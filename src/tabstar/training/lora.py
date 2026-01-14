from os.path import exists

from peft import LoraConfig, get_peft_model, PeftModel

from sap_rpt_oss.constants import ModelSize
from tabstar.arch.arch import TabStarModel
from tabstar.arch.rpt_model import TabStarRPTModel


def load_pretrained(
    model_version: str,
    lora_r: int,
    lora_alpha: int,
    dropout: float,
    encoder_backend: str = "tabstar",
    rpt_model_size: ModelSize = ModelSize.base,
    rpt_checkpoint_path: str | None = None,
) -> PeftModel:
    if encoder_backend == "rpt":
        print("🤩 Loading RPT-backed TabSTAR model")
        model = TabStarRPTModel(
            rpt_model_size=rpt_model_size,
            rpt_checkpoint_path=rpt_checkpoint_path,
        )
    else:
        print(f"🤩 Loading pretrained model version: {model_version}")
        model = TabStarModel.from_pretrained(model_version, local_files_only=True)
    # TODO: probably best if this is written more generic and not so hard-coded
    lora_modules = ["query", "key", "value", "out_proj", "linear1", "linear2",
                    "cls_head.layers.0", "reg_head.layers.0"]
    to_exclude = []
    if encoder_backend == "tabstar":
        to_freeze = range(6)
        prefixes = tuple(f"text_encoder.encoder.layer.{i}." for i in to_freeze)
        to_exclude = [name for name, _ in model.named_modules() if name.startswith(prefixes)]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * lora_alpha,
        target_modules=lora_modules,
        exclude_modules=to_exclude,
        lora_dropout=dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model


def load_finetuned(
    save_dir: str,
    tabstar_version: str,
    encoder_backend: str = "tabstar",
    rpt_model_size: ModelSize = ModelSize.base,
    rpt_checkpoint_path: str | None = None,
) -> PeftModel:
    if not exists(save_dir):
        raise FileNotFoundError(f"Checkpoint path {save_dir} does not exist.")
    if encoder_backend == "rpt":
        base_model = TabStarRPTModel(
            rpt_model_size=rpt_model_size,
            rpt_checkpoint_path=rpt_checkpoint_path,
        )
    else:
        base_model = TabStarModel.from_pretrained(tabstar_version, local_files_only=True)
    model = PeftModel.from_pretrained(base_model, save_dir, device_map='cpu', local_files_only=True)
    return model
