import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from tabstar.tabstar_verbalizer import TabSTARData
from tabstar.training.hyperparams import GLOBAL_BATCH


class TabSTARDataset(Dataset):
    def __init__(self, data: TabSTARData):
        self.x_txt = data.x_txt
        self.x_num = data.x_num
        self.rpt_data = data.rpt_data
        self.rpt_column_embeddings = data.rpt_column_embeddings
        data_len = len(data)
        if isinstance(data.y, pd.Series):
            self.y = data.y.reset_index(drop=True)
        elif isinstance(data.y, np.ndarray):
            self.y = pd.Series(data.y, dtype=np.float32)
        else:
            # Dummy target for convenience
            self.y = pd.Series(np.zeros(data_len), dtype=np.float32)
        self.d_output = data.d_output

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        y = self.y.iloc[idx]
        if self.rpt_data is not None:
            rpt_row = {k: v[idx] for k, v in self.rpt_data.items()}
            return {"rpt_data": rpt_row, "rpt_column_embeddings": self.rpt_column_embeddings,
                    "y": y, "d_output": self.d_output}
        x_txt = self.x_txt[idx]
        x_num = self.x_num[idx]
        return {"x_txt": x_txt, "x_num": x_num, "y": y, "d_output": self.d_output}
    
def get_dataloader(data: TabSTARData, is_train: bool, batch_size: int = GLOBAL_BATCH) -> DataLoader:
    dataset = TabSTARDataset(data)
    return DataLoader(dataset, shuffle=is_train, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)


def collate_fn(batch) -> TabSTARData:
    d_output = batch[0]["d_output"]
    y = pd.Series([item["y"] for item in batch])
    if "rpt_data" in batch[0]:
        rpt_keys = batch[0]["rpt_data"].keys()
        rpt_data = {k: np.stack([item["rpt_data"][k] for item in batch]) for k in rpt_keys}
        rpt_column_embeddings = batch[0]["rpt_column_embeddings"]
        rpt_data["column_embeddings"] = rpt_column_embeddings
        return TabSTARData(
            d_output=d_output,
            x_txt=None,
            x_num=None,
            y=y,
            rpt_data=rpt_data,
            rpt_column_embeddings=rpt_column_embeddings,
        )
    x_txt = np.stack([item["x_txt"] for item in batch])
    x_num = np.stack([item["x_num"] for item in batch])
    return TabSTARData(d_output=d_output, x_txt=x_txt, x_num=x_num, y=y)
