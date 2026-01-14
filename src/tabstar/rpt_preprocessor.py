from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from sap_rpt_oss.data.tokenizer import Tokenizer
from tabstar.preprocessing.nulls import raise_if_null_target
from tabstar.preprocessing.target import fit_preprocess_y, transform_preprocess_y
from tabstar.preprocessing.texts import replace_column_names
from tabstar.tabstar_verbalizer import TabSTARData


@dataclass
class RPTTokenizedData:
    data: Dict[str, np.ndarray]
    column_embeddings: np.ndarray


class TabSTARRPTPreprocessor:
    def __init__(
        self,
        is_cls: bool,
        verbose: bool = False,
        regression_type: str = "l2",
        classification_type: str = "cross-entropy",
        num_regression_bins: int = 16,
    ):
        self.is_cls = is_cls
        self.verbose = verbose
        self.tokenizer = Tokenizer(
            regression_type=regression_type,
            classification_type=classification_type,
            num_regression_bins=num_regression_bins,
            random_seed=0,
            is_valid=False,
        )
        self.target_transformer = None
        self.d_output: Optional[int] = None
        self.y_name: Optional[str] = None
        self.y_values: Optional[List[str]] = None
        self.target_tokens: Optional[List[str]] = None
        self.target_token_columns: Optional[List[str]] = None

    def fit(self, X: DataFrame, y: Series) -> None:
        if not isinstance(X, DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if not isinstance(y, Series):
            raise ValueError("y must be a pandas Series.")
        raise_if_null_target(y)
        self._assert_no_duplicate_columns(X)

        x = X.copy()
        y = y.copy()
        x, y = replace_column_names(x=x, y=y)

        self.target_transformer = fit_preprocess_y(y=y, is_cls=self.is_cls)
        if self.is_cls:
            self.d_output = len(self.target_transformer.classes_)
            self.y_values = sorted(self.target_transformer.classes_)
        else:
            self.d_output = 1
        self.y_name = str(y.name)
        self.target_tokens = self._build_target_tokens(self.y_name, self.y_values)
        self.target_token_columns = [f"TABSTAR_TARGET_TOKEN_{i}" for i in range(len(self.target_tokens))]

    def transform(self, X: DataFrame, y: Optional[Series]) -> TabSTARData:
        if self.d_output is None or self.target_transformer is None:
            raise ValueError("Preprocessor is not fitted yet. Call fit() first.")
        x = X.copy()
        self._assert_no_duplicate_columns(x)
        if y is not None:
            y = y.copy()
            raise_if_null_target(y)
        x, y = replace_column_names(x=x, y=y)

        y_transformed = self.transform_target(y)
        x = self._append_target_tokens_end(x, self.target_tokens, self.target_token_columns)

        tokenized = self._tokenize_with_rpt(x=x, y=y)
        return TabSTARData(
            d_output=self.d_output,
            x_txt=None,
            x_num=None,
            y=y_transformed,
            rpt_data=tokenized.data,
            rpt_column_embeddings=tokenized.column_embeddings,
        )

    def transform_target(self, y: Optional[Series]) -> Optional[Series | np.ndarray]:
        if y is None:
            return None
        return transform_preprocess_y(y=y, scaler=self.target_transformer)

    @staticmethod
    def _build_target_tokens(y_name: str, y_values: Optional[List[str]]) -> List[str]:
        if y_values:
            values = [str(v) for v in y_values]
            return [f"Target Feature: {y_name}\nFeature Value: {v}" for v in values]
        return [f"Numerical Target Feature: {y_name}"]

    @staticmethod
    def _append_target_tokens_end(
        x: DataFrame, tokens: List[str], token_columns: List[str]
    ) -> DataFrame:
        target_df = DataFrame(
            {col: [token] * len(x) for col, token in zip(token_columns, tokens)},
            index=x.index,
        )
        return pd.concat([x, target_df], axis=1)

    def _tokenize_with_rpt(self, x: DataFrame, y: Optional[Series]) -> RPTTokenizedData:
        if y is None:
            y_context = pd.Series(np.zeros(len(x), dtype=np.float32), name=self.y_name, index=x.index)
        else:
            y_context = y
        y_context = y_context.to_frame()
        x_context = x
        # Tokenizer expects a non-empty query set for standardization; use a single dummy row.
        x_query = x_context.iloc[:1].copy()
        y_query = y_context.iloc[:1].copy()        

        task = "classification" if self.is_cls else "regression"
        data, _, _ = self.tokenizer(
            X_context=x_context,
            y_context=y_context,
            X_query=x_query,
            y_query=y_query,
            classification_or_regression=task,
        )
        # Drop the dummy query row to keep dataset size aligned with input.
        context_size = len(y_context)
        data = {k: v[:context_size] for k, v in data.items()}
        
        # Remove any target leakage in the last (target) column.
        data["text_embeddings"][:, -1] = 0
        data["target"].fill_(-100)
        data["target_delta"].zero_()

        column_embeddings = data.pop("column_embeddings")
        data_np = {k: v.cpu().numpy() for k, v in data.items()}
        return RPTTokenizedData(data=data_np, column_embeddings=column_embeddings.cpu().numpy())

    @staticmethod
    def _assert_no_duplicate_columns(x: DataFrame) -> None:
        if len(set(x.columns)) != len(x.columns):
            raise ValueError("Duplicate column names found in DataFrame!")
