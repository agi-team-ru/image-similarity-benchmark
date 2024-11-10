import logging
import os
from typing import Any, Dict, List, cast
import httpx
from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import f1_score, r2_score, root_mean_squared_error
from dto import ImgSimilarityRequest, ImgSimilarityResponse
from server_params import (
    LocalServerMethod,
    LocalServerOptions,
    LOCAl_SERVER_ENDPOINT,
    LOCAl_SERVER_PORT,
)
import time

from dataset_params import (
    ds_source_dir,
    ds_info_file,
    ds_key_score,
    ds_key_images,
    ds_results_file,
)
from utils import base64_encode, file_exists, pandas_append, read_binary_file

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper())
logger = logging.getLogger(__name__)


@dataclass
class TestedMethod:
    name: str
    endpoint: str
    options: Dict[str, Any]
    threshold: float


local_server_base_url = f"http://localhost:{LOCAl_SERVER_PORT}{LOCAl_SERVER_ENDPOINT}"

tested_methods = [
    TestedMethod(
        name="random",
        endpoint=local_server_base_url,
        options=LocalServerOptions(method=LocalServerMethod.ALWAYS_RANDOM).model_dump(),
        threshold=0.5,
    ),
    # TestedMethod(
    #     name="const_true",
    #     endpoint=local_server_base_url,
    #     options=LocalServerOptions(method=LocalServerMethod.ALWAYS_TRUE).model_dump(),
    #     threshold=0.5,
    # ),
    # TestedMethod(
    #     name="const_false",
    #     endpoint=local_server_base_url,
    #     options=LocalServerOptions(method=LocalServerMethod.ALWAYS_FALSE).model_dump(),
    #     threshold=0.5,
    # ),
    # TestedMethod(
    #     name="lib_sim",
    #     endpoint=local_server_base_url,
    #     options=LocalServerOptions(
    #         method=LocalServerMethod.LIB_SIMILARITIES
    #     ).model_dump(),
    #     threshold=0.7,
    # ),
    # TestedMethod(
    #     name="llm1",
    #     endpoint="http://31.128.44.189:8000/score",
    #     options={},
    #     threshold=0.75,
    # ),
]


def main():
    dataset_df = pd.read_csv(ds_info_file).sample(frac=1)

    db = (
        pd.read_csv(ds_results_file, index_col="#")
        if file_exists(ds_results_file)
        else pd.DataFrame(
            columns=cast(
                Any, ["method"] + [key for key in ds_key_images] + ["y_true", "y_pred"]
            )
        )
    )

    http_client = httpx.Client()
    y_true: List[float] = []
    y_true_bin: List[bool] = []
    y_pred_map: Dict[str, List[float]] = {}
    y_pred_bin_map: Dict[str, List[bool]] = {}

    for _, row in dataset_df.iterrows():
        y_true_value = float(row[ds_key_score])
        images: List[str] = []
        for img_key in ds_key_images:
            img_path = f"{ds_source_dir}/{row[img_key]}"
            images.append(base64_encode(read_binary_file(img_path)))

        y_true_bin.append(y_true_value > 0.5)
        y_true.append(y_true_value)

        for method in tested_methods:
            name = method.name
            while True:
                try:
                    r = http_client.post(
                        method.endpoint,
                        json=ImgSimilarityRequest(
                            images=images, options=method.options
                        ).model_dump(mode="json"),
                        timeout=120,
                    )
                    if not httpx.codes.is_success(r.status_code):
                        logger.error(f"Request failed ({r.status_code}): {r.text}")
                    else:
                        break
                except:
                    logger.exception("Request failed")
                retry_timeout = 3
                logger.error(f"Retrying in {retry_timeout}s")
                time.sleep(retry_timeout)

            response = ImgSimilarityResponse.model_validate_json(r.text)
            y_pred_value = response.score
            y_pred_bin_map.setdefault(name, []).append(y_pred_value > method.threshold)
            y_pred_map.setdefault(name, []).append(y_pred_value)
            db = pandas_append(
                db,
                {img_key: row[img_key] for img_key in ds_key_images}
                | {
                    "y_true": y_true_value,
                    "y_pred": y_pred_value,
                    "method": method.name,
                },
            )
            db.to_csv(ds_results_file, index_label="#")

        if len(y_true) > 1:
            df_name: List[str] = []
            df_r2: List[float] = []
            df_f1: List[float] = []
            df_mse: List[float] = []
            for method in tested_methods:
                name = method.name
                df_name.append(name)
                df_r2.append(float(r2_score(y_pred=y_pred_map[name], y_true=y_true)))
                df_f1.append(
                    f1_score(
                        y_pred=y_pred_bin_map[name],
                        y_true=y_true_bin,
                        zero_division=cast(str, 1.0),
                    )
                )
                df_mse.append(
                    float(
                        root_mean_squared_error(y_pred=y_pred_map[name], y_true=y_true)
                    )
                )

            df = pd.DataFrame(
                {
                    "Method": df_name,
                    "F1 Score": df_f1,
                    "R2 Score": df_r2,
                    "MSE Score": df_mse,
                }
            )
            print(f"\n\n{df.to_markdown()}")


if __name__ == "__main__":
    main()
