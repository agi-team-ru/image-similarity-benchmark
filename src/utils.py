import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict
import pandas as pd


def base64_encode(b: bytes) -> str:
    output = BytesIO()
    base64.encode(BytesIO(b), output)
    output.seek(0)
    return output.read().decode("ascii")


def base64_decode(s: str) -> bytes:
    return base64.b64decode(s)


def read_binary_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def file_exists(path: str):
    return Path(path).is_file()


def pandas_append(db: pd.DataFrame, row: Dict[str, Any]):
    return pd.concat(
        [
            db,
            pd.DataFrame([pd.Series(row)]),
        ],
        ignore_index=True,
    )
