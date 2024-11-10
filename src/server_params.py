from enum import Enum
from pydantic import BaseModel

LOCAl_SERVER_ENDPOINT="/img-similarity"
LOCAl_SERVER_PORT=8888

class LocalServerMethod(Enum):
    ALWAYS_TRUE = "always_true"
    ALWAYS_RANDOM = "always_random"
    ALWAYS_FALSE = "always_false"
    LIB_SIMILARITIES = "lib_similarities"
    THREAT_EXCHANGE = "threat_exchange"


class LocalServerOptions(BaseModel):
    method: LocalServerMethod
