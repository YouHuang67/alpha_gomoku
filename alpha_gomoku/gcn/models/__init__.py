from .base import *
from .residual import *


embedding_classes = {
    k.lower(): v for k, v in locals().items()
    if isinstance(v, type) and issubclass(v, EmbeddingBase)
}

network_classes = {
    k.lower(): v for k, v in locals().items()
    if isinstance(v, type) and issubclass(v, NetworkBase)
}