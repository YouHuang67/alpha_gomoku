from collections import Iterable


tolist = lambda x: x if isinstance(x, Iterable) else [x]