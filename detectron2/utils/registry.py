class Registry:
    """Minimal name-to-object registry compatible with BaFormer."""

    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def register(self, obj=None):
        if obj is None:
            return lambda cls: self.register(cls)
        self._obj_map[obj.__name__] = obj
        return obj

    def get(self, name):
        try:
            return self._obj_map[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._obj_map))
            raise KeyError(f"{name!r} is not registered in {self._name}. Available: {available}") from exc
