"""Configuration helpers required by BaFormer's model constructors."""

from functools import wraps


def configurable(init_func=None, *, from_config=None):
    """Support Detectron2's ``@configurable`` constructor convention.

    BaFormer passes its YACS config directly to registered modules.  The
    full Detectron2 decorator also supports explicit keyword arguments, which
    this wrapper preserves for ordinary constructor calls.
    """
    if init_func is None:
        return lambda func: configurable(func, from_config=from_config)

    @wraps(init_func)
    def wrapped(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and hasattr(type(self), "from_config"):
            cfg = args[0]
            params = type(self).from_config(cfg)
            return init_func(self, **params)
        return init_func(self, *args, **kwargs)

    return wrapped
