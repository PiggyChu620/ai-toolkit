import importlib


_JOB_CLASS_MODULES = {
    "BaseJob": ".BaseJob",
    "ExtractJob": ".ExtractJob",
    "TrainJob": ".TrainJob",
    "MergeJob": ".MergeJob",
    "ModJob": ".ModJob",
    "GenerateJob": ".GenerateJob",
    "ExtensionJob": ".ExtensionJob",
}

__all__ = list(_JOB_CLASS_MODULES.keys())


def __getattr__(name):
    module_name = _JOB_CLASS_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
