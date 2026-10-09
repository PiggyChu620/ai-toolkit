import importlib


_PROCESS_CLASS_MODULES = {
    "BaseExtractProcess": ".BaseExtractProcess",
    "ExtractLoconProcess": ".ExtractLoconProcess",
    "ExtractLoraProcess": ".ExtractLoraProcess",
    "BaseProcess": ".BaseProcess",
    "BaseTrainProcess": ".BaseTrainProcess",
    "TrainVAEProcess": ".TrainVAEProcess",
    "BaseMergeProcess": ".BaseMergeProcess",
    "TrainSliderProcess": ".TrainSliderProcess",
    "TrainSliderProcessOld": ".TrainSliderProcessOld",
    "TrainSDRescaleProcess": ".TrainSDRescaleProcess",
    "ModRescaleLoraProcess": ".ModRescaleLoraProcess",
    "GenerateProcess": ".GenerateProcess",
    "BaseExtensionProcess": ".BaseExtensionProcess",
    "TrainESRGANProcess": ".TrainESRGANProcess",
    "BaseSDTrainProcess": ".BaseSDTrainProcess",
}

__all__ = list(_PROCESS_CLASS_MODULES.keys())


def __getattr__(name):
    module_name = _PROCESS_CLASS_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
