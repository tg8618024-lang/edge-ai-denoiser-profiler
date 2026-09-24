"""VST3 / CLAP Audio Plugin Bridge Package."""

from src.plugins.vst.vst_bridge import (
    AudioCircularFIFO,
    VST3PluginProcessor,
    PARAM_BYPASS,
    PARAM_DENOISE_AMOUNT,
    PARAM_MODEL_SELECT,
    PARAM_CROSSFADE,
    PARAM_PRECISION,
    create_vst_instance,
    destroy_vst_instance,
)
from src.plugins.vst.host_simulator import DAWHostSimulator, HostSimulationResult

__all__ = [
    "AudioCircularFIFO",
    "VST3PluginProcessor",
    "PARAM_BYPASS",
    "PARAM_DENOISE_AMOUNT",
    "PARAM_MODEL_SELECT",
    "PARAM_CROSSFADE",
    "PARAM_PRECISION",
    "create_vst_instance",
    "destroy_vst_instance",
    "DAWHostSimulator",
    "HostSimulationResult",
]
