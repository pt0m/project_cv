import src.pysot.core.config as cfg
from src.pysot.tracker.siamrpn_tracker import SiamRPNTracker
from src.pysot.tracker.siammask_tracker import SiamMaskTracker
from src.pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
