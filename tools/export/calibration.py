"""Explicit calibration reduction for the pinned AI Edge Quantizer.

Its profiler default is an exponential moving average. That can exclude rare
positive detector logits and cap sigmoid scores at 0.5. Use observed global
extrema and retain the statistics so quantization is reproducible and auditable.
"""
import json
import numpy as np
from ai_edge_quantizer import calibrator
from ai_edge_quantizer.utils import qsv_utils

def calibrate_minmax(qt,fp,samples,output):
    runner=calibrator.Calibrator(str(fp),num_threads=4,
        mode=calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED,
        qsv_update_func=qsv_utils.min_max_update)
    # The pinned package's Quantizer.calibrate does not expose reduction policy.
    runner.calibrate(samples,qt._recipe_manager)
    stats=runner.get_model_qsvs()
    output.write_text(json.dumps(stats,default=lambda v:np.asarray(v).tolist(),indent=2)+'\n')
    return stats
