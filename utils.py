from contextlib import contextmanager
import torch

@contextmanager
def event_measure():
    """ Measure GPU execution time by CUDA event """
    result = {}
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    try:
        yield result
    finally:
        end_event.record()
        end_event.synchronize()
        result["time"] = start_event.elapsed_time(end_event)
