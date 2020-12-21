from typing import List
from contextlib import contextmanager

import torch

class Prefetcher:
    def __init__(self, num_prefetch_streams):
        self.prefetch_streams: List[torch.cuda.Stream] = []
        if num_prefetch_streams == 0:
            self.prefetch_streams.append(torch.cuda.default_stream())
        else:
            for _ in range(num_prefetch_streams):
                self.prefetch_streams.append(torch.cuda.Stream())
        self.num_prefetch_blocks_list: List[int] = []
        self.current_prefetch_idx: int = 0

    @contextmanager
    def record_malloc(self):
        result = {}
        torch._C._cuda_enablePrefetchRecording()

        try:
            yield result
        finally:
            result['num_blocks'] = torch._C._cuda_disablePrefetchRecording()
            self.num_prefetch_blocks_list.append(result["num_blocks"])

    def prefetch_async(self, prefetch_stride=1):
        num_blocks_to_prefetch = sum(self.num_prefetch_blocks_list[self.current_prefetch_idx:self.current_prefetch_idx+prefetch_stride])
        current_prefetch_stream = self.prefetch_streams[self.current_prefetch_idx % len(self.prefetch_streams)]
        torch._C._cuda_memPrefetchAsync(current_prefetch_stream._cdata, num_blocks_to_prefetch)
        self.current_prefetch_idx += prefetch_stride
        
        if self.current_prefetch_idx == len(self.num_prefetch_blocks_list):
            self.current_prefetch_idx = 0
            torch._C._cuda_clearPrefetchIdx()
