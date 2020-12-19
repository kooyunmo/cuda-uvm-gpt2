from dataclasses import dataclass, field
from typing import List
from contextlib import contextmanager

import torch

@dataclass
class Prefetcher:
    prefetch_stream: torch.cuda.Stream = field(default_factory=torch.cuda.Stream)
    num_prefetch_blocks_list: List[int] = field(default_factory=list)
    current_prefetch_idx: int = 0

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
        torch._C._cuda_memPrefetchAsync(self.prefetch_stream._cdata, num_blocks_to_prefetch)
        self.current_prefetch_idx += prefetch_stride
        
        if self.current_prefetch_idx == len(self.num_prefetch_blocks_list):
            self.current_prefetch_idx = 0
            torch._C._cuda_clearPrefetchIdx()
