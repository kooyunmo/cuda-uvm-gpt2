import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from contextlib import contextmanager
from tqdm import tqdm
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from models.gpt2 import GPT2LM
from models.gpt2_prefetch import PrefetchGPT2LM


class MockDataset(Dataset):
    def __init__(self, dsize):
        self.dsize = dsize
        self.data = []
        for i in range(dsize):
            self.data.append(torch.randint(low=0, high=50256, size=(1024,)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return d


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-nsys', action='store_true', help='whether to use nsys')
    return parser.parse_args()


def main():
    args = get_args()
    prefetch_stream = torch.cuda.Stream() 

    #model = GPT2LM(embed_dim=768, num_heads=12, num_layers=12).train().cuda()
    #model = GPT2LM().cuda()
    #model = PrefetchGPT2LM(embed_dim=768, num_heads=12, num_layers=12, prefetch_stream=prefetch_stream).eval().cuda()
    #model = PrefetchGPT2LM().train().cuda()

    synthetic_dataset = MockDataset(5)
    dataloader = DataLoader(synthetic_dataset, batch_size=1, shuffle=False)

    fw_times = []
    for inp in tqdm(dataloader):
        # TODO: eval with no_grad
        with event_measure() as result:
            out = model(inp.cuda())
        fw_times.append(result['time'])

    avg_fw_time = np.mean(fw_times)
    print(f"avg step time: {avg_fw_time} ms")


if __name__ == '__main__':
    main()