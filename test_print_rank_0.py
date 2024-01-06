import print_rank_0
import pytest
import torch


@pytest.fixture()
def process_group():
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1,
                                         init_method="tcp://localhost:3080")
    yield
    torch.distributed.destroy_process_group()


def test_print_rank_0(process_group):
    print_rank_0("Hello World!")
    print_rank_0.print_rank_0("Hello World!")
    print_rank_0.print_rank_last("Hello World!")
