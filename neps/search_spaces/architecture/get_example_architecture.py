from __future__ import annotations

from torch import nn

import neps
from neps.search_spaces.architecture import primitives as ops
from neps.search_spaces.architecture import topologies as topos
from neps.search_spaces.architecture.primitives import AbstractPrimitive
from omegaconf import DictConfig


def get_architecture() -> neps.ArchitectureParameter:
    class DownSampleBlock(AbstractPrimitive):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__(locals())
            self.conv_a = ReLUConvBN(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
            self.conv_b = ReLUConvBN(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
                ),
            )

        def forward(self, inputs):
            basicblock = self.conv_a(inputs)
            basicblock = self.conv_b(basicblock)
            residual = self.downsample(inputs)
            return residual + basicblock


    class ReLUConvBN(AbstractPrimitive):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
            super().__init__(locals())

            self.kernel_size = kernel_size
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
            )

        def forward(self, x):
            return self.op(x)


    class AvgPool(AbstractPrimitive):
        def __init__(self, **kwargs):
            super().__init__(kwargs)
            self.op = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        def forward(self, x):
            return self.op(x)


    primitives = {
        "Sequential15": topos.get_sequential_n_edge(15),
        "DenseCell": topos.get_dense_n_node_dag(4),
        "down": {"op": DownSampleBlock},
        "avg_pool": {"op": AvgPool},
        "id": {"op": ops.Identity},
        "conv3x3": {"op": ReLUConvBN, "kernel_size": 3, "stride": 1, "padding": 1},
        "conv1x1": {"op": ReLUConvBN, "kernel_size": 1, "stride": 1, "padding": 0},
    }


    structure = {
        "S": ["Sequential15(C, C, C, C, C, down, C, C, C, C, C, down, C, C, C, C, C)"],
        "C": ["DenseCell(OPS, OPS, OPS, OPS, OPS, OPS)"],
        "OPS": ["id", "conv3x3", "conv1x1", "avg_pool"],
    }


    def set_recursive_attribute(op_name, predecessor_values):
        in_channels = 16 if predecessor_values is None else predecessor_values["out_channels"]
        out_channels = in_channels * 2 if op_name == "DownSampleBlock" else in_channels
        return dict(in_channels=in_channels, out_channels=out_channels)
    
    prior_distr = {
        "S": [1],
        "C": [1],
        "OPS": [1/4, 1/4, 1/4, 1/4],
    }
        
    return neps.ArchitectureParameter(
        set_recursive_attribute=set_recursive_attribute,
        structure=structure,
        primitives=primitives,
        prior=prior_distr
    )

def get_cfg_architecture() -> neps.CFGArchitectureParameter:
    primitives = [
        "unet",
        "conv_encoder",
        "res_encoder",
        "down",
        "conv_decoder",
        "res_decoder",
        "up",
        "block",
        "instance_norm",
        "batch_norm",
        "leaky_relu",
        "relu",
        "elu",
        "prelu",
        "gelu",
        "1b",
        "2b",
        "3b",
        "4b",
        "5b",
        "6b",
        "dropout",
        "no_dropout",
    ]
    
    # We want to keep at least half of the stages to ensure that the network is deep enough
    n_stages = 4
    possible_n_stages = range(n_stages // 2, n_stages + 1)
    starting_rule = [f"unet {n}E {n}D" for n in possible_n_stages]

    def get_productions_dict(part: str, _n_stages: range) -> dict[str, list[str]]:
        result = {}
        for n in _n_stages:
            if part == "encoder":
                result[f"{n}E"] = [f"conv_{part} NORM, NONLIN, DROPOUT, {', '.join([f'{_n}EB, down' for _n in range(1, n)])}, {n}EB"]
                result[f"{n}E"] = [f"res_{part} NORM, NONLIN, DROPOUT, {', '.join([f'{_n}EB, down' for _n in range(1, n)])}, {n}EB"]

            elif part == "decoder":
                result[f"{n}D"] = [f"conv_{part} NORM, NONLIN, DROPOUT, {', '.join([f'up, {_n}DB' for _n in range(1, n)])}"]
                result[f"{n}D"] = [f"res_{part} NORM, NONLIN, DROPOUT, {', '.join([f'up, {_n}DB' for _n in range(1, n)])}"]

            else:
                raise ValueError(f"Unknown part: {part}")
            
        for _n in range(1, max(_n_stages) + 1):
            result[f"{_n}EB"] = ["1b", "2b", "3b", "4b", "5b", "6b"]

        for _n in range(1, max(_n_stages) + 1):
            result[f"{_n}DB"] = ["1b", "2b", "3b", "4b", "5b", "6b"]
        return result

    encoder_rules = get_productions_dict("encoder", possible_n_stages) 
    decoder_rules = get_productions_dict("decoder", possible_n_stages)

    structure = {
        "S": starting_rule,
        **encoder_rules,
        **decoder_rules,
        "NORM": ["instance_norm", "batch_norm"],
        "NONLIN": ["leaky_relu", "relu", "prelu", "gelu"],
        "DROPOUT": ["dropout", "no_dropout"],
    }

    return neps.CFGArchitectureParameter(
        structure=structure,
        primitives=primitives,
    )

def get_default_cfg_architecture() -> neps.CFGArchitectureParameter:
    return "(S unet (3E res_encoder (NORM batch_norm) (NONLIN relu) (DROPOUT dropout) (1EB 4b) down (2EB 6b) down (3EB 1b)) (3D res_decoder (NORM batch_norm) (NONLIN relu) (DROPOUT dropout) up (1DB 6b) up (2DB 3b)))"