# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch
from mmcv import Config, DictAction
import torch.nn as nn
import ptflops
import thop


def prepare_inputs(shape_nchw):
    x = torch.zeros(shape_nchw)
    return x


class Net(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride)

    def forward(self, x):
        x = self.conv(x)
        return x


class TestProfile:
    def __init__(self,
                 shape_nchw=(1, 128, 32, 32),
                 in_channels=128,
                 out_channels=128,
                 kernel_size=3,
                 stride=1):
        self.shape_nchw = shape_nchw
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        print('(n,c,h,w) is :', shape_nchw)
        input_height = shape_nchw[2] - kernel_size + 1
        input_width = shape_nchw[3] - kernel_size + 1
        default_flops = 2*input_height*input_width*(in_channels*kernel_size*kernel_size+1)*out_channels
        default_mflops = default_flops/1e6
        default_macs = default_mflops/2
        print('manual flops is %.4f MFLOPs' % default_mflops)
        print('manual macs is %.4f MFLOPs' % default_macs)

    def init_model(self):
        model = Net(self.in_channels, self.out_channels, self.kernel_size, self.stride)
        input_buffer = prepare_inputs(self.shape_nchw)
        return model, input_buffer

    def profile_thop(self, show_params=False):
        model, input_buffer = self.init_model()
        with torch.no_grad():
            macs, params = thop.profile(model, inputs=(input_buffer, ), verbose=True)
            from thop import clever_format
            macs, params = clever_format([macs, params], "%.3f")
            print('THOP macs : %s' % macs)
            if show_params:
                print('THOP params : %s' % params)

    def profile_pt(self, show_params=False):
        model, input_buffer = self.init_model()
        with torch.no_grad():
            macs, params = ptflops.get_model_complexity_info(model,
                                                             input_res=(self.shape_nchw[1], self.shape_nchw[2], self.shape_nchw[3], ),
                                                             as_strings=True,
                                                             print_per_layer_stat=True, verbose=True)
            print('ptflops {:<30}  {:<8}'.format('Computational complexity: ', macs))
            if show_params:
                print('ptflops {:<30}  {:<8}'.format('Number of parameters: ', params))


    def profile_pytorch(self):
        model, input_buffer = self.init_model()
        with torch.no_grad():
            with torch.autograd.profiler.profile(with_stack=False, enabled=True, use_cuda=False, record_shapes=True,
                                                 with_flops=True, profile_memory=True) as prof:
                outputs = model(input_buffer)
            print(prof.key_averages(group_by_stack_n=5).table(row_limit=-1))

if __name__ == '__main__':
    test_tool = TestProfile()
    # test_tool.profile_pytorch()
    # test_tool.profile_thop()
    test_tool.profile_pt()
    pass
