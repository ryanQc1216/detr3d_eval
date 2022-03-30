# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch
from mmcv import Config, DictAction

from mmdet3d.models import build_model
import os
import thop
try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
import numpy as np
from tools.analysis_tools.ptflops.flops_counter import get_model_complexity_info as ptflops_get_model_complexity_info

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',
                        default='./projects/configs/detr3d/detr3d_res101_gridmask_eval.py',
                        help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[40000, 4],
        help='input point cloud size')
    parser.add_argument(
        '--modality',
        type=str,
        default='multi',
        choices=['point', 'image', 'multi'],
        help='input data modality')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def input_constructor(shape):
    img = np.zeros(shape, dtype=np.float)
    img = torch.tensor(img).float().cuda()
    cams_pose = torch.zeros((1, 6, 4, 4)).float().cuda()
    inputs = dict(
        imgs=img,
        cams_pose=cams_pose
    )
    return inputs


def main():
    args = parse_args()

    if args.modality == 'point':
        assert len(args.shape) == 2, 'invalid input shape'
        input_shape = tuple(args.shape)
    elif args.modality == 'image':
        if len(args.shape) == 1:
            input_shape = (3, args.shape[0], args.shape[0])
        elif len(args.shape) == 2:
            input_shape = (3,) + tuple(args.shape)
        else:
            raise ValueError('invalid input shape')
    elif args.modality == 'multi':
        input_shape = (6, 3, 960, 1280)

    cfg = Config.fromfile(args.config)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not supported for {}'.format(
                model.__class__.__name__))

    inputs = input_constructor(input_shape)
    imgs = inputs['imgs']
    cams_pose = inputs['cams_pose']

    '''
    use mmdetection flops tool
    '''
    # with torch.no_grad():
    #     res = model(imgs, cams_pose)
    #     flops, params = get_model_complexity_info(model, input_shape, input_constructor=input_constructor)
    #     split_line = '=' * 30
    #     print(f'{split_line}\nInput shape: {input_shape}\n'
    #           f'Flops: {flops}\nParams: {params}\n{split_line}')

    '''
    use torchscript export
    '''
    # with torch.no_grad():
    #     res = model(imgs, cams_pose)
    #     model_export = torch.jit.trace(model,
    #                                    (imgs, cams_pose),
    #                                    strict=False)
    #     torch.jit.save(model_export, './outputs/detr3d_export.pt')
    #     graph = model_export.graph.copy()
    #     torch._C._jit_pass_onnx_function_substitution(graph)
    #     print(graph)

    '''
    use onnx export
    '''
    # with torch.no_grad():
    #     res = model(imgs, cams_pose)
    #     torch.onnx.export(model,
    #                       (imgs, cams_pose),
    #                       './outputs/detr3d_export.onnx',
    #                       opset_version=13,
    #                       do_constant_folding=False)


    '''
    use pytorch profile tools
    '''
    # with torch.no_grad():
    #     with torch.autograd.profiler.profile(with_stack=False, enabled=True, use_cuda=False, record_shapes=True,
    #                                          with_flops=True, profile_memory=True) as prof:
    #         outputs = model(imgs, cams_pose)
    #     print(prof.key_averages(group_by_stack_n=5).table(row_limit=-1))

    '''
    use thop profile
    '''
    # with torch.no_grad():
    #     macs, params = thop.profile(model, (imgs, cams_pose), verbose=True)
    #     from thop import clever_format
    #     flops, params = clever_format([macs*2, params], "%.3f")
    #     print('flops : %s' % flops)
    #     print('params : %s' % params)
    #     kk = 1

    '''
    use pt profile
    '''
    with torch.no_grad():
        res = model(imgs, cams_pose)
        macs, params = ptflops_get_model_complexity_info(model, input_shape,
                                                         input_constructor=input_constructor,
                                                         as_strings=True,
                                                         ignore_modules = [''],
                                                         print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if __name__ == '__main__':
    main()
