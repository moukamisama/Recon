import torch
from collections import OrderedDict

from archs.modules_recon import *
from utils.registry import ARCH_REGISTRY

import json

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv2d_recon(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d_recon(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d_recon
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = ReLU_recon(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if len(identity) == len(out):
            for i in range(len(identity)):
                out[i] += identity[i]
        elif len(identity) == 1 and len(out) > 1:
            for i in range(len(identity)):
                out[i] += identity[0]
        elif len(identity) > 1 and len(out) == 1:
            o = []
            for i in range(len(identity)):
                o.append(out[0] + identity[i])
            out = o

        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d_recon
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = ReLU_recon(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_recon(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,
                 # added
                 tasks=None, branch_type='no_branch', topK=1, conflict_scores_file=None):
        """
        :param tasks: list of tasks
        :param branch_type: 'no_branch' or 'branched'
        :param topK: topK layers that are branch_type
        :param conflict_scores_file: file that contains the conflict scores
        """

        super(ResNet_recon, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d_recon
        self._norm_layer = norm_layer

        # added: tasks relevant
        if tasks is None:
            tasks = ['Task1']
        self.tasks = tasks
        self.n_tasks = len(self.tasks)

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2d_recon(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = ReLU_recon(inplace=True)
        self.maxpool = MaxPool2d_recon(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = AdaptiveAvgPool2d_recon((1, 1))
        self.fc = Linear_recon(512 * block.expansion, num_classes)


        # define the shared parameters
        self._shared_parameters = OrderedDict()

        # define the task-specific parameters
        if branch_type == 'no_branch':
            layers_branched = []
        elif branch_type == 'branched':
            if conflict_scores_file is None:
                raise ValueError("conflict_scores_file should be provided when branch_type is 'branched'")

            with open(conflict_scores_file, "r") as fp:
                layers_branched = json.load(fp)
                layers_branched = list(layers_branched.keys())
        else:
            raise ValueError("branch_type should be 'no_branch' or 'branched'")


        # turn the topK layers to task-specific parameters
        layers_branched = layers_branched[:topK]
        self.turn(layers_branched)

        # get shared parameters
        self.get_name_of_shared_parameters(layers_branched)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def shared_modules(self):
        return self

    def shared_parameters(self):
        return self._shared_parameters

    def turn(self, layers_branched):
        """
        Turn the layers in layers_branched to task-specific parameters
        """
        self._shared_parameters.clear()

        # if there are multiple tasks, set the number of tasks for branch layers
        if self.n_tasks > 1:
            for idx, m in self.named_modules():
                if idx in layers_branched:
                    m.set_n_tasks(n_tasks=self.n_tasks)

    def get_name_of_shared_parameters(self, layers_branched):
        """
        Get the name of shared parameters
        """
        for idx, m in self.named_modules():
            if '.m_list' in idx:
                idx = idx[:idx.index('.m_list')]

            if idx not in layers_branched:
                members = m._parameters.items()
                memo = set()
                for k, v in members:
                    if v is None or v in memo:
                        continue
                    memo.add(v)
                    name = idx + ('.' if idx else '') + k
                    self._shared_parameters[name] = v

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = [torch.flatten(x_i, 1) for x_i in x]
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet18_recon(**kwargs):
    return ResNet_recon(BasicBlock, [2, 2, 2, 2], **kwargs)


@ARCH_REGISTRY.register()
class MnistResNet_recon(torch.nn.Module):
    def __init__(self, tasks=None, branch_type='no_branch', topK=1, conflict_scores_file=None):
        super(MnistResNet_recon, self).__init__()
        if tasks is None:
            self.tasks = ['Task1']
        else:
            self.tasks = tasks

        self.n_tasks = len(tasks)
        self.feature_extractor = resnet18_recon(num_classes=100, tasks=tasks, branch_type=branch_type, topK=topK,
                                             conflict_scores_file=conflict_scores_file)

        self.ReLU = ReLU_recon(inplace=True)
        self.pred = Linear_recon(100, 10, n_tasks=self.n_tasks)

    def shared_parameters(self):
        return self.feature_extractor.shared_parameters()

    def turn(self, layers_branched):
        self.feature_extractor.turn(layers_branched)

    # zero the gradients of shared parameters
    def zero_grad_shared_modules(self):
        for name, p in self.shared_parameters().items():
            if p.grad is not None:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()

    # return the size of the model
    def model_size(self, unit='MB'):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()

        if unit == 'MB':
            size_all_out = param_size / 1024 ** 2
        elif unit == 'B':
            size_all_out = param_size
        else:
            raise ValueError(f'Error: Do not support unit {unit}')

        return size_all_out

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.ReLU(x)
        x = self.pred(x)

        x = {task: x[k] for k, task in enumerate(self.tasks)}

        return x


if __name__ == '__main__':
    # net = MnistResNet_recon(branched='branch')
    # sz = net.model_size()
    # print(f'branch: {sz} MB')

    # net = MnistResNet_recon(branched='empty')
    # normal_sz = net.model_size()
    # print(f'Normal: {normal_sz:.2f} MB')
    #
    # net = MnistResNet_recon(n_tasks=1, branched='empty')
    # sz = net.model_size()
    # print(f'STL: {sz * 2:.2f} MB')
    #
    # net = MnistResNet_recon(branched='branch', topK=25)
    # sz = net.model_size()
    # print(f'Recon: {sz:.2f} MB')
    #
    # net = MnistResNet_recon(branched='bmtas')
    # sz = net.model_size()
    # print(f'BMTAS: {sz:.2f} MB')
    #
    # n_tasks = 2
    # roto_param = nn.Parameter(torch.eye(100))
    # roto_param_size = roto_param.nelement() * roto_param.element_size() * n_tasks / 1024 ** 2
    #
    # print(f'Roto: {roto_param_size + normal_sz:.2f}')

    def parameter_sz(p_list, unit='B'):
        param_size = 0
        for param in p_list:
            param_size += param.nelement() * param.element_size()

        if unit == 'MB':
            size = param_size / 1024 ** 2
        elif unit == 'B':
            size = param_size
        else:
            raise ValueError(f'Error: Do not support unit {unit}')

        return size


    #
    #
    def get_layer_dict(m):
        shared_parameters = m.shared_parameters()
        name_list = list(shared_parameters.keys())
        param_list = list(shared_parameters.values())

        layer_dict = {}
        for i, name in enumerate(name_list):
            if '.weight' in name:
                name = name.replace('.weight', '')
            elif '.bias' in name:
                name = name.replace('.bias', '')

            if name not in layer_dict:
                layer_dict[name] = [param_list[i]]
            else:
                layer_dict[name].append(param_list[i])

        return layer_dict


    import numpy as np
    import random

    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    #
    # randomly select layers and turn them to task-specific parameters
    net = MnistResNet_recon(branched='empty')
    layer_dict = get_layer_dict(net)
    layer_names = list(layer_dict.keys())

    # fist K layers
    topK = 25
    # first_K_layers = layer_names[:topK]
    # saved_file = f'./saved/seed{seed}_first_{topK}Layers.json'
    # with open(saved_file, "w") as fp:
    #     json.dump(first_K_layers, fp)
    #
    # last_K_layers = layer_names[-topK:]
    # saved_file = f'./saved/seed{seed}_last_{topK}Layers.json'
    # with open(saved_file, "w") as fp:
    #     json.dump(last_K_layers, fp)

    saved_file = f'./saved/seed{seed}_first_{topK}Layers.json'
    with open(saved_file, "r") as fp:
        b = json.load(fp)
    net = MnistResNet_recon(branched='ablation', conflict_scores_file=saved_file)
    print(f'first{topK} sz: {net.model_size():.2f}')

    saved_file = f'./saved/seed{seed}_last_{topK}Layers.json'
    with open(saved_file, "r") as fp:
        b = json.load(fp)
    net = MnistResNet_recon(branched='ablation', conflict_scores_file=saved_file)

    print(f'last{topK} sz: {net.model_size():.2f}')

    #
    # # fix the number of layers
    # topK=25
    # n_layers = len(layer_names)
    # permute = np.random.permutation(n_layers)
    # layer_name_permute = [layer_names[p] for p in permute]
    # random_layer_names = layer_name_permute[:topK]
    #
    # saved_file = f'./saved/seed{seed}_random_select_{topK}Layers.json'
    #
    # with open(saved_file, "w") as fp:
    #     json.dump(random_layer_names, fp)
    #
    # #--------------------------------------------------------------------------------------------------------------------------
    # # fix the size of models
    # branch_net = MnistResNet_recon(branched='branch', topK=25)
    # branch_net_sz = branch_net.model_size(unit='B')
    # sz = net.model_size(unit='B')
    # sz_bound = branch_net_sz - sz
    # param_sz = [parameter_sz(layer_dict[name]) for name in layer_name_permute]
    #
    # p_sz_total = 0
    # n_select_layers = 0
    # for i, p_sz in enumerate(param_sz):
    #     p_sz_total += p_sz
    #     if p_sz_total > sz_bound:
    #         n_select_layers = i + 1
    #         break
    #
    # random_layer_names = layer_name_permute[:n_select_layers]
    # saved_file = f'./saved/seed{seed}_random_select_comparable_size.json'
    #
    # with open(saved_file, "w") as fp:
    #     json.dump(random_layer_names, fp)

    # saved_file = f'./saved/seed{0}_random_select_comparable_size.json'
    # with open(saved_file, "r") as fp:
    #     b = json.load(fp)
    # net = MnistResNet_recon(branched='ablation', conflict_scores_file=saved_file)
    # print(f'saved_file sz: {net.model_size():.2f}')
    #
    # saved_file = f'./saved/seed{1}_random_select_comparable_size.json'
    # with open(saved_file, "r") as fp:
    #     b = json.load(fp)
    # net = MnistResNet_recon(branched='ablation', conflict_scores_file=saved_file)
    # print(f'saved_file sz: {net.model_size():.2f}')
    #
    #
    # saved_file = f'./saved/seed{2}_random_select_comparable_size.json'
    # with open(saved_file, "r") as fp:
    #     b = json.load(fp)
    # net = MnistResNet_recon(branched='ablation', conflict_scores_file=saved_file)
    # print(f'saved_file sz: {net.model_size():.2f}')
    #
    # saved_file = f'./saved/seed{0}_random_select_25Layers.json'
    # with open(saved_file, "r") as fp:
    #     b = json.load(fp)
    # net = MnistResNet_recon(branched='ablation', conflict_scores_file=saved_file)
    # print(f'saved_file sz: {net.model_size():.2f}')
    #
    # saved_file = f'./saved/seed{1}_random_select_25Layers.json'
    # with open(saved_file, "r") as fp:
    #     b = json.load(fp)
    # net = MnistResNet_recon(branched='ablation', conflict_scores_file=saved_file)
    # print(f'saved_file sz: {net.model_size():.2f}')
    #
    # saved_file = f'./saved/seed{2}_random_select_25Layers.json'
    # with open(saved_file, "r") as fp:
    #     b = json.load(fp)
    # net = MnistResNet_recon(branched='ablation', conflict_scores_file=saved_file)
    # print(f'saved_file sz: {net.model_size():.2f}')

    print('finished!')

