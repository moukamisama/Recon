import torch
import torch.nn as nn
import json
from collections import OrderedDict
from utils.registry import ARCH_REGISTRY
from archs.modules_recon import Conv2d_recon, BatchNorm2d_recon, ReLU_recon, Sigmoid_recon, \
    MaxUnpool2d_recon, MaxPool2d_recon, cat_recon, dot_recon, interpolate_recon

def is_father_str(str, str_list):
    for s in str_list:
        if s in str:
            return True
    return False

@ARCH_REGISTRY.register()
class SegNet_recon(nn.Module):
    def __init__(self, class_nb, tasks, branch_type='no_branch', topK=0, conflict_scores_file=None):
        """
        :param class_nb: number of classes for segmentation
        :param tasks: list of tasks
        :param branch_type: 'no_branch' or 'branched'
        :param topK: topK layers that are branch_type
        :param conflict_scores_file: file that contains the conflict scores
        """

        super(SegNet_recon, self).__init__()
        self.class_nb = class_nb

        # added: tasks relevant
        if tasks is None:
            tasks = ['Task1']
        self.tasks = tasks
        self.n_tasks = len(self.tasks)

        # initialise network parameters
        filter = [64, 128, 256, 512, 512]

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))

        self.pred_task1, self.pred_task2, self.pred_task3 = None, None, None

        for task in self.tasks:
            if task == 'semantic':
                self.semantic_pred = self.conv_layer([filter[0], self.class_nb], pred=True)
            if task == 'depth':
                self.depth_pred = self.conv_layer([filter[0], 1], pred=True)
            if task == 'normal':
                self.normal_pred = self.conv_layer([filter[0], 3], pred=True)

        # define pooling and unpooling functions
        self.max_pool = MaxPool2d_recon(kernel_size=2, stride=2)

        self.down_sampling = MaxPool2d_recon(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = MaxUnpool2d_recon(kernel_size=2, stride=2)

        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

        self._shared_parameters = OrderedDict()

        # added: obtain the defined task dependent layers
        td_layers = []
        for idx, m in self.named_modules():
            members = m._parameters.items()
            memo = set()
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = idx + ('.' if idx else '') + k

                if is_father_str(name, self.task_dependent_modules_names()):
                    td_layers.append(name[:idx.index('.m_list')])

        self.td_layers = list(set(td_layers))

        #--------------------------------------------------------------------------------------------------
        # added: insert task branches
        if branch_type == 'no_branch':
            layers_branched = []
        elif branch_type == 'branched':
            if conflict_scores_file is None:
                raise ValueError("conflict_scores_file should be provided when branch_type is 'branched'")

            with open(conflict_scores_file, "r") as fp:
                layers_branched = json.load(fp)
                layers_branched = list(layers_branched.keys())
                layers_branched = layers_branched[:topK]
        else:
            raise ValueError("branch_type should be 'no_branch' or 'branched'")

        self.turn(layers_branched)
        self.get_name_of_shared_parameters(layers_branched)

        #--------------------------------------------------------------------------------------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

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
        layers_branched = layers_branched + self.td_layers

        # obtain the shared parameters
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

        # delete logsigma
        del self._shared_parameters['logsigma']

    def shared_parameters(self):
        return self._shared_parameters

    def shared_modules_name(self):
        return ['encoder_block', 'decoder_block',
                'conv_block_enc', 'conv_block_dec',
                'down_sampling', 'up_sampling']

    def task_dependent_modules_names(self):
        return ['semantic_pred', 'depth_pred', 'normal_pred']

    def zero_grad_shared_modules(self):
        for name, p in self.shared_parameters().items():
            if p.grad is not None:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()

    def add_branch_layers(self, layer_list):
        for idx, m in self.named_modules():
            if idx in layer_list:
                m.set_n_tasks(n_tasks=self.n_tasks)

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                OrderedDict([
                    ('conv1', Conv2d_recon(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1)),
                    ('bn1', BatchNorm2d_recon(num_features=channel[1])),
                    ('relu1', ReLU_recon(inplace=True))
                ])
            )
        else:
            conv_block = nn.Sequential(
                OrderedDict([
                    ('conv1', Conv2d_recon(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1)),
                    ('conv2', Conv2d_recon(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0))
                ])
            )
        return conv_block

    def att_layer(self, channel, n_tasks=1):
        att_block = nn.Sequential(
            OrderedDict([
                ('conv1',
                 Conv2d_recon(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0, n_tasks=n_tasks)),
                ('bn1', BatchNorm2d_recon(channel[1], n_tasks=n_tasks)),
                ('relu1', ReLU_recon(inplace=True)),
                ('conv2',
                 Conv2d_recon(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0, n_tasks=n_tasks)),
                ('bn2', BatchNorm2d_recon(channel[2], n_tasks=n_tasks)),
                ('Sigmoid2', Sigmoid_recon())
            ])
        )
        return att_block

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
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])


        # assert len(g_decoder[-1][-1]) == 1, 'Error'

        output = {}
        if len(g_decoder[-1][-1]) == 1:
            for task in self.tasks:
                if task == 'semantic':
                    output[task] = self.semantic_pred(g_decoder[-1][-1][0])[0]
                if task == 'depth':
                    output[task] = self.depth_pred(g_decoder[-1][-1][0])[0]
                if task == 'normal':
                    t3_pred = self.normal_pred(g_decoder[-1][-1][0])[0]
                    output[task] = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)
        if len(g_decoder[-1][-1]) == self.n_tasks:
            for task in self.tasks:
                if task == 'semantic':
                    output[task] = self.semantic_pred(g_decoder[-1][-1][0])[0]
                if task == 'depth':
                    output[task] = self.depth_pred(g_decoder[-1][-1][1])[0]
                if task == 'normal':
                    t3_pred = self.normal_pred(g_decoder[-1][-1][2])[0]
                    output[task] = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return output