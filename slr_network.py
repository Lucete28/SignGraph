import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet as resnet
from itertools import groupby

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs

class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()

        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None, sample_ids=None):
        # 영상 입력일 경우: x.shape == [B, T, C, H, W]
        if len(x.shape) == 5:
            batch, temp, channel, height, width = x.shape
            # [B, T, C, H, W] → [B, C, T, H, W] then pass through conv2d → 결과 reshape하여 [B, C, T]
            framewise = self.conv2d(x.permute(0, 2, 1, 3, 4)).view(batch, temp, -1).permute(0, 2, 1)
        else:
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: [T, B, C] from conv1d visual feature
        x = conv1d_outputs['visual_feat']

        # lgt를 보장: conv1d_outputs['feat_len']가 배치 당 하나의 값이어야 합니다.
        lgt = conv1d_outputs['feat_len']
        if not isinstance(lgt, torch.Tensor):
            lgt = torch.tensor(lgt, device=x.device)
        lgt = lgt.view(-1)  # [B] 형태로

        # temporal model: 입력은 [T, B, C]와 lgt (기본 CPU tensor)
        tm_outputs = self.temporal_model(x, lgt.cpu())
        outputs = self.classifier(tm_outputs['predictions'])

        if not self.training:
            pred = self.decoder.decode(outputs, lgt, batch_first=False, probs=False, sample_ids=sample_ids)
            conv_pred = self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False, sample_ids=sample_ids)
        else:
            pred = None
            conv_pred = None
        # print(f"[forward] lgt (feat_len): {lgt.shape}, values: {lgt}")

        return {
            "framewise_features": framewise,
            "visual_features": x,
            "temproal_features": tm_outputs['predictions'],
            "feat_len": lgt.to(x.device),  # [B] torch.Tensor
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred if not self.training else None,
            "recognized_sents": pred if not self.training else None,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0

        # 👇 input_lengths (feat_len): must be [B] shape
        feat_len = ret_dict["feat_len"]
        if isinstance(feat_len, (list, tuple)):
            feat_len = torch.cat([f.view(-1) if isinstance(f, torch.Tensor) else torch.tensor([f]) for f in feat_len], dim=0)
        elif isinstance(feat_len, torch.Tensor):
            feat_len = feat_len.view(-1)
        else:
            raise TypeError(f"Unsupported feat_len type: {type(feat_len)}")

        feat_len = feat_len.cpu().int()  # ✅ CTC 요구 형식

        label_lgt_cpu = label_lgt.cpu().int()
        label_cpu = label.cpu().int()

        batch_size = label_lgt_cpu.size(0)

        # # ✅ 디버깅 메시지 (선택적)
        # print(f"[criterion] feat_len: {feat_len.shape}, values: {feat_len}")
        # print(f"[criterion] label_lgt: {label_lgt_cpu.shape}, values: {label_lgt_cpu}")

        if feat_len.size(0) != batch_size:
            print(f"⚠️ Warning: feat_len ({feat_len.size(0)}) != label_lgt ({batch_size}), attempting to align...")
            min_len = min(feat_len.size(0), batch_size)
            feat_len = feat_len[:min_len]
            label_lgt_cpu = label_lgt_cpu[:min_len]
            label_cpu = label_cpu[:label_lgt_cpu.sum()]  # target 전체 길이 축소

        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](
                    ret_dict["conv_logits"].log_softmax(-1),
                    label_cpu,
                    feat_len,
                    label_lgt_cpu
                ).mean()

            elif k == 'SeqCTC':
                loss_ctc = self.loss['CTCLoss'](
                    ret_dict["sequence_logits"].log_softmax(-1),
                    label_cpu,
                    feat_len,
                    label_lgt_cpu
                )
                loss += weight * loss_ctc.mean()

            elif k == 'Dist':
                loss += weight * self.loss['distillation'](
                    ret_dict["conv_logits"],
                    ret_dict["sequence_logits"].detach(),
                    use_blank=False
                )

            elif k == 'LengthPenalty':
                from itertools import groupby
                logits = ret_dict["sequence_logits"].permute(1, 0, 2)
                pred_ids = torch.argmax(logits, dim=-1)

                predicted_lengths = []
                for b in range(pred_ids.shape[0]):
                    curr_len = int(feat_len[b].item())
                    pred = pred_ids[b, :curr_len]
                    grouped = [x for x, _ in groupby(pred.tolist()) if x != 0]
                    predicted_lengths.append(len(grouped))

                true_lens = label_lgt_cpu.tolist()
                penalty = sum((p - t) ** 2 for p, t in zip(predicted_lengths, true_lens)) / len(predicted_lengths)
                loss += weight * penalty

        return loss



    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
