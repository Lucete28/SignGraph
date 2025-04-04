import os
import pdb
import time
import torch
import ctcdecode
import numpy as np
from itertools import groupby
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ⬇ 프레임 길이 저장용 전역 리스트
frame_lengths = []

def plot_timewise_predictions(nn_output, gloss_dict, vid_lgt, batch_idx=0, blank_id=0, sample_id=None):
    i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
    probs = torch.softmax(nn_output, dim=-1)
    pred_ids = torch.argmax(probs, dim=-1)

    length = int(vid_lgt[batch_idx].item())
    pred_seq = pred_ids[batch_idx, :length].cpu().numpy()
    x = np.arange(length)

    plt.figure(figsize=(15, 4))
    plt.plot(x, pred_seq, drawstyle='steps-post', label='Predicted Gloss ID', linewidth=1.5)

    blank_indices = np.where(pred_seq == blank_id)[0]
    if len(blank_indices) > 0:
        plt.scatter(blank_indices, pred_seq[blank_indices], color='red', marker='x', label='CTC Blank (0)', zorder=5)

    title_str = "Predicted Gloss ID over Time (CTC Blank Highlighted)"
    if sample_id:
        title_str += f"\nSample: {sample_id}"
    plt.title(title_str)
    plt.xlabel("Time Step")
    plt.ylabel("Gloss ID")
    plt.yticks(np.unique(pred_seq))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0, beam_width=10):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(vocab, beam_width=beam_width, blank_id=blank_id, num_processes=10)

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False, sample_ids=None):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)

        # ⬇ 프레임 길이 수집
        global frame_lengths
        for i in range(vid_lgt.size(0)):
            frame_lengths.append(int(vid_lgt[i].item()))

        # sample_id가 존재하면 시각화
        sample_id = sample_ids[0] if sample_ids else None
        # plot_timewise_predictions(nn_output, self.i2g_dict, vid_lgt, batch_idx=0, sample_id=sample_id)

        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            ret_list.append([self.i2g_dict[int(gloss_id)] for gloss_id in first_result])
        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [x for x in group_result if x != self.blank_id]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in enumerate(max_result)])
        return ret_list


# ✅ 테스트 루프 끝에서 프레임 길이 분석 (예: seq_eval() 마지막에)
def analyze_frame_lengths():
    if not frame_lengths:
        print("⚠ 분석할 frame_lengths가 없습니다.")
        return

    print("\n📊 Test Video Frame Length Analysis:")
    print(f"- Total samples: {len(frame_lengths)}")
    print(f"- Mean length : {np.mean(frame_lengths):.2f}")
    print(f"- Min length  : {np.min(frame_lengths)}")
    print(f"- Max length  : {np.max(frame_lengths)}")

    # 히스토그램 시각화
    plt.figure(figsize=(10, 5))
    plt.hist(frame_lengths, bins=20, color='skyblue', edgecolor='black')
    plt.title("Test Video Frame Length Distribution")
    plt.xlabel("Frame Length")
    plt.ylabel("Number of Samples")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
