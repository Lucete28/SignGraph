import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from einops import rearrange
from collections import defaultdict
from utils.metrics import wer_list 
from utils.misc import *



def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    optimizer.scheduler.step(epoch_idx)
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    scaler = GradScaler()
    for batch_idx, data in enumerate(tqdm(loader)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        optimizer.zero_grad()
        with autocast():

            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            if len(device.gpu_list)>1:
                loss = model.module.criterion_calculation(ret_dict, label, label_lgt)
            else:
                loss = model.criterion_calculation(ret_dict, label, label_lgt)

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('loss is nan')
            print(str(data[1])+'  frames', str(data[3])+'  glosses')
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update() 
        if len(device.gpu_list)>1:
            torch.cuda.synchronize() 
            torch.distributed.reduce(loss, dst=0)

        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0 and is_main_process():
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
        del ret_dict
        del loss
    optimizer.scheduler.step()
    if is_main_process():
        recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return


import csv 
from jiwer import wer as jiwer_wer
def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder, evaluate_tool="python"):
    model.eval()
    results = defaultdict(dict)

    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        info = [d['fileid'] for d in data[-1]]
        gloss = [d['label'] for d in data[-1]]
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            for inf, conv_sents, recognized_sents, gl in zip(info, ret_dict['conv_sents'], ret_dict['recognized_sents'], gloss):
                results[inf]['conv_sents'] = conv_sents
                results[inf]['recognized_sents'] = recognized_sents
                results[inf]['gloss'] = gl

    gls_hyp = [' '.join(results[n]['conv_sents']) for n in results]
    gls_ref = [results[n]['gloss'] for n in results]
    wer_results_con = wer_list(hypotheses=gls_hyp, references=gls_ref)

    gls_hyp = [' '.join(results[n]['recognized_sents']) for n in results]
    wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)

    reg_per = wer_results if wer_results['wer'] < wer_results_con['wer'] else wer_results_con

    recoder.print_log('\tEpoch: {} {} done. Conv wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results_con['wer'], wer_results_con['ins'], wer_results_con['del']),
        f"{work_dir}/{mode}.txt")

    recoder.print_log('\tEpoch: {} {} done. LSTM wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results['wer'], wer_results['ins'], wer_results['del']),
        f"{work_dir}/{mode}.txt")

    # ✅ 전체 결과 CSV로 저장
    save_folder = os.path.join(work_dir, f"{mode}_detailed_results")
    os.makedirs(save_folder, exist_ok=True)
    csv_path = os.path.join(save_folder, "wer_results.csv")

    rows = []
    for file_id in results:
        gt = results[file_id]['gloss']
        conv_pred = ' '.join(results[file_id]['conv_sents'])
        lstm_pred = ' '.join(results[file_id]['recognized_sents'])
        conv_wer = jiwer_wer(gt, conv_pred)
        lstm_wer = jiwer_wer(gt, lstm_pred)

        rows.append([
            file_id,
            gt,
            conv_pred,
            f"{conv_wer:.4f}",
            lstm_pred,
            f"{lstm_wer:.4f}"
        ])

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_id', 'gt', 'conv_pred', 'conv_wer', 'lstm_pred', 'lstm_wer'])
        writer.writerows(rows)
    print(len(rows))

    print(f"\n✅ 전체 결과가 다음 경로에 저장되었습니다:\n{csv_path}\n")



    # WER 기준 상위 5개 샘플 출력
    sample_wers = []
    for file_id in results:
        gt = results[file_id]['gloss']
        conv_pred = ' '.join(results[file_id]['conv_sents'])
        lstm_pred = ' '.join(results[file_id]['recognized_sents'])
    
        conv_wer = jiwer_wer(gt, conv_pred)
        lstm_wer = jiwer_wer(gt, lstm_pred)
    
        sample_wers.append({
            'file_id': file_id,
            'gt': gt,
            'conv_pred': conv_pred,
            'conv_wer': conv_wer,
            'lstm_pred': lstm_pred,
            'lstm_wer': lstm_wer,
            'max_wer': max(conv_wer, lstm_wer)
        })

    top5 = sorted(sample_wers, key=lambda x: x['max_wer'], reverse=True)[:5]
    # 전체 샘플 WER 평균 계산
    avg_conv_wer = 100 * np.mean([sample['conv_wer'] for sample in sample_wers])
    avg_lstm_wer = 100 * np.mean([sample['lstm_wer'] for sample in sample_wers])

    print("\n📊 전체 평균 WER")
    print(f"- Conv 방식 평균 WER: {avg_conv_wer:.2f}%")
    print(f"- LSTM 방식 평균 WER: {avg_lstm_wer:.2f}%")

    print("\n📢 WER 상위 5개 샘플 (Conv vs LSTM):\n")
    for sample in top5:
        print(f"[{sample['file_id']}] WER (Conv: {sample['conv_wer']:.4f}, LSTM: {sample['lstm_wer']:.4f})")
        print(f"GT   : {sample['gt']}")
        print(f"Conv : {sample['conv_pred']}")
        print(f"LSTM : {sample['lstm_pred']}")
        print("-" * 60)
























    return {"wer": reg_per['wer'], "ins": reg_per['ins'], 'del': reg_per['del']}
