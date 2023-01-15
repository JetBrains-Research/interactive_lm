import torch
import catalyst

from easse.sari import corpus_sari
from rouge import Rouge 
from copy import deepcopy
from typing import Callable, Union, Tuple
from catalyst.core import IRunner
from catalyst.metrics._additive import AdditiveMetric
from catalyst.callbacks.metric import BatchMetricCallback, ICallbackBatchMetric
from collections import defaultdict
from collections import OrderedDict
from nltk.tokenize import sent_tokenize

from .config import Config
from .difflibparser import DifflibParser, DiffCode
from .dataset_utils import extract_com8text_from_tgt, extract_text8docs_from_src

def get_diffs_for_em(src: Tuple[str], tgt: Tuple[str]) -> Tuple[str]:
    diff_result = list(DifflibParser(src, tgt))
    result = []
    for dif_id, dif_line in enumerate(diff_result):
        if dif_line['code'] != DiffCode.SIMILAR:
            nl_ = ''
            ol_ = dif_line['line']
            code = dif_line['code']
            if 'newline' in dif_line:
                nl_ = dif_line['newline']
            dif_line_str = f'{ol_}_{nl_}_{code}'
            result.append(dif_line_str)
    return set(result)

def topN_diff_exact_match_one(src: str, 
                              tgt: str, 
                              pred: Tuple[str],
                              sentence_tokenizer: Callable = sent_tokenize) -> float:
    '''
    Checks if any of pred hypotheses matches with target
    '''
    src_tokenized = sentence_tokenizer(src)
    tgt_tokenized = sentence_tokenizer(tgt)
    pred_tokenized = [sentence_tokenizer(x) for x in pred]
    
    
    tgt_diff = get_diffs_for_em(src_tokenized, tgt_tokenized)
    pred_diff = [get_diffs_for_em(src_tokenized, prediction) for prediction in pred_tokenized]
    
    metrics = []
    for p_dif in pred_diff:
        max_ = max([len(tgt_diff), len(p_dif), 1])
        inter_ = len(tgt_diff.intersection(p_dif))
        metrics.append(inter_ / max_)
             
    return max(metrics)

def topN_exact_match_one(src: str, tgt: str, pred: Tuple[str]) -> float:
    '''
    Checks if any of pred hypotheses matches with target
    '''
    return float(tgt in pred)


def sari_one(src: str, tgt: str, pred: Tuple[str]) -> float:
    '''
    Returns max sari for pred hypotheses
    '''
    sari_metrics = []
    for prediction in pred:
        sari_metrics.append(
            corpus_sari(orig_sents=[src],  
                        sys_sents=[prediction], 
                        refs_sents=[[tgt]])
        )
    return max(sari_metrics)


def rouge_one(src: str, tgt: str, pred: Tuple[str]) -> float:
    '''
    Returns max agregated rouge metrics dict for pred hypotheses
    '''
    rouge = Rouge()
    rouge_metrics = defaultdict(float)
    for prediction in pred:
        prediction_rouge = [{
                'rouge-1': {'f': 0.0},
                'rouge-2': {'f': 0.0},
                'rouge-l': {'f': 0.0}
        }]
        if len(tgt) > 0 and len(prediction) > 0:
            try:
                prediction_rouge = rouge.get_scores([prediction], [tgt])
            except:
                continue
        rouge_metrics['rouge-1'] = max(rouge_metrics['rouge-1'], prediction_rouge[0]['rouge-1']['f'])
        rouge_metrics['rouge-2'] = max(rouge_metrics['rouge-2'], prediction_rouge[0]['rouge-2']['f'])
        rouge_metrics['rouge-l'] = max(rouge_metrics['rouge-l'], prediction_rouge[0]['rouge-l']['f'])
    return rouge_metrics


class PeerMetricsPattern(ICallbackBatchMetric):
    def __init__(self, metric_names_list, compute_on_call=False):
        super().__init__(compute_on_call=compute_on_call, prefix='', suffix='')
        self.metric_names_list = metric_names_list
        self.metrics: List[AdditiveMetric] = [
            AdditiveMetric(compute_on_call=compute_on_call) for _ in range(len(self.metric_names_list))
        ]

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def update(self, values, n_samples):
        for value, metric in zip(values, self.metrics):
            metric.update(value, n_samples)
        return values

    def update_key_value(self, values, n_samples):
        values = self.update(values, n_samples)
        output = {
            f"{self.prefix}{key}{self.suffix}": round(value, 6)
            for key, value in zip(self.metric_names_list, values)
        }
        return output

    def compute(self):
        means, stds = zip(*(metric.compute() for metric in self.metrics))
        return means, stds

    def compute_key_value(self):
        means, stds = self.compute()
        output_mean = {
            f"{self.prefix}{key}{self.suffix}": round(value, 6)
            for key, value in zip(self.metric_names_list, means)
        }
        output_std = {
            f"{self.prefix}{key}{self.suffix}/std": round(value, 6)
            for key, value in zip(self.metric_names_list, stds)
        }
        return {**output_mean, **output_std}


class PeerEditMetricsCallback(BatchMetricCallback):

    def __init__(self, 
                 metric_names: Tuple[str],
                 tokenizer,
                 bs: Union[int, None] = None):
        bs_arr = []
        metric_dict = OrderedDict()
        for fmn_b in metric_names:
            assert fmn_b.count('@') == 1, f'{fmn_b} metric does not contain bs (throught @)'
            assert fmn_b.count('__') == 1, f'{fmn_b} metric does not contain part (throught __)'
            
            fmn, bs = fmn_b.split('@')
            part, mn = fmn.split('__')
            if part not in metric_dict:
                metric_dict[part] = OrderedDict()
            if mn not in metric_dict[part]:
                metric_dict[part][mn] = OrderedDict()
            metric_dict[part][mn][bs] = 0.0
            bs_arr.append(int(bs))
            
        metrics_arr_names = []
        for part, mn2bs2m in metric_dict.items():
            for mn, bs2m in mn2bs2m.items():
                for bs, m in bs2m.items():
                    metrics_arr_names.append(f"{part}__{mn}@{bs}")
                 
        super().__init__(
            metric = PeerMetricsPattern(metrics_arr_names),
            input_key = 'features', 
            target_key = 'targets', 
            log_on_batch = True
        )
        
        self.tokenizer = tokenizer
        self.metric_names = metrics_arr_names
        
        max_bs = max(bs_arr)
        if isinstance(bs, int):
            assert bs >= max_bs, 'Given bs must be more or equal to max bs in metrics'
            self.bs = bs
        else:
            self.bs = max_bs
            
        self.metric_dict = metric_dict

    def on_batch_end(self, runner: "IRunner") -> None:

        if runner.loader_key == 'train':
            runner.model.train()
            return None
        
        if runner.loader_key.startswith('valid'):
            runner.model.eval()

            src = runner.batch['features'][0]
            tgt = runner.batch['features'][1]
            max_len = tgt.shape[1]
            with torch.no_grad():
                pred = runner.model.pretrained.generate(
                    src.to(runner.engine.device),
                    attention_mask=(src != 0).float().to(runner.engine.device),
                    num_beams=self.bs,
                    num_return_sequences=self.bs,
                    max_length=max_len
                )

            batch_metrics = deepcopy(self.metric_dict)
            
            pred = pred.view(-1, self.bs, pred.shape[1])
            for i in range(tgt.shape[0]):
                tgt_subseq = tgt[i, :pred.shape[2]].to(runner.engine.device)
                
                tgt_full = self.tokenizer.decode(tgt_subseq, skip_special_tokens=True).strip()
                tgt_comment, tgt_text = extract_com8text_from_tgt(tgt_full)
                
                src_full = self.tokenizer.decode(src[i], skip_special_tokens=True).strip()
                src_text, _ = extract_text8docs_from_src(src_full)
                
                pred_full = []
                pred_texts = []
                pred_comments = []
                
                for pred_item in pred[i]:
                    txt_pred = self.tokenizer.decode(pred_item, skip_special_tokens=True)
                    com_prediction, text_prediction = extract_com8text_from_tgt(txt_pred)
                    
                    pred_full.append(txt_pred)
                    pred_texts.append(text_prediction)
                    pred_comments.append(com_prediction)
                
                batch_metrics = self.update_metrics_part('full', batch_metrics, src_full, tgt_full, pred_full)
                batch_metrics = self.update_metrics_part('text', batch_metrics, src_text, tgt_text, pred_texts)
                batch_metrics = self.update_metrics_part('comment', batch_metrics, '', tgt_comment, pred_comments)
                                
            metrics_arr = []
            for part, mn2bs2m in batch_metrics.items():
                for mn, bs2m in mn2bs2m.items():
                    for bs, m in bs2m.items():
                        metrics_arr.append(m / len(src))

            metrics = self.metric.update_key_value(metrics_arr, len(src))
            runner.batch_metrics.update(metrics)
    
    @staticmethod
    def update_metrics_part(part: str, metrics: dict, src: str, tgt: str, pred: Tuple[str]) -> int:
        '''
        Updates metric dict for one object
        '''
        if part not in metrics:
            return metrics
        
        for mn, bs2m in metrics[part].items():
            for bs, m in bs2m.items():
                bs_int = int(bs)

                if 'sari' in mn:
                    metrics[part][mn][bs] = m + sari_one(src, tgt, pred[:bs_int])
                    
                if 'diff_exact_match' in mn:
                    metrics[part][mn][bs] = m + topN_diff_exact_match_one(src, tgt, pred[:bs_int])
                elif 'exact_match' in mn:
                    metrics[part][mn][bs] = m + topN_exact_match_one(src, tgt, pred[:bs_int])

                if 'rouge' in mn:
                    r_metrics = rouge_one(src, tgt, pred[:bs_int])
                    metrics[part][mn][bs] = m + r_metrics[mn]     
        return metrics

    def on_loader_end(self, runner: "IRunner") -> None:
        if runner.loader_key.startswith('valid'):
            metrics = self.metric.compute_key_value()
            metrics = runner.engine.mean_reduce_ddp_metrics(metrics)
            runner.loader_metrics.update(metrics)
            
            
class PeerMetricsPattern(ICallbackBatchMetric):
    def __init__(self, metric_names_list, compute_on_call=False):
        super().__init__(compute_on_call=compute_on_call, prefix='', suffix='')
        self.metric_names_list = metric_names_list
        self.metrics: List[AdditiveMetric] = [
            AdditiveMetric(compute_on_call=compute_on_call) for _ in range(len(self.metric_names_list))
        ]

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def update(self, values, n_samples):
        for value, metric in zip(values, self.metrics):
            metric.update(value, n_samples)
        return values

    def update_key_value(self, values, n_samples):
        values = self.update(values, n_samples)
        output = {
            f"{self.prefix}{key}{self.suffix}": round(value, 6)
            for key, value in zip(self.metric_names_list, values)
        }
        return output

    def compute(self):
        means, stds = zip(*(metric.compute() for metric in self.metrics))
        return means, stds

    def compute_key_value(self):
        means, stds = self.compute()
        output_mean = {
            f"{self.prefix}{key}{self.suffix}": round(value, 6)
            for key, value in zip(self.metric_names_list, means)
        }
        output_std = {
            f"{self.prefix}{key}{self.suffix}/std": round(value, 6)
            for key, value in zip(self.metric_names_list, stds)
        }
        return {**output_mean, **output_std}


class PeerExplainMetricsCallback(BatchMetricCallback):

    def __init__(self, 
                 metric_names: Tuple[str],
                 tokenizer,
                 bs: Union[int, None] = None):
        bs_arr = []
        metric_dict = OrderedDict()
        for fmn_b in metric_names:
            assert fmn_b.count('@') == 1, f'{fmn_b} metric does not contain bs (throught @)'
            assert fmn_b.count('__') == 1, f'{fmn_b} metric does not contain part (throught __)'
            
            fmn, bs = fmn_b.split('@')
            part, mn = fmn.split('__')
            if part not in metric_dict:
                metric_dict[part] = OrderedDict()
            if mn not in metric_dict[part]:
                metric_dict[part][mn] = OrderedDict()
            metric_dict[part][mn][bs] = 0.0
            bs_arr.append(int(bs))
            
        metrics_arr_names = []
        for part, mn2bs2m in metric_dict.items():
            for mn, bs2m in mn2bs2m.items():
                for bs, m in bs2m.items():
                    metrics_arr_names.append(f"{part}__{mn}@{bs}")
                 
        super().__init__(
            metric = PeerMetricsPattern(metrics_arr_names),
            input_key = 'features', 
            target_key = 'targets', 
            log_on_batch = True
        )
        
        self.tokenizer = tokenizer
        self.metric_names = metrics_arr_names
        
        max_bs = max(bs_arr)
        if isinstance(bs, int):
            assert bs >= max_bs, 'Given bs must be more or equal to max bs in metrics'
            self.bs = bs
        else:
            self.bs = max_bs
            
        self.metric_dict = metric_dict

    def on_batch_end(self, runner: "IRunner") -> None:

        if runner.loader_key == 'train':
            runner.model.train()
            return None
        
        if runner.loader_key.startswith('valid'):
            runner.model.eval()

            src = runner.batch['features'][0]
            tgt = runner.batch['features'][1]
            max_len = tgt.shape[1]
            with torch.no_grad():
                pred = runner.model.pretrained.generate(
                    src.to(runner.engine.device),
                    attention_mask=(src != 0).float().to(runner.engine.device),
                    num_beams=self.bs,
                    num_return_sequences=self.bs,
                    max_length=max_len
                )

            batch_metrics = deepcopy(self.metric_dict)
            
            pred = pred.view(-1, self.bs, pred.shape[1])
            for i in range(tgt.shape[0]):
                tgt_subseq = tgt[i, :pred.shape[2]].to(runner.engine.device)
                
                tgt_full = self.tokenizer.decode(tgt_subseq, skip_special_tokens=True).strip()
                
                pred_full = []
                for pred_item in pred[i]:
                    txt_pred = self.tokenizer.decode(pred_item, skip_special_tokens=True)
                    com_prediction, text_prediction = extract_com8text_from_tgt(txt_pred)
                    
                    pred_full.append(txt_pred)
                    
                
                batch_metrics = self.update_metrics_part('full', batch_metrics, '', tgt_full, pred_full)
                # batch_metrics = self.update_metrics_part('text', batch_metrics, src_text, tgt_text, pred_texts)
                # batch_metrics = self.update_metrics_part('comment', batch_metrics, '', tgt_comment, pred_comments)
                                
            metrics_arr = []
            for part, mn2bs2m in batch_metrics.items():
                for mn, bs2m in mn2bs2m.items():
                    for bs, m in bs2m.items():
                        metrics_arr.append(m / len(src))

            metrics = self.metric.update_key_value(metrics_arr, len(src))
            runner.batch_metrics.update(metrics)
    
    @staticmethod
    def update_metrics_part(part: str, metrics: dict, src: str, tgt: str, pred: Tuple[str]) -> int:
        '''
        Updates metric dict for one object
        '''
        if part not in metrics:
            return metrics
        
        for mn, bs2m in metrics[part].items():
            for bs, m in bs2m.items():
                bs_int = int(bs)

                if 'sari' in mn:
                    metrics[part][mn][bs] = m + sari_one(src, tgt, pred[:bs_int])
                    
                if 'diff_exact_match' in mn:
                    metrics[part][mn][bs] = m + topN_diff_exact_match_one(src, tgt, pred[:bs_int])
                elif 'exact_match' in mn:
                    metrics[part][mn][bs] = m + topN_exact_match_one(src, tgt, pred[:bs_int])

                if 'rouge' in mn:
                    r_metrics = rouge_one(src, tgt, pred[:bs_int])
                    metrics[part][mn][bs] = m + r_metrics[mn]     
        return metrics

    def on_loader_end(self, runner: "IRunner") -> None:
        if runner.loader_key.startswith('valid'):
            metrics = self.metric.compute_key_value()
            metrics = runner.engine.mean_reduce_ddp_metrics(metrics)
            runner.loader_metrics.update(metrics)