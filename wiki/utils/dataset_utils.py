import re
import os
import torch
import pandas as pd

from torch import nn
from typing import Callable, Union, Tuple
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from .config import Config

COM_SEP = 'COM_SEP'
TEXT_SEP_SRC = 'TEXT_SEP'
TEXT_SEP_TGT = 'TEXT_SEP'
DOCS_SEP = 'DOCS_SEP'

def get_tgt(row: pd.core.series.Series, 
            text_to_lower: bool = True, 
            comment_to_lower: bool = True, 
            text_delimiter: str = TEXT_SEP_TGT, 
            comment_delimiter: str = COM_SEP) -> str:
    '''
    Gets dataset row and returns target text
    '''
    assert text_delimiter != '', 'Text separator can not be empty'
    text_delimiter = f' {text_delimiter} '
    comment_delimiter = f' {comment_delimiter} '
    
    new_text = row.new_text.lower() if text_to_lower else row.new_text
    coms = row.comment.lower() if comment_to_lower else row.comment
    tgt = comment_delimiter + coms + text_delimiter + new_text
    return tgt.strip()

def get_src(row: pd.core.series.Series, 
            text_to_lower: bool = True, 
            comment_to_lower: bool = True, 
            text_delimiter: str = TEXT_SEP_SRC,
            doc_delimiter: str = DOCS_SEP) -> str:
    '''
    Gets dataset row and returns source text
    '''
    assert doc_delimiter != '', 'Docs separator can not be empty'
    text_delimiter = f' {text_delimiter} '
    doc_delimiter = f' {doc_delimiter} '
    
    old_text = row.old_text.lower() if text_to_lower else row.old_text 
    docs = row.docs_processed.lower() if text_to_lower else row.docs_processed 
    src = text_delimiter + old_text + doc_delimiter + docs
    return src.strip()

def get_tgt_explain(row: pd.core.series.Series, 
            text_to_lower: bool = True, 
            comment_to_lower: bool = True, 
            text_delimiter: str = TEXT_SEP_TGT, 
            comment_delimiter: str = COM_SEP) -> str:
    '''
    Gets dataset row and returns target text
    '''
    assert text_delimiter != '', 'Text separator can not be empty'
    text_delimiter = f' {text_delimiter} '
    comment_delimiter = f' {comment_delimiter} '
    
    # new_text = row.new_text.lower() if text_to_lower else row.new_text
    coms = row.comment.lower() if comment_to_lower else row.comment
    # tgt = comment_delimiter + coms + text_delimiter + new_text
    return coms

def get_src_explain(row: pd.core.series.Series, 
            text_to_lower: bool = True, 
            comment_to_lower: bool = True, 
            text_delimiter: str = TEXT_SEP_SRC,
            doc_delimiter: str = DOCS_SEP) -> str:
    '''
    Gets dataset row and returns source text
    '''
    assert doc_delimiter != '', 'Docs separator can not be empty'
    text_delimiter = f' {text_delimiter} '
    doc_delimiter = f' {doc_delimiter} '
    
    old_text = row.old_text.lower() if text_to_lower else row.old_text 
    new_text = row.new_text.lower() if text_to_lower else row.new_text
    docs = row.docs_processed.lower() if text_to_lower else row.docs_processed 
    src = old_text + text_delimiter + new_text + doc_delimiter + docs
    return src.strip()

def extract_com8text_from_tgt(tgt: str, text_delimiter: str = TEXT_SEP_TGT) -> Tuple[str, str]:
    assert text_delimiter != '', 'Text separator can not be empty'
    if text_delimiter in tgt:
        splitted = tgt.split(text_delimiter)
        tgt_comment = splitted[0].strip()
        tgt_text = ' '.join(splitted[1:]).strip()
        return tgt_comment, tgt_text
    return 'error', 'error'

def extract_text8docs_from_src(src: str, 
                               text_delimiter: str = TEXT_SEP_SRC, 
                               doc_delimiter: str = DOCS_SEP) -> Tuple[str, str]:
    assert doc_delimiter != '', 'Docs separator can not be empty'
    
    if text_delimiter != '' and text_delimiter in src: 
        src_text = ' '.join(src.split(text_delimiter)[1:]).strip()
    else: 
        src_text = src
        
    if doc_delimiter in src_text:
        splitted = src_text.split(doc_delimiter)
        src_text = splitted[0].strip()
        docs_text = ' '.join(splitted[1:]).strip()
        return src_text, docs_text
    return 'error', 'error'

    
class EditDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset: pd.DataFrame, 
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
                 config: Config,
                 text_to_lower: bool = True, 
                 comment_to_lower: bool = True):
        self.db = dataset
        self.tokenizer = tokenizer
        
        src_text = self.db.apply(lambda x: get_src(x, text_to_lower, comment_to_lower), axis=1).values
        tgt_text = self.db.apply(lambda x: get_tgt(x, text_to_lower, comment_to_lower), axis=1).values
        
        self.src_text_tokenized = [tokenizer(x,
                                       max_length=config.src_max_len,
                                       truncation=True,
                                       return_attention_mask=False,
                                       ) for x in src_text]
        self.tgt_text_tokenized = [tokenizer(x,
                                       max_length=config.tgt_max_len,
                                       truncation=True,
                                       return_attention_mask=False,
                                       ) for x in tgt_text]

    def __len__(self):
        return len(self.db)

    def __getitem__(self, 
                    idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.src_text_tokenized[idx]
        tgt = self.tgt_text_tokenized[idx]
        return src, tgt

    @staticmethod
    def collate_fn(samples: Tuple[torch.Tensor, torch.Tensor], 
                   tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
                   config: Config) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        src_samples = [x[0] for x in samples]
        tgt_samples = [x[1] for x in samples]

        src_samples = tokenizer.pad(src_samples,
                                    padding='longest',
                                    max_length=config.src_max_len,
                                    return_attention_mask=False,
                                    return_tensors='pt')['input_ids']

        tgt_samples = tokenizer.pad(tgt_samples,
                                    padding='longest',
                                    max_length=config.tgt_max_len,
                                    return_attention_mask=False,
                                    return_tensors='pt')['input_ids']

        return (src_samples, tgt_samples), torch.ones(len(samples), 1)
    
    
class ExplainDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset: pd.DataFrame, 
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
                 config: Config,
                 text_to_lower: bool = True, 
                 comment_to_lower: bool = True):
        self.db = dataset
        self.tokenizer = tokenizer
        
        src_text = self.db.apply(lambda x: get_src_explain(x, text_to_lower, comment_to_lower), axis=1).values
        tgt_text = self.db.apply(lambda x: get_tgt_explain(x, text_to_lower, comment_to_lower), axis=1).values
        
        self.src_text_tokenized = [tokenizer(x,
                                       max_length=config.src_max_len,
                                       truncation=True,
                                       return_attention_mask=False,
                                       ) for x in src_text]
        self.tgt_text_tokenized = [tokenizer(x,
                                       max_length=config.tgt_max_len,
                                       truncation=True,
                                       return_attention_mask=False,
                                       ) for x in tgt_text]

    def __len__(self):
        return len(self.db)

    def __getitem__(self, 
                    idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.src_text_tokenized[idx]
        tgt = self.tgt_text_tokenized[idx]
        return src, tgt

    @staticmethod
    def collate_fn(samples: Tuple[torch.Tensor, torch.Tensor], 
                   tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
                   config: Config) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        src_samples = [x[0] for x in samples]
        tgt_samples = [x[1] for x in samples]

        src_samples = tokenizer.pad(src_samples,
                                    padding='longest',
                                    max_length=config.src_max_len,
                                    return_attention_mask=False,
                                    return_tensors='pt')['input_ids']

        tgt_samples = tokenizer.pad(tgt_samples,
                                    padding='longest',
                                    max_length=config.tgt_max_len,
                                    return_attention_mask=False,
                                    return_tensors='pt')['input_ids']

        return (src_samples, tgt_samples), torch.ones(len(samples), 1)