import wikitextparser as wtp
import nltk
from difflib import Differ 
from typing import Callable, Union, Tuple
from .difflibparser import DifflibParser, DiffCode
import re
from os import listdir
from os.path import isfile, join
from collections import Counter
import os
import numpy as np

MIN_CHANGE_SYMB_LEN = 5
MAX_CHANGE_SYMB_LEN = 300
MAX_PAGE_SYMB_LEN = 20000
MAX_ABSTRACT_LEN = 800
MIN_ABSTRACT_LEN = 10
MIN_COMMENT_LEN = 5
abstarct_tokenizer = lambda x: x

def filter_page(page_name: str):
    '''
    Checks if page name contains latin symbols
    '''
    return bool(re.search('[a-zA-Z]', page_name))

def filter_comment(comment_text: str, 
                   user_name: str):
    '''
    Filters comments that were made by bot or contain only abstract name
    '''
    com_text = comment_text.strip()
    if 'bot' in com_text or 'bot' in user_name:
        return False
    
    if com_text[-2:] == '*/':
        return False
    
    if 'link' in com_text and 'fix' in com_text:
        return False
    
    if ' image ' in com_text or ' image' in com_text or 'image ' in com_text:
        return False
    
    if 'log in' in com_text or 'sorry' in com_text:
        return False
    
    return True

def find_nearest(l: int, r: int, arr: Tuple[int]):
    l_ans, r_ans = -1, -1
    for sent_idx, (l_arr, r_arr) in enumerate(zip(arr, arr[1:])):
        if l_arr <= l < r_arr:
            l_ans = sent_idx
        if l_arr <= r < r_arr:
            r_ans = sent_idx
    if l_ans == -1:
        l_ans = len(arr) - 1
    if r_ans == -1:
        r_ans = len(arr) - 1
    return l_ans, r_ans

def clean_closing(text: str):
    text_sents = text.split('\n')
    ans = []
    for snt in text_sents:
        snt = snt.strip()
        snt = snt.strip()
        if len(snt) > 0:
            if '[' in snt and snt.strip()[-2:] == ']]':
                continue
            if snt.strip()[0] == '|' or snt.strip()[:2] == '{|' or snt.strip()[-2:] == '|}':
                continue
            if snt.strip()[-2:] == '--' or snt.strip()[0] == ':' or snt.strip()[0] == '!':
                continue
        ans.append(snt)
    return '\n'.join(ans).strip()

def clean_text(text: str):
    '''
    Cleans text from rudiments
    '''
    text = text.replace('=====', '==').replace('====', '==').replace('===', '==')
    text = wtp.remove_markup(text)
    text = re.sub('\[\[File.*?]]', '', text, count=0, flags=0)
    text = re.sub('\[\[Category:.*?]]', '', text, count=0, flags=0)
    text = re.sub('\[\[category:.*?]]', '', text, count=0, flags=0)
    # text = wtp.remove_markup(text)    
    text = text.replace('\t', '').replace('\n\n\n', '\n\n').replace('\n\n\n', '\n\n')
    text = re.sub(r'\n..:[^\n]*', '', text)
    text = re.sub(r'\nCategoria:[^\n]*', '', text)
    text = re.sub(r'\nCategory:[^\n]*', '', text)
    text = re.sub(r'\nFile:[^\n]*', '', text)
    text = re.sub(r'\nimage:[^\n]*', '', text)
    text = re.sub(r'\nWikipedia:[^\n]*', '', text)
    text = re.sub(r'\nTemplate:[^\n]*', '', text)
    text = re.sub(r'\nvls:[^\n]*', '', text)
    text = re.sub(r'\nImage:[^\n]*', '', text)
    text = re.sub(r'\n..:[^\n]*', '', text)
    text = re.sub(r'thumb\|[^\n]*', '', text)
    # text = clean_closing(text)
    return text
    
def clean_section_text(text: str):
    text = re.sub('==.*?==+', '', text, count=0, flags=0)
    return text.strip()

def filter_edited_abstract(text: str):
    if '(UTC)' in text:
        return False
    
    if 'mathematical equation' in text:
        return False
    return True

def text2sentences(text:str, 
                   sent_tokenizer: Callable = nltk.sent_tokenize):
    idxs_arr = []
    sents = sent_tokenizer(text)
    cur_str = text[:]
    cur_skip = 0
    idxs2sent = {}
    for sent in sents:
        match_idx = cur_str.find(sent)
        start_idx = match_idx + cur_skip
        idxs_arr.append(start_idx)
        finish_idx = match_idx + cur_skip + len(sent) - 1
        idxs2sent[(start_idx, finish_idx)] = sent
        if finish_idx + 1 < len(cur_str):
            cur_skip = finish_idx + 1
            cur_str = cur_str[match_idx + len(sent):]
    return idxs2sent, np.array(sents), idxs_arr

def extract_important_sections(text: str):
    parsed_text = wtp.parse(text)
    section_titles, section_texts = [], []
    for sec in parsed_text.sections:
        if not sec.title:
            #for par in sec.string.split('\n\n'):
            section_titles.append(sec.title)
            section_texts.append(clean_section_text(sec.string))
            continue
        if 'external links' in sec.title.lower():
            continue
        if 'references' in sec.title.lower():
            continue
        if 'notes' in sec.title.lower():
            continue
        if 'see also' in sec.title.lower():
            continue
        
        #for par in sec.string.split('\n\n'):
        section_titles.append(sec.title)
        section_texts.append(clean_section_text(sec.string))
    return section_titles, section_texts

def get_diff_num(prev_sections_texts: Tuple[str], 
                 new_sections_texts: Tuple[str]):
    dif_result = list(DifflibParser(prev_sections_texts, new_sections_texts))
    result = []
    result_idxs = []
    old_text, new_text, last_diff_id = [], [], -1000
    for dif_id, dif_line in enumerate(dif_result):
        if dif_line['code'] != DiffCode.SIMILAR:
            if np.abs(dif_id - last_diff_id) > 0:
                if filter_edited_abstract(dif_line['line']):
                    if 'newline' in dif_line and not filter_edited_abstract(dif_line['newline']):
                        continue
                    
                    if dif_line['code'] != DiffCode.CHANGED and dif_id > 0 and dif_line['code'] != DiffCode.LEFTONLY:
                        prev_text = dif_result[dif_id - 1]['line']
                        dif_line['new_line'] = prev_text + ' ' + dif_line['line']
                        dif_line['line'] = prev_text
                    result.append(dif_line)
                    result_idxs.append(dif_id)
                    last_diff_id = dif_id
    return result_idxs, result    

def get_changes(diffs: Tuple[dict]):
    all_changes = []
    all_changes_sents = []
    for diff_id, diff_obj in enumerate(diffs):
        if diff_obj['code'] == DiffCode.RIGHTONLY:
            if len(abstarct_tokenizer(diff_obj['line'])) > MAX_ABSTRACT_LEN:
                continue
            if len(abstarct_tokenizer(diff_obj['line'])) < MIN_ABSTRACT_LEN:
                continue
            all_changes.append(([diff_obj['line']], 'r'))
            _, sents, _ = text2sentences(diff_obj['line'])
            all_changes_sents.append(sents)
            
        elif diff_obj['code'] == DiffCode.LEFTONLY:
            if len(abstarct_tokenizer(diff_obj['line'])) > MAX_ABSTRACT_LEN:
                continue
            if len(abstarct_tokenizer(diff_obj['line'])) < MIN_ABSTRACT_LEN:
                continue
            all_changes.append(([diff_obj['line']], 'l'))
            _, sents, _ = text2sentences(diff_obj['line'])
            all_changes_sents.append(sents)
            
        elif diff_obj['code'] == DiffCode.CHANGED:
            if len(abstarct_tokenizer(diff_obj['line'])) > MAX_ABSTRACT_LEN:
                continue
            if len(abstarct_tokenizer(diff_obj['newline'])) > MAX_ABSTRACT_LEN:
                continue
            idxs2sent, sents, idxs_arr = text2sentences(diff_obj['newline'])
            all_changes_sents = []
            r_change = diff_obj['rightchanges']
            cur_ch = -10
            prev_ch = -10
            all_r_changes = []
            changed_sents = []
            for ch in r_change:
                if prev_ch < 0:
                    prev_ch = ch
                    cur_ch = ch
                if np.abs(ch - cur_ch) > 1:
                    new_change = diff_obj['newline'][prev_ch:cur_ch+1]
                    if new_change.strip() != '' and len(new_change.strip()) > MIN_CHANGE_SYMB_LEN:
                        all_r_changes.append(new_change)
                        sents_idxs_l, sents_idxs_r = find_nearest(prev_ch, cur_ch+1, idxs_arr)
                        changed_sents += list(range(sents_idxs_l, sents_idxs_r+1))
                    prev_ch = ch
                cur_ch = ch
            new_change = diff_obj['newline'][prev_ch:cur_ch+1]
            if new_change.strip() != '' and len(new_change.strip()) > MIN_CHANGE_SYMB_LEN:
                all_r_changes.append(new_change)
                sents_idxs_l, sents_idxs_r = find_nearest(prev_ch, cur_ch+1, idxs_arr)
                changed_sents += list(range(sents_idxs_l, sents_idxs_r+1))
            all_changes.append((all_r_changes, 'c'))
            changed_sents = sorted(list(set(changed_sents)))
            all_changes_sents.append(sents[changed_sents])
    return all_changes, all_changes_sents