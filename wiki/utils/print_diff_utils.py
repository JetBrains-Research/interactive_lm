import difflib
from IPython.display import HTML as html_print

def colored(s, color='black'):
    if color == 'green':
        return f'\033[93m {s}'
    if color == 'yellow':
        return f'\033[92m {s}'
    if color == 'red':
        return f'\033[91m {s}'
    if color == 'blue':
        return f'\033[94m {s}'
    if color == 'pink':
        return f'\033[95m {s}'
    if color == 'bold':
        return f'\x1B[1m {s} \x1b[0m'
    return f'\033[39m {s}'

def diff_print(src_text, tgt_text):
    src_text_tok, tgt_text_tok = src_text.split(), tgt_text.split()
    matcher = difflib.SequenceMatcher(a=src_text_tok, b=tgt_text_tok)
    
    sti = []
    cur_idx = 0
    for match in matcher.get_matching_blocks():
        if match.size != 0:
            if match.b == cur_idx:
                sti.append((match.b, match.b + match.size, 'same'))
                cur_idx = match.b + match.size
            else:
                sti.append((cur_idx, match.b, 'diff'))
                sti.append((match.b, match.b + match.size, 'same'))
                cur_idx = match.b + match.size
    if cur_idx < len(tgt_text_tok) - 1:
        sti.append((cur_idx, len(tgt_text_tok), 'diff'))
    
    new_str = ''
    for (st_id, en_id, idxs_type) in sti:
        cur_txt = ' '.join(tgt_text_tok[st_id:en_id])
        if idxs_type != 'same':
            new_str += colored(cur_txt, color='bold') + ' '
        else:
            new_str += cur_txt + ' '
    return new_str