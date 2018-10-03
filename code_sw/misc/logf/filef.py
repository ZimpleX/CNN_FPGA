"""
utility script for file operations:
    * create dir
    * file permission
    * log to file
"""

from __future__ import print_function
import os
from zython.logf.stringf import stringf
from zython.logf.printf import printf
import stat
import re

import pdb

_LOG_DIR_DEFAULT = './output.log.ignore'

def grep_string(s,pattern):
    """
    ref: [http://stackoverflow.com/questions/1921894/grep-and-python#answer-25181706]
    s should be a string.
    """
    return '\n'.join(re.findall(r'^.*%s.*?$'%pattern,s,flags=re.M))
def grep_file(f,pattern):
    """
    f should be a file name
    """
    with open(f) as in_f:
        s = '\n'.join([l for l in in_f])
        ret = grep_string(s,pattern)
    return ret
def grep_file_multi_keys(f,pattern_l):
    """
    grep multiple keys from the file named as f.
    """
    ret = dict()
    with open(f) as in_f:
        s = '\n'.join([l for l in in_f])
        for pattern in pattern_l:
            ret[pattern] = grep_string(s,pattern)
    return ret
    


def mkdir_r(dir_r):
    """
    recursively mkdir if not exist
    dir_r of 'a/b/c' or 'a/b/c/' will both create directory a, b and c

    WARNING:
    no explicit error checking:
    e.g.: if there is a file (not dir) called 'a/b', then this function will fail
    """
    dir_parent = os.path.dirname(dir_r)
    dir_parent = (dir_parent != '') and dir_parent or '.'
    if not os.path.exists(dir_parent):
        mkdir_r(dir_parent)
    if not os.path.exists(dir_r):
        os.mkdir(dir_r)
        printf("created dir: {}", dir_r, separator=None)


def _perm_to_int(perm):
    """
    convert perm (which can be a str or int) to int (understandable by os module)
    e.g.: perm='0444' if for read-only policy
    However, I won't process the first char for now
    """
    if type(perm) == type(0):
        return perm
    ERROR_PERM_FORMAT = 'format of perm is wrong!'
    try:
        assert len(perm) == 4
    except AssertionError:
        printf(ERROR_PERM_FORMAT, type='ERROR')
        exit()
    p_pos = ['','USR', 'GRP', 'OTH']   # don't care, owner, group, others
    p_ret = 0
    eval_str = 'stat.S_I{}{}'
    for n in range(1,4):
        p_int = int(perm[n])
        try:
            assert p_int <= 7 and p_int >= 0
        except AssertionError:
            printf(ERROR_PERM_FORMAT, type='ERROR')
            exit()
        if p_int >= 4:
            p_ret |= eval(eval_str.format('R', p_pos[n]))
        if p_int in [2,3,6,7]:
            p_ret |= eval(eval_str.format('W', p_pos[n]))
        if p_int%2 == 1:
            p_ret |= eval(eval_str.format('X', p_pos[n]))
    return p_ret


def set_f_perm(f_name, perm):
    """
    set permission of file (f_name) to 'perm'
    perm: int / string containing 4 digits (e.g.: '0777')

    NOTE:
    I am not checking existence here.
    """
    perm = _perm_to_int(perm)
    os.chmod(f_name, perm)


def print_to_file(f_name, msg, *reflex, type='INFO', separator='default', 
        log_dir=_LOG_DIR_DEFAULT, mode='a', perm='default'):
    """
    write message into file
        * create dir / file if not exist
        * read-only policy
    argument:
        f_name      name of the log file
        msg,
        reflex, 
        type, 
        separator   see logf.stringf
        log_dir     the directory to be appended in front of f_name
        mode        'w' for truncate; 'a' for append
        perm        permission of f_name after write to it
            Policy of perm='default':
              if f_name exists before, then don't change permission
              if f_name doesn't exist, then set permission to '0444'
            Policy of perm='xxxx' (string consisting of 4 digits):
              set permission to 'xxxx' no matter f_name exists or not
    """
    msg = str(msg)
    f_full = '{}/{}'.format(log_dir, f_name)
    if perm == 'default':
        perm = (os.path.exists(f_full)) and os.stat(f_full).st_mode or '0444'
    # make sure directory exists
    mkdir_r(os.path.dirname(f_full))
    if os.path.exists(f_full):
        set_f_perm(f_full, '0222')
    f = open(f_full, mode)
    print(stringf(msg, *reflex, type=type, separator=separator), file=f)
    #printf('write to file: {}', f_full, separator=None)
    f.close()
    
    set_f_perm(f_full, perm)
