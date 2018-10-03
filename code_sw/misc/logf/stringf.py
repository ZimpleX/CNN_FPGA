"""
formatting input string by adding tags: [INFO] / [WARN] / [ERROR] / ...
To be used by logf.printf & logf.filef
"""

from functools import reduce
import pdb

_STRINGF_SEPA = {'INFO': '-',
                'WARN': '=',
                'ERROR': '*',
                'DEBUG': '~',
                'NET': '><',    # specific to ANN training
                'others': ''}

def stringf(string, *reflex, type='INFO', separator='default'):
    """
    You can provide arbitrary type and separator (can be multiple char):
    e.g.: string='{} is handy!', reflex='stringf', type='FOO', separator='><',
        then the return string will be:
        ><><><><><><><><><><><>
        [FOO] stringf is handy!
        ><><><><><><><><><><><>
    
    If you don't want to print type / separator, then set it / them to '' or None
    If you don't provide the separator, then it will be assigned according to _STRINGF_SEPA.
    """
    if separator == 'default':
        separator = _STRINGF_SEPA['others']
        separator = (type in _STRINGF_SEPA.keys()) and _STRINGF_SEPA[type] or separator
    if reflex:
        string = string.format(*reflex)
    if type is None or len(type) == 0:
        string = string
    else:
        string = '[{}] {}'.format(type, string)

    if not separator:
        return string
    else:
        maxLen = reduce(lambda l1,strL2: (l1 < len(strL2)) and len(strL2) or l1, string.split("\n"), 0)
        sepLine = separator*(maxLen//len(separator)) \
                + separator[0:maxLen%len(separator)]
        return '{}\n{}\n{}'.format(sepLine, string, sepLine)
