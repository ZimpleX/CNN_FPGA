"""
print formatted string to console / terminal
"""

from zython.logf.stringf import stringf

def printf(string, *reflex, type='INFO', separator='default'):
    """
    for usage, check out logf.stringf
    """
    print(stringf(string, *reflex, type=type, separator=separator))


def think_twice(about_what):
    printf('Are u sure to {}?', about_what, type='WARN', separator=None)
    ans = input('[Y/n]: ')
    return ans == 'Y'
        
