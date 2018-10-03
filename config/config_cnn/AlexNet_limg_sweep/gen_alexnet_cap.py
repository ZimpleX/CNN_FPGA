# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 12:38:30 2017

@author: zengh
"""

_NAME = 'AlexNet_Limg{}.yaml'

_template = """
name: AlexNet_Limg_{name}
conv_layers:
  #- name: conv1 # conv layer i
  #  l_img: {c1} # image width/height
  #  l_kern: 11  # kernel width/height
  #  f_in: 3     # input feature maps
  #  f_out: 96   # output feature maps
  #  stride: 4   # conv stride
  #  pad: 0
  - name: conv2
    l_img: {c2}
    l_kern: 5
    f_in: 96
    f_out: 256
    stride: 1
    pad: 2
  - name: conv3
    l_img: {c3}
    l_kern: 3
    f_in: 256
    f_out: 384
    stride: 1
    pad: 1
  - name: conv4
    l_img: {c4}
    l_kern: 3
    f_in: 384
    f_out: 384
    stride: 1
    pad: 1
  - name: conv5
    l_img: {c5}
    l_kern: 3
    f_in: 384
    f_out: 256
    stride: 1
    pad: 1
"""


def generate_file(limg_size):
    """
    limg_size is the image size for the last conv layer.
    """
    with open(_NAME.format(limg_size),'w') as f:
        f.write(_template.format(name=limg_size,
                                 c5=limg_size,
                                 c4=limg_size,
                                 c3=limg_size,
                                 c2=2*limg_size,
                                 c1=16*limg_size))
        
def sweep(limg_size_start,limg_size_stride,num_limg_size):
    for i in range(num_limg_size):
        generate_file(limg_size_start+limg_size_stride*i)
        

if __name__ == "__main__":
    sweep(10,1,10)