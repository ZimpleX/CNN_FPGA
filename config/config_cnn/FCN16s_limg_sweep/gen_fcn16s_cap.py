# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:39:12 2017

@author: zengh
"""

_NAME = "FCN16s_Limg{}.yaml"

_template = """
name: FCN-16s_Limg_{name}
conv_layers:
  - name: conv1_1   #1
    l_img: {conv1}      # c1
    l_kern: 3  
    f_in: 3     
    f_out: 64   
    stride: 1   
    pad: 1
  - name: conv1_2   # 2
    l_img: {conv1}      # c1
    l_kern: 3  
    f_in: 64
    f_out: 64   
    stride: 1   
    pad: 1
  - name: conv2_1   # 3
    l_img: {conv2}      # c2
    l_kern: 3
    f_in: 64
    f_out: 128   
    stride: 1  
    pad: 1
  - name: conv2_2   # 4
    l_img: {conv2}      # c2
    l_kern: 3 
    f_in: 128
    f_out: 128   
    stride: 1   
    pad: 1
  - name: conv3_1   # 5
    l_img: {conv3}      # c3
    l_kern: 3 
    f_in: 128
    f_out: 256   
    stride: 1   
    pad: 1
  - name: conv3_2   # 6
    l_img: {conv3}      # c3
    l_kern: 3
    f_in: 256
    f_out: 256   
    stride: 1   
    pad: 1
  - name: conv3_3   # 7
    l_img: {conv3}      # c3
    l_kern: 3
    f_in: 256
    f_out: 256   
    stride: 1   
    pad: 1
  - name: conv4_1   # 8
    l_img: {conv4}       # c4
    l_kern: 3
    f_in: 256
    f_out: 512   
    stride: 1   
    pad: 1
  - name: conv4_2   # 9
    l_img: {conv4}       # c4
    l_kern: 3
    f_in: 512
    f_out: 512   
    stride: 1   
    pad: 1
  - name: conv4_3   # 10
    l_img: {conv4}       # c4
    l_kern: 3
    f_in: 512
    f_out: 512   
    stride: 1   
    pad: 1
  - name: conv5_1   # 11
    l_img: {conv5}       # c5
    l_kern: 3
    f_in: 512
    f_out: 512   
    stride: 1   
    pad: 1
  - name: conv5_2   # 12
    l_img: {conv5}       # c5
    l_kern: 3
    f_in: 512
    f_out: 512  
    stride: 1   
    pad: 1
  - name: conv5_3   # 13
    l_img: {conv5}       # c5
    l_kern: 3
    f_in: 512
    f_out: 512   
    stride: 1   
    pad: 1
  - name: fc6       # 14
    l_img: {conv6}       # c6
    l_kern: 7
    f_in: 512
    f_out: 4096   
    stride: 1 
    pad: 0
  - name: fc7       # 15
    l_img: {conv7}       # c7
    l_kern: 1
    f_in: 4096
    f_out: 4096   
    stride: 1 
    pad: 0
  - name: score     # 16
    l_img: {conv7}       # c7
    l_kern: 1
    f_in: 4096
    f_out: 21   
    stride: 1   
    pad: 0
  - name: score_2   # 17
    l_img: {conv8}       # c8
    l_kern: 4
    f_in: 21
    f_out: 21   
    stride: 2
    pad: 0
  - name: score_pool4 # 18
    l_img: {conv9}       # c9
    l_kern: 1
    f_in: 512
    f_out: 21   
    stride: 1   
    pad: 0
"""

def generate_file(limg_size):
    """
    limg_size is the image size for the last conv layer.
    """
    with open(_NAME.format(limg_size),'w') as f:
        f.write(_template.format(name=limg_size,
                                 conv6=limg_size,
                                 conv5=2*limg_size,
                                 conv4=4*limg_size,
                                 conv3=8*limg_size,
                                 conv2=16*limg_size,
                                 conv1=32*limg_size,
                                 conv7=limg_size-6,
                                 conv8=2*(limg_size-6),
                                 conv9=2*limg_size))
        
def sweep(limg_size_start,limg_size_stride,num_limg_size):
    for i in range(num_limg_size):
        generate_file(limg_size_start+limg_size_stride*i)
        

if __name__ == "__main__":
    sweep(10,1,10)
