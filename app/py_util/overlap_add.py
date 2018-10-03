import numpy as np
import scipy.signal

import pdb
from zython.logf.printf import printf   # my own lib for logging info
from math import ceil


# np.fft.fft2(a)
np.set_printoptions(precision=2,linewidth=200)
np.random.seed(0)

def _mat_shape_check(*mat_l):
    """
    check if mat is 2D and its shape is square (N x N)
    """
    try:
        for mat in mat_l:
            assert len(mat.shape) == 2
            assert mat.shape[0] == mat.shape[1]
    except AssertionError:
        printf("Currently only support 2D square matrix, get {}", mat.shape,type="ERROR")


def mat_padding(orig_k, ret_size, 
            x_shift=0, y_shift=0):
    """
    zero padding the small kernel 'orig_k' to size ret_size x ret_size.
    overlap the pixel (0,0) of orig_k with (0,0) of ret_k, then do
    x direction shift & y direction shift of ret_k.
    do cyclic shifting if kernel across the boundary.
    """
    ret_k = np.zeros((ret_size, ret_size))
    temp_k = np.zeros((ret_size, ret_size))
    #_mat_shape_check(orig_k)
    temp_k[0:orig_k.shape[0],0:orig_k.shape[1]] = orig_k
    ret_k[y_shift:ret_size,:] = temp_k[0:ret_size-y_shift,:]
    ret_k[0:y_shift,:] = temp_k[ret_size-y_shift:ret_size,:]
    temp_k[:] = ret_k
    ret_k[:,x_shift:ret_size] = temp_k[:,0:ret_size-x_shift]
    ret_k[:,0:x_shift] = temp_k[:,ret_size-x_shift:ret_size]
    #_ret_fft = np.fft.fft2(ret_k)
    #ret_fft_real = _ret_fft.real
    #ret_fft_imag = _ret_fft.imag
    #printf('ret_k:\n{}',ret_k)
    #printf('fft_real:\n{}\nfft_imag:\n{}',ret_fft_real,ret_fft_imag)
    return ret_k


def conv_compare(ip_name):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    #kernel = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
    from PIL import Image
    ip_img = np.asarray(Image.open(ip_name))
    # clip image to be square
    img_size = min(ip_img.shape[0],ip_img.shape[1])
    ip_img = ip_img[0:img_size,0:img_size]
    from scipy import signal
    op_img = signal.convolve2d(ip_img,kernel)
    # output image
    op_name = '{}_{}.{}'.format('.'.join(ip_name.split('.')[0:-1]),'opNative',ip_name.split('.')[-1])
    Image.fromarray(np.uint8(op_img.clip(0,255))).save(op_name)


def overlap_add(kern, base, N):
    """
    Do the N point FFT using overlap and add method.
    Suppose: 
        kern        K x K
        base        B x B
        base_window W x W
        ret_mat     R x R
    Do the N-point KKT:
      - W + K - 1 = N
      - Zero padding kern & base to N x N
      - KKT padded N x N matrices.

    Note:
      - Currently support only square shaped kernel & base.
    """
    # actually OaA has already done the padding for you.
    #pdb.set_trace()

    _mat_shape_check(kern, base)
    K = kern.shape[0]
    B = base.shape[0]
    stride = 1
    padding = K - 1
    R = int((B - K + 2*padding)/stride + 1)
    assert K <= N

    W = N + 1 - K           # l_sub
    overlap = K - 1
    R_prime = ceil(B/W)*W + K - 1     # temp matrix after padding
    ret_mat = np.zeros((R_prime, R_prime),dtype=complex)
    kern_fft = np.fft.fft2(mat_padding(kern,N))
    for i in range(0,B,W):
        for j in range(0,B,W):
            base_win_fft = np.fft.fft2(mat_padding(base[i:i+W,j:j+W],N))
            ret_tile = np.fft.ifft2(kern_fft*base_win_fft)
            x_off = i
            y_off = j
            temp_mat = np.zeros((R_prime,R_prime),dtype=complex)
            #printf("(x_off,y_off): ({},{})  in  ({},{})", x_off, y_off, R_prime,R_prime)
            try:
                temp_mat[x_off:x_off+N,y_off:y_off+N] = ret_tile
            except Exception:
                pdb.set_trace()
                printf("exception", type="ERROR")
            ret_mat += temp_mat
    # debug
    assert not np.any(np.around(ret_mat.imag,decimals=10))
    #printf("fft conv:\n{}", np.around(ret_mat[0:R,0:R].real,decimals=10))
    #printf("normal conv:{}\n{}", scipy.signal.convolve2d(base,kern).shape, scipy.signal.convolve2d(base,kern))
    return ret_mat[0:R,0:R].real


def overlap_add_1D(kern,base,N):
    # Default padding & stride:
    #   padding = N-1
    #   stride = 1
    K = kern.shape[0]
    B = base.shape[0]
    R = K + B - 1
    W = N + 1 - K
    kern_pad = np.zeros(N)
    kern_pad[0:K] = kern
    kern_fft = np.fft.fft(kern_pad,N)
    ret = np.zeros(R)
    for i in range(0,B,W):
        base_pad = np.zeros(N)
        base_pad[0:W] = base[i:i+W]
        base_win_fft = np.fft.fft(base_pad,N)
        ret_win = np.fft.ifft(kern_fft*base_win_fft)
        temp = np.zeros(R)
        temp[i:i+N] = ret_win
        ret += temp
    # debug
    printf("normal conv:\n{}", np.convolve(kern,base))
    printf("fft conv:\n{}", ret)



if __name__ == "__main__":
    N = 8
    K = 4
    B = 9
    kern = np.random.randint(10,size=K*K).reshape(K,K)
    base = np.random.randint(10,size=B*B).reshape(B,B)
    stride = 1
    overlap_add(kern,base,N)
    """
    N = 4
    #kern = np.random.randint(10,size=3)
    #base = np.random.randint(10,size=12)
    kern = np.array([1])
    base = np.arange(8)
    overlap_add_1D(kern,base,N)
    """
