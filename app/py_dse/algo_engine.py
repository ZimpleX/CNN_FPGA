import numpy as np
from zython.logf.printf import stringf
import zython.arithmetic as arizhmetic

from abc import ABCMeta,abstractmethod


class conv_complexity:
    __metaclass__ = ABCMeta
    def __init__(self,layer,params_algo):
        """
        computes the computation complexity for multiple layers

        INPUT:
            layer              {'name': ..,
                                'f_in': ..,
                                'f_out': ..,
                                'l_img': ..,
                                'l_kern': ..,
                                'stride': ..,
                                'pad': ..}
            params_algo        {'d max': .., 
                                'name': .., 
                                'N avail': ..}
        OUTPUT:
            NONE
        """
        self.cnn_name    = layer['name']
        self.params_algo = params_algo
        self.f_in   = layer['f_in']
        self.f_out  = layer['f_out']
        self.l_img  = layer['l_img']
        self.l_kern = layer['l_kern']
        self.stride = layer['stride']
        self.pad    = layer['pad']
        # -------------
        self.num_layers = len(self.f_in)
        self.ops_count = np.zeros(self.num_layers, dtype=np.int64)

    @abstractmethod
    def count(self):
        pass

    def str_compare_algo(self,*other_algo):
        """
        compare the baseline algo with other algorithms.
        For example, let self be an object of spatial_complexity, and oaa, cap be objects of fft_complexity
        You can do self.str_compare_algo(oaa,cap) to check how much reduction of computation complexity can
        be achieved by frequency domain convolution. The comparison gives you both layer by layer info as 
        well as the statistics for the complete CNN. 

        INPUT:
            other_algo          OBJECTs of the type 'conv_complexity' ('spatial_complexity'/'fft_complexity')
        OUTPUT:
            s                   STRING showing the comparison results. 
        """
        num_algo = len(other_algo)+1
        name_list = [self.params_algo['name']] + [cnn.params_algo['name'] for cnn in other_algo]
        ops_count_algo = [cnn.ops_count for cnn in other_algo]
        ops_count_algo = np.concatenate(ops_count_algo)
        ops_count_algo = np.concatenate((self.ops_count,ops_count_algo)).reshape((-1,self.num_layers))
        ops_ratio_algo = (ops_count_algo/self.ops_count).T
        s = stringf('{} -- COMPARE DIFFERENT CONV ALGORITHMS',
                self.cnn_name,type=None,separator='=')
        s += '\n'
        s += stringf('layer    l_img   l_kern'+'    {:>10s}'*num_algo,
                *name_list,type=None,separator='-')
        s += '\n'
        for li in range(self.num_layers):
            s += stringf('{:>5d}    {:>5d}    {:>5d}'+'    {:>10.3f}'*num_algo+'\n',
                    li,self.l_img[li],self.l_kern[li],*ops_ratio_algo[li],type=None)
        s += '-'*(23+14*num_algo) + '\n'
        ops_total_algo = ops_count_algo.sum(axis=1)
        ops_total_algo_norm = ops_total_algo/ops_total_algo[0]
        s += stringf('{:>23s}'+'    {:>10.3f}'*num_algo+'\n',
                'TOTAL OPS:',*ops_total_algo_norm,type=None)
        ops_total_algo_G = ops_total_algo/1e9
        s += stringf('{:>23s}'+'    {:>10.3f}'*num_algo+'\n',
                'TOTAL OPS (G):',*ops_total_algo_G,type=None)
        return s
        



class spatial_complexity(conv_complexity):
    def __init__(self,layer,params_algo):
        super().__init__(layer,params_algo)

    def count(self):
        """
        Get the total number of operations using spatial convolution. 
        We count both addition and multiplication. Note that in many CNNs, the padding 
        is equal to (l_kern-1)/2 rather than (l_kern-1). 

        INPUT:
            NONE
        OUTPUT:
            NONE
        """
        num_pixel_out = ((self.l_img - self.l_kern\
            + 2*self.pad) / self.stride + 1)**2
        ops_per_pixel = 2*self.l_kern**2
        self.ops_count = self.f_in*self.f_out\
                *num_pixel_out*ops_per_pixel
        return self.ops_count

    def __str__(self):
        s = stringf("{}: CONVOLUTION IN SPATIAL DOMAIN",
                self.cnn_name,type=None,separator='=')
        s += '\n'
        s += stringf('layer    l_img   l_kern  ops (M)',type=None,separator='-')
        s += '\n'
        for li in range(self.num_layers):
            s += stringf('{:5d}    {:5d}    {:5d}    {:5d}\n',
                li,self.l_img[li],self.l_kern[li],
                int(self.ops_count[li]/1e6),type=None)
        s += '-'*32 + '\n'
        s += stringf('TOTAL OPS: {:5.3f} G\n', self.ops_count.sum()/1e9,type=None)
        return s



class fft_complexity(conv_complexity):
    def __init__(self,layer,params_algo,
            OPS_CONST={'fft':1.5,'hadamard':4.0,'ifft':1.5}):  # why 4? cuz complex multiplication requires 6, and accumulation requires 2. Real and Imag are both useful pixels, so (6+2)/2
        """
        INPUT:
            OPS_CONST           constants for computation complexity. 
                                hadamard product involves a complexity number multiplication and addition for each pixel.
                                complex mul needs 4 mul and 2 add, complex add need 2 add --> 8 ops in total each pixel.
                                shared between the real and imag pixels, then each image pixel gets 8/2 = 4 ops. 
        OUTPUT:
            NONE
        """
        super().__init__(layer,params_algo)
        # chosen_params_algo: the folding factor and fft size selected by the algo engine. 
        self.chosen_params_algo = {'batch fold':np.zeros(self.num_layers, dtype=np.int64), 
                            'fft Ni':np.zeros(self.num_layers, dtype=np.int64)}
        self.CONST = OPS_CONST

    def count(self,layer_list=None,global_d_N=False):
        """
        Get the optimal computation complexity by figuring out the optimal configuration of 
        batch folding factor and fft size using the CaP technique. 

        INPUT:
            layer_list          LIST of layer index to be considered for complexity calculation. 
                                None if you want to consider all layers.
            global_d_N          True, if we choose a single value of d and N for all layers of a CNN
                                False, if we choose value of d and N for each layer separately.
        OUTPUT:
            self.ops_count      Total number of operations using CaP
        """
        if self.params_algo['d max'] == -1:
            # the gcd method may not give you the optimal bound, but will give you "good enough" configuration
            d_max = (self.params_algo['N avail'].max()-self.l_kern+1)/\
                arizhmetic.gcd(self.l_img+self.l_kern-1,self.params_algo['N avail'].max()-self.l_kern+1)
        else:
            d_max = np.array([self.params_algo['d max']]*self.num_layers)
        if layer_list is None:
            layer_list = range(self.num_layers)
        if global_d_N:
            best_ops_total = float('inf')
            for di in range(1,int(d_max.max())+1):
                for Nk in self.params_algo['N avail']:
                    if Nk <= self.l_kern.max()-1:
                        continue
                    ops_nn = [self._count_i(di,Nk,li) for li in layer_list]
                    ops_total = np.array(ops_nn).sum()
                    if ops_total < best_ops_total:
                        best_ops_total = ops_total
                        self.ops_count = np.array(ops_nn)
                        for li in layer_list:
                            self.set_chosen_params_algo(di,Nk,li)
        else:
            for li in layer_list:
                best_ops_li = float('inf')
                for dij in range(1,int(d_max[li])+1):
                    for Nk in self.params_algo['N avail']:
                        if Nk <= self.l_kern[li]-1:   
                            continue
                        ops_li = self._count_i(dij,Nk,li)
                        if ops_li < best_ops_li:
                            best_ops_li = ops_li
                            self.ops_count[li] = best_ops_li
                            self.set_chosen_params_algo(dij,Nk,li)
        return self.ops_count


    def _count_i(self,di,Ni,li):
        """
        count the total number of operations for a single CNN layer, given the CaP configuration
        INPUT:
            d                   INTEGER, batch folding factor
            N                   INTEGER, fft size
        OUTPUT:
            li_count            ops count for layer i
        """
        if Ni-self.l_kern[li]+1 <= 0:
            return float('inf')
        # ops per tile
        tile_fft = 2*self.CONST['fft']*Ni**2*np.log(Ni)/np.log(2)*self.f_in[li]
        tile_ifft = 2*self.CONST['ifft']*Ni**2*np.log(Ni)/np.log(2)*self.f_out[li]
        tile_hadamard = self.CONST['hadamard']*Ni**2*self.f_in[li]*self.f_out[li]
        # num tiles
        l_img_fd = di*self.l_img[li] + (di-1)*(self.l_kern[li]-1)
        num_tiles = np.ceil(l_img_fd/(Ni-self.l_kern[li]+1))**2
        li_count = num_tiles*(tile_fft+tile_ifft+tile_hadamard)/di**2
        return li_count

    def set_chosen_params_algo(self,di,Ni,li):
        """
        li:     layer i
        """
        self.chosen_params_algo['batch fold'][li] = di
        self.chosen_params_algo['fft Ni'][li] = Ni

    def __str__(self):
        s = stringf("{}: CONVOLUTION IN FREQUENCY DOMAIN - {}",
                self.cnn_name,self.params_algo['name'],type=None,separator="=")
        s += '\n'
        s += stringf('layer    l_img   l_kern        N        d  ops (M)',type=None,separator='-')
        s += '\n'
        for li in range(self.num_layers):
            s += stringf('{:5d}    {:5d}    {:5d}    {:5d}    {:5d}    {:5d}\n',
                li,self.l_img[li],self.l_kern[li],
                self.chosen_params_algo['fft Ni'][li],self.chosen_params_algo['batch fold'][li],
                int(self.ops_count[li]/1e6),type=None)
        s += '-'*50 + '\n'
        s += stringf('TOTAL OPS: {:5.3f} G\n', self.ops_count.sum()/1e9,type=None)
        return s


class fft_complexity_ideal(fft_complexity):
    def __init__(self,layer,params_algo,
            OPS_CONST={'fft':1.5,'hadamard':4.0,'ifft':1.5}):
        super().__init__(layer,params_algo,OPS_CONST=OPS_CONST)

    def count_ideal(self):
        count_i = self.CONST['hadamard']*(self.l_img+self.l_kern-1)**2*self.f_in*self.f_out
        return count_i.sum()