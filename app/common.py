class CNN:
    def __init__(self,params):
        self.name   = params['name']
        self.f_in   = params['f_in']
        self.f_out  = params['f_out']
        self.l_img  = params['l_img']
        self.l_kern = params['l_kern']
        self.stride = params['stride']
        self.pad    = params['pad']

    def get_layer_i(self,li):
        """
        li:     layer i --> can be a list.
                e.g., if li=[1,2,3], then it will return layers 1,2,3
        """
        return {'name':     self.name,
                'f_in':     self.f_in[li],
                'f_out':    self.f_out[li],
                'l_img':    self.l_img[li],
                'l_kern':   self.l_kern[li],
                'stride':   self.stride[li],
                'pad':      self.pad[li]}

    def get_cnn(self):
        return {'name':     self.name,
                'f_in':     self.f_in,
                'f_out':    self.f_out,
                'l_img':    self.l_img,
                'l_kern':   self.l_kern,
                'stride':   self.stride,
                'pad':      self.pad}
