import torch
import torch.ao.quantization as quant

class QuantConv(torch.nn.Conv2d):
    def __init__(self,
                *args,
                input_q = True,
                weight_q = True,
                output_q = False,
                **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_quantizer = None
        self.weight_quantizer = None
        self.output_quantizer = None
        if input_q:
            self.input_quantizer = quant.FakeQuantize(quant.MinMaxObserver,dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric)
        if weight_q:
            self.weight_quantizer = quant.FakeQuantize(quant.MovingAveragePerChannelMinMaxObserver,dtype=torch.qint8, ch_axis=1, qscheme=torch.per_channel_symmetric, quant_min=-128, quant_max=127)
        if output_q:
            self.output_quantizer = quant.FakeQuantize(quant.MinMaxObserver,dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric)
    def forward(self, x):
        if self.input_quantizer:
            x = self.input_quantizer(x)
        if self.weight_quantizer:
            x = self._conv_forward(x, self.weight_quantizer(self.weight), self.bias)
        else: 
            x = self._conv_forward(x, self.weight, self.bias)
        if self.output_quantizer:
            return self.output_quantizer(x)
        return x
    
class QuantReLU(torch.nn.ReLU):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)
        self.output_quantizer = quant.FakeQuantize(quant.MinMaxObserver,dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric)
    def forward(self, x):
        return self.output_quantizer(super().forward(x))
