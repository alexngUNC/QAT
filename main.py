import torchvision
import torch
from quant_conv import QuantConv, QuantReLU
from functools import reduce

def set_module_by_name(module,
                       access_string, val):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep='.')
    p = reduce(getattr, names[:-1], module)
    setattr(p, names[-1], val)

conv_q_list = [("1.0", 1), ("1.2", 2), ("1.2", 3), ("2.0", 1), ("2.3", 2), ("2.3", 3), ("3.0", 1), ("3.5", 2), ("3.5", 3), ("4.0", 1), ("4.2", 2), ("4.2", 3)]
q_relu = ["1.0", "1.1", "2.0", "2.1", "2.2", "3.0", "3.1", "3.2", "3.3", "3.4", "4.0", "4.1"]

def quantize_model(model):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            print(name)
            q = False
            for a, b in conv_q_list:
                if f"{a}.conv{b}" in name:
                    q=True
            print(q)
            if q or "downsample" in name or name == "conv1":
                new_conv = QuantConv(mod.in_channels, mod.out_channels, mod.kernel_size, stride=mod.stride, padding=mod.padding, dilation=mod.dilation, groups=mod.groups, bias=True if mod.bias is not None else False, padding_mode=mod.padding_mode)
            else:
                new_conv = QuantConv(mod.in_channels, mod.out_channels, mod.kernel_size, stride=mod.stride, padding=mod.padding, dilation=mod.dilation, groups=mod.groups, bias=True if mod.bias is not None else False, padding_mode=mod.padding_mode, input_q=False)
            new_conv.weight = mod.weight
            set_module_by_name(model, name, new_conv)
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ReLU):
            q = False
            print(name)
            for n in q_relu:
                if n in name:
                    q = True
            if q:
                new_relu = QuantReLU(mod.inplace)
                set_module_by_name(model, name, new_relu)
    return model
    

def main():
    # torch.nn.Conv2d()
    model = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    # x = torch.zeros([1, 3, 224, 224], dtype=torch.float32)
    # model.eval()
    file_name = "resnet50-original.onnx"
    x = torch.zeros(1,3,224,224)
    torch.onnx.export(model, x, file_name, opset_version=17)
    return
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            print(name)
            q = False
            for a, b in conv_q_list:
                if f"{a}.conv{b}" in name:
                    q=True
            print(q)
            if q or "downsample" in name or name == "conv1":
                new_conv = QuantConv(mod.in_channels, mod.out_channels, mod.kernel_size, stride=mod.stride, padding=mod.padding, dilation=mod.dilation, groups=mod.groups, bias=True if mod.bias is not None else False, padding_mode=mod.padding_mode)
            else:
                new_conv = QuantConv(mod.in_channels, mod.out_channels, mod.kernel_size, stride=mod.stride, padding=mod.padding, dilation=mod.dilation, groups=mod.groups, bias=True if mod.bias is not None else False, padding_mode=mod.padding_mode, input_q=False)
            new_conv.weight = mod.weight
            set_module_by_name(model, name, new_conv)
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ReLU):
            q = False
            print(name)
            for n in q_relu:
                if n in name:
                    q = True
            if q:
                new_relu = QuantReLU(mod.inplace)
                set_module_by_name(model, name, new_relu)
    
    model(x)
    model.apply(torch.ao.quantization.disable_observer)
    # model.apply(torch.ao.quantization.disable_fake_quant)
    model.eval()
    # x = torch.zeros([1, 3, 224, 224], dtype=torch.float32)
    # model.eval()
    file_name = "resnet50-quant.onnx"
    torch.onnx.export(model, x, file_name, opset_version=17)

if __name__ == "__main__":
    main()