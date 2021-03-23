import torch
from collections import OrderedDict

path = 'SiamMask_VOT_LD.pth'
model = = None

if(not torch.cuda.is_available()):
    model = torch.load(path,
        map_location=torch.device('cpu'))
else:
    model = torch.load(path,
        map_location=lambda storage, loc: storage.cuda(device))

new_model = OrderedDict()

for k, v in model.items():
    if k.startswith('features.features'):
        k = k.replace('features.features', 'backbone')
    elif k.startswith('features'):
        k = k.replace('features', 'neck')
    elif k.startswith('rpn_model'):
        k = k.replace('rpn_model', 'rpn_head')
    elif k.startswith('mask_model'):
        k = k.replace('mask_model.mask', 'mask_head')
    elif k.startswith('refine_model'):
        k = k.replace('refine_model', 'refine_head')
    new_model[k] = v

torch.save(new_model, 'model.pth')
