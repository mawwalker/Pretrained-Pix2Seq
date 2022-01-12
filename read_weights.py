from re import M
import torch

# pretrained_weights = torch.load('./coco_ap370.pth')

pretrained_weights = torch.load('output/ships_v3/checkpoint_best.pth', map_location='cpu')
pretrained_weights = torch.load('output/ships_v2/eval/150.pth', map_location='cpu')

num_classes = 12 + 1
num_bins = 2000
num_vocal = num_bins + 1 + num_classes + 2
'''
input_proj.0.weight
input_proj.0.bias
input_proj.1.weight
input_proj.1.bias
transformer.vocal_classifier.weight: torch.Size([2094, 256])
transformer.vocal_classifier.bias: torch.Size([2094])
transformer.det_embed.weight: torch.Size([1, 256])
transformer.vocal_embed.weight: torch.Size([2092, 256])
'''

# model = pretrained_weights['model']
# print(model)

print(pretrained_weights.keys())
print(pretrained_weights['counts'])

# for x in model:
#     if isinstance(model[x], torch.Tensor):
#         print('{}: {}'.format(x, model[x].shape))

# pretrained_weights["model"]["class_embed.weight"].resize_(num_class+1,256)
# pretrained_weights["model"]["class_embed.bias"].resize_(num_class+1)

# resize the class shape
# pretrained_weights['model']['transformer.vocal_classifier.weight'].resize_(num_vocal, 256)
# pretrained_weights['model']['transformer.vocal_classifier.bias'].resize_(num_vocal)
# pretrained_weights['model']['transformer.vocal_embed.weight'].resize_(num_bins + num_classes + 1, 256)

# for x in pretrained_weights['model']:
#     if isinstance(model[x], torch.Tensor):
#         print('{}: {}'.format(x, model[x].shape))

# torch.save(pretrained_weights,'pix2seq_r50_%d.pth'%num_classes)

# print('input_proj.0.weight: {}, input_proj.1.weight: {}'.format(model["input_proj.0.weight"].shape,
#                                                                 model["input_proj.1.weight"].shape))
