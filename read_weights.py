import torch
import argparse

parser = argparse.ArgumentParser('Resize vocal', add_help=False)
parser.add_argument('--num', default=90, type=int, help="class num")
parser.add_argument('--weight', type=str, default="./coco_ap370.pth",
                    help="source weight path")
parser.add_argument('--type', type=str, default="res",
                    help="pretrained model type, res or Swin_L, Swin_B, ")
args = parser.parse_args()

pretrained_weights = torch.load(args.weight)

num_classes = args.num + 1
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
# for x in pretrained['model']:
    # print(x)
print('before resize: ')
model = pretrained_weights['model']
for x in model:
    if isinstance(model[x], torch.Tensor):
        print('{}: {}'.format(x, model[x].shape))

# pretrained_weights["model"]["class_embed.weight"].resize_(num_class+1,256)
# pretrained_weights["model"]["class_embed.bias"].resize_(num_class+1)
pretrained_weights['model']['transformer.vocal_classifier.weight'].resize_(num_vocal, 256)
pretrained_weights['model']['transformer.vocal_classifier.bias'].resize_(num_vocal)
pretrained_weights['model']['transformer.vocal_embed.weight'].resize_(num_bins + num_classes + 1, 256)

print('after resize: ')
for x in pretrained_weights['model']:
    if isinstance(model[x], torch.Tensor):
        print('{}: {}'.format(x, model[x].shape))
if args.type == 'res':
    torch.save(pretrained_weights,'pix2seq_r50_%d.pth'%num_classes)
else:
    torch.save(pretrained_weights,f'pix2seq_{type}_{num_classes}.pth')
# print('input_proj.0.weight: {}, input_proj.1.weight: {}'.format(model["input_proj.0.weight"].shape,
#                                                                 model["input_proj.1.weight"].shape))
