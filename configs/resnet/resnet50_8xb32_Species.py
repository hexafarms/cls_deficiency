_base_ = [
    '../_base_/models/resnet50_Species.py', '../_base_/datasets/species_bs8_pil_resize_autoaug.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
