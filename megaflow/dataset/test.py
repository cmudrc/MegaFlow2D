from MegaFlow2D import MegaFlow2D

if __name__ == '__main__':
    dataset = MegaFlow2D(root='C:/research/data', download=False, split_scheme='mixed', transform=None, pre_transform=None, split_ratio=[0.5, 0.5])