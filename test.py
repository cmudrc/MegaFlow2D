from megaflow.dataset.MegaFlow2D import MegaFlow2D

if __name__ == '__main__':
    # Create a dataset object
    dataset = MegaFlow2D(root='D:/Work/data', download=True, transform='normalize', pre_transform=None)