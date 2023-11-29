import torch
from torchvision.datasets import Cityscapes
from torchvision import transforms
import unittest

class CustomCityscapes(Cityscapes):
    """
        The PyTorch implementation of the torchvision.datasets.Cityscapes
        seems to have a bug. It does not apply the transform attribute to
        the masks. Thus, the __getitem__ method is overridden here.

    """
    def __getitem__(self, idx):
        images, masks = super().__getitem__(idx)
        return images, self.transform(masks.convert('RGB'))


"""
    Unit Test class

"""

class TestDataset(unittest.TestCase):
    def test_dataset_transform(self):
        """
            Tests that the transforms as applied to both images and masks
            in the dataset.

        """
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor()
            ])

        dataset = CustomCityscapes(root="/home/zak/datasets/cityscapes",
                    split='val', mode='fine', target_type='color',
                    transform=transform)
        
        image, mask = dataset[1]
        self.assertEqual(type(image), torch.Tensor)
        self.assertEqual(type(mask), torch.Tensor)
        self.assertEqual(image.shape, mask.shape)


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
