import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


def display_image(img):
    image = Image.open(img)
    t = transforms.ToTensor()(image)
    composed = transforms.Compose([transforms.ToPILImage(),transforms.Resize((571,480),interpolation=Image.NEAREST)])
    trans_img = composed(t)
    #slicedImage = trans_img[63:318, 70:424]
    plt.imshow(image)


    plt.show()

#file = "scaned_bars.jpg"
file = "bars.png"
display_image(file)
