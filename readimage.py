from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

FILE = 'map_space.npy'

def load(FILE):
    grid = np.load(FILE)
    plt.imshow(grid)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img = Image.open('map_sample.png')
    img = ImageOps.grayscale(img)
    np_img = np.array(img)
    np_img = ~np_img
    np_img[np_img > 0] = 1
    plt.set_cmap('binary')
    np.save('map_space.npy', np_img)
    load(FILE)




