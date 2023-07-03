#A no-compression test on how fast the SVD algorithm runs.

from PIL import Image
import argparse
import numpy as np

from versions.svd_0 import svd
#from versions.svd_1 import svd

def image_compression(path: str):
    image = Image.open(path)
    img_arr = np.array(image)
    print(image.size)
    img_arr_decomposed = [img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]]

    for seq,each in enumerate(img_arr_decomposed):
        U, Sigma, V_hermit = svd(each)
        result = np.matmul(np.matmul(U,Sigma),V_hermit)    
        img_arr[:,:,seq] = result

    output = Image.fromarray(img_arr)
    output.save("output.png")



def main():
    parser = argparse.ArgumentParser(description="A small implementation of SVD compression.")
    parser.add_argument('filename', help="You need to parse a filename of the image you want to compress.")
    args = parser.parse_args()

    image_compression(args.filename)

if __name__ == "__main__":
    main()