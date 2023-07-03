from PIL import Image
import argparse
import numpy as np
from sympy import Matrix

from versions.svd_0 import svd
#from versions.svd_1 import svd

def image_compression(path: str, rank: int):
    image = Image.open(path)
    img_arr = np.array(image)
    print(image.size)
    img_arr_decomposed = [img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]]

    for seq,each in enumerate(img_arr_decomposed):
        U, Sigma, V_hermit = svd(each)
        Sigma_Compressed = Sigma #Provisional - to test the algorithm without compression

        result = np.matmul(np.matmul(U,Sigma_Compressed),V_hermit)    
        img_arr[:,:,seq] = result

    output = Image.fromarray(img_arr)
    output.save("output.png")



def main():
    parser = argparse.ArgumentParser(description="A small implementation of SVD compression.")
    parser.add_argument('filename', help="You need to parse a filename of the image you want to compress.")
    parser.add_argument("-r", "--rank", type=float, help="With this parameter you specifiy the rank of the final compressed matrix", default=1)
    args = parser.parse_args()

    image_compression(args.filename, args.rank)

if __name__ == "__main__":
    main()