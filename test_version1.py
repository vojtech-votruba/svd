#An actual test with compression.

import cv2
import argparse

from versions.svd_1 import svd

def image_compression(path: str, rank: int, iter: int):
    img = cv2.imread(path)
    img_arr_decomposed = list(cv2.split(img))
    
    for seq,each in enumerate(img_arr_decomposed):
        U, Sigma, V_hermit = svd(each, iter)
        U_compressed = U[:,:rank]
        Sigma_compressed = Sigma[0:rank,:rank]
        V_hermit_compressed = V_hermit[:rank,:]
        result = U_compressed @ Sigma_compressed @ V_hermit_compressed 
        img_arr_decomposed[seq] = result

    output = (cv2.merge(img_arr_decomposed))
    cv2.imwrite("output.png", output)

def main():
    parser = argparse.ArgumentParser(description="A small implementation of SVD compression.")
    parser.add_argument('filename', help="You need to parse a filename of the image you want to compress.")
    parser.add_argument("-r", "--rank", type=int, help="With this parameter you specifiy the rank of the final compressed matrix")
    parser.add_argument("-i", "--iterations", type=int, help="Number of iterations the program should perform.")
    args = parser.parse_args()

    image_compression(args.filename, args.rank, args.iterations)

if __name__ == "__main__":
    main()