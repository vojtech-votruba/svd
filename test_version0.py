#A no-compression test on how fast the SVD algorithm runs.

import cv2
import argparse

from versions.svd_0 import svd

def image_compression(path: str):
    img = cv2.imread(path)
    img_arr_decomposed = list(cv2.split(img))

    for seq,each in enumerate(img_arr_decomposed):
        U, Sigma, V_hermit = svd(each)
        result = U @ Sigma @ V_hermit    
        img_arr_decomposed[seq] = result

    output = (cv2.merge(img_arr_decomposed))
    cv2.imwrite("output.png", output)

def main():
    parser = argparse.ArgumentParser(description="A small implementation of SVD compression.")
    parser.add_argument('filename', help="You need to parse a filename of the image you want to compress.")
    args = parser.parse_args()

    image_compression(args.filename)

if __name__ == "__main__":
    main()