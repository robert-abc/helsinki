import argparse
import re
import os
from utils import process
from utils import deblur

parser = argparse.ArgumentParser(description=
            'Deblur images with different levels of blur.')

parser.add_argument('input_path', type=str,
                    help='Path with images to be deblurred')
parser.add_argument('output_path', type=str,
                    help='Path to save deblurred images')
parser.add_argument('deblur_level', type=int,
                    choices=range(0,20), metavar='[0-19]',
                    help='Level of blur')
parser.add_argument('--extension', dest='extension',
                    type=str, default='tif', required=False,
                    help='Image extension (default: tif)')

args = parser.parse_args()

img_names=os.listdir(args.input_path)
r=re.compile(".*"+args.extension)
img_names=list(filter(r.match,img_names))
print(f"{len(img_names)} images were found.")

for img in img_names:
    path_in=os.path.join(args.input_path,img)
    path_out=os.path.join(args.output_path,img)
    path_out=path_out[0:-3]+'png'

    img_arr,orig_dim=process.load_img(path_in,width=512,enforse_div32='EXTEND')
    print(orig_dim)

    img_pil=process.np_to_pil(img_arr)
    img_pil.save(path_out)
