#!/usr/bin/env python
# Predict flower name from an image with predict.py along with the probability of that name.
# That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
# Basic usage: python predict.py /path/to/image checkpoint
# Options:
# Return top KK most likely classes: python predict.py input checkpoint --top_k 3
# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# Use GPU for inference: python predict.py input checkpoint --gpu


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import my_models

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', default='input/image_07970.jpg', help='Input Image (eq. input/image_08041.jpg')
parser.add_argument('checkpoint', default='checkpoint.pth', help='Checkpoint to load (.pth file)')
parser.add_argument('--top_k', default=5, help='Number of classes to return')
parser.add_argument('--category_names', default='cat_to_name.json', help='json with category names')
parser.add_argument('--gpu', default=False, help='Set to True if GPU should be used')

args = parser.parse_args()


def main():
    # font colors
    CRED = '\033[90m'
    CEND = '\033[0m'

    # Test image prediction
    #     print('(102)')
    image_path = args.data_dir
    checkpoint = args.checkpoint
    top_k = int(args.top_k)
    category_names = args.category_names
    gpu = args.category_names

    print('')
    print('FLOWER CLASSIFICATION')
    print(CRED + '  Image: data_dir: ' + image_path +
          ', checkpoint: ' + category_names +
          ', top K: ' + str(top_k) + CEND)

    probs, classes = my_models.predict(image_path, checkpoint, top_k, category_names, gpu)


# Call to get_input_args function to run the program
if __name__ == "__main__":
    main()





