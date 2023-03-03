import json
import cv2
import os
import numpy as np
LOCAL_ROOT='./'
LOCAL_IMAGE_LIST_PATH='image_list.json'
def get_image(max_number=None):
    with open(LOCAL_ROOT+LOCAL_IMAGE_LIST_PATH) as f:
        image_list = json.load(f)
    print("got local image list, {} image".format(len(image_list.keys())))
    Batch_size = 1024
    print("Batch_size=, {}".format(Batch_size))
    n = 0
    final_image = []
    final_image_id = []
    for idx,image_id in enumerate(image_list):
        # get image from local file
        try:
            image = cv2.imread(image_id)
            final_image.append(image)
            final_image_id.append(image_id)
            n += 1
        except:
            print("Failed to read image: {}".format(image_id))
            raise

        if n == Batch_size or idx == len(image_list) - 1:
            np_final_image_id = np.array(final_image_id)
            np_final_image = np.array(final_image)
            n = 0
            final_image = []
            final_image_id = []
            yield np_final_image_id, np_final_image
if __name__=='__main__':
    df=get_image(300)
    