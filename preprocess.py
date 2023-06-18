import glob
import random
from PIL import Image
import pickle
from datasets import Dataset
from datasets import load_from_disk


ds_test = {'image_file_path': [], 'image': [], 'labels': []}
# ds_train = {'image_file_path': [], 'image': [], 'labels': []}
# ds_validation = {'image_file_path': [], 'image': [], 'labels': []}
# ds_test = {'image_file_path': [], 'image': [], 'labels': []}

image_file_path1 = []
image1 = []
labels1 = []

imagelist = [f for f in glob.glob('data_before/*/*',
                                  recursive=True)]
random.shuffle(imagelist)
imagelist=imagelist[:5000]
print(len(imagelist))
print(imagelist[1])
i = 0
for path in imagelist:
    if '.jpg' or '.png' in path:
        if '.txt' not in path:
                temp = (path).split('\\', 6)
                s2 = temp[1]
                image_file_path1.append(path)
                image1.append(Image.open(path))
                if s2 == 'live':
                    labels1.append(1)
                else:
                    labels1.append(0)
    i += 1
    print(i)

# ds_train['image_file_path'] = image_file_path1
# ds_train['image'] = image1
# ds_train['labels'] = labels1
#
# ds_train = Dataset.from_dict(ds_train)
# ds_test = Dataset.from_dict(ds_test)
# ds_validation = Dataset.from_dict(ds_validation)

ds_test['image_file_path'] = image_file_path1
ds_test['image'] = image1
ds_test['labels'] = labels1

ds_test = Dataset.from_dict(ds_test)



print(ds_test)

ds_test.save_to_disk('beforeall')
