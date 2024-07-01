import os
import shutil
import random

# Split the dataset to Training, Validation and Testing

os.chdir(r"C:\Users\Mo Khaled\PycharmProjects\OpenCv\Sign-Language-Digits-Dataset")
if os.path.isdir(r'train/0/') is False:
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')

    for i in range(0,10):
        shutil.move(f'{i}', 'train')
        os.mkdir(f'valid/{i}')
        os.mkdir(f'test/{i}')

    for i in range(0, 10):
        valid_sample = random.sample(os.listdir(f'train/{i}'),30)
        for j in valid_sample:
            shutil.move(f'train/{i}/{j}', f'valid/{i}')

        test_sample = random.sample(os.listdir(f'train/{i}'), 5)
        for j in test_sample:
            shutil.move(f'train/{i}/{j}', f'test/{i}')






