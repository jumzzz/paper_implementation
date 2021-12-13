import os

# from torchtext.utils import download_from_url, extract_archive

import tensorflow.keras as keras
from PIL import Image
from random import shuffle



def make_dir(target_dir):
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)



class CatDogDataset:

    def __init__(self, download_root, 
                    keep_zip=False, resize_to=(256,256)):
        self.download_root = download_root

        make_dir(self.download_root)

        download_path = os.path.join(download_root, 'cats_vs_dogs.zip')
        cats_dog_uri = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'
        self.dataset_folder = os.path.join(download_root, 'cats_vs_dogs')

        make_dir(self.download_root)

        self.resize_to = resize_to

        keras.utils.get_file('cats_vs_dogs', 
                    origin=cats_dog_uri, cache_dir=self.download_root, extract=True)

        # download_from_url(cats_dog_uri, download_path)
        
        # self.dataset_folder = os.path.join(download_root, 'cats_vs_dogs')
        # self.extracted_files = extract_archive(download_path, self.dataset_folder)

        # if not keep_zip:
        #     os.remove(download_path)

        # self.init_directories()



    def init_directories(self):

        self.dataset_source = os.path.join(self.dataset_folder, 'PetImages')

        self.dataset_resized = os.path.join(self.dataset_folder, 'resized')
        self.dataset_train = os.path.join(self.dataset_resized, 'train')
        self.dataset_test = os.path.join(self.dataset_resized, 'test')

        make_dir(self.dataset_resized)
        make_dir(self.dataset_train)
        make_dir(self.dataset_test)

        self.train_cat_dir = os.path.join(self.dataset_train, 'Cat')
        self.train_dog_dir = os.path.join(self.dataset_train, 'Dog')

        self.test_cat_dir = os.path.join(self.dataset_test, 'Cat')
        self.test_dog_dir = os.path.join(self.dataset_test, 'Dog')

        make_dir(self.train_cat_dir)
        make_dir(self.train_dog_dir)

        make_dir(self.test_cat_dir)
        make_dir(self.test_dog_dir)


    def has_processed(self):
        
        has_train_dog = len(os.listdir(self.train_dog_dir)) > 10000
        has_train_cat = len(os.listdir(self.train_cat_dir)) > 10000

        has_test_dog = len(os.listdir(self.test_dog_dir)) > 1000
        has_test_cat = len(os.listdir(self.test_cat_dir)) > 1000

        has_data = has_train_dog and has_train_cat and \
            has_test_dog and has_test_cat

        if has_data:
            print('Files already preprocessed. No need to preprocess again.')
        else:
            print('Files for preprocessing is missing. Will Preprocess data.')

        return has_data
        
    
    def prepocess_files(self):
        if not self.has_processed():
            cat_src_dir = os.path.join(self.dataset_source, 'Cat')
            dog_src_dir = os.path.join(self.dataset_source, 'Dog')

            self.acquire_train_test(cat_src_dir, dog_src_dir, self.dataset_train, self.dataset_test)

        return self.dataset_train, self.dataset_test



    def list_paths(self, target_dir):
        fnames = os.listdir(target_dir)
        return [os.path.join(target_dir, fn) for fn in fnames if fn.endswith('.jpg')]

    def generate_dst(self, src, target_dir):
        
        parent_dir = src.split(os.sep)[-2]
        fname = src.split(os.sep)[-1]
        
        if parent_dir == 'Cat':
            return os.path.join(target_dir, 'Cat', fname)
        else:
            return os.path.join(target_dir, 'Dog', fname)
        


    def create_train_test_paths(self, file_paths, train_dir, test_dir, train_split=0.9):
        
        fps_copy = file_paths.copy()
        
        shuffle(fps_copy)
        
        total = len(fps_copy)
        train_len = int(total * train_split)
        
        srcs_train = fps_copy[0:train_len]
        srcs_test = fps_copy[train_len:]
        
        
        dsts_train = [self.generate_dst(src, train_dir) for src in srcs_train]
        dsts_test = [self.generate_dst(src, test_dir) for src in srcs_test]
        
        src_paths = srcs_train + srcs_test
        dst_paths  = dsts_train + dsts_test
        
        return src_paths, dst_paths
    
    
    
    def resize_image_file(self, src, dst):
        
        with Image.open(src) as img:
            im_resized = img.resize(self.resize_to)
            im_resized.save(dst)
        
        

    def acquire_train_test(self, cats_dir, dogs_dir, train_dir, test_dir, 
                        train_split=0.9):

        cat_img_src_paths = self.list_paths(cats_dir)
        dog_img_src_paths = self.list_paths(dogs_dir)
        

        raw_src_paths = cat_img_src_paths + dog_img_src_paths
        
        img_src_paths, img_dst_paths = self.create_train_test_paths(raw_src_paths, 
                                                            train_dir,
                                                            test_dir, train_split)
        
        assert len(img_src_paths) == len(img_dst_paths), 'Mismatch on src and dst of filepaths.'
        
        total_images = len(img_src_paths)
        
        for idx, sd in enumerate(zip(img_src_paths, img_dst_paths)):
            try:
                self.resize_image_file(sd[0], sd[1])
                print(f'[{idx}/{total_images}] Resizing from {sd[0]} to {sd[1]} success.', end='\r')
            
            except Exception as e:
                print('')
                print(f'[{idx}/{total_images}] Resizing from {sd[0]} to {sd[1]} failed.', end='\r')
            
            if os.path.isfile(sd[1]):
                if os.path.getsize(sd[1]) == 0:
                    # Discarding zero byte file
                    print(f'[{idx}/{total_images}] Removing zero-byte file {sd[1]}', end='\r')
                    os.remove(sd[1])



