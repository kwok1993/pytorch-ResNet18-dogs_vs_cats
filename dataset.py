import shutil
import os
import random


class DataClean:
    def __init__(self):
        if not os.path.exists(os.path.join("dogs_vs_cats", "train")):
            os.mkdir(os.path.join("dogs_vs_cats", "train"))
            os.mkdir(os.path.join("dogs_vs_cats", "train", "dog"))
            os.mkdir(os.path.join("dogs_vs_cats", "train", "cat"))

        if not os.path.exists(os.path.join("dogs_vs_cats", "valid")):
            os.mkdir(os.path.join("dogs_vs_cats", "valid"))
            os.mkdir(os.path.join("dogs_vs_cats", "valid", "dog"))
            os.mkdir(os.path.join("dogs_vs_cats", "valid", "cat"))

        self.files = os.listdir(os.path.join("dogs_vs_cats", "train1"))

    def classification(self):
        valid_files = random.sample(self.files, 2000)
        for valid_file in valid_files:
            folder = valid_file.split(".")[0]
            shutil.copyfile(os.path.join("dogs_vs_cats", "train1", valid_file),
                            os.path.join("dogs_vs_cats", "valid", folder, valid_file))
            self.files.remove(valid_file)

        for train_file in self.files:
            folder = train_file.split(".")[0]
            shutil.copyfile(os.path.join("dogs_vs_cats", "train1", train_file),
                            os.path.join("dogs_vs_cats", "train", folder, train_file))


if __name__ == '__main__':
    data_clean = DataClean()
    data_clean.classification()
