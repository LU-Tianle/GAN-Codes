# -*- encoding: utf-8 -*-
# @Software: PyCharm 
# @File    : utils.py 
# @Time    : 2018/11/15 10:07
# @Author  : LU Tianle

"""
"""

import os
import matplotlib.pyplot as plt
import shutil


def create_folder(path, clean_folder=False):
    """
    create a folder if it's not exist
    :param path: folder path,
    :param clean_path: if true, all folders and files in the path will be deleted recursively
    """
    if os.path.exists(path):
        if not clean_folder:
            shutil.rmtree(path)  # Recursively delete a directory tree
            os.makedirs(path)
    else:
        os.makedirs(path)




    # def __latest_check_point_epoch(self):
    #     """
    #     :return: the number of latest check point epochs that had been saved in the '.../checkpoints/'
    #     """
    #     for folder in os.listdir(self.check_points_path):  # delete illegal files and folders
    #         path = self.check_points_path + os.path.sep + folder
    #         if os.path.isfile(path):
    #             os.remove(path)
    #         if re.match(r'epoch_', folder) is None:
    #             shutil.rmtree(path)
    #     dir_list = os.listdir(self.check_points_path)
    #     dir_list.sort()
    #     return dir_list.pop().split('_')[1]
