# coding=utf-8
import os
import subprocess
from configuration_file import configuration_file
import xlrd

path = configuration_file().rootpath
jar_path = configuration_file().jar_path


def mkdir(path):
    path = path.strip()

    path = path.rstrip("\\")


    isExists = os.path.exists(path)


    if not isExists:

        os.makedirs(path)
        return True
    else:

        return False


