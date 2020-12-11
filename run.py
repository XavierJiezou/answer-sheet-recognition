import os

path = os.getcwd()
pics = os.listdir(path)
pics.remove('run.py')
for index, name in enumerate(pics):
    os.rename(name,index+1)