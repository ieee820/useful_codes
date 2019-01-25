#遍历所有文件夹下面的文件
import glob
for filename in glob.iglob('src/**/*.c', recursive=True):
    print(filename)
