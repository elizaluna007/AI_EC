import os
import pandas as pd

def print_csv_filenames_and_headers(dir_path):
    # 遍历指定目录下所有csv文件
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            # 构建文件完整路径
            file_path = os.path.join(dir_path, filename)

            # 检查文件是否为空
            if os.path.getsize(file_path) == 0:
                print(f"{filename}()")
            else:
                # 读入csv文件的前一行，获取列名
                df = pd.read_csv(file_path, nrows=0)
                headers = ",".join(df.columns)
                # 打印文件名和列名
                print(f"{filename}({headers})")

# 指定"data"文件夹路径
dir_path = "E:\mypyhon\e-commerce-ai\data"

print_csv_filenames_and_headers(dir_path)
