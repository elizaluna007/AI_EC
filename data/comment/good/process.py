# 打开 good.txt 文件进行读取
with open(r'D:\360MoveData\Users\Luna\Desktop\实训\e-commerce-ai\data\comment\good\good.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()  # 读取所有行

# 写入前100行到 good_1.txt 文件
with open(r'D:\360MoveData\Users\Luna\Desktop\实训\e-commerce-ai\data\comment\good\good_1.txt', 'w', encoding='utf-8') as file:
    file.writelines(lines[:1000000])  # 写入前1000000行
    
# 写入前100行到 good_1.txt 文件
with open(r'D:\360MoveData\Users\Luna\Desktop\实训\e-commerce-ai\data\comment\good\good_2.txt', 'w', encoding='utf-8') as file:
    file.writelines(lines[1000000:])  # 写入后面的行
