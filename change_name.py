import os
import shutil


# 文件A和文件B的路径
path_A = 'D:/flir-adv-mine/evaluate_black/patch/'
path_B = 'D:/flir-adv-mine/evaluate_black/clean/'

# 获取文件A和文件B的文件名列表
files_A = os.listdir(path_A)
files_B = os.listdir(path_B)

# 确保文件A和文件B中的文件名数量一致
if len(files_A) != len(files_B):
    print("文件数量不匹配")
else:
    # 遍历文件A和文件B的文件名，并逐一替换
    for i in range(len(files_A)):
        old_file_name = os.path.join(path_A, files_A[i])
        new_file_name = os.path.join(path_B, files_B[i])

        # 使用os.rename()函数将文件名替换
        shutil.move(old_file_name, new_file_name)
        print(f"已将 {old_file_name} 替换为 {new_file_name}")

print("文件名替换完成")
