# 打开文件并读取内容
file_path = 'ASR.txt'  # 请将路径替换为实际的文件路径
with open(file_path, 'r') as file:
    content = file.readlines()

# 初始化计数器
total_images = len(content)
no_detection_count = 0

# 遍历文件内容
for line in content:
    # 检查每一行是否包含 "(no detections)"
    if "(no detections)" in line:
        no_detection_count += 1

# 计算占比
no_detection_percentage = (no_detection_count / total_images) * 100

# 打印结果
print(f"总共有 {total_images} 张图片，其中有 {no_detection_count} 张图片没有检测到目标。")
print(f"没有检测到目标的图片占总体的 {no_detection_percentage:.2f}%。")
