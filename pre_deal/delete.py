import os

def remove_images_from_dataset(imgs_dir, delete_names):
    for root, dirs, imgs in os.walk(imgs_dir):
        print(len(imgs))  # 10000
        for img in imgs:
            img_name = os.path.splitext(img)[0]  # 获取文件名（去除扩展名）
            
            if img_name in delete_names:
                img_path = os.path.join(root, img)
                os.remove(img_path)

imgs_dir = "/home/wqliu/Workspace/2023/preprocess/train_resized/labels"
delete_txt_path = "/home/wqliu/Workspace/2023/preprocess/dirty.txt"

# 读取待删除的图像名称列表
with open(delete_txt_path, "r") as f:
    delete_names = [line.strip() for line in f]

# 移除指定的图像文件
remove_images_from_dataset(imgs_dir, delete_names)