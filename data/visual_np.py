import numpy as np

def print_npy(file_path):
    # 加载 .npy 文件
    data = np.load(file_path,allow_pickle=True)
    
    # 输出文件的基本信息
    print(f"Loaded data from {file_path}:")
    print(f"Shape: {data.shape}, Data type: {data.dtype}")
    
    # 输出数据内容
    print("Data content:")
    print(data)

def print_npz(file_path):
    # 加载 .npz 文件
    data = np.load(file_path)
    
    # 输出文件的基本信息
    print(f"Loaded .npz file: {file_path}")
    print("Available arrays:")
    print(data.files)
    
    # 输出每个数组的内容
    for array_name in data.files:
        array_data = data[array_name]
        print(f"\nArray '{array_name}':")
        print(f"Shape: {array_data.shape}, Data type: {array_data.dtype}")
        print("Array content:")
        print(array_data)

# 示例用法：
# 你可以直接调用这些函数来输出文件内容
print_npy('COCO2014_idx/COCO2014_val_images.npy')
print_npy('COCO2014_idx/COCO2014_val_label_to_idx_dict.npy')
# print_npz('VG_idx/VG_test_labels.npz')
