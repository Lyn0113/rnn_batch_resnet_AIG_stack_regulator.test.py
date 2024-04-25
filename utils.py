import numpy as np
import tensorflow as tf
input_num = 16
output_num = 8
lut_num = 20
lut_num1 = 20
lut_num2 = 20
lut_num3 = 20
lut_num4 = 20
lut_num5 = 20
lut_num6 = 20
reg_num = 8

ic_input_num = 2 * (input_num + reg_num) + 1
ic1_input_num = 4 * lut_num + 1
ic2_input_num = 4 * lut_num1 + 1
ic3_input_num = 4 * lut_num2 + 1
ic4_input_num = 4 * lut_num3 + 1
ic5_input_num = 4 * lut_num4 + 1
ic6_input_num = 4 * lut_num5 + 1
ic7_input_num = 4 * lut_num6 + 1


# 创建相反的数据
def generate_convert(binary_dimension,total_samples):
    def int_to_binary(n):
        string_num=bin(n)[2:]
        # 使用np.array将列表转换为NumPy数组
        int_list_left = [int(char) for char in string_num]
        num_len=len(int_list_left)
        binary = [0]*(binary_dimension-num_len)+int_list_left
        return binary
    inputs = []
    outputs = []
    for _ in range (total_samples):
        num = np.random.randint(0, 64)
    # 将数字转换为二进制数组
        binary_left = int_to_binary(num)
        inputs.append(binary_left[:])
        for i in range (len(binary_left)):
            binary_left[i]=1-binary_left[i]
        outputs.append(binary_left)
    return inputs,outputs



# 创建相等的数据
def generate_same(binary_dimension,total_samples):
    def int_to_binary(n):
        string_num=bin(n)[2:]
        # 使用np.array将列表转换为NumPy数组
        int_list_left = [int(char) for char in string_num]
        num_len=len(int_list_left)
        binary = [0]*(binary_dimension-num_len)+int_list_left
        return binary
    inputs = []
    outputs = []
    for _ in range (total_samples):
        num = np.random.randint(0, 64)
    # 将数字转换为二进制数组
        binary_left = int_to_binary(num)
        inputs.append(binary_left)
        outputs.append(binary_left)
    return inputs,outputs

# 创建加法数据
def generate_add(binary_dimension,total_samples):
    def int_to_binary(n):
        string_num=bin(n)[2:]
        # 使用np.array将列表转换为NumPy数组
        int_list_left = [int(char) for char in string_num]
        num_len=len(int_list_left)
        binary = [0]*(binary_dimension-num_len)+int_list_left
        return binary
    inputs=[]
    outputs=[]
    for _ in range (total_samples):
        left=np.random.randint(0,64)
        right=np.random.randint(0,64)
        # 将数字转换为二进制数组
        binary_left = int_to_binary(left)
        binary_right = int_to_binary(right)
        # 计算
        sum_value = left + right
        sum_value=int_to_binary(sum_value)
        # 将二进制输入和真实和作为一对数据添加到输入列表中
        inputs.append(binary_left+binary_right)
        outputs.append(sum_value)
    # split dataset
    split_train = int(0.7 * len(inputs))
    split_val = int(0.8 * len(inputs))
    x_train = inputs[:split_train]
    y_train = outputs[:split_train]
    x_val = inputs[split_train:split_val]
    y_val = outputs[split_train:split_val]
    x_test = inputs[split_val:]
    y_test = outputs[split_val:]
    return x_train, y_train, x_val, y_val, x_test, y_test


# 创建二进制取and数据
def generate_binary_and(binary_dimension,total_samples):
    def int_to_binary(n):
        string_num=bin(n)[2:]
        # 使用np.array将列表转换为NumPy数组
        int_list_left = [int(char) for char in string_num]
        num_len=len(int_list_left)
        binary = [0]*(binary_dimension-num_len)+int_list_left
        return binary
    inputs=[]
    outputs=[]
    for _ in range (total_samples):
        left=np.random.randint(0,64)
        right=np.random.randint(0,64)
        # 将数字转换为二进制数组
        binary_left = int_to_binary(left)
        binary_right = int_to_binary(right)
        res=[]
        for i in range (len(binary_left)):
            res.append(binary_left[i]*binary_right[i])
        # 将二进制输入和真实和作为一对数据添加到输入列表中
        inputs.append(binary_left+binary_right)
        outputs.append(res)
    # split dataset
    split_train = int(0.7 * len(inputs))
    split_val = int(0.8 * len(inputs))
    x_train = inputs[:split_train]
    y_train = outputs[:split_train]
    x_val = inputs[split_train:split_val]
    y_val = outputs[split_train:split_val]
    x_test = inputs[split_val:]
    y_test = outputs[split_val:]
    return x_train, y_train, x_val, y_val, x_test, y_test

# 创建二进制取or数据
def generate_binary_or(binary_dimension,total_samples):
    def int_to_binary(n):
        string_num=bin(n)[2:]
        # 使用np.array将列表转换为NumPy数组
        int_list_left = [int(char) for char in string_num]
        num_len=len(int_list_left)
        binary = [0]*(binary_dimension-num_len)+int_list_left
        return binary
    inputs=[]
    outputs=[]
    for _ in range (total_samples):
        left=np.random.randint(0,256)
        right=np.random.randint(0,256)
        # 将数字转换为二进制数组
        binary_left = int_to_binary(left)
        binary_right = int_to_binary(right)
        res=[]
        for i in range (len(binary_left)):
            res.append(binary_left[i]+binary_right[i]-binary_left[i]*binary_right[i])
        # 将二进制输入和真实和作为一对数据添加到输入列表中
        inputs.append(binary_left+binary_right)
        outputs.append(res)
    # split dataset
    split_train = int(0.7 * len(inputs))
    split_val = int(0.8 * len(inputs))
    x_train = inputs[:split_train]
    y_train = outputs[:split_train]
    x_val = inputs[split_train:split_val]
    y_val = outputs[split_train:split_val]
    x_test = inputs[split_val:]
    y_test = outputs[split_val:]
    return x_train, y_train, x_val, y_val, x_test, y_test
# 创建二进制取xor数据
def generate_binary_xor(binary_dimension,total_samples):
    def int_to_binary(n):
        string_num=bin(n)[2:]
        # 使用np.array将列表转换为NumPy数组
        int_list_left = [int(char) for char in string_num]
        num_len=len(int_list_left)
        binary = [0]*(binary_dimension-num_len)+int_list_left
        return binary
    inputs=[]
    outputs=[]
    for _ in range (total_samples):
        left=np.random.randint(0,2**binary_dimension)
        right=np.random.randint(0,2**binary_dimension)
        # 将数字转换为二进制数组
        binary_left = int_to_binary(left)
        binary_right = int_to_binary(right)
        res=[]
        for i in range (len(binary_left)):
            res.append(binary_left[i]^binary_right[i])
        # 将二进制输入和真实和作为一对数据添加到输入列表中
        inputs.append(binary_left+binary_right)
        outputs.append(res)
    # split dataset
    split_train = int(0.7 * len(inputs))
    split_val = int(0.8 * len(inputs))
    x_train = inputs[:split_train]
    y_train = outputs[:split_train]
    x_val = inputs[split_train:split_val]
    y_val = outputs[split_train:split_val]
    x_test = inputs[split_val:]
    y_test = outputs[split_val:]
    return x_train, y_train, x_val, y_val, x_test, y_test
# 创建二进制取not数据
def generate_binary_not(binary_dimension,total_samples):
    def int_to_binary(n):
        string_num=bin(n)[2:]
        # 使用np.array将列表转换为NumPy数组
        int_list_left = [int(char) for char in string_num]
        num_len=len(int_list_left)
        binary = [0]*(binary_dimension-num_len)+int_list_left
        return binary
    inputs=[]
    outputs=[]
    for _ in range (total_samples):
        left=np.random.randint(0,64)
        right=np.random.randint(0,64)
        # 将数字转换为二进制数组
        binary_left = int_to_binary(left)
        binary_right = int_to_binary(right)
        res=[]
        for i in range (len(binary_left)):
            res.append(1-binary_left[i])
        # 将二进制输入和真实和作为一对数据添加到输入列表中
        inputs.append(binary_left[:]+binary_right)
        outputs.append(res)
    # split dataset
    split_train = int(0.7 * len(inputs))
    split_val = int(0.8 * len(inputs))
    x_train = inputs[:split_train]
    y_train = outputs[:split_train]
    x_val = inputs[split_train:split_val]
    y_val = outputs[split_train:split_val]
    x_test = inputs[split_val:]
    y_test = outputs[split_val:]
    return x_train, y_train, x_val, y_val, x_test, y_test
def lut_operation(num_luts, inputs):

    outputs = []
    for i in range(num_luts):
        a = inputs[:, 2 * i]
        b = inputs[:, 2 * i + 1]
        output = a * b
        outputs.append(output)
    return tf.stack(outputs, axis=-1)
def compute_accuracy(output, y_test):
    # 将输出和测试标签展平为一维张量
    output = tf.reshape(output, [-1])
    y_test = tf.reshape(y_test, [-1])

    # 将输出和测试标签转换为numpy数组
    output = output.numpy()
    y_test = y_test.numpy()

    # 四舍五入输出
    output = (output > 0.5).astype(int)

    # 计算准确率
    correct = (output == y_test).sum().item()
    # 计算准确率
    accuracy = correct / len(y_test)

    return accuracy