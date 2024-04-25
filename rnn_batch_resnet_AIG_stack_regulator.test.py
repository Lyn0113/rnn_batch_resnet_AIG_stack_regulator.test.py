import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Activation
from keras.initializers import GlorotUniform
from utils import generate_add, compute_accuracy, generate_same, generate_convert, generate_binary_and, \
    generate_binary_or, generate_binary_not, generate_binary_xor

input_num = 8
output_num = 4
lut_num = 8
lut_num1 = 8
lut_num2 = 8
lut_num3 = 8
lut_num4 = 8
lut_num5 = 8
lut_num6 = 8
reg_num = 8

ic_input_num = 2 * input_num + 1
ic1_input_num = 4 * lut_num + 1
ic2_input_num = 4 * lut_num1 + 1
ic3_input_num = 4 * lut_num2 + 1
ic4_input_num = 4 * lut_num3 + 1
ic5_input_num = 4 * lut_num4 + 1
ic6_input_num = 4 * lut_num5 + 1
ic7_input_num = 4 * lut_num6 + 1


def lut_operation(num_luts, inputs):
    # | id | Operator             | AB=00 | AB=01 | AB=10 | AB=11 |
    # |----|----------------------|-------|-------|-------|-------|
    # | 0  | 0                    | 0     | 0     | 0     | 0     |
    # | 1  | A and B              | 0     | 0     | 0     | 1     |
    # | 2  | not(A implies B)     | 0     | 0     | 1     | 0     |
    # | 3  | A                    | 0     | 0     | 1     | 1     |
    # | 4  | not(B implies A)     | 0     | 1     | 0     | 0     |
    # | 5  | B                    | 0     | 1     | 0     | 1     |
    # | 6  | A xor B              | 0     | 1     | 1     | 0     |
    # | 7  | A or B               | 0     | 1     | 1     | 1     |
    # | 8  | not(A or B)          | 1     | 0     | 0     | 0     |
    # | 9  | not(A xor B)         | 1     | 0     | 0     | 1     |
    # | 10 | not(B)               | 1     | 0     | 1     | 0     |
    # | 11 | B implies A          | 1     | 0     | 1     | 1     |
    # | 12 | not(A)               | 1     | 1     | 0     | 0     |
    # | 13 | A implies B          | 1     | 1     | 0     | 1     |
    # | 14 | not(A and B)         | 1     | 1     | 1     | 0     |
    # | 15 | 1                    | 1     | 1     | 1     | 1     |
    outputs = []
    for i in range(num_luts):
        # Extract weights for the i-th LUT
        a = inputs[:, 2 * i]
        b = inputs[:, 2 * i + 1]
        # output = 2.0* tf.maximum(0.0, a*b - 0.5)
        # output = 1 / (1 + tf.exp(-50 * (a*b - 0.5)))
        output = a * b
        # Apply the LUT logic
        outputs.append(output)
    return tf.stack(outputs, axis=-1)


class FeedForward(keras.Model):
    def __init__(self, units, batch_size=200):
        self.units = units
        self.batch_size = batch_size
        super(FeedForward, self).__init__()
        self.wic = self.add_weight(shape=(4 * lut_num, ic_input_num),
                                   initializer=GlorotUniform(),
                                   trainable=True,
                                   )
        self.wic1 = self.add_weight(shape=(4 * lut_num1, ic1_input_num),
                                    initializer=GlorotUniform(),
                                    trainable=True,
                                    )
        self.wic2 = self.add_weight(shape=(4 * lut_num2, ic2_input_num),
                                    initializer=GlorotUniform(),
                                    trainable=True,
                                    )
        self.wic3 = self.add_weight(shape=(4 * lut_num3, ic3_input_num),
                                    initializer=GlorotUniform(),
                                    trainable=True,
                                    )
        self.wic4 = self.add_weight(shape=(output_num, ic4_input_num),
                                    initializer=GlorotUniform(),
                                    trainable=True,
                                    )

    def call(self, inputs):
        softmax_weights = tf.nn.softmax(self.wic)
        ones = tf.expand_dims(tf.ones([tf.shape(inputs)[0]], dtype=inputs.dtype), -1)
        combined_inputs = tf.concat([inputs, 1 - inputs, ones], axis=-1)
        ic_output = tf.matmul(combined_inputs, tf.transpose(softmax_weights))
        ic_outputa, ic_outputb = tf.split(ic_output, 2, axis=-1)
        lut_output = lut_operation(lut_num, ic_outputa)

        softmax_weights1 = tf.nn.softmax(self.wic1)
        combined_inputs1 = tf.concat([ic_outputb, lut_output, 1 - lut_output, ones], axis=-1)
        ic_output1 = tf.matmul(combined_inputs1, tf.transpose(softmax_weights1))
        ic_outputa, ic_outputb = tf.split(ic_output1, 2, axis=-1)
        lut_output1 = lut_operation(lut_num1, ic_outputa)

        # softmax_weights2 = tf.nn.softmax(self.wic2)
        # combined_inputs2 = tf.concat([ic_outputb, lut_output1, 1 - lut_output1, ones], axis=-1)
        # ic_output2 = tf.matmul(combined_inputs2, tf.transpose(softmax_weights2))
        # ic_outputa, ic_outputb = tf.split(ic_output2, 2, axis=-1)
        # lut_output2 = lut_operation(lut_num2, ic_outputa)

        # softmax_weights3 = tf.nn.softmax(self.wic3)
        # combined_inputs3 = tf.concat([ic_outputb, lut_output, 1 - lut_output, ones], axis=-1)
        # ic_output3 = tf.matmul(combined_inputs3, tf.transpose(softmax_weights3))
        # ic_outputa, ic_outputb = tf.split(ic_output3, 2, axis=-1)
        # lut_output3 = lut_operation(lut_num3, ic_outputa)

        softmax_weights4 = self.wic4
        ones = tf.expand_dims(tf.ones([tf.shape(lut_output1)[0]], dtype=inputs.dtype), -1)
        combined_inputs4 = tf.concat([ic_outputb, lut_output1, 1 - lut_output1, ones], axis=-1)
        ic_output4 = tf.matmul(combined_inputs4, tf.transpose(softmax_weights4))

        return ic_output4


num_epochs = 20
batch_size = 32
# 创建FeedForward层的实例，神经元个数为units
model = FeedForward(units=256)

# Generate data using the modified function
x_train, y_train, x_val, y_val, x_test, y_test = generate_binary_xor(output_num, batch_size * 3000)
# inputs,outputs=generate_binary_and(8,batch_size*1000)
# inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
# outputs = tf.convert_to_tensor(outputs, dtype=tf.float32)

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

# 编译模型
model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1, min_delta=0.0001)
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, callbacks=[early_stopping])

print("Weights wic:", model.wic4.numpy())
# 进行预测
predictions = model.predict(x_test)
accuracy = compute_accuracy(predictions, y_test)
print("accuracy=", accuracy)
