import tensorflow as tf
import numpy as np

class LinearModel(tf.Module):
    def __init__(self):
        # Initialize slope (m) and intercept (c)
        self.m = tf.Variable(0.0, name='slope')
        self.c = tf.Variable(0.0, name='intercept')

    @tf.function
    def __call__(self, x):
        return self.m * x + self.c

model = LinearModel()

# Sample training data (x, y) - y = 2x + 1
x_train = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y_train = np.array([1, 3, 5, 7, 9], dtype=np.float32)

def loss_fn(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Stochastic Gradient Descent optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.01)

def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
    gradients = tape.gradient(loss, [model.m, model.c])
    optimizer.apply_gradients(zip(gradients, [model.m, model.c]))
    return loss

for epoch in range(1000):
    loss = train_step(x_train, y_train)

print(f'Trained slope (m): {model.m.numpy()}')
print(f'Trained intercept (c): {model.c.numpy()}')
