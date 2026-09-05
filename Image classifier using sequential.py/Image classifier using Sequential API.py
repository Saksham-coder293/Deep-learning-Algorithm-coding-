import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train =  X_train_full[:-5000], y_train_full[:-5000]
X_valid,y_valid = X_train_full[-5000:], y_train_full[-5000:]

X_train.shape
y_train.shape

class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

class_names[y_train[0]]


tf.random.set_seed(42)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[28,28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300,activation = "relu"))
model.add(tf.keras.layers.Dense(100,activation = "relu"))
model.add(tf.keras.layers.Dense(10,activation = "softmax"))

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_size = [28,28]),
    tf.keras.layers.Dense(300, activation = "relu"),
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(10, activation = "softmax")
])


model.summary()
model.layers

hidden1 = model.layers[1]
hidden1.name
model.get_layer('dense') is hidden1

weights, biases = hidden1.get_weights
weights
weights.shape
biases.shape

#compiling the model
model.compile(loss = "sparse_categorical_crossentropy",optimizer = "sgd",metrics = ["accuracy"])

history = model.fit(X_train,y_train,epochs = 30,validation_data = (X_valid,y_valid))