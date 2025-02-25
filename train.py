import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Carregar dataset corretamente
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar imagens (0 a 1) e adicionar dimensão extra para (28,28,1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)  # De (60000, 28, 28) para (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)

# Criar modelo
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar modelo
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.0003),
              metrics=['accuracy'])

# Treinar modelo
model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))

# Salvar no formato correto para Keras 3
model.save("model.keras")  # Antes era "model", agora usa ".keras"

# Salvar pesos separadamente, se necessário
model.save_weights("weights.weights.h5")
