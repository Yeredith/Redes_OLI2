from model_drn import DRN
from data_drn import data_load,data_decode,data_prepare
from data_drn import data_patch,data_augment,data_normalize
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam,SGD
import matplotlib.pyplot as plt
import os
import math
import tensorflow as tf

data_path = os.path.join(os.getcwd(), 'train_OLI','train_hr_png')
input_size = 64
channel = 3
scale = 4
dual = True #indicador booleano que determina si se utiliza el modelo dual durante el entrenamiento.
input_shape = (input_size,input_size,channel)

dataset = tf.data.Dataset.from_tensor_slices(data_load(data_path))

dataset = dataset.map(data_decode,tf.data.experimental.AUTOTUNE)
dataset = dataset.map(lambda x: data_prepare(x,scale,input_shape),tf.data.experimental.AUTOTUNE)
dataset = dataset.cache()
dataset = dataset.map(lambda x,y: data_patch(x,y,scale,input_shape),tf.data.experimental.AUTOTUNE)
dataset = dataset.map(data_augment,tf.data.experimental.AUTOTUNE)
dataset = dataset.map(data_normalize,tf.data.experimental.AUTOTUNE)
dataset = dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

model = DRN(input_shape=input_shape,model='DRN-S',scale=scale,dual=dual)

#tasas de aprendizaje basado en el coseno anidado.
#Ajusta dinámicamente la tasa de aprendizaje durante el entrenamiento.
def CosineAnnealingScheduler(T_max=30,lr_max=0.001,lr_min=0.00009,Pi=tf.constant(math.pi)):
    def scheduler(epoch, lr):
        lr = lr_min + (lr_max - lr_min) * 0.5*(1 + tf.math.cos(Pi * epoch / T_max))
        return lr
    return scheduler

#funciones de pérdida para el entrenamiento. loss se utiliza si 
#no se utiliza el modelo dual; dual_loss se utiliza si se utiliza el modelo dual.
def loss(y_true, y_pred):
    loss = tf.math.reduce_mean(tf.keras.losses.MAE(y_true,y_pred))
    return loss
def dual_loss(y_true, y_pred):
    lr, sr2lr = tf.split(y_pred, 2, axis=-1)
    loss = tf.math.reduce_mean(tf.keras.losses.MAE(lr,sr2lr))
    return 0.1*loss

# Model checkpoint setup
model_path = os.path.join(os.getcwd(), 'models_drn','DRN_S')
model_name = "weight-{epoch:03d}-{loss:.4f}.h5"
if not os.path.exists(model_path):
    os.mkdir(model_path)

    
checkpoint = ModelCheckpoint(os.path.join(model_path, model_name), period=20,save_best_only=False,save_weights_only=True)
lrscheduler = LearningRateScheduler(CosineAnnealingScheduler())
opt = Adam(1e-3)
if dual:
    model.compile(loss=[loss]+[dual_loss for i in range(int(math.log(scale,2)))], optimizer=opt)
else:
    model.compile(loss='mean_absolute_error', optimizer=opt)
    
history = model.fit(dataset, epochs=400, callbacks=[checkpoint, lrscheduler])

# Plot the loss
plt.plot(history.history['loss'], label='Entrenamiento')
plt.title('Perdidas del modelo DRN')
plt.xlabel('Epocas')
plt.ylabel('Perdida')
plt.legend()
plt.show()
