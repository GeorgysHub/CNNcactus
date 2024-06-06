import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, Callback
import matplotlib.pyplot as plt

class EarlyStopAtThreshold(Callback):
    def __init__(self, monitor='val_accuracy', threshold=0.99):
        super().__init__()
        self.monitor = monitor
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val = logs.get(self.monitor)
        if val is not None and val >= self.threshold:
            print(f"\nTraining stopped: {self.monitor} reached {self.threshold}")
            self.model.stop_training = True

def augment_training_data():
    return ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',
        brightness_range=[0.1, 1],
        channel_shift_range=150.0
    )

def create_validation_data_gen():
    return ImageDataGenerator()

def adjust_learning_rate(epoch, lr):
    return lr if epoch < 10 else lr * tf.math.exp(-0.1)

def fit_model(model, data, labels, batch_size=32, epochs=50):
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    train_data_gen = augment_training_data()
    val_data_gen = create_validation_data_gen()

    train_gen = train_data_gen.flow(X_train, y_train, batch_size=batch_size)
    val_gen = val_data_gen.flow(X_val, y_val, batch_size=batch_size)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    lr_sched = LearningRateScheduler(adjust_learning_rate)
    model_checkpoint = ModelCheckpoint('cactus_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop_threshold = EarlyStopAtThreshold()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr, lr_sched, model_checkpoint, early_stop_threshold]
    )

    for i, (loss, acc, val_loss, val_acc) in enumerate(zip(history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy'])):
        print(f"Epoch {i+1}/{epochs} - loss: {loss:.4f} - accuracy: {acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

    display_training_plots(history)
    return model


def display_training_plots(history):
    epochs = range(1, len(history.history['accuracy']) + 1)

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(14, 5))

    # Plotting Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
