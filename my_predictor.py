import tensorflow as tf
import numpy as np



def predict_with_model(model, imgpath):

    image = tf.io.read_file(imgpath)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [150,150]) # (60,60,3)
    image = tf.expand_dims(image, axis=0) # (1,60,60,3)

    predictions = model.predict(image) # [0.005, 0.00003, 0.99, 0.00 ....]
    predictions = np.argmax(predictions) # 2

    return predictions



if __name__=="__main__":

    img_path = "C:\\Users\\Domaschk\\Documents\\01_Malibu\\00_MaLiBu_Framework\\Daten\\Preprocessed\\3_Test\\1\\kreisförmige Position_sample 6_layer-00142.png"
    #img_path = "C:\\Users\\Domaschk\\Documents\\01_Malibu\\00_MaLiBu_Framework\\Daten\\Preprocessed\\3_Test\\0\\kreisförmige Position_sample 1_layer-00094.png"
    model = tf.keras.models.load_model('./Models')
    prediction = predict_with_model(model, img_path)

    print(f"prediction = {prediction}")