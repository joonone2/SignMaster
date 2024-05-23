import tensorflow as tf

# TensorFlow 모델 로드
model = tf.keras.models.load_model('models/0518_02.h5')

# TensorFlow Lite 변환기 생성
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

# TensorFlow Lite 모델로 변환
tflite_model = converter.convert()

# 변환된 모델 저장
with open('my_model.tflite', 'wb') as f:
    f.write(tflite_model)
