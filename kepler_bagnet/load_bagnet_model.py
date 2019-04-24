import keras
from keras.models import load_model

def load_bagnet_model(
):
    # model_urls = {
    #     'bagnet9': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/d413271344758455ac086992beb579e256447839/bagnet8.h5',
    #     'bagnet17': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/d413271344758455ac086992beb579e256447839/bagnet16.h5',
    #     'bagnet33': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/d413271344758455ac086992beb579e256447839/bagnet32.h5',
    # }

    # model_path = keras.utils.get_file(
    #     'bagnet32.h5',
    #     model_urls['bagnet33'],
    #     cache_subdir='models',
    #     file_hash='96d8842eec8b8ce5b3bc6a5f4ff3c8c0278df3722c12bc84408e1487811f8f0f')

    # keras_model = load_model(model_path)
    
    keras_model = load_model("data/bagnet_models/bagnet32.h5")

    return keras_model
