# load model
import sys
import keras
import numpy as np
from keras.models import load_model, Model

from load_bagnet_model import load_bagnet_model
from load_test_image import load_test_image
from generate_prediction import generate_prediction
from utilities.convert_h5_to_pb import convert_h5_to_pb

from data.image_net.classes import image_net_classes


def main():


        np.set_printoptions(threshold=sys.maxsize)





        bagnet_model = load_bagnet_model()


        def remove_last_layer_from_model(
                model,
        ):
                layer_names = [
                layer.name
                for layer
                in model.layers
                ]

                last_layer_name = layer_names[-2]

                model_with_last_layer_removed = Model(
                inputs=model.input,
                outputs=model.get_layer(last_layer_name).output,
                )

                return model_with_last_layer_removed


        bagnet_model_with_last_layer_removed = remove_last_layer_from_model(
        bagnet_model,
        )

        # bagnet_model_with_last_layer_removed.save("bagnet32_without_last_layer.h5")


        test_image = load_test_image()

        result = generate_prediction(
            bagnet_model,
            test_image,
            image_net_classes,
        )


        print("result: ", result)



main()

# convert_h5_to_pb(
#     path_to_h5="bagnet32_without_last_layer.h5",
#     export_path="bagnet32_without_last_layer",
# )


# result = bagnet_model_with_last_layer_removed.predict(
#     np.array([test_image]),
#     batch_size=1,
# )

