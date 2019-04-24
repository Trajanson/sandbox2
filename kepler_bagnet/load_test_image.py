# import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# import cv2


def load_test_image(
    filename="example_2.png",
):

    # this is a PIL image
    pil_image = load_img(
        "data/image_net/test_images/" + filename,
        target_size=(224, 224),
    )
    pil_image = img_to_array(pil_image)
    channels_first_pil_image = np.rollaxis(pil_image, 2, 0)

    # print("channels_first_pil_image", channels_first_pil_image)
    # print("channels_first_pil_image.shape", channels_first_pil_image.shape)

    preprocessed_pil_image = channels_first_pil_image / 255.
    preprocessed_pil_image -= np.array([0.485, 0.456, 0.406])[:, None, None]
    preprocessed_pil_image /= np.array([0.229, 0.224, 0.225])[:, None, None]

    return preprocessed_pil_image

    # # print("img", img)



    # image_bgr = cv2.imread(f"data/image_net/test_images/{filename}")

    # # print("image_bgr", image_bgr)

    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # resized_image = cv2.resize(image_rgb, (224, 224))

    # channels_first_image = np.rollaxis(resized_image, 2, 0)

    # print("channels_first_image", channels_first_image)
    # print("channels_first_image.shape", channels_first_image.shape)

    # # print("channels_first_image", channels_first_image)

    # sample_image = channels_first_image / 255.
    # sample_image -= np.array([0.485, 0.456, 0.406])[:, None, None]
    # sample_image /= np.array([0.229, 0.224, 0.225])[:, None, None]

    # # print("\n\n\n\n\n")
    # # print(sample_image)
    # # print("\n\n\n\n\n")

    # # print(preprocessed_pil_image == sample_image)

    # # print(channels_first_pil_image.shape)
    # # print(sample_image.shape)

    # return sample_image
