import numpy as np

def generate_prediction(
        model,
        test_image,
        classes,
):
        result = model.predict(
            np.array([test_image]),
            batch_size=1,
        )

        winner_index = np.argmax(result)

        winner = classes[winner_index]

        return winner
