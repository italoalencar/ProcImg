import numpy as np


def rle_encode(data):
    encoded_data = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded_data.append((data[i - 1], count))
            count = 1
    encoded_data.append((data[-1], count))
    return encoded_data


def rle_decode(encoded_data):
    decoded_data = []
    for value, count in encoded_data:
        decoded_data.extend([value] * count)
    return decoded_data


def encode_image(image_array):
    pixels = image_array.flatten()
    encoded_data = rle_encode(pixels)
    return encoded_data


def decode_image(encoded_data, shape):
    decoded_data = rle_decode(encoded_data)
    image_array = np.array(decoded_data).reshape(shape)
    return image_array
