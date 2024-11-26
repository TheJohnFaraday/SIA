import numpy as np

from dataset.font_data import Font3


def get_letters_labels():
    return [
        '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
        'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
        't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'DEL'
    ]

def convert_fonts_to_binary_matrix(font_array):
    binary_matrix = []
    for character in font_array:
        char_matrix = []
        for hex_value in character:
            binary_value = bin(hex_value)[2:].zfill(8)[-5:]
            # Reemplazo 0 por -1
            char_matrix.append([1 if bit == '1' else -1 for bit in binary_value])
        binary_matrix.append(char_matrix)
    return np.array(binary_matrix)

def get_letters():
    # Convert font data to binary matrix and reshape
    binary_matrix = convert_fonts_to_binary_matrix(Font3)
    binary_matrix = np.reshape(
        binary_matrix, (32, 35, 1)
    )  # Reshape input to (32, 35, 1) for compatibility

    return binary_matrix
