import heapq
import pickle

# from bitarray import bitarray
from collections import Counter

import bitarray
import numpy as np
from PIL import Image


class HuffmanNode:
    def __init__(self, value=None, freq=None, left=None, right=None):
        self.value = value
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

    def is_leaf(self):
        return not self.left and not self.right


def get_image_pixels(image_path):
    img = Image.open(image_path)
    pixels = np.array(img)
    return pixels


def build_huffman_tree(frequencies):
    heap = [
        HuffmanNode(value=pixel, freq=freq)
        for pixel, freq in enumerate(frequencies)
        if freq > 0
    ]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged_node = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged_node)

    return heap[0]


def build_huffman_codes(node, current_code="", codes=None):
    if codes is None:
        codes = {}

    if node is not None:
        if node.value is not None:
            codes[node.value] = current_code
        build_huffman_codes(node.left, current_code + "0", codes)
        build_huffman_codes(node.right, current_code + "1", codes)

    return codes


def compress_image(image_path, output_path):
    global pixels
    pixels = get_image_pixels(image_path)
    frequencies = np.bincount(pixels.flatten())
    huffman_tree = build_huffman_tree(frequencies)
    huffman_codes = build_huffman_codes(huffman_tree)
    compressed_data = "".join([huffman_codes[pixel] for pixel in pixels.flatten()])

    # salvar
    with open(output_path, "wb") as file:
        pickle.dump(huffman_codes, file)
        compressed_bits = bitarray.bitarray(compressed_data)
        compressed_bits.tofile(file)
    print("Compressed data saved to:", output_path)


def build_huffman_tree_from_codes(huffman_codes):
    root = HuffmanNode()
    for value, code in huffman_codes.items():
        node = root
        for bit in code:
            if bit == "0":
                if not node.left:
                    node.left = HuffmanNode()
                node = node.left
            elif bit == "1":
                if not node.right:
                    node.right = HuffmanNode()
                node = node.right
        node.value = value
    return root


def decompress_huffman_data(compressed_bits, huffman_tree):
    current_node = huffman_tree
    decompressed_data = []

    for bit in compressed_bits:
        if bit:
            current_node = current_node.right
        else:
            current_node = current_node.left

        if current_node.is_leaf():
            decompressed_data.append(current_node.value)
            current_node = huffman_tree

    return np.array(decompressed_data, dtype=np.uint8)


def decompress_image(compressed_path, output_path):
    with open(compressed_path, "rb") as file:
        huffman_codes = pickle.load(file)
        compressed_bits = bitarray.bitarray()
        compressed_bits.fromfile(file)

    huffman_tree = build_huffman_tree_from_codes(huffman_codes)
    decompressed_data = decompress_huffman_data(compressed_bits, huffman_tree)
    decompressed_data = np.reshape(decompressed_data, pixels.shape)

    # Salvar
    decompressed_img = Image.fromarray(decompressed_data.astype(np.uint8))
    decompressed_img.save(output_path)
    print("Decompressed data saved to:", output_path)
