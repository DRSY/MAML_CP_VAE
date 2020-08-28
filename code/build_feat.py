import json
import numpy as np
import argparse
import os
import pandas as pd
import config


def get_glove_embeds(in_file, out_file):
    glove_file = "data/glove.840B.300d.txt"
    word_vec = {}
    with open(glove_file) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            word_vec[word] = np.fromstring(vec, sep=' ')

    embeds = []
    with open(in_file) as f:
        for line in f:
            tokens = line.strip().split()
            vec = np.zeros(300, dtype=np.float32)
            sent_len = 0
            for token in tokens:
                if token in word_vec:
                    vec += word_vec[token]
                    sent_len += 1
            if sent_len > 0:
                vec = np.true_divide(vec, sent_len)
            vec = vec.reshape(1, 300)
            embeds.append(vec)
        embeds = np.concatenate(embeds)
        np.save(out_file, embeds)


def main():
    for s in [1, 2]:
        for t in range(1, 8):
            for label in ["train", "val"]:
                in_file = "./data/s{}/{}/t{}.all".format(s, label, t)
                out_file = "./data/s{}/{}/t{}_glove.npy".format(s, label, t)
                if not os.path.exists(out_file):
                    get_glove_embeds(in_file=in_file, out_file=out_file)
                    print(f"s{s}, t{t}, {label} done")


if __name__ == "__main__":
    main()
