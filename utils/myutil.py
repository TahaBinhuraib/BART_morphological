import codecs
import re
from collections import defaultdict
from imp import get_tag
from random import choice, random

import numpy as np


def get_chars(l):
    flat_list = [char for word in l for char in word]
    return list(set(flat_list))


def get_tags(l):
    flat_list = [tag for sublist in l for tag in sublist]
    return list(set(flat_list))


def read_data(filename):
    with codecs.open(filename, "r", "utf-8") as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    tags = []
    for l in lines:
        l = l.strip().split("\t")
        if l:
            inputs.append(l[0])
            outputs.append(l[1])
            tags.append(" ".join(l[2].split(";")))

    return inputs, outputs, tags


def _read_data(filename):
    with codecs.open(filename, "r", "utf-8") as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    tags = []
    for l in lines:
        l = l.strip().split("\t")
        if l:
            inputs.append(l[0])
            outputs.append(l[1])
            tags.append(re.split("\W+", l[2]))

    return inputs, outputs, tags


def read_data_for_tokenizer(filename):
    with codecs.open(filename, "r", "utf-8") as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    for l in lines:
        l = l.strip().split("\t")
        if l:
            inputs.append(l[0])
            outputs.append(l[1])

    return inputs, outputs


def read_test_data(filename):
    with codecs.open(filename, "r", "utf-8") as inp:
        lines = inp.readlines()
    inputs = []
    tags = []
    for l in lines:
        l = l.strip().split("\t")
        if l:
            inputs.append(list(l[0]))
            tags.append(re.split("\W+", l[1]))
    return inputs, tags


def get_char_acc(lstr1, lstr2):
    true_char = 0
    len_ref = len(lstr1)
    len_pred_here = len(lstr2)

    N = max(len_ref, len_pred_here)

    for y, y_hat in zip(lstr1, lstr1):
        if y == y_hat:
            true_char += 1

    acc_here = true_char / N
    return acc_here


def make_data_for_tokenizer(train_path, dev_path, language):
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

    train_input, train_output = read_data_for_tokenizer(train_path)
    validate_input, validate_output = read_data_for_tokenizer(dev_path)
    raw_data = train_input + train_output + validate_input + validate_output

    with open(f"generated_data/{language}_data.txt", "w") as f:
        for item in raw_data:
            f.write("%s\n" % item)

    _, _, train_tags = _read_data(train_path)
    _, _, validate_tags = _read_data(dev_path)
    tags = get_tags(train_tags + validate_tags)
    for tag in tags:
        special_tokens.append("tag")

    return f"generated_data/{language}_data.txt", special_tokens


def generate_data(
    train_input,
    train_tags,
    validate_input,
    validate_tags,
    train_output,
    validate_output,
):
    """Generate the data needed for the low_resource language task"""
    x_train = []
    for input, tag in zip(train_input, train_tags):
        x_train.append(input + " " + tag)

    x_validation = []
    for input, tag in zip(validate_input, validate_tags):
        x_validation.append(input + " " + tag)

    train = []
    for x, y in zip(x_train, train_output):
        train.append((x, y))

    test = []
    for x, y in zip(x_validation, validate_output):
        test.append((x, y))

    max_x = len(max(x_train, key=len))
    max_y = len(max(train_output, key=len))

    return train, test, max_x, max_y
