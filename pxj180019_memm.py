from nltk.corpus import brown
import numpy
import scipy
import sklearn
import copy
import re


def word_ngram_features(i, words):
    words_copy = copy.copy(words)
    result = list()
    if i < 0 or i >= len(words_copy):
        return result

    words_copy.insert(0, "<s>")
    words_copy.insert(1, "<s>")
    words_copy.append("</s>")
    words_copy.append("</s>")
    index = i + 2
    prevbigram = "prevbigram-[" + words_copy[index - 1] + "]"
    nextbigram = "nextbigram-[" + words_copy[index + 1] + "]"
    prevskip = "prevskip-[" + words_copy[index - 2] + "]"
    nextskip = "nextskip-[" + words_copy[index + 2] + "]"
    prevtrigram = "prevtrigram-[" + words_copy[index - 1] + "]-[" + words_copy[index - 2] + "]"
    nexttrigram = "nexttrigram-[" + words_copy[index + 1] + "]-[" + words_copy[index + 2] + "]"
    centertrigram = "centertrigram-[" + words_copy[index - 1] + "]-[" + words_copy[index + 1] + "]"
    result.extend([prevbigram, nextbigram, prevskip, nextskip, prevtrigram, nexttrigram, centertrigram])

    return result


def word_features(word, rare_words):
    result = list()
    if word not in rare_words:
        result.append("word-[" + word + "]")
    if word.isupper():
        result.append("capital")
    if re.search(r"\d+", word):
        result.append("number")
    if re.search(r"-", word):
        result.append("hyphen")

    prefix_feature = list()
    suffix_feature = list()

    for j in range(4):
        if j > len(word) - 1:
            break
        else:
            prefix_feature.append("prefix[" + str(j + 1) + "]-[" + word[:(j+1)] + "]")
            suffix_feature.append("suffix[" + str(j + 1) + "]-[" + word[-(j+1):] + "]")

    if prefix_feature is not [] and suffix_feature is not []:
        result = result + prefix_feature
        result = result + suffix_feature

    return result


def get_features(i, words, prevtag, rare_words):
    result = word_ngram_features(i, words) + word_features(words[i], rare_words)
    result.append("tagbigram-[" + prevtag + "]")
    lower_result = list()
    for feature in result:
        lower_result.append(feature.lower())
    return lower_result


def build_Y(tags, tag_dict=dict()):
    Y = list()
    for sen_tag in tags:
        for tag in sen_tag:
            index = tag_dict[tag]
            Y.append(index)

    return numpy.array(Y)



def main():
    brown_sentences = brown.tagged_sents(tagset='universal')
    print(brown_sentences)
    print(len(brown_sentences))


    # Initializing training sentences and training tags.

    train_sentences = list(list())
    train_tags = list(list())
    words_count = dict()
    rare_words = set()
    for sen_tag_list in brown_sentences:
        temp_word = list()
        temp_tag = list()
        for word, tag in sen_tag_list:
            temp_word.append(word)
            temp_tag.append(tag)
            count = words_count.get(word, 0)
            words_count[word] = count + 1
        train_sentences.append(temp_word)
        train_tags.append(temp_tag)
    for k, v in words_count.items():
        if v < 5:
            rare_words.add(k)
    # print(train_sentences)
    # print(train_tags)
    # print(rare_words)
    word_ngram_features(train_sentences[0].__len__() - 1, train_sentences[0])
    # print(str(word_features("pankaj", rare_words)))
    # print(str(word_features("PANKAJ", rare_words)))
    # print(str(word_features("pank123", rare_words)))
    # print(str(word_features("pan-", rare_words)))
    # print(str(word_features("pan", rare_words)))
    # print(str(get_features(4, train_sentences[0], "AA", rare_words)))

    # Generating features for all training sentences.

    training_features = list(list())
    for i in range(len(train_sentences)):
        temp_sen_features = list()
        for j in range(len(train_sentences[i])):
            temp_word_features = list()
            if j == 0:
                temp_word_features = temp_word_features + get_features(j, train_sentences[i], "<S>", rare_words)
            else:
                temp_word_features = temp_word_features + get_features(j, train_sentences[i], train_tags[i][j - 1], rare_words)
            temp_sen_features.append(temp_word_features)
        training_features.append(temp_sen_features)

    print(training_features[0][0][0])
    print(training_features[0][0][1])
    print(training_features[0][0][2])

    with open("original_feature.txt", "w+") as f:
        f.write(str(training_features))

    training_features, non_rare_feature_set = remove_rare_features(training_features, 5)

    with open("updated_feature.txt", "w+") as f:
        f.write(str(training_features))

    with open("non_rare_feature.txt", "w+") as f:
        f.write(str(non_rare_feature_set))

    # Creating feature dict from non-rare feature set.

    feature_dict = dict()
    tag_dict = dict()

    index = 0
    for feature in non_rare_feature_set:
        feature_dict[feature] = index
        index = index + 1

    tag_set = set()

    for sen_tags in train_tags:
        for tags in sen_tags:
            tag_set.add(tags)

    print(set(tag_set))
    index = 0
    for tag in tag_set:
        tag_dict[tag] = index
        index = index + 1

    build_Y(train_tags, tag_dict)

def remove_rare_features(features, n):
    feature_count = dict()
    for sen_features in features:
        for word_feature in sen_features:
            for feature in word_feature:
                count = feature_count.get(feature, 0)
                feature_count[feature] = count + 1
    # print(feature_count)

    rare_feature_set = set()
    non_rare_feature_set = set()

    for name, count in feature_count.items():
        if count < n:
            rare_feature_set.add(name)
        else:
            non_rare_feature_set.add(name)

    updated_features = list(list())
    for sen_features in features:
        updated_sen_feature = list()
        for word_feature in sen_features:
            non_rare_feature = [feature for feature in word_feature if feature not in rare_feature_set]
            updated_sen_feature.append(non_rare_feature)
        updated_features.append(updated_sen_feature)

    # print("Updated feature: " + str(updated_features))
    # print("Non rare: " + str(non_rare_feature_set))
    return updated_features, non_rare_feature_set


if __name__ == '__main__':
    main()