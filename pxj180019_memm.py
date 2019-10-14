from nltk.corpus import brown
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
import numpy
import scipy
import sklearn
import copy
import re
tagset = list()
tag_dict = dict()
feature_dict = dict()
rare_words = set()
obj = LogisticRegression(class_weight='balanced', solver='saga', multi_class='multinomial')

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


def build_Y(tags):
    Y = list()
    for sen_tag in tags:
        for tag in sen_tag:
            index = tag_dict[tag]
            Y.append(index)
    return numpy.array(Y)


def build_X(features):
    examples_list, features_list, count = [], [], -1

    for m in range(len(features)):
        for n in range(len(features[m])):
            count = count + 1
            for i in range(len(features[m][n])):
                index = feature_dict.get(features[m][n][i], -1)
                if index != -1:
                    features_list.append(index)
                    examples_list.append(count)
    values = [1] * len(examples_list)

    data = numpy.array(values)
    row = numpy.array(examples_list)
    col = numpy.array(features_list)
    return csr_matrix((data, (row, col)), shape=(count + 1, len(feature_dict)))


def load_test(filename):
    result = list(list())
    with open(filename, "r") as f:
        for line in f:
            temp = line.strip().split(" ")
            result.append(temp)
    return result


def get_predictions(test_sentence, model):
    sentence = test_sentence
    n = len(sentence)
    T = len(tagset)
    Y_pred = numpy.empty([n - 1, T, T], dtype=float)

    for i in range(1, len(sentence)):
        for j in range(len(tagset)):
            test_feature = get_features(i, sentence, tagset[j], rare_words)
            X = build_X([[test_feature]])
            Y_pred[i-1][j] = model.predict_log_proba(X)
    test_feature = get_features(0, sentence, "<S>", rare_words)
    first_sen = build_X([[test_feature]])
    Y_start = model.predict_log_proba(first_sen)
    return Y_pred, Y_start


def viterbi(Y_start, Y_pred):
    n = len(Y_pred) + 1
    T = len(tagset)
    V = numpy.empty([n, T])
    BP = numpy.empty([n, T], dtype=int)
    T1 = numpy.empty(len(tagset))
    for j in range(len(tagset)):
        V[0, j] = Y_start[0, j]
        BP[0, j] = -1

    for i in range(len(Y_pred)):
        for k in range(len(tagset)):
            for j in range(len(tagset)):
                T1[j] = V[i, j] + Y_pred[i, j, k]
            V[i + 1, k] = numpy.amax(T1)
            BP[i + 1, k] = numpy.argmax(T1)

    backward_indices = []
    index = numpy.argmax(V[n - 1])
    backward_indices.append(index)
    while n != 1:
        index = BP[n - 1, index]
        backward_indices.append(index)
        n = n - 1
    backward_indices.reverse()
    inv_map = {v: k for k, v in tag_dict.items()}
    result = []
    for i in backward_indices:
        result.append(inv_map[i])
    return result


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


def main():
    global tagset, tag_dict, feature_dict, rare_words 
    brown_sentences = brown.tagged_sents(tagset='universal')
    print("Loaded the corpus.")

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

    # Generating features for all training sentences.

    training_features = list(list())
    for i in range(len(train_sentences)):
        temp_sen_features = list()
        for j in range(len(train_sentences[i])):
            temp_word_features = list()
            if j == 0:
                temp_word_features = temp_word_features + get_features(j, train_sentences[i], "<S>", rare_words)
            else:
                temp_word_features = temp_word_features + get_features(j, train_sentences[i], train_tags[i][j - 1],
                                                                       rare_words)
            temp_sen_features.append(temp_word_features)
        training_features.append(temp_sen_features)


    training_features, non_rare_feature_set = remove_rare_features(training_features, 5)

    # Creating feature dict from non-rare feature set.

    index = 0
    for feature in non_rare_feature_set:
        feature_dict[feature] = index
        index = index + 1

    tag_set = set()
    for sen_tags in train_tags:
        for tags in sen_tags:
            tag_set.add(tags)

    tagset = list(tag_set)

    index = 0
    for tag in tag_set:
        tag_dict[tag] = index
        index = index + 1

    # Training the model...
    print("Training the model.")
    X_train = build_X(training_features)
    Y_train = build_Y(train_tags)
    obj.fit(X_train, Y_train)

    # Testing the model...
    print("Testing the model.")
    test_data = load_test("test.txt")
    for line in test_data:
        if line is not None:
            Y, X = get_predictions(line, obj)
            res = viterbi(X, Y)
            print(res)


if __name__ == '__main__':
    main()