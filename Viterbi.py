import collections
import math
import numpy as np

def BuildMap(sentence, emission, transition):
    tokens = sentence.split("\n")
    pre_tag = "start"

    for i in range(tokens.__len__()):
        if tokens[i] != "":
            word = tokens[i].split("\t")[0]
            tag = tokens[i].split("\t")[1]
            if word in words_count:
                words_count[word] += 1
            else:
                words_count[word] = 1
            if tag in tags_count:
                tags_count[tag] += 1
            else:
                tags_count[tag] = 1

            if tag in emission:
                if word in emission[tag]:
                    count = emission[tag][word] + 1
                    emission[tag][word] = count
                else:
                    emission[tag][word] = 1.0
            else:
                emission[tag] = dict()
                emission[tag][word] = 1.0

            if i < (tokens.__len__()-1):
                if i == 0:
                    if tag in transition["start"]:
                        count = transition["start"][tag] + 1
                        transition["start"][tag] = count
                    else:
                        transition["start"][tag] = 1.0

            if tokens[i + 1] != "":
                if tag in transition:
                    if tokens[i + 1].split("\t")[1] in transition[tag]:
                        count = transition[tag][tokens[i + 1].split("\t")[1]] + 1
                        transition[tag][tokens[i + 1].split("\t")[1]] = count
                    else:
                        transition[tag][tokens[i + 1].split("\t")[1]] = 1.0
                else:
                    transition[tag] = dict()
                    transition[tag][tokens[i + 1].split("\t")[1]] = 1.0

def HMM(test_file):
    output = open("./WSJ_24.test", 'w')
    sentence = ""
    tagline = ""
    words = []
    tags = []

    for line in test_file:
        if line != "\n":
            sentence = sentence + line.strip() + " "
        else:
            sentence = sentence[:-1]
            tagline = Viterbi(sentence)
            words = sentence.split(" ")
            sentence=""
            tags = tagline.split(" ")
            for i in range(words.__len__()):
                if words[i] not in emission[tags[i]]:
                    if words[i].__len__() >= 3:
                        if i == 0:
                            tags[i] = UnknownWords("", words[i], tags[i])
                        else:
                            tags[i] = UnknownWords(words[i - 1], words[i], tags[i])

            for i in range(words.__len__()):
                output.write(words[i] + "\t" + tags[i] + "\n")
            output.write("\n")
    output.close()

def Viterbi(sentence):
    tagline = ""
    end_t = ""
    unknown = -10.0
    max_probability = -1000

    path = []
    backtrace = []
    preTag = set()
    prePro = collections.defaultdict(float)

    preTag.add("start")
    prePro["start"] = 0.0

    words = sentence.split(" ")

    for i in range(words.__len__()):
        next_tag = set()
        next_pro = collections.defaultdict(float)
        back_tags = collections.defaultdict(str)
        
        for pre_tag in preTag:
            if pre_tag in transition and transition[pre_tag] != {}:
                for tag in transition[pre_tag]:
                    next_tag.add(tag)
                    if tag in emission and words[i] in emission[tag]:
                        probability = prePro[pre_tag] + transition[pre_tag][tag] + emission[tag][words[i]]
                    else:
                        probability = prePro[pre_tag] + transition[pre_tag][tag] + unknown                

                    if tag not in next_pro or probability > next_pro[tag]:
                        next_pro[tag] = probability
                        back_tags[tag] = pre_tag                     
                        if backtrace.__len__() > i:
                            del backtrace[i]

                        backtrace.append(back_tags)
        prePro = next_pro
        preTag = next_tag

    for prob in prePro:
        if max_probability < prePro[prob]:
            max_probability = prePro[prob]
            end_t = prob

    path.append(end_t)

    for i in range(words.__len__() - 1, 0, -1):
        path.append(backtrace[i][path[len(path)-1]])   

    for j in range(path.__len__()):
        token = path.pop()
        if tagline == None:
            tagline = token + " "
        else:
            tagline = tagline + token + " "

    return tagline

def UnknownWords(pre, word, tag):
    length = word.__len__()

    if word[0].isupper():
        tag = "NNP"
    elif pre == "be":
        tag = "JJ"
    elif pre == "it":
        tag = "VBZ"
    elif pre == "would":
        tag = "VB"
    elif word[length-2: length] == "ss":
        tag = "NN"
    elif word[length-1: length] == "s":
        tag = "NNS"
    elif "-" in word:
        tag = "JJ"
    elif "." in word:
        tag = "CD"
    elif pre == "$":
        tag = "CD"
    elif length >= 3 and word[length-3: length] == "ble":
        tag ="JJ"
    elif length >= 3 and word[length-3: length] == "ive":
        tag = "JJ"
    elif word[length-2: length] == "us":
        tag = "JJ"
    else:
        tag = "NN"

    return tag

print ("start")

training_file = open("./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos", 'r')
test_file = open("./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words", 'r')

emission = dict()
transition = dict()
transition["start"] = dict()
words_count = collections.defaultdict(int)
tags_count = collections.defaultdict(int)
sentence = ""
for line in training_file:
    if line != "\n":
        sentence += line 
    else:
        BuildMap(sentence, emission, transition)
        sentence = ""

for a in transition:
    count = 0
    for b in transition[a]:
        count += transition[a][b]
    for b in transition[a]:
        logprob = math.log10(transition[a][b] / count)
        transition[a][b] = logprob

for a in emission:
    count = 0
    for b in emission[a]:
        count += emission[a][b]
    for b in emission[a]:
        logprob = math.log10(emission[a][b]/count)
        emission[a][b] = logprob

print ("going")
HMM(test_file)

print ("finish.")
