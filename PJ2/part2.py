import numpy as np
import sklearn_crfsuite

train_file = "./NER/Chinese/train.txt"
test_file = "./NER/Chinese/test.txt"
out_file = "./NER/Chinese/Part2_my_result.txt"
language = "Chinese"

eng_tags = [
    "O", 
    "B-PER", "I-PER", 
    "B-ORG", "I-ORG", 
    "B-LOC", "I-LOC", 
    "B-MISC", "I-MISC"
]

ch_tags = [
    'O',
    'B-NAME', 'M-NAME', 'E-NAME', 'S-NAME', 
    'B-CONT', 'M-CONT', 'E-CONT', 'S-CONT',
    'B-EDU', 'M-EDU', 'E-EDU', 'S-EDU',
    'B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE',
    'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG',
    'B-RACE', 'M-RACE', 'E-RACE', 'S-RACE',
    'B-PRO', 'M-PRO', 'E-PRO', 'S-PRO',
    'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC'
]

def init():
    wordlist, taglist = [], []
    wordlists_train, taglists_train = [], []
    wordlists_test = []
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = f.readlines()
    for line in train_data:
        if line == '\n':
            wordlists_train.append(wordlist)
            taglists_train.append(taglist)
            wordlist, taglist = [], []
        else:
            word, tag = line.strip().split()
            wordlist.append(word)
            taglist.append(tag)
    if len(wordlist) != 0:
        wordlists_train.append(wordlist)
        taglists_train.append(taglist) 
        wordlist, taglist = [], []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = f.readlines()
    for line in test_data:
        if line == '\n':
            wordlists_test.append(wordlist)
            wordlist = []
        else:
            word, _ = line.strip().split()
            wordlist.append(word)
    if len(wordlist) != 0:
        wordlists_test.append(wordlist)
        wordlist = []
    return wordlists_train, taglists_train, wordlists_test

def word2features(sent, i):
    cur = sent[i]
    if i == 0:
        pre = "<b>"
    else:
        pre = sent[i - 1]
    
    if i == len(sent) - 1:
        nxt = "<e>"
    else:
        nxt = sent[i + 1]
    
    features = {
        'w': cur,
        'w-1': pre,
        'w+1': nxt,
        'w-1:w': pre + cur,
        'w:w+1': cur + nxt,
        'bias': 1
    }
    if language == "English":
        features.update({'is_upper': cur[0].isupper()})
    
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

class CRF:
    def __init__(self):
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
            verbose=False
        )
    
    def train(self, wordlists, taglists):
        features = [sent2features(s) for s in wordlists]
        self.model.fit(features, taglists)

    def test(self, wordlists):
        features = [sent2features(s) for s in wordlists]
        pred = self.model.predict(features)
        f = open(out_file, 'w')
        for wordlist, taglist in zip(wordlists, pred):
            for i in range(len(wordlist)):
                f.write(f'{wordlist[i]} {taglist[i]}\n')
            f.write('\n')
        f.close()

if __name__ == '__main__':
    wordlists_train, taglists_train, wordlists_test = init()
    crf = CRF()
    print("training")
    crf.train(wordlists_train, taglists_train)
    print("testing")
    crf.test(wordlists_test)