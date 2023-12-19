import numpy as np

train_file = "./NER/Chinese/train.txt"
test_file = "./NER/Chinese/test.txt"
out_file = "./NER/Chinese/Part1_my_result.txt"
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
    word_set = set()
    wordlists_train, taglists_train = [], []
    wordlists_test = []
    words = []
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
            word_set.add(word)
    if len(wordlist) != 0:
        wordlists_train.append(wordlist)
        taglists_train.append(taglist) 
        wordlist, taglist = [], []
    
    words = list(word_set)
    
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
    return wordlists_train, taglists_train, words, wordlists_test


class HMM:
    def __init__(self, words, tags):
        self.words = words
        self.tags = tags
        self.M = len(words)
        self.N = len(tags)
        
        self.init_prob = np.zeros(len(tags))
        self.transition_prob = np.zeros((len(tags), len(tags)))
        self.emit_prob = np.zeros((len(tags), len(words)))
    
    def train(self, wordlists, taglists):
        for wordlist, taglist in zip(wordlists, taglists):
            self.init_prob[self.tags.index(taglist[0])] += 1
            for j in range(len(wordlist)):
                word, tag = wordlist[j], taglist[j]
                self.emit_prob[self.tags.index(tag)][self.words.index(word)] += 1
                if j < len(wordlist) - 1:
                    trans_tag = taglist[j + 1]
                    self.transition_prob[self.tags.index(tag)][self.tags.index(trans_tag)] += 1
        
        # self.init_prob = (self.init_prob) / (self.init_prob.sum())
        # for i, v in enumerate(np.sum(self.transition_prob, axis=1, keepdims=True)):
        #     if v == 0: continue
        #     self.transition_prob[i, :] = self.transition_prob[i, :] / v
        # for i, v in enumerate(np.sum(self.emit_prob, axis=1, keepdims=True)):
        #     if v == 0: continue
        #     self.emit_prob[i, :] = self.emit_prob[i, :] / v
        # 拉普拉斯平滑
        self.init_prob = (self.init_prob + 1) / (self.init_prob.sum() + self.N)
        self.transition_prob = (self.transition_prob + 1) / (np.sum(self.transition_prob, axis=1, keepdims=True) + self.N)
        self.emit_prob = (self.emit_prob + 1) / (np.sum(self.emit_prob, axis=1, keepdims=True) + self.M)
        

    def viterbi(self, wordlist):
        transition_prob = np.log(self.transition_prob)
        emit_prob = np.log(self.emit_prob)
        init_prob = np.log(self.init_prob)
        dp = np.zeros((self.N, len(wordlist)))
        pre = np.zeros((self.N, len(wordlist)))
        
        if wordlist[0] in self.words:
            dp[:, 0] = init_prob + emit_prob[:, self.words.index(wordlist[0])]
        else:
            dp[:, 0] = init_prob + np.log(np.ones(self.N) / self.N)
        
        for t in range(1, len(wordlist)):
            if wordlist[t] in self.words:
                emit = emit_prob[:, self.words.index(wordlist[t])]
            else:
                emit = np.log(np.ones(self.N) / self.N)
            for i in range(self.N):
                p = dp[:, t - 1] + transition_prob[:, i]
                dp[i, t] = np.max(p) + emit[i]
                pre[i, t] = np.argmax(p)
        path = np.zeros((len(wordlist)))
        path[-1] = np.argmax(dp[:, -1])
        for t in range(len(wordlist) - 1, 0, -1):
            path[t - 1] = pre[int(path[t]), t]
        return [self.tags[int(index)] for index in path]
    
    def test(self, wordlists):
        f = open(out_file, 'w')
        for wordlist in wordlists:
            taglist = self.viterbi(wordlist)
            for i in range(len(wordlist)):
                f.write(f'{wordlist[i]} {taglist[i]}\n')
            f.write('\n')
        f.close()
        

if __name__ == '__main__':
    wordlists_train, taglists_train, words, wordlists_test = init()
    if (language == "Chinese"):
        tags = ch_tags
    else:
        tags = eng_tags
    hmm = HMM(words, tags)
    print("training")
    hmm.train(wordlists_train, taglists_train)
    print("testing")
    hmm.test(wordlists_test)
    
    