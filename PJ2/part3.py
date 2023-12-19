import pickle
import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

torch.manual_seed(76)

batch_size = 32
embedding_dim = 128
hidden_size = 512
device = 'cpu'

epochs = 50
lr = 0.001
weight_decay = 1e-4

language = "Chinese"
train_file = "./NER/Chinese/train.txt" if language == "Chinese" else "./NER/English/train.txt"
model_file = "./ckpts/Chinese_BiLSTM_CRF_Model.pkl" if language == "Chinese" else "./ckpts/English_BiLSTM_CRF_Model.pkl"
test_file = "./NER/Chinese/test.txt" if language == "Chinese" else "./NER/English/test.txt"
out_file = "./NER/Chinese/Part3_my_result.txt" if language == "Chinese" else "./NER/English/Part3_my_result.txt"

eng_tag_dict = {
    'O': 0, 
    'B-PER': 1, 'I-PER': 2, 
    'B-ORG': 3, 'I-ORG': 4, 
    'B-LOC': 5, 'I-LOC': 6, 
    'B-MISC': 7, 'I-MISC': 8
}

ch_tag_dict = {
    'O': 0,
    'B-NAME': 1, 'M-NAME': 2, 'E-NAME': 3, 'S-NAME': 4,
    'B-CONT': 5, 'M-CONT': 6, 'E-CONT': 7, 'S-CONT': 8,
    'B-EDU': 9, 'M-EDU': 10, 'E-EDU': 11, 'S-EDU': 12,
    'B-TITLE': 13, 'M-TITLE': 14, 'E-TITLE': 15, 'S-TITLE': 16,
    'B-ORG': 17, 'M-ORG': 18, 'E-ORG': 19, 'S-ORG': 20,
    'B-RACE': 21, 'M-RACE': 22, 'E-RACE': 23, 'S-RACE': 24,
    'B-PRO': 25, 'M-PRO': 26, 'E-PRO': 27, 'S-PRO': 28,
    'B-LOC': 29, 'M-LOC': 30, 'E-LOC': 31, 'S-LOC': 32
}

word_dict = {}
tag_dict = ch_tag_dict if language == "Chinese" else eng_tag_dict
tag_dict_d = {v: k for k, v in tag_dict.items()}

def read_file(file):
    wordlist, taglist = [], []
    sentences = []
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    for line in data:
        if line == '\n':
            sentences.append([wordlist, taglist])
            wordlist, taglist = [], []
        else:
            word, tag = line.strip().split()
            wordlist.append(word)
            taglist.append(tag)
    if len(wordlist) != 0:
        sentences.append([wordlist, taglist])
        wordlist, taglist = [], []
    return sentences

def prep_word_dict():
    word_dict['UNK'] = 0
    word_dict['PAD'] = 1
    with open(train_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    for line in data:
        if line == '\n': continue
        word, _ = line.strip().split()
        if word not in word_dict:
            word_dict[word] = len(word_dict)
    tag_dict["<START>"] = len(tag_dict)
    tag_dict["<STOP>"] = len(tag_dict)

def log_sum_exp(vec):
    max_score, _ = torch.max(vec, dim=-1)
    max_score_broadcast = max_score.unsqueeze(-1).repeat_interleave(vec.shape[-1], dim=-1)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1))

class BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_size, words_size, tag_dict, device='cpu'):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  
        self.hidden_size = hidden_size  
        self.vocab_size = words_size  
        self.tagset_size = len(tag_dict)  
        self.device = device
        self.state = 'train' 

        # 嵌入层
        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_size // 2, num_layers=2, bidirectional=True, batch_first=True)
        # 全连接层
        self.hidden2tag = nn.Linear(hidden_size, self.tagset_size, bias=True)
        # CRF层
        self.crf = CRF(tag_dict, device)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, sentence, seq_len, tags=''):
        embeds = self.word_embeds(sentence)
        self.dropout(embeds)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, seq_len, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        seq_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        seqence_output = self.layer_norm(seq_unpacked)
        feats = self.hidden2tag(seqence_output)
        
        if self.state == 'train':
            loss = self.crf.calc_loss(feats, tags, seq_len)
            return loss
        elif self.state == 'eval':
            all_tag = []
            for i, feat in enumerate(feats):
                all_tag.append(self.crf.viterbi(feat[:seq_len[i]])[1])
            return all_tag
        else:
            return self.crf.viterbi(feats[0])[1]

class CRF(nn.Module):
    def __init__(self, label_map, device='cpu'):
        super(CRF, self).__init__()
        self.label_map = label_map
        self.label_map_inv = {v: k for k, v in label_map.items()}
        self.tagset_size = len(self.label_map)
        self.device = device

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.transitions.data[self.label_map[self.START_TAG], :] = -10000
        self.transitions.data[:, self.label_map[self.STOP_TAG]] = -10000

    def _all_path(self, feats, seq_len):
        init_alphas = torch.full((self.tagset_size,), -10000.)
        init_alphas[self.label_map[self.START_TAG]] = 0.

        forward_var = torch.zeros(feats.shape[0], feats.shape[1] + 1, feats.shape[2], dtype=torch.float32, device=self.device)
        # 初始化前向矩阵
        forward_var[:, 0, :] = init_alphas
        transitions = self.transitions.unsqueeze(0).repeat(feats.shape[0], 1, 1)
        for t in range(feats.shape[1]):
            emit_score = feats[:, t, :]
            # 迭代: 前向变量，转移矩阵，发射矩阵
            tag_var = forward_var[:, t, :].unsqueeze(1).repeat(1, feats.shape[2], 1)  + transitions + emit_score.unsqueeze(2).repeat(1, 1, feats.shape[2])
            forward_var[:, t + 1, :] = log_sum_exp(tag_var)
        forward_var = forward_var[range(feats.shape[0]), seq_len, :]
        terminal_var = forward_var + self.transitions[self.label_map[self.STOP_TAG]].unsqueeze(0).repeat(feats.shape[0], 1)
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _real_path(self, feats, tags, seq_len):
        score = torch.zeros(feats.shape[0], device=self.device)
        # eg: label_map->2; [2]; [[2]]; [[2], [2], [2]] 
        start = torch.tensor([self.label_map[self.START_TAG]], device=self.device).unsqueeze(0).repeat(feats.shape[0], 1)
        tags = torch.cat([start, tags], dim=1)
        for batch_i in range(feats.shape[0]):
            score[batch_i] = torch.sum(self.transitions[tags[batch_i, 1:seq_len[batch_i] + 1], tags[batch_i, :seq_len[batch_i]]]) + torch.sum(feats[batch_i, range(seq_len[batch_i]), tags[batch_i][1:seq_len[batch_i] + 1]])
            score[batch_i] += self.transitions[self.label_map[self.STOP_TAG], tags[batch_i][seq_len[batch_i]]]
        return score
    
    def calc_loss(self, feats, tags, seq_len):
        all_score = self._all_path(feats, seq_len)
        real_score = self._real_path(feats, tags, seq_len)
        return torch.mean(all_score - real_score)

    def viterbi(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_vvars[0][self.label_map[self.START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            forward_var = forward_var.repeat(feat.shape[0], 1)
            next_tag_var = forward_var + self.transitions
            bptrs_t = torch.max(next_tag_var, 1)[1].tolist()
            viterbivars_t = next_tag_var[range(forward_var.shape[0]), bptrs_t]
            forward_var = (viterbivars_t + feat).view(1, -1)
            # 回溯
            backpointers.append(bptrs_t)
        terminal_var = forward_var + self.transitions[self.label_map[self.STOP_TAG]]
        best_tag_id = torch.max(terminal_var, 1)[1].item()
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.label_map[self.START_TAG] 
        best_path.reverse()
        return path_score, best_path
    
class NERdataset(Dataset):
    def __init__(self, file, word_dict, tag_dict):
        self.sentences = read_file(file)
        self.word_dict = word_dict
        self.tag_dict = tag_dict

    def __getitem__(self, index):
        wordlist = self.sentences[index][0]
        taglist = self.sentences[index][1]
        wordlist_idx = [self.word_dict.get(word, 0) for word in wordlist]
        taglist_idx = [self.tag_dict[tag] for tag in taglist]
        return [wordlist_idx, taglist_idx]

    def __len__(self):
        return len(self.sentences)

    def collate_fn(self, batch):
        wordlists = [wordlist for wordlist, _ in batch]
        taglists = [taglist for _, taglist in batch]
        wordlists_len = [len(wordlist) for wordlist in wordlists]
        max_len = max(wordlists_len)
        # 填充成相同长度
        wordlists = [wordlist + [self.word_dict['PAD']] * (max_len - len(wordlist)) for wordlist in wordlists]
        taglists = [taglist + [self.tag_dict['O']] * (max_len - len(taglist)) for taglist in taglists]
        
        wordlists = torch.tensor(wordlists, dtype=torch.long)
        taglists = torch.tensor(taglists, dtype=torch.long)
        wordlists_len = torch.tensor(wordlists_len, dtype=torch.long)

        return wordlists, taglists, wordlists_len

def train():
    train_dataset = NERdataset(train_file, word_dict, tag_dict)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        pin_memory=True, 
        shuffle=True, 
        collate_fn=train_dataset.collate_fn)
    model = BiLSTM_CRF(embedding_dim, hidden_size, len(train_dataset.word_dict), train_dataset.tag_dict, device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        model.state = 'train'
        for step, (wordlists, taglists, max_len) in enumerate(train_dataloader, start=1):
            wordlists = wordlists.to(device)
            taglists = taglists.to(device)
            max_len = max_len.to(device)

            loss = model(wordlists, max_len, taglists)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch: [{epoch + 1}/{epochs}],'
                  f'  step: {step / len(train_dataloader) * 100:2.2f}%,'
                  f'  loss: {loss.item():2.4f},')
        with open(model_file, 'wb') as file:
            pickle.dump(model, file)

def test():
    test_dataset = NERdataset(test_file, word_dict, tag_dict)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        pin_memory=False, 
        shuffle=False, 
        collate_fn=test_dataset.collate_fn)
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    pred = []
    model.eval()
    model.state = 'eval'
    with torch.no_grad():
        for wordlists, taglists, max_len in tqdm(test_dataloader, desc='eval: '):
            wordlists = wordlists.to(device)
            max_len = max_len.to(device)
            idxlist_batch = model(wordlists, max_len, taglists)
            taglist_batch = []
            for idxlist in idxlist_batch:
                taglist_batch.append([tag_dict_d[idx] for idx in idxlist])
            pred.extend(taglist_batch)
    with open(out_file, 'w') as f:
        for sentence, taglist in zip(test_dataset.sentences, pred):
            for word, tag in zip(sentence[0], taglist):
                f.write(f"{word} {tag}\n")
            f.write("\n")

if __name__ == '__main__':
    prep_word_dict()
    # 训练
    # train()
    
    # 测试
    test()
   
