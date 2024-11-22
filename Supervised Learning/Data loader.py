import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# 工具函数
def sequence_padding(inputs, length=None, padding=0):
    """对序列进行补全（padding）"""
    if length is None:
        length = max(len(x) for x in inputs)
    return np.array([
        np.concatenate([x, [padding] * (length - len(x))]) if len(x) < length else x
        for x in inputs
    ])


def convert_index_to_text(index_list, entity_type):
    """将索引转换为实体文本"""
    return f"{'-'.join(map(str, index_list))}-{entity_type}"


def preprocess_data(sentence, entities, maxlen, tokenizer, categories, dis2idx):
    """
    预处理句子和实体信息，生成模型需要的特征。
    Args:
        sentence: 输入句子
        entities: 实体信息
        maxlen: 最大长度
        tokenizer: 分词器
        categories: 实体类别映射
        dis2idx: 距离索引映射
    Returns:
        tokens_ids, pieces2word, dist_inputs, grid_labels, grid_mask2d, entity_text
    """
    tokens = [tokenizer.tokenize(word)[1:-1] for word in sentence[:maxlen - 2]]
    pieces = [piece for pieces in tokens for piece in pieces]
    tokens_ids = [tokenizer._token_start_id] + tokenizer.tokens_to_ids(pieces) + [tokenizer._token_end_id]

    length = len(tokens)
    _pieces2word = np.zeros((length, len(tokens_ids)), dtype=np.bool_)
    e_start = 0
    for i, pieces in enumerate(tokens):
        if len(pieces) == 0:
            continue
        pieces = list(range(e_start, e_start + len(pieces)))
        _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
        e_start += len(pieces)

    _dist_inputs = np.zeros((length, length), dtype=np.int64)
    for k in range(length):
        _dist_inputs[k, :] += k
        _dist_inputs[:, k] -= k
    for i in range(length):
        for j in range(length):
            if _dist_inputs[i, j] < 0:
                _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
            else:
                _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
    _dist_inputs[_dist_inputs == 0] = 19

    _grid_labels = np.zeros((length, length), dtype=np.int64)
    _grid_mask2d = np.ones((length, length), dtype=np.bool_)
    for entity in entities:
        e_start, e_end, e_type = entity[0], entity[1] + 1, entity[-1]
        if e_end >= maxlen - 2:
            continue
        index = list(range(e_start, e_end))
        for i in range(len(index)):
            if i + 1 >= len(index):
                break
            _grid_labels[index[i], index[i + 1]] = 1
        _grid_labels[index[-1], index[0]] = categories[e_type]
    _entity_text = set([convert_index_to_text(list(range(e[0], e[1] + 1)), categories[e[-1]]) for e in entities])

    return tokens_ids, _pieces2word, _dist_inputs, _grid_labels, _grid_mask2d, _entity_text


# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, filename):
        self.data = self.load_data(filename)

    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for l in tqdm(f.split('\n\n'), desc='Load data'):
                if not l:
                    continue
                sentence, entities = [], []
                for i, c in enumerate(l.split('\n')):
                    char, flag = c.split(' ')
                    sentence += char
                    if flag[0] == 'B':
                        entities.append([i, i, flag[2:]])
                    elif flag[0] == 'I':
                        entities[-1][1] = i
                if len(sentence) > maxlen - 2:
                    continue
                features = preprocess_data(
                    sentence=sentence,
                    entities=entities,
                    maxlen=maxlen,
                    tokenizer=tokenizer,
                    categories=categories,
                    dis2idx=dis2idx
                )
                D.append(features)
        return D

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 定义collate_fn函数
def collate_fn(data):
    tokens_ids, pieces2word, dist_inputs, grid_labels, grid_mask2d, _entity_text = map(list, zip(*data))

    sent_length = torch.tensor([i.shape[0] for i in pieces2word], dtype=torch.long, device=device)
    max_wordlen = torch.max(sent_length).item()
    max_tokenlen = np.max([len(x) for x in tokens_ids])
    tokens_ids = torch.tensor(sequence_padding(tokens_ids), dtype=torch.long, device=device)
    batch_size = tokens_ids.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = torch.tensor(x, dtype=torch.long, device=device)
        return new_data

    dis_mat = torch.zeros((batch_size, max_wordlen, max_wordlen), dtype=torch.long, device=device)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_wordlen, max_wordlen), dtype=torch.long, device=device)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_wordlen, max_wordlen), dtype=torch.bool, device=device)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_wordlen, max_tokenlen), dtype=torch.bool, device=device)
    pieces2word = fill(pieces2word, sub_mat)

    return [tokens_ids, pieces2word, dist_inputs, sent_length, grid_mask2d], [grid_labels, grid_mask2d, _entity_text]


# 加载数据集
train_dataloader = DataLoader(MyDataset(''), ##数据集文件位置
                              batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(MyDataset(''), ##数据集文件位置
                              batch_size=batch_size, collate_fn=collate_fn)
