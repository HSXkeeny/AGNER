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

import random
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SubsetSequentialSampler
from typing import List, Dict


class NERLoader:
    def __init__(self, entity_task_list: List[List[str]], bert_model_dir: str, max_len: int = 512, seed: int = 42):

        self.entity_task_list = entity_task_list
        self.num_tasks = len(entity_task_list)
        self.max_len = max_len
        self.bert_model_dir = bert_model_dir
        self.seed = seed

        # 实体与任务的映射
        self.entity_to_task = {entity: tid for tid, entities in enumerate(entity_task_list) for entity in entities}

        # 数据读取器
        self.data_reader = NerDataReader(bert_model_dir, max_len, ent_file_or_ent_lst=self.get_all_entities())

    def get_all_entities(self) -> List[str]:
        """获取所有任务的实体集合"""
        return sum(self.entity_task_list, [])

    def load_data(self, datafiles: Dict[str, str], setup: str = 'split', batch_size: int = 16, quick_test: bool = False):
        """
        加载数据集，并构建任务数据集与加载器。
        Args:
            datafiles: 包含 train/dev/test 数据文件路径的字典
            setup: 数据划分方式（split 或 filter）
            batch_size: 批量大小
            quick_test: 是否快速测试（使用 dev 替代 train）
        """
        self.datafiles = datafiles
        self.batch_size = batch_size

        # 训练集加载
        if not quick_test:
            self.train_examples, self.train_task_to_indices = self._load_task_data(datafiles["train"], setup)

        # 验证集加载
        self.dev_examples, self.dev_task_to_indices = self._load_task_data(datafiles["dev"], setup)
        if quick_test:
            self.train_examples = self.dev_examples
            self.train_task_to_indices = self.dev_task_to_indices

        # 测试集加载
        self.test_examples, self.test_task_to_indices = self._load_task_data(datafiles["test"], 'filter')

        # 构建数据集
        self.train_dataset = self.data_reader.build_dataset(self.train_examples, arch="span", loss_type="sigmoid")
        self.dev_dataset = self.data_reader.build_dataset(self.dev_examples, arch="span", loss_type="sigmoid")
        self.test_dataset = self.data_reader.build_dataset(self.test_examples, arch="span", loss_type="sigmoid")

        # 初始化加载器
        self.init_loaders()

    def _load_task_data(self, file_path: str, setup: str = 'split') -> (List, Dict[int, set]):
        """
        加载任务相关数据并根据任务划分样本。
        Args:
            file_path: 数据文件路径
            setup: 数据划分方式（split 或 filter）
        Returns:
            数据样本列表及任务到样本索引的映射
        """
        examples = NerExample.load_from_jsonl(file_path, token_deli=' ',
                                              external_attrs=['task_id', 'bert_tok_char_lst', 'ori_2_tok'])
        task_to_indices = {tid: set() for tid in range(self.num_tasks)}

        if setup == 'split':
            random.seed(self.seed)
            random.shuffle(examples)
            num_per_task = len(examples) // self.num_tasks

            for tid in range(self.num_tasks):
                start = tid * num_per_task
                end = start + num_per_task if tid < self.num_tasks - 1 else len(examples)
                for idx in range(start, end):
                    examples[idx].task_id = tid
                    task_to_indices[tid].add(idx)

        elif setup == 'filter':
            for idx, example in enumerate(examples):
                for entity in example.entities:
                    if entity in self.entity_to_task:
                        task_to_indices[self.entity_to_task[entity]].add(idx)

        else:
            raise ValueError(f"Unsupported setup: {setup}")

        return examples, task_to_indices

    def init_loaders(self):
        """初始化任务加载器"""
        self.train_loaders = []
        for tid, indices in self.train_task_to_indices.items():
            sampler = SubsetRandomSampler(list(indices))
            loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler,
                                collate_fn=self.data_reader.get_batcher_fn(arch="span"))
            self.train_loaders.append(loader)

        self.dev_loaders = []
        for tid, indices in self.dev_task_to_indices.items():
            sampler = SubsetSequentialSampler(list(indices))
            loader = DataLoader(self.dev_dataset, batch_size=self.batch_size, sampler=sampler,
                                collate_fn=self.data_reader.get_batcher_fn(arch="span"))
            self.dev_loaders.append(loader)

        self.test_loaders = []
        for tid, indices in self.test_task_to_indices.items():
            sampler = SubsetSequentialSampler(list(indices))
            loader = DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=sampler,
                                collate_fn=self.data_reader.get_batcher_fn(arch="span"))
            self.test_loaders.append(loader)

    def get_loader(self, mode: str, task_id: int = None):
        """
        获取指定模式与任务的加载器。
        Args:
            mode: 加载器模式（train/dev/test）
            task_id: 指定任务 ID（如果为 None，则返回所有任务加载器）
        """
        if task_id is None:
            return getattr(self, f"{mode}_loaders")
        if mode == "train":
            return self.train_loaders[task_id]
        elif mode == "dev":
            return self.dev_loaders[task_id]
        elif mode == "test":
            return self.test_loaders[task_id]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

# 加载数据集
train_dataloader = DataLoader(MyDataset(''), ##数据集文件位置
                              batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(MyDataset(''), ##数据集文件位置
                              batch_size=batch_size, collate_fn=collate_fn)

