import os
import json
import torch
import logging
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertModel
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from data_loader import MyDataset, collate_fn
from model import NERModel

# 日志设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置参数
CONFIG = {
    "datasets": {
        "conll2003": {"train": "data/datasets/conll2003/conll2003_train.json",
                      "valid": "data/datasets/conll2003/conll2003_dev.json"},
        "ontonotes5": {"train": "data/datasets/ontonotes5/ontonotes5_train.json",
                       "valid": "data/datasets/ontonotes5/ontonotes5_dev.json"},
        "msra": {"train": "data/datasets/msra/msra_train.json",
                 "valid": "data/datasets/msra/msra_dev.json"},
        "resume": {"train": "data/datasets/resume/resume_train.json",
                   "valid": "data/datasets/resume/resume_dev.json"},
        "weibo": {"train": "data/datasets/weibo/weibo_train.json",
                  "valid": "data/datasets/weibo/weibo_dev.json"},
        "ace2004": {"train": "data/datasets/ace2004/ace2004_train.json",
                    "valid": "data/datasets/ace2004/ace2004_dev.json"},
        "ace2005": {"train": "data/datasets/ace2005/ace2005_train.json",
                    "valid": "data/datasets/ace2005/ace2005_dev.json"},
        "agner": {"train": "data/datasets/agner/agner_train.json",
                  "valid": "data/datasets/agner/agner_dev.json"}
    },
    "model": {
        "name": "bert-base-chinese",
        "hidden_size": 768,
        "dropout": 0.1
    },
    "training": {
        "batch_size": 16,
        "epochs": 50,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "log_steps": 10
    },
    "save_dir": "results/"
}

# 加载数据集
def load_data(dataset_name):
    dataset_config = CONFIG["datasets"].get(dataset_name)
    if not dataset_config:
        raise ValueError(f"Dataset {dataset_name} is not configured.")
    train_dataset = MyDataset(dataset_config["train"])
    valid_dataset = MyDataset(dataset_config["valid"])
    return train_dataset, valid_dataset

# 初始化模型
def initialize_model(device):
    tokenizer = BertTokenizer.from_pretrained(CONFIG["model"]["name"])
    model = NERModel(CONFIG["model"])
    model.to(device)
    return model, tokenizer

# 训练
def train_model(model, train_loader, valid_loader, device, save_path):
    optimizer = AdamW(model.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    best_f1 = 0.0
    for epoch in range(CONFIG["training"]["epochs"]):
        model.train()
        logging.info(f"Epoch {epoch + 1}/{CONFIG['training']['epochs']}")
        train_loss = 0.0
        for step, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training")):
            inputs = [x.to(device) for x in inputs]
            labels = [x.to(device) for x in labels[:-1]]
            optimizer.zero_grad()
            loss = model(inputs, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), CONFIG["training"]["max_grad_norm"])
            optimizer.step()
            train_loss += loss.item()

            if (step + 1) % CONFIG["training"]["log_steps"] == 0:
                logging.info(f"Step {step + 1}, Loss: {train_loss / (step + 1):.4f}")

        # 验证阶段
        eval_f1 = evaluate_model(model, valid_loader, device)
        logging.info(f"Epoch {epoch + 1}, Validation F1: {eval_f1:.4f}")
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pt"))
            logging.info(f"New best model saved with F1: {best_f1:.4f}")

# 验证
def evaluate_model(model, valid_loader, device):
    model.eval()
    total_correct, total_predicted, total_gold = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc="Evaluating"):
            inputs = [x.to(device) for x in inputs]
            pred_entities = model.predict(inputs)
            gold_entities = labels[-1]  # _entity_text
            # 比较预测和真实实体
            total_correct += len(set(pred_entities) & set(gold_entities))
            total_predicted += len(pred_entities)
            total_gold += len(gold_entities)

    precision = total_correct / total_predicted if total_predicted > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return f1

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    for dataset_name in CONFIG["datasets"]:
        logging.info(f"Processing dataset: {dataset_name}")
        train_dataset, valid_dataset = load_data(dataset_name)
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=CONFIG["training"]["batch_size"], collate_fn=collate_fn)

        save_path = os.path.join(CONFIG["save_dir"], dataset_name)
        os.makedirs(save_path, exist_ok=True)
        model, tokenizer = initialize_model(device)

        train_model(model, train_loader, valid_loader, device, save_path)

if __name__ == "__main__":
    main()
