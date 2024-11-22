import os
import torch
import logging
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from ner_loader import NERLoader
from model import AGNERModel
from loss import AGNERLoss

# 日志设置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Diffusion Memory Buffer
class DiffusionMemoryBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add(self, examples):
        """
        添加样本到 DMB 中，确保总样本数量不超过最大限制。
        """
        self.buffer.extend(examples)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def sample(self, batch_size):
        """
        从 DMB 中随机采样指定数量的样本。
        """
        if len(self.buffer) == 0:
            return []
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def generate_memory_samples(self, model, tokenizer, device, num_samples=32):
        """
        利用扩散机制生成记忆样本。
        Args:
            model: 当前训练的模型
            tokenizer: 分词器
            device: 训练设备
            num_samples: 要生成的样本数
        Returns:
            生成的样本
        """
        generated_samples = []
        for _ in range(num_samples):
            # 使用扩散机制生成记忆样本（占位实现，根据具体需求实现扩散逻辑）
            sample = model.generate_fake_sample(tokenizer, device)
            generated_samples.append(sample)
        return generated_samples


# 加载数据集
def load_task_data(task_config, loader, setup="split"):
    logging.info(f"Loading data for task: {task_config['name']}")
    datafiles = {"train": task_config["train"], "dev": task_config["valid"], "test": None}
    loader.load_data(datafiles, setup=setup, batch_size=CONFIG["training"]["batch_size"])
    return loader.get_loader(mode="train"), loader.get_loader(mode="dev")


# 初始化模型
def initialize_model(device):
    tokenizer = BertTokenizer.from_pretrained(CONFIG["model"]["name"])
    model = AGNERModel(CONFIG["model"])
    model.to(device)
    return model, tokenizer


# 持续学习的训练
def continual_learning(model, tasks, device):
    dmb = DiffusionMemoryBuffer(CONFIG["training"]["buffer_size"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    loss_fn = AGNERLoss(entity_type_count=CONFIG["model"]["num_entity_types"], device=device, optimizer=optimizer,
                        scheduler=None, max_grad_norm=CONFIG["training"]["max_grad_norm"], nil_weight=0.1,
                        match_class_weight=1.0, match_boundary_weight=1.0, loss_class_weight=1.0,
                        loss_boundary_weight=1.0)

    for task_id, task_config in enumerate(tasks):
        train_loader, dev_loader = load_task_data(task_config, loader)
        logging.info(f"Training on Task {task_id + 1}/{len(tasks)}: {task_config['name']}")

        for epoch in range(CONFIG["training"]["epochs_per_task"]):
            model.train()
            logging.info(f"Epoch {epoch + 1}/{CONFIG['training']['epochs_per_task']}")
            train_loss = 0.0

            for step, batch in enumerate(train_loader):
                inputs, labels = batch
                inputs = [x.to(device) for x in inputs]
                labels = [x.to(device) for x in labels]
                optimizer.zero_grad()

                # Diffusion Memory Buffer 数据采样
                replay_samples = dmb.sample(CONFIG["training"]["buffer_sampling_size"])
                if replay_samples:
                    replay_inputs, replay_labels = zip(*replay_samples)
                    replay_inputs = [torch.cat(tensors, dim=0).to(device) for tensors in zip(*replay_inputs)]
                    replay_labels = [torch.cat(tensors, dim=0).to(device) for tensors in zip(*replay_labels)]
                    inputs = [torch.cat((inputs[i], replay_inputs[i]), dim=0) for i in range(len(inputs))]
                    labels = [torch.cat((labels[i], replay_labels[i]), dim=0) for i in range(len(labels))]

                # 损失计算与更新
                loss = loss_fn.step(model(inputs), labels)
                train_loss += loss

                if (step + 1) % CONFIG["training"]["log_steps"] == 0:
                    logging.info(f"Step {step + 1}, Loss: {train_loss / (step + 1):.4f}")

            # 生成扩散记忆样本并加入 DMB
            generated_samples = dmb.generate_memory_samples(model, tokenizer, device, num_samples=32)
            dmb.add(list(zip(inputs, labels)) + generated_samples)

            # 验证模型
            eval_f1 = evaluate_model(model, dev_loader, device)
            logging.info(f"Validation F1 for Task {task_id + 1}: {eval_f1:.4f}")


# 验证
def evaluate_model(model, valid_loader, device):
    model.eval()
    total_correct, total_predicted, total_gold = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc="Evaluating"):
            inputs = [x.to(device) for x in inputs]
            pred_entities = model.predict(inputs)
            gold_entities = labels[-1]
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

    loader = NERLoader(entity_task_list=[[...]], bert_model_dir=CONFIG["model"]["name"])
    tasks = CONFIG["datasets"]["tasks"]

    model, tokenizer = initialize_model(device)
    continual_learning(model, tasks, device)


if __name__ == "__main__":
    main()
