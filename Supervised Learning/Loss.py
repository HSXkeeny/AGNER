import torch
from torch import nn
from torch.nn import functional as F
from .matcher import HopcroftKarpMatcher


class AGNERLoss:
    def __init__(self, entity_type_count, device, optimizer, scheduler, max_grad_norm,
                 nil_weight, match_class_weight, match_boundary_weight,
                 loss_class_weight, loss_boundary_weight):
        """
        初始化 AGNER 损失函数。
        Args:
            entity_type_count: 实体类别数量
            device: 计算设备
            optimizer: 优化器
            scheduler: 学习率调度器
            max_grad_norm: 梯度裁剪阈值
            nil_weight: nil 类别的权重
            match_class_weight: 分类匹配权重
            match_boundary_weight: 边界匹配权重
            loss_class_weight: 分类损失权重
            loss_boundary_weight: 边界损失权重
        """
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm

        # 定义权重和损失
        self.weight_dict = {
            "loss_class": loss_class_weight,
            "loss_boundary": loss_boundary_weight
        }
        self.matcher = HopcroftKarpMatcher(
            cost_class=match_class_weight,
            cost_span=match_boundary_weight
        )
        self.nil_weight = nil_weight

        # 初始化类别权重
        empty_weight = torch.ones(entity_type_count)
        empty_weight[0] = self.nil_weight
        self.empty_weight = empty_weight.to(device)

    def compute_loss(self, outputs, targets):
        """
        计算总损失，包括分类损失和边界损失。
        Args:
            outputs: 模型的输出，包括预测的 logits 和边界信息
            targets: 标签，包括真实的类别和边界
        Returns:
            总损失值
        """
        # 获取匹配
        indices = self.matcher(outputs, targets)

        # 分类损失
        loss_class = self._compute_class_loss(outputs, targets, indices)

        # 边界损失
        loss_boundary = self._compute_boundary_loss(outputs, targets, indices)

        # 加权组合损失
        total_loss = (
            self.weight_dict["loss_class"] * loss_class +
            self.weight_dict["loss_boundary"] * loss_boundary
        )
        return total_loss

    def _compute_class_loss(self, outputs, targets, indices):
        """
        分类损失的计算（交叉熵损失）。
        Args:
            outputs: 模型输出，包括预测 logits
            targets: 标签，包括真实类别
            indices: Hopcroft-Karp Matcher 的匹配结果
        Returns:
            分类损失值
        """
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=self.device)
        target_classes[idx] = torch.cat([targets["labels"][i] for _, i in indices])
        loss_class = F.cross_entropy(src_logits.view(-1, src_logits.size(-1)),
                                     target_classes.view(-1),
                                     weight=self.empty_weight)
        return loss_class

    def _compute_boundary_loss(self, outputs, targets, indices):
        """
        边界损失的计算（Binary Cross-Entropy 损失）。
        Args:
            outputs: 模型输出，包括边界预测
            targets: 标签，包括真实边界
            indices: Hopcroft-Karp Matcher 的匹配结果
        Returns:
            边界损失值
        """
        idx = self._get_src_permutation_idx(indices)
        pred_left, pred_right = outputs["pred_left"][idx], outputs["pred_right"][idx]
        target_left = torch.cat([targets["gt_left"][i] for _, i in indices])
        target_right = torch.cat([targets["gt_right"][i] for _, i in indices])

        left_loss = F.binary_cross_entropy_with_logits(pred_left, target_left.float())
        right_loss = F.binary_cross_entropy_with_logits(pred_right, target_right.float())

        return left_loss + right_loss

    def _get_src_permutation_idx(self, indices):
        """
        根据 Hopcroft-Karp Matcher 的结果重新排列输出。
        Args:
            indices: Hopcroft-Karp Matcher 的匹配结果
        Returns:
            重新排列的索引
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def step(self, outputs, targets):
        """
        损失反向传播并更新参数。
        Args:
            outputs: 模型输出
            targets: 标签
        """
        loss = self.compute_loss(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return loss.item()
