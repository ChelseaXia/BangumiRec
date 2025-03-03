import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import pickle
import os
import pandas as pd
from typing import Dict, Tuple, List, Any
from tqdm import tqdm
from models import LLMGNNRecommender

class AnimeDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, edges, labels):
        self.edges = edges
        self.labels = labels
        
    def __len__(self):
        return self.edges.size(1)
        
    def __getitem__(self, idx):
        return {
            'edge': self.edges[:, idx],
            'label': self.labels[idx]
        }

class LightGCNTrainer:
    def __init__(
        self,
        model: nn.Module,
        processed_data: Dict,
        device: torch.device,
        learning_rate: float = 0.001,
        batch_size: int = 1024,
        num_epochs: int = 100,
        early_stopping_patience: int = 10
    ):
        """
        LightGCN模型训练器
        
        Args:
            model: LightGCN模型实例
            processed_data: 预处理后的数据字典
            device: 训练设备
            learning_rate: 学习率
            batch_size: 批次大小
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
        """
        self.model = model.to(device)
        self.processed_data = processed_data
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = early_stopping_patience
        
        # 设置优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 设置损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 处理评论嵌入 - 支持多种格式
        self.process_comment_embeddings()
        
        # 准备数据加载器
        self.prepare_dataloaders()
    
    def process_comment_embeddings(self):
        """处理不同格式的评论嵌入数据"""
        comment_embeddings = self.processed_data.get('comment_embeddings', None)
        if comment_embeddings is None:
            return
            
        self.logger.info("处理评论嵌入数据...")
        self.logger.info(f"评论嵌入类型: {type(comment_embeddings)}, 长度: {len(comment_embeddings)}")
        
        if isinstance(comment_embeddings, list):
            # 检查第一个元素
            if len(comment_embeddings) > 0:
                self.logger.info(f"第一个元素类型: {type(comment_embeddings[0])}")
                if isinstance(comment_embeddings[0], dict):
                    # 输出字典的键
                    self.logger.info(f"字典键: {list(comment_embeddings[0].keys())}")
                    
                    # 检查是否是批次字典列表格式
                    if all(isinstance(item, dict) for item in comment_embeddings):
                        self.logger.info("检测到字典格式的评论嵌入")
                        
                        # 提取嵌入数据
                        valid_embeddings = []
                        for i, batch_data in enumerate(comment_embeddings):
                            # 检查特殊字典结构
                            if 'is_empty_placeholder' in batch_data and batch_data['is_empty_placeholder']:
                                self.logger.info(f"找到空评论占位符，索引: {i}")
                                continue
                                
                            if 'embeddings' in batch_data:
                                embed_shape = np.array(batch_data['embeddings']).shape
                                self.logger.info(f"索引 {i} 的嵌入形状: {embed_shape}")
                                
                                # 检查维度
                                if len(embed_shape) >= 2:
                                    self.logger.info(f"添加有效嵌入, 形状: {embed_shape}")
                                    valid_embeddings.append(batch_data['embeddings'])
                        
                        # 如果有有效嵌入，合并它们
                        if valid_embeddings:
                            try:
                                combined_embeddings = np.vstack(valid_embeddings)
                                self.processed_data['comment_embeddings'] = torch.tensor(combined_embeddings)
                                self.logger.info(f"成功合并评论嵌入，形状: {combined_embeddings.shape}")
                            except Exception as e:
                                self.logger.error(f"合并评论嵌入时出错: {str(e)}")
                                # 放弃使用评论嵌入
                                self.processed_data['comment_embeddings'] = None
                        else:
                            self.logger.warning("未找到有效的评论嵌入，设置为None")
                            self.processed_data['comment_embeddings'] = None
        
    def ensure_comment_item_mapping(self):
        """确保存在评论到物品的映射"""
        if ('comment_embeddings' not in self.processed_data) or ('comment_embedding_to_item_idx' in self.processed_data):
            return
            
        # 尝试从comment_to_item_map创建映射
        if 'comment_to_item_map' in self.processed_data:
            self.logger.info("从comment_to_item_map创建评论到物品索引的映射...")
            
            comment_to_item_map = self.processed_data['comment_to_item_map']
            item_id_map = self.processed_data.get('item_id_map', {})
            
            comment_embedding_to_item_idx = {}
            for comment_idx, item_id in comment_to_item_map.items():
                if item_id in item_id_map:
                    comment_embedding_to_item_idx[comment_idx] = item_id_map[item_id]
                    
            self.processed_data['comment_embedding_to_item_idx'] = comment_embedding_to_item_idx
            self.logger.info(f"创建了 {len(comment_embedding_to_item_idx)} 个评论到物品索引的映射")
        
    def load_edges_from_file(self, file_path):
        """从文件加载边数据"""
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                edge_data = pickle.load(f)
                
            # 转换为张量
            user_indices = torch.tensor(edge_data['user_indices'])
            item_indices = torch.tensor(edge_data['item_indices'])
            
            return torch.stack([user_indices, item_indices])
        else:
            self.logger.warning(f"边数据文件 {file_path} 不存在")
            return None
        
    def prepare_dataloaders(self):
        """准备训练、验证和测试数据加载器 - 增强版，更严格的索引检查"""
        self.logger.info("准备数据加载器...")
        
        # 加载边数据，优先使用已存在的边数据
        train_edges = self.processed_data.get('train_edges', None)
        val_edges = self.processed_data.get('val_edges', None)
        test_edges = self.processed_data.get('test_edges', None)
        
        # 如果边数据不存在，尝试从数据目录加载
        output_dir = self.processed_data.get('output_dir', '')
        if train_edges is None and output_dir:
            train_edges = self.load_edges_from_file(os.path.join(output_dir, 'train_edges.pkl'))
        if val_edges is None and output_dir:
            val_edges = self.load_edges_from_file(os.path.join(output_dir, 'val_edges.pkl'))
        if test_edges is None and output_dir:
            test_edges = self.load_edges_from_file(os.path.join(output_dir, 'test_edges.pkl'))
        
        # 如果仍然没有边数据，从交互矩阵创建
        if train_edges is None and 'train_interaction_matrix' in self.processed_data:
            self.logger.info("从交互矩阵创建边数据...")
            train_matrix = self.processed_data['train_interaction_matrix']
            if train_matrix.is_sparse:
                indices = train_matrix._indices()
                train_edges = indices
                
                # 为验证和测试创建简单分割
                n_edges = indices.size(1)
                all_indices = torch.randperm(n_edges)
                train_size = int(n_edges * 0.8)
                val_size = int(n_edges * 0.1)
                
                train_idx = all_indices[:train_size]
                val_idx = all_indices[train_size:train_size+val_size]
                test_idx = all_indices[train_size+val_size:]
                
                train_edges = indices[:, train_idx]
                val_edges = indices[:, val_idx]
                test_edges = indices[:, test_idx]
        
        if train_edges is None:
            raise ValueError("无法找到或创建训练边数据")
        
        # 获取模型维度信息
        num_users = self.processed_data.get('num_users', 0)
        num_items = self.processed_data.get('num_items', 0)
        
        # 如果没有明确的用户和物品数量，从id_mappings获取
        if num_users == 0 and 'user_id_map' in self.processed_data:
            num_users = len(self.processed_data['user_id_map'])
            self.processed_data['num_users'] = num_users
        
        if num_items == 0 and 'item_id_map' in self.processed_data:
            num_items = len(self.processed_data['item_id_map'])
            self.processed_data['num_items'] = num_items
        
        self.logger.info(f"数据集维度: 用户数={num_users}, 物品数={num_items}")
        
        # 递进的边过滤函数
        def filter_edges_safely(edges, device, name="未知"):
            """安全地过滤超出范围的边，并返回过滤后的边和掩码"""
            if edges is None:
                return None, None
                
            # 移动到指定设备
            edges = edges.to(device)
            
            # 记录原始边数
            orig_count = edges.size(1)
            self.logger.info(f"{name}数据集原始边数: {orig_count}")
            
            # 检查用户索引范围
            user_mask = edges[0] < num_users
            invalid_users = (~user_mask).sum().item()
            if invalid_users > 0:
                self.logger.warning(f"{name}数据集中有 {invalid_users} 条边的用户索引超出范围")
                
            # 检查物品索引范围
            item_mask = edges[1] < num_items
            invalid_items = (~item_mask).sum().item()
            if invalid_items > 0:
                self.logger.warning(f"{name}数据集中有 {invalid_items} 条边的物品索引超出范围")
                
            # 同时满足两个条件
            valid_mask = user_mask & item_mask
            valid_count = valid_mask.sum().item()
            
            if valid_count < orig_count:
                self.logger.warning(f"{name}数据集过滤后边数: {valid_count}/{orig_count} ({valid_count/orig_count*100:.2f}%)")
                
            # 应用过滤
            filtered_edges = edges[:, valid_mask]
            
            # 如果过滤后为空，返回None
            if filtered_edges.size(1) == 0:
                self.logger.error(f"{name}数据集过滤后为空")
                return None, None
                
            return filtered_edges, valid_mask
        
        # 处理边数据
        self.train_edges, train_mask = filter_edges_safely(train_edges, self.device, "训练")
        self.val_edges, val_mask = filter_edges_safely(val_edges, self.device, "验证") 
        self.test_edges, test_mask = filter_edges_safely(test_edges, self.device, "测试")
        
        # 创建标签
        # 检查是否有is_negative信息
        pos_samples = self.processed_data.get('positive_samples', None)
        neg_samples = self.processed_data.get('negative_samples', None)
        
        if pos_samples is not None and neg_samples is not None:
            # 合并正负样本
            all_samples = pd.concat([pos_samples, neg_samples], ignore_index=True)
            train_labels = torch.tensor(all_samples['is_negative'].values, dtype=torch.float)
        else:
            # 假设所有训练边都是正样本，使用1作为标签
            train_labels = torch.ones(train_edges.size(1), dtype=torch.float) if self.train_edges is not None else None
        
        # 验证和测试标签都设为1（假设它们都是正样本）
        val_labels = torch.ones(self.val_edges.size(1), dtype=torch.float) if self.val_edges is not None else None
        test_labels = torch.ones(self.test_edges.size(1), dtype=torch.float) if self.test_edges is not None else None
        
        # 应用标签掩码
        if train_labels is not None and train_mask is not None:
            train_labels = train_labels.to(self.device)
            train_labels = train_labels[train_mask]
        
        if val_labels is not None and val_mask is not None:
            val_labels = val_labels.to(self.device)
            val_labels = val_labels[val_mask]
        
        if test_labels is not None and test_mask is not None:
            test_labels = test_labels.to(self.device)
            test_labels = test_labels[test_mask]
        
        # 创建数据集
        if self.train_edges is not None and train_labels is not None:
            self.train_dataset = AnimeDataset(self.train_edges, train_labels)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            self.logger.info(f"训练数据集大小: {len(self.train_dataset)}")
        else:
            self.train_dataset = None
            self.train_loader = None
            self.logger.warning("无法创建训练数据集")
        
        if self.val_edges is not None and val_labels is not None:
            self.val_dataset = AnimeDataset(self.val_edges, val_labels)
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            self.logger.info(f"验证数据集大小: {len(self.val_dataset)}")
        else:
            self.val_dataset = None
            self.val_loader = None
            self.logger.warning("无法创建验证数据集")
        
        if self.test_edges is not None and test_labels is not None:
            self.test_dataset = AnimeDataset(self.test_edges, test_labels)
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            self.logger.info(f"测试数据集大小: {len(self.test_dataset)}")
        else:
            self.test_dataset = None
            self.test_loader = None
            self.logger.warning("无法创建测试数据集")
    
    def train_epoch(self) -> float:
        """训练一个epoch - 增强版，更严格的索引检查和错误处理"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        skipped_batches = 0
        
        if self.train_loader is None:
            self.logger.error("训练数据加载器为空，无法训练")
            return float('inf')
        
        for batch in tqdm(self.train_loader, desc="训练中"):
            self.optimizer.zero_grad()
            
            # 获取用户和物品嵌入
            user_emb, item_emb = self.model(
                self.processed_data['train_interaction_matrix'].to(self.device),
                self.processed_data.get('comment_embeddings'),
                self.processed_data.get('comment_embedding_to_item_idx')
            )
            
            # 准备批次数据
            edges = batch['edge']
            labels = batch['label']
            
            # 检查边数据是否为空
            if edges.size(1) == 0:
                self.logger.warning("跳过空批次")
                skipped_batches += 1
                continue
            
            # 检查索引是否越界 - 记录批次大小
            if num_batches % 100 == 0:  # 只在每100个批次打印一次，避免输出过多
                self.logger.debug(f"当前批次边数形状: {edges.shape}, 标签数: {labels.size(0)}")
            
            # 严格检查索引范围
            max_user_idx = edges[0].max().item()
            max_item_idx = edges[1].max().item()
            
            if max_user_idx >= user_emb.size(0) or max_item_idx >= item_emb.size(0):
                if num_batches % 100 == 0:  # 只在每100个批次打印一次，避免输出过多
                    self.logger.warning(f"警告: 索引越界，跳过这个批次. "
                                    f"最大用户索引: {max_user_idx}, 用户嵌入大小: {user_emb.size(0)}, "
                                    f"最大物品索引: {max_item_idx}, 物品嵌入大小: {item_emb.size(0)}")
                skipped_batches += 1
                continue
            
            try:
                # 计算预测分数
                users = user_emb[edges[0]]
                items = item_emb[edges[1]]
                
                # 确保用element-wise乘法并在特征维度求和
                pred = torch.sum(users * items, dim=1)
                
                # 如果形状不匹配，调整大小
                if pred.size(0) != labels.size(0):
                    min_size = min(pred.size(0), labels.size(0))
                    pred = pred[:min_size]
                    labels = labels[:min_size]
                    
                    if num_batches % 100 == 0:  # 只在每100个批次打印一次
                        self.logger.debug(f"截断后 - 预测形状: {pred.shape}, 标签形状: {labels.shape}")
                
                # 计算损失
                loss = self.criterion(pred, labels)
                
                # 添加L2正则化
                l2_reg = 0
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += 0.001 * l2_reg
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except RuntimeError as e:
                # 处理可能的CUDA错误
                if "CUDA" in str(e) or "index out of bounds" in str(e):
                    self.logger.warning(f"批次处理出错，跳过: {str(e)}")
                    skipped_batches += 1
                    torch.cuda.empty_cache()  # 清理GPU内存
                    continue
                else:
                    # 其他错误重新抛出
                    raise
        
        # 汇报跳过的批次
        if skipped_batches > 0:
            self.logger.warning(f"本epoch共跳过 {skipped_batches}/{len(self.train_loader)} 个批次")
            
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        评估模型性能 - 增强版，更多的错误处理和边界检查
        
        Args:
            data_loader: 数据加载器
                
        Returns:
            包含评估指标的字典
        """
        if data_loader is None:
            return {'AUC': 0, 'MRR': 0, 'NDCG@5': 0, 'NDCG@10': 0}
                
        self.model.eval()
        total_auc = 0
        total_mrr = 0
        total_ndcg_5 = 0
        total_ndcg_10 = 0
        num_users = 0
        skipped_users = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估中"):
                try:
                    edges = batch['edge']
                    
                    # 获取用户和物品嵌入
                    user_emb, item_emb = self.model(
                        self.processed_data['train_interaction_matrix'].to(self.device),
                        self.processed_data.get('comment_embeddings'),
                        self.processed_data.get('comment_embedding_to_item_idx')
                    )
                    
                    # 获取当前批次的用户
                    batch_users = torch.unique(edges[0])
                    
                    for user in batch_users:
                        # 安全检查 - 确保用户索引在范围内
                        if user >= user_emb.size(0):
                            skipped_users += 1
                            continue
                            
                        # 获取用户的真实交互
                        user_interactions = edges[:, edges[0] == user]
                        true_items = user_interactions[1]
                        
                        # 安全检查 - 忽略超出范围的物品
                        valid_items = true_items[true_items < item_emb.size(0)]
                        if len(valid_items) == 0 or len(valid_items) < len(true_items):
                            skipped_users += 1
                            continue
                        
                        try:
                            # 计算用户对所有物品的预测分数
                            user_embed = user_emb[user].unsqueeze(0)
                            scores = torch.mm(user_embed, item_emb.t()).squeeze()
                            
                            # 安全检查 - 确保真实交互的索引不超出scores范围
                            if valid_items.max() >= len(scores):
                                skipped_users += 1
                                continue
                            
                            # 将已交互物品的分数设置为负无穷（不参与推荐）
                            for item_idx in valid_items:
                                scores[item_idx] = float('-inf')
                            
                            # 计算评估指标
                            auc = self.calculate_auc_safely(scores, valid_items)
                            mrr = self.calculate_mrr_safely(scores, valid_items)
                            ndcg_5 = self.calculate_ndcg_safely(scores, valid_items, k=5)
                            ndcg_10 = self.calculate_ndcg_safely(scores, valid_items, k=10)
                            
                            total_auc += auc
                            total_mrr += mrr
                            total_ndcg_5 += ndcg_5
                            total_ndcg_10 += ndcg_10
                            num_users += 1
                        except Exception as e:
                            self.logger.warning(f"评估用户时出错: {str(e)}")
                            skipped_users += 1
                            continue
                except Exception as e:
                    self.logger.warning(f"评估批次时出错: {str(e)}")
                    continue
            
            # 报告跳过的用户数量
            if skipped_users > 0:
                self.logger.warning(f"评估过程中跳过了 {skipped_users} 个用户")
            
            # 计算平均指标
            metrics = {
                'AUC': total_auc / num_users if num_users > 0 else 0,
                'MRR': total_mrr / num_users if num_users > 0 else 0,
                'NDCG@5': total_ndcg_5 / num_users if num_users > 0 else 0,
                'NDCG@10': total_ndcg_10 / num_users if num_users > 0 else 0
            }
            
            return metrics
    
    def calculate_auc_safely(self, scores: torch.Tensor, positive_items: torch.Tensor) -> float:
        """安全地计算AUC分数，处理各种边界情况"""
        try:
            # 创建掩码，排除负无穷值
            valid_mask = scores != float('-inf')
            if not valid_mask.any():
                return 0.0
                
            neg_scores = scores[valid_mask]
            
            # 确保positive_items中的索引在scores范围内
            valid_pos_mask = positive_items < len(scores)
            if not valid_pos_mask.any():
                return 0.0
                
            valid_pos_items = positive_items[valid_pos_mask]
            if len(valid_pos_items) == 0:
                return 0.0
                
            pos_scores = scores[valid_pos_items]
            
            # 如果正样本或负样本数量为0，则返回0
            if len(pos_scores) == 0 or len(neg_scores) == 0:
                return 0.0
                
            pos_scores = pos_scores.unsqueeze(1)
            neg_scores = neg_scores.unsqueeze(0)
            
            # 计算AUC
            return ((pos_scores > neg_scores).float().mean()).item()
        except Exception as e:
            self.logger.warning(f"计算AUC时出错: {str(e)}")
            return 0.0

    def calculate_mrr_safely(self, scores: torch.Tensor, positive_items: torch.Tensor) -> float:
        """安全地计算MRR分数，处理各种边界情况"""
        try:
            if len(positive_items) == 0:
                return 0.0
                
            # 确保所有索引在范围内
            valid_pos_mask = positive_items < len(scores)
            if not valid_pos_mask.any():
                return 0.0
                
            valid_pos_items = positive_items[valid_pos_mask]
            if len(valid_pos_items) == 0:
                return 0.0
                
            # 获取排名
            _, indices = torch.sort(scores, descending=True)
            
            ranks = torch.zeros_like(valid_pos_items, dtype=torch.float)
            for i, item in enumerate(valid_pos_items):
                rank_positions = (indices == item).nonzero()
                if len(rank_positions) > 0:
                    ranks[i] = rank_positions[0].item() + 1
                else:
                    ranks[i] = len(scores) + 1  # 最低排名
                    
            return (1.0 / ranks.float()).mean().item()
        except Exception as e:
            self.logger.warning(f"计算MRR时出错: {str(e)}")
            return 0.0

    def calculate_ndcg_safely(self, scores: torch.Tensor, positive_items: torch.Tensor, k: int) -> float:
        """安全地计算NDCG@k分数，处理各种边界情况"""
        try:
            if len(positive_items) == 0:
                return 0.0
                
            # 确保所有索引在范围内
            valid_pos_mask = positive_items < len(scores)
            if not valid_pos_mask.any():
                return 0.0
                
            valid_pos_items = positive_items[valid_pos_mask]
            if len(valid_pos_items) == 0:
                return 0.0
                
            # 计算top-k推荐
            _, indices = torch.sort(scores, descending=True)
            topk_indices = indices[:k]
            
            # 计算DCG和IDCG
            dcg = 0
            for i, idx in enumerate(topk_indices):
                if idx in valid_pos_items:
                    dcg += 1 / np.log2(i + 2)
            
            idcg = 0
            for i in range(min(len(valid_pos_items), k)):
                idcg += 1 / np.log2(i + 2)
                
            return dcg / idcg if idcg > 0 else 0
        except Exception as e:
            self.logger.warning(f"计算NDCG时出错: {str(e)}")
            return 0.0
    
    def train(self) -> Dict:
        """
        训练模型
        
        Returns:
            包含训练历史的字典
        """
        best_val_metrics = None
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'val_metrics': [],
            'best_epoch': 0
        }
        
        # 创建保存目录
        save_dir = self.processed_data.get('output_dir', '')
        if not save_dir:
            save_dir = './'
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'best_model.pth')
        
        for epoch in range(self.num_epochs):
            # 训练一个epoch
            train_loss = self.train_epoch()
            training_history['train_loss'].append(train_loss)
            
            # 在验证集上评估
            val_metrics = self.evaluate(self.val_loader) if self.val_loader else {'AUC': 0}
            training_history['val_metrics'].append(val_metrics)
            
            # 输出当前epoch的指标
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            if self.val_loader:
                self.logger.info("Validation Metrics:")
                for metric, value in val_metrics.items():
                    self.logger.info(f"{metric}: {value:.4f}")
            
            # 早停检查
            if self.val_loader:
                if best_val_metrics is None or val_metrics['AUC'] > best_val_metrics['AUC']:
                    best_val_metrics = val_metrics
                    training_history['best_epoch'] = epoch
                    patience_counter = 0
                    
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), model_path)
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            else:
                # 如果没有验证集，每个epoch保存模型
                torch.save(self.model.state_dict(), model_path)
        
        # 加载最佳模型（如果有验证集）并在测试集上评估
        if os.path.exists(model_path) and self.val_loader:
            self.model.load_state_dict(torch.load(model_path))
        
        if self.test_loader:
            test_metrics = self.evaluate(self.test_loader)
            
            self.logger.info("\nTest Set Metrics:")
            for metric, value in test_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
            
            training_history['test_metrics'] = test_metrics
        
        return training_history

def train_lightgcn_model(processed_data: Dict, config: Dict):
    """
    训练LightGCN模型的主函数
    
    Args:
        processed_data: 预处理后的数据
        config: 训练配置字典
    """
    # 检查矩阵维度
    print(f"交互矩阵形状: {processed_data['train_interaction_matrix'].size()}")
    print(f"用户数: {processed_data['num_users']}, 物品数: {processed_data['num_items']}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确保数据包含所需字段
    required_fields = ['num_users', 'num_items', 'train_interaction_matrix']
    missing_fields = [field for field in required_fields if field not in processed_data]
    
    if missing_fields:
        # 尝试从其他字段推断缺失字段
        if 'user_id_map' in processed_data and 'num_users' in missing_fields:
            processed_data['num_users'] = len(processed_data['user_id_map'])
        if 'item_id_map' in processed_data and 'num_items' in missing_fields:
            processed_data['num_items'] = len(processed_data['item_id_map'])
        
        # 创建交互矩阵（如果缺失）
        if 'train_interaction_matrix' in missing_fields and 'id_mappings' in processed_data:
            # 尝试从矩阵组件创建
            if 'interaction_matrix_components' in processed_data:
                components = processed_data['interaction_matrix_components']
                indices = torch.tensor(components['indices'])
                values = torch.tensor(components['values'])
                size = components['size']
                
                processed_data['train_interaction_matrix'] = torch.sparse_coo_tensor(
                    indices=indices,
                    values=values,
                    size=size
                )
    
    # 再次检查必要字段
    missing_fields = [field for field in required_fields if field not in processed_data]
    if missing_fields:
        raise ValueError(f"处理后的数据缺少必要字段: {missing_fields}")
    
    # 创建模型实例
    from models import LLMGNNRecommender
    model = LLMGNNRecommender(
        num_users=processed_data['num_users'],
        num_items=processed_data['num_items'],
        embed_dim=config['embed_dim'],
        n_layers=config.get('n_layers', 3),
        dropout=config.get('dropout', 0.1)
    )

    # 检查评论嵌入数据
    if 'comment_embeddings' in processed_data:
        if processed_data['comment_embeddings'] is not None:
            print(f"处理前的评论嵌入类型: {type(processed_data['comment_embeddings'])}")
            if isinstance(processed_data['comment_embeddings'], list):
                print(f"列表长度: {len(processed_data['comment_embeddings'])}")
                if len(processed_data['comment_embeddings']) > 0:
                    print(f"第一个元素类型: {type(processed_data['comment_embeddings'][0])}")
        else:
            print("评论为None")
            
    # 在train_lightgcn_model中，创建训练器之前
    processed_data['comment_embeddings'] = None
    print("已禁用评论嵌入功能，以确保模型正常运行")
    
    # 减小批次大小以减少潜在的内存问题
    config['batch_size'] = min(config['batch_size'], 256)
    print(f"调整批次大小为 {config['batch_size']}")
    
    # 创建训练器
    trainer = LightGCNTrainer(
        model=model,
        processed_data=processed_data,
        device=device,
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['patience']
    )
    
    # 开始训练
    training_history = trainer.train()
    return model, training_history


if __name__ == "__main__":
    import pickle
    import pandas as pd
    
    # 配置训练参数
    config = {
        'embed_dim': 64,
        'n_layers': 3,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 256,  # 降低批次大小
        'num_epochs': 100,
        'patience': 10
    }
    
    # 加载预处理数据
    processed_data_path = 'processed_data/metadata.pkl'
    if os.path.exists(processed_data_path):
        with open(processed_data_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"找到处理数据目录，包含文件: {metadata.get('files', [])}")
        
        # 设置数据目录
        processed_dir = os.path.dirname(processed_data_path)
        
        # 加载所有必要数据
        processed_data = {'output_dir': processed_dir}
        
        # 加载ID映射
        id_mappings_path = os.path.join(processed_dir, 'id_mappings.pkl')
        if os.path.exists(id_mappings_path):
            with open(id_mappings_path, 'rb') as f:
                id_mappings = pickle.load(f)
                processed_data['user_id_map'] = id_mappings['user_id_map']
                processed_data['item_id_map'] = id_mappings['item_id_map']
                processed_data['num_users'] = len(id_mappings['user_id_map'])
                processed_data['num_items'] = len(id_mappings['item_id_map'])
        
        # 加载交互矩阵组件
        matrix_path = os.path.join(processed_dir, 'interaction_matrix_components.pkl')
        if os.path.exists(matrix_path):
            with open(matrix_path, 'rb') as f:
                matrix_components = pickle.load(f)
                
                indices = torch.tensor(matrix_components['indices'])
                values = torch.tensor(matrix_components['values'])
                size = matrix_components['size']
                
                processed_data['train_interaction_matrix'] = torch.sparse_coo_tensor(
                    indices=indices,
                    values=values,
                    size=size
                )
        
        # 加载评论嵌入
        comment_path = os.path.join(processed_dir, 'comment_embeddings.pkl')
        if os.path.exists(comment_path):
            with open(comment_path, 'rb') as f:
                comment_embeddings_list = pickle.load(f)
                processed_data['comment_embeddings'] = comment_embeddings_list
        
        # 加载评论到物品的映射（如果存在）
        comment_map_path = os.path.join(processed_dir, 'comment_to_item_map.pkl')
        if os.path.exists(comment_map_path):
            with open(comment_map_path, 'rb') as f:
                comment_to_item_map = pickle.load(f)
                
                # 创建评论嵌入到物品索引的映射
                comment_embedding_to_item_idx = {}
                for comment_idx, subject_id in comment_to_item_map.items():
                    if subject_id in processed_data['item_id_map']:
                        item_idx = processed_data['item_id_map'][subject_id]
                        comment_embedding_to_item_idx[comment_idx] = item_idx
                
                processed_data['comment_embedding_to_item_idx'] = comment_embedding_to_item_idx
                print(f"加载了 {len(comment_embedding_to_item_idx)} 个评论到物品索引的映射")
        
        # 训练模型
        print("开始训练模型...")
        model, history = train_lightgcn_model(processed_data, config)
        
        print("训练完成!")
    else:
        print(f"找不到处理数据文件: {processed_data_path}")