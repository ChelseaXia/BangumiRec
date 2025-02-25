import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import pickle
import os
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
        
        # 处理评论嵌入
        if isinstance(self.processed_data.get('comment_embeddings', None), list):
            self.logger.info("合并评论嵌入列表...")
            self.processed_data['comment_embeddings'] = torch.tensor(
                np.concatenate(self.processed_data['comment_embeddings'], axis=0)
            )
        
        # 准备数据加载器
        self.prepare_dataloaders()
        
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
        """准备训练、验证和测试数据加载器"""
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
            
        # 处理边数据
        self.train_edges = train_edges.to(self.device)
        self.val_edges = val_edges.to(self.device) if val_edges is not None else None
        self.test_edges = test_edges.to(self.device) if test_edges is not None else None
        
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
            train_labels = torch.ones(self.train_edges.size(1), dtype=torch.float)
        
        # 验证和测试标签都设为1（假设它们都是正样本）
        val_labels = torch.ones(self.val_edges.size(1), dtype=torch.float) if self.val_edges is not None else None
        test_labels = torch.ones(self.test_edges.size(1), dtype=torch.float) if self.test_edges is not None else None
        
        # 创建数据集
        self.train_dataset = AnimeDataset(self.train_edges, train_labels.to(self.device))
        self.val_dataset = AnimeDataset(self.val_edges, val_labels.to(self.device)) if self.val_edges is not None else None
        self.test_dataset = AnimeDataset(self.test_edges, test_labels.to(self.device)) if self.test_edges is not None else None
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            self.val_loader = None
            
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            self.test_loader = None
            
        self.logger.info(f"训练数据集大小: {len(self.train_dataset)}")
        if self.val_dataset:
            self.logger.info(f"验证数据集大小: {len(self.val_dataset)}")
        if self.test_dataset:
            self.logger.info(f"测试数据集大小: {len(self.test_dataset)}")
    
    def train_epoch(self) -> float:
        """
        训练一个epoch
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc="训练中"):
            self.optimizer.zero_grad()
            
            # 获取用户和物品嵌入
            user_emb, item_emb = self.model(
                self.processed_data['train_interaction_matrix'].to(self.device),
                self.processed_data.get('comment_embeddings', None).to(self.device) if self.processed_data.get('comment_embeddings') is not None else None,
                self.processed_data.get('comment_embedding_to_item_idx', None)
            )
            
            # 准备批次数据
            edges = batch['edge']
            labels = batch['label']
            
            # 计算预测分数
            users = user_emb[edges[0]]
            items = item_emb[edges[1]]
            pred = torch.sum(users * items, dim=1)
            
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
            
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        评估模型性能
        
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
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估中"):
                edges = batch['edge']
                
                # 获取用户和物品嵌入
                user_emb, item_emb = self.model(
                    self.processed_data['train_interaction_matrix'].to(self.device),
                    self.processed_data.get('comment_embeddings', None).to(self.device) if self.processed_data.get('comment_embeddings') is not None else None,
                    self.processed_data.get('comment_embedding_to_item_idx', None)
                )
                
                # 获取当前批次的用户
                batch_users = torch.unique(edges[0])
                
                for user in batch_users:
                    # 获取用户的真实交互
                    user_interactions = edges[:, edges[0] == user]
                    true_items = user_interactions[1]
                    
                    # 计算用户对所有物品的预测分数
                    user_embed = user_emb[user].unsqueeze(0)
                    scores = torch.mm(user_embed, item_emb.t()).squeeze()
                    
                    # 将已交互物品的分数设置为负无穷（不参与推荐）
                    scores[true_items] = float('-inf')
                    
                    # 计算评估指标
                    auc = self.calculate_auc(scores, true_items)
                    mrr = self.calculate_mrr(scores, true_items)
                    ndcg_5 = self.calculate_ndcg(scores, true_items, k=5)
                    ndcg_10 = self.calculate_ndcg(scores, true_items, k=10)
                    
                    total_auc += auc
                    total_mrr += mrr
                    total_ndcg_5 += ndcg_5
                    total_ndcg_10 += ndcg_10
                    num_users += 1
        
        # 计算平均指标
        metrics = {
            'AUC': total_auc / num_users if num_users > 0 else 0,
            'MRR': total_mrr / num_users if num_users > 0 else 0,
            'NDCG@5': total_ndcg_5 / num_users if num_users > 0 else 0,
            'NDCG@10': total_ndcg_10 / num_users if num_users > 0 else 0
        }
        
        return metrics
    
    @staticmethod
    def calculate_auc(scores: torch.Tensor, positive_items: torch.Tensor) -> float:
        """计算AUC分数"""
        pos_scores = scores[positive_items]
        neg_scores = scores[scores != float('-inf')]  # 获取所有未交互物品的分数
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.0
            
        pos_scores = pos_scores.unsqueeze(1)
        neg_scores = neg_scores.unsqueeze(0)
        
        # 确保计算可行
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return 0.0
            
        return ((pos_scores > neg_scores).float().mean()).item()
    
    @staticmethod
    def calculate_mrr(scores: torch.Tensor, positive_items: torch.Tensor) -> float:
        """计算MRR分数"""
        # 获取正样本的排名
        _, indices = torch.sort(scores, descending=True)
        
        if len(positive_items) == 0:
            return 0.0
            
        ranks = torch.zeros_like(positive_items, dtype=torch.float)
        for i, item in enumerate(positive_items):
            rank_positions = (indices == item).nonzero()
            if len(rank_positions) > 0:
                ranks[i] = rank_positions[0].item() + 1
            else:
                ranks[i] = len(scores) + 1  # 最低排名
                
        return (1.0 / ranks.float()).mean().item()
    
    @staticmethod
    def calculate_ndcg(scores: torch.Tensor, positive_items: torch.Tensor, k: int) -> float:
        """计算NDCG@k分数"""
        if len(positive_items) == 0:
            return 0.0
            
        _, indices = torch.sort(scores, descending=True)
        topk_indices = indices[:k]
        
        dcg = 0
        idcg = 0
        for i, idx in enumerate(topk_indices):
            if idx in positive_items:
                dcg += 1 / np.log2(i + 2)
        
        for i in range(min(len(positive_items), k)):
            idcg += 1 / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0
    
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

# 使用示例
if __name__ == "__main__":
    import pickle
    import pandas as pd
    
    # 配置训练参数
    config = {
        'embed_dim': 64,
        'n_layers': 3,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 1024,
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