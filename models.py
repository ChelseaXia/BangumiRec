import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class AttentionFusion(nn.Module):
    """实现评论嵌入和物品嵌入的注意力融合"""
    
    def __init__(self, embed_dim):
        super(AttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        
        # 注意力权重计算
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        # 输出融合层
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, item_embeds, comment_embeds):
        """
        融合物品嵌入和评论嵌入
        
        Args:
            item_embeds: 物品嵌入，形状 [num_items, embed_dim]
            comment_embeds: 评论嵌入，形状 [num_items, embed_dim]
            
        Returns:
            融合后的物品嵌入
        """
        # 计算注意力权重
        Q = self.W_q(item_embeds)
        K = self.W_k(comment_embeds)
        V = self.W_v(comment_embeds)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 加权聚合评论特征
        attended_comments = torch.matmul(attention_weights, V)
        
        # 连接物品嵌入和加权评论嵌入
        concat_embeds = torch.cat([item_embeds, attended_comments], dim=-1)
        
        # 融合生成最终嵌入
        fused_embeds = self.fusion(concat_embeds)
        
        return fused_embeds


class LLMGNNRecommender(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, n_layers=3, dropout=0.1):
        """
        结合LightGCN和评论嵌入的推荐模型
        
        Args:
            num_users: 用户数量
            num_items: 物品数量
            embed_dim: 嵌入维度
            n_layers: 图卷积层数
            dropout: dropout比率
        """
        super(LLMGNNRecommender, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化用户和物品嵌入
        self.user_embeds = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(num_users, embed_dim))
        )
        self.item_embeds = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(num_items, embed_dim))
        )
        
        # 评论嵌入转换层 (假设评论嵌入是768维，常见的BERT/LLM输出维度)
        self.comment_transform = nn.Linear(768, embed_dim)
        
        # 注意力融合层
        self.attention = AttentionFusion(embed_dim)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, interaction_matrix, comment_embeddings=None):
        """
        前向传播
        
        Args:
            interaction_matrix: 用户-物品交互矩阵(稀疏格式)
            comment_embeddings: 评论嵌入，可选
            
        Returns:
            用户嵌入和物品嵌入
        """
        # 确保交互矩阵是稀疏格式
        if not interaction_matrix.is_sparse:
            self.logger.warning("交互矩阵不是稀疏格式，尝试转换...")
            interaction_matrix = interaction_matrix.to_sparse()
            
        # LightGCN部分: 图卷积传播
        embeds = torch.cat([self.user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]
        
        for layer in range(self.n_layers):
            # 消息传递
            embeddings = torch.sparse.mm(interaction_matrix, embeds_list[-1])
            # 应用dropout
            embeddings = self.dropout_layer(embeddings)
            embeds_list.append(embeddings)
        
        # 聚合所有层的嵌入
        all_embeddings = torch.stack(embeds_list, dim=0)
        all_embeddings = torch.mean(all_embeddings, dim=0)  # 使用平均而不是求和，避免数值过大
        
        # 分离用户和物品嵌入
        user_embeddings = all_embeddings[:self.num_users]
        item_embeddings = all_embeddings[self.num_users:]
        
        # 如果提供了评论嵌入，应用注意力融合
        if comment_embeddings is not None:
            # 处理评论嵌入维度兼容性
            if comment_embeddings.size(0) != self.num_items:
                self.logger.warning(f"评论嵌入数量({comment_embeddings.size(0)})与物品数量({self.num_items})不匹配，尝试调整...")
                if comment_embeddings.size(0) > self.num_items:
                    comment_embeddings = comment_embeddings[:self.num_items]
                else:
                    # 如果评论数量不足，填充零
                    padding = torch.zeros(self.num_items - comment_embeddings.size(0), comment_embeddings.size(1))
                    padding = padding.to(comment_embeddings.device)
                    comment_embeddings = torch.cat([comment_embeddings, padding], dim=0)
            
            # 转换评论嵌入维度
            transformed_comments = self.comment_transform(comment_embeddings)
            
            # 应用注意力融合机制
            item_embeddings = self.attention(item_embeddings, transformed_comments)
        
        return user_embeddings, item_embeddings
    
    def predict(self, users, items):
        """
        预测指定用户-物品对的得分
        
        Args:
            users: 用户索引
            items: 物品索引
            
        Returns:
            预测评分
        """
        user_embeds = self.user_embeds[users]
        item_embeds = self.item_embeds[items]
        return torch.sum(user_embeds * item_embeds, dim=1)
    
    def full_predict(self, users, interaction_matrix, comment_embeddings=None):
        """
        为指定用户预测所有物品的得分
        
        Args:
            users: 用户索引
            interaction_matrix: 用户-物品交互矩阵
            comment_embeddings: 评论嵌入，可选
            
        Returns:
            用户对所有物品的预测评分
        """
        user_embeds, item_embeds = self.forward(interaction_matrix, comment_embeddings)
        user_embeds = user_embeds[users]
        return torch.matmul(user_embeds, item_embeds.t())
    
    def cal_loss(self, users, pos_items, neg_items, interaction_matrix, comment_embeddings=None):
        """
        计算BPR损失
        
        Args:
            users: 用户索引
            pos_items: 正向物品索引
            neg_items: 负向物品索引
            interaction_matrix: 用户-物品交互矩阵
            comment_embeddings: 评论嵌入，可选
            
        Returns:
            BPR损失
        """
        user_embeds, item_embeds = self.forward(interaction_matrix, comment_embeddings)
        
        user_embeds = user_embeds[users]
        pos_embeds = item_embeds[pos_items]
        neg_embeds = item_embeds[neg_items]
        
        # 计算正样本和负样本得分
        pos_scores = torch.sum(user_embeds * pos_embeds, dim=1)
        neg_scores = torch.sum(user_embeds * neg_embeds, dim=1)
        
        # 计算BPR损失 (正样本得分应高于负样本)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # L2正则化
        reg_loss = 0.001 * (
            user_embeds.norm(2).pow(2) +
            pos_embeds.norm(2).pow(2) +
            neg_embeds.norm(2).pow(2)
        )
        
        return loss + reg_loss