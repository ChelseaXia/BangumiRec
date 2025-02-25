import torch
import logging
import pickle
import os
import numpy as np
from typing import List, Dict, Tuple, Any

class AnimeRecommender:
    def __init__(self, model, processed_data, device):
        """
        动画推荐器
        
        Args:
            model: 训练好的LightGCN模型
            processed_data: 预处理后的数据
            device: 计算设备
        """
        self.model = model.to(device)
        self.processed_data = processed_data
        self.device = device
        self.model.eval()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 构建反向映射（索引到ID）
        self.user_idx_to_id = {v: k for k, v in self.processed_data.get('user_id_map', {}).items()}
        self.item_idx_to_id = {v: k for k, v in self.processed_data.get('item_id_map', {}).items()}
        
        # 加载动画元数据（如果可用）
        self.anime_metadata = {}
        if 'anime_data' in processed_data:
            self.anime_metadata = processed_data['anime_data']
        
        # 如果有的话，预加载交互边数据
        self.user_interactions = self._preload_user_interactions()
        
    def _preload_user_interactions(self) -> Dict[int, List[int]]:
        """预加载用户交互数据，加速推荐"""
        user_interactions = {}
        
        # 检查边数据存在性
        if 'train_edges' in self.processed_data:
            edges = self.processed_data['train_edges']
            for edge_idx in range(edges.size(1)):
                user_idx = edges[0, edge_idx].item()
                item_idx = edges[1, edge_idx].item()
                
                if user_idx not in user_interactions:
                    user_interactions[user_idx] = []
                user_interactions[user_idx].append(item_idx)
        
        # 如果没有预加载的边数据，尝试从交互矩阵获取
        elif 'train_interaction_matrix' in self.processed_data:
            matrix = self.processed_data['train_interaction_matrix']
            if matrix.is_sparse:
                indices = matrix._indices()
                for i in range(indices.size(1)):
                    user_idx = indices[0, i].item()
                    item_idx = indices[1, i].item()
                    
                    if user_idx not in user_interactions:
                        user_interactions[user_idx] = []
                    user_interactions[user_idx].append(item_idx)
        
        self.logger.info(f"预加载了 {len(user_interactions)} 个用户的交互数据")
        return user_interactions
        
    def get_recommendations(self, user_id: int, top_k: int = 10) -> List[Dict]:
        """
        为指定用户获取推荐
        
        Args:
            user_id: 用户ID
            top_k: 推荐数量
            
        Returns:
            推荐列表，每个推荐包含动画ID和预测分数
        """
        with torch.no_grad():
            # 获取用户索引
            user_idx = self.processed_data['user_id_map'].get(user_id)
            if user_idx is None:
                raise ValueError(f"User ID {user_id} not found in training data")
                
            # 获取用户和物品嵌入
            user_emb, item_emb = self.model(
                self.processed_data['train_interaction_matrix'].to(self.device),
                self.processed_data.get('comment_embeddings', None).to(self.device) if self.processed_data.get('comment_embeddings') is not None else None,
                self.processed_data.get('comment_embedding_to_item_idx', None)
            )
            
            # 计算用户对所有物品的预测分数
            user_embed = user_emb[user_idx].unsqueeze(0)
            scores = torch.mm(user_embed, item_emb.t()).squeeze()
            
            # 获取用户已交互的物品
            interacted_items = self.user_interactions.get(user_idx, [])
            
            # 将已交互物品的分数设为负无穷
            scores[interacted_items] = float('-inf')
            
            # 获取top-k推荐
            top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores) - len(interacted_items)))
            
            # 转换回原始物品ID并添加元数据
            recommendations = []
            for idx, score in zip(top_indices.cpu().numpy(), top_scores.cpu().numpy()):
                item_id = self.item_idx_to_id.get(idx.item())
                
                # 创建推荐项
                rec_item = {
                    'anime_id': item_id,
                    'score': float(score)
                }
                
                # 添加元数据（如果可用）
                if isinstance(self.anime_metadata, dict) and item_id in self.anime_metadata:
                    anime_info = self.anime_metadata[item_id]
                    rec_item['title'] = anime_info.get('title', '')
                    rec_item['score'] = anime_info.get('score', rec_item['score'])
                    rec_item['genres'] = anime_info.get('genres', '')
                
                recommendations.append(rec_item)
            
            return recommendations
    
    def get_similar_items(self, item_id: int, top_k: int = 10) -> List[Dict]:
        """
        获取相似动画推荐
        
        Args:
            item_id: 动画ID
            top_k: 推荐数量
            
        Returns:
            相似动画列表
        """
        with torch.no_grad():
            # 获取物品索引
            item_idx = self.processed_data['item_id_map'].get(item_id)
            if item_idx is None:
                raise ValueError(f"Item ID {item_id} not found in training data")
            
            # 获取物品嵌入
            _, item_emb = self.model(
                self.processed_data['train_interaction_matrix'].to(self.device),
                self.processed_data.get('comment_embeddings', None).to(self.device) if self.processed_data.get('comment_embeddings') is not None else None,
                self.processed_data.get('comment_embedding_to_item_idx', None)
            )
            
            # 计算与所有物品的相似度
            item_embed = item_emb[item_idx].unsqueeze(0)
            similarities = torch.mm(item_embed, item_emb.t()).squeeze()
            
            # 将自身的相似度设为负无穷
            similarities[item_idx] = float('-inf')
            
            # 获取top-k相似物品
            top_similarities, top_indices = torch.topk(similarities, k=top_k)
            
            # 转换回原始物品ID并添加元数据
            similar_items = []
            for idx, similarity in zip(top_indices.cpu().numpy(), top_similarities.cpu().numpy()):
                similar_id = self.item_idx_to_id.get(idx.item())
                
                # 创建相似项
                similar_item = {
                    'anime_id': similar_id,
                    'similarity': float(similarity)
                }
                
                # 添加元数据（如果可用）
                if isinstance(self.anime_metadata, dict) and similar_id in self.anime_metadata:
                    anime_info = self.anime_metadata[similar_id]
                    similar_item['title'] = anime_info.get('title', '')
                    similar_item['genres'] = anime_info.get('genres', '')
                
                similar_items.append(similar_item)
            
            return similar_items

    def explain_recommendation(self, user_id: int, anime_id: int) -> Dict:
        """
        解释推荐结果
        
        Args:
            user_id: 用户ID
            anime_id: 动画ID
            
        Returns:
            推荐解释字典
        """
        with torch.no_grad():
            # 获取用户和物品索引
            user_idx = self.processed_data['user_id_map'].get(user_id)
            item_idx = self.processed_data['item_id_map'].get(anime_id)
            
            if user_idx is None or item_idx is None:
                raise ValueError("User ID or Anime ID not found in training data")
            
            # 获取用户和物品嵌入
            user_emb, item_emb = self.model(
                self.processed_data['train_interaction_matrix'].to(self.device),
                self.processed_data.get('comment_embeddings', None).to(self.device) if self.processed_data.get('comment_embeddings') is not None else None,
                self.processed_data.get('comment_embedding_to_item_idx', None)
            )
            
            # 获取用户和物品的嵌入向量
            user_embed = user_emb[user_idx]
            item_embed = item_emb[item_idx]
            
            # 计算预测分数
            score = torch.dot(user_embed, item_embed).item()
            
            # 获取用户观看历史
            user_history = self.user_interactions.get(user_idx, [])
            
            # 获取用户观看历史中的相似动画
            similar_watched = []
            for hist_item_idx in user_history:
                hist_item_embed = item_emb[hist_item_idx]
                similarity = torch.dot(item_embed, hist_item_embed).item()
                
                if similarity > 0.5:  # 设置相似度阈值
                    hist_item_id = self.item_idx_to_id.get(hist_item_idx)
                    similar_item = {
                        'anime_id': hist_item_id,
                        'similarity': similarity
                    }
                    
                    # 添加元数据（如果可用）
                    if isinstance(self.anime_metadata, dict) and hist_item_id in self.anime_metadata:
                        anime_info = self.anime_metadata[hist_item_id]
                        similar_item['title'] = anime_info.get('title', '')
                    
                    similar_watched.append(similar_item)
            
            # 对相似动画按相似度排序
            similar_watched.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 创建解释
            explanation = {
                'predicted_score': score,
                'similar_watched_anime': similar_watched[:5]  # 返回最相似的5部动画
            }
            
            # 添加当前动画信息（如果可用）
            if isinstance(self.anime_metadata, dict) and anime_id in self.anime_metadata:
                anime_info = self.anime_metadata[anime_id]
                explanation['anime_title'] = anime_info.get('title', '')
                explanation['anime_genres'] = anime_info.get('genres', '')
            
            # 添加评论信息（如果可用）
            if self.processed_data.get('comment_embedding_to_item_idx') and anime_id in self.anime_metadata:
                comment_indices = []
                for comment_idx, item_idx_mapped in self.processed_data['comment_embedding_to_item_idx'].items():
                    if item_idx_mapped == item_idx:
                        comment_indices.append(comment_idx)
                
                if comment_indices:
                    explanation['has_comment_embedding'] = True
                    explanation['comment_indices'] = comment_indices
            
            return explanation
    
    def batch_recommend(self, user_ids: List[int], top_k: int = 10) -> Dict[int, List[Dict]]:
        """
        批量为多个用户生成推荐
        
        Args:
            user_ids: 用户ID列表
            top_k: 每个用户的推荐数量
            
        Returns:
            用户ID到推荐列表的映射
        """
        recommendations = {}
        for user_id in user_ids:
            try:
                user_recs = self.get_recommendations(user_id, top_k)
                recommendations[user_id] = user_recs
            except ValueError as e:
                self.logger.warning(f"为用户 {user_id} 生成推荐时出错: {str(e)}")
                recommendations[user_id] = []
        
        return recommendations
    
    def analyze_comment_influence(self, item_id: int) -> Dict:
        """
        分析评论对物品嵌入的影响
        
        Args:
            item_id: 物品ID
            
        Returns:
            影响分析结果
        """
        with torch.no_grad():
            # 获取物品索引
            item_idx = self.processed_data['item_id_map'].get(item_id)
            if item_idx is None:
                raise ValueError(f"Item ID {item_id} not found in training data")
            
            # 先获取没有评论嵌入的物品嵌入
            _, item_emb_without_comments = self.model(
                self.processed_data['train_interaction_matrix'].to(self.device),
                None,
                None
            )
            
            # 再获取有评论嵌入的物品嵌入
            _, item_emb_with_comments = self.model(
                self.processed_data['train_interaction_matrix'].to(self.device),
                self.processed_data.get('comment_embeddings', None).to(self.device) if self.processed_data.get('comment_embeddings') is not None else None,
                self.processed_data.get('comment_embedding_to_item_idx', None)
            )
            
            # 获取特定物品的嵌入
            item_embed_without = item_emb_without_comments[item_idx]
            item_embed_with = item_emb_with_comments[item_idx]
            
            # 计算余弦相似度
            cos_sim = torch.nn.functional.cosine_similarity(
                item_embed_without.unsqueeze(0),
                item_embed_with.unsqueeze(0)
            ).item()
            
            # 计算欧几里得距离
            euclidean_dist = torch.norm(item_embed_without - item_embed_with).item()
            
            # 检查是否有评论映射到这个物品
            has_comment_mapping = False
            comment_indices = []
            if self.processed_data.get('comment_embedding_to_item_idx'):
                for comment_idx, mapped_item_idx in self.processed_data['comment_embedding_to_item_idx'].items():
                    if mapped_item_idx == item_idx:
                        has_comment_mapping = True
                        comment_indices.append(comment_idx)
            
            return {
                'has_comment_mapping': has_comment_mapping,
                'comment_indices': comment_indices,
                'cosine_similarity': cos_sim,
                'euclidean_distance': euclidean_dist,
                'embedding_changed': cos_sim < 0.99  # 假设相似度小于0.99表示显著变化
            }
    
    @staticmethod
    def load_from_directory(model_path: str, data_dir: str, device=None):
        """
        从目录加载推荐器
        
        Args:
            model_path: 模型路径
            data_dir: 数据目录
            device: 计算设备
            
        Returns:
            加载的推荐器实例
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # 加载模型
        from models import LLMGNNRecommender
        
        # 加载数据
        processed_data = {}
        
        # 加载ID映射
        id_mappings_path = os.path.join(data_dir, 'id_mappings.pkl')
        if os.path.exists(id_mappings_path):
            with open(id_mappings_path, 'rb') as f:
                id_mappings = pickle.load(f)
                processed_data['user_id_map'] = id_mappings['user_id_map']
                processed_data['item_id_map'] = id_mappings['item_id_map']
                processed_data['num_users'] = len(id_mappings['user_id_map'])
                processed_data['num_items'] = len(id_mappings['item_id_map'])
        
        # 加载交互矩阵组件
        matrix_path = os.path.join(data_dir, 'interaction_matrix_components.pkl')
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
        comment_path = os.path.join(data_dir, 'comment_embeddings.pkl')
        if os.path.exists(comment_path):
            with open(comment_path, 'rb') as f:
                comment_embeddings_list = pickle.load(f)
                # 合并评论嵌入
                if isinstance(comment_embeddings_list, list):
                    if all(isinstance(item, dict) and 'embeddings' in item for item in comment_embeddings_list):
                        # 新格式：包含嵌入和物品ID
                        all_embeddings = []
                        all_subject_ids = []
                        for batch_data in comment_embeddings_list:
                            all_embeddings.append(batch_data['embeddings'])
                            all_subject_ids.extend(batch_data['subject_ids'])
                        
                        comment_embeddings = np.concatenate(all_embeddings, axis=0)
                        processed_data['comment_embeddings'] = torch.tensor(comment_embeddings)
                        
                        # 创建评论到物品索引的映射
                        comment_embedding_to_item_idx = {}
                        for i, subject_id in enumerate(all_subject_ids):
                            if subject_id in processed_data['item_id_map']:
                                item_idx = processed_data['item_id_map'][subject_id]
                                comment_embedding_to_item_idx[i] = item_idx
                        
                        processed_data['comment_embedding_to_item_idx'] = comment_embedding_to_item_idx
                    else:
                        # 旧格式：只有嵌入列表
                        processed_data['comment_embeddings'] = torch.tensor(
                            np.concatenate(comment_embeddings_list, axis=0)
                        )
                else:
                    processed_data['comment_embeddings'] = comment_embeddings_list
        
        # 加载评论到物品的映射（如果存在）
        comment_map_path = os.path.join(data_dir, 'comment_to_item_map.pkl')
        if os.path.exists(comment_map_path):
            with open(comment_map_path, 'rb') as f:
                comment_to_item_map = pickle.load(f)
                
                # 创建评论嵌入到物品索引的映射
                if 'comment_embedding_to_item_idx' not in processed_data:
                    comment_embedding_to_item_idx = {}
                    for comment_idx, subject_id in comment_to_item_map.items():
                        if subject_id in processed_data['item_id_map']:
                            item_idx = processed_data['item_id_map'][subject_id]
                            comment_embedding_to_item_idx[comment_idx] = item_idx
                    
                    processed_data['comment_embedding_to_item_idx'] = comment_embedding_to_item_idx
        
        # 加载动画数据
        anime_data_path = os.path.join(data_dir, 'anime_data.pkl')
        if os.path.exists(anime_data_path):
            with open(anime_data_path, 'rb') as f:
                processed_data['anime_data'] = pickle.load(f)
        
        # 创建模型
        model = LLMGNNRecommender(
            num_users=processed_data['num_users'],
            num_items=processed_data['num_items'],
            embed_dim=64  # 使用默认值，可以根据实际情况调整
        )
        
        # 加载模型参数
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 创建推荐器
        recommender = AnimeRecommender(model, processed_data, device)
        
        return recommender

def main():
    # 加载推荐器
    recommender = AnimeRecommender.load_from_directory(
        model_path='processed_data/best_model.pth',
        data_dir='processed_data'
    )
    
    # 为用户生成推荐
    try:
        # 选择第一个用户
        first_user = next(iter(recommender.processed_data['user_id_map'].keys()))
        recommendations = recommender.get_recommendations(first_user, top_k=10)
        
        print(f"为用户 {first_user} 的推荐:")
        for i, rec in enumerate(recommendations, 1):
            title = rec.get('title', f"Anime #{rec['anime_id']}")
            print(f"{i}. {title} (分数: {rec['score']:.4f})")
        
        # 如果有推荐结果，为第一个推荐项生成解释
        if recommendations:
            first_rec = recommendations[0]
            explanation = recommender.explain_recommendation(first_user, first_rec['anime_id'])
            
            print("\n推荐解释:")
            print(f"预测分数: {explanation['predicted_score']:.4f}")
            
            # 检查是否有评论影响
            if explanation.get('has_comment_embedding'):
                print(f"该动画有对应的评论嵌入，评论索引: {explanation.get('comment_indices')}")
            
            print("用户观看过的相似动画:")
            for i, similar in enumerate(explanation.get('similar_watched_anime', []), 1):
                title = similar.get('title', f"Anime #{similar['anime_id']}")
                print(f"{i}. {title} (相似度: {similar['similarity']:.4f})")
                
            # 分析评论影响
            influence = recommender.analyze_comment_influence(first_rec['anime_id'])
            if influence['has_comment_mapping']:
                print("\n评论影响分析:")
                print(f"该动画有 {len(influence['comment_indices'])} 条评论映射")
                print(f"评论对嵌入的影响 - 余弦相似度: {influence['cosine_similarity']:.4f}")
                print(f"评论是否显著改变了嵌入: {'是' if influence['embedding_changed'] else '否'}")
                
    except Exception as e:
        print(f"生成推荐时出错: {str(e)}")

if __name__ == "__main__":
    main()
