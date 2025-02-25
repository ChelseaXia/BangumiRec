import torch
import numpy as np
import pandas as pd
import pickle
import os
import logging
import random
from typing import Dict, List, Tuple
import sys

# 添加父目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型和训练器类
from models import LLMGNNRecommender
from model_trainer import LightGCNTrainer
from recommender import AnimeRecommender

def create_mini_dataset(processed_data_dir: str, output_dir: str, sample_size: Dict[str, int]) -> Dict:
    """
    从处理好的数据中采样创建小规模测试数据集
    
    Args:
        processed_data_dir: 处理好的数据目录
        output_dir: 输出目录
        sample_size: 包含各部分采样大小的字典
            {
                'users': 用户采样数量,
                'items': 物品采样数量,
                'interactions': 交互采样数量
            }
    
    Returns:
        小规模数据集字典
    """
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"从 {processed_data_dir} 加载数据并创建小规模数据集")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载ID映射
    logger.info("加载ID映射...")
    id_mapping_path = os.path.join(processed_data_dir, 'id_mappings.pkl')
    if not os.path.exists(id_mapping_path):
        raise FileNotFoundError(f"找不到ID映射文件: {id_mapping_path}")
        
    with open(id_mapping_path, 'rb') as f:
        id_mappings = pickle.load(f)
        
    user_id_map = id_mappings['user_id_map']
    item_id_map = id_mappings['item_id_map']
    
    # 采样用户和物品
    sampled_users = random.sample(
        list(user_id_map.keys()),
        min(sample_size['users'], len(user_id_map))
    )
    sampled_items = random.sample(
        list(item_id_map.keys()),
        min(sample_size['items'], len(item_id_map))
    )
    
    logger.info(f"采样了 {len(sampled_users)} 个用户和 {len(sampled_items)} 个物品")
    
    # 创建新的ID映射
    new_user_id_map = {uid: idx for idx, uid in enumerate(sampled_users)}
    new_item_id_map = {iid: idx for idx, iid in enumerate(sampled_items)}
    
    # 保存新的ID映射
    new_id_mappings = {
        'user_id_map': new_user_id_map,
        'item_id_map': new_item_id_map,
        'matrix_size': (len(new_user_id_map), len(new_item_id_map))
    }
    
    with open(os.path.join(output_dir, 'id_mappings.pkl'), 'wb') as f:
        pickle.dump(new_id_mappings, f)
    
    # 加载交互矩阵组件
    logger.info("加载交互矩阵组件...")
    matrix_path = os.path.join(processed_data_dir, 'interaction_matrix_components.pkl')
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"找不到交互矩阵组件文件: {matrix_path}")
        
    with open(matrix_path, 'rb') as f:
        matrix_components = pickle.load(f)
    
    indices = matrix_components['indices']
    values = matrix_components['values']
    
    # 过滤并重新映射交互
    logger.info("过滤并重新映射交互...")
    filtered_interactions = []
    filtered_values = []
    
    # 反转ID映射用于查找
    orig_user_idx_to_id = {v: k for k, v in user_id_map.items()}
    orig_item_idx_to_id = {v: k for k, v in item_id_map.items()}
    
    # 筛选交互
    for i in range(indices.shape[1]):
        user_idx = indices[0, i]
        item_idx = indices[1, i]
        
        # 获取原始ID
        user_id = orig_user_idx_to_id.get(user_idx)
        item_id = orig_item_idx_to_id.get(item_idx)
        
        # 检查是否在采样集中
        if user_id in new_user_id_map and item_id in new_item_id_map:
            filtered_interactions.append([
                new_user_id_map[user_id],
                new_item_id_map[item_id]
            ])
            filtered_values.append(values[i])
            
            # 如果达到采样数量，停止
            if len(filtered_interactions) >= sample_size['interactions']:
                break
    
    # 如果没有足够的交互，随机添加一些
    if len(filtered_interactions) < sample_size['interactions']:
        logger.warning(f"只找到 {len(filtered_interactions)} 个交互，少于请求的 {sample_size['interactions']} 个")
        
        # 随机添加一些交互
        remaining = sample_size['interactions'] - len(filtered_interactions)
        for _ in range(remaining):
            user_idx = random.randrange(len(new_user_id_map))
            item_idx = random.randrange(len(new_item_id_map))
            
            filtered_interactions.append([user_idx, item_idx])
            filtered_values.append(random.randint(1, 5))  # 随机评分1-5
    
    # 创建小规模交互矩阵
    filtered_indices = np.array(filtered_interactions).T
    filtered_values = np.array(filtered_values)
    
    # 保存交互矩阵组件
    new_matrix_components = {
        'indices': filtered_indices,
        'values': filtered_values,
        'size': (len(new_user_id_map), len(new_item_id_map))
    }
    
    with open(os.path.join(output_dir, 'interaction_matrix_components.pkl'), 'wb') as f:
        pickle.dump(new_matrix_components, f)
    
    # 构建稀疏张量
    logger.info("构建稀疏交互矩阵...")
    mini_interaction_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor(filtered_indices),
        values=torch.tensor(filtered_values, dtype=torch.float),
        size=(len(new_user_id_map), len(new_item_id_map))
    )
    
    # 处理评论嵌入和映射
    logger.info("处理评论嵌入和映射...")
    sampled_comment_embeddings = None
    comment_embedding_to_item_idx = {}
    
    # 检查是否有评论到物品的映射
    comment_map_path = os.path.join(processed_data_dir, 'comment_to_item_map.pkl')
    has_comment_map = os.path.exists(comment_map_path)
    
    # 加载并处理评论嵌入
    comment_path = os.path.join(processed_data_dir, 'comment_embeddings.pkl')
    if os.path.exists(comment_path):
        with open(comment_path, 'rb') as f:
            comment_data_list = pickle.load(f)
            
            # 处理评论嵌入和对应的物品ID
            all_embeddings = []
            all_subject_ids = []
            
            # 检查评论数据格式并提取嵌入和物品ID
            for batch_data in comment_data_list:
                if isinstance(batch_data, dict) and 'embeddings' in batch_data and 'subject_ids' in batch_data:
                    # 新格式：包含嵌入和物品ID
                    all_embeddings.append(batch_data['embeddings'])
                    all_subject_ids.extend(batch_data['subject_ids'])
                else:
                    # 旧格式：只有嵌入
                    all_embeddings.append(batch_data)
            
            # 合并评论嵌入
            all_comment_embeddings = np.concatenate(all_embeddings, axis=0)
            
            # 如果有物品ID，创建映射
            if all_subject_ids:
                # 创建旧物品ID到新物品索引的映射
                old_item_id_to_new_idx = {old_id: new_idx for old_id, new_idx in new_item_id_map.items()}
                
                # 为每个评论找到对应的新物品索引
                for i, subject_id in enumerate(all_subject_ids):
                    if subject_id in old_item_id_to_new_idx:
                        comment_embedding_to_item_idx[i] = old_item_id_to_new_idx[subject_id]
            
            # 如果有评论到物品映射但没有通过评论数据获取到，尝试从映射文件加载
            if not comment_embedding_to_item_idx and has_comment_map:
                with open(comment_map_path, 'rb') as f:
                    comment_to_item_map = pickle.load(f)
                    
                    # 创建评论索引到新物品索引的映射
                    for comment_idx, subject_id in comment_to_item_map.items():
                        if subject_id in old_item_id_to_new_idx:
                            comment_embedding_to_item_idx[comment_idx] = old_item_id_to_new_idx[subject_id]
            
            # 采样评论嵌入
            if len(all_comment_embeddings) > 0:
                sampled_comment_embeddings = all_comment_embeddings
                logger.info(f"加载了 {len(all_comment_embeddings)} 个评论嵌入和 {len(comment_embedding_to_item_idx)} 个映射")
            else:
                logger.warning("无法加载评论嵌入，将创建随机嵌入")
                sampled_comment_embeddings = np.random.randn(len(new_item_id_map), 768)  # 模拟BERT嵌入维度
    else:
        logger.warning(f"找不到评论嵌入文件: {comment_path}，创建随机嵌入")
        # 创建随机嵌入
        sampled_comment_embeddings = np.random.randn(len(new_item_id_map), 768)
    
    # 保存评论嵌入和映射
    with open(os.path.join(output_dir, 'comment_embeddings.pkl'), 'wb') as f:
        pickle.dump([sampled_comment_embeddings], f)
    
    with open(os.path.join(output_dir, 'comment_embedding_to_item_idx.pkl'), 'wb') as f:
        pickle.dump(comment_embedding_to_item_idx, f)
    
    # 创建边数据
    logger.info("创建边数据...")
    train_edges = torch.tensor(filtered_indices)
    
    # 分割为训练、验证和测试集
    n_edges = train_edges.size(1)
    indices = torch.randperm(n_edges)
    
    train_size = int(n_edges * 0.8)
    val_size = int(n_edges * 0.1)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    
    train_edges_data = {
        'user_indices': train_edges[0, train_idx].numpy(),
        'item_indices': train_edges[1, train_idx].numpy()
    }
    
    val_edges_data = {
        'user_indices': train_edges[0, val_idx].numpy(),
        'item_indices': train_edges[1, val_idx].numpy()
    }
    
    test_edges_data = {
        'user_indices': train_edges[0, test_idx].numpy(),
        'item_indices': train_edges[1, test_idx].numpy()
    }
    
    # 保存边数据
    with open(os.path.join(output_dir, 'train_edges.pkl'), 'wb') as f:
        pickle.dump(train_edges_data, f)
        
    with open(os.path.join(output_dir, 'val_edges.pkl'), 'wb') as f:
        pickle.dump(val_edges_data, f)
        
    with open(os.path.join(output_dir, 'test_edges.pkl'), 'wb') as f:
        pickle.dump(test_edges_data, f)
    
    # 加载和采样动画数据
    logger.info("处理动画数据...")
    anime_data = None
    anime_path = os.path.join(processed_data_dir, 'anime_data.pkl')
    
    if os.path.exists(anime_path):
        with open(anime_path, 'rb') as f:
            full_anime_data = pickle.load(f)
            
            # 过滤只保留采样的物品
            if isinstance(full_anime_data, pd.DataFrame):
                anime_data = full_anime_data[full_anime_data.index.isin(sampled_items)]
            elif isinstance(full_anime_data, dict):
                anime_data = {k: v for k, v in full_anime_data.items() if k in sampled_items}
                
        # 保存动画数据
        with open(os.path.join(output_dir, 'anime_data.pkl'), 'wb') as f:
            pickle.dump(anime_data, f)
    
    # 保存评论到物品的映射
    if has_comment_map:
        with open(comment_map_path, 'rb') as f:
            original_comment_to_item_map = pickle.load(f)
            
        # 过滤只保留采样的物品
        filtered_comment_to_item_map = {
            k: v for k, v in original_comment_to_item_map.items() 
            if v in sampled_items
        }
        
        with open(os.path.join(output_dir, 'comment_to_item_map.pkl'), 'wb') as f:
            pickle.dump(filtered_comment_to_item_map, f)
    
    # 保存元数据
    metadata = {
        'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_users': len(new_user_id_map),
        'num_items': len(new_item_id_map),
        'files': os.listdir(output_dir)
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    # 构建mini数据集
    mini_dataset = {
        'train_interaction_matrix': mini_interaction_matrix,
        'comment_embeddings': torch.tensor(sampled_comment_embeddings),
        'comment_embedding_to_item_idx': comment_embedding_to_item_idx,
        'train_edges': torch.stack([
            torch.tensor(train_edges_data['user_indices']),
            torch.tensor(train_edges_data['item_indices'])
        ]),
        'val_edges': torch.stack([
            torch.tensor(val_edges_data['user_indices']),
            torch.tensor(val_edges_data['item_indices'])
        ]),
        'test_edges': torch.stack([
            torch.tensor(test_edges_data['user_indices']),
            torch.tensor(test_edges_data['item_indices'])
        ]),
        'user_id_map': new_user_id_map,
        'item_id_map': new_item_id_map,
        'num_users': len(new_user_id_map),
        'num_items': len(new_item_id_map),
        'anime_data': anime_data,
        'output_dir': output_dir
    }
    
    logger.info(f"小规模数据集创建完成，保存到 {output_dir}")
    
    return mini_dataset


def run_mini_pipeline(processed_data_dir: str, mini_data_dir: str = "mini_data"):
    """
    运行小规模测试pipeline
    
    Args:
        processed_data_dir: 处理好的数据目录
        mini_data_dir: 小规模数据输出目录
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(levelname)s - %(message)s',
                      force=True)
    logger = logging.getLogger(__name__)
    
    # 确保数据目录存在
    if not os.path.exists(processed_data_dir):
        raise FileNotFoundError(f"找不到处理数据目录: {processed_data_dir}")
    
    logger.info(f"开始小规模测试 pipeline，使用数据源: {processed_data_dir}")
    
    # 创建小规模数据集
    sample_size = {
        'users': 100,    # 采样100个用户
        'items': 50,     # 采样50个物品
        'interactions': 1000  # 采样1000个交互
    }
    
    try:
        logger.info("创建小规模数据集...")
        mini_dataset = create_mini_dataset(processed_data_dir, mini_data_dir, sample_size)
        
        # 配置训练参数
        config = {
            'embed_dim': 32,          # 降低嵌入维度
            'n_layers': 2,            # 减少层数
            'dropout': 0.1,           # 设置dropout
            'learning_rate': 0.001,
            'batch_size': 128,        # 减小批次大小
            'num_epochs': 5,          # 减少训练轮数
            'patience': 2             # 减少早停耐心值
        }
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        
        # 创建模型实例
        logger.info("创建模型...")
        model = LLMGNNRecommender(
            num_users=mini_dataset['num_users'],
            num_items=mini_dataset['num_items'],
            embed_dim=config['embed_dim'],
            n_layers=config['n_layers'],
            dropout=config['dropout']
        )
        
        # 创建训练器
        logger.info("初始化训练器...")
        trainer = LightGCNTrainer(
            model=model,
            processed_data=mini_dataset,
            device=device,
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            num_epochs=config['num_epochs'],
            early_stopping_patience=config['patience']
        )
        
        # 开始训练
        logger.info("开始小规模训练...")
        training_history = trainer.train()
        
        # 保存训练历史
        with open(os.path.join(mini_data_dir, 'training_history.pkl'), 'wb') as f:
            pickle.dump(training_history, f)
        
        # 测试推荐功能
        logger.info("测试推荐功能...")
        recommender = AnimeRecommender(model, mini_dataset, device)
        
        # 随机选择一个用户进行推荐测试
        test_users = random.sample(list(mini_dataset['user_id_map'].keys()), min(5, len(mini_dataset['user_id_map'])))
        
        for test_user in test_users:
            try:
                recommendations = recommender.get_recommendations(test_user, top_k=5)
                
                logger.info(f"\n为用户 {test_user} 的推荐:")
                for i, rec in enumerate(recommendations, 1):
                    title = "未知标题"
                    if 'title' in rec:
                        title = rec['title']
                    elif mini_dataset.get('anime_data') is not None:
                        if isinstance(mini_dataset['anime_data'], pd.DataFrame) and rec['anime_id'] in mini_dataset['anime_data'].index:
                            title = mini_dataset['anime_data'].loc[rec['anime_id']].get('title', f"Anime #{rec['anime_id']}")
                        elif isinstance(mini_dataset['anime_data'], dict) and rec['anime_id'] in mini_dataset['anime_data']:
                            title = mini_dataset['anime_data'][rec['anime_id']].get('title', f"Anime #{rec['anime_id']}")
                    
                    logger.info(f"{i}. {title} (分数: {rec['score']:.4f})")
                
                # 为第一个推荐生成解释
                if recommendations:
                    try:
                        first_rec = recommendations[0]
                        explanation = recommender.explain_recommendation(test_user, first_rec['anime_id'])
                        
                        logger.info(f"\n推荐 '{title}' 的解释:")
                        logger.info(f"预测分数: {explanation['predicted_score']:.4f}")
                        
                        # 显示评论映射信息
                        if 'has_comment_embedding' in explanation and explanation['has_comment_embedding']:
                            logger.info(f"该动画有对应的评论嵌入，评论索引: {explanation.get('comment_indices')}")
                        
                        if 'similar_watched_anime' in explanation and explanation['similar_watched_anime']:
                            logger.info("用户观看过的相似动画:")
                            for i, similar in enumerate(explanation['similar_watched_anime'], 1):
                                similar_title = "未知标题"
                                if 'title' in similar:
                                    similar_title = similar['title']
                                
                                logger.info(f"{i}. {similar_title} (相似度: {similar['similarity']:.4f})")
                        else:
                            logger.info("没有找到用户观看过的相似动画")
                            
                        # 分析评论影响
                        try:
                            influence = recommender.analyze_comment_influence(first_rec['anime_id'])
                            if influence['has_comment_mapping']:
                                logger.info("\n评论影响分析:")
                                logger.info(f"该动画有 {len(influence['comment_indices'])} 条评论映射")
                                logger.info(f"评论对嵌入的影响 - 余弦相似度: {influence['cosine_similarity']:.4f}")
                                logger.info(f"评论是否显著改变了嵌入: {'是' if influence['embedding_changed'] else '否'}")
                        except AttributeError:
                            logger.info("评论影响分析功能未实现")
                        
                    except Exception as e:
                        logger.error(f"生成解释时出错: {str(e)}")
            except Exception as e:
                logger.error(f"为用户 {test_user} 生成推荐时出错: {str(e)}")
        
        logger.info("\n小规模 pipeline 执行完成!")
        return model, training_history, recommender
        
    except Exception as e:
        logger.error(f"执行 pipeline 时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def main():
    # 检查命令行参数
    if len(sys.argv) > 1:
        processed_data_dir = sys.argv[1]
    else:
        processed_data_dir = "processed_data"
    
    mini_data_dir = "mini_data"
    
    # 运行mini pipeline
    model, history, recommender = run_mini_pipeline(processed_data_dir, mini_data_dir)
    print("\nMini pipeline 执行成功!")


if __name__ == "__main__":
    main()