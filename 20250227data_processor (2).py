import os
import pandas as pd
import numpy as np
import torch
import logging
import pickle
import json
from typing import Dict, Tuple, List, Any
import re
import gc
from datetime import datetime
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from torch.cuda.amp import autocast
from tqdm import tqdm
import psutil
import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

default_config = None  # 配置应从config.py导入
try:
    from config import default_config
except ImportError:
    logging.warning("未找到config.py，使用默认数据路径")

# 配置数据路径
data_dir = getattr(default_config, 'data_dir', '/mnt/RecLLM/bangumi') if default_config else '/mnt/RecLLM/bangumi'
processed_data_dir = getattr(default_config, 'processed_data_dir', 'processed_data') if default_config else 'processed_data'

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转换为MB

class AnimeDataset(Dataset):
    """Enhanced Anime Dataset with GPU support"""
    def __init__(self, edges, labels, device='cuda'):
        self.edges = edges.to(device)  
        self.labels = labels.to(device)
        self.device = device
        
    def __len__(self):
        return self.edges.size(1)
        
    def __getitem__(self, idx):
        return {
            'edge': self.edges[:, idx],
            'label': self.labels[idx]
        }

class IncrementalAnimeDataProcessor:
    def __init__(self, anime_df: pd.DataFrame, interactions_df: pd.DataFrame, users_df: pd.DataFrame, output_dir: str, device='cuda'):
        self.logger = self.setup_logging()
        self.logger.info(f"初始化处理器 - 当前内存使用: {get_memory_usage():.2f} MB")
        
        start_time = time.time()
        self.device = device
        self.anime_df = anime_df.copy()
        self.interactions_df = interactions_df.copy()
        self.users_df = users_df.copy()

        # 创建ID到索引的映射，提高查找性能
        self.anime_id_map = {anime_id: idx for idx, anime_id in enumerate(self.anime_df['subject_id'].unique())}
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(self.users_df['用户ID'].unique())}
        
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"数据复制完成，耗时: {time.time() - start_time:.2f}秒")
        self.setup_bert_model()
        
    @staticmethod
    def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger = logging.getLogger(__name__)
        logger.info(f"开始加载数据文件，当前内存使用: {get_memory_usage():.2f} MB")
        
        start_time = time.time()
        anime_df = pd.read_csv(os.path.join(data_dir, "anime.tsv"), sep='\t')
        logger.info(f"动画数据加载完成，形状: {anime_df.shape}")
        
        interactions_df = pd.read_csv(os.path.join(data_dir, "interactions.tsv"), sep='\t')
        logger.info(f"交互数据加载完成，形状: {interactions_df.shape}")
        
        users_df = pd.read_csv(os.path.join(data_dir, "user.tsv"), sep='\t')
        logger.info(f"用户数据加载完成，形状: {users_df.shape}")
        
        logger.info(f"所有数据加载完成，总耗时: {time.time() - start_time:.2f}秒")
        logger.info(f"加载后内存使用: {get_memory_usage():.2f} MB")
        
        return anime_df, interactions_df, users_df

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def setup_bert_model(self):
        start_time = time.time()
        try:
            self.logger.info("开始加载LLM...")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            self.bert_model = AutoModel.from_pretrained('bert-base-chinese')
            self.bert_model.eval()
            if torch.cuda.is_available():
                self.bert_model = self.bert_model.to(self.device)
                self.logger.info("LLM已转移到GPU")
            self.logger.info(f"LLM加载完成，耗时: {time.time() - start_time:.2f}秒")
        except Exception as e:
            self.logger.error(f"LLM加载失败: {str(e)}")
            raise
            
    def clean_comment_text(self, text: str) -> str:
        """清理评论文本"""
        if pd.isna(text):
            return ""
            
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', str(text))
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        # 移除多余空白
        text = ' '.join(text.split())
        
        return text
        
    def process_time_windows(self, interactions_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        start_time = time.time()
        self.logger.info("开始处理时间窗口...")
        
        interactions_df['updated_at'] = pd.to_datetime(interactions_df['updated_at'])
        
        train_end = pd.Timestamp('2022-04-01')
        val_end = pd.Timestamp('2022-07-01')
        test_end = pd.Timestamp('2022-10-01')
        
        train_data = interactions_df[interactions_df['updated_at'] < train_end]
        val_data = interactions_df[
            (interactions_df['updated_at'] >= train_end) & 
            (interactions_df['updated_at'] < val_end)
        ]
        test_data = interactions_df[
            (interactions_df['updated_at'] >= val_end) & 
            (interactions_df['updated_at'] < test_end)
        ]
        
        self.logger.info(f"数据集划分完成 - 训练集: {len(train_data)}条, 验证集: {len(val_data)}条, 测试集: {len(test_data)}条")
        self.logger.info(f"时间窗口处理完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 保存时间窗口数据
        self.save_data(train_data, 'train_data.pkl')
        self.save_data(val_data, 'val_data.pkl')
        self.save_data(test_data, 'test_data.pkl')
        
        # 释放内存
        self.logger.info(f"释放未使用数据前内存使用: {get_memory_usage():.2f} MB")
        
        time_windows = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        # 保存索引信息
        self.save_data({
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data)
        }, 'time_windows_meta.pkl')
        
        return time_windows
        
    def process_comments_in_batches(self, interaction_df: pd.DataFrame, batch_size=512, output_file='comment_embeddings.pkl'):
        """分批处理评论并保存到文件，只处理非空评论"""
        start_time = time.time()
        self.logger.info("开始增量处理评论文本，仅处理非空评论...")
        
        # 检查点和输出文件路径
        checkpoint_path = os.path.join(self.output_dir, 'comment_processing_checkpoint.json')
        embeddings_path = os.path.join(self.output_dir, output_file)
        map_path = os.path.join(self.output_dir, 'comment_to_item_map.pkl')
        
        # 首先过滤出非空评论
        interaction_df['comment'] = interaction_df['comment'].fillna("").astype(str)
        non_empty_mask = interaction_df['comment'].str.strip() != ''
        empty_mask = ~non_empty_mask
        
        # 统计空评论和非空评论数量
        num_empty = empty_mask.sum()
        num_non_empty = non_empty_mask.sum()
        
        self.logger.info(f"评论总数: {len(interaction_df)}, 非空评论: {num_non_empty} ({num_non_empty/len(interaction_df)*100:.2f}%), 空评论: {num_empty}")
        
        # 获取非空评论的索引和数据
        non_empty_indices = list(interaction_df.index[non_empty_mask])
        non_empty_df = interaction_df.loc[non_empty_indices]
        
        # 获取空评论的索引和物品ID
        empty_indices = list(interaction_df.index[empty_mask])
        empty_subject_ids = interaction_df.loc[empty_mask, 'subject_id'].values.tolist()
        
        # 创建/加载评论到物品ID的映射
        comment_to_item_map = {}
        if os.path.exists(map_path):
            try:
                with open(map_path, 'rb') as f:
                    comment_to_item_map = pickle.load(f)
                    self.logger.info(f"已加载评论到物品映射，包含 {len(comment_to_item_map)} 条记录")
            except Exception as e:
                self.logger.warning(f"加载评论到物品映射失败: {str(e)}，将重新创建映射")
        
        # 为空评论添加映射
        for idx, subject_id in zip(empty_indices, empty_subject_ids):
            comment_to_item_map[idx] = subject_id
        
        # 为空评论创建一个零向量作为嵌入
        embedding_dim = 768  # BERT-base的维度
        empty_embedding = np.zeros(embedding_dim, dtype=np.float16)
        
        # 加载检查点信息，确定哪些非空评论已处理
        processed_indices = set()
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                    processed_indices = set([int(idx) for idx in checkpoint_data.get('processed_indices', [])])
                    self.logger.info(f"已加载检查点，找到 {len(processed_indices)} 条已处理评论")
            except Exception as e:
                self.logger.warning(f"加载检查点文件失败: {str(e)}，将重新处理所有非空评论")
        
        # 找出未处理的非空评论
        indices_to_process = [idx for idx in non_empty_indices if idx not in processed_indices]
        
        # 加载已有的嵌入结果（如果存在）
        all_batch_data = []
        has_empty_placeholder = False
        if os.path.exists(embeddings_path):
            try:
                with open(embeddings_path, 'rb') as f:
                    all_batch_data = pickle.load(f)
                    # 检查是否已有空评论占位符
                    for batch in all_batch_data:
                        if isinstance(batch, dict) and batch.get('is_empty_placeholder', False):
                            has_empty_placeholder = True
                            break
                    self.logger.info(f"已加载嵌入数据，包含 {len(all_batch_data)} 个批次")
            except Exception as e:
                self.logger.warning(f"加载嵌入数据失败: {str(e)}，将重新处理所有嵌入")
                all_batch_data = []
        
        # 如果没有找到空评论占位符，添加一个
        if not has_empty_placeholder:
            empty_batch = {
                'is_empty_placeholder': True,
                'empty_embedding': empty_embedding,
                'empty_indices': empty_indices
            }
            all_batch_data.append(empty_batch)
            self.logger.info(f"已添加空评论占位符，包含 {len(empty_indices)} 条空评论")
        
        if not indices_to_process:
            self.logger.info(f"所有非空评论已处理完成，无需额外处理")
            
            # 保存所有数据
            self.save_data(all_batch_data, output_file)
            self.save_data(comment_to_item_map, 'comment_to_item_map.pkl')
            
            # 更新检查点
            with open(checkpoint_path, 'w') as f:
                all_indices = list(processed_indices) + empty_indices
                json.dump({
                    'processed_indices': all_indices,
                    'non_empty_processed': list(processed_indices),
                    'empty_indices': empty_indices,
                    'total_processed': len(all_indices),
                    'last_update': datetime.now().isoformat(),
                    'processing_complete': True
                }, f)
                
            return output_file
        
        self.logger.info(f"找到 {len(indices_to_process)} 条未处理的非空评论，开始处理")
        
        # 获取要处理的评论文本和物品ID
        comments_to_process = non_empty_df.loc[indices_to_process, 'comment'].astype(str)
        subject_ids_to_process = non_empty_df.loc[indices_to_process, 'subject_id'].values.tolist()
        
        # 向量化清理文本
        comments_to_process = comments_to_process.str.replace(r'<[^>]+>', '', regex=True)  # 移除HTML标签
        comments_to_process = comments_to_process.str.replace(r'[^\w\s\u4e00-\u9fff]', ' ', regex=True)  # 移除特殊字符 
        comments_to_process = comments_to_process.str.split().str.join(' ')  # 移除多余空白
        
        # 估计总批次数
        total_batches = (len(comments_to_process) + batch_size - 1) // batch_size
        
        # 配置BERT混合精度计算
        torch.backends.cudnn.benchmark = True
        
        # 使用自适应批处理大小避免OOM
        adaptive_batch_size = batch_size
        min_batch_size = 32  # 最小批处理大小
        max_retries = 3  # 最大重试次数，防止无限循环
        
        # 处理新的批次
        total_processed = 0
        current_processed = list(processed_indices)  # 跟踪已处理的非空评论
        
        i = 0
        while i < len(comments_to_process):
            try:
                # 计算当前批次的结束索引
                end_idx = min(i + adaptive_batch_size, len(comments_to_process))
                
                # 获取当前批次的索引、评论和物品ID
                batch_indices = indices_to_process[i:end_idx]
                batch_comments = comments_to_process.iloc[i:end_idx].values.tolist()
                batch_subject_ids = subject_ids_to_process[i:end_idx]
                
                # 使用Tokenizer处理
                inputs = self.tokenizer(
                    batch_comments,
                    padding='max_length',
                    truncation=True,
                    max_length=256,
                    return_tensors='pt'
                )
                
                # 将输入数据移到GPU
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                    outputs = self.bert_model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].detach()
                    
                    # 转移到CPU以释放GPU内存
                    batch_embeddings_cpu = batch_embeddings.cpu()
                    del batch_embeddings
                    
                    # 保存批次结果
                    batch_data = {
                        'embeddings': batch_embeddings_cpu.numpy().astype(np.float16),
                        'subject_ids': batch_subject_ids,
                        'global_indices': batch_indices  # 保存全局索引，便于恢复
                    }
                    
                    all_batch_data.append(batch_data)
                    
                    # 更新评论到物品ID的映射
                    for idx, subject_id in zip(batch_indices, batch_subject_ids):
                        comment_to_item_map[idx] = subject_id
                    
                    # 更新已处理的索引
                    current_processed.extend(batch_indices)
                    
                # 清理缓存
                del inputs, outputs
                torch.cuda.empty_cache()
                
                # 如果处理成功，尝试增加批处理大小
                if adaptive_batch_size < batch_size:
                    adaptive_batch_size = min(adaptive_batch_size * 2, batch_size)
                    
                # 更新处理进度
                total_processed += len(batch_comments)
                if total_processed % (batch_size * 5) == 0 or end_idx == len(comments_to_process):
                    self.logger.info(f"已处理 {total_processed}/{len(comments_to_process)} 条新非空评论")
                    self.logger.info(f"当前内存使用: {get_memory_usage():.2f} MB")
                    
                    # 定期保存检查点
                    with open(checkpoint_path, 'w') as f:
                        all_indices = current_processed + empty_indices
                        json.dump({
                            'processed_indices': all_indices,
                            'non_empty_processed': current_processed,
                            'empty_indices': empty_indices,
                            'total_processed': len(all_indices),
                            'last_update': datetime.now().isoformat()
                        }, f)
                    
                    # 保存嵌入和映射
                    self.save_data(all_batch_data, output_file)
                    self.save_data(comment_to_item_map, 'comment_to_item_map.pkl')
                    
                    gc.collect()  # 触发垃圾回收
                
                # 移动到下一批
                i = end_idx
                max_retries = 3  # 重置重试计数器
                    
            except RuntimeError as e:
                # 处理OOM错误
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # 减小批处理大小
                    adaptive_batch_size = max(adaptive_batch_size // 2, min_batch_size)
                    self.logger.warning(f"CUDA内存不足，将批处理大小减小到 {adaptive_batch_size}")
                    
                    # 防止无限循环
                    max_retries -= 1
                    if max_retries <= 0 and adaptive_batch_size <= min_batch_size:
                        self.logger.error(f"在处理批次时遇到持续的OOM错误，保存当前进度并退出。当前处理进度: {total_processed}/{len(comments_to_process)}")
                        
                        # 保存当前进度
                        with open(checkpoint_path, 'w') as f:
                            all_indices = current_processed + empty_indices
                            json.dump({
                                'processed_indices': all_indices,
                                'non_empty_processed': current_processed,
                                'empty_indices': empty_indices,
                                'total_processed': len(all_indices),
                                'last_update': datetime.now().isoformat()
                            }, f)
                        
                        self.save_data(all_batch_data, output_file)
                        self.save_data(comment_to_item_map, 'comment_to_item_map.pkl')
                        break
                    
                    continue
                else:
                    # 遇到其他错误，保存进度后抛出异常
                    self.logger.error(f"处理评论时遇到错误: {str(e)}")
                    with open(checkpoint_path, 'w') as f:
                        all_indices = current_processed + empty_indices
                        json.dump({
                            'processed_indices': all_indices,
                            'non_empty_processed': current_processed,
                            'empty_indices': empty_indices,
                            'total_processed': len(all_indices),
                            'last_update': datetime.now().isoformat()
                        }, f)
                    
                    self.save_data(all_batch_data, output_file)
                    self.save_data(comment_to_item_map, 'comment_to_item_map.pkl')
                    raise
        
        # 最终保存所有结果
        self.logger.info("将所有处理结果写入文件...")
        self.save_data(all_batch_data, output_file)
        self.save_data(comment_to_item_map, 'comment_to_item_map.pkl')
        
        # 更新最终检查点
        with open(checkpoint_path, 'w') as f:
            all_indices = current_processed + empty_indices
            json.dump({
                'processed_indices': all_indices,
                'non_empty_processed': current_processed,
                'empty_indices': empty_indices,
                'total_processed': len(all_indices),
                'last_update': datetime.now().isoformat(),
                'processing_complete': True
            }, f)
        
        self.logger.info(f"评论增量处理完成，总耗时: {time.time() - start_time:.2f}秒")
        self.logger.info(f"总共处理了 {total_processed} 条新非空评论和 {len(empty_indices)} 条空评论")
        
        return output_file
    
    def append_to_pickle(self, data, filepath):
        """将数据追加到pickle文件"""
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            # 如果文件不存在或为空，创建新文件和列表
            with open(filepath, 'wb') as f:
                pickle.dump([data], f)
        else:
            # 读取现有数据
            with open(filepath, 'rb') as f:
                existing_data = pickle.load(f)
            
            # 追加新数据
            existing_data.append(data)
            
            # 重写文件
            with open(filepath, 'wb') as f:
                pickle.dump(existing_data, f)
        
    def build_interaction_matrix(self, interactions_df: pd.DataFrame, save=True):
        start_time = time.time()
        self.logger.info("开始构建交互矩阵...")
        
        unique_users = sorted(interactions_df['user_id'].unique())
        unique_items = sorted(interactions_df['subject_id'].unique())
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        
        self.logger.info(f"用户数量: {len(unique_users)}, 物品数量: {len(unique_items)}")
        
        # 批量处理索引转换
        user_indices = torch.tensor([self.user_id_map[uid] for uid in tqdm(interactions_df['user_id'], desc="处理用户索引")])
        item_indices = torch.tensor([self.item_id_map[iid] for iid in tqdm(interactions_df['subject_id'], desc="处理物品索引")])
        
        values = torch.FloatTensor(interactions_df['rate'].values)
        
        matrix = torch.sparse_coo_tensor(
            indices=torch.stack([user_indices, item_indices]),
            values=values,
            size=(len(unique_users), len(unique_items))
        )
        
        if save:
            # 保存映射和矩阵的索引和值
            mapping_data = {
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map,
                'matrix_size': (len(unique_users), len(unique_items))
            }
            self.save_data(mapping_data, 'id_mappings.pkl')
            
            # 保存矩阵组件
            matrix_data = {
                'indices': torch.stack([user_indices, item_indices]).cpu().numpy(),
                'values': values.cpu().numpy(),
                'size': (len(unique_users), len(unique_items))
            }
            self.save_data(matrix_data, 'interaction_matrix_components.pkl')
        
        self.logger.info(f"交互矩阵构建完成，耗时: {time.time() - start_time:.2f}秒")
        self.logger.info(f"矩阵大小: {matrix.size()}, 非零元素数量: {len(values)}")
        
        return matrix
        
    def generate_negative_samples_matrix(self, pos_interactions: pd.DataFrame, num_negatives: int = 1, batch_size=10000):
        """
        使用矩阵运算进行高效的负采样（修复版本）
        
        Args:
            pos_interactions: 正样本交互数据
            num_negatives: 每个正样本生成的负样本数
            batch_size: 处理的批次大小，控制内存使用
        """
        import scipy.sparse as sp
        start_time = time.time()
        self.logger.info(f"开始使用矩阵方法生成负样本（修复版本），每个正样本生成 {num_negatives} 个负样本...")
        
        # 准备输出文件
        output_file = os.path.join(self.output_dir, 'negative_samples.pkl')
        
        # 将正样本标记并保存
        pos_interactions['is_negative'] = 0
        self.save_data(pos_interactions, 'positive_samples.pkl')
        
        # 获取唯一用户和物品 - 首先要确保只使用存在的ID
        unique_users = sorted(pos_interactions['user_id'].unique())
        user_mapping = {uid: i for i, uid in enumerate(unique_users)}
        
        # 获取所有交互中出现的物品ID和动画数据框中的物品ID
        interacted_items = set(pos_interactions['subject_id'].unique())
        anime_items = set(self.anime_df['subject_id'])  # 使用subject_id列而不是索引
        
        # 确保使用的物品ID同时存在于交互数据和动画数据中
        # 我们主要关心能够在self.anime_df中查找到score的物品
        valid_items = sorted(interacted_items.intersection(anime_items))
        
        # 如果交互中有些物品ID不在anime_df中，记录日志
        if len(interacted_items - anime_items) > 0:
            self.logger.warning(f"发现 {len(interacted_items - anime_items)} 个物品ID存在于交互数据但不在动画数据中")
        
        # 构建物品映射
        item_mapping = {iid: i for i, iid in enumerate(valid_items)}
        reverse_item_mapping = {i: iid for iid, i in item_mapping.items()}
        
        # 创建用户-物品交互矩阵 (1表示交互过)
        self.logger.info(f"构建交互矩阵...用户数: {len(unique_users)}, 有效物品数: {len(valid_items)}")
        n_users = len(unique_users)
        n_items = len(valid_items)
        
        # 过滤只包含映射中存在的物品ID的交互
        valid_interactions = pos_interactions[pos_interactions['subject_id'].isin(valid_items)]
        self.logger.info(f"有效交互数: {len(valid_interactions)}/{len(pos_interactions)} ({len(valid_interactions)/len(pos_interactions)*100:.2f}%)")
        
        # 获取行、列索引
        user_idx = []
        item_idx = []
        try:
            for _, row in valid_interactions.iterrows():
                if row['user_id'] in user_mapping and row['subject_id'] in item_mapping:
                    user_idx.append(user_mapping[row['user_id']])
                    item_idx.append(item_mapping[row['subject_id']])
        except Exception as e:
            self.logger.error(f"处理交互数据时出错: {str(e)}")
            self.logger.error(f"当前处理的用户ID: {row['user_id']}, 物品ID: {row['subject_id']}")
            raise
            
        # 创建稀疏矩阵
        interaction_matrix = sp.csr_matrix(
            (np.ones(len(user_idx)), (user_idx, item_idx)),
            shape=(n_users, n_items)
        )
        
        # 预处理物品权重 - 更健壮的版本
        self.logger.info("预处理物品权重...")
        item_scores = np.ones(n_items)  # 默认所有物品权重为1
        
        for i in range(n_items):
            item_id = reverse_item_mapping[i]
            # 查找对应的行而不是使用索引
            matching_rows = self.anime_df[self.anime_df['subject_id'] == item_id]
            if not matching_rows.empty:
                try:
                    score = matching_rows.iloc[0]['score']
                    if not pd.isna(score):
                        item_scores[i] = score
                except Exception as e:
                    self.logger.warning(f"获取物品 {item_id} 的评分时出错: {str(e)}，使用默认值1.0")
        
        # 归一化权重，避免除零错误
        sum_scores = item_scores.sum()
        if sum_scores > 0:
            item_scores = item_scores / sum_scores
        else:
            item_scores = np.ones(n_items) / n_items
            self.logger.warning("物品评分之和为零，使用均匀分布")
        
        # 分批处理用户以控制内存使用
        negative_count = 0
        neg_samples_df_list = []  # 用于存储所有批次的负样本
        
        for start_idx in tqdm(range(0, n_users, batch_size), desc="处理用户批次"):
            end_idx = min(start_idx + batch_size, n_users)
            batch_users_idx = list(range(start_idx, end_idx))
            batch_size_actual = len(batch_users_idx)
            
            self.logger.info(f"处理用户批次 {start_idx}-{end_idx-1} (共 {batch_size_actual} 个用户)")
            
            # 获取这批用户的交互矩阵
            batch_interactions = interaction_matrix[batch_users_idx]
            
            # 批量生成负样本
            batch_negatives = []
            
            for user_idx in tqdm(range(batch_size_actual), desc="生成负样本", leave=False):
                real_user_idx = batch_users_idx[user_idx]
                user_id = unique_users[real_user_idx]
                
                # 获取用户交互过的物品
                interacted = batch_interactions[user_idx].nonzero()[1]
                interacted_set = set(interacted)
                
                # 计算未交互物品
                non_interacted = [i for i in range(n_items) if i not in interacted_set]
                
                if not non_interacted:
                    self.logger.info(f"用户 {user_id} 已经与所有有效物品交互过，跳过负采样")
                    continue
                    
                # 为未交互物品计算采样权重
                try:
                    item_weights = np.array([item_scores[i] for i in non_interacted])
                    # 防止权重和为0
                    if np.sum(item_weights) == 0:
                        item_weights = np.ones_like(item_weights)
                    item_weights = item_weights / np.sum(item_weights)
                except Exception as e:
                    self.logger.error(f"计算物品权重时出错: {str(e)}")
                    self.logger.error(f"非交互物品索引数量: {len(non_interacted)}, 物品评分数组大小: {len(item_scores)}")
                    raise
                
                # 计算该用户在正样本中的行数
                user_pos_count = len(interacted)
                
                # 计算需要采样的负样本数量
                to_sample = min(user_pos_count * num_negatives, len(non_interacted))
                
                if to_sample > 0:
                    # 采样负样本物品索引
                    try:
                        sampled_items_idx = np.random.choice(
                            len(non_interacted),
                            size=to_sample,
                            p=item_weights,
                            replace=False
                        )
                        
                        # 转换为物品ID
                        for idx in sampled_items_idx:
                            if idx >= len(non_interacted):
                                self.logger.error(f"索引超出范围: {idx} >= {len(non_interacted)}")
                                continue
                                
                            non_interacted_idx = non_interacted[idx]
                            if non_interacted_idx in reverse_item_mapping:
                                item_id = reverse_item_mapping[non_interacted_idx]
                                
                                # 创建负样本记录
                                batch_negatives.append({
                                    'user_id': user_id,
                                    'subject_id': item_id,
                                    'rate': 0,
                                    'is_negative': 1
                                })
                            else:
                                self.logger.warning(f"无法找到索引 {non_interacted_idx} 对应的物品ID")
                                
                    except Exception as e:
                        self.logger.warning(f"为用户 {user_id} 采样负样本时出错: {str(e)}")
                        self.logger.warning(f"非交互物品: {len(non_interacted)}, 权重: {len(item_weights)}, 要采样: {to_sample}")
                else:
                    self.logger.info(f"用户 {user_id} 无需采样负样本 (交互: {user_pos_count}, 可用物品: {len(non_interacted)})")
            
            # 保存批次负样本
            if batch_negatives:
                batch_neg_df = pd.DataFrame(batch_negatives)
                neg_samples_df_list.append(batch_neg_df)
                negative_count += len(batch_negatives)
                    
                self.logger.info(f"已处理 {end_idx}/{n_users} 个用户，累计生成 {negative_count} 个负样本")
                self.logger.info(f"当前内存使用: {get_memory_usage():.2f} MB")
                
                # 每处理5个批次保存一次数据
                if len(neg_samples_df_list) >= 5:
                    combined_df = pd.concat(neg_samples_df_list, ignore_index=True)
                    self.append_df_to_pickle(combined_df, output_file)
                    neg_samples_df_list = []  # 清空列表
                    self.logger.info(f"已保存负样本批次到文件，累计 {negative_count} 个负样本")
            
            # 清理内存
            del batch_interactions, batch_negatives
            gc.collect()
        
        # 保存剩余批次
        if neg_samples_df_list:
            combined_df = pd.concat(neg_samples_df_list, ignore_index=True)
            self.append_df_to_pickle(combined_df, output_file)
            self.logger.info(f"已保存剩余负样本批次到文件")
        
        self.logger.info(f"负样本生成完成，总共生成 {negative_count} 个负样本，耗时: {time.time() - start_time:.2f}秒")
        return output_file
    
    def append_df_to_pickle(self, df, filepath):
        """将DataFrame追加到pickle文件"""
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            # 如果文件不存在或为空，创建新文件和DataFrame
            df.to_pickle(filepath)
        else:
            # 读取现有数据
            existing_df = pd.read_pickle(filepath)
            
            # 追加新数据
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            
            # 重写文件
            combined_df.to_pickle(filepath)
    
    def create_edge_data(self, data_type, data_df):
        """为指定数据类型创建边数据并保存"""
        self.logger.info(f"创建{data_type}边数据...")
        
        # 确保数据中的ID在映射中存在
        valid_users = [uid for uid in data_df['user_id'] if uid in self.user_id_map]
        valid_items = [iid for iid in data_df['subject_id'] if iid in self.item_id_map]
        
        if len(valid_users) < len(data_df):
            self.logger.warning(f"{len(data_df) - len(valid_users)} 个用户ID在{data_type}数据中不存在于映射中")
        
        if len(valid_items) < len(data_df):
            self.logger.warning(f"{len(data_df) - len(valid_items)} 个物品ID在{data_type}数据中不存在于映射中")
        
        # 过滤有效数据
        valid_df = data_df[data_df['user_id'].isin(valid_users) & data_df['subject_id'].isin(valid_items)]
        
        # 创建边数据
        user_indices = [self.user_id_map[uid] for uid in valid_df['user_id']]
        item_indices = [self.item_id_map[iid] for iid in valid_df['subject_id']]
        
        # 保存为NumPy数组
        edge_data = {
            'user_indices': np.array(user_indices),
            'item_indices': np.array(item_indices)
        }
        
        self.save_data(edge_data, f'{data_type}_edges.pkl')
        self.logger.info(f"{data_type}边数据创建完成，保存了 {len(user_indices)} 条边")
        
        return edge_data
        
    def save_data(self, data, filename):
        """保存数据到文件"""
        filepath = os.path.join(self.output_dir, filename)
        start_time = time.time()
        self.logger.info(f"保存数据到 {filepath}")
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"数据保存完成，耗时: {time.time() - start_time:.2f}秒")
        except Exception as e:
            self.logger.error(f"数据保存失败: {str(e)}")
            raise
            
    def save_metadata(self):
        """保存处理后数据的元数据"""
        metadata = {
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_users': len(self.user_id_map) if hasattr(self, 'user_id_map') else 0,
            'num_items': len(self.item_id_map) if hasattr(self, 'item_id_map') else 0,
            'files': os.listdir(self.output_dir)
        }
        
        self.save_data(metadata, 'metadata.pkl')
        
    def process(self):
        """执行增量数据处理流程"""
        total_start_time = time.time()
        self.logger.info(f"开始增量数据处理流程，初始内存使用: {get_memory_usage():.2f} MB")

        # 检查是否已经完成评论嵌入处理
        comment_embeddings_file = os.path.join(self.output_dir, 'comment_embeddings.pkl')
        checkpoint_path = os.path.join(self.output_dir, 'comment_processing_checkpoint.json')
        comments_already_processed = False
        
        if os.path.exists(comment_embeddings_file) and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                    if checkpoint_data.get('processing_complete', False):
                        comments_already_processed = True
                        self.logger.info("检测到评论嵌入已完全处理，将跳过此步骤")
            except Exception:
                self.logger.warning("检查点读取失败，将正常执行所有处理步骤")

        
        # 步骤1: 保存原始数据
        self.logger.info("步骤1: 保存原始数据引用...")
        self.save_data(self.anime_df, 'anime_data.pkl')
        self.save_data(self.users_df, 'user_data.pkl')
        
        # 步骤2: 按时间窗口划分数据
        self.logger.info("步骤2: 按时间窗口划分数据...")
        time_split_data = self.process_time_windows(self.interactions_df)
        
        # 释放原始交互数据以节省内存
        del self.interactions_df
        torch.cuda.empty_cache()
        self.logger.info(f"释放原始交互数据后内存使用: {get_memory_usage():.2f} MB")
        
        # 步骤3: 构建交互矩阵
        self.logger.info("步骤3: 构建交互矩阵...")
        train_interaction_matrix = self.build_interaction_matrix(time_split_data['train'])
        
        # 步骤4: 处理评论嵌入 - 仅在未处理时执行
        if not comments_already_processed:
            self.logger.info("步骤4: 增量处理评论嵌入，仅处理非空评论...")
            comment_embeddings_file = self.process_comments_in_batches(time_split_data['train'], batch_size=512)
        else:
            self.logger.info("步骤4: 跳过评论嵌入处理，使用已有结果")
        
        # 释放评论数据以节省内存
        time_split_data['train'] = time_split_data['train'].drop(columns=['comment'])
        torch.cuda.empty_cache()
        self.logger.info(f"释放评论数据后内存使用: {get_memory_usage():.2f} MB")
        
        # 步骤5: 生成负样本
        self.logger.info("步骤5: 生成负样本...")
        neg_samples_file = self.generate_negative_samples_matrix(time_split_data['train'])  # 使用矩阵版本
        
        # 步骤6: 创建边数据
        self.logger.info("步骤6: 创建边数据...")
        self.create_edge_data('train', time_split_data['train'])
        self.create_edge_data('val', time_split_data['val'])
        self.create_edge_data('test', time_split_data['test'])
        
        # 步骤7: 保存元数据
        self.logger.info("步骤7: 保存处理元数据...")
        self.save_metadata()
        
        total_time = time.time() - total_start_time
        self.logger.info(f"增量数据处理完成，总耗时: {total_time:.2f}秒")
        self.logger.info(f"最终内存使用: {get_memory_usage():.2f} MB")
        self.logger.info(f"所有处理后的数据文件已保存到: {self.output_dir}")
        
        # 返回输出目录
        return self.output_dir

def create_data_loader(processor_dir, data_type='train', device='cuda'):
    """根据已处理的数据创建数据加载器"""
    logger = logging.getLogger(__name__)
    logger.info(f"从{processor_dir}加载{data_type}数据加载器...")
    
    try:
        # 加载边数据
        with open(os.path.join(processor_dir, f'{data_type}_edges.pkl'), 'rb') as f:
            edge_data = pickle.load(f)
            
        user_indices = torch.tensor(edge_data['user_indices'], device=device)
        item_indices = torch.tensor(edge_data['item_indices'], device=device)
        
        # 创建边和标签
        edges = torch.stack([user_indices, item_indices])
        
        # 对于训练数据，我们需要负样本标签
        if data_type == 'train':
            # 加载正样本和负样本
            with open(os.path.join(processor_dir, 'positive_samples.pkl'), 'rb') as f:
                positive_samples = pickle.load(f)
                
            try:
                with open(os.path.join(processor_dir, 'negative_samples.pkl'), 'rb') as f:
                    negative_samples = pickle.load(f)
            except:
                logger.warning("未找到负样本文件，仅使用正样本")
                negative_samples = pd.DataFrame()
                
            # 合并样本
            all_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)
            
            # 创建标签
            labels = torch.tensor(all_samples['is_negative'].values, dtype=torch.float, device=device)
        else:
            # 对于验证和测试数据，我们可以只用评分作为标签
            labels = torch.ones(edges.size(1), device=device)
            
        # 创建数据集
        dataset = AnimeDataset(edges, labels, device=device)
        
        logger.info(f"{data_type}数据加载器创建完成，数据集大小: {len(dataset)}")
        return dataset
        
    except Exception as e:
        logger.error(f"创建数据加载器失败: {str(e)}")
        raise

def load_processed_data(processor_dir):
    """加载处理后的数据以供模型使用"""
    logger = logging.getLogger(__name__)
    logger.info(f"加载处理后的数据从: {processor_dir}")
    
    try:
        # 加载元数据
        with open(os.path.join(processor_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            
        # 加载ID映射
        with open(os.path.join(processor_dir, 'id_mappings.pkl'), 'rb') as f:
            id_mappings = pickle.load(f)
            
        # 加载交互矩阵组件
        with open(os.path.join(processor_dir, 'interaction_matrix_components.pkl'), 'rb') as f:
            matrix_components = pickle.load(f)
            
        # 重建交互矩阵
        indices = torch.tensor(matrix_components['indices'])
        values = torch.tensor(matrix_components['values'])
        size = matrix_components['size']
        
        train_interaction_matrix = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=size
        )
        
        # 加载评论嵌入和评论到物品的映射
        with open(os.path.join(processor_dir, 'comment_embeddings.pkl'), 'rb') as f:
            comment_data_list = pickle.load(f)
        
        # 如果有评论到物品的映射，则加载
        comment_to_item_map = {}
        try:
            with open(os.path.join(processor_dir, 'comment_to_item_map.pkl'), 'rb') as f:
                comment_to_item_map = pickle.load(f)
        except FileNotFoundError:
            logger.warning("未找到评论到物品的映射文件，将尝试从评论嵌入数据重建")
        
        # 处理评论嵌入和物品ID
        all_embeddings = []
        all_subject_ids = []
        empty_embedding = None
        empty_indices = []
        
        # 处理嵌入数据，区分普通批次和空评论占位符
        for batch_data in comment_data_list:
            if isinstance(batch_data, dict):
                if batch_data.get('is_empty_placeholder', False):
                    # 找到空评论占位符
                    empty_embedding = batch_data['empty_embedding']
                    empty_indices = batch_data['empty_indices']
                    continue
                elif 'embeddings' in batch_data and 'subject_ids' in batch_data:
                    # 非空评论数据
                    all_embeddings.append(batch_data['embeddings'])
                    all_subject_ids.extend(batch_data['subject_ids'])
            else:
                # 旧格式数据
                all_embeddings.append(batch_data)
        
        # 合并所有非空评论嵌入
        if all_embeddings:
            comment_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            comment_embeddings = np.array([])
        
        # 处理空评论嵌入
        if empty_embedding is not None and empty_indices:
            # 为所有空评论创建相同的嵌入
            empty_embeddings = np.tile(empty_embedding, (len(empty_indices), 1))
            
            # 将空评论嵌入与非空评论嵌入合并
            if comment_embeddings.size > 0:
                comment_embeddings = np.vstack([comment_embeddings, empty_embeddings])
            else:
                comment_embeddings = empty_embeddings
            
            # 获取空评论对应的物品ID
            for idx in empty_indices:
                if idx in comment_to_item_map:
                    all_subject_ids.append(comment_to_item_map[idx])
        
        # 转换为张量
        comment_embeddings = torch.tensor(comment_embeddings)
        
        # 如果我们有物品ID，创建一个评论嵌入到物品索引的映射
        comment_embedding_to_item_idx = {}
        if all_subject_ids:
            item_id_to_idx = id_mappings['item_id_map']
            for i, subject_id in enumerate(all_subject_ids):
                if subject_id in item_id_to_idx:
                    comment_embedding_to_item_idx[i] = item_id_to_idx[subject_id]
        
        # 创建结果字典
        result = {
            'metadata': metadata,
            'user_id_map': id_mappings['user_id_map'],
            'item_id_map': id_mappings['item_id_map'],
            'train_interaction_matrix': train_interaction_matrix,
            'comment_embeddings': comment_embeddings,
            'comment_to_item_map': comment_to_item_map,
            'comment_embedding_to_item_idx': comment_embedding_to_item_idx,
            'num_users': metadata['num_users'],
            'num_items': metadata['num_items']
        }
        
        # 加载动画和用户数据
        with open(os.path.join(processor_dir, 'anime_data.pkl'), 'rb') as f:
            result['anime_data'] = pickle.load(f)
            
        with open(os.path.join(processor_dir, 'user_data.pkl'), 'rb') as f:
            result['user_data'] = pickle.load(f)
            
        logger.info(f"数据加载完成，用户数: {result['num_users']}, 物品数: {result['num_items']}")
        return result
        
    except Exception as e:
        logger.error(f"加载处理后的数据失败: {str(e)}")
        raise

def main(data_dir: str = data_dir, output_dir: str = processed_data_dir):
    logger = logging.getLogger(__name__)
    total_start_time = time.time()
    
    try:
        logger.info("=== 开始增量数据处理主流程 ===")
        logger.info(f"数据目录: {data_dir}")
        logger.info(f"输出目录: {output_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        logger.info("步骤1: 加载原始数据文件...")
        anime_df, interactions_df, users_df = IncrementalAnimeDataProcessor.load_data(data_dir)
        
        # 创建处理器实例并处理数据
        logger.info("步骤2: 初始化数据处理器...")
        processor = IncrementalAnimeDataProcessor(anime_df, interactions_df, users_df, output_dir)
        
        logger.info("步骤3: 执行增量数据处理...")
        processed_dir = processor.process()
        
        total_time = time.time() - total_start_time
        logger.info(f"=== 所有处理完成，总耗时: {total_time:.2f}秒 ===")
        logger.info(f"处理后的数据保存在: {processed_dir}")
        
        return processed_dir
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"处理失败: {str(e)}")