import os
import pandas as pd
import numpy as np
import torch
import logging
import pickle
from typing import Dict, Tuple, List, Any
import re
from datetime import datetime
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
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
            self.logger.info("开始加载BERT模型...")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            self.bert_model = AutoModel.from_pretrained('bert-base-chinese')
            self.bert_model.eval()
            if torch.cuda.is_available():
                self.bert_model = self.bert_model.to(self.device)
                self.logger.info("BERT模型已转移到GPU")
            self.logger.info(f"BERT模型加载完成，耗时: {time.time() - start_time:.2f}秒")
        except Exception as e:
            self.logger.error(f"BERT模型加载失败: {str(e)}")
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
        
    def process_comments_in_batches(self, comments: pd.Series, batch_size=2048, output_file='comment_embeddings.pkl'):
        """分批处理评论并保存到文件"""
        start_time = time.time()
        self.logger.info("开始分批处理评论文本...")
        
        total_batches = (len(comments) + batch_size - 1) // batch_size
        
        # 创建空文件以便追加
        with open(os.path.join(self.output_dir, output_file), 'wb') as f:
            pass
            
        total_processed = 0
        for i in tqdm(range(0, len(comments), batch_size), total=total_batches, desc="处理评论批次"):
            batch_comments = comments[i:i + batch_size].fillna('')
            cleaned_comments = [self.clean_comment_text(text) for text in batch_comments]
            
            inputs = self.tokenizer(
                cleaned_comments,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 将输入数据移到GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                
                # 保存批次结果
                batch_data = batch_embeddings.cpu().numpy()  # 转移到CPU并转为NumPy数组
                
                # 追加到文件
                self.append_to_pickle(batch_data, os.path.join(self.output_dir, output_file))
                
            total_processed += len(batch_comments)
            if total_processed % (batch_size * 10) == 0:
                self.logger.info(f"已处理 {total_processed}/{len(comments)} 条评论")
                torch.cuda.empty_cache()  # 清理GPU缓存
                self.logger.info(f"当前内存使用: {get_memory_usage():.2f} MB")
        
        self.logger.info(f"评论处理完成，总耗时: {time.time() - start_time:.2f}秒")
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
        
    def generate_negative_samples_in_batches(self, pos_interactions: pd.DataFrame, batch_size=2048, num_negatives: int = 1):
        start_time = time.time()
        self.logger.info(f"开始分批生成负样本，每批 {batch_size} 条，每个正样本生成 {num_negatives} 个负样本...")
        
        all_items = set(self.anime_df.index)
        output_file = os.path.join(self.output_dir, 'negative_samples.pkl')
        
        # 将正样本标记为非负样本并保存
        pos_interactions['is_negative'] = 0
        self.save_data(pos_interactions, 'positive_samples.pkl')
        
        # 分批处理
        total_batches = (len(pos_interactions) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(total_batches), desc="生成负样本批次"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(pos_interactions))
            batch = pos_interactions.iloc[start_idx:end_idx]
            
            negative_samples = []
            for _, row in batch.iterrows():
                user_id = row['user_id']
                
                user_interacted = set(
                    pos_interactions[pos_interactions['user_id'] == user_id]['subject_id']
                )
                
                available_items = list(all_items - user_interacted)
                
                item_weights = self.anime_df.loc[available_items, 'score'].values
                item_weights = np.nan_to_num(item_weights, nan=1.0)
                item_weights = item_weights / item_weights.sum()
                
                neg_items = np.random.choice(
                    available_items,
                    size=min(num_negatives, len(available_items)),
                    p=item_weights,
                    replace=False
                )
                
                for neg_item in neg_items:
                    negative_samples.append({
                        'user_id': user_id,
                        'subject_id': neg_item,
                        'rate': 0,
                        'is_negative': 1
                    })
            
            # 保存批次负样本
            batch_neg_df = pd.DataFrame(negative_samples)
            self.append_df_to_pickle(batch_neg_df, output_file)
            
            # 清理内存
            if batch_idx % 10 == 0:
                self.logger.info(f"已处理 {end_idx}/{len(pos_interactions)} 条正样本")
                self.logger.info(f"当前内存使用: {get_memory_usage():.2f} MB")
                
        self.logger.info(f"负样本生成完成，耗时: {time.time() - start_time:.2f}秒")
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
        
        # 步骤4: 处理评论嵌入
        self.logger.info("步骤4: 处理评论嵌入...")
        comment_embeddings_file = self.process_comments_in_batches(time_split_data['train']['comment'])
        
        # 释放评论数据以节省内存
        time_split_data['train'] = time_split_data['train'].drop(columns=['comment'])
        torch.cuda.empty_cache()
        self.logger.info(f"释放评论数据后内存使用: {get_memory_usage():.2f} MB")
        
        # 步骤5: 生成负样本
        self.logger.info("步骤5: 生成负样本...")
        neg_samples_file = self.generate_negative_samples_in_batches(time_split_data['train'])
        
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
        
        # 加载评论嵌入
        with open(os.path.join(processor_dir, 'comment_embeddings.pkl'), 'rb') as f:
            comment_embeddings_list = pickle.load(f)
            
        # 合并评论嵌入
        comment_embeddings = np.concatenate(comment_embeddings_list, axis=0)
        comment_embeddings = torch.tensor(comment_embeddings)
        
        # 创建结果字典
        result = {
            'metadata': metadata,
            'user_id_map': id_mappings['user_id_map'],
            'item_id_map': id_mappings['item_id_map'],
            'train_interaction_matrix': train_interaction_matrix,
            'comment_embeddings': comment_embeddings,
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