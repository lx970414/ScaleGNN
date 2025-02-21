import numpy as np
import random

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)

def random_walk_interface(adj_matrix, num_steps):
    """
    进行随机游走，返回不同步长下的节点索引关系。

    参数:
    adj_matrix: list of lists (邻接表，每个索引存储该节点的邻居)
    num_steps: int (最大步长)

    返回:
    adj_matrix_rw_buffer: list of tuples (存储步长结果 [(row_indices, col_indices, size)] )
    """
    num_nodes = len(adj_matrix)  # 节点数量
    
    # 初始化存储结构
    adj_matrix_rw_buffer = []

    # 进行随机游走
    for step in range(1, num_steps + 1):
        row_records = []
        col_records = []

        for node in range(num_nodes):
            current_node = node
            flag = True  # 是否成功完成随机游走

            for _ in range(step):
                if not adj_matrix[current_node]:  # 没有邻居，终止游走
                    flag = False
                    break
                current_node = random.choice(adj_matrix[current_node])  # 随机选择下一个节点
            
            if flag:
                row_records.append(node)
                col_records.append(current_node)

        # 记录当前步长的数据
        adj_matrix_rw_buffer.append((
            np.array(row_records, dtype=np.int32),  # 起点
            np.array(col_records, dtype=np.int32),  # 终点
            np.array([len(row_records)], dtype=np.int32)  # 该步长的随机游走数
        ))
    
    return adj_matrix_rw_buffer
