import numpy as np
from collections import defaultdict
from tqdm import tqdm
import gc
import psutil
import os


def triplet_to_str(triplet):
    return f"({triplet[0]},{triplet[1]},{triplet[2]})"


def unique_preserve_order(input_list):
    seen = set()
    unique_list = []
    for item in input_list:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list


def remove_same_head_tail(triplets, mode):
    if 'rmht' not in mode:
        return triplets

    new_triplets = []
    seen = set()
    for triplet in triplets:
        item_1 = ','.join([str(triplet[0]), str(triplet[2])])
        item_2 = ','.join([str(triplet[2]), str(triplet[0])])
        if item_1 not in seen and item_2 not in seen:
            seen.add(item_1)
            seen.add(item_2)
            new_triplets.append(triplet)
    return new_triplets


def merge_tuples(tuple_list, mode=0):
    if mode == 0:
        merged_dict = defaultdict(lambda: [[], None, None])
        for t in tuple_list:
            key = (t[1], t[2])  # Group by the second and third elements
            merged_dict[key][0].append(t[0])  # Append the first element to the list
            merged_dict[key][1] = t[1]  # Set the second element
            merged_dict[key][2] = t[2]  # Set the third element

        # Convert the dictionary back to a list of merged tuples
        return [('[' + ','.join(v[0]) + ']', v[1], v[2]) for v in merged_dict.values()]
    else:
        assert mode == 2
        merged_dict = defaultdict(lambda: [None, None, []])
        for t in tuple_list:
            key = (t[0], t[1])
            merged_dict[key][2].append(t[2])
            merged_dict[key][0] = t[0]
            merged_dict[key][1] = t[1]
        return [(v[0], v[1], '[' + ','.join(v[2]) + ']') for v in merged_dict.values()]


##############################################################################################################################/
def build_graph_from_triplets(triplets):
    """
    構建圖結構，返回鄰接表
    每個節點可以連接到其他節點，支持雙向連接
    """
    graph = defaultdict(list)
    
    for triplet in triplets:
        head, relation, tail = triplet[0], triplet[1], triplet[2]
        # 添加正向連接: head -> tail
        graph[head].append((tail, relation, 'forward'))
        # 添加反向連接: tail -> head  
        graph[tail].append((head, relation, 'backward'))
    
    return graph


def find_paths_from_graph_incremental(graph, query_entities, max_path_length=5, batch_size=100):
    """
    增量生成結構感知路徑，分批處理避免記憶體過載
    每找到一批路徑就立即處理，不累積在記憶體中
    """
    visited_entities = set()
    total_paths_processed = 0
    MAX_TOTAL_PATHS = 5000
    
    def bfs_path_incremental(start_node, max_depth, callback):
        """使用BFS增量搜索路徑，每找到一批就回調處理"""
        from collections import deque
        
        queue = deque([(start_node, [], set())])  # (current_node, path, visited)
        batch_paths = []
        paths_found = 0
        MAX_PATHS_PER_ENTITY = 1000
        
        while queue and paths_found < MAX_PATHS_PER_ENTITY and total_paths_processed < MAX_TOTAL_PATHS:
            current_node, current_path, visited = queue.popleft()
            
            # 如果路徑長度達到限制，跳過
            if len(current_path) >= max_depth:
                continue
                
            # 如果當前路徑不為空，加入批次
            if len(current_path) > 0:
                batch_paths.append(current_path.copy())
                paths_found += 1
                
                # 當批次達到指定大小時，立即處理
                if len(batch_paths) >= batch_size:
                    callback(batch_paths)
                    batch_paths.clear()
                
            # 遍歷當前節點的所有鄰居，限制鄰居數量
            neighbors = graph[current_node][:50]  # 限制每個節點最多50個鄰居
            for neighbor, relation, direction in neighbors:
                if neighbor not in visited:  # 避免環路
                    # 優化：避免深拷貝
                    new_path = current_path + [(neighbor, relation, direction)]
                    new_visited = visited | {neighbor}
                    queue.append((neighbor, new_path, new_visited))
        
        # 處理剩餘的批次
        if batch_paths:
            callback(batch_paths)
    
    # 路徑處理回調函數
    def process_path_batch(paths):
        nonlocal total_paths_processed
        total_paths_processed += len(paths)
        # 這裡可以立即格式化並處理路徑，而不是累積在記憶體中
        return [format_path_to_string(path) for path in paths]
    
    # 從query entities開始動態BFS
    for query_entity in query_entities:
        if query_entity in graph and query_entity not in visited_entities:
            bfs_path_incremental(query_entity, max_path_length, process_path_batch)
            visited_entities.add(query_entity)
            
            # 如果已經找到足夠路徑，提前退出
            if total_paths_processed >= MAX_TOTAL_PATHS:
                break
    
    # 如果沒有從query entities找到路徑，從其他節點開始（限制數量）
    if total_paths_processed == 0:
        start_nodes = list(graph.keys())[:10]  # 限制起始節點數量
        for start_node in start_nodes:
            if start_node not in visited_entities:
                bfs_path_incremental(start_node, max_path_length, process_path_batch)
                visited_entities.add(start_node)
                
                # 如果已經找到足夠路徑，提前退出
                if total_paths_processed >= MAX_TOTAL_PATHS:
                    break
    
    return total_paths_processed


def find_paths_from_graph(graph, query_entities, max_path_length=5):
    """
    從圖中動態生成所有相關路徑
    基於ReG方法的結構感知重組模塊，從query entities開始動態BFS
    不預先指定路徑數量，讓算法自動探索所有相關連接
    """
    all_paths = []
    visited_entities = set()
    
    # 限制最大路徑數量，防止CPU過載
    MAX_PATHS_PER_ENTITY = 1000
    MAX_TOTAL_PATHS = 5000
    
    def bfs_path(start_node, max_depth):
        """使用BFS從起始節點開始動態搜索路徑，優化版本"""
        from collections import deque
        
        queue = deque([(start_node, [], set())])  # (current_node, path, visited)
        paths_found = 0
        
        while queue and paths_found < MAX_PATHS_PER_ENTITY and len(all_paths) < MAX_TOTAL_PATHS:
            current_node, current_path, visited = queue.popleft()
            
            # 如果路徑長度達到限制，跳過
            if len(current_path) >= max_depth:
                continue
                
            # 如果當前路徑不為空，保存路徑
            if len(current_path) > 0:
                all_paths.append(current_path.copy())
                paths_found += 1
                
            # 遍歷當前節點的所有鄰居，限制鄰居數量
            neighbors = graph[current_node][:50]  # 限制每個節點最多50個鄰居
            for neighbor, relation, direction in neighbors:
                if neighbor not in visited:  # 避免環路
                    # 優化：避免深拷貝，使用frozenset和tuple
                    new_path = current_path + [(neighbor, relation, direction)]
                    new_visited = visited | {neighbor}
                    queue.append((neighbor, new_path, new_visited))
    
    # 從query entities開始動態BFS
    for query_entity in query_entities:
        if query_entity in graph and query_entity not in visited_entities:
            bfs_path(query_entity, max_path_length)
            visited_entities.add(query_entity)
            
            # 如果已經找到足夠路徑，提前退出
            if len(all_paths) >= MAX_TOTAL_PATHS:
                break
    
    # 如果沒有從query entities找到路徑，從其他節點開始（限制數量）
    if len(all_paths) == 0:
        start_nodes = list(graph.keys())[:10]  # 限制起始節點數量
        for start_node in start_nodes:
            if start_node not in visited_entities:
                bfs_path(start_node, max_path_length)
                visited_entities.add(start_node)
                
                # 如果已經找到足夠路徑，提前退出
                if len(all_paths) >= MAX_TOTAL_PATHS:
                    break
    
    return all_paths


def format_path_to_string(path):
    """
    將路徑轉換為字符串格式
    格式: (entity1,relation1,entity2) -> (entity2,relation2,entity3) -> ...
    """
    if not path:
        return ""
    
    # 重新構建路徑，從起始節點開始
    if len(path) == 1:
        # 單個節點
        entity, relation, direction = path[0]
        return f"({entity})"
    
    # 構建完整的triplet路徑
    triplets = []
    current_entity = None
    
    for i, (entity, relation, direction) in enumerate(path):
        if i == 0:
            # 第一個節點作為起始點
            current_entity = entity
        else:
            if direction == 'forward':
                # 正向連接: current_entity -> entity
                triplets.append(f"({current_entity},{relation},{entity})")
                current_entity = entity
            else:
                # 反向連接: entity -> current_entity
                triplets.append(f"({entity},{relation},{current_entity})")
                current_entity = entity
    
    return " -> ".join(triplets)


def check_memory_usage():
    """檢查記憶體使用情況，返回是否記憶體不足"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # 如果記憶體使用超過80%，返回True
        if memory_percent > 80:
            print(f"Warning: High memory usage detected: {memory_percent:.1f}%")
            return True
        return False
    except:
        return False


def generate_structure_aware_paths_streaming(triplets, query_entities, max_path_length=4, batch_size=100):
    """
    流式生成結構感知路徑，分批處理避免記憶體過載
    每找到一批路徑就立即格式化並返回，不累積在記憶體中
    """
    if not triplets:
        return []
    
    # 檢查初始記憶體使用
    if check_memory_usage():
        print("High memory usage detected at start, using conservative settings")
        batch_size = min(batch_size, 20)
        max_path_length = min(max_path_length, 3)
    
    # 限制triplets數量，防止過大的圖
    if len(triplets) > 10000:
        print(f"Warning: Large graph detected ({len(triplets)} triplets). Sampling to prevent CPU overload.")
        import random
        triplets = random.sample(triplets, 10000)
    
    # 構建圖結構
    graph = build_graph_from_triplets(triplets)
    
    # 檢查圖的大小，如果太大則提前返回
    total_edges = sum(len(neighbors) for neighbors in graph.values())
    if total_edges > 100000:  # 100K edges threshold
        print(f"Warning: Very large graph detected ({total_edges} edges). Using fallback to prevent CPU overload.")
        # 回退到簡單的triplet格式
        return [f"({t[0]},{t[1]},{t[2]})" for t in triplets[:100]]
    
    # 使用增量BFS生成路徑
    formatted_paths = []
    MAX_OUTPUT_PATHS = 2000  # 限制最終輸出的路徑數量
    
    def process_path_batch(paths):
        """處理路徑批次，立即格式化並累積"""
        nonlocal formatted_paths
        
        # 檢查記憶體使用
        if check_memory_usage():
            print("Memory pressure detected, reducing batch size")
            gc.collect()  # 強制垃圾回收
        
        for path in paths:
            if len(formatted_paths) >= MAX_OUTPUT_PATHS:
                return False  # 停止處理
            formatted_path = format_path_to_string(path)
            if formatted_path:
                formatted_paths.append(formatted_path)
        return True  # 繼續處理
    
    try:
        # 使用增量BFS
        find_paths_from_graph_incremental(graph, query_entities, max_path_length, batch_size)
    except MemoryError:
        print("Memory error during path generation, falling back to simple triplets")
        return [f"({t[0]},{t[1]},{t[2]})" for t in triplets[:100]]
    except Exception as e:
        print(f"Error during path generation: {e}")
        return [f"({t[0]},{t[1]},{t[2]})" for t in triplets[:100]]
    
    return formatted_paths


def generate_structure_aware_paths(triplets, query_entities, max_path_length=4):
    """
    動態生成結構感知的路徑
    基於ReG方法的實現，從query entities開始動態BFS
    不預先指定路徑數量，讓算法自動探索所有相關連接
    """
    if not triplets:
        return []
    
    # 限制triplets數量，防止過大的圖
    if len(triplets) > 10000:
        print(f"Warning: Large graph detected ({len(triplets)} triplets). Sampling to prevent CPU overload.")
        import random
        triplets = random.sample(triplets, 10000)
    
    # 構建圖結構
    graph = build_graph_from_triplets(triplets)
    
    # 檢查圖的大小，如果太大則提前返回
    total_edges = sum(len(neighbors) for neighbors in graph.values())
    if total_edges > 100000:  # 100K edges threshold
        print(f"Warning: Very large graph detected ({total_edges} edges). Using fallback to prevent CPU overload.")
        # 回退到簡單的triplet格式
        return [f"({t[0]},{t[1]},{t[2]})" for t in triplets[:100]]
    
    # 動態生成路徑，從query entities開始BFS
    paths = find_paths_from_graph(graph, query_entities, max_path_length=max_path_length)
    
    # 格式化路徑，限制輸出數量
    formatted_paths = []
    MAX_OUTPUT_PATHS = 2000  # 限制最終輸出的路徑數量
    
    for i, path in enumerate(paths):
        if i >= MAX_OUTPUT_PATHS:
            break
        formatted_path = format_path_to_string(path)
        if formatted_path:
            formatted_paths.append(formatted_path)
    
    return formatted_paths
##############################################################################################################################\

def get_prompts(each_qa, mode, sys_prompt, cot_prompt, thres, seed=0):
    question_prompt = "Question:\n" + each_qa['question']
    if question_prompt[-1] != '?':
        question_prompt += '?'

    if 'rog' in mode:
        num_sampled_triplets = int(mode.split('_')[1])
        good_triplets_rog = each_qa['good_triplets_rog']
        input_triplets = remove_same_head_tail(good_triplets_rog, mode)
        # sampled_triplets = np.array(each_qa[f'sampled_triplets_{num_sampled_triplets}'])
        # input_triplets = np.concatenate([good_triplets_rog, sampled_triplets]) if len(good_triplets_rog) > 0 else sampled_triplets
        input_triplets = [triplet_to_str(triplet) for triplet in input_triplets]
        other_triplets = remove_same_head_tail(each_qa['scored_triplets'], mode)
        other_triplets = [triplet_to_str(triplet) for triplet in other_triplets]
        input_triplets = unique_preserve_order(input_triplets + other_triplets)
        input_triplets = input_triplets[:num_sampled_triplets]
        # input_triplets = np.random.permutation(input_triplets)
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)
    elif 'scored' in mode:
        num_sampled_triplets = int(mode.split('_')[1])
        input_triplets = each_qa['scored_triplets']
        if thres:
            input_triplets = [(triplet[0], triplet[1], triplet[2]) for triplet in input_triplets if triplet[3] >= thres]
        else:
            input_triplets = [(triplet[0], triplet[1], triplet[2]) for triplet in input_triplets]

        input_triplets = unique_preserve_order(input_triplets)
        input_triplets = input_triplets[:num_sampled_triplets]
        input_triplets = [triplet_to_str(triplet) for triplet in input_triplets]
        if 'rev' in mode:
            input_triplets.reverse()
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)

    elif 'rand' in mode:
        num_sampled_triplets = int(mode.split('_')[1])
        np.random.seed(seed)
        input_triplets = np.random.permutation(np.array(each_qa['graph']))
        if 'randNoA' in mode:
            for each_a in each_qa['a_entity']:
                input_triplets = [triplet for triplet in input_triplets if each_a not in triplet[0] and each_a not in triplet[2]]

        input_triplets = unique_preserve_order([triplet_to_str(triplet) for triplet in input_triplets])
        input_triplets = input_triplets[:num_sampled_triplets]
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)
    ##############################################################################################################################/
    elif 'struct' in mode:
        # 解析最大路徑長度，如果沒有指定則使用默認值4
        max_path_length = int(mode.split('_')[2]) if len(mode.split('_')) > 2 else 4
        
        # 從graph中獲取triplets
        input_triplets = each_qa['graph']
        
        # 獲取query entities (a_entity)
        query_entities = each_qa.get('a_entity', [])
        
        # 使用流式處理生成結構感知路徑，避免記憶體過載
        try:
            structure_paths = generate_structure_aware_paths_streaming(
                input_triplets, 
                query_entities,
                max_path_length=max_path_length,
                batch_size=50  # 小批次處理
            )
        except Exception as e:
            print(f"Warning: Struct-aware path generation failed: {e}")
            print("Falling back to simple triplet format")
            structure_paths = []
        
        if structure_paths:
            triplet_prompt = "Paths:\n" + "\n".join(structure_paths)
        else:
            # 如果沒有生成路徑，回退到原始triplet格式
            input_triplets = [triplet_to_str(triplet) for triplet in input_triplets]
            triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)
    ##############################################################################################################################\
            
    elif 'noevi' in mode:
        triplet_prompt = ''
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if 'firstq' in mode:
        all_query = "\n\n".join([sys_prompt, question_prompt, triplet_prompt])
        user_query = "\n\n".join([question_prompt, triplet_prompt])
    else:
        all_query = "\n\n".join([sys_prompt, triplet_prompt, question_prompt])
        user_query = "\n\n".join([triplet_prompt, question_prompt])
        if triplet_prompt == '':
            user_query = question_prompt

    each_qa['sys_query'] = sys_prompt
    each_qa['user_query'] = user_query
    each_qa['all_query'] = all_query
    each_qa['cot_query'] = cot_prompt
    return each_qa


def get_prompts_for_data(data, mode, sys_prompt, cot_prompt, thres):
    new_data = []
    iterator = data
    if 'struct' in mode:
        iterator = tqdm(data, total=len(data), desc="Generating struct-aware prompts")
    for each_qa in iterator:
        new_data.append(get_prompts(each_qa, mode, sys_prompt, cot_prompt, thres))
    return new_data
