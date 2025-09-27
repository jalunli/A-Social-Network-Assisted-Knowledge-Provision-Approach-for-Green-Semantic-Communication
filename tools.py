import math
import random
import networkx as nx
import community as community_louvain
import pandas as pd

# energy consumed in computing
def Ecomp(f,msize,dsize):
    ecomp = (msize+dsize)*(f*f)*(1e-28)
    return ecomp

# energy consumed in transmission
def Etrans(p,t):
    etrans = p*t
    return etrans

# computation time
def Tcomp(f,msize,dsize):
    tcomp = (msize+dsize)/(f)
    return tcomp

# transmission time
def Ttrans(bandwidth,snr,data_size):
    snr_linear = 10 **(snr / 10)
    rate = bandwidth * math.log2(1 + snr_linear)
    ttrans = (data_size)*8/(rate)
    return ttrans

# initialize node attributes
def init_node_attributes(num_nodes):
    node_dict = {}
    
    for node_id in range(0, num_nodes):
        # generate properties
        cpu_freq = round(random.uniform(0.5, 2.5), 2)*1e9 
        snr = round(random.uniform(-1, 5), 2) # dB          
        community_id = 0  
        degree = 0                 
        model_size = round(random.uniform(0.4, 1.2), 2)*1e9 
        dataset_size = round(random.uniform(1, 4), 2)*1e9  
        model_performance = round(random.uniform(40, 60), 2)
        model_dimension = round(random.uniform(5, 35), 2)*1e3
        degree_centrality = 0
        betweeness_centrality = 0
        social_score = 0
        # 添加到字典
        node_dict[node_id] = {
            "cpu_frequency": cpu_freq,
            "snr": snr,
            "community_id": community_id,
            "degree": degree,
            "model_size": model_size,
            "dataset_size": dataset_size,
            'model_performance': model_performance,
            "model_dimension": model_dimension,
            "degree_centrality": degree_centrality,
            "betweeness_centrality": betweeness_centrality,
            "social_score": social_score
        }
    return node_dict

# normalize function
def normalize(value, min_val, max_val):
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)

# calculate semantic qos
def calculate_qos(snr, model_performance, gradient_update, social_score,
                 snr_range=(-5, 1),
                 performance_range=(0, 100),
                 social_score_range=(0, 1)):
    
    # 对各参数进行归一化
    snr_norm = normalize(snr, snr_range[0], snr_range[1])
    perf_norm = normalize(model_performance, performance_range[0], performance_range[1])
    social_norm = normalize(social_score, social_score_range[0], social_score_range[1])
    
    # 计算各部分得分（按权重分配）
    snr_score = snr_norm * 0.05
    performance_score = perf_norm * 0.05
    gradient_score = gradient_update * 0.4  
    social_score = social_norm * 0.5
    
    # 计算总QoS得分
    total_qos = snr_score + performance_score + gradient_score + social_score
    
    # 确保结果在0-1范围内（处理可能的浮点误差）
    return round(max(0.0, min(1.0, total_qos)), 4)

def community_clustering_with_centrality(G, target_k, node_dict, degree_weight=200, betweenness_weight=2000):
   
    # clutering
    communities = community_clustering_with_k(G, target_k)

    # centrality measure
    degree_centrality = nx.degree_centrality(G)  # 度数中心性（归一化到[0,1]）
    betweenness_centrality = nx.betweenness_centrality(G)  # 介数中心性（归一化到[0,1]）
    
    # 3. 处理每个社区
    result = {}
    for comm_idx, community in enumerate(communities, 1):
        # 计算社区内每个节点的综合得分
        node_info = []
        for node in community:
            # 加权求和（确保权重和为1）
            combined_score = (degree_centrality[node] * degree_weight + 
                             betweenness_centrality[node] * betweenness_weight)
            node_info.append({
                'node': node,
                'degree_centrality': round(degree_centrality[node], 4),
                'betweenness_centrality': round(betweenness_centrality[node], 4),
                'combined_score': round(combined_score, 4)
            })
            node_dict[node]['community_id'] = comm_idx
            node_dict[node]['degree_centrality'] = round(degree_centrality[node], 4)
            node_dict[node]['betweenness_centrality'] = round(betweenness_centrality[node], 4)
            node_dict[node]['social_score'] = round(combined_score, 4)
        
        # Arrange in descending order of social score
        node_info_sorted = sorted(node_info, key=lambda x: x['combined_score'], reverse=True)
        
        # store results
        result[f"community_{comm_idx}"] = {
            'size': len(community),
            'nodes_ranked': node_info_sorted
        }
    
    return result

def community_clustering_with_k(G, target_k):

    if target_k >= G.number_of_nodes():
        raise ValueError("The number of target communities cannot exceed the total number of nodes.")
    
    # Louvain clustering
    partition = community_louvain.best_partition(G)
    initial_communities = {}
    for node, comm_id in partition.items():
        if comm_id not in initial_communities:
            initial_communities[comm_id] = []
        initial_communities[comm_id].append(node)
    communities = list(initial_communities.values())
    current_k = len(communities)
    print(f" {current_k} latent communites detected, adjusted to {target_k} target communities")
    
    # adjust community number
    if current_k > target_k:
        while len(communities) > target_k:
            communities.sort(key=lambda x: len(x))
            merged = communities[0] + communities[1]
            communities = [merged] + communities[2:]
    elif current_k < target_k:
        while len(communities) < target_k:
            communities.sort(key=lambda x: len(x), reverse=True)
            largest = communities[0]
            if len(largest) < 2:
                raise ValueError("Unable to split communities with only one node; please reduce the target quantity.")
            
            subgraph = G.subgraph(largest)
            node_degrees = [(node, subgraph.degree(node)) for node in largest]
            node_degrees.sort(key=lambda x: x[1])
            split_idx = max(1, len(largest) // 2)
            part1 = [node for node, _ in node_degrees[:split_idx]]
            part2 = [node for node, _ in node_degrees[split_idx:]]
            communities = [part1, part2] + communities[1:]
    
    return communities

def policy_calculate(policy_result,node_attributes,bandwidth_limit=20e6,ft_limit=4,bt_limit=4):
    # iterate every node:
    total_energy = 0
    total_trans_energy = 0
    total_comp_energy = 0
    total_qos = 0
    batch_size = 1000
    f_d = 100
    for node, update in policy_result.items():
        ftcomp = 0
        ftrans = 0
        btrans = 0
        bcomp = 0
        node_str = str(node)
        # calculate forward-computation time 
        ftcomp = Tcomp(f=node_attributes[node_str]['cpu_frequency'],msize=node_attributes[node_str]['model_size'],dsize=node_attributes[node_str]['dataset_size'])
        # calculate forward-transmit time
        ftrans = Ttrans(bandwidth_limit,snr=node_attributes[node_str]['snr'],data_size=batch_size*f_d)
        # calculate f-trans process
        total_trans_energy += Etrans(p=0.2,t=ftrans)
        # calculate f-se process
        total_comp_energy += Ecomp(f=node_attributes[node_str]['cpu_frequency'],msize=node_attributes[node_str]['model_size'],dsize=node_attributes[node_str]['dataset_size'])
        # compare forward time limit
        if ftrans + ftcomp > ft_limit:
            qos = 0
        else:
            if update == 1:
                # compare bandwidth limit
                # calculate b-trans process
                if (batch_size*node_attributes[node_str]['model_dimension'] > bandwidth_limit):
                    # add retransmission penalty
                    total_trans_energy += Etrans(p=0.2,t=btrans)
                    btrans = Ttrans(bandwidth_limit,snr=node_attributes[node_str]['snr'],data_size=batch_size*node_attributes[node_str]['model_dimension'])
                    total_trans_energy += Etrans(p=0.2,t=btrans)
                    qos = 0
                else:
                    btrans = Ttrans(bandwidth_limit,snr=node_attributes[node_str]['snr'],data_size=batch_size*node_attributes[node_str]['model_dimension'])
                    # calculate b-comp time
                    bcomp = Tcomp(f=node_attributes[node_str]['cpu_frequency'],msize=node_attributes[node_str]['model_size'],dsize=batch_size*node_attributes[node_str]['model_dimension'])
                    # calculate b-comp process
                    total_comp_energy += Ecomp(f=node_attributes[node_str]['cpu_frequency'],msize=node_attributes[node_str]['model_size'],dsize=batch_size*node_attributes[node_str]['model_dimension'])
                    total_trans_energy += Etrans(p=0.2,t=btrans)
                    # compare b-time limit
                    if (btrans + bcomp) > bt_limit:
                        qos = 0
                    else:
                        qos = calculate_qos(node_attributes[node_str]['snr'],
                                        node_attributes[node_str]['model_performance'],
                                        update,
                                        node_attributes[node_str]['social_score'])
            else:
                qos = calculate_qos(node_attributes[node_str]['snr'],
                                    node_attributes[node_str]['model_performance'],
                                    update,
                                    node_attributes[node_str]['social_score'])
                # 0-update penalty
                total_trans_energy +=Etrans(p=0.2,t=ftrans)
                # calculate f-se process
                total_comp_energy +=Ecomp(f=node_attributes[node_str]['cpu_frequency'],msize=node_attributes[node_str]['model_size'],dsize=node_attributes[node_str]['dataset_size'])
          
        total_qos += qos
    total_energy =total_trans_energy + total_comp_energy
    energy_efficiency = total_qos / total_energy
    return energy_efficiency
if __name__ == '__main__':
    #pre-defining area
    ss = social_score = 0.9
    ms = model_score = 0.9
    ds = dataset_score = 0.9
    msize = 1.2e9
    dsize = 1e9
    model_dimension = 1e5
    f = device_frequency = 1e9
    p = transmit_power = 0.2
    d = data_batch = 1e+6
    bandwidth = 20e6
    snr = 1
