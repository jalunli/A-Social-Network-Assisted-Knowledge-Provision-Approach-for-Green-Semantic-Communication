import random
import networkx as nx
import community as community_louvain

def minmax_normalize(values):
    if not values or len(set(values)) == 1:
        return [0.0 for _ in values]
    min_val = min(values)
    max_val = max(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

def SCSR(g, node_attributes, K):
    # Lovain Clustering
    partition = community_louvain.best_partition(g)  # 返回 {node: community_id}
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)
    social_scores = {}
    for comm_id, nodes in communities.items():
        subgraph = g.subgraph(nodes).copy()
        # scaling factor
        s = 20
        degree_centrality = nx.degree_centrality(subgraph)        
        betweenness_centrality = nx.betweenness_centrality(subgraph)
        dc_values = [degree_centrality[node] for node in nodes]
        bc_values = [betweenness_centrality[node] for node in nodes]
        dc_normalized = minmax_normalize(dc_values)
        bc_normalized = minmax_normalize(bc_values)
        for i, node in enumerate(nodes):
            dc = dc_normalized[i]
            bc = bc_normalized[i]
            social_score = s * 0.4 * dc + s * 0.6 * bc  
            social_scores[node] = round(social_score, 4)  
    for node,score in social_scores.items():
        node_attributes[str(node)]['social_score'] = score
    sorted_nodes = sorted(social_scores.items(), key=lambda x: x[1], reverse=True)
    selected_nodes = sorted_nodes[:K]
    result = {node: random.choice([0, 1]) for node in selected_nodes}    
    return result

def SCS_GA(SCSR_result,node_attributes,threshold):
    result = {}
    for node,update in SCSR_result.items():
        # scaling factor
        s = 10
        # Knowledge-Similarity Based Score
        KS = node_attributes[str(node)]['social_score'] * node_attributes[str(node)]['model_dimension']*node_attributes[str(node)]['model_performance']*10
        if KS < threshold:
            update = 1
        else:
            update = 0
        result[node] = update
    return result

def CPUS0(g,node_attributes,K):
    valid_nodes = []
    for node in g.nodes:
        valid_nodes.append((node, node_attributes[str(node)]['snr']))
    valid_nodes.sort(key=lambda x: x[1], reverse=True)
    selected_nodes = [node for node, _ in valid_nodes[:K]]
    result = {node: 0 for node in selected_nodes}
    return result
def CPUS1(g,node_attributes,K):
    valid_nodes = []
    for node in g.nodes:
        valid_nodes.append((node, node_attributes[str(node)]['snr']))
    valid_nodes.sort(key=lambda x: x[1], reverse=True)
    selected_nodes = [node for node, _ in valid_nodes[:K]]
    result = {node: 1 for node in selected_nodes}
    return result
def CPUSR(g,node_attributes,K):
    valid_nodes = []
    for node in g.nodes:
        valid_nodes.append((node, node_attributes[str(node)]['snr']))
    valid_nodes.sort(key=lambda x: x[1], reverse=True)
    selected_nodes = [node for node, _ in valid_nodes[:K]]
    result = {node: random.choice([0, 1]) for node in selected_nodes}
    return result
def RUS1(g,node_attributes,K):
    # randomly select K nodes
    selected_nodes = random.sample(list(g.nodes), K)
    # all gradient allocated
    result = {node: 1 for node in selected_nodes}
    return result
def RUS0(g,node_attributes,K):
    # randomly select K nodes
    selected_nodes = random.sample(list(g.nodes), K)
    # all gradient allocated
    result = {node: 0 for node in selected_nodes}
    return result
def RUSR(g,node_attributes,K):
    # randomly select K nodes
    selected_nodes = random.sample(list(g.nodes), K)
    # all gradient allocated
    result = {node: random.choice([0, 1]) for node in selected_nodes}
    return result
G = nx.Graph()
if __name__ == '__main__':
    G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # 创建节点属性字典（这里用随机数作为示例属性）
    attributes = {node: random.randint(1, 100) for node in G.nodes}
    
    # 选择5个节点
    selected = RUS1(G, attributes, 5)
    
    print("选择的节点及是否更新梯度:")
    for node, update in selected.items():
        print(f"节点 {node}: {'更新梯度' if update == 1 else '不更新梯度'}")