import networkx as nx
import pandas as pd
import math,random
from tools import init_node_attributes
import json

# construct user node property
node_dict = init_node_attributes(4039)

# degree_centrality = nx.degree_centrality(G)  # 度数中心性（归一化到[0,1]）
# betweenness_centrality = nx.betweenness_centrality(G)  # 介数中心性（归一化到[0,1]）
# print('Maximum degree centrality:', max(degree_centrality.values()))


# print(degree_centrality)
# social community detection
# result = community_clustering_with_centrality(
#         G, 
#         target_k=16,
#         node_dict=node_dict,
#         degree_weight=0.6,
#         betweenness_weight=0.4
#     )

# for comm_name, comm_data in result.items():
#         print(f"\n{comm_name}（节点数：{comm_data['size']}）：")
#         print(f"排名 | 节点 | 度数中心性 | 介数中心性 | 综合得分")
#         print("-" * 50)
#         for rank, node_info in enumerate(comm_data['nodes_ranked'], 1):
#             print(f"{rank:4d} | {node_info['node']:4d} | {node_info['degree_centrality']:12.4f} | "
#                   f"{node_info['betweenness_centrality']:14.4f} | {node_info['combined_score']:.4f}")
with open("node_attribute.json", "w", encoding="utf-8") as f:
    # ensure_ascii=False 确保中文正常显示
    json.dump(node_dict, f, ensure_ascii=False, indent=4)

