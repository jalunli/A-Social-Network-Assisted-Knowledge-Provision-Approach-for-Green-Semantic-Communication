import json
import networkx as nx
import pandas as pd
from policy import RUS0,RUS1,RUSR,CPUS0,CPUS1,CPUSR,SCSR,SCS_GA
from tools import policy_calculate

# load network dataset
facebook = pd.read_csv(
        filepath_or_buffer='facebook_combined.txt',
        sep=" ",
        names=["start_node", "end_node"],
    )
# create networkx graph object
G = nx.from_pandas_edgelist(facebook, "start_node", "end_node")

# load node_attributes as a dict
with open("node_attribute.json", "r", encoding="utf-8") as f:
    node_attributes = json.load(f)

# comparing baselines
# social approach
selected = SCSR(G,node_attributes,3000)
result_EESCR = {}
result_EESC0 = {}
result_EESC1 = {}
for node, update in selected.items():
        result_EESCR[node[0]] = update
        result_EESC0[node[0]] = 0
        result_EESC1[node[0]] = 1
result_SCSGA = SCS_GA(result_EESCR,node_attributes,15e6)
result_RUS0 = RUS0(G,node_attributes,3000)
result_RUS1 = RUS1(G,node_attributes,3000)
result_RUSR = RUSR(G,node_attributes,3000)
result_CPUS0 = CPUS0(G,node_attributes,3000)
result_CPUS1 = CPUS1(G,node_attributes,3000)
result_CPUSR = CPUSR(G,node_attributes,3000)

for bandwidth_limit in [10e6,15e6,20e6,25e6,30e6,35e6]:
    print('----------Energy Efficiency Result for bandwidth: ',bandwidth_limit)
    EESCSR = policy_calculate(result_EESCR,node_attributes,bandwidth_limit)
    EESCS0 = policy_calculate(result_EESC0,node_attributes,bandwidth_limit)
    EESCS1 = policy_calculate(result_EESC1,node_attributes,bandwidth_limit)
    EESCSGA = policy_calculate(result_SCSGA,node_attributes,bandwidth_limit)
    EERUS0 = policy_calculate(result_RUS0,node_attributes,bandwidth_limit)
    EERUS1 = policy_calculate(result_RUS1,node_attributes,bandwidth_limit)
    EERUSR = policy_calculate(result_RUSR,node_attributes,bandwidth_limit)
    EECPUS0 = policy_calculate(result_CPUS0,node_attributes,bandwidth_limit)
    EECPUS1 = policy_calculate(result_CPUS1,node_attributes,bandwidth_limit)
    EECPUSR = policy_calculate(result_CPUSR,node_attributes,bandwidth_limit)
    print('Random User Selection with No Gradient Updates: ',EERUS0)
    print('Random User Selection with All Gradient Updates: ',EERUS1)
    print('Random User Selection with Random Gradient Updates: ',EERUSR)
    print('Communication Prefered User Selection with No Gradient Updates: ',EECPUS0)
    print('Communication Prefered User Selection with All Gradient Updates: ',EECPUS1)
    print('Communication Prefered User Selection with Random Gradient Updates: ',EECPUSR)
    print('Social Cooperative User Selection with No Gradient Updates: ',EESCS0)
    print('Social Cooperative User Selection with All Gradient Updates: ',EESCS1)
    print('Social Cooperative User Selection with Random Gradient Updates: ',EESCSR)
    print('Social Cooperative User Selection and Gradient Updates: ',EESCSGA)
for time_limit in [4,6,8,10,12,14]:
    bandwidth_limit = 20e6
    f_limit=b_limit=time_limit/2
    print('----------Energy Efficiency Result for time: ',time_limit)
    EESCSR = policy_calculate(result_EESCR,node_attributes,bandwidth_limit,f_limit,b_limit)
    EESCS0 = policy_calculate(result_EESC0,node_attributes,bandwidth_limit,f_limit,b_limit)
    EESCS1 = policy_calculate(result_EESC1,node_attributes,bandwidth_limit,f_limit,b_limit)
    EESCSGA = policy_calculate(result_SCSGA,node_attributes,bandwidth_limit,f_limit,b_limit)
    EERUS0 = policy_calculate(result_RUS0,node_attributes,bandwidth_limit,f_limit,b_limit)
    EERUS1 = policy_calculate(result_RUS1,node_attributes,bandwidth_limit,f_limit,b_limit)
    EERUSR = policy_calculate(result_RUSR,node_attributes,bandwidth_limit,f_limit,b_limit)
    EECPUS0 = policy_calculate(result_CPUS0,node_attributes,bandwidth_limit,f_limit,b_limit)
    EECPUS1 = policy_calculate(result_CPUS1,node_attributes,bandwidth_limit,f_limit,b_limit)
    EECPUSR = policy_calculate(result_CPUSR,node_attributes,bandwidth_limit,f_limit,b_limit)
    print('Random User Selection with No Gradient Updates: ',EERUS0)
    print('Random User Selection with All Gradient Updates: ',EERUS1)
    print('Random User Selection with Random Gradient Updates: ',EERUSR)
    print('Communication Prefered User Selection with No Gradient Updates: ',EECPUS0)
    print('Communication Prefered User Selection with All Gradient Updates: ',EECPUS1)
    print('Communication Prefered User Selection with Random Gradient Updates: ',EECPUSR)
    print('Social Cooperative User Selection with No Gradient Updates: ',EESCS0)
    print('Social Cooperative User Selection with All Gradient Updates: ',EESCS1)
    print('Social Cooperative User Selection with Random Gradient Updates: ',EESCSR)
    print('Social Cooperative User Selection and Gradient Updates: ',EESCSGA)