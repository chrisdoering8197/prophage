import os
import pickle
import graph_tool.all as gt
import networkx as nx
import argparse
from cdlib import algorithms
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
HOME = os.path.expandvars('$HOME')+'/'

def leiden_calc(gt_graph_path,image_out_folder,out_name,nx_graph_path):
    
    print('Loading graph-tools graph...')
    gt_graph = gt.load_graph(gt_graph_path)

    if not nx_graph_path:
        print('Converting graph-tools graph to networkx...')
        nx_graph = nx.Graph()

        for v in gt_graph.vertices():
            u = gt_graph.vp['name'][v]
            sfpd_pos = (gt_graph.vp['SFDP_pos'][v][0],gt_graph.vp['SFDP_pos'][v][1])
            block_comm = gt_graph.vp['block_state'][v]
            nx_graph.add_node(u,sfpd_pos=sfpd_pos,block_comm=block_comm)

        for e in gt_graph.edges():
            u, v = gt_graph.vp['name'][e.source()], gt_graph.vp['name'][e.target()]
            weight = gt_graph.ep['weight'][e]
            alpha = gt_graph.ep['alpha'][e]
            nx_graph.add_edge(u,v,weight=weight,alpha=alpha)
    else:
        print('Loading networkx graph...')
        with open(nx_graph_path, 'rb') as f:
            nx_graph = pickle.load(f)
        
    print('Calculating Leiden communities...')
    leiden_comms =  algorithms.leiden(nx_graph,weights='weight')
    communities = leiden_comms.communities
    for comm_num, community in enumerate(communities):
        for node in community:
            nx_graph.nodes[node]['leiden_comm'] = comm_num
        
    print('Adding Leiden community information to graph-tool graph...')
    leiden_communities = {node: data['leiden_comm'] for node, data in nx_graph.nodes(data=True)}

    #Map leiden states in networkx graph to graph-tools
    leiden_state = gt_graph.new_vp("double")
    leiden_state_labels = gt_graph.new_vp('string')
    for v in gt_graph.vertices():
        leiden_comm_number = leiden_communities[gt_graph.vp['name'][v]]
        leiden_state[v] = leiden_comm_number
        leiden_state_labels[v] = str(leiden_comm_number)
    gt_graph.vp['leiden_state'] = leiden_state

    # Normalize block IDs to a 0–1 range for colormap mapping
    norm = Normalize(vmin=min(leiden_state.a), vmax=max(leiden_state.a))
    colormap = plt.get_cmap("hsv") 

    # Map leiden communities to colors
    vertex_colors = gt_graph.new_vp("vector<double>")
    for v in gt_graph.vertices():
        vertex_colors[v] = colormap(norm(leiden_state[v]))[:3]

    print('Saving graph-tools graph...')
    gt_graph.save(f'{os.path.splitext(gt_graph_path)[0]}_Leiden.gt.gz')
        
    print('Plotting...')
    gt.graph_draw(gt_graph,pos=gt_graph.vp['SFDP_pos'],
            output=os.path.join(image_out_folder,f'{out_name}_leiden.png'),
            vertex_fill_color=vertex_colors,
            output_size=(5000, 5000),
            edge_color=gt_graph.ep['alpha'])

    gt.graph_draw(gt_graph,pos=gt_graph.vp['SFDP_pos'],
            output=os.path.join(image_out_folder,f'{out_name}_leiden_textlabeled.png'),
            edge_pen_width=0,output_size=(10000, 10000),
            vertex_text_color='black',vertex_fill_color='white',
            vertex_text=leiden_state_labels)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering of phage genomes by shared protein clusters, visualized with spring-force directed graphs.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gt_graph', type=str, help='Path to graph-tool graph on which to perform Leiden clustering')
    parser.add_argument('out_folder', type=str, help='Path to folder to save image results')
    parser.add_argument('out_name', type=str, help='Name to save image results under')
    parser.add_argument('--nx_graph', type=str, help='Path to networkx graph to use instead of converting graph-tool graph if available')


    args = parser.parse_args()
    leiden_calc(args.gt_graph,args.out_folder,args.out_name,args.nx_graph)