import os
import argparse
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from joblib import Parallel,delayed
from tqdm import tqdm
import graph_tool.all as gt
HOME = os.path.expandvars('$HOME')+'/'
THREADS=48
SMALLEST_GRAPH = 100

def self_cluster(out_name,out_folder,in_protein,in_genome,gene2genome_file,TMPDIR):
    prot_clust_tsv = os.path.join(out_folder,f'{out_name}_proteins_cluster.tsv')
    if not os.path.isfile(prot_clust_tsv):
        #Cluster all phage proteins at 25% identity and 80% coverage
        clust_cmd = ['conda run -n MyEnv','mmseqs','easy-cluster',os.path.join(out_folder,f'{out_name}_proteins'),in_protein,TMPDIR,
                '--min-seq-id','0.25','-c','0.8']
        os.system(' '.join(clust_cmd))
    else:
        print('Found protein cluster file, reusing...')

    #Read in protein clustering and save as dictionaries
    prot_clustDF = pd.read_csv(prot_clust_tsv,sep='\t',names=['rep_seq','member'])
    prot2rep = dict(zip(prot_clustDF['member'],prot_clustDF['rep_seq']))
    rep2mem = prot_clustDF.groupby('rep_seq')['member'].apply(list).to_dict()
    protclust_sizes = prot_clustDF['rep_seq'].value_counts().to_dict()

    print('Loading needed tables...')
    #Cluster phage genomes strictly based on DNA similarity
    phage_genome_clu_tsv = os.path.join(out_folder,f'{out_name}_cluster.tsv')
    if not os.path.isfile(phage_genome_clu_tsv):
        #Cluster all phage genomes at 95% identity and 95% coverage
        clust_cmd = ['conda run -n MyEnv','mmseqs','easy-cluster',os.path.join(out_folder,f'{out_name}'),in_genome,TMPDIR,
                '--min-seq-id','0.95','-c','0.95','--cluster-mode','1']
        os.system(' '.join(clust_cmd))
    else:
        print('Found genome cluster file, reusing...')
        
    #Read strict DNA clustering in as dictionary
    phage_genome_clu = pd.read_csv(phage_genome_clu_tsv,sep='\t',names=['rep_genome','mem_genome'])
    phage_genome_clu = phage_genome_clu.groupby('rep_genome')['mem_genome'].count().to_dict()
    total_genomes = len(phage_genome_clu)

    gene2genomeDF = pd.read_csv(gene2genome_file)
    genome2gene = gene2genomeDF.groupby('contig_id')['protein_id'].apply(list).to_dict()
    gene2genome = dict(zip(gene2genomeDF['protein_id'],gene2genomeDF['contig_id']))

    #Create dictionary to get protein clusters present in a genome if the genome shares at least one protein cluster with another genome                                     
    genome2PC = {genome:{prot2rep[protein] for protein in prot_list} 
                for genome,prot_list in genome2gene.items()
                if any(protclust_sizes[prot2rep[protein]] > 1 for protein in prot_list)}

    print(f'Filtered {len(phage_genome_clu) - len(genome2PC)} fully singleton genomes')
    #Filter phage_genome_clu dictionary to only include genomes that share at least one protein cluster with another genome
    phage_genome_clu = {genome:clu_size for genome,clu_size in phage_genome_clu.items() if genome in genome2PC}

    #For a set of genome pairs, calculate the edge weight based on the hypergeometric test and only return edges with pval <= pval_threshold
    def edge_weight(genome_pairs_set,total_PC,pval_threshold):
        edges = []
        for genome1, genome2 in genome_pairs_set:
            PC1, PC2 = genome2PC[genome1], genome2PC[genome2]
            overlap = len(PC1 & PC2)
            pval = sp.stats.hypergeom.sf(overlap - 1, total_PC, len(PC1), len(PC2))
            if pval <= pval_threshold:
                edges.append((genome1,genome2,overlap,pval))
        return edges

    #P-value threshold set following those used in vcontact2
    total_PC = len(set(prot_clustDF['rep_seq']))
    pairwise_comparisons = (total_genomes *(total_genomes-1))/2 #Includes all genomes included those filtered at DNA similarity level
    pval_threshold = 0.1/pairwise_comparisons

    edge_data = os.path.join(out_folder,f'{out_name}_edge_table.parquet')
    if not os.path.isfile(edge_data):
        seen = set()
        genome_pairs = []
        print('Indentifying genome pairs to consider...')
        #Create a list of genome pairs to compare where the genomes share at least one protein cluster
        for genome, prot_set in tqdm(genome2PC.items()):
            seen.add(genome)
            genome2compare = {gene2genome[mem_prot] for protein in prot_set for mem_prot in rep2mem[protein]}
            genome2compare -= seen
            genome_pairs.append([frozenset([genome,compare]) for compare in genome2compare])
            
        print('Calculating genome pair weights...')
        edges=Parallel(n_jobs=THREADS)(delayed(edge_weight)(pairs_list,total_PC,pval_threshold) 
                                    for pairs_list in tqdm(genome_pairs))

        edges = [edge for edges_group in edges for edge in edges_group]
        edgesDF = pd.DataFrame(edges,columns=['Genome1','Genome2','Overlap','Pval'])
        edgesDF.to_parquet(edge_data)
    else:
        print('Found edge data, reusing...')
        edgesDF = pd.read_parquet(edge_data)

    graph_data = os.path.join(out_folder,f'{out_name}_prophage.gt.gz')
    if not os.path.isfile(graph_data):
        
        print('Creating graph compatible edge list...')
        #Convert p-values to e-values and initialize graph. Save e-values as edge weights
        zero_equivalent = np.finfo(float).eps #Representing pval=0 as the smallest floating point number
        edges4graph = [(row.Genome1,row.Genome2,-np.log10(max(row.Pval,zero_equivalent)*pairwise_comparisons)) 
                    for row in edgesDF.itertuples()]
        phages_graph = gt.Graph(directed=False)
        index2genome = phages_graph.add_edge_list(edges4graph,hashed=True,eprops=[('weight','double')])
        phages_graph.vp['name'] = index2genome
        
        print('Adding genome cluster sizes to graph...')
        vsize = phages_graph.new_vp("double")
        for v in phages_graph.vertices():
            vsize[v] = phage_genome_clu[phages_graph.vp['name'][v]]
        phages_graph.vp['size'] = vsize
        
        print('Creating color mapping for cluster sizes...')
        #Save cluster sizes as plasma colormap
        plasma_colormap = plt.get_cmap('plasma')
        size_values = [vsize[v] for v in phages_graph.vertices()]
        norm_color = Normalize(vmin=min(size_values),vmax=max(size_values))
        vcolor = phages_graph.new_vp('vector<double>')
        for v in phages_graph.vertices():
            norm_value = norm_color(vsize[v])
            rbga_color = plasma_colormap(norm_value)
            vcolor[v] = rbga_color[:3]
        phages_graph.vp['color'] = vcolor
        
        print('Calculating edge alpha...')
        #Save normalize e-values/weights as edge alpha normalized to largest weight
        edge_alpha = phages_graph.new_ep('vector<double>')
        max_weight = max(phages_graph.ep['weight'][e] for e in phages_graph.edges())
        for e in phages_graph.edges():
            alpha = phages_graph.ep['weight'][e] / max_weight
            edge_alpha[e] = [0.0,0.0,0.0,alpha]
        phages_graph.ep['alpha'] = edge_alpha
        
        print('Saving graph...')
        phages_graph.save(graph_data)
    else:
        print('Found graph data, reusing...')
        phages_graph = gt.load_graph(graph_data)
        
    component_graphs, component_sizes = gt.label_components(phages_graph)
    print(f'{len(set(component_graphs))} subcomponent graphs present')
    sns.histplot(component_sizes)
    plt.title('Prophage Subgraph Sizes')
    plt.savefig(os.path.join(out_folder,f'{out_name}_subgraph_hist.png'))
    plt.close()

    subgraphs = []
    filtered_vertices = 0
    for component_id, component_size in enumerate(component_sizes):
        if component_size >= SMALLEST_GRAPH:
            subgraph = gt.GraphView(phages_graph,vfilt=lambda v: component_graphs[v] == component_id)
            subgraphs.append(subgraph)
        else:
            filtered_vertices += component_size
    print(f'Filtered {filtered_vertices} genomes belonging to subgraphs smaller than {SMALLEST_GRAPH} genomes')
    print(f'{len(subgraphs)} subgraphs remaining after filtering')

    for i, subgraph in enumerate(subgraphs):
        subgraph_out = os.path.join(out_folder,f'subgraph_plots/{out_name}_prophage_{i}_size_{len(subgraph)}.gt.gz')
        if not os.path.isfile(subgraph_out):
            print(f'Saving subgraph {i}')
            subgraph.save(subgraph_out)
        print(f'Plotting subgraph {i}')
        print('Performing Scalable Force-Directed Placement calculation...')
        SFDP_pos = gt.sfdp_layout(subgraph,eweight=subgraph.ep['weight'],verbose=True)
        print('Performing Fruchterman Reingold calculation...')
        FR_pos = gt.fruchterman_reingold_layout(subgraph,weight=subgraph.ep['weight'],n_iter=50)
        subgraph.vp['FR_pos'] = FR_pos
        subgraph.vp['SFDP_pos'] = SFDP_pos
        subgraph.save(os.path.join(out_folder,f'subgraph_plots/{out_name}_prophage_{i}_size_{len(subgraph)}_with_SFDP_FR.gt.gz'))
        gt.graph_draw(subgraph,pos=FR_pos,edge_color=subgraph.ep['alpha'],
                    output_size=(5000, 5000),output=os.path.join(out_folder,f'subgraph_plots/FR_{out_name}_prophage_{i}_size_{len(subgraph)}.png'))
        gt.graph_draw(subgraph,pos=SFDP_pos,edge_color=subgraph.ep['alpha'],
                output_size=(5000, 5000),output=os.path.join(out_folder,f'subgraph_plots/SFDP_{out_name}_prophage_{i}_size_{len(subgraph)}.png'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering of phage genomes by shared protein clusters, visualized with spring-force directed graphs.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('out_name', type=str, help='Name under which to save results')
    parser.add_argument('out_folder', type=str, help='Path to folder to save results')
    parser.add_argument('in_protein', type=str, help='Path to file containing all phage proteins')
    parser.add_argument('in_genomes', type=str, help='Path to file containing all phage DNA genomes')
    parser.add_argument('gene2genome_file', type=str, help='Path to file containing gene to genome mapping. Proteins much be present in the first column with corresponding genome in the second.')
    parser.add_argument('--TMPDIR', type=str, default=os.path.expandvars('$TMPDIR'), help='Path to temporary directory')

    args = parser.parse_args()
    self_cluster(args.out_name,args.out_folder,args.in_protein,args.in_genome,args.gene2genome_file,args.TMPDIR)