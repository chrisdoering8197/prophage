import glob
import os
import argparse
import multiprocessing
import pandas as pd
import numpy as np
import networkx as nx
import shutil
from Bio import SeqIO
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CPU_AVAIL = multiprocessing.cpu_count()

def ProteinClusters(out_name,in_folder,out_folder,annotate_clusters):
    #Protein cluster computation

    #Make folder to store clustering files
    out_folder_cluster = os.path.join(out_folder,'cluster_files/')
    if not os.path.isdir(out_folder_cluster):
        os.mkdir(out_folder_cluster)
    
    clust_result_tsv = os.path.join(out_folder_cluster,out_name+'_cluster.tsv')
    if not os.path.isfile(clust_result_tsv):
        print('Clustering proteins in viral hits with mmseqs')

        #Get all proteins to cluster and write to single fasta file
        phage_genomes = glob.glob(os.path.join(in_folder,'*.faa'))
        viral_seqs = [record for genome in phage_genomes for record in SeqIO.parse(genome,'fasta')]
        fasta_for_cluster = os.path.join(out_folder_cluster,f'{out_name}_all_proteins.faa')
        SeqIO.write(viral_seqs,fasta_for_cluster,'fasta')
        
        #Cluster proteins at 80% coverage requirement and E-value of 0.001 (mmseqs defaults)
        cluster = ['conda run -n MyEnv','mmseqs','easy-cluster',fasta_for_cluster,os.path.join(out_folder_cluster,out_name),'tmp','-v','1','--remove-tmp-files','1']
        os.system(' '.join(cluster))

    else:
        print('Found cluster database and results file, resuing...')
        
    
    clust_results = pd.read_csv(clust_result_tsv,sep='\t',names=['rep_seq','member'])
    
    if annotate_clusters:
        print('Annotating protein clusters with Phrokka and DefenseFinder')

        #Input and output file and folder names
        out_folder_anno = os.path.join(out_folder,'anno_files/')
        anno_out = os.path.join(out_folder_anno,'pharokka_proteins_full_merged_output.tsv')
        defense_out = os.path.join(out_folder_anno,out_name+'_rep_seqs_defense_finder_genes.tsv')

        #Check if annotation files are present, run if not and overwrite any partial results
        if not os.path.isfile(anno_out) or not os.path.isfile(defense_out):
            #Overwrite annotation results folder if present and summary file is not
            if os.path.isdir(out_folder_anno):
                print('Overwriting existing annotation folder')
                shutil.rmtree(out_folder_anno)

            #Load in all protein records
            all_proteins_file = os.path.join(out_folder_cluster,f'{out_name}_all_proteins.faa')
            all_protein_dict = SeqIO.to_dict(SeqIO.parse(all_proteins_file,'fasta'))

            #Write representative sequences to file
            rep_seq_records = [all_protein_dict[ID] for ID in set(clust_results['rep_seq'])]
            for record in rep_seq_records: #Need to remove descriptions because they contain # and will cause pharokka to error
                record.description=''
            clust_rep_file = os.path.join(TMPDIR,f'{out_name}_rep_seqs.faa')
            SeqIO.write(rep_seq_records,clust_rep_file,'fasta')

            #Phrokka protein annotation
            pharokka_db = os.path.join(DBDIR,'pharokka_dbs/')
            cmd = ['conda run -n pharokka','pharokka_proteins.py','-i',clust_rep_file,'-d',pharokka_db,'-o',out_folder_anno,'-t',str(CPU_AVAIL)]
            os.system(' '.join(cmd))

            #Read phrokka annotation summary into pandas
            anno_results = pd.read_csv(anno_out,sep='\t',usecols=['ID','phrog','annot','category','vfdb_hit','CARD_hit'])
            anno_results['vfdb_hit'] = anno_results['vfdb_hit'].replace({np.nan:None,'None':None})
            anno_results['CARD_hit'] = anno_results['CARD_hit'].replace({np.nan:None,'None':None})
            
            #DefenseFinder protein annotation
            cmd = ['conda run -n MyEnv','defense-finder','run','-o',out_folder_anno,'-w',str(CPU_AVAIL),
                   '--log-level','WARNING',clust_rep_file]
            os.system(' '.join(cmd))
            
            #Read DefenseFinder summary into pandas
            defense_results = pd.read_csv(defense_out,sep='\t',usecols=['hit_id','gene_name'])
        else:
            print('Found pharokka annotation and DefenseFinder summaries, reusing...')
            anno_results = pd.read_csv(anno_out,sep='\t',usecols=['ID','phrog','annot','category','vfdb_hit','CARD_hit'])
            anno_results['vfdb_hit'] = anno_results['vfdb_hit'].replace({np.nan:None,'None':None})
            anno_results['CARD_hit'] = anno_results['CARD_hit'].replace({np.nan:None,'None':None})
            defense_results = pd.read_csv(defense_out,sep='\t',usecols=['hit_id','gene_name'])
    else:
        anno_results = None
        defense_results = None

    
    PC_in_genomes = defaultdict(set)
    #All genome annotations were performed with Prodigal-gv and follow the standard naming convention
    for _, row in clust_results.iterrows():
        PC_in_genomes[row['rep_seq']].add(('_').join(row['member'].split('_')[:-1]))
    PC_in_genomes = {k:len(v) for k,v in PC_in_genomes.items()}
    PC_in_genomes = pd.DataFrame.from_dict(PC_in_genomes,orient='index')
    PC_in_genomes.to_csv(os.path.join(out_folder,f'{out_name}_PC_per_genome.txt'),sep='\t')
    sns.histplot(PC_in_genomes,bins=100)
    plt.xlabel('Phage Genomes per Protein Cluster')
    plt.ylabel('Number of Genomes')
    plt.savefig(os.path.join(out_folder,f'{out_name}_PC_per_genome.png'),bbox_inches = 'tight')
    plt.close()
    
    return clust_results, anno_results, defense_results

def Synteny(in_folder,out_name,out_folder,phage_ordered,clust_results,start_protein,end_protein,
            total_phage,circular,clust_anno,defense_anno,draw_node_labels):
    print('Calculating synteny of phage genome hits')
    #Check if protein cluster annotations have been performed
    anno = clust_anno is not None
     
    #If not specified, set as reference the phage with the largest average cluster size with genome length within 1 standard deviation of above the mean
    if not phage_ordered:
        protein_clust_sizes = defaultdict(int)
        for index, row in clust_results.iterrows():
            protein_clust_sizes[row['rep_seq']] += 1
        mem2rep = {row['member']:row['rep_seq'] for index, row in clust_results.iterrows()}
        phage_genomes = glob.glob(os.path.join(in_folder,'*.faa'))

        #Get genome length and average cluster size for each phage genome
        phage_attr = []
        for genome in phage_genomes:
            protein_names = [record.id for record in SeqIO.parse(genome,'fasta')]
            clust_sizes = [protein_clust_sizes[mem2rep[prot]] for prot in protein_names]
            phage_attr.append((genome,np.sum(clust_sizes)/len(clust_sizes),len(protein_names)))
        mean_len, stdev_len = np.mean([x[2] for x in phage_attr]),np.std([x[2] for x in phage_attr])
        phage_attr = [x for x in phage_attr if mean_len <= x[2] <= mean_len + stdev_len] #subselect on phages with genome length within 1 stdev above the mean

        selected_phage = max(phage_attr,key=lambda x: x[1])[0]
        selected_phage_path = [path for path in phage_genomes if selected_phage in path][0]
        phage_ordered = [record.id for record in SeqIO.parse(selected_phage_path,'fasta')]


    #Assumes protein IDs from fasta/gbff file are in order on the phage.
    #Also assumes proteins in target viral genomes are in Prodigal annotation format
    max_index = len(phage_ordered)-1
    
    #If not specified set start and end proteins to first and last in list of IDs
    if not start_protein:
        start_protein = phage_ordered[0]
    if not end_protein:
        end_protein = phage_ordered[-1]
        
    #Check to make sure start and end proteins cover the whole prophage
    start_protein_pos = phage_ordered.index(start_protein)
    end_protein_pos = phage_ordered.index(end_protein)
    start_end_touch = abs(end_protein_pos - start_protein_pos) == 1
    start_end_cover = (start_protein_pos == 0 and end_protein_pos == max_index) or (start_protein_pos == max_index and end_protein_pos == 0)
    if start_end_touch == False and start_end_cover == False:
        raise Exception('Start and end proteins do not encompass the whole phage')

    #If the start and end proteins actually lie in the middle of the published order, rearrange so that they are on the edges
    if start_end_touch:
        first_in_order, last_in_order = min(start_protein_pos,end_protein_pos), max(start_protein_pos,end_protein_pos)
        phage_ordered = list(reversed(phage_ordered[:first_in_order+1])) + list(reversed(phage_ordered[last_in_order:]))
    
    #Put cluster annotation information in dictionary format
    anno_dict = {}
    if anno:
        for index, row in clust_anno.iterrows():
            if row['CARD_hit']:
                anno_dict[row['ID']] = 'AMR'
            if row['vfdb_hit']:
                anno_dict[row['ID']] = 'virulence'
            else:
                anno_dict[row['ID']] = row['category']
        #override annotations with defense annotation if present
        for index, row in defense_anno.iterrows():
            anno_dict[row['hit_id']] = 'defense'
    #Get protein cluster information into dictionary format and set reference sequences to representative sequences
    protein_clusts = defaultdict(set)
    for index, row in clust_results.iterrows():
        protein_clusts[row['rep_seq']].add(row['member'])
    for rep_seq, members in protein_clusts.copy().items():
        ref_proteins_in_clust = [ID for ID in phage_ordered if ID in members]
        if any(ref_proteins_in_clust):
            
            if anno:
                anno_category = anno_dict[rep_seq]
                del anno_dict[rep_seq]
                anno_dict[ref_proteins_in_clust[0]] = anno_category
                
            del protein_clusts[rep_seq]
            protein_clusts[ref_proteins_in_clust[0]] = members
            if len(ref_proteins_in_clust) > 1:
                print(f"proteins {ref_proteins_in_clust} cluster together, {ref_proteins_in_clust[0]} listed as representative")
                
    #Dictionary to get representative sequence from member proteins            
    mem2repseq = {}
    for rep_seq, member_set in protein_clusts.items():
        for member in member_set:
            mem2repseq[member] = rep_seq

    
    #Establish networkx graph with reference connections
    phage_graph = nx.Graph()
    for i in range(len(phage_ordered)-1):
        node1 = mem2repseq[phage_ordered[i]]
        node2 = mem2repseq[phage_ordered[i+1]]
        
        if (node1,node2) in phage_graph.edges(): #To account for reference proteins that cluster together
            phage_graph.edges[node1,node2]['weight'] += 1
        else:
            phage_graph.add_edge(node1,node2,weight=1,reference=True)
            
        phage_graph.nodes[node1]['reference'] = True
        phage_graph.nodes[node2]['reference'] = True
        
    if circular:
        node1 = mem2repseq[phage_ordered[0]]
        node2 = mem2repseq[phage_ordered[-1]]
        
        if (node1,node2) in phage_graph.edges():
            phage_graph.edges[node1,node2]['weight'] += 1
        else:
            phage_graph.add_edge(node1,node2,weight=1,reference=True)
            
        phage_graph.nodes[node1]['reference'] = True
        phage_graph.nodes[node2]['reference'] = True
    #Add non-reference nodes/connnections
    seen = set()
    ref_set = set(phage_ordered)
    for rep_protein, member_proteins in protein_clusts.items():
        
        filtered_members = member_proteins.difference(ref_set)
        for protein_id in filtered_members:
                        
            contig = ('_').join(protein_id.split('_')[:-1])
            prot_num = int(protein_id.split('_')[-1])

            up_ID = f"{contig}_{prot_num + 1}"
            if up_ID not in seen:
                up_rep_seq_hit = mem2repseq.get(up_ID,None)
                if up_rep_seq_hit:
                    if (rep_protein,up_rep_seq_hit) in phage_graph.edges():
                        phage_graph.edges[rep_protein,up_rep_seq_hit]['weight'] += 1
                    else:
                        phage_graph.add_edge(rep_protein,up_rep_seq_hit,weight=1,reference=False)

            down_ID = f"{contig}_{prot_num - 1}"
            if down_ID not in seen:
                down_rep_seq_hit = mem2repseq.get(down_ID,None)
                if down_rep_seq_hit:
                    if (rep_protein,down_rep_seq_hit) in phage_graph.edges():
                        phage_graph.edges[rep_protein,down_rep_seq_hit]['weight'] += 1
                    else:
                        phage_graph.add_edge(rep_protein,down_rep_seq_hit,weight=1,reference=False)
                    
            seen.add(protein_id)
    
    #Normalize edge weights for alpha by total number of phage hits
    edge_weights = nx.get_edge_attributes(phage_graph, 'weight')
    edge_alpha = {edge: min(weight / total_phage,1) for edge, weight in edge_weights.items()}  # Normalize weights to range [0, 1]
    
    #figure initalization
    plt.figure(figsize=(40,40))
    reference_pos = nx.circular_layout(phage_ordered,scale=1000)
    pos = nx.spring_layout(phage_graph,pos=reference_pos,k=5,fixed=phage_ordered,seed=7)
    
    #Scale node size by cluster size
    node_size_scale_factor = 2000
    node_size_dict = {ID: len(members) for ID, members in protein_clusts.items()} 
    max_clust_size = max(node_size_dict.values())
    node_size_dict = {ID:(size/max_clust_size)*node_size_scale_factor for ID,size in node_size_dict.items()}
    
    
    if anno:
        #Create a color map for the categories in anno_dict
        category_to_color = {'head and packaging':'#6db6ff',
                             'unknown function':'#db6d00',
                             'DNA, RNA and nucleotide metabolism':'#004949',
                             'lysis':'#920000',
                             'connector':'#b6dbff',
                             'moron, auxiliary metabolic gene and host takeover':'#009292',
                             'transcription regulation':'#b66dff',
                             'integration and excision':'#490092',
                             'tail':'#006ddb',
                             'other':'#924900',
                             'defense':'#ffff6d',
                             'virulence':'#ff6db6',
                             'AMR':'#ffb6db'}
        #Outline reference nodes with red edge
        ref_nodes = [n for n in phage_graph.nodes() if phage_graph.nodes[n].get('reference',False) == True]
        ref_node_colors = [category_to_color.get(anno_dict.get(n,None),'lime') for n in ref_nodes]
        ref_node_sizes = [node_size_dict[n] for n in ref_nodes]
        nx.draw_networkx_nodes(phage_graph, pos,nodelist=ref_nodes,node_size=ref_node_sizes,
                               node_color=ref_node_colors,edgecolors='red')
        #Outline non-reference nodes with black edge
        nonref_nodes = [n for n in phage_graph.nodes() if phage_graph.nodes[n].get('reference',False) == False]
        nonref_node_colors = [category_to_color.get(anno_dict.get(n,None),'lime') for n in nonref_nodes]
        nonref_node_sizes = [node_size_dict[n] for n in nonref_nodes]
        # nx.draw_networkx_nodes(phage_graph, pos,nodelist=nonref_nodes,node_size=nonref_node_sizes,
        #                        node_color=nonref_node_colors,edgecolors='black')
        nx.draw_networkx_nodes(phage_graph, pos,nodelist=nonref_nodes,node_size=nonref_node_sizes,
                               node_color=nonref_node_colors)
        
        #Create the legend for node coloring
        legend_elements = [Patch(facecolor=patch_color, edgecolor='black', label=category) 
                           for category,patch_color in category_to_color.items()]
        node_legend = plt.legend(handles=legend_elements, loc='upper right',bbox_to_anchor=(1,1), fontsize='xx-large', title='Pharokka Annotation')
        
        #Create legend for reference/non-reference edgecolors
        plt.gca().add_artist(node_legend)
        edge_legend = [Patch(facecolor='gray',edgecolor='red',label='Reference Protein Groups'),
                       Patch(facecolor='gray',edgecolor='black',label='Non-Reference Protein Groups')]
        plt.legend(handles=edge_legend,loc='upper right',bbox_to_anchor=(1,0.85),fontsize='xx-large')

        
    else:
        #Draw start node in green and end node in red
        nx.draw_networkx_nodes(phage_graph, pos, nodelist=[start_protein], node_color='green', node_size=node_size_dict[start_protein])
        nx.draw_networkx_nodes(phage_graph, pos, nodelist=[end_protein], node_color='red', node_size=node_size_dict[end_protein])
        
        #Get non-start and end nodes, split into reference and non-reference
        middle_nodes = [n for n in phage_graph.nodes() if n not in {start_protein, end_protein}]

        ref_nodes = [n for n in middle_nodes if middle_nodes[n].get('reference',False) == True]
        ref_node_sizes = [node_size_dict[n] for n in ref_nodes]
        nx.draw_networkx_nodes(phage_graph, pos,nodelist=ref_nodes,node_size=ref_node_sizes,edgecolors='red')
        
        nonref_nodes = [n for n in middle_nodes if middle_nodes[n].get('reference',False) == False]
        nonref_node_sizes = [node_size_dict[n] for n in nonref_nodes]
        nx.draw_networkx_nodes(phage_graph, pos,nodelist=nonref_nodes,node_size=nonref_node_sizes,edgecolors='black')

        edge_legend = [Patch(facecolor='gray',edgecolor='red',label='Reference Protein Groups'),
               Patch(facecolor='gray',edgecolor='black',label='Non-Reference Protein Groups')]
        plt.legend(handles=edge_legend,loc='upper right')
    
    #Draw node labels
    if draw_node_labels:
        nx.draw_networkx_labels(phage_graph, pos, labels={n: n for n in set([mem2repseq[ID] for ID in phage_ordered])})
    
    #edges
    nx.draw_networkx_edges(phage_graph,pos,edge_color='r',alpha=[edge_alpha[(u,v)] for u, v in phage_graph.edges()],width=5)

    plt.axis("off")
    plt.savefig(os.path.join(out_folder,f'{out_name}_synteny_graph.png'),bbox_inches = 'tight')
    nx.draw_networkx_labels(phage_graph, pos, labels={n: n for n in set([mem2repseq[ID] for ID in phage_ordered])})
    plt.savefig(os.path.join(out_folder,f'{out_name}_synteny_graph_labeled.png'),bbox_inches='tight')
    plt.close()

    #If present, add average log odds of defense to each node and plot
    pred_defense_file = os.path.join(DBDIR,'cd_for_pcd/phage_predictions.pq')
    if os.path.isfile(pred_defense_file):
        pred_defense = pd.read_parquet(pred_defense_file)
        prot2defense = dict(zip(pred_defense['center_id'],pred_defense['log_odds']))

        for node in phage_graph.nodes():
            mems = protein_clusts[node]
            node_pred_defense = np.mean([prot2defense[mem] for mem in mems])
            phage_graph.nodes[node]['defense'] = node_pred_defense
        #Initiate coolwarm cmap and set between -25 and 25
        defense_cmap = plt.cm.ScalarMappable(cmap='coolwarm')
        defense_cmap.set_clim(vmin=-25,vmax=25)
        defense_colors = [defense_cmap.to_rgba(phage_graph.nodes[node]['defense']) for node in phage_graph.nodes()]
        node_sizes = [node_size_dict[node] for node in phage_graph.nodes()]
        
        #figure re-initalization
        plt.figure(figsize=(40,40))
        reference_pos = nx.circular_layout(phage_ordered,scale=1000)
        pos = nx.spring_layout(phage_graph,pos=reference_pos,k=5,fixed=phage_ordered,seed=7)

        nx.draw_networkx_nodes(phage_graph, pos, node_color=defense_colors,node_size=node_sizes)
        nx.draw_networkx_edges(phage_graph,pos,edge_color='r',alpha=[edge_alpha[(u,v)] for u, v in phage_graph.edges()],width=5)

        plt.axis("off")
        plt.savefig(os.path.join(out_folder,f'{out_name}_synteny_graph_defense.png'),bbox_inches = 'tight')
        plt.close()
    
    
    nx.write_graphml(phage_graph,os.path.join(out_folder,f'{out_name}_graph.xml'))
    return phage_graph


def viral_group_analysis(out_name,in_folder,out_folder,ref_phage,calc_PCs,calc_synteny,
                         circular,start_protein,end_protein,annotate_clusters,draw_node_labels):
    
    #Make output folder if needed
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    #Histogram of genome sizes
    protein_files = glob.glob(os.path.join(in_folder,'*.faa'))
    if len(protein_files) == 0:
        raise Exception('No protein files found in input folder')
    genome_lengths = [len([record.id for record in SeqIO.parse(file,'fasta')]) for file in protein_files]
    sns.histplot(genome_lengths,binwidth=1)
    plt.title('Genome Lengths in # Proteins')
    plt.savefig(os.path.join(out_folder,f'{out_name}_genome_size.png'))
    plt.close()
    
    #Genomes analyzed written to file
    genome_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in protein_files]
    genome_names = pd.DataFrame(genome_names,columns=['genome_name'])
    genome_names.to_csv(os.path.join(out_folder,f'{out_name}_genomes_list.txt'),index=False,sep='\t')
                                
    if ref_phage:
        phage_ordered = [record.id for record in SeqIO.parse(ref_phage,'fasta')]
    else:
        phage_ordered = None

    if calc_PCs:
        clust_results, clust_anno, defense_anno = ProteinClusters(out_name,in_folder,out_folder,annotate_clusters)
        if calc_synteny:
            total_phage = len(glob.glob(os.path.join(in_folder,'*.faa')))
            phage_graph = Synteny(in_folder,out_name,out_folder,phage_ordered,clust_results,start_protein,end_protein,
                                  total_phage,circular,clust_anno,defense_anno,draw_node_labels)
    else:
        print('Not performing protein cluster or synteny analysis')
             
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synteny analysis of viral sequence group",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('out_name', type=str, help='Name under which to save results')
    parser.add_argument('in_folder', type=str, help='Folder where fasta files of phage genomes can be found')
    parser.add_argument('out_folder', type=str, help='Folder to output results')
    parser.add_argument('--ref_phage', type=str, help='Fasta file of phage genome to use as reference sequence. If not provided will pick as reference the genome with the largest average protein cluster size')

    add_group = parser.add_argument_group('Additional calculations','Calculate additional characteristics from phage search like synteny calculations')
    add_group.add_argument('--no_calc_PCs', action="store_true", help="Don't calculate protein clusters for all hits and produce a histogram of protein clusters per genome. Also stops synteny calculations.")
    add_group.add_argument('--no_calc_synteny', action="store_true", help="Don't calculate synteny of provided phage genome and produce graph of protein connections")
    add_group.add_argument('--no_annotate_clusters', action='store_true', help="Don't perform phrokka annotation of protein clusters.")
    add_group.add_argument('--draw_node_labels', action="store_true", help='Create additional synteny graph with labels drawn.')
    add_group.add_argument('--circular', action="store_true", help='Specify if provided genome is circular. Used for synteny calculations.')
    add_group.add_argument('--start_protein', type=str, help='First protein in phage genome if not first protein in fasta/gbff file. Used for synteny calculations.')
    add_group.add_argument('--end_protein', type=str, help='Last protein in phage genome if not last protein in fasta/gbff file. Used for synteny calculations.')

    path_group = parser.add_argument_group('Path variables','Variables for paths to tmp and database directories/files')
    path_group.add_argument('--TMPDIR', type=str, help='Path to temporary directory', default=os.path.expandvars('$TMPDIR'))
    path_group.add_argument('--DBDIR', type=str, help='Path to database directories and files', default=os.path.expandvars('$HOME/LaubLab_shared/'))

    args = parser.parse_args()

    global TMPDIR, DBDIR
    TMPDIR = args.TMPDIR
    DBDIR = args.DBDIR

    if args.no_calc_PCs == False:
        calc_PCs = True
    else:
        calc_PCs = False
    
    if args.no_calc_synteny == False:
        calc_synteny = True
    else:
        calc_synteny = False

    if args.no_annotate_clusters == False:
        annotate_clusters = True
    else:
        annotate_clusters = False

    viral_group_analysis(args.out_name,args.in_folder,args.out_folder,args.ref_phage,calc_PCs,calc_synteny,
                         args.circular,args.start_protein,args.end_protein,annotate_clusters,args.draw_node_labels)
