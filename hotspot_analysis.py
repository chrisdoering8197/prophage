import glob
import os
import sys
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages


HOME = os.path.expandvars('$HOME')
sys.path.append(os.path.join(HOME,'hotspots/'))
from HelperFunctions import *

def hhpred(file,DBs,hhsuite_dbs,transfer_dbs=True):
    basename, ext = os.path.splitext(file)
    a3m = basename+'.a3m'
    if ext != '.a2m' and ext != '.faa':
        raise ValueError('Input file must be in either a2m or fasta format')
    if transfer_dbs:
        if ext == '.a2m':
            hhblits_cmd = ['conda run -n HHsuiteEnv','hhblits','-o','/dev/null',
                    '-i',file,'-M a2m','-d',os.path.join(TMPDIR,'UniRef30_2019_11'),
                    '-oa3m',a3m,'-cpu','2','-n','2','-v','1']
            os.system(' '.join(hhblits_cmd))
        if ext == '.faa':
            hhblits_cmd = ['conda run -n HHsuiteEnv','hhblits','-o','/dev/null',
            '-i',file,'-d',os.path.join(TMPDIR,'UniRef30_2019_11'),
            '-oa3m',a3m,'-cpu','2','-n','2','-v','1']
            os.system(' '.join(hhblits_cmd))
    else:
        if ext == '.a2m':
            hhblits_cmd = ['conda run -n HHsuiteEnv','hhblits','-o','/dev/null',
                    '-i',file,'-M a2m','-d',os.path.join(hhsuite_dbs,'UniRef30_2019_11'),
                    '-oa3m',a3m,'-cpu','2','-n','2','-v','1']
            os.system(' '.join(hhblits_cmd))
        if ext == '.faa':
            hhblits_cmd = ['conda run -n HHsuiteEnv','hhblits','-o','/dev/null',
            '-i',file,'-d',os.path.join(hhsuite_dbs,'UniRef30_2019_11'),
            '-oa3m',a3m,'-cpu','2','-n','2','-v','1']
            os.system(' '.join(hhblits_cmd))
    hhr_out = basename+'.hhr'
    hhsearch_cmd = ['conda run -n HHsuiteEnv','hhsearch','-i',a3m,'-o',hhr_out,'-v','1']
    dbcommand = ' '.join([f'-d {db}' for db in DBs])
    hhsearch_cmd = ' '.join(hhsearch_cmd) + ' ' + dbcommand
    os.system(hhsearch_cmd)
    

def hotspot_proteins(out_folder,protein_file,defenseDF,anno_dict,clust_dict,mem2rep,
                     protein1,protein2,max_distance,hhpred_prob_cutoff,show_clusters,dbs2search,transfer_dbs,hhsuite_dbs):
    #Get all homologs of each protein and check for overlap in genomes using Prodigal naming convention
    prot1_homologs = clust_dict[mem2rep[protein1]]
    prot1_genomes = {'_'.join(prot.split('_')[:-1]) for prot in prot1_homologs}
    prot2_homologs = clust_dict[mem2rep[protein2]]
    prot2_genomes = {'_'.join(prot.split('_')[:-1]) for prot in prot2_homologs}
    covered_genomes = prot1_genomes.intersection(prot2_genomes)
    
    #Check genomes for hotspots at least max_distance apart
    hotspots = []
    for phage_genome in covered_genomes:
        prot1_hits = {prot for prot in prot1_homologs if phage_genome == '_'.join(prot.split('_')[:-1])}
        prot2_hits = {prot for prot in prot2_homologs if phage_genome == '_'.join(prot.split('_')[:-1])}
        potential_hotspots = []
        for prot1 in prot1_hits:
            for prot2 in prot2_hits:
                low_num, high_num = sorted([int(prot1.split('_')[-1]),int(prot2.split('_')[-1])])
                prots_between = [f'{phage_genome}_{num}' for num in range(low_num+1,high_num)]
                if len(prots_between) <= max_distance:
                    potential_hotspots.append([f'{phage_genome}_{low_num}']+prots_between+[f'{phage_genome}_{high_num}'])
        if potential_hotspots:
            smallest_hotspot = min(potential_hotspots,key=len)
            hotspots.append((phage_genome,smallest_hotspot))
            
    #Save DefenseFinder and Pharokka annotations in dataframe for hotspots
    hotspots_with_reps = [(phage_genome,hotspot,[mem2rep[prot] for prot in hotspot]) for phage_genome,hotspot in hotspots]
    hotspotDF = pd.DataFrame(hotspots_with_reps,columns=['Genome','Hotspot Proteins','RepSeq Hotspot Proteins'])

    defense_dict = dict(defenseDF[['hit_id', 'gene_name']].values)
    hotspotDF['DefenseFinder Proteins'] = hotspotDF['RepSeq Hotspot Proteins'].apply(lambda prot_list: [defense_dict.get(prot,'') for prot in prot_list])
    hotspotDF['Protein Class'] = hotspotDF['RepSeq Hotspot Proteins'].apply(lambda prot_list: [anno_dict[prot] for prot in prot_list])
    
    #HHpred annotation of protein clusters in hotspot
    clusters2annotate = set([item for sublist in hotspotDF['RepSeq Hotspot Proteins'] for item in sublist])
    hhpred_folder = os.path.join(out_folder,f'protNums_{protein1.split("_")[-1]}_{protein2.split("_")[-1]}_hhpred_files/')
    if not os.path.isdir(hhpred_folder):
        os.mkdir(hhpred_folder)
    protein_seqs = SeqIO.to_dict(SeqIO.parse(protein_file,'fasta'))
    files4hhpred = []
    cluster_file_name_mapping = {}
    #Write clusters to individual fasta files and build MSAs
    for i, repseq in enumerate(clusters2annotate):
        cluster_file_name_mapping[i] = repseq
        
        clust_file_name = os.path.join(hhpred_folder,f'cluster_{i}.faa')
        clust_seqs = [protein_seqs[protID] for protID in clust_dict[repseq]]
        SeqIO.write(clust_seqs,clust_file_name,'fasta')
            
        if len(clust_seqs) > 1:
            a2m_file_name = os.path.join(hhpred_folder,f'cluster_{i}.a2m')
            if not os.path.isfile(a2m_file_name):
                clustalo = ['conda run -n MyEnv','clustalo','-i',clust_file_name,'-o',a2m_file_name,'--outfmt','a2m','--threads',str(CPU_AVAIL)]
                os.system(' '.join(clustalo))
            files4hhpred.append(a2m_file_name)
        else:
            files4hhpred.append(clust_file_name)
            
    pd.DataFrame.from_dict(cluster_file_name_mapping,orient='index').to_csv(os.path.join(hhpred_folder, 'ClusterNumberMapping.txt'), sep='\t')

    files4hhpred = [file for file in files4hhpred if not os.path.isfile(os.path.splitext(file)[0]+'.hhr')]
    if files4hhpred:

        #Transfer UniRef database to temporary directory
        if 'UniRef30_2019_11_a3m.ffdata' not in os.listdir(TMPDIR) and transfer_dbs:
            print('Transferring the Uniclust database to temporary filesystem')
            os.system(f'cp -r {os.path.join(hhsuite_dbs,'UniRef30_2019*')} {TMPDIR}')
            os.system(f'chmod a+r {os.path.join(TMPDIR,'UniRef30_2019*')}')


        #Search against databases with hhpred
        db_locs = {'pfam':os.path.join(hhsuite_dbs,'pfam'),
           'innate':os.path.join(hhsuite_dbs,'innate'),
           'defense':os.path.join(hhsuite_dbs,'df'),
           'CD':os.path.join(hhsuite_dbs,'NCBI_CD'),
           'pdb':os.path.join(hhsuite_dbs,'pdb70'),
           'uniref': os.path.join(hhsuite_dbs,'UniRef30_2019_11')}
        DBs = [db_locs[db] for db in dbs2search]

        #Run hhsearch on fasta alignment files
        print('Running HHpred annotations')
        n_jobs = int(CPU_AVAIL/2)
        _ = Parallel(n_jobs=n_jobs)(delayed(hhpred) (afa_file, DBs,hhsuite_dbs,transfer_dbs) for afa_file in tqdm(files4hhpred))
    
    #Compile hhearch results into single dataframe
    hhr_files = glob.glob(os.path.join(hhpred_folder,'*.hhr'))
    hhpred_results = []
    for hhr_file in hhr_files:
        #Parse hhr file into pandas dataframe
        repseq = cluster_file_name_mapping[int(os.path.splitext(hhr_file)[0].split('_')[-1])]
        single_hhpred = parse_hhblits(hhr_file,repseq)
        single_hhpred['RepSeq'] = repseq
        single_hhpred = single_hhpred[single_hhpred['Prob'] >= hhpred_prob_cutoff]
        hhpred_results.append(single_hhpred)
    hhpred_results = pd.concat(hhpred_results,ignore_index=True)
    hhpred_results.to_csv(os.path.join(hhpred_folder,f'{protein1}_{protein2}_hhpred_combined.txt'),sep='\t')
    
    top_hotspot_DF = []
    with PdfPages(os.path.join(hhpred_folder,f'{protein1}_{protein2}_top_hotspots.pdf')) as pdf:
        for hotspot_proteins,count in hotspotDF['RepSeq Hotspot Proteins'].value_counts()[:show_clusters].items():

            setDF = []
            prot_num = 1
            for i, prot in enumerate(hotspot_proteins):
                protein_record = protein_seqs[prot]
                protDF = hhpred_results[hhpred_results['query'].str.contains(prot)].reset_index()
                if not protDF.empty:
                    protDF.rename(columns={'query':'System Name','Prob':'Domain Prob','qstart':'Domain Start','qend':'Domain End','hit_name':'Domain Name'},inplace=True)
                    protDF['Protein #'] = prot_num
                    protDF['Domain Prob'] = protDF['Domain Prob'] / 100
                    protDF['Protein Length'] = len(protein_record.seq)*3
                    #List median strand for protein of those involved in this hotspot cluster
                    protein_group = [prot_list[i] for prot_list in 
                                     hotspotDF[hotspotDF['RepSeq Hotspot Proteins'].apply(lambda x: x == hotspot_proteins)]['Hotspot Proteins']]
                    protein_strands = [int(protein_seqs[ID].description.split('#')[3]) for ID in protein_group]
                    protDF['Strand'] = np.median(protein_strands)
                    for index, row in protDF.copy().iterrows():
                        domain_parts = row['Domain Name'].split(';')
                        if domain_parts[0].startswith('PF'):
                            protDF.loc[index,'Domain Name'] = domain_parts[1]
                        elif domain_parts[0].startswith('cd'):
                            protDF.loc[index,'Domain Name'] = domain_parts[0].split(' ')[1]
                        elif domain_parts[0].startswith('UniRef50'):
                            protDF.loc[index,'Domain Name'] = domain_parts[0].split('~')[0].split(' ')[1]
                        elif 'Cas' in domain_parts[0]:
                            protDF.loc[index,'Domain Name'] = domain_parts[0].split(' ')[1]
                        #If name starts with the pattern of XXXX_X where those are a mix of letters and numbers remove that section and take everything after until a comma
                        
                        else:
                            protDF.loc[index,'Domain Name'] = domain_parts[0]
                    setDF.append(protDF)
                    top_hotspot_DF.append(protDF)
                else:
                    prot_data = {'Hit':' ','Domain Prob':0,'E-value':None,'P-value':None,'Score':None,'SS':None,
                                 'Cols':None,'Query HMM':None,'Template HMM':None,'query':prot,'Strand':1,
                                 'Domain Start':len(protein_seqs[prot].seq)/2,'Domain End':len(protein_seqs[prot].seq)/2,
                                 'Domain Name':' ','System Name':prot,'Protein #':prot_num,'Protein Length':len(protein_seqs[prot].seq)*3}
                    setDF.append(pd.DataFrame(prot_data,index=[0]))
                prot_num += 1
                
                #Generate intergenic ghost proteins to properly space proteins when plotting. Based on median intergenic distance for proteins this cluster.
                if i < len(hotspot_proteins) - 1:
                    prot_groups = [(prot_list[i],prot_list[i+1]) for prot_list in
                                   hotspotDF[hotspotDF['RepSeq Hotspot Proteins'].apply(lambda x: x == hotspot_proteins)]['Hotspot Proteins']]
                    inter_dists = [int(protein_seqs[prot2].description.split('#')[1]) - int(protein_seqs[prot1].description.split('#')[2])
                                   for prot1, prot2 in prot_groups]
                    median_dist = np.median([lengths for lengths in inter_dists])
 
                    inter_data = {'Hit':' ','Domain Prob':0,'E-value':None,'P-value':None,'Score':None,'SS':None,
                         'Cols':None,'Query HMM':None,'Template HMM':None,'query':'Intergenic','Strand':1,
                         'Domain Start':median_dist/2,'Domain End':median_dist/2,
                         'Domain Name':' ','System Name':'Intergenic','Protein #':prot_num,'Protein Length':median_dist}
                    setDF.append(pd.DataFrame(inter_data,index=[0]))
                    prot_num += 1

            setDF = pd.concat(setDF,ignore_index=True)
            figs = []    
            if not setDF.empty:
                fig = plot_system(setDF,display_name=f'Cluster size {count}',
                                  save_name=os.path.join(hhpred_folder,f'cluster_size_{count}.svg'))
                pdf.savefig(fig,bbox_inches='tight')
                plt.close(fig)
        top_hotspot_DF = pd.concat(top_hotspot_DF,ignore_index=True)
        top_hotspot_DF.to_csv(os.path.join(hhpred_folder,f'{protein1}_{protein2}_top_hotspots.txt'),sep='\t')
        
    hotspotDF.to_csv(os.path.join(out_folder,f'{protein1}_{protein2}_hotspot_summary.txt'),sep='\t')

        

def hotspot_analysis(protein1,protein2,in_clusters,in_proteins,out_folder,anno_file,defense_file,
                 max_distance,hhpred_prob_cutoff,show_clusters,dbs2search,transfer_dbs,hhsuite_dbs):
    #Create out_folder if needed
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
        
    #Read in pharokka and DefenseFinder annotations
    if anno_file:
        print('Reading in annotations...')
        annotations = pd.read_csv(anno_file,sep='\t',usecols=['ID','phrog','annot','category','vfdb_hit','CARD_hit'])
        annotations['vfdb_hit'] = annotations['vfdb_hit'].replace({np.nan:None,'None':None})
        annotations['CARD_hit'] = annotations['CARD_hit'].replace({np.nan:None,'None':None})
        anno_dict = {}
        for index, row in annotations.iterrows():
            if row['CARD_hit']:
                anno_dict[row['ID']] = 'AMR'
            if row['vfdb_hit']:
                anno_dict[row['ID']] = 'virulence'
            else:
                anno_dict[row['ID']] = row['category']
        if defense_file:
            defense = pd.read_csv(defense_file,sep='\t',usecols=['hit_id','gene_name'])
            #override annotations with defense annotation if present
            for index, row in defense.iterrows():
                anno_dict[row['hit_id']] = 'defense'
    
    #Get protein cluster information into dictionary format
    print('Reading in cluster information...')
    protein_df = pd.read_csv(in_clusters,sep='\t',names=['rep_seq','member'])
    protein_clusts = protein_df.groupby('rep_seq')['member'].apply(set).to_dict()
        
    #Dictionary to get representative sequence from member proteins
    mem2rep = dict(protein_df[['member','rep_seq']].values)        

    hotspot_proteins(out_folder,in_proteins,defense,anno_dict,protein_clusts,mem2rep,
                        protein1,protein2,max_distance,hhpred_prob_cutoff,show_clusters,dbs2search,transfer_dbs,hhsuite_dbs)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze hotspot identified in phage synteny plot",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('protein1', type=str, help='One half of boundary proteins for hotspot to analyze')
    parser.add_argument('protein2',type=str, help='Other half of boundary proteins for hotspot to analyze')
    parser.add_argument('in_clusters', type=str, help='tab separated file cluster file from mmseqs where one column shows the representative sequences and the other the members of each cluster')
    parser.add_argument('in_proteins', type=str, help='Fasta file containing protein sequences for all involved proteins')
    parser.add_argument('out_folder', type=str, help='Folder to output results')
    parser.add_argument('--anno_file',type=str,help='File containing pharokka annotations of protein clusters.')
    parser.add_argument('--defense_file',type=str,help='File containing DefenseFinder annotation of protein clusters.')
    parser.add_argument('--max_distance', type=int, default=15, help='Maximum hotspot size to consider.')
    parser.add_argument('--hhpred_prob_cutoff', type=float, default=0.5, help='Probability cutoff for display of hhpred domain predictions.')
    parser.add_argument('--show_clusters', type=int, default=20, help='Number of top hhpred cluster results to show.')
    parser.add_argument('-d','--databases',nargs='*',
                    default=['pfam','innate','defense','CD','pdb'],
                    choices=['pfam','innate','defense','CD','pdb','uniref'],
                    help='Databases with which to perform hhsearch.')
    
    demo_vars = parser.add_argument_group('Demo Variables',description='Variables used when running demonstration or not on SuperCloud system')
    demo_vars.add_argument('--TMPDIR',type=str,default=os.path.expandvars('$TMPDIR'),help='Temporary directory for running hhsearch')
    demo_vars.add_argument('--CPU_AVAIL',type=int,default=multiprocessing.cpu_count(),help='Number of CPUs available for running hhsearch')
    demo_vars.add_argument('--hhsuite_dbs',type=str,default=os.path.expandvars('$HOME/LaubLab_shared/hhsuite_dbs/'),help='Directory containing hhsuite databases')
    demo_vars.add_argument('--transfer_dbs',action='store_true',help='Transfer databases to local filesystem for faster searching. Set to false if running locally.')

    args = parser.parse_args()

    global TMPDIR, CPU_AVAIL
    TMPDIR = args.TMPDIR
    CPU_AVAIL = args.CPU_AVAIL

    hotspot_analysis(args.protein1,args.protein2,args.in_clusters,args.in_proteins,args.out_folder,args.anno_file,
                     args.defense_file,args.max_distance,args.hhpred_prob_cutoff,args.show_clusters,args.databases,args.transfer_dbs,args.hhsuite_dbs)