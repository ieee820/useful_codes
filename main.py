from Bio.Seq import Seq
from Bio import SeqIO
from Bio.Align.Applications import MafftCommandline
from glob import glob
from Bio.Seq import Seq, reverse_complement
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO.FastaIO import FastaTwoLineWriter
import requests
import os
# import pickle
from bs4 import BeautifulSoup
import pandas as pd
import math
import shutil
# my_seq = Seq("AGTACACTGGT")
# print(my_seq)

error_ab1_list = []


def igblast_post(nn_fasta_filepath):
    url = 'https://www.ncbi.nlm.nih.gov/igblast/igblast.cgi'
    playload = {
        "organism": "human",
        "germline_db_V": "IG_DB/imgt.Homo_sapiens.V.f.orf.p",
        "germline_db_D": "IG_DB/imgt.Homo_sapiens.D.f.orf",
        "germline_db_J": "IG_DB/imgt.Homo_sapiens.J.f.orf",
        "program": "blastn",
        "mismatch_penalty": -1,
        "min_D_match": 5,
        "D_penalty": -2,
        "min_V_length": 9,
        "min_J_length": 0,
        "J_penalty": -2,
        "num_alignments_V": 3,
        "num_alignments_D": 3,
        "num_alignments_J": 3,
        "translation": "true",
        "domain": "imgt",
        "num_clonotype": 100,
        "outfmt": 3,
        "v_focus": "true",
        "num_alignments_additional": 10,
        "evalue": 1,
        "SEARCH_TYPE": "IG",
        "igsource": "new",
        "analyze": "on",
        "CMD": "request"
    }

    files = {
        "queryfile": open(nn_fasta_filepath, "rb")
    }
    response = requests.post(url, data=playload, files=files)

    return response
    # dump response to pkl
    # file_r = open('response_90nn.pkl', 'wb')
    # pickle.dump(response, file_r, pickle.HIGHEST_PROTOCOL)



def table_to_pd(table):
    rows = []
    for child in table.children:
        row = []
        for td in child:
            try:
                row.append(td.text.replace('\n', ''))
            except:
                continue
        if len(row) > 0:
            rows.append(row)
    df = pd.DataFrame(rows[1:], columns=rows[0])
    # print(df_sum)
    return df



def df_unfold(df_detail):
    res = pd.DataFrame(columns=['ID', 'Count', 'Clone_name'])
    for index, row in df_detail.iterrows():
        # print(index, row['ID'], row['Count'], row['All query name(s)'])
        clone_names = row['All query name(s)']
        # print(clone_names)
        clone_rep_list = clone_names.split(',')
        if len(clone_rep_list) > 1:
            for clone_name in clone_names.split(','):
                # print(clone_name)
                res = res.append({'ID': row['ID'], 'Count': row['Count'], 'Clone_name': clone_name}, ignore_index=True)
        else:
            res = res.append({'ID': row['ID'], 'Count': row['Count'], 'Clone_name': clone_names}, ignore_index=True)

    return res



def parse_igblast_result(response):
    # print(response.text)
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table', attrs={'border': 1})
    # print(table)
    for table in tables:
        # print(table.previous.string)
        previous_element = table.previous.string
        if 'Clonotype summary' in previous_element:
            # table to pd
            df_sum = table_to_pd(table)

        elif 'grouped by clonotypes' in previous_element:
            # print(table)
            df_detail = table_to_pd(table)

    # print(df_sum, df_detail)
    res_rep = df_unfold(df_detail)
    res_merge = res_rep.merge(df_sum, on='ID', how='left')
    res_merge['Clone_name_abv'] = res_merge.apply(lambda row: row['Clone_name'].split('_')[0], axis=1)
    Clone_name_abv = res_merge['Clone_name_abv']
    res_merge = res_merge.drop(columns=['Clone_name_abv'])
    res_merge.insert(loc=1, column='Clone_name_abv', value=Clone_name_abv)
    # print(res_merge)
    # res_merge.to_excel('output.xlsx')
    return res_merge



def step2(nn_fasta_filepath, output_xls):
    response = igblast_post(nn_fasta_filepath)
    res_df = parse_igblast_result(response)
    res_df.to_excel(output_xls)




def seq_test():
    for seq_record in SeqIO.parse("F-99-NN.fas", "fasta"):
        print(seq_record.id)
        # print(seq_record.seq)
        print(len(seq_record))



def read_abi():
    handle = open("F/SHS001-570_Seq-F_TSS20210711-021-03705_C01.ab1", "rb")
    for record in SeqIO.parse(handle, "abi"):
        print(record.seq)



def do_align():
    mafft_cline = MafftCommandline(input="3_seq.fasta")
    print(mafft_cline)
    stdout, stderr = mafft_cline()
    # output.write(stdout)



def dna_enzyme_cutting(seq, cutting_site_left, cutting_site_right, ab1_filename):
    seq_str =str(seq)
    # print(seq_str)
    if (cutting_site_left in seq_str) and (cutting_site_right in seq_str):
        #cut left seq
        pos_left = seq_str.find(cutting_site_left)
        step_left = len(cutting_site_left)
        cut_range_left = pos_left + step_left
        left_seq_str = seq_str[cut_range_left:]
        # print(new_seq_str)
        #cut right seq
        pos_right = left_seq_str.find(cutting_site_right)
        # step_right = len(cutting_site_right)
        # cut_range_right = pos_right + step_right
        right_seq_str = left_seq_str[0:pos_right]
    else:
        error_ab1_list.append(ab1_filename)
        right_seq_str = ''

    return right_seq_str




def abi_to_fasta(ab1_floder_path, cutting_site_left, cutting_site_right, output_nn_path):
    all_sequences = []
    for filename in glob(ab1_floder_path):
        handle = open(filename, "rb")
        for record in SeqIO.parse(handle, "abi"):
            seq_cut = dna_enzyme_cutting(record.seq, cutting_site_left, cutting_site_right, record.name)
            if seq_cut != '':
                simple_seq = Seq(seq_cut)
                simple_seq_r = SeqRecord(simple_seq)
                simple_seq_r.id = record.name
                simple_seq_r.description = ''
                all_sequences.append(simple_seq_r)

    # SeqIO.write(all_sequences, "all_sequences.fasta", "fasta")
    handle = open(output_nn_path, 'w')
    writer = FastaTwoLineWriter(handle)
    writer.write_file(all_sequences)
    handle.close()



def dna_to_protein(nn_fasta_file, output_aa_path, if_reverse_complement):
    all_aa_seqs = []
    for seq_record in SeqIO.parse(nn_fasta_file, 'fasta'):
        # print(seq_record.id)
        # print(repr(seq_record.seq))
        seq = seq_record.seq
        if if_reverse_complement:
            # seq = seq.complement()
            seq = reverse_complement(seq)
        rna = seq.transcribe()
        aa_seq = rna.translate()
        # print(protein)
        simple_seq = Seq(aa_seq)
        simple_seq_r = SeqRecord(simple_seq)
        simple_seq_r.id = seq_record.name
        simple_seq_r.description = ''
        all_aa_seqs.append(simple_seq_r)

    handle = open(output_aa_path, 'w')
    writer = FastaTwoLineWriter(handle)
    writer.write_file(all_aa_seqs)
    handle.close()



#F_cuat = ['CCATGGCC' , 'TAATAA']
def step1(F_ab1_folder, R_ab1_folder, F_cut, R_cut, work_dir):
    F_NN_path = os.path.join(work_dir, 'F_NN.fasta')
    R_NN_path = os.path.join(work_dir, 'R_NN.fasta')
    F_AA_path = os.path.join(work_dir, 'F_AA.fasta')
    R_AA_path = os.path.join(work_dir, 'R_AA.fasta')
    abi_to_fasta(F_ab1_folder, F_cut[0], F_cut[1], F_NN_path)
    abi_to_fasta(R_ab1_folder, R_cut[0], R_cut[1], R_NN_path)
    dna_to_protein(F_NN_path, F_AA_path, False)
    dna_to_protein(R_NN_path, R_AA_path, True)



def merge_F_R(F_igblast_path, R_igblast_path, merge_igblast_path):
    F_df = pd.read_excel(F_igblast_path,'Sheet1')
    R_df = pd.read_excel(R_igblast_path,'Sheet1')
    F_mini_df = F_df[["ID", "Clone_name_abv", "V gene"]]
    F_mini_df.columns = ["ID-F", "Clone_name_abv", "F-V gene"]

    R_mini_df = R_df[["ID", "Clone_name_abv", "V gene"]]
    R_mini_df.columns = ["ID-R", "Clone_name_abv", "R-V gene"]
    # R_mini_df["ID-R"] = pd.astype(R_mini_df["ID-R"])
    # R_mini_df.astype('object').dtypes

    merge_df = F_mini_df.merge(R_mini_df, on='Clone_name_abv', how='inner')
    # merge_df = merge_df.astype(object)
    merge_df['ID-type'] = merge_df.apply(lambda row: str(row['ID-F']) + '-' + str(row['ID-R']), axis=1)
    # print(merge_df.dtypes)
    # merge_df.astype('object').dtypes
    # merge_df = merge_df.astype(object)
    merge_df = merge_df[["Clone_name_abv", "ID-F","ID-R", 'ID-type', "F-V gene", "R-V gene"]]
    merge_df.to_excel(merge_igblast_path)




if __name__ == "__main__":
    # print(error_ab1_list)

    pwd = os.getcwd()
    work_dir = os.path.join(pwd, 'working')
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    # else:
    #     shutil.rmtree(work_dir)
    #     os.mkdir(work_dir)

    F_igblast_path = os.path.join(work_dir, 'F_igblast.xlsx')
    R_igblast_path = os.path.join(work_dir, 'R_igblast.xlsx')
    merge_igblast_path = os.path.join(work_dir, 'merge_igblast.xlsx')
    F_NN_path = os.path.join(work_dir, 'F_NN.fasta')
    R_NN_path = os.path.join(work_dir, 'R_NN.fasta')
    # step1('./F/*.ab1', './R/*.ab1', ['CCATGGCC' , 'TAATAA'], ['GCGGCCGC', 'AGCAGA'], work_dir)
    # step2(F_NN_path, F_igblast_path)
    # step2(R_NN_path, R_igblast_path)
    merge_F_R(F_igblast_path, R_igblast_path, merge_igblast_path)
