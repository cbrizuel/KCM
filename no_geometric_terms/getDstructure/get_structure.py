"get 3D structure"
import requests
import time
from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
import torch
import esm
from scipy.special import softmax
from ...config import esmfold_url
"""import torch
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
from scipy.special import softmax
import gc"""

"Funcion para determinaer el PDB haciendo una llamada al api"
def _request_esmfold_prediction(
                                sequence):
  # usamos el API de ESMFold para predecir la estructura de la secuencia
  response = requests.post(esmfold_url, 
                           data=sequence,verify=False)
  if response.status_code != 200:
    time.sleep(1.5)
    return _request_esmfold_prediction(sequence)
  #  time.sleep(random.uniform(0.1, 4.0))
  #  response = requests.post('https://api.esmatlas.com/foldSequence/v1/pdb/', 
  #                           data=sequence)
  #if response.status_code != 200:
  #  raise RuntimeError('a call to the ESMFold remote API failed.')
  return response.content.decode()






def esmfold_predict_structure(sequence: str, 
                              pdbFilename: str,
                              model_3d = None
                              ) -> any:
  if not model_3d: 
    model_3d = esm.pretrained.esmfold_v1()
    model_3d.eval().cuda().requires_grad_(False)
    model_3d.set_chunk_size(128)
  
  prediction_3d = model_3d.infer_pdb(sequence)
  torch.cuda.empty_cache()

  with open(pdbFilename, 'wt', encoding='utf-8') as pdb_file:
    pdb_file.write(prediction_3d)
  return model_3d,prediction_3d


def parse_output(output):
  pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
  plddt = output["plddt"][0,:,1]

  bins = np.append(0,np.linspace(2.3125,21.6875,63))
  sm_contacts = softmax(output["distogram_logits"],-1)[0]
  sm_contacts = sm_contacts[...,bins<8].sum(-1)
  xyz = output["positions"][-1,0,:,1]
  mask = output["atom37_atom_exists"][0,:,1] == 1
  o = {"pae":pae[mask,:][:,mask],
       "plddt":plddt[mask],
       "sm_contacts":sm_contacts[mask,:][:,mask],
       "xyz":xyz[mask]}
  return o

def get_hash(x): return hashlib.sha1(x.encode()).hexdigest()

def genera(sequence,model):
  alphabet_list = list(ascii_uppercase+ascii_lowercase)
  jobname = "test" #@param {type:"string"}
  jobname = re.sub(r'\W+', '', jobname)[:50]
  sequence = re.sub("[^A-Z:]", "", sequence.replace("/",":").upper())
  sequence = re.sub(":+",":",sequence)
  sequence = re.sub("^[:]+","",sequence)
  sequence = re.sub("[:]+$","",sequence)
  copies = 1 #@param {type:"integer"}
  if copies == "" or copies <= 0: copies = 1
  sequence = ":".join([sequence] * copies)
  num_recycles = 3 #@param ["0", "1", "2", "3", "6", "12", "24"] {type:"raw"}
  chain_linker = 25

  ID = jobname+"_"+get_hash(sequence)[:5]
  seqs = sequence.split(":")
  lengths = [len(s) for s in seqs]
  length = sum(lengths)
  print("length",length)

  u_seqs = list(set(seqs))
  if len(seqs) == 1: mode = "mono"
  elif len(u_seqs) == 1: mode = "homo"
  else: mode = "hetero"
  torch.cuda.empty_cache()
  output = model.infer(sequence,
                      num_recycles=num_recycles,
                      chain_linker="X"*chain_linker,
                      residue_index_offset=512)

  pdb_str = model.output_to_pdb(output)[0]
  output = tree_map(lambda x: x.cpu().numpy(), output)
  ptm = output["ptm"][0]
  plddt = output["plddt"][0,...,1].mean()
  return pdb_str, output,ptm,plddt

