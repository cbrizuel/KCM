"Tools from pdb and sequences"
from Bio.PDB import PDBParser, PPBuilder
import io

"Extrae la secuencia de aminoacidos a partir del pdb almacenado en el directorio de trabajo"
def extract_amino_acid_sequence_pdb(input_text):
    parser = PDBParser()
    structure = parser.get_structure('protein', input_text)
    ppb=PPBuilder()
    for pp in ppb.build_peptides(structure):
        pp.get_sequence()
    seq = pp.get_sequence().__str__()
    return seq

def extract_amino_acid_sequence_pdb(input_text):
    parser = PDBParser()
    structure = parser.get_structure('protein', input_text)
    ppb = PPBuilder()
    sequences = []
    for pp in ppb.build_peptides(structure):
        sequence = pp.get_sequence().__str__()
        sequences.append(sequence)
    sequences = ''.join(sequences)
    return sequences

"Funcion para extraer el Backbone de los atomos cuando se tiene el PDB almacenado en el directorio de trabajo"
def extract_backbone_atoms(pdb_file):
    backbone_atoms = []
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)

    # Obtener el primer modelo (normalmente es el índice 0)
    model = structure[0]

    # Obtener la cadena principal (normalmente es la cadena 'A')
    chain_id_principal = 'A'
    chain_principal = model[chain_id_principal]
   

    for residue in chain_principal:
        if residue.get_id()[0] == ' ' and residue.get_resname() in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']:
            for atom in residue:
                atom_name = atom.get_name()
                if atom_name in ['N', 'CA', 'C', 'O']:
                    backbone_atoms.append(atom.get_coord())                  
    return backbone_atoms 
    

"Funcion para construir la secuencia de aminoacidos y para llama al ESMFold"
def det_sec(posiciones,amino):
    letras_seleccionadas = [amino[posicion] for posicion in posiciones]
    return ''.join(letras_seleccionadas)


"Funcion para extraer el Backbone de los atomos cuando se tiene el PDB almacenado en variable"
def extract_backbone_atoms_str(pdb_str):
    backbone_atoms = []
    pdb_io = io.StringIO(pdb_str)

    # Crear el parser y obtener la estructura
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_io)

    # Obtener el primer modelo (normalmente es el índice 0)
    model = structure[0]

    # Obtener la cadena principal (normalmente es la cadena 'A')
    chain_id_principal = 'A'
    chain_principal = model[chain_id_principal]
   

    for residue in chain_principal:
        if residue.get_id()[0] == ' ' and residue.get_resname() in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']:
            for atom in residue:
                atom_name = atom.get_name()
                if atom_name in ['N', 'CA', 'C', 'O']:
                    backbone_atoms.append(atom.get_coord())                  
    return backbone_atoms 

def extract_amino_acid_sequence(input_text):
    pdb_io = io.StringIO(input_text)
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_io)
    ppb=PPBuilder()
    for pp in ppb.build_peptides(structure):
        pp.get_sequence()
    seq = pp.get_sequence().__str__()
    return seq