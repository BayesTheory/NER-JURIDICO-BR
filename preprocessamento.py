import os
import glob
import json
import pickle

def montar_caminhos(base_path_lener):
    paths = {}
    paths['dev'] = os.path.join(base_path_lener, 'dev')
    paths['metadata'] = os.path.join(base_path_lener, 'metadata')
    paths['raw_text'] = os.path.join(base_path_lener, 'raw_text')
    paths['scripts'] = os.path.join(base_path_lener, 'scripts')
    paths['test'] = os.path.join(base_path_lener, 'test')
    paths['train'] = os.path.join(base_path_lener, 'train')

    # Arquivos principais
    paths['file_train_conll'] = os.path.join(paths['train'], 'train.conll')
    paths['file_dev_conll'] = os.path.join(paths['dev'], 'dev.conll')
    paths['file_test_conll'] = os.path.join(paths['test'], 'test.conll')

    # Metadata JSON
    metadata_json_files = glob.glob(os.path.join(paths['metadata'], '*.json'))
    paths['file_metadata_json'] = metadata_json_files[0] if metadata_json_files else None

    # Scripts
    paths['file_script_text_to_conll'] = os.path.join(paths['scripts'], 'textToConll.py')
    paths['file_script_abbrev_pkl'] = os.path.join(paths['scripts'], 'abbrev_list.pkl')

    # Raw text
    raw_text_txt_files = glob.glob(os.path.join(paths['raw_text'], '*.txt'))
    paths['file_raw_text_txt'] = raw_text_txt_files[0] if raw_text_txt_files else None
    raw_text_json_files = glob.glob(os.path.join(paths['raw_text'], '*.json'))
    paths['file_raw_text_json'] = raw_text_json_files[0] if raw_text_json_files else None

    return paths

def checar_arquivos(paths):
    files_to_check = {
        "Treino (.conll)": paths['file_train_conll'],
        "Dev (.conll)": paths['file_dev_conll'],
        "Teste (.conll)": paths['file_test_conll'],
        "Metadata (.json)": paths['file_metadata_json'],
        "Script (textToConll.py)": paths['file_script_text_to_conll'],
        "Script (abbrev_list.pkl)": paths['file_script_abbrev_pkl'],
        "Raw Text (.txt)": paths['file_raw_text_txt'],
        "Raw Text (.json)": paths['file_raw_text_json'],
    }
    print("\n--- Verificando Arquivos ---")
    all_found = True
    for name, path in files_to_check.items():
        if path and os.path.exists(path):
            print(f"[OK] {name}: {path}")
        elif path:
            print(f"[ERRO] {name}: Arquivo NÃO encontrado em {path}")
            all_found = False
        else:
            print(f"[AVISO] {name}: Nenhum arquivo correspondente encontrado na pasta {os.path.dirname(path) if path else ''}")
    return all_found

def ler_conll(caminho_arquivo):
    sentences = []
    current_sentence = []
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    current_sentence.append(parts)
                elif current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
        if current_sentence:
            sentences.append(current_sentence)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {caminho_arquivo}")
    except Exception as e:
        print(f"Erro ao ler {caminho_arquivo}: {e}")
    return sentences

def ler_json(caminho_arquivo):
    data = None
    try:
        with open(caminho_arquivo, ' 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {caminho_arquivo}")
    except json.JSONDecodeError:
        print(f"Erro: Arquivo JSON mal formatado em {caminho_arquivo}")
    except Exception as e:
        print(f"Erro ao ler {caminho_arquivo}: {e}")
    return data

def ler_pickle(caminho_arquivo):
    data = None
    try:
        with open(caminho_arquivo, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {caminho_arquivo}")
    except Exception as e:
        print(f"Erro ao ler {caminho_arquivo}: {e}")
    return data

def ler_txt(caminho_arquivo):
    lines = []
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {caminho_arquivo}")
    except Exception as e:
        print(f"Erro ao ler {caminho_arquivo}: {e}")
    return lines
