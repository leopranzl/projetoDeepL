import os
import random
import shutil
from pathlib import Path

# Parâmetros
origem = 'garbage_classification'
destino = 'dataset_dividido'
proporcao_treino = 0.7
proporcao_validacao = 0.15
proporcao_teste = 0.15
seed = 123

# Garantir reprodutibilidade
random.seed(seed)

# Criar pastas de destino
for pasta in ['train', 'val', 'test']:
    for classe in os.listdir(origem):
        caminho_classe = os.path.join(origem, classe)
        if os.path.isdir(caminho_classe):
            os.makedirs(os.path.join(destino, pasta, classe), exist_ok=True)

# Separar os arquivos
for classe in os.listdir(origem):
    caminho_classe = os.path.join(origem, classe)
    if not os.path.isdir(caminho_classe):
        continue

    imagens = os.listdir(caminho_classe)
    random.shuffle(imagens)

    n = len(imagens)
    n_treino = int(n * proporcao_treino)
    n_val = int(n * proporcao_validacao)

    conjuntos = {
        'train': imagens[:n_treino],
        'val': imagens[n_treino:n_treino + n_val],
        'test': imagens[n_treino + n_val:]
    }

    for conjunto, arquivos in conjuntos.items():
        for arquivo in arquivos:
            origem_arquivo = os.path.join(caminho_classe, arquivo)
            destino_arquivo = os.path.join(destino, conjunto, classe, arquivo)
            shutil.copy2(origem_arquivo, destino_arquivo)

print('✅ Divisão concluída com sucesso!')