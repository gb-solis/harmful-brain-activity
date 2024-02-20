import pandas as pd
import numpy as np
import scipy as sp
from pyarrow import parquet as pq
from matplotlib import pyplot as plt

BASEPATH = '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/'

SAMPLE_RATE = 200 # amostra/segundo
COLUMN_NAMES = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']

def espectrograma(dados, nome='', cutoff=None):
    ''' Calcula e plota um espectrograma.

    Dado uma array `dados`, e opcionalmente o `nome` do conjunto de dados e
    uma frequência de `cutoff` em Herz, plota o espectrograma correspondente
    aos dados.
    '''
    freqs, times, spec = sp.signal.spectrogram(dados)

    times /= SAMPLE_RATE # segundo
    freqs *= SAMPLE_RATE # Hz
    
    i_cut = next((i for i,f in enumerate(freqs) if f >= cutoff), None)
    
    _ = plt.pcolormesh(times, freqs[:i_cut], spec[:i_cut])#, norm='log')
    plt.xlabel('time ($s$)')
    plt.ylabel(r'frequency ($\mathrm{Hz}$)')
    plt.title(f'Espectrograma {nome}')


def histograma(dados, bins=100, nome=''):
    ''' Calcula e plota um histograma.

    Dada uma array `dados` e, opcionalmente o número de calhas `bins` e o
    `nome` do conjunto de dados, plota o histograma dos valores dos dados.
    '''
    plt.hist(dados, bins=bins)
    plt.xlabel('Leitura (u.a.)')
    plt.ylabel('Contagens')
    plt.title(f'Histograma {nome}')


def série_temporal(dados, nome='', size=(50,5)):
    ''' Calcula e plota uma série temporal.

    Plota a série temporal dos `dados`, com opções para configurar o nome e o
    tamanho do plot (em polegadas).
    '''
    plt.plot([i/SAMPLE_RATE for i in range(len(dados))], dados, '.-')
    plt.xlabel('tempo (s)')
    plt.ylabel('leitura (u.a.)')
    plt.title(f'Leituras {nome}')
    fig = plt.gcf()
    fig.set_size_inches(*size)


def data_iterator(path=BASEPATH):
    ''' Retorna um iterador, contendo os dados dos arquivos passados em `path`.
    `path` pode ser tanto uma lista de arquivos quanto um diretório.
    '''
    data = ds.dataset(path, format='parquet')
    iterator = data.to_batches()
    return iterator

def combined_hist(path=BASEPATH):
    ''' Retorna os histogramas conjuntos de todos os arquivos em `path`, um para
    cada electrodo.
    '''
    counters = { name : Counter() for name in COLUMN_NAMES }
    iterator = data_iterator(path)
    for i in iterator:
        for name in i.column_names:
            col = i.column(name).drop_null().cast(pa.int32(), safe=False)
            counts = { x[0].as_py() : x[1].as_py() for x in pc.value_counts(col)}
            counters[name].update(counts)
    return counters
