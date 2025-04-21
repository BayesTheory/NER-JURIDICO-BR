# Reconhecimento de Entidades Nomeadas para Textos Jurídicos Brasileiros (LeNER-Br)

Este repositório contém o código e os experimentos realizados para o projeto final da disciplina CPE 783 - Processamento de Linguagem Natural (Período 2025/1) do PEE/COPPE/UFRJ. O foco é o Reconhecimento de Entidades Nomeadas (NER) em documentos jurídicos brasileiros.
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.6%2B-green)

## 1. Descrição do Problema

 O projeto aborda o desafio de identificar e classificar automaticamente entidades nomeadas (como Legislação, Jurisprudência, Pessoas, Organizações, Localidades e Tempo) em textos do domínio jurídico brasileiro. Essa tarefa é complexa devido à linguagem técnica específica, estruturas textuais variadas e padrões de nomenclatura próprios da área. A extração dessas entidades é fundamental para diversas aplicações, como recuperação de informação jurídica, análise de precedentes e automação de processos legais.

## 2. Dataset Utilizado

 Foi utilizado o dataset **LeNER-Br (Legal Named Entity Recognition for Brazilian Portuguese)**.
* **Fonte:** Luz de Araujo et al., 2018 [1]
* **Conteúdo:** 70 documentos jurídicos (STF, STJ, TJMG, TCU)
* **Tamanho:** Aprox. 10.4k sentenças e 318k tokens
* **Entidades Anotadas:** PESSOA, JURISPRUDENCIA, TEMPO, LOCAL, LEGISLACAO, ORGANIZACAO
* **Formato:** BIO, com divisões pré-definidas de treino/desenvolvimento/teste.

## 3. Metodologia

Foram implementadas e comparadas duas abordagens principais:

### 3.1. Abordagem 1: Baseline Clássico (CRF)

* Modelo estatístico **Conditional Random Fields (CRF)**.
* **Características:** Features linguísticas (ex: Part-of-Speech tags), n-gramas.
* **Representações Vetoriais:** TF-IDF e/ou embeddings de palavras pré-treinados (Word2Vec, FastText).
* **Objetivo:** Estabelecer um baseline sólido com menor custo computacional.

### 3.2. Abordagem 2: Deep Learning Sequencial (BiLSTM-CRF)

* Arquitetura de rede neural recorrente **Bidirecional LSTM (BiLSTM)** seguida por uma camada **CRF**.
* **Objetivo:** Capturar dependências sequenciais de forma mais robusta, sendo uma alternativa viável a Transformers completos em hardware limitado.
* **Embeddings:** Uso de embeddings pré-treinados estáticos (Word2Vec/FastText) e contextuais (extraídos do BERTimbau [2] sem fine-tuning).
* **Regularização:** Dropout.

### (Opcional) 3.3. Abordagem 3: Exploração com Transformers

* Investigação limitada (se realizada) com fine-tuning do BERTimbau ou uso de seus embeddings em camadas de classificação simples.
