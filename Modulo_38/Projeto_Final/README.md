# Projeto Final

Neste projeto, o modelo de aprendizado foi construído através de um credit scoring para cartão de crédito, em um desenho amostral com 15 safras, e utilizando 12 meses de performance. O trabalho foi desenvolvido seguindo os passos abaixo:

  ## Subir no GITHUB todos os jupyter notebooks/códigos que você desenvolveu nesse ultimo módulo: 
   - O notebook "Mod38Projeto.ipynb" foi desenvolvido utilizando bibiiotecas como o pandas, numpy, seaborn, matplotlib, statsmodels, Scikit-Learn, SciPy e PyCaret; e realizou processos de análise descritiva (univariadas e bivariadas), desenvolvimento do modelo (tratamento dos dados e aplicação do statsmodels) e avaliação do modelo. Em seguida, foi trabalhada uma Pipeline para substitução de nulos, remoção de outliers, execução do PCA e criação de dummy. No final foi utilizada a biblioteca PyCaret para criação de modelo com o LightGBM, salvando no final o arquivo "lightgbm_model_final.pkl".
     
  ## Gerar um arquivo python (.py) com todas as funções necessárias para rodar no streamlit a escoragem do arquivo de treino: 
  - Foi gerado o arquivo "app_pycaret.py" para realizar as etapas de carregamento de um arquivo CSV, de limpeza, de análise descritiva univariada e bivariada, de predição de valores e salvamento de arquivo com os preditos para o usuário.
- link: https://ebac-curso-knlrroxfjkheeyuy4vgayh.streamlit.app/
    
  ## Criar um carregador de csv no streamlit: 
  - O arquivo "credit_scoring.csv" contempla os últimos 03 (três) meses da base original utilizada para a criação do modelo.

  ## Subir o csv no streamlit:
  - O usuário deverá fazer upload do arquivo "credit_scoring.csv".
    
  ## Criar um pipeline de pré processamento dos dados: 
  - O pipeline "lightgbm_model_final.pkl" é oriundo do processamento do "Mod38Projeto.ipynb".
    
  ## Utilizar o modelo treinado para escorar a base:
  - O modelo treinado será executado logo após a análise descritiva (ativada com o carregamento do arquivo CSV.
     
  ## nome_arquivo = 'model_final.pkl':
  - Nome dados ao arquivo pipeline.
    
  ## Gravar um vídeo da tela do streamlit em funcionamento:
  - Realizar a gravação do vídeo para mostrar o funcionamento da ferramenta conforme apresentado nas aulas gravadas do curso.
    
  ## Subir no Github o vídeo de funcionamento da ferramenta como README.md:
https://github.com/AntonioSCoelho97/EBAC-Curso/assets/132955062/94a044b4-677b-4cc3-9f6d-620060e49b22



