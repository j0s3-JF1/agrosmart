# Relatório Técnico: AgroSmart - Protótipo de Visão Computacional

**Projeto:** Sistema de Identificação de Doenças em Folhas via IA
**Objetivo:** Desenvolver uma solução local e offline para processar imagens de folhas (saudáveis e doentes) e auxiliar produtores na tomada de decisão sem dependência de APIs em nuvem.
**Disciplina/Instituição:** FIAP - Inteligência Artificial e Visão Computacional

---

## 1. Tecnologias Utilizadas e Propósito

Para a construção deste sistema, fundamentou-se em uma arquitetura de aprendizado profundo (Deep Learning) executada localmente, garantindo que os dados não precisassem ser enviados para servidores externos.

*   **Python 3:** Linguagem base escolhida pela sua vasta compatibilidade com bibliotecas matemáticas e de Machine Learning.
*   **TensorFlow & Keras:** O pacote central do nosso cérebro artificial. Utilizamos a API de alto nível do Keras montada em cima do TensorFlow para construir uma Rede Neural Convolucional (CNN). Esta arquitetura foi escolhida por ser o estado da arte para extração de qualidades em imagens (textura, bordas e cores vitais para diferenciar pragas).
*   **OpenCV & Pillow (cv2/PIL):** O OpenCV se responsabiliza pela análise dimensional de cores em tempo real do HUD da Webcam e a criação de Bounding Boxes em cima de formas biológicas. Já o pacote Pillow renderiza de forma veloz estes arrays visuais para interface desktop.
*   **Tkinter:** Utilizado para elevar a usabilidade de um projeto puramente de prompt terminal para um software interativo (Graphical User Interface - GUI).
*   **Pandas & NumPy:** Organização algébrica. O NumPy lida com a redução e conversão de matrizes de pixels para a IA processar; O Pandas assume a organização estruturada em `DataFrames` e exporta com segurança para arquivos de atividade local (Logs).

---

## 2. Metodologia: O Processo de Criação

O desenvolvimento foi segmentado nas seguintes fases metodológicas:

**I. Coleta e Organização de Amostras (Data Preparation)**
Foram baixadas pouco mais de 15.000 imagens brutas segmentadas. Foi realizada uma organização do *dataset* achatando as diversas doenças numa única categoria binária: **DOENTE** vs **SAUDÁVEL**. Em seguida, um *split* separou a massa de aprendizado em `Train` (70%), `Valid` (20%) e `Test` (10%).

**II. Treinamento do Modelo (Model Training - `train.py`)**
O Keras *Data Generator* padronizou as imagens para 128x128 em RGB, e aplicou o *data augmentation* para combater o *Overfitting*. Utilizamos uma CNN de 3 camadas convolucionais Max-Pooling + Dropout. O modelo treinou por 10 épocas e resultou em um *Brain Model* leve focado para CPU: `.h5`.

**III. Lógica Visual de Segmentação (HSV Thresholding)**
Com a janela abrindo e a câmera injetando sinal de 30 FPS, o algoritmo não processa computação inútil. Em tempo real ele converte os frames para tons HSV (Hue Saturation Value) isolando limites perfeitos de plantas vivos e secas (Saturação 60, Hue 30 a 85). Se a sombra isolada tiver densidade em pixels (>8000), o classificador envia um recorte inteligente à Inteligência Artificial.

**IV. A Inferência (Prototype - `app.py`)**
A IA recebe o *crop* foliar, recalcula canais em RGB e joga num ambiente matemático 0 a 1. Nossa arquitetura mapeou a classe Saudável alfabeticamente logo um retorno `>0.5` projeta a planta saudável com Bounding Box Verde e abaixo disso planta Anômala (Doente) apontada via Bounding Box Vermelha perante alerta na interface. A confiança (%) da predição atesta a solidez do processo e valida em tempo real a performance do projeto.

**V. Banco de Dados Local (Exportação)**
No software rodando, após uma detecção satisfatória e validada, o usuário aciona o botão de Exportar Evento na tela. O sistema compila o horário real, a nomenclatura da doença e a nota avaliada convertendo em dicionário Pandas em `mode='append'`. O `dados_agrosmart.csv` então funciona perfeitamente como sistema centralizado e não-perigoso para manipulação estatística agrária posterior.

---

## 3. Aplicabilidade do Protótipo

A solução apresentada sendo totalmente offline, apresenta grande valor no agronegócio de precisão:
1.  **Monitoramento em Drones / Equipamentos Rurais Simples:** Como o processamento computacional ocorre na CPU de maneira focada (sem nuvem e apenas rodando inferência após análise orgânica HSV de luz), qualquer notebook modesto embarcado pode efetivamente julgar pragas.
2.  **Rastreabilidade Agronômica:** Através do documento exportado autonomamente, tabelas ou PowerBI podem relatar picos de infecção numa fazenda no final do período.
3.  **UI Versátil:** Fatorado com Tkinter, os botões tornam acessível para a equipe em laboratórios de triagem ativarem as microscópicas ou conectores com facilidade, incluindo opções de fotos que chegam esporadicamente via WhatsApp de produtores locais.
