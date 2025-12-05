🇪🇸 Word2Vec en Español - Modelado de Embeddings Vectoriales
Este repositorio contiene un script de Python diseñado para entrenar un modelo Word2Vec a partir de un corpus masivo en español. El objetivo es generar representaciones vectoriales densas (embeddings) de palabras que capturan sus relaciones semánticas y sintácticas. [Imagen de Arquitectura Word2Vec]
El script realiza la descarga, el pre-procesamiento del corpus y el entrenamiento del modelo, finalizando con la exportación de los vectores para su visualización y un modo interactivo de pruebas.
✨ Funcionalidades Clave
Lo que hemos implementado hasta ahora en el script principal:
Configuración de PLN (NLTK): Descarga automática de recursos (stopwords, punkt) y definición de stop words en español.
Carga de Corpus: Utiliza el dataset josecannete/large_spanish_corpus (subconjunto de 1 millón de registros) para garantizar un entrenamiento rápido y de alta calidad.
Pre-procesamiento Intensivo:
Conversión a minúsculas.
Eliminación de URLs, menciones (@) y hashtags (#).
Limpieza de puntuación, números y palabras cortas (menores a 2 caracteres).
Tokenización y filtrado de stop words.
Entrenamiento Word2Vec: Entrena un modelo con el algoritmo Skip-gram (300 dimensiones, ventana de contexto de 10) para capturar relaciones semánticas.
Persistencia del Modelo: Guarda el modelo entrenado (word2vec_large_spanish_corpus.model) para evitar re-entrenamientos futuros.
Exportación para Visualización: Genera automáticamente los archivos embeddings.tsv y labels.tsv necesarios para el TensorFlow Projector.
Modo Interactivo: Permite al usuario interactuar con el modelo para consultar la similitud (Producto Coseno) entre palabras y encontrar los términos más similares.
🛠️ Requisitos Previos
Necesitas tener Python 3.8+ instalado y las siguientes librerías de Python.
Instalación de Librerías
Recomendamos crear un entorno virtual e instalar todas las dependencias usando pip:
pip install datasets pandas nltk gensim


🚀 Inicialización y Ejecución
Para iniciar el proceso de entrenamiento y acceder al modo interactivo, simplemente ejecuta el script principal:
python nombre_del_script.py


(Asume que el archivo principal se llama nombre_del_script.py)
Flujo de Ejecución
El script iniciará la configuración de NLTK y la descarga del corpus.
Realizará el pre-procesamiento del corpus (esta es la parte más intensiva en recursos).
Entrenará el modelo Word2Vec.
Guardará el modelo y generará los archivos embeddings.tsv y labels.tsv.
Ejecutará pruebas básicas de similitud.
Ingresará automáticamente al Modo Interactivo, donde podrás probar las capacidades semánticas del modelo.
📊 Visualización de Embeddings
Al finalizar la ejecución, se habrán creado dos archivos clave:
embeddings.tsv: Contiene la matriz de vectores (las 300 dimensiones de cada palabra).
labels.tsv: Contiene la lista de las palabras (etiquetas) correspondientes a cada vector.
Estos archivos permiten la visualización 3D de tus embeddings mediante técnicas de reducción de dimensionalidad (como PCA o t-SNE) en el TensorFlow Projector.
Pasos para visualizar:
Abre http://projector.tensorflow.org/.
Haz clic en "Load" (Cargar) en el panel izquierdo.
Sube embeddings.tsv como el archivo de vectores.
Sube labels.tsv como el archivo de etiquetas.
¡Explora las relaciones semánticas de las palabras en el espacio 3D!
