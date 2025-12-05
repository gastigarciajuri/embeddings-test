🇪🇸 Word2Vec en Español - Modelado de Embeddings Vectoriales
Este repositorio contiene un script de Python diseñado para entrenar un modelo Word2Vec a partir de un corpus masivo en español. El objetivo es generar representaciones vectoriales densas (embeddings) de palabras que capturan sus relaciones semánticas y sintácticas.
El script realiza la descarga, el pre-procesamiento del corpus y el entrenamiento del modelo, finalizando con la exportación de los vectores para su visualización y un modo interactivo de pruebas.
✨ Funcionalidades Clave
El script principal (main.py o nombre_del_script.py) integra las siguientes características fundamentales del pipeline de PLN:
Configuración de PLN (NLTK): Descarga automática de recursos necesarios (stopwords, punkt) y definición de stop words en español.
Carga de Corpus Masivo: Utiliza el dataset josecannete/large_spanish_corpus (un subconjunto de 1 millón de registros) para garantizar un entrenamiento rápido y de alta calidad.
Pre-procesamiento Intensivo:
Conversión de texto a minúsculas.
Eliminación de URLs, menciones (@) y hashtags (#).
Limpieza de puntuación, números y palabras cortas (menores a 2 caracteres).
Tokenización y filtrado de stop words.
Entrenamiento Word2Vec: Entrena el modelo con el algoritmo Skip-gram (sg=1), utilizando una dimensionalidad de 300 vectores y una ventana de contexto de 10.
Persistencia del Modelo: Guarda el modelo entrenado como word2vec_large_spanish_corpus.model para permitir la carga sin necesidad de re-entrenamiento.
Exportación para Visualización: Genera automáticamente los archivos embeddings.tsv y labels.tsv para su uso en el TensorFlow Projector.
Modo Interactivo: Permite al usuario consultar la similitud (Producto Coseno) entre palabras y encontrar los términos más similares una vez finalizado el entrenamiento.
🛠️ Requisitos Previos
Necesitas tener Python 3.8+ instalado en tu sistema.
Instalación de Librerías
Recomendamos el uso de un entorno virtual. Instala todas las dependencias requeridas con pip:
pip install datasets pandas nltk gensim


🚀 Inicialización y Ejecución
Para iniciar el proceso de entrenamiento y acceder al modo interactivo, ejecuta el script principal (asegúrate de reemplazar main.py por el nombre de tu archivo si es diferente):
python main.py


Flujo de Ejecución
El script seguirá la siguiente secuencia, con mensajes de progreso en consola:
Configuración inicial de NLTK y descarga de recursos.
Carga y pre-procesamiento por lotes del corpus.
Entrenamiento del modelo Word2Vec.
Guardado del modelo en disco y generación de los archivos .tsv.
Ejecución de pruebas de verificación semántica (similitud entre "rey" y "reina", similares a "españa").
Ingreso al Modo Interactivo (usa sim, comp, o salir).
📊 Visualización de Embeddings
Una vez generados, los archivos .tsv son esenciales para la inspección 3D del espacio vectorial:
embeddings.tsv: La matriz de vectores generada por Word2Vec.
labels.tsv: Las etiquetas (palabras) asociadas a cada vector.
Pasos para la Visualización en 3D:
Abre el TensorFlow Projector en tu navegador.
Haz clic en el botón "Load" (Cargar) en el panel izquierdo.
Sube embeddings.tsv como el archivo de vectores.
Sube labels.tsv como el archivo de etiquetas.
Utiliza los métodos de reducción de dimensionalidad (como PCA o t-SNE) en el panel derecho para explorar la agrupación semántica de las palabras.
