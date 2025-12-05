import sys
import datasets 
import pandas as pd
import nltk
import re
from gensim.models import Word2Vec 
from gensim.parsing.preprocessing import strip_punctuation, strip_numeric, strip_short
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

# ======================================================================================
# --- 1. CONFIGURACIÓN INICIAL (CLASE 1) ---
# ======================================================================================

def setup_environment():
    """Configura y descarga los recursos necesarios de NLTK."""
    print("Iniciando configuración y descarga de recursos de NLTK...")
    
    # Descarga de recursos necesarios (punkt para word_tokenize, stopwords para limpieza)
    nltk.download('stopwords', quiet=True) 
    nltk.download('punkt', quiet=True)
    
    # Definir y retornar las stopwords en español para el proceso de limpieza
    spanish_stopwords = set(stopwords.words('spanish'))
    
    print("Importaciones y configuración de NLTK completadas.")
    return spanish_stopwords

# ======================================================================================
# --- 2. CARGA Y PRE-PROCESAMIENTO (CLASE 6: VECTORIZACIÓN TRADICIONAL) ---
# ======================================================================================

def load_and_preprocess_corpus(spanish_stopwords):
    """Carga el corpus y define la función de pre-procesamiento/tokenización."""
    
    # --- 2.1. Carga y Selección del Corpus ---
    print("\nIniciando la carga del corpus 'josecannete/large_spanish_corpus'...")
    try:
        # Intenta cargar el dataset de forma normal
        dataset_corpus = datasets.load_dataset('josecannete/large_spanish_corpus', 'ParaCrawl')
    except Exception as e:
        print(f"Error al cargar el dataset: {e}. Intentando forzar redescarga...")
        # Lógica de fallback: si falla la carga, intenta forzar la redescarga.
        dataset_corpus = datasets.load_dataset('josecannete/large_spanish_corpus', 'ParaCrawl', download_mode='force_redownload')

    # Seleccionar un subconjunto de 1,000,000 registros para entrenamiento rápido (sin cambiar lógica)
    subset = dataset_corpus['train'].select(range(1000000))
    print(f"Corpus cargado. Subconjunto de entrenamiento seleccionado con {len(subset)} registros.")

    # --- 2.2. Función de Pre-procesamiento y Tokenización (Clase 6) ---
    def process_batch(batch):
        """Limpia, tokeniza y elimina stopwords de un batch de sentencias."""
        cleaned_text_list = []      
        for text in batch['text']:
            if not text:
                cleaned_text_list.append([])
                continue

            # 1. Convertir a minúsculas
            text = text.lower()

            # 2. Eliminar URLs, menciones y puntuación
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            text = re.sub(r'\@\w+|#\w+', '', text)
            text = strip_punctuation(text)

            # 3. Eliminar números y palabras cortas
            text = strip_numeric(text)
            text = strip_short(text, minsize=2)
            
            # 4. Tokenizar y filtrar Stopwords
            tokens = word_tokenize(text)
            cleaned_tokens = [
                word for word in tokens 
                if word.isalpha() and word not in spanish_stopwords 
            ]
            cleaned_text_list.append(cleaned_tokens)
            
        # ESTA ESTRUCTURA ES NECESARIA PARA LA API DE HUGGING FACE DATASETS
        return {'text': cleaned_text_list} 

    # --- 2.3. Aplicación del Procesamiento ---
    print("Iniciando tokenización y limpieza del corpus con map(batched=True)...")
    sentences_corpus = subset.map(process_batch, batched=True)

    # Extraer el corpus tokenizado (lista de listas de strings)
    tokenized_corpus = sentences_corpus['text'] 
    print(f"Tokenización y limpieza completadas. Total de sentencias tokenizadas: {len(tokenized_corpus)}")
    
    return tokenized_corpus


# ======================================================================================
# --- 3. ENTRENAMIENTO, CARGA Y EXPORTACIÓN (CLASES 8, 9) ---
# ======================================================================================

def export_to_projector(wv):
    """Exporta los vectores y etiquetas a archivos TSV para visualización en TensorFlow Projector."""
    print("\n--- 3.3. 💾 Exportando datos para el Embedding Projector (TensorFlow) ---")

    try:
        # wv (Word Vectors): La matriz de embeddings final (la Matriz de Datos).
        vectors = wv.vectors # Matriz de NumPy con todos los embeddings (Nx300)
        words = wv.index_to_key # Lista de todas las palabras (N)
        
        # 1. Exportar los vectores a embeddings.tsv
        df_vectors = pd.DataFrame(vectors)
        df_vectors.to_csv(
            path_or_buf='embeddings.tsv', 
            sep='\t', 
            index=False,
            header=False # El archivo de embeddings no debe tener encabezados
        )
        print("✅ Archivo 'embeddings.tsv' (Vectores) creado exitosamente.")
        
        # 2. Exportar las etiquetas (palabras) a labels.tsv
        df_words = pd.DataFrame(words)
        df_words.to_csv(
            path_or_buf='labels.tsv', 
            sep='\t', 
            index=False,
            header=False # El archivo de etiquetas no debe tener encabezados
        )
        print("✅ Archivo 'labels.tsv' (Etiquetas/Palabras) creado exitosamente.")
        
        print("\n¡Listo! Puedes subir ambos archivos al Embedding Projector (projector.tensorflow.org).")
        
    except ImportError:
        print("\n❌ ERROR: La librería 'pandas' no está instalada. Ejecuta: pip install pandas")
    except Exception as e:
        print(f"\n❌ ERROR al exportar archivos TSV: {e}")


def train_and_load_model(tokenized_corpus):
    """Entrena el modelo Word2Vec y devuelve el objeto WordVector (wv)."""
    
    print("\n--- 3.1. Iniciando el Entrenamiento del Modelo Word2Vec (Skip-gram, sg=1) ---")

    # Parámetros del modelo (vector_size=300: dimensión de los vectores)
    word2vec_model = Word2Vec(
        sentences=tokenized_corpus, # Corpus tokenizado
        vector_size=300, # Dimensiones del vector. Concepto de "Dimensionalidad".
        window=10, # Contexto de la ventana (palabras a la izquierda y derecha).
        min_count=2, # Mínimo de apariciones para que una palabra sea considerada.
        workers=12, # Ajusta según los núcleos de tu CPU para paralelización. (Mi cpu tiene 20, deje 2 libres)
        sg=1, # Algoritmo de entrenamiento: 1 para Skip-gram (mejor para semántica).
    )
    
    # Iniciar el entrenamiento formalmente
    word2vec_model.train(
        tokenized_corpus,
        total_examples=word2vec_model.corpus_count,
        epochs=10
    )
    print("Entrenamiento de Word2Vec completado.")

    # --- 3.2. Guardar y Cargar WordVectors ---
    model_filename = "word2vec_large_spanish_corpus.model"
    word2vec_model.save(model_filename) # Permite la Persistencia (no re-entrenar).
    print(f"Modelo guardado como: {model_filename}")

    # wv (Word Vectors): Es la matriz de embeddings final (la Matriz de Datos).
    wv = word2vec_model.wv 
    
    return word2vec_model, wv


# ======================================================================================
# --- 4. VERIFICACIÓN Y PRUEBAS (APLICACIÓN DEL ÁLGEBRA LINEAL) ---
# ======================================================================================

def run_tests(word2vec_model, wv):
    """Ejecuta pruebas básicas para verificar la calidad semántica del modelo."""
    
    print("\n--- 4.1. Verificación del Vocabulario y Vectores ---")
    
    # Mostrar las primeras 20 palabras del vocabulario (Índice de la matriz de embeddings)
    vocabulario = wv.index_to_key
    print("\n--- 🧠 Primeras 20 palabras más frecuentes en el vocabulario: ---")
    print(vocabulario[:20])

    # Acceder al vector de una palabra específica y su dimensionalidad
    palabra_a_vectorizar = "tecnología"
    if palabra_a_vectorizar in wv:
        vector = wv[palabra_a_vectorizar]
        # Vemos el vector de 300 dimensiones
        print(f"\n--- 📊 Vector de '{palabra_a_vectorizar}' ({vector.shape[0]} dimensiones): ---")
        print(vector[:10]) 
    else:
        print(f"\n❌ La palabra '{palabra_a_vectorizar}' no se encontró en el vocabulario.")

    # --- 4.2. Prueba de Similitud (Producto Coseno) ---
    print("\n--- 4.2. Pruebas de Similitud (Álgebra Lineal) ---")
    palabra_1 = "rey"
    palabra_2 = "reina"
    if palabra_1 in wv and palabra_2 in wv:
        # wv.similarity() usa el Producto Coseno: mide la alineación direccional de los vectores.
        similarity = wv.similarity(palabra_1, palabra_2)
        print(f"Similitud (Producto Coseno) entre '{palabra_1}' y '{palabra_2}': {similarity:.4f}")
    else:
        print(f"Una o ambas palabras ('{palabra_1}', '{palabra_2}') no están en el vocabulario.")

    # --- 4.3. Palabras más similares (Búsqueda Vectorial) ---
    test_word = "españa"
    if test_word in wv:
        # most_similar realiza una búsqueda de Proyección Ortogonal por Similitud Coseno
        most_similar = wv.most_similar(test_word, topn=5)
        print(f"\n5 palabras más similares a '{test_word}' (Búsqueda Vectorial):")
        for word, score in most_similar:
            print(f"- {word}: {score:.4f}")
    else:
        print(f"La palabra '{test_word}' no está en el vocabulario.")

# ======================================================================================
# --- 5. MODO INTERACTIVO (USO CONTINUO DEL MODELO) ---
# ======================================================================================

def interactive_mode(wv):
    """Permite al usuario interactuar con el modelo para buscar similitudes y comparaciones."""

    print("\n==============================================")
    print("         MODO INTERACTIVO DE WORD2VEC         ")
    print("==============================================")
    print("Escribe 'salir' para terminar.")
    print("Escribe 'sim' para encontrar palabras similares (Búsqueda Vectorial).")
    print("Escribe 'comp' para comparar dos palabras (Similitud Coseno).")

    while True:
        try:
            action = input("\nElige acción (sim/comp/salir): ").lower()

            if action == 'salir':
                print("Modo interactivo finalizado. ¡Adiós!")
                break
            
            elif action == 'sim':
                # --- FUNCIONALIDAD: ENCONTRAR SIMILARES (BÚSQUEDA VECTORIAL) ---
                input_word = input("Palabra (similares): ").lower()
                try:
                    top_n = int(input("Cantidad (Ej: 5): "))
                except ValueError:
                    print("Usando 5 por defecto.")
                    top_n = 5
                    
                if input_word not in wv:
                    print(f"❌ La palabra '{input_word}' no está en el vocabulario.")
                    continue

                print(f"\n-- 🔎 Palabras más similares a '{input_word}' (Top {top_n}):")
                most_similar = wv.most_similar(input_word, topn=top_n)
                for word, score in most_similar:
                    print(f"   -> {word}: {score:.4f}")

            elif action == 'comp':
                # --- FUNCIONALIDAD: COMPARAR 2 PALABRAS (SIMILITUD COSEN) ---
                palabra1 = input("Primera palabra: ").lower()
                palabra2 = input("Segunda palabra: ").lower()

                if palabra1 not in wv or palabra2 not in wv:
                    print(f"❌ Una o ambas palabras ('{palabra1}', '{palabra2}') no están en el vocabulario.")
                    continue
                
                # Esto es la aplicación directa de la fórmula del Producto Coseno (Álgebra Lineal)
                similitud = wv.similarity(palabra1, palabra2)
                
                print("\n-- ⚖️ Resultado de la Comparación:")
                print(f"   Similitud Coseno entre '{palabra1}' y '{palabra2}': {similitud:.4f}")
                print("   (Cerca de 1 = Muy similares; Cerca de 0 = Neutral; Cerca de -1 = Opuestos)")

            else:
                print("Opción no válida. Usa 'sim', 'comp' o 'salir'.")
            
        except Exception as e:
            if "KeyboardInterrupt" in str(e):
                 print("\nModo interactivo finalizado debido a una interrupción.")
            else:
                print(f"\nOcurrió un error inesperado: {e}")
            break

# ======================================================================================
# --- FUNCIÓN PRINCIPAL DE EJECUCIÓN ---
# ======================================================================================

def main():
    # Asegura que pandas esté disponible para la exportación a TSV (requiere instalación manual)
    if 'pandas' not in sys.modules:
        try:
            global pd
            import pandas as pd
        except ImportError:
            print("\n🚨 Advertencia: 'pandas' no está instalado. La exportación a TSV fallará. Ejecuta: pip install pandas")

    try:
        # 1. Configuración (Clase 1)
        spanish_stopwords = setup_environment()
        
        # 2. Carga y Pre-procesamiento (Clase 6)
        tokenized_corpus = load_and_preprocess_corpus(spanish_stopwords)
        
        # 3. Entrenamiento y Carga del Modelo (Clases 8, 9)
        word2vec_model, wv = train_and_load_model(tokenized_corpus)
        
        # 3.3. Exportación para Visualización (CLASE DE PCA/PROYECCIÓN)
        if 'pandas' in sys.modules:
            export_to_projector(wv)
        
        # 4. Verificación y Pruebas (Clases 9, 10)
        run_tests(word2vec_model, wv)
        
        # 5. Modo Interactivo (Uso del Modelo)
        interactive_mode(wv)
        
    except Exception as e:
        print(f"\n[ERROR FATAL EN LA EJECUCIÓN] El script se ha detenido: {e}")

if __name__ == "__main__":
    main()