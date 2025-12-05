## 🧩 Código Fuente

### Ver Código: `word2vec_trainer.py`

[Enlace al Código Fuente](https://github.com/gastigarciajuri/embeddings-test/blob/main/test_2.py)

---

## 🧠 Análisis y Arquitectura

<details>
<summary>Explicación Detallada de la Lógica y NLP</summary>

Este **pipeline** sigue las etapas estándar de un proyecto de *Word Embeddings*, desde la ingesta de datos masivos hasta la aplicación de técnicas de álgebra lineal para demostrar la comprensión semántica.

### 1\. **Fase de Pre-procesamiento y Limpieza (`load_and_preprocess_corpus`)**

Esta es la etapa **crítica** para preparar el lenguaje natural para el modelo. Se utiliza la eficiencia de la librería `datasets` con la función `.map(batched=True)` para procesar el corpus de 1 millón de registros de manera **rápida y paralela**.

El *pipeline* de limpieza dentro de `process_batch` es robusto e incluye:

* Conversión a minúsculas (`.lower()`).
* Eliminación de entidades de *web* o redes sociales (URLs, menciones (`@`), hashtags (`#`)).
* Limpieza de puntuación y números (vía `gensim`).
* **Tokenización y Filtrado de *Stopwords***: Se utiliza `nltk.word_tokenize` para separar las palabras y, posteriormente, se eliminan las *stopwords* en español para asegurar que el modelo se enfoque solo en el significado y no en palabras de función (como "el", "la", "de").

---

### 2\. **Fase de Entrenamiento (`train_and_load_model`)**

Aquí se entrena el modelo `Word2Vec` utilizando el algoritmo **Skip-gram** (`sg=1`), que ha demostrado ser más eficaz para capturar relaciones semánticas complejas que el modelo CBOW.

* **Dimensionalidad (300):** Cada palabra será representada por un vector de 300 números, lo que permite capturar múltiples rasgos de significado.
* **Ventana (10):** El modelo considera 10 palabras a la izquierda y 10 a la derecha de la palabra objetivo para definir su contexto semántico.
* **Trabajadores (12):** Utiliza 12 núcleos de CPU para acelerar el entrenamiento del corpus masivo.

El modelo se entrena durante 10 épocas y se guarda en disco para garantizar la persistencia.


---

### 3\. **Verificación Semántica y Álgebra Lineal (`run_tests` & `interactive_mode`)**

La calidad de los *embeddings* se valida aplicando directamente el álgebra lineal sobre los vectores:

* **Similitud Coseno (`wv.similarity`):** Esta métrica mide el ángulo entre dos vectores. Una puntuación cercana a $1.0$ indica que los vectores apuntan en direcciones muy similares, lo que significa que las palabras tienen un significado o contexto muy relacionado (ej. "rey" y "reina").
* **Búsqueda Vectorial (`wv.most_similar`):** Encuentra las palabras más cercanas en el espacio vectorial a una palabra dada, demostrando la capacidad del modelo para "buscar" sinónimos o palabras relacionadas por significado.

---

### 4\. **Exportación para Visualización (`export_to_projector`)**

Esta función es clave para fines de documentación y exploración. Exporta la matriz de vectores (`embeddings.tsv`) y el vocabulario (`labels.tsv`) en el formato compatible con **TensorFlow Embedding Projector**. Esto permite reducir las 300 dimensiones a 3 (mediante PCA o t-SNE) y ver la nube de palabras.
</details>

---

## 🛑 Manejo de Errores y Excepciones

<details>
<summary>Robustez y Fallas Comunes</summary>

El script incorpora varias salvaguardas para asegurar una ejecución fluida en diferentes entornos:

* **Fallo en Carga de Corpus:** La función `load_and_preprocess_corpus` incluye un bloque `try-except` para intentar forzar la redescarga del *dataset* si la carga inicial falla, mitigando problemas comunes de caché de `datasets`.
* **Dependencia Opcional de Pandas:** La librería `pandas` se verifica dinámicamente en `main()`. Si no está instalada, el script lanza una **advertencia** y omite la exportación a TSV, permitiendo que el entrenamiento y el modo interactivo sigan funcionando.
* **Manejo de Vocabulario:** Las funciones `run_tests` y `interactive_mode` verifican si la palabra consultada (`if input_word not in wv:`) existe en el vocabulario del modelo, evitando errores de clave al intentar acceder a un vector inexistente.
* **`try-except` Interactivos:** El `interactive_mode` está envuelto en un `try-except` general que captura errores inesperados o interrupciones de teclado (`KeyboardInterrupt`), cerrando el modo interactivo de forma limpia.

</details>
