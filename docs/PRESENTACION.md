# LeadScore — Apunte para la presentación

Guía breve para presentar y defender el proyecto. Ordenada como conviene contarlo,
con el vocabulario de las semanas 5 y 6 (secuencias, RNN/GRU, atención, Transformers).

---

## 1. El pitch en 30 segundos

> En e-commerce solo el **4,3 %** de las sesiones termina en compra. Gastar el mismo
> esfuerzo comercial en todos los visitantes es tirar plata. **LeadScore lee la secuencia
> de clics de una sesión en vivo y estima la probabilidad de que termine en compra**, para
> priorizar a quién contactar y con qué acción.
>
> La secuencia es el dato: `view → view → addtocart` no significa lo mismo que
> `addtocart → view → view`. Por eso el proyecto compara un modelo **sin memoria**
> (baseline tabular), uno **con memoria** (GRU) y uno **con atención** (Transformer),
> y los combina en un **ensamble híbrido**.

## 2. La problemática (por qué esto es un problema de negocio)

- 166.957 sesiones reales de RetailRocket, 137.124 visitantes. Solo 4,3 % convierte.
- LeadScore entrega un score y una **acción recomendada** por banda:

| Banda | p_final | Acción comercial |
|---|---|---|
| 🔥 caliente | ≥ 0,70 | Priorizar contacto; **no** quemar descuento (ya va a comprar) |
| 🌡️ tibio | 0,40 – 0,70 | Descuento o recordatorio de carrito (aquí el incentivo sí mueve la aguja) |
| ❄️ frío | < 0,40 | No invertir; retargeting barato |

- Además segmenta la sesión con **K-means (k=4)**: `decidido`, `comparador`,
  `explorador`, `carrito abandonado`. El score dice *cuánto*; el segmento dice *cómo*.

## 3. Cómo se conecta con la materia (la sección que importa)

Cada concepto de las semanas 5 y 6 tiene un lugar concreto en el código:

| Concepto de clase | Dónde vive en LeadScore |
|---|---|
| **Datos secuenciales** (el orden importa) | Una sesión es una secuencia de eventos ordenada en el tiempo |
| **Por qué un MLP no basta** (largo variable, sin memoria) | Las sesiones van de 3 a 50 eventos; el baseline tabular es justamente ese MLP-sin-memoria contra el que comparo |
| **Texto → números** (one-hot, embedding) | Mismo problema, con eventos en vez de palabras: vocabulario `{PAD:0, view:1, addtocart:2}` → **Embedding de 32 dim** |
| **Por qué no BoW / TF-IDF** | Bolsa de palabras **pierde el orden**, y el orden *es* mi señal. Un baseline de conteos ya lo tengo: es el modelo tabular |
| **RNN / estado oculto `h_t`** | El `h_t` de la GRU es la **intención acumulada** de la sesión hasta ese clic. Uso el último `h_t` como vector-resumen → capa lineal → probabilidad |
| **LSTM vs GRU** | Elegí **GRU**: menos compuertas (solo *reset* R y *update* Z), más liviana, y la clase la recomienda para secuencias. Con sesiones de 4 eventos, la mochila de la LSTM sería sobredimensionada |
| **BPTT / vanishing gradient** | Lo que motiva usar GRU y no una RNN simple: las compuertas dejan pasar el gradiente en secuencias largas (mi cola llega a 50 eventos) |
| **Padding** | Las secuencias del batch tienen largos distintos → hay que rellenar. La GRU usa `pack_padded_sequence` |
| **Masked Attention (ignora el padding)** | El Transformer usa `src_key_padding_mask`: **el relleno no recibe atención**. Es literalmente la slide de *Scaled Dot Product Attention*. Tengo un **test que prueba que la predicción no cambia al agregar padding** |
| **Q, K, V + softmax** | Cada evento se proyecta a consulta/clave/valor; el softmax reparte los pesos de atención entre los eventos de la sesión |
| **Self-attention** | Cada evento mira a los demás **directamente**, sin recurrencia: el `addtocart` final puede mirar al primer `view` sin pasar por los del medio |
| **Positional Encoding** | Sin recurrencia el modelo no sabe el orden → encoding sinusoidal para devolvérselo |
| **Multi-head attention** | 4 cabezas, 2 capas encoder, d_model 64, feed-forward 128, dropout 0,1 |
| **Atención = interpretabilidad / atribución de errores** | Extraigo los pesos de la última capa → `results/attention_heatmap.png`. **Se ve en qué clic se fijó el modelo** |
| **Matriz de confusión, Accuracy, Precision, Recall, F1, AUC** | Las 5 métricas para los 4 modelos sobre el **mismo test set** → `results/metrics.json` + 4 matrices en `results/` |

**Un aporte propio sobre la materia:** además del token de evento, el modelo recibe el
**`delta_t`** (segundos entre clics), proyectado a 16 dimensiones y concatenado al
embedding. El texto no tiene esto: entre dos palabras no pasa el tiempo. Entre dos clics
sí, y **dudar 5 minutos no es lo mismo que hacer clic al toque**.

## 4. Los números (test set, n = 25.032)

| Modelo | Accuracy | Precision | Recall | F1 | AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline (sin memoria) | 0,871 | 0,236 | 0,913 | 0,375 | 0,916 |
| GRU (memoria `h_t`) | 0,885 | 0,263 | 0,944 | 0,411 | 0,945 |
| Transformer (atención) | 0,883 | 0,260 | 0,945 | 0,407 | 0,943 |
| **Híbrido (stacking)** | **0,942** | **0,397** | 0,680 | **0,501** | **0,945** |

**Cómo leerlos, en una frase cada uno:**

1. **La secuencia aporta señal:** GRU y Transformer suben el AUC de 0,916 a ~0,944 sobre
   el baseline. Leer *el orden* de los clics gana sobre contarlos.
2. **El híbrido gana:** F1 0,375 → **0,501** y precision 0,24 → **0,40**. Combinar los tres
   (stacking) supera a cualquiera por separado.
3. **Ojo con la accuracy:** con 4,3 % de conversión, un modelo que diga "nadie compra"
   saca 95,7 % de accuracy y es inútil. Por eso miro **F1 y AUC**, no accuracy.
4. **El híbrido cambia recall por precision** (0,94 → 0,68) a propósito: el umbral se
   calibró maximizando F1 en validación. En negocio: prefiero llamar a 100 leads y que
   40 compren, que llamar a 400 para pescar los mismos.

## 5. La pregunta difícil: ¿por qué el Transformer NO le ganó a la GRU?

**Es un empate técnico (0,407 vs 0,411 de F1), y era lo esperable. La razón está en los datos:**

> **La mediana de una sesión es de 4 eventos. El 75 % tiene 5 o menos.**

La propia tabla comparativa del curso lo dice: GRU/LSTM sirven para **secuencias cortas**;
los Transformers ganan cuando hay **contexto largo** y **ambigüedad** que resolver. Aquí:

- **No hay contexto largo:** en 4 clics, la memoria `h_t` de la GRU no alcanza a
  degradarse. No hay vanishing gradient que salvar.
- **No hay polisemia:** mi vocabulario tiene **2 tokens** (`view`, `addtocart`). El ejemplo
  de "manta" (frazada vs pez) no existe acá: un `view` es siempre un `view`. La atención
  brilla desambiguando significados según el contexto — no tengo qué desambiguar.

**Entonces, ¿el Transformer fue en vano?** No, por dos razones:
1. **Aporta al ensamble:** comete errores *distintos* a los de la GRU, y de eso vive el
   stacking. El híbrido con los tres es mejor que con dos.
2. **Es el único que explica su decisión:** el heatmap de atención muestra en qué clic se
   fijó. La GRU no ofrece eso.

*(Y es un resultado honesto: forzar que el Transformer ganara habría requerido inflar el
experimento. Reportar el empate y explicarlo con los datos vale más.)*

## 6. Las tres decisiones que defienden el rigor

1. **Anti-fuga de datos (lo más importante).** La etiqueta `converted` se calcula sobre la
   sesión completa, pero de la secuencia de entrada **se borra la transacción y todo lo
   posterior**. Si no, el modelo "predeciría" la compra viendo la compra. Está
   **verificado por un test automático**, no solo por confianza.
2. **Split 70/15/15 estratificado POR VISITANTE**, no por sesión. Si un visitante tuviera
   una sesión en train y otra en test, el modelo lo reconocería y las métricas serían
   mentira. Hay un test que prueba que los conjuntos de visitantes son disjuntos.
3. **Un solo test set para los 4 modelos**, y el umbral y el stacking se ajustan **en
   validación, nunca en test**. Por eso las cifras son comparables entre sí.

Extra: seed 42 en numpy/torch/sklearn → reproducible.

## 7. Guion de la demo (3 minutos)

```bash
docker compose up --build      # → http://localhost:8080
```

1. **Simulador** (`/`): agregar `view` → el gauge marca frío ❄️. Agregar otro `view` →
   sube. Agregar **`addtocart`** → salta a **caliente 🔥 (~0,92)** y el segmento cambia a
   *decidido*. **Ese salto es la tesis del proyecto en vivo:** el mismo número de eventos,
   en distinto orden y tipo, da una intención distinta.
2. Mostrar el **desglose por modelo**: baseline, GRU, Transformer y el híbrido. Se ve que
   los tres coinciden y el ensamble los combina.
3. **Dashboard** (`/dashboard.html`): subir un CSV de leads → tabla ordenada por
   probabilidad, con 🔥/🌡️/❄️ y segmento. Es la vista que usaría un jefe de ventas.
4. Cerrar mostrando `results/attention_heatmap.png`: **en qué clic se fijó el Transformer**.

## 8. Preguntas probables y respuestas cortas

| Pregunta | Respuesta |
|---|---|
| **¿Por qué no usaste BERT / fine-tuning?** | BERT está preentrenado en **texto**. Mi dato son secuencias de eventos de un e-commerce: no existe un modelo preentrenado que transferir, y el vocabulario son 2 tokens. Transfer learning aquí no aplica; entrenar desde cero es lo correcto. La *arquitectura* Transformer sí la uso — que es lo que enseña la unidad. |
| **¿Por qué GRU y no LSTM?** | Menos compuertas y más liviana con rendimiento equivalente (lo que dice la clase). Con sesiones de mediana 4 eventos, las 3 compuertas de la LSTM son sobredimensionadas. |
| **¿Por qué la accuracy del baseline es 0,87 y parece buena?** | Espejismo del desbalance (4,3 % positivos). Su precision es 0,24: de cada 4 leads que marca como calientes, 3 no compran. Por eso F1. |
| **¿El modelo no está viendo la compra?** | No, y no es una promesa: hay un test que lo verifica (punto 6.1). El token `transaction` ni siquiera existe en el vocabulario. |
| **¿Cómo sé que la atención "sirve"?** | El heatmap muestra los pesos reales de la última capa, y un test verifica que cada fila de atención suma 1 (es una distribución válida, el softmax hace lo que debe). Que la máscara funciona lo prueba el test de invariancia al padding. |
| **¿Qué mejorarías?** | (a) Sumar la **categoría del ítem** al embedding — quedó fuera de v1 y es donde el Transformer podría despegar, porque ahí sí aparece ambigüedad; (b) sesiones más largas de usuarios recurrentes; (c) probar el umbral contra el margen real por lead en vez de F1. |

## 9. Datos duros para citar

- **Repositorio:** https://github.com/gustavorh/LeadScore
- **Dataset:** RetailRocket (Kaggle) — 166.957 sesiones, 137.124 visitantes, 4,3 % conversión.
- **Split:** 116.747 train / 25.178 val / 25.032 test (por visitante).
- **Arquitecturas:** GRU (embed 32 + delta 16 → hidden 64) · Transformer (d_model 64,
  4 cabezas, 2 capas, FF 128) · stacking logístico + umbral calibrado (0,891) · K-means k=4.
- **Entrenamiento:** CPU, ~6 minutos, seed 42.
- **Infra:** 2 contenedores (FastAPI + nginx) con docker-compose, local y producción.
- **Tests:** 37, incluyendo anti-fuga (`test_no_transaction_token_in_any_sequence`),
  disjunción de splits por visitante (`test_make_splits_grouped_no_visitor_leakage`) e
  invariancia al padding en ambos modelos (`test_padding_invariance`).
