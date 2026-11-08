# Simple Oral Q&A Defense Guide — NLP1 Project 1 (Lecture 2a)

> **Hoe het mondeling werkt:** Je geeft **geen** presentatie. De TA leest eerst jullie **6 slides** (`slides_final/presentation_6slides.pdf`) om te zien welke dataset en extra experimenten jullie hebben gedaan, en stelt daarna gerichte vragen over jullie keuzes, resultaten en de theorie uit **Lecture 2a (*Text Classification*)**.

---

## 1. Overzicht van Alle Experimenten & Cijfers (Spiekbriefje)

| Onderdeel | Tutorial Baseline (`project.ipynb`) | Onze Extra Experimenten | Belangrijkste Cijfers (5-Fold CV & Test) |
| :--- | :--- | :--- | :--- |
| **1. Dataset (`tweet_eval` sentiment)** | Train ($45,615$), Val ($2,000$), Test ($12,284$) | — | **Prior Shift**: Negative verdubbelt van **15.5%** (Train) naar **32.3%** (Test); Positive halveert van **39.2%** naar **19.3%**. |
| **2. Tokenizers** (vast: `BoW` + `Log-Linear`) | **Whitespace** ($V=8,192$) | **NLTK `TweetTokenizer`** ($V=8,192$) & **Byte-Level BPE** ($V \in \{2\text{k}, 4\text{k}, 8\text{k}, 16\text{k}\}$) | • Whitespace: **0.6091 F1** (63.42% Acc, 6.4% OOV)<br>• `TweetTokenizer`: **0.6248 F1** (64.95% Acc, 4.1% OOV)<br>• BPE $V=2,048$: **0.6142 F1** (63.95% Acc, 0% OOV)<br>• BPE $V=4,096$: **0.6221 F1** (64.68% Acc, 0% OOV)<br>• **BPE $V=8,192$**: **0.6263 F1** (**65.08% Acc**, 0% OOV — **Optimaal**)<br>• BPE $V=16,384$: **0.6258 F1** (65.01% Acc, $2\times$ params) |
| **3. Representaties** (vast: `BPE V=8,192` + `Log-Linear`) | **Token Indicators** ($\{0,1\}$) & **Bag-of-Words** (counts) | **Unigram TF-IDF** ($\text{TF}_{\text{tweet}} \times \text{IDF}_{\text{dataset}}$, $L_2$-genormaliseerd per tweet) | • Token Indicators: **0.6259 CV F1** (65.10% Acc) $\mid$ **0.5791 Test F1** (58.85% Acc)<br>• Bag-of-Words: **0.6263 CV F1** (65.08% Acc) $\mid$ **0.5799 Test F1** (58.91% Acc)<br>• TF-IDF: **0.6058 CV F1** (62.89% Acc) $\mid$ **0.5584 Test F1** (56.72% Acc) |
| **4. Modellen** (vast: `BPE V=8,192` + `Bag-of-Words`) | **Log-Linear** ($24,579$ params) & **Shallow MLP** 1-laag ($2,098,179$ params) | **Deep MLP** 2-lagen `8192 -> 256 -> 128 -> 3` ($2,130,691$ params) | • Log-Linear: **0.6263 CV F1** (65.08% Acc) $\mid$ **0.5799 Test F1** (58.91% Acc)<br>• Shallow MLP: **0.6270 CV F1** (65.13% Acc) $\mid$ **0.5862 Test F1** (59.46% Acc)<br>• Deep MLP: **0.6122 CV F1** (63.85% Acc) $\mid$ **0.5695 Test F1** (57.88% Acc) |
| **5. Bayesian ROPE** ($\rho=0.20, \text{ROPE}=\pm 0.01$) | Correlated Bayesian $t$-test op 5-Fold Stratified CV | Volledig $3 \times 3$ Grid (Representaties $\times$ Modellen) | • `BoW` $\approx$ `Indicators`: **$P(\text{ROPE}) = 98.8\%$** (Log-Lin) / **$99.6\%$** (MLP)<br>• `TF-IDF` $<$ `BoW`: **$P(\text{Drop} > 0.01) = 99.4\%$**<br>• `Shallow MLP` $\approx$ `Log-Linear`: **$P(\text{ROPE}) = 99.1\%$**<br>• `Deep MLP` $<$ `Shallow MLP`: **$P(\text{Drop} > 0.01) = 97.7\%$** |

---

## 2. Korte Antwoorden per Slide (30-Seconden Antwoorden voor het Mondeling)

### Slide 1: Dataset, Class Imbalance & Evaluatiemetrieken
- **Vraag 1: Waarom gebruiken jullie naast Accuracy ook Macro-F1 én een gewogen loss-functie?**
  - **Antwoord:** In de trainingsset ($45,615$ tweets) is **Negative** een kleine minderheidsklasse (**15.5%**), terwijl **Negative** in de testset ($12,284$ tweets) plotseling verdubbelt naar **32.3%** (*Temporal Prior Shift*).
  - Als je alleen naar **Accuracy** kijkt tijdens training, negeert het model de Negative klasse omdat die maar 15.5% van de punten oplevert. **Macro-F1** telt de $F_1$-score van alle 3 de klassen even zwaar mee ($\frac{1}{3} F_{1,\text{Neg}} + \frac{1}{3} F_{1,\text{Neu}} + \frac{1}{3} F_{1,\text{Pos}}$). Daarnaast geven we in de cross-entropy loss een hoger gewicht aan Negative ($w_{\text{Neg}} = 2.14$), zodat het model de minderheidsklasse niet negeert.

---

### Slide 2: Tokenizers & Waarom BPE $V = 8,192$ Wordt Gekozen
- **Vraag 2: Waarom hebben álle BPE vocabulary sizes ($2048, 4096, 8192, 16384$) precies 0.0% Out-of-Vocabulary (OOV) op de testset?**
  - **Antwoord:** Omdat **Byte-Level BPE** zijn basis-vocabulaire begint met alle **256 individuele UTF-8 bytes/karakters**. Elk onbekend woord, emoji of spelfout in de testset kan daardoor altijd worden opgebroken in kleinere subwords of losse letters zonder ooit een `<unk>` token te geven. Woord-tokenizers zoals **Whitespace** (6.4% OOV) en **`TweetTokenizer`** (4.1% OOV) kunnen dat niet.
- **Vraag 3: Als ze allemaal 0% OOV hebben, waarom kiezen jullie dan specifiek $V = 8,192$?**
  - **Antwoord:** De vocab-grootte $V$ bepaalt **hoe fijn of grof woorden in stukjes worden gehakt**:
    1. Bij **$V = 2,048$** (**0.6142 F1**) worden woorden als `"Unbelievable disappointment"` in kleine 3-lettergrepen gehakt (`'bel'`, `'app'`). Omdat Bag-of-Words geen volgorde kent, botst `'app'` van `"disappointment"` (Negatief) met `'app'` van `"apple"` (Neutraal)!
    2. Bij **$V = 8,192$** (**0.6263 F1**) blijven complete sentiment-stammen intact (`'believable'`, `'_disappointment'`, `'fail'`), terwijl voorvoegsels zoals `'_Un'` netjes worden losgekoppeld.
    3. Bij **$V = 16,384$** (**0.6258 F1**, $P(\text{ROPE}) = 100\%$) wordt de F1-score niet hoger, maar verdubbelt wel het aantal parameters van het model ($2\times$). Daarom is $V = 8,192$ de optimale keuze.

---

### Slide 3: Document Representaties (Token Indicators, Bag-of-Words & TF-IDF)
- **Vraag 4: Waarop is jullie TF-IDF precies berekend? (Zijn het n-grams of tokens? Gedeeld door de zin of door de hele dataset?)**
  - **Antwoord:** Onze TF-IDF is berekend op **unigram BPE tokens** ($V = 8,192$ losse tokens, geen multi-woord $n$-grams) in 3 stappen:
    1. **TF (per losse tweet)**: Hoe vaak token $i$ voorkomt in **die ene tweet $x$** (exact gelijk aan Bag-of-Words).
    2. **IDF (over de hele trainingsdataset)**: Gedeeld door het aantal tweets $\text{df}(i)$ waarin token $i$ voorkomt over **alle $45,615$ trainingstweets**: $\text{idf}(i) = \log\left(\frac{1 + 45,615}{1 + \text{df}(i)}\right) + 1$. Zeldzame woorden zoals `"win"` ($\text{df}=920$, $\text{IDF}=4.90$) krijgen daardoor een veel hoger gewicht dan `"the"` ($\text{df}=18,420$, $\text{IDF}=1.91$).
    3. **$L_2$-normalisatie (gedeeld door de lengte van de losse tweet)**: Tot slot delen we de vector van de tweet door zijn eigen Euclidische lengte $\|v(x)\|_2$.
- **Vraag 5: Waarom presteren Token Indicators en Bag-of-Words vrijwel exact hetzelfde ($P(\text{ROPE}) = 98.8\%$)?**
  - **Antwoord:** Een tweet is gemiddeld maar ~19 woorden lang. Uit onze data-analyse blijkt dat **97.6% van de aanwezige tokens in een tweet maar 1 keer voorkomt** (slechts 2.4% wordt herhaald). Voor 97.6% van de woorden is tellen (`1`) dus exact hetzelfde als aan-/afwezigheid (`1`).
- **Vraag 6: Waarom presteert TF-IDF slechter dan gewone Bag-of-Words ($0.6058$ vs. $0.6263$ F1, $P(\text{Drop}) = 99.4\%$)?**
  - **Antwoord:** Door stap 3: het delen door de **$L_2$-norm van de losse tweet**. Als een tweet van 20 woorden één sterk sentiment-woord bevat (`"great"`) en 19 neutrale opvulwoorden, deelt de $L_2$-normalisatie het gewicht van `"great"` door de lengte van alle 20 woorden samen (waardoor de waarde van `"great"` krimpt van $0.83$ naar $0.32$).

---

### Slide 4: Modellen & Waarom de 2-Layer Deep MLP Nauwelijks Meer Parameters Heeft
- **Vraag 7: Waarom heeft de Two-Hidden-Layer MLP (`Deep MLP`, $2.13\text{M}$ params) maar $1.5\%$ meer parameters dan de Single-Hidden-Layer MLP (`Shallow MLP`, $2.10\text{M}$ params)?**
  - **Antwoord:** Zoals te zien in ons netwerkdiagram op Slide 4 zitten bijna alle parameters in **Laag 1 (`8,192 -> 256`)**:
    - De input-vector heeft dimensie $8,192$. Om $8,192$ inputs te verbinden met $256$ neuronen in Laag 1 heb je $8,192 \times 256 + 256 = \mathbf{2,097,408}$ parameters nodig (**99.96%** van het hele Shallow MLP netwerk!).
    - In Laag 1 is de vector dus al gecomprimeerd van $8,192$ naar $256$.
    - De extra tweede laag in `Deep MLP` (**Laag 2: `256 -> 128`**) verbindt alleen nog maar $256$ neuronen met $128$ neuronen: $256 \times 128 + 128 = \mathbf{32,896}$ parameters (slechts **$+1.54\%$** extra!).

---

### Slide 5: 5-Fold Cross-Validation & Correlated Bayesian ROPE
- **Vraag 8: Hoe werkt de Correlated Bayesian $t$-test en wat betekent ROPE?**
  - **Antwoord:** Bij 5-Fold Cross-Validation overlappen de trainingssets van elke fold voor $80\%$ ($\rho = 1/K = 0.20$). Een gewone $t$-test onderschat daardoor de variantie en geeft valse significantie. De **Correlated Bayesian $t$-test** ( uit `project.ipynb`) corrigeert de variantie met factor $\left(\frac{1}{K} + \frac{\rho}{1-\rho}\right)$ en berekent de kans dat het echte verschil $\Delta$ binnen de **ROPE** (*Region of Practical Equivalence*, $[-0.01, +0.01]$ F1) valt.
- **Vraag 9: Waarom wint `Shallow MLP` ($2.10\text{M}$ params) niet van `Log-Linear` ($24.6\text{k}$ params) ($P(\text{ROPE}) = 99.1\%$), en waarom gaat `Deep MLP` zelfs achteruit ($P(\text{Drop}) = 97.7\%$)?**
  - **Antwoord:**
    1. **`Shallow MLP` $\approx$ `Log-Linear`**: In een hoog-dimensionale ruimte ($D = 8,192$) zijn unigram features al lineair scheidbaar. Een niet-lineaire laag kan de ontbrekende woordvolgorde niet terugtoveren uit een Bag-of-Words vector.
    2. **`Deep MLP` $<$ `Shallow MLP`**: Dit bevestigt **Lecture 2a Slide 35** (*"start simple"*): op een $99.71\%$ sparse vector zorgt een extra verborgen laag ervoor dat het model overfit op toevallige woordcombinaties in de training-folds.

---

### Slide 6: Test Set ($N = 12,284$) & Lecture 2a Foutenanalyse
- **Vraag 10: Waarom dalen alle modellen met ~0.04 F1 op de testset ten opzichte van CV, en welke fouten maken ze?**
  - **Antwoord:** De daling van ~0.04 F1 komt doordat de testset uit een latere tijdsperiode komt met een zware **Prior Shift** (Negative verdubbelt van 15.5% naar 32.3%). Belangrijk is dat **alle rangordes uit onze CV exact hetzelfde blijven op de testset** (`Indicators` $\approx$ `BoW` $>$ `TF-IDF` en `Shallow MLP` $\approx$ `Log-Linear` $>$ `Deep MLP`).
  - Onze 3 foutvoorbeelden op de testset laten precies de 3 uitdagingen uit **Lecture 2a** zien:
    1. **Negation Scope (*Uninformative Representations*)**: In `"An interesting security vulnerability ... isn't so bad"` ziet Bag-of-Words niet dat `"isn't"` bij `"bad"` hoort omdat woordvolgorde ontbreekt.
    2. **Topic Bias (*Spurious Correlation*)**: Neutrale nieuws-tweets met `#Palestine` of `#Israel` worden als Negatief voorspeld omdat die hashtags in de trainingsset bijna altijd in negatieve tweets voorkwamen.
    3. **Sarcasme (*Arbitrariness of Argmax*)**: In `"Wow... You always pick the right heroes"` voorspellen beide modellen met $>78\%$ zekerheid Positief op basis van de letterlijke woorden `"Wow"` en `"heroes"`, terwijl de tweet sarcastisch (Negatief) bedoeld is.
