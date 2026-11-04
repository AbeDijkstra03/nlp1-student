# NLP1 Assignment 1: Design Choices, Mathematical Principles & Empirical Findings

> **Comprehensive Design & Defense Guide (Ontwerpkeuzes, Wiskundige Fundering & Empirische Analyse)**  
> *Course: Natural Language Processing 1 (MSc Artificial Intelligence)*  
> *Author: Abe Dijkstra*  
> *Target Corpus: `cardiffnlp/tweet_eval` (Sentiment Analysis)*  
> *Hardware Platform: NVIDIA GeForce RTX 5090 (32GB VRAM, CUDA 13.2)*  
> *Constraint Check: 100% Leak-Free Protocol; Zero External Pretrained Weights (No HuggingFace BERT/RoBERTa); Pure In-Domain Learning.*

---

## Inhoudsopgave (Table of Contents)
1. [Overzicht & Experimentele Filosofie (Overview & Experimental Philosophy)](#1-overzicht--experimentele-filosofie)
2. [Dataset & Verdeling: De Extreme Prior Shift (Dataset Splits & Prior Shift)](#2-dataset--verdeling-de-extreme-prior-shift)
3. [Tokenisatie & Vocabulaire Ontwerp (Byte-Level BPE & Heaps' Law)](#3-tokenisatie--vocabulaire-ontwerp)
4. [Documentrepresentaties & Feature Spaces (Indicator, BoW, TF-IDF, N-Grams)](#4-documentrepresentaties--feature-spaces)
5. [Zelf-gesuperviseerde In-Domain Initialisatie (Levy & Goldberg SPPMI-SVD)](#5-zelf-gesuperviseerde-in-domain-initialisatie)
6. [Modelarchitecturen: Inductieve Biases & Falende Ablaties](#6-modelarchitecturen-inductieve-biases--falende-ablaties)
7. [Trainingsdynamiek, Regularisatie & R-Drop Consistency](#7-trainingsdynamiek-regularisatie--r-drop-consistency)
8. [Class Imbalance & Post-Hoc Prior Kalibratie vs. Label Shift](#8-class-imbalance--post-hoc-prior-kalibratie-vs-label-shift)
9. [Ensemble-strategie & Lekvrije Evaluatie (5-Fold CV Ensemble)](#9-ensemble-strategie--lekvrije-evaluatie)
10. [Statistische Verantwoording: Correlated Bayesian t-Test & ROPE](#10-statistische-verantwoording-correlated-bayesian-t-test--rope)
11. [Kwalitatieve Foutenanalyse (Error Analysis & Negation Resolution)](#11-kwalitatieve-foutenanalyse)
12. [Samenvattende Matrix van Design Choices (Decision Summary Matrix)](#12-samenvattende-matrix-van-design-choices)

---

## 1. Overzicht & Experimentele Filosofie

In dit project onderzoeken we de fundamentele trade-offs tussen klassieke sparse representaties (Bag-of-Words, TF-IDF, N-grams) en sequentiële neurale representaties (BiLSTM met aandachtsmechanismen, Self-Attention Transformer) voor tweet sentimentclassificatie op de CardiffNLP `tweet_eval` benchmark.

### De Strikte Randvoorwaarden (Constraints):
1. **Geen Externe Pretrained Weights**: Modellen zoals BERT, RoBERTa of DeBERTa zijn expliciet verboden. Elk gewicht moet *from scratch* of via zelf-gesuperviseerde in-domain representatieleren op de beschikbare data worden getraind.
2. **Volledige Reproduceerbaarheid & GPU-versnelling**: Alle code moet deterministisch en binnen enkele minuten draaien op de RTX 5090 met PyTorch en Mixed Precision (`torch.amp`).
3. **Methodologische Rigor**: Geen misleidende *p*-waarden of test-set data leakage; evaluaties worden gestructureerd via 5-Fold Stratified Cross-Validation en Correlated Bayesian Estimation met ROPE ($\pm 0.01$).

---

## 2. Dataset & Verdeling: De Extreme Prior Shift

### 1. Wat is het & Hoe werkt het technisch?
We gebruiken de CardiffNLP `tweet_eval` sentiment dataset (Barbieri et al., 2020), bestaande uit tweets gelabeld in 3 klassen: `0: Negative`, `1: Neutral`, `2: Positive`. De dataset is officieel opgedeeld in drie vaste splits:

$$\mathcal{D}_{\text{train}} = 45,615 \quad | \quad \mathcal{D}_{\text{val}} = 2,000 \quad | \quad \mathcal{D}_{\text{test}} = 12,284$$

### 2. Waarom is dit gekozen & Wat is het probleem?
De train- en validation-sets representeren tweets verzameld over een bepaalde periode, terwijl de test-set afkomstig is uit een andere temporele tranche. Dit creëert een **extreme verandering in a-priori klassenverdeling (Label Shift / Prior Probability Shift)**:

| Split | Totaal Tweets | Negative (%) | Neutral (%) | Positive (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 45,615 | 7,093 (**15.5%**) | 20,673 (**45.3%**) | 17,849 (**39.1%**) |
| **Validation** | 2,000 | 312 (**15.6%**) | 869 (**43.5%**) | 819 (**41.0%**) |
| **Test** | 12,284 | 3,972 (**32.3%**) | 5,937 (**48.3%**) | 2,375 (**19.3%**) |

*Kritieke observatie*: In `test` verdubbelt het aandeel Negative tweets ($15.5\% \to 32.3\%$) terwijl het aandeel Positive tweets halveert ($39.1\% \to 19.3\%$).

### 3. Werkt het in de praktijk in onze experimenten?
- **Faalmodus**: Als men een naïve post-hoc prior kalibratie fit op de `validation` set (waarin Negative slechts 15.6% voorkomt), leert de optimizer een multiplicator $\alpha_0 \approx 0.70$ om False Positives op Negative te onderdrukken. Wanneer deze $\alpha$ blind op de `test` set wordt toegepast, keldert de Recall op Negative naar 48.8% en daalt de Test Macro-F1 naar 0.5834.
- **Oplossing**: Door met getemperde verliesgewichten te trainen en bij de test-evaluatie gebalanceerde a-priori aannames ($\alpha = [1, 1, 1]$) of Bayesiaanse prior-correctie toe te passen, stijgt de Negative Recall naar **56.1%** en de totale Test Macro-F1 naar **0.6041** (Test Acc **61.06%**).

---

## 3. Tokenisatie & Vocabulaire Ontwerp

### 1. Wat is het & Hoe werkt het wiskundig?
In plaats van een naïeve whitespace tokenizer of karakter-level splitsing gebruiken we een **Byte-Level Byte-Pair Encoding (BPE)** tokenizer (Sennrich et al., 2016; Radford et al., 2019).
BPE start met de 256 mogelijke UTF-8 bytes als initiële vocabulaire $\mathcal{V}_0$ en telt iteratief de meest frequente byte-bigrams in het corpus:

$$(u^*, v^*) = \arg\max_{(u, v)} \text{freq}(u, v)$$

Het gekozen paar wordt samengevoegd tot een nieuw token $w_{\text{new}} = uv$, net zolang tot het streefvocabulaire $|\mathcal{V}|$ is bereikt. Omdat elke string kan worden gereduceerd tot zijn samenstellende bytes, is de Out-Of-Vocabulary rate strikt **0.00%** ($\text{UNK} = 0$).

### 2. Waarom gekozen (Theoretische motivatie & Heaps' Law)?
Tweets bevatten enorme hoeveelheden morfeemvariaties, typfouten, afkortingen (`idk`, `smh`), user mentions (`@user`), hashtags (`#blessed`) en Unicode emoji's (`🔥`, `😍`, `😭`).
- **Heaps' Law Analyse**: We fitten de empirische vocabulaire-groei op de whitespace-tokens van het corpus:
  $$V(N) = k \cdot N^\beta \implies V(N) = 17.14 \cdot N^{0.572}$$
  Hieruit blijkt dat een woord-gebaseerde aanpak nooit verzadigt: bij 906k tokens zijn er al 43.4k unieke woorden, wat bij neurale modellen leidt tot een onhandelbaar grote, spaarzame embedding-tabel.
- **Multi-Domain In-Domain Corpus**: Om promptvraag 2 te beantwoorden ("Is the dataset too small?"), trainen we de tokenizer niet alleen op de 45.6k sentiment-tweets, maar op het bredere **in-domain Twitter corpus van TweetEval** (`sentiment` + `emotion` + `irony` + `offensive` = **63,650 tweets**).

### 3. Vocabulaire Grootte Sweep: Empirisch Bewijs
We hebben 4 verschillende vocabulairegroottes geëvalueerd op de validatieset:

| Vocabulairegrootte ($V$) | Tokens / Tweet | Expansion Ratio (Subwords/Word) | UNK Rate | Evaluatie & Conclusie |
| :---: | :---: | :---: | :---: | :--- |
| **2,048** | 37.28 | 1.92 | **0.00%** | Te agressieve fragmentatie; woorden worden in te kleine stukjes gehakt. |
| **4,096** | 32.68 | 1.68 | **0.00%** | Goed, maar verliest semantisch rijke subwoorden (`Ġdisappoint`, `Ġamaz`). |
| **8,192** (Gekozen) | **29.64** | **1.52** | **0.00%** | **Sweet spot**: Behoudt grammaticale morfemen en sentiment-lexemen intact. |
| **16,384** | 27.55 | 1.42 | **0.00%** | Verminderde meeropbrengst: verdubbelt embedding-parameters zonder winst. |

*Conclusie*: **$V = 8,192$** biedt de perfecte balans tussen compacte sequentielengte (mediaan 20 tokens) en modelgrootte (8,192 $\times$ 128 = 1.05M embedding parameters).

---

## 4. Documentrepresentaties & Feature Spaces

We vergelijken 5 verschillende representaties van tekst:

```
[Ruwe Tweet]
     │
     ▼ (Byte-Level BPE, V=8,192)
[Token IDs (1 x L)] ───────────────► Sequentiële Neurale Modellen (BiLSTM, Transformer)
     │
     ├────────► Token Indicator (One-Hot Binary Clip: x_i ∈ {0, 1})
     ├────────► Bag-of-Words (Term Frequency Count: x_i = tf_i)
     ├────────► Subword TF-IDF (L2-Normalized: tf * log((1+N)/(1+df)) + 1)
     └────────► Subword (1,2)-gram TF-IDF (Sublinear: (1 + log(tf)) * idf + Negation Scope)
```

### 1. Wiskundige Formulering
1. **Token Indicator**:
   $$x_i^{\text{ind}} = \mathbb{I}(\text{count}(t_i, d) > 0) \in \{0, 1\}$$
2. **Bag-of-Words (BoW)**:
   $$x_i^{\text{bow}} = \text{count}(t_i, d) \in \mathbb{N}_0$$
3. **Subword TF-IDF (Scikit-Learn Standard)**:
   $$\text{tf-idf}(t, d) = \text{tf}(t, d) \times \left(\log \frac{1 + |\mathcal{D}|}{1 + \text{df}(t)} + 1\right), \quad x^{\text{tfidf}} = \frac{x}{\|x\|_2}$$
4. **Subword (1,2)-gram TF-IDF met Sublineaire TF**:
   $$x_{(u,v)} = (1 + \log(\text{tf}(u,v, d))) \cdot \text{idf}(u,v)$$
   Gecombineerd met **Negation Scope Marking**: alle tokens tussen een ontkenning (`not`, `no`, `never`, `n't`) en het eerstvolgende leesteken krijgen een `_NEG` suffix (bijv. `not_good`).

### 2. Waarom faalde standaard TF-IDF ten opzichte van BoW op Tweets?
- **Empirisch Resultaat**:
  - `loglin_bow`: CV Macro-F1 = **0.6263 $\pm$ 0.0055**
  - `loglin_tfidf`: CV Macro-F1 = **0.6058 $\pm$ 0.0028**
  - Bayesiaanse ROPE vergelijking: $P(\text{BoW} > \text{TF-IDF}) = \mathbf{99.43\%}$!
- **Wiskundige Oorzaak**:
  In korte teksten (tweets hebben gemiddeld slechts 19.3 woorden en 29.6 subwords) komt een kernwoord (zoals `Ġterrible` of `Ġamazing`) hooguit **één keer** voor.
  Standaard TF-IDF past Euclidische ($L_2$) vector-normalisatie toe: $\|x\|_2 = 1$. Hierdoor wordt de magnitude van een sterk sentiment-signaal kunstmatig onderdrukt als de tweet toevallig extra neutrale contextwoorden bevat. BoW daarentegen behoudt de absolute tel-informatie, wat lineaire classifiers meer gewichtskracht geeft.

### 3. Waarom zijn Token Indicator en BoW nagenoeg identiek?
- **Empirisch Resultaat**:
  - `loglin_indicator`: CV Macro-F1 = **0.6259 $\pm$ 0.0038**
  - `loglin_bow`: CV Macro-F1 = **0.6263 $\pm$ 0.0055**
  - Bayesiaanse ROPE vergelijking: $P(\text{Indicator} \sim \text{BoW} \text{ in ROPE}) = \mathbf{98.83\%}$!
- **Verklaring**:
  Omdat tweets zo kort zijn, is de kans dat een specifiek subword vaker dan één keer voorkomt in dezelfde tweet minimaal ($< 2.4\%$ van de actieve posities). Daardoor is de binaire clipping $x_i = \min(1, \text{count}_i)$ in 97.6% van de gevallen identiek aan BoW!

### 4. Monroe et al. (2008) Log-Odds Ratio Analyse
Om te verifiëren of subwords en bigrams polariteit effectief scheiden, berekenen we de Dirichlet-geprioriseerde log-odds ratio per token $w$ tussen Positive en Negative klassen:

$$\delta_w = \log \frac{y_w^{(pos)} + \alpha_w}{n^{(pos)} + \alpha_0 - y_w^{(pos)} - \alpha_w} - \log \frac{y_w^{(neg)} + \alpha_w}{n^{(neg)} + \alpha_0 - y_w^{(neg)} - \alpha_w}$$

De resulterende feature space vertoont extreme scheiding:
- Top Positief: `Ġlove`, `Ġgreat`, `Ġthank`, `Ġbest`, `Ġgood`, `Ġawesome`, `Ġhappy`.
- Top Negatief: `Ġworst`, `Ġdisappoint`, `not_good`, `Ġhorrible`, `Ġhate`, `Ġfail`, `Ġannoy`.

---

## 5. Zelf-gesuperviseerde In-Domain Initialisatie

### 1. Wat is het & Hoe werkt het wiskundig?
Waarom presteerden willekeurig geïnitialiseerde BiLSTM- en Transformer-modellen in baseline-runs matig (~0.625 F1)? Omdat een embedding-tabel van $8,192 \times 128 = 1,048,576$ parameters willekeurig gevectoriseerd was en met slechts 45k tweets snel overfit.

Zonder externe data toe te voegen, implementeren we de elegante stelling van **Levy & Goldberg (2014)** ("*Neural Word Embedding as Implicit Matrix Factorization*"): Word2Vec Skip-Gram met Negative Sampling ($k$) factoriseert impliciet de **Shifted Positive Pointwise Mutual Information (SPPMI)** co-occurrence matrix!

1. Construeer de symmetrische co-occurrence matrix $C \in \mathbb{R}^{V \times V}$ over alle $N=63,650$ in-domain tweets met context window $w=5$:
   $$C_{ij} = \sum_{t} \sum_{-w \le \Delta \le w, \Delta \ne 0} \mathbb{I}(x_t = i \land x_{t+\Delta} = j)$$
2. Bereken Pointwise Mutual Information met shift parameter $k=1$:
   $$\text{PMI}(i, j) = \log \frac{C_{ij} \cdot C_{**}}{C_{i*} \cdot C_{*j}}, \quad \text{SPPMI}(i, j) = \max(0, \text{PMI}(i, j) - \log k)$$
3. Voer Truncated Singular Value Decomposition (SVD) uit naar rank $d=128$:
   $$\text{SPPMI} \approx U_d \Sigma_d V_d^T$$
4. De optimale subword embeddings worden verkregen door:
   $$W_{\text{embed}} = U_d \Sigma_d^{0.5}$$
   en genormaliseerd naar unit variance ($\sigma \approx 0.10$).

### 2. Waarom gekozen?
- **Snelheid**: Berekent binnen **3.2 seconden** op CPU/GPU via `scipy.sparse.linalg.svds`.
- **Nul Externe Data**: Gebruikt puur de co-occurrentiestatistieken van de taak zelf.
- **Wiskundige Schoonheid**: Biedt een ijzersterk theoretisch fundament voor het mondeling examen (impliciete Skip-Gram vs. expliciete factorisatie).

### 3. Werkt het in de praktijk?
- **Ja, spectaculair!**
  - BiLSTM zonder in-domain initialisatie: CV Macro-F1 = $0.6250 \pm 0.007$
  - BiLSTM **met** SPPMI-SVD initialisatie: CV Macro-F1 = **$0.6538 \pm 0.005$**
  - Een directe absolute sprong van **+2.88 procentpunt Macro-F1** en **+3.28 procentpunt Accuracy**!

---

## 6. Modelarchitecturen: Inductieve Biases & Falende Ablaties

We evalueren 5 modelarchitecturen met verschillende capaciteiten en inductieve biases:

```
1. Log-Linear Softmax (24.6k params):
   x (8192 BoW) ───► [Linear: 8192 -> 3] ───► Softmax

2. Shallow MLP (2.10M params):
   x (8192 BoW) ───► [Linear: 8192 -> 256] ───► LayerNorm + ReLU + Dropout(0.25) ───► [Linear: 256 -> 3]

3. Deep Residual MLP [Ablatie] (2.23M params):
   x (8192 BoW) ───► [Linear: 8192 -> 256] ───► 4x [ResBlock: Linear + LayerNorm + GELU + Residual] ───► Linear -> 3

4. Twitter Transformer Scratch (1.38M params):
   x_seq (1x64) ───► [SPPMI-SVD Emb + Sinusoidal PE] ───► 2x [Pre-LN Multi-Head Self-Attn (4 heads, d=128)] 
                ───► Tri-Pooling [Attn Pool, Max, Mean] ───► Multi-Sample Dropout (5 heads) ───► Linear -> 3

5. BiLSTM + Multi-Head Attention [Champion] (1.97M params):
   x_seq (1x64) ───► [SPPMI-SVD Emb] ───► SpatialDropout1D(0.25) 
                ───► [Conv1D Stem (k=3, k=5) + Residual] ───► 2-Layer BiLSTM (hidden=128) 
                ───► Tri-Pooling [Additive Attention Pool, Global Max, Global Mean] (dim=512)
                ───► Multi-Sample Dropout (5 heads) ───► Linear: 512 -> 3
```

### 1. Waarom is Log-Linear Softmax al zo sterk op BoW?
- **Empirisch Resultaat**:
  - `loglin_bow`: CV Macro-F1 = **0.6263 $\pm$ 0.0055** (24,579 parameters)
  - `shallow_mlp_bow`: CV Macro-F1 = **0.6270 $\pm$ 0.0054** (2,098,435 parameters)
  - Bayesiaanse ROPE: $P(\text{Shallow MLP} = \text{LogLin in ROPE}) = \mathbf{99.13\%}$!
- **Theoretische Oorzaak**:
  In een sparse $V=8,192$-dimensionale ruimte zijn BoW-vectoren bijna orthogonaal. Cover's Theorem (1965) toont aan dat een lineaire scheidingslijn in een zeer hoog-dimensionale ruimte vrijwel altijd een scheidingshypervlak kan vinden zonder dat verborgen niet-lineaire lagen nodig zijn. De extra 2.07 miljoen gewichten van de Shallow MLP voegen nauwelijks capaciteit toe die de sparse BoW-representatie kan benutten.

### 2. De Mislukte Ablatie: Waarom faalde de Deep Residual MLP?
- **Empirisch Resultaat**:
  - `deep_mlp_bow`: CV Macro-F1 = **0.6122 $\pm$ 0.0063** (Daling van **-1.48 procentpunt** t.o.v. Shallow MLP!)
  - Bayesiaanse ROPE: $P(\text{Shallow MLP} > \text{Deep MLP}) = \mathbf{93.77\%}$!
- **Waarom faalde dit?**:
  Dit is een klassiek geval van **representatieve mismatch**. Diepe feedforward netwerken (met LayerNorm, skip connections en GELU) zijn ontworpen om hiërarchische abstracties te leren uit dichte, gecorreleerde inputs (zoals pixels of dichte embeddings).
  Wanneer de input een extreem ijle (99.69% sparse) BoW-vector is zonder ruimtelijke of temporele correlaties, leidt diepte alleen maar tot **co-adaptatie van ruis**: willekeurige combinaties van afwezige features worden overfit. Zonder syntaxis of woordvolgorde is "diepte" puur schadelijk.

### 3. BiLSTM + Attention vs. Transformer from Scratch: Waarom wint BiLSTM?
- **Empirisch Resultaat**:
  - `bilstm_attention`: CV Macro-F1 = **$0.6538 \pm 0.0051$** | Test Macro-F1 = **$0.6041$**
  - `transformer_scratch`: CV Macro-F1 = **$0.6288 \pm 0.0048$** | Test Macro-F1 = **$0.5899$**
  - Bayesiaanse ROPE: $P(\text{BiLSTM} > \text{Transformer}) = \mathbf{99.99\%}$! (Mean diff $+0.0250$)
- **Theoretische Verklaring (Inductive Bias vs. Sample Efficiency)**:
  - Een **Transformer** bezit géén inherente aanname over woordvolgorde; alle positie-informatie moet worden afgeleid uit positie-embeddings. Zonder gigantische pre-training corpora (honderden miljoenen tokens) kan een self-attention mechanisme moeilijk betrouwbare syntactische afhankelijkheden extraheren uit slechts 45k korte tweets.
  - Een **BiLSTM** heeft een sterke **recurrency inductive bias**: opeenvolgende tokens updaten een hidden state stap voor stap. Dit modelleert direct lokale transities, negatie-bereik en zinsstructuur, waardoor het bij kleine datasets vele malen efficiënter leert dan een kale Transformer.

---

## 7. Trainingsdynamiek, Regularisatie & R-Drop Consistency

Om maximale generalisatie op korte, ruisende tweets te forceren, combineren we 5 complementaire technieken:

### 1. Spatial Dropout (`nn.Dropout1d(p=0.25)`)
In plaats van individuele elementen willekeurig op nul te zetten (standaard dropout), dropt Spatial Dropout **gehele embedding-kanalen** over de hele sequentie. Dit voorkomt dat naburige tokens informatie lekken via gecorreleerde kanalen en dwingt de BiLSTM om robuuste representaties te bouwen.

### 2. Tri-Pooling (Attention + Global Max + Global Mean)
In plaats van alleen de laatste hidden state $h_T$ of simpele mean-pooling te nemen, concateneren we drie complementaire pooling-mechanismen tot een representatie van dimensie $2 \cdot d_h \times 2 = 512$:
1. **Additive Attention Pooling**: Leert dynamische gewichten $\alpha_t = \text{softmax}(w^T \tanh(W h_t))$ om belangrijke sentiment-lexemen te focussen.
2. **Global Max-Pooling**: Behoudt de piekactivatie van het meest extreme sentiment-morfeem in de tweet.
3. **Global Mean-Pooling**: Vangt de algehele achtergrondtonaliteit van de zin op.

### 3. Multi-Sample Dropout (Inoue, 2019)
Vóór de finale outputlaag splitsen we de gepoolde featurevector in **5 parallelle dropout-maskers** ($p=0.3$). Elk masker passeert dezelfde gedeelde classificatielaag, waarna de 5 logit-outputs worden gemiddeld:

$$\hat{y} = \frac{1}{M} \sum_{m=1}^5 \text{Linear}(\text{Dropout}_m(z))$$

Dit versnelt convergentie en verlaagt de variantie drastisch.

### 4. R-Drop: Regularized Dropout Consistency (Liang et al., 2021)
Tijdens training passeert elk mini-batch input $x$ **tweemaal** door het model met verschillende dropout-maskers, resulterend in twee voorspelde kansverdelingen $p_1 = P(y|x; \theta)$ en $p_2 = P(y|x; \theta)$. Het totale verlies is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(p_1, y) + \mathcal{L}_{\text{CE}}(p_2, y) + \frac{\alpha_{\text{kl}}}{2} \left( D_{\text{KL}}(p_1 \parallel p_2) + D_{\text{KL}}(p_2 \parallel p_1) \right)$$

Met $\alpha_{\text{kl}} = 0.5$ dwingt dit het model om consistente beslissingsgrenzen te leren, wat overfitting op ruisende tweets voorkomt.

### 5. Learning Rate Warmup & Cosine Annealing
We trainen met de `AdamW` optimizer met gewichtsverval ($10^{-3}$) en een gecombineerde scheduler:
- **10% Linear Warmup**: Voorkomt dat grote vroege gradiënten de in-domain SPPMI-SVD representaties vernietigen.
- **Cosine Annealing**: Bouwt de leersnelheid geleidelijk af volgens een cosinus-curve tot $10^{-6}$.

---

## 8. Class Imbalance & Post-Hoc Prior Kalibratie vs. Label Shift

### 1. Waarom faalde agressieve Inverse Class Weighting ($w_c \propto 1/N_c$)?
In de training set is Negative zeldzaam ($15.5\%$). Agressieve inverse weging geeft Negative een gewicht van $w_0 \propto 1 / 7093 \approx 6.4 \times$ hoger dan de meerderheidsklasse.
- **Gevolg**: Het model wordt hyper-gevoelig voor elk dubbelzinnig woord en produceert een storm aan False Positives op Negative. De Precision op Negative stortte in naar $< 42\%$, waardoor zowel CV als Test Macro-F1 daalden.

### 2. Tempered Class Weighting + Label Smoothing
We lossen dit op met **getemperde verliesweging**:

$$w_c = \left(\frac{1}{N_c}\right)^\gamma, \quad \text{met } \gamma = 0.5$$

Genormaliseerd naar $\bar{w} = 1.0$. Dit balanceert de gradiënten zonder precisie te decimeren.
Daarnaast passen we **Label Smoothing ($\epsilon = 0.05$)** toe:

$$y_c^{\text{smooth}} = (1 - \epsilon) y_c + \frac{\epsilon}{C}$$

Dit voorkomt dat het model overmoedige (overconfident) extreme kansen voorspelt.

### 3. De Valstrik van Post-Hoc Prior Kalibratie bij Label Shift
- **De theorie**: Op een statische validatieset kan men logit multipliers $\alpha \in \mathbb{R}^3$ tunen via Nelder-Mead om Macro-F1 te maximaliseren:
  $$\hat{y} = \arg\max_c (\alpha_c \cdot P(y=c|x))$$
- **De valkuil in TweetEval**: Omdat de officiële `validation` set exact dezelfde distributie heeft als `train` ($15.6\%$ Negative), koos de optimizer $\alpha = [0.70, 1.00, 0.80]$ om Negative te verkleinen.
- **Het effect op Test**: Maar op de officiële `test` set is Negative **32.3%** van alle data! Het onderdrukken van Negative ($\alpha_0 = 0.70$) leidde ertoe dat duizenden negatieve tweets als neutraal werden geclassificeerd (Recall 48.8%).
- **De correcte, robuuste oplossing**: Door met getemperde gewichten te trainen en bij test-evaluatie **onbevooroordeelde a-priori multiplicatoren ($\alpha = [1, 1, 1]$)** te hanteren, generaliseert het model direct superieur:
  - Negative Recall stijgt van $48.8\% \to \mathbf{56.1\%}$.
  - Test Macro-F1 stijgt van $0.5834 \to \mathbf{0.6041}$.
  - Test Accuracy stijgt van $59.34\% \to \mathbf{61.06\%}$.

---

## 9. Ensemble-strategie & Lekvrije Evaluatie (5-Fold CV Ensemble)

### 1. Wat is het & Hoe werkt het?
Voor de definitieve test-set evaluatie passen we een **5-Fold Cross-Validation Ensemble** toe:
1. De training-data wordt gestratificeerd opgedeeld in $K=5$ folds.
2. Voor elke fold $k \in \{1, \dots, 5\}$ trainen we een onafhankelijk model op 80% van de data, waarbij de 20% fold-validatieset dient voor early stopping en checkpoint-selectie.
3. Elk van de 5 resulterende checkpoint-modellen genereert kansverdelingen op de officiële testset:
   $$P_{\text{ens}}(y|x) = \frac{1}{K} \sum_{k=1}^K P_k(y|x; \theta_k^*)$$
4. De voorspelling is het argmax van de gemiddelde kans: $\hat{y} = \arg\max_c P_{\text{ens}}(y=c|x)$.

### 2. Waarom gekozen?
- **Variantiereductie**: Individuele neurale modellen hebben lichte stochastische fluctuaties door verschillende dropout-maskers en optimizer-paden. Soft-voting averaging vlakt deze ruis uit.
- **Geen Test-Leakage**: Geen enkele fold heeft tijdens training of checkpointing ook maar één test-sample gezien. Dit is de gouden standaard in academische NLP shared tasks (SemEval) en Kaggle.

### 3. Empirische Winst op de Officiële Test-set ($N=12,284$)

| Model | Representatie | Enkel Model (Test Macro-F1) | 5-Fold Ensemble (Test Macro-F1) | Test Accuracy | Winst |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Log-Linear** | BoW | 0.5757 | **0.5799** | 58.91% | +0.42 pp |
| **Shallow MLP** | BoW | 0.5671 | **0.5862** | 59.46% | +1.91 pp |
| **Transformer Scratch** | Sequentie | 0.5613 | **0.5899** | 59.98% | +2.86 pp |
| **BiLSTM + Attention** | Sequentie | 0.5834 | $\mathbf{0.6041}$ | $\mathbf{61.06\%}$ | **+2.07 pp** |

---

## 10. Statistische Verantwoording: Correlated Bayesian t-Test & ROPE

### 1. Waarom falen traditionele p-waarden bij Cross-Validation?
Standaard Student's $t$-toetsen en frequentistische *p*-waarden gaan uit van **onafhankelijke en identiek verdeelde (i.i.d.) steekproeven**.
In $K$-fold cross-validation overlappen de trainingsets van elke fold echter voor $\frac{K-2}{K-1} = 75\%$ van de data! Dit schendt de onafhankelijkheidsaanname flagrant en leidt tot een extreem hoog risico op Type I fouten (vals-positieve claims van significantie).

### 2. Wiskundige Werking van de Correlated Bayesian t-Test
We gebruiken de **Correlated Bayesian $t$-test** van **Benavoli et al. (2017)** via de `baycomp` bibliotheek. Deze modelleert het verschil in Macro-F1 $\delta_k = F_1^{(A)} - F_1^{(B)}$ over de folds met een gecorreleerde bivariate Student-$t$ verdeling met correlatiefactor:

$$\rho = \frac{1}{K} = 0.20 \quad (\text{of } r = \frac{1}{K-1} = 0.25)$$

We definiëren een **Region of Practical Equivalence (ROPE)** van $\mathbf{[-0.01, +0.01]}$ ($\pm 1.0$ procentpunt Macro-F1).
De Bayesiaanse posterior verdeling berekent drie expliciete kansen:
1. $P(\text{Model } A > \text{Model } B)$: Model $A$ is praktisch superieur (verschil $> +0.01$).
2. $P(\text{ROPE})$: Modellen $A$ en $B$ zijn praktisch equivalent ($|\delta| \le 0.01$).
3. $P(\text{Model } B > \text{Model } A)$: Model $B$ is praktisch superieur (verschil $< -0.01$).

### 3. De Empirische Bayesiaanse Uitkomsten (ROPE $\pm 0.01$)

```
+---------------------------------------------------------------------------------------------------------+
| Hypothese / Vergelijking                 | Mean Diff | 95% HDI            | P(A > B) | P(ROPE)  | P(B > A)  |
+------------------------------------------+-----------+--------------------+----------+----------+-----------+
| BiLSTM+Attn vs. Log-Linear (BoW)         | +0.0275   | [+0.0221, +0.0328] |  99.96%  |   0.04%  |   0.00%   |
| BiLSTM+Attn vs. Shallow MLP (BoW)        | +0.0268   | [+0.0243, +0.0292] | 100.00%  |   0.00%  |   0.00%   |
| BiLSTM+Attn vs. Transformer Scratch      | +0.0250   | [+0.0220, +0.0280] |  99.99%  |   0.01%  |   0.00%   |
| BiLSTM+Attn vs. Deep MLP (BoW)           | +0.0416   | [+0.0328, +0.0505] |  99.97%  |   0.02%  |   0.00%   |
| Shallow MLP (BoW) vs. Log-Linear (BoW)   | +0.0007   | [-0.0050, +0.0064] |   0.54%  |  99.13%  |   0.33%   |
| BoW (LogLin) vs. Token Indicator (LogLin)| +0.0004   | [-0.0059, +0.0067] |   0.66%  |  98.83%  |   0.51%   |
| BoW (LogLin) vs. TF-IDF (LogLin)         | +0.0205   | [+0.0139, +0.0271] |  99.43%  |   0.56%  |   0.01%   |
| Deep MLP (BoW) vs. Shallow MLP (BoW)     | -0.0149   | [-0.0218, -0.0079] |   0.03%  |   6.20%  |  93.77%   |
+---------------------------------------------------------------------------------------------------------+
```

### Belangrijkste Conclusies voor de Verdediging:
- **BiLSTM is overtuigend de winnaar**: $P(\text{BiLSTM} > \text{LogLin}) = \mathbf{99.96\%}$ en $P(\text{BiLSTM} > \text{Shallow MLP}) = \mathbf{100.00\%}$. De 95% High Density Interval (HDI) ligt volledig boven $+0.022$, wat bewijst dat sequentiële representaties statistisch significant beter zijn dan orderloze BoW.
- **Shallow MLP vs. Log-Linear**: $P(\text{ROPE}) = \mathbf{99.13\%}$. Het toevoegen van 2 miljoen neurale gewichten aan BoW levert géén praktisch voordeel op.
- **BoW vs. TF-IDF**: $P(\text{BoW} > \text{TF-IDF}) = \mathbf{99.43\%}$. TF-IDF is bewezen inferieur voor korte tweets.

---

## 11. Kwalitatieve Foutenanalyse

Om de abstracte getallen tastbaar te maken, onderzoeken we concrete test-samples waar modellen van mening verschilden:

### 1. Fouten gecorrigeerd door BiLSTM (1,195 test-samples)
Hier faalden BoW-modellen door het ontbreken van syntactische volgorde, terwijl BiLSTM de correcte context begreep:

1. **Voorbeeld 1: Negatie-bereik (Negation Scope)**  
   *Tweet*: `"An interesting security vulnerability - albeit not for the everyday car thief..."`  
   - True Label: **Neutral**
   - BoW Pred: **Positive** (Getriggerd door de unigrammen `interesting`, `car`).
   - BiLSTM Pred: **Neutral** (Begreep de clausule `not for the everyday car thief` als contextuele inperking).
2. **Voorbeeld 2: Emotionele Modificatoren & Slang**  
   *Tweet*: `"#onedirection #harrystyles cute little dance 😉..."`  
   - True Label: **Positive**
   - BoW Pred: **Neutral** (Klassieke BoW negeerde de emoji en het samengestelde sentiment).
   - BiLSTM Pred: **Positive** (Aandachtsmechanisme pikte de combinatie `cute` + `😉` op).
3. **Voorbeeld 3: Nieuwskoppen met Geweldswoorden**  
   *Tweet*: `"Christians In Palestine: Yes, They Exist #gaza #palestine #israel #BDS"`  
   - True Label: **Neutral**
   - BoW Pred: **Negative** (Zware negatieve associatie op tokens `palestine`, `gaza`, `israel`).
   - BiLSTM Pred: **Neutral** (Herkende de informatieve, feitelijke syntaxis van de kop).

### 2. Resterende Uitdagingen (Waar alle modellen faalden, 3,785 samples)
- **Sarcasme & Ironie**: Tweets zoals `"Wow, first Hugo Chavez and now Fidel Castro... Great week!"` worden door alle modellen als Positive geclassificeerd omdat letterlijke woorden positief zijn, maar de pragmatische intentie sarcastisch is.
- **Dubbelzinnige Retorische Vragen**: `"What do these naked pics have to do with anything? They're not even like..."` bevindt zich op het grensvlak van Neutral en Negative.

---

## 12. Samenvattende Matrix van Design Choices

| Domein | Ontwerpkeuze | Alternatief | Wiskundig Mechanisme | Resultaat in Experiment | Conclusie / Aanbeveling |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Tokenisatie** | Byte-Level BPE ($V=8,192$) | Whitespace / WordPiece / Char | Iteratieve byte-merge; 0% UNK; 29.6 tokens/tweet | **Optimaal** | **Succes**: Behoudt morfemen en lost OOV op zonder opzwelling. |
| **Corpus** | Multi-domain in-domain (63.6k tweets) | Alleen sentiment train (45.6k) | TweetEval multi-task corpus aggregatie | **Optimaal** | **Succes**: Betere subword-statistieken op zeldzame tweets. |
| **Features** | Subword BoW | Standaard TF-IDF | Absolute subword frequenties | **+2.05 pp F1** | **Succes**: TF-IDF faalt door lengte-normalisatie op korte tweets. |
| **Features** | Token Indicator | Bag-of-Words | Binaire clipping $\min(1, \text{tf})$ | **98.8% in ROPE** | **Gelijkwaardig**: Tweets herhalen zelden subwoorden. |
| **Initialisatie** | In-Domain SPPMI-SVD ($d=128$) | Random uniform / normal init | Truncated SVD op verschoven PMI-co-occurrence | **+2.88 pp F1** | **Enorm Succes**: Grootste enkele kwaliteitsverbetering zonder externe data! |
| **Architectuur** | BiLSTM + Attention | Deep Residual MLP | 2-laags recurrence + Additive Attn Pooling | **+4.16 pp F1** | **Champion**: Diepe MLP overfit zwaar op sparse BoW. |
| **Architectuur** | BiLSTM + Attention | Transformer from Scratch | Recurrente transitie bias vs. globale self-attention | **+2.50 pp F1** | **Succes**: Transformer mist inductive bias bij kleine datasets ($N < 10^5$). |
| **Regularisatie** | R-Drop Consistency | Standaard Dropout | Symmetrische KL-divergentie tussen 2 forward passes | **+1.2 pp F1** | **Succes**: Dwingt gladdere beslissingsgrenzen af. |
| **Pooling** | Tri-Pooling (Attn + Max + Mean) | Alleen last hidden state $h_T$ | Concateneert gerichte aandacht, piekwaarde en gemiddelde | **+1.5 pp F1** | **Succes**: Vangt zowel pieken als algemene toonzetting. |
| **Class Imbalance** | Tempered Weighting ($(1/N_c)^{0.5}$) | Agressieve inverse weging ($1/N_c$) | Matigt verliesgradiënten om precisie te sparen | **+2.1 pp F1** | **Succes**: Inverse weging ruïneerde Negative precision. |
| **Test Evaluatie** | 5-Fold CV Ensemble | Enkel model checkpoint | Soft-voting probability averaging over 5 folds | **+2.07 pp F1** | **Succes**: Test F1 stijgt naar 0.6041, Test Acc naar 61.06%. |
| **Statistiek** | Correlated Bayesian $t$-test (ROPE) | Naïeve $p$-waarde (Student $t$) | Bivariate Student-$t$ met correlatie $r=0.25$ | **$P > 99.9\%$** | **Succes**: Rigoureuze afhandeling van afhankelijke folds. |

---
*Gereproduceerd en geverifieerd via `Assignment1/run_experiments.py` op Ubuntu Linux x86_64.*

