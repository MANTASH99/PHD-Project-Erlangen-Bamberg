# Emotion Classification and Reasoning Correctness: Corpus Overview

## 1 Introduction

Emotion recognition in text (ERT) has become a cornerstone task in affective computing and social NLP, with applications in dialogue systems, mental health monitoring, and social media analysis. Despite remarkable progress in recent years, especially with large pre-trained language models (LLMs) such as BERT and RoBERTa, an important question remains largely unanswered:

**When emotion classifiers make correct predictions, are they correct for the right reasons?**

Modern neural models often achieve high accuracy while relying on superficial lexical patterns, dataset artifacts, or topical biases rather than genuine emotional semantics. This challenge undermines trust in affective AI systems and limits their interpretability and fairness. While explainable AI (XAI) methods—such as SHAP, LIME, and Integrated Gradients—can highlight influential tokens in individual predictions, we still lack a systematic understanding of whether these explanations correspond to valid reasoning processes.

This work introduces a multi-stage framework to bridge that gap. Our goal is to develop a pipeline capable of distinguishing between predictions made for the *right reasons* (semantically grounded in emotional cues) and those made for the *wrong reasons* (driven by spurious or misleading evidence). To achieve this, we:

1. **Assemble and unify multiple large-scale emotion recognition datasets** into a single, harmonized corpus.
2. **Train and fine-tune several emotion classification models**, both general-purpose and emotion-specific.
3. **Apply explainability methods (SHAP)** to interpret model decisions.
4. **Manually annotate a subset of correct predictions** according to whether they were made for valid reasons.
5. **Train meta-models** that automatically predict whether a model’s correct classification was semantically justified.

This framework enables an empirical and scalable investigation of reasoning correctness in emotion recognition. Ultimately, our goal is to automate the detection of *“right-for-right-reasons”* predictions — a critical step toward interpretable and accountable NLP systems.

---

## 2 Corpus Construction

### 2.1 Overview

We constructed a unified emotion corpus by combining a large, publicly available resource — the **Super Emotion Dataset** — with a newly collected proprietary dataset (**Hidden Emotions**). Together, these datasets offer both **breadth** (diverse textual sources and label sets) and **depth** (rich, contextually grounded emotional narratives).

The resulting merged dataset contains approximately **65,000 high-quality samples**, each labeled with one or more emotion categories. The dataset serves as the foundation for model training, explainability analysis, and human annotation in later phases.

---

### 2.2 Public Component: Super Emotion Dataset

The **Super Emotion Dataset (SED)** (Cirimus et al., 2024) aggregates and harmonizes several popular emotion classification corpora. Its goal is to offer a large-scale, multi-domain dataset for benchmarking affective language models. The dataset combines six established sources:

| Source                                  | Domain          | Approx. Size | Example Labels                                        |
| --------------------------------------- | --------------- | ------------ | ----------------------------------------------------- |
| GoEmotions (Demszky et al., 2020)       | Reddit comments | 58k          | joy, anger, sadness, disgust, fear, surprise, neutral |
| MELD (Poria et al., 2019)               | TV dialogues    | 13k          | joy, sadness, anger, disgust, surprise, fear, neutral |
| ISEAR (Scherer & Wallbott, 1994)        | Self-reports    | 7k           | joy, anger, sadness, fear, disgust, shame, guilt      |
| CrowdFlower Emotion                     | Tweets          | 40k          | happiness, sadness, anger, fear, surprise             |
| SemEval Affect (Mohammad et al., 2018)  | Tweets          | 10k          | joy, sadness, anger, fear                             |
| Twitter Emotion Corpus (Mohammad, 2012) | Hashtag-based   | 160k         | joy, sadness, anger, fear, love, surprise             |

Each sub-corpus was preprocessed by the SED creators to standardize columns and labels. The unified schema is:

```
label, text
```

The Super Emotion Dataset provides a rich, heterogeneous source of emotional text ranging from conversational (MELD) to social media (Twitter, Reddit) to self-report narratives (ISEAR). It covers multiple linguistic registers and degrees of emotional explicitness, making it an ideal foundation for evaluating generalization in emotion recognition.

---

### 2.3 Proprietary Component: Hidden Emotions Dataset

The **Hidden Emotions Dataset** is a privately collected corpus created to capture real-world emotional narratives beyond existing benchmarks. It consists of textual responses to open-ended prompts, where participants describe emotionally charged situations and self-report the emotion they experienced.

**Example entries:**

| Label   | Example Text                                                                                                                        |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| anger   | “People get under my skin. For example, when an entitled customer demands to speak to my manager for a simple issue I could solve.” |
| fear    | “I was driving on the highway and someone cut me off and brake-checked me, almost causing a car accident.”                          |
| sadness | “Someone misunderstood me and took it out of context.”                                                                              |

Compared to the Super Emotion dataset, this collection contains longer, more introspective text — often full sentences or short paragraphs — capturing subtler emotional reasoning. The dataset currently includes **~15k samples**, covering:

**anger, boredom, disgust, fear, guilt, joy, pride, relief, sadness, shame, surprise, trust**

This dataset adds multiple rare affective categories, enriching the unified corpus.

---

### 2.4 Label Harmonization

Merging the two datasets required harmonizing differing label spaces. Key steps:

**1. Label normalization:**

* lowercased all labels
* removed punctuation

**2. Synonym merging:**

* happiness → joy
* content → joy
* annoyance → anger
* shame → guilt (in selected cases)

**3. Composite filtering:**

* multi-label entries were either split or removed depending on dominance

**4. Minority exclusion:**

* categories with <10 samples removed for statistical reliability

After harmonization, the final label set included **14 emotion categories**:

**anger, boredom, disgust, fear, guilt, joy, love, pride, relief, sadness, shame, surprise, trust, neutral**

---

### 2.5 Corpus Statistics

| Statistic           | Value                                      |
| ------------------- | ------------------------------------------ |
| Total samples       | 35,421                                     |
| Average text length | 21.3 tokens                                |
| Unique labels       | 14                                         |
| Label distribution  | Long-tailed (joy, sadness, anger dominant) |
| Domains             | Conversational, social media, narrative    |
| Languages           | English                                    |
| Format              | CSV (label, text)                          |

A stratified 80/10/10 train–validation–test split was applied to maintain class balance. Minimal filtering was used to avoid rare-class splitting errors.

---

### 2.6 Data Utility and Motivation

By combining large-scale, publicly available emotion datasets with a proprietary, deeply contextual corpus, we achieve a unique balance between **breadth** (diversity of emotional expressions) and **depth** (richness of reasoning behind emotion). This hybrid design supports our ultimate goal: to not only classify emotions accurately but to study *why* models make those predictions — particularly whether their reasoning aligns with human emotional interpretation.


## 3. Models

This work evaluates three pretrained transformer-based emotion classifiers, each finetuned on standard emotion-recognition corpora.

### 3.1 BERT-base-uncased-emotions

We employ a BERT-base model finetuned for multi-class emotion classification. Evaluated on 5,209 samples, it achieved 4,251 correct predictions (0.8161 accuracy). Its balanced architecture provides strong lexical sensitivity, making it a suitable baseline for explainability comparisons.

### 3.2 RoBERTa-base-emotions

RoBERTa-base, trained with a larger corpus and improved masking strategy, delivered the best performance among the tested models. On the same 5,209-sample evaluation, it achieved 4,270 correct predictions (0.8197). RoBERTa’s contextual embedding stability makes it ideal for SHAP-based interpretability.

### 3.3 DistilBERT-base-emotions

DistilBERT, a compact and computationally efficient variant of BERT, was included to test whether lighter architectures still yield meaningful explanations. It produced 4,182 correct predictions (0.8028), demonstrating a performance–efficiency trade-off.

## 4. SHAP Methodology

*To be expanded.* This section will explain:

* Why SHAP was selected
* How SHAP values were computed over token embeddings
* Aggregation strategies and visualization methods
* How explanations were paired with model predictions for annotation

## 5. Annotation Pipeline (Label Studio)

The annotation interface was implemented in Label Studio, with JSON inputs containing text, embedded SHAP heatmaps, token-level features, and metadata. Annotators evaluated whether the model's explanation correctly highlighted emotionally relevant features.

The core Label Studio configuration:

```xml
<View>
  <Header value="SHAP Explanation Evaluation"/>

  <Image name="shap_image" value="$image" zoom="true" zoomControl="true"/>

  <Header value="Text analyzed:"/>
  <Text name="text" value="$text"/>

  <Header value="Model Information:"/>
  <Table>
    <TableItem name="true_label" value="True Label: $true_label"/>
    <TableItem name="pred_label" value="Predicted: $pred_label"/>
    <TableItem name="confidence" value="Confidence: $confidence"/>
    <TableItem name="model_name" value="Model: $model"/>
    <TableItem name="id" value="ID: $global_id"/>
  </Table>

  <Header value="Is the SHAP explanation highlighting the correct features?"/>
  <Choices name="explanation_quality" toName="shap_image" choice="single-radio" required="true">
    <Choice value="correct_reasons" hint="SHAP correctly highlights relevant words/features"/>
    <Choice value="wrong_reasons" hint="SHAP highlights irrelevant or wrong features"/>
    <Choice value="partially_correct" hint="Mix of correct and incorrect highlights"/>
    <Choice value="unclear" hint="Cannot determine / ambiguous"/>
  </Choices>

  <Header value="Notes (optional):"/>
  <TextArea name="notes" toName="shap_image"
            placeholder="Explain your reasoning, mention specific words/features, or add comments..."
            rows="4"
            maxSubmissions="1"/>
</View>
```

## 6. Meta-classifier

*Placeholder.* Describe how SHAP-derived features were used to train a classifier predicting whether an explanation is correct.

## 7. Results

*To be completed.*

## 8. Discussion

*To be completed.*

## 9. Ethical Considerations

*To be completed.*

## 10. Limitations

*To be completed.*

## 11. References

*To be completed.*
