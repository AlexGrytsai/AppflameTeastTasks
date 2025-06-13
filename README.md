# Test Task Solution

## 1. System for Classifying User-Generated Text

**Task:**  
Classify user-generated text into three categories: Safe, Offensive, Needs Human Review.

**Architecture:**  
Diagram: [`Design a system that classifies user-generated text.drawio`](./Design%20a%20system%20that%20classifies%20user-generated%20text.drawio)

**Component Description:**
- **Message Input** — incoming user message.
- **General Analysis** — basic assessment (sentiment, emotions) via third-party services (AWS Comprehend, GCP Natural Language, Azure Text Analytics).
- **Third-party Toxicity API** — specialized services (e.g., Perspective API) for toxicity, aggression, threat assessment.
- **LLM Analysis** — analysis using LLM (OpenAI, Anthropic, etc.) by multiple criteria (different prompts).
- **External Resource Analysis** — checking links to external resources (Web Risk API, VirusTotal, IPQualityScore, URLScan.io).
- **Aggregation & Decision** — aggregating results, making a decision (Safe/Offensive/Needs Human Review).
- **Manual Review** — manual moderation for ambiguous/uncertain cases.
- **Message Broker** — ensures fault tolerance and scalability (Kafka, RabbitMQ, NATS).

**Tools and Models:**
- Perspective API (Google Jigsaw) — for toxicity.
- OpenAI/Anthropic LLM — for complex cases and custom criteria.
- Web Risk API, VirusTotal, IPQualityScore — for link analysis.
- Custom ML module — for specific tasks.

**Handling edge cases and low-confidence:**
- If the result is uncertain (low probability or conflicting between services) — send for manual moderation.
- Fallback between services: if the main API is unavailable or the result is "uncertain", use a backup.

**Evaluation and improvement:**
- Metrics: accuracy, recall, F1-score, share of manual moderation.
- Regular error analysis, model retraining, adding new criteria.

---

## 2. Auto-Reply Bot for Headspace Support

**Task:**  
Automate responses to frequent user requests using LLM.

**Selected intents and prompts:**  
See [`Task_2.md`](./Task_2.md)

- **Intent 1:** Subscription cancellation  
  Prompt: detailed cancellation instructions for iOS, Android, Web, offer alternatives.
- **Intent 2:** Login issues  
  Prompt: clarifying questions, simple steps to restore access, no technical terms.

**Response quality assessment:**
- Check for completeness, correctness, absence of hallucinations.
- Clarity, friendliness, relevance of steps.
- Testing on real/synthetic examples.

**Examples of generated responses:**  
Contained in [`Task_2.md`](./Task_2.md)

---

## 3. NSFW Content Moderation (Images & Videos)

**Task:**  
Evaluate third-party solutions for image/video moderation.

**Comparison:**  
See [`NSFW_Content_Moderation.md`](./NSFW_Content_Moderation.md)

| Solution                  | Accuracy | Latency         | Cost                | Integration      |
|---------------------------|----------|-----------------|---------------------|------------------|
| API4AI NSFW Recognition   | 92–95%   | 300–600 ms      | $7.50/10k req       | Medium (REST)    |
| Google Cloud Vision       | 93–96%   | 500–800 ms      | $1.50/1k req        | High (REST/SDK)  |

**Recommendation:**  
API4AI — optimal balance of price/quality/speed.  
Google Cloud Vision — backup/enterprise.

**Fallback:**  
If the main API is unavailable or the result is "uncertain", use a backup service, then manual moderation.

---

## 4. Toxic Comment Classification (ML)

**Task:**  
Train an ML model to classify text as toxic/non-toxic.

**Source code and pipeline:**
- [`/train_ml_model`](./train_ml_model) — data preparation, NLTK resources, model training.
- [`main.py`](./main.py) — training and evaluation launch.
- [`train_simple_ML_model.md`](./train_simple_ML_model.md) — brief description of results.

**Models:**
- Logistic Regression (best result, F1-score 0.75 for toxic).
- Multinomial Naive Bayes (F1-score 0.28 for toxic).

**Best preprocessing steps:**
- Stopword removal, lemmatization, TF-IDF vectorization.

**How to improve:**
- Class balancing, transformer integration (BERT, RoBERTa), increasing recall for toxic.

---

## Launch

1. Install dependencies:
   ```
   poetry install
   ```
2. Activate virtual environment (optional):
   ```
   poetry shell
   ```
3. Start training:
   ```
   poetry run python main.py
   ``` 