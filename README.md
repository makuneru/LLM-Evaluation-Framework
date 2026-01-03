# LLM Evaluation Framework

A simple framework for evaluating Large Language Model (LLM) responses using DeepEval.

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Set Up API Key
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

> 💡 Get your API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 3. Run Tests
```bash
# Run all examples
python3 -m pytest -v

# Or run individual examples
python3 metric_geval_example.py
python3 metric_answer_relevancy_example.py
python3 metric_conversational_geval_example.py
python3 metric_faithfulness_example.py
python3 metric_hallucination_example.py
python3 metric_ragas_example.py
python3 metric_golden_dataset_geval_example.py
```

---

## 📚 What is DeepEval?

**DeepEval** is an evaluation framework that uses LLMs (like GPT-4) to judge the quality of other LLM responses.

**How it works:**
```
Your LLM → Generates response → DeepEval (GPT-4) → Evaluates quality → Pass/Fail + Score
```

**Key Benefits:**
- ✅ **No manual evaluation** - Automated quality checks
- ✅ **Semantic understanding** - Judges meaning, not just exact text matches
- ✅ **Flexible criteria** - Define custom evaluation rules
- ✅ **Consistent** - Same evaluation logic every time

---

## 🎯 Evaluation Metrics - When to Use What

### 1. **GEval Metric** 📏
**What it does:** Custom evaluation based on criteria you define

**Use when:**
- ✅ Testing correctness (math, facts, technical explanations)
- ✅ You need domain-specific evaluation criteria
- ✅ Evaluating single-turn responses
- ✅ Measuring specific qualities (accuracy, completeness, clarity)

**Example:** "Is this math calculation correct?" or "Does the code explanation cover key concepts?"

**File:** `metric_geval_example.py` (5 scenarios: Math, Facts, Technical, Summarization, Data Analysis)

---

### 2. **Answer Relevancy** 🎯
**What it does:** Measures if the response actually answers the question

**Use when:**
- ✅ Testing chatbots and Q&A systems
- ✅ Checking if responses stay on-topic
- ✅ Evaluating customer support interactions
- ✅ Detecting rambling or off-topic responses

**Example:** User asks "Do you have this in stock?" - Does the answer address availability or just talk about the product?

**File:** `metric_answer_relevancy_example.py` (6 scenarios with hardcoded responses)

---

### 3. **Conversational GEval** 💬
**What it does:** Evaluates multi-turn conversations (back-and-forth dialogue)

**Use when:**
- ✅ Testing conversational AI and chatbots
- ✅ Evaluating multi-step interactions
- ✅ Assessing dialogue quality (professionalism, helpfulness, problem-solving)
- ✅ Testing customer service conversations

**Example:** A 5-turn conversation about booking an appointment - Was it professional? Did it solve the problem?

**File:** `metric_conversational_geval_example.py` (15 conversations across HR, ERP, Finance)

---

### 4. **Faithfulness** 📄
**What it does:** Checks if the response is grounded in provided source documents (no hallucinations)

**Use when:**
- ✅ Building RAG (Retrieval-Augmented Generation) systems
- ✅ Testing knowledge base Q&A
- ✅ Ensuring responses match company policies/documents
- ✅ Preventing LLM from making up information

**Example:** User asks about parental leave policy - Does the answer match the actual HR policy document?

**File:** `metric_faithfulness_example.py` (8 HR policy scenarios)

---

### 5. **Hallucination** ⚠️
**What it does:** Detects when LLM invents data instead of admitting it doesn't have the information

**Use when:**
- ✅ Testing database queries (only return fields that exist)
- ✅ Validating API responses (don't invent fields)
- ✅ Product catalogs (don't make up inventory/features)
- ✅ Medical/HR systems (CRITICAL - no fake data!)
- ✅ Any system where accuracy is more important than "helpfulness"

**Example:** Database has email and role, but LLM invents phone number and address to please the user

**File:** `metric_hallucination_example.py` (8 critical scenarios)

---

### 6. **RAGAS** 🎯
**What it does:** Comprehensive RAG pipeline evaluation (combines 4 metrics in one)

**The 4 RAGAS Components:**
1. **Answer Relevancy** - Is the response relevant to the question?
2. **Faithfulness** - Is the response grounded in retrieved documents?
3. **Contextual Precision** - Are relevant chunks ranked higher in retrieval?
4. **Contextual Recall** - Does context contain all info needed?

**Use when:**
- ✅ Evaluating RAG (Retrieval-Augmented Generation) systems
- ✅ Testing document retrieval + generation pipeline
- ✅ Measuring both retriever and generator quality
- ✅ Need holistic RAG performance score (0-1)

**Example:** HR chatbot retrieves leave policy docs and generates answer - RAGAS tests if retrieval was good AND answer was accurate

**File:** `metric_ragas_example.py` (8 HR leave policy scenarios)

---

### 7. **Golden Datasets** 🌟
**What it does:** Batch evaluation using predefined correct input-output pairs

**Use when:**
- ✅ Regression testing (did new model version break anything?)
- ✅ Comparing multiple models on the same test set
- ✅ Tracking performance over time
- ✅ Automated CI/CD testing

**Example:** Test 100 known HR policy questions with expected answers to see how well your LLM performs

**File:** `metric_golden_dataset_geval_example.py` (14 HR leave policy scenarios)

---

## 📊 Comparison Table

| Metric | Single Response | Multi-Turn | Requires Context | Use Case |
|--------|----------------|------------|------------------|----------|
| **GEval** | ✅ | ❌ | ❌ | Custom criteria, correctness testing |
| **Answer Relevancy** | ✅ | ❌ | ❌ | Q&A relevance, chatbot responses |
| **Conversational GEval** | ❌ | ✅ | ❌ | Multi-turn dialogues, conversations |
| **Faithfulness** | ✅ | ❌ | ✅ | RAG systems, policy compliance |
| **Hallucination** | ✅ | ❌ | ✅ | Detect invented data, field validation |
| **RAGAS** | ✅ | ❌ | ✅ | Comprehensive RAG evaluation (4-in-1) |
| **Golden Dataset** | ✅ | ✅ | ✅/❌ | Batch testing, regression, comparison |

---

## 📁 Project Structure

```
LLM-Evaluation-Framework/
├── .env                                      # Your OpenAI API key (create this)
├── .env.example                              # Template for environment variables
├── .gitignore                                # Git ignore rules
├── requirements.txt                          # Python dependencies (includes ragas)
│
├── metric_geval_example.py                   # GEval: Correctness testing (5 scenarios)
├── metric_answer_relevancy_example.py        # Answer Relevancy: Hardcoded responses (6 scenarios)
├── metric_conversational_geval_example.py    # Conversational: Multi-turn dialogues (15 scenarios)
├── metric_faithfulness_example.py            # Faithfulness: RAG/policy testing (8 scenarios)
├── metric_hallucination_example.py           # Hallucination: Detect invented data (8 scenarios)
├── metric_ragas_example.py                   # RAGAS: Comprehensive RAG evaluation (8 HR scenarios)
├── metric_golden_dataset_geval_example.py    # Golden Dataset: HR leave policies (14 scenarios)
│
└── README.md                                 # This file
```

---

## 🔧 Troubleshooting

### `openai.RateLimitError: Error code: 429`
**Problem:** No OpenAI credits or billing not set up

**Solution:**
1. Visit [platform.openai.com/account/billing](https://platform.openai.com/account/billing)
2. Add payment method
3. Add credits ($5-10 is plenty for testing)

---

### `ModuleNotFoundError: No module named 'deepeval'`
**Problem:** Dependencies not installed

**Solution:**
```bash
pip3 install -r requirements.txt
```

---

### Tests fail with "expected_output mismatch"
**Problem:** Threshold too high or actual response doesn't match expected

**Solution:**
- Lower threshold (e.g., `0.5` instead of `0.8`)
- Check if `actual_output` makes sense
- Review evaluation criteria

---

## 📖 Real-World Decision Tree

**"Which metric should I use?"**

```
Are you testing a RAG system (retrieval + generation)?
├─ YES → Use RAGAS (comprehensive 4-in-1 evaluation)
└─ NO ↓

Are you testing multi-turn conversations?
├─ YES → Use Conversational GEval
└─ NO ↓

Is your main concern LLM inventing/fabricating data?
├─ YES → Use Hallucination
└─ NO ↓

Do you have source documents/context the LLM should follow?
├─ YES → Use Faithfulness
└─ NO ↓

Are you testing if the answer addresses the question?
├─ YES → Use Answer Relevancy
└─ NO ↓

Do you have 50+ test cases to run repeatedly?
├─ YES → Use Golden Dataset
└─ NO ↓

Need custom evaluation criteria?
└─ YES → Use GEval
```

---

## 🎯 Next Steps

1. **Run the examples** - See how each metric works
2. **Modify test cases** - Change inputs/outputs to match your domain
3. **Add real LLM calls** - Replace simulated responses with actual API calls
4. **Build golden dataset** - Collect 20-50 real test cases from your product
5. **Automate** - Add to CI/CD pipeline for regression testing

---

## 💡 Pro Tips

1. **Start small** - Test with 5-10 cases before scaling to 100+
2. **Adjust thresholds** - `0.5-0.7` is a good starting point, tune based on results
3. **RAG systems** - Use RAGAS for comprehensive evaluation (tests retrieval + generation)
4. **Combine metrics** - Use Answer Relevancy + Faithfulness + Hallucination together
5. **Track over time** - Save results to see if model quality improves/degrades
6. **Use Golden Datasets** - Best for regression testing and model comparison
7. **Hallucination first** - For critical systems (medical, financial), test Hallucination before deployment
8. **Individual RAGAS metrics** - Use separately for detailed debugging of RAG components

---

## 📚 Learn More

- [DeepEval Documentation](https://docs.deepeval.com/)
- [DeepEval Metrics Guide](https://docs.deepeval.com/metrics)
- [OpenAI API Documentation](https://platform.openai.com/docs)
