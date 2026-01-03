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
python3 metric_conversational_geval_example.py
python3 metric_answer_relevancy_example.py
python3 metric_faithfulness_example.py
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

## 🎯 Evaluation Metrics

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

## 📊 Comparison Table

| Metric | Single Response | Multi-Turn | Requires Context | Use Case |
|--------|----------------|------------|------------------|----------|
| **GEval** | ✅ | ❌ | ❌ | Custom criteria, correctness testing |
| **Answer Relevancy** | ✅ | ❌ | ❌ | Q&A relevance, chatbot responses |
| **Conversational GEval** | ❌ | ✅ | ❌ | Multi-turn dialogues, conversations |
| **Faithfulness** | ✅ | ❌ | ✅ | RAG systems, policy compliance |

---

## 📁 Project Structure

```
LLM-Evaluation-Framework/
├── .env                                   # Your OpenAI API key (create this)
├── .env.example                           # Template for environment variables
├── .gitignore                             # Git ignore rules
├── requirements.txt                       # Python dependencies
│
├── metric_geval_example.py                # GEval: Correctness testing (5 scenarios)
├── metric_answer_relevancy_example.py     # Answer Relevancy: Hardcoded responses (6 scenarios)
├── metric_conversational_geval_example.py # Conversational: Multi-turn dialogues (15 scenarios)
├── metric_faithfulness_example.py         # Faithfulness: RAG/policy testing (8 scenarios)
│
└── README.md                              # This file
```

---

## 🧪 Example Use Cases

### E-Commerce Chatbot
```python
# Test if chatbot gives relevant product information
metric = AnswerRelevancyMetric(threshold=0.7)
test_case = LLMTestCase(
    input="Do you have this in stock?",
    actual_output="Yes! In stock. Black and White. $199."
)
```
**Use:** Answer Relevancy

---

### HR Policy Bot
```python
# Ensure responses match official HR policy
metric = FaithfulnessMetric(threshold=0.7)
test_case = LLMTestCase(
    input="How many vacation days do I get?",
    actual_output="You receive 15 vacation days per year.",
    retrieval_context=["Company Policy: Employees receive 15 vacation days annually."]
)
```
**Use:** Faithfulness

---

### Customer Support Conversation
```python
# Evaluate multi-turn support interaction
test_case = ConversationalTestCase(
    turns=[
        {"role": "user", "content": "I need help with my order"},
        {"role": "assistant", "content": "I'd be happy to help! What's your order number?"},
        {"role": "user", "content": "12345"},
        {"role": "assistant", "content": "Found it! Your order ships tomorrow."}
    ]
)
metric = GEval(
    name="Helpfulness",
    criteria="Rate how helpful and professional the support interaction was"
)
```
**Use:** Conversational GEval

---

### Math Calculator App
```python
# Verify calculation accuracy
metric = GEval(
    name="Correctness",
    criteria="Check if the numerical answer is correct"
)
test_case = LLMTestCase(
    input="What is 5 divided by 2?",
    expected_output="2.5",
    actual_output="2.5"
)
```
**Use:** GEval

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

## 📖 Decision Tree

**"Which metric should I use?"**

```
Are you testing multi-turn conversations?
├─ YES → Use Conversational GEval
└─ NO ↓

Do you have source documents/context the LLM should follow?
├─ YES → Use Faithfulness
└─ NO ↓

Are you testing if the answer addresses the question?
├─ YES → Use Answer Relevancy
└─ NO → Use GEval with custom criteria
```

---

## 🎯 Next Steps

1. **Run the examples** - See how each metric works
2. **Modify test cases** - Change inputs/outputs to match your domain
3. **Add real LLM calls** - Replace simulated responses with actual API calls
4. **Build test suite** - Collect 20-50 real test cases from your product
5. **Automate** - Add to CI/CD pipeline for regression testing

---

## 💡 Pro Tips

1. **Start small** - Test with 5-10 cases before scaling to 100+
2. **Adjust thresholds** - `0.5-0.7` is a good starting point, tune based on results
3. **Combine metrics** - Use Answer Relevancy + Faithfulness together for RAG systems
4. **Track over time** - Save results to see if model quality improves/degrades

---

## 📚 Learn More

- [DeepEval Documentation](https://docs.deepeval.com/)
- [DeepEval Metrics Guide](https://docs.deepeval.com/metrics)
- [OpenAI API Documentation](https://platform.openai.com/docs)

---

## 📄 License

MIT
