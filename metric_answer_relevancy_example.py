"""
Answer Relevancy with Simulated LLM Responses
This demonstrates how to evaluate LLM responses using simulated outputs.
Replace simulate_LLM_response() with actual API calls for production use.

⚠️ IMPORTANT: AnswerRelevancyMetric requires OpenAI API access for evaluation.
Even though we're using simulated responses for the LLM being tested,
DeepEval uses OpenAI (GPT-4) internally to evaluate answer relevancy.

Make sure you have:
1. A valid OPENAI_API_KEY in your .env file
2. Sufficient OpenAI credits/quota for evaluation

To run these tests:
    pytest metric_answer_relevancy_with_api_example.py -v -s
    or
    python3 metric_answer_relevancy_with_api_example.py
"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric


def simulate_LLM_response(prompt: str, context: str = "helpful assistant") -> str:
    """
    Simulates LLM responses for testing purposes.
    In production, replace with actual API call to your LLM.
    
    Args:
        prompt: The user's question/input
        context: The system context/role for the response
    
    Returns:
        str: Simulated LLM response
    """
    # Simulated responses based on context type
    responses = {
        # E-commerce responses
        'Do you have the Samsung Galaxy S24 in stock?': {
            'helpful assistant': 'Yes, we currently have the Samsung Galaxy S24 in stock. It is available in Phantom Black, Cream, and Violet colors. The price is $799.99. Would you like to place an order or need more information about its specifications?',
            'e-commerce': 'In stock! Samsung Galaxy S24 available in 3 colors: Phantom Black, Cream, Violet. $799.99. Free shipping on orders over $50. Add to cart?',
            'verbose': 'Thank you for your interest in the Samsung Galaxy S24! Samsung has a long history of innovation in the smartphone industry. The Galaxy S series, first introduced in 2010, has become one of the most popular smartphone lines globally. The latest S24 model features cutting-edge AI capabilities, an impressive camera system, and a stunning display. To answer your question: Yes, we do have it in stock in multiple colors.'
        },
        
        # Travel responses
        'What are the cheapest flights from New York to London in December?': {
            'helpful assistant': 'The cheapest flights from New York to London in December typically range from $450-$600 round-trip. Tuesday and Wednesday departures tend to be least expensive. Airlines like Norse Atlantic ($480), TAP Air Portugal ($520), and Icelandair ($550) usually offer competitive prices with one stopover. Non-stop flights are available from $700+ on major carriers.',
            'travel agent': 'Great question! For December NYC-London routes, I recommend: Norse Atlantic (JFK-LGW) from $480 with carry-on, TAP via Lisbon from $520, or Icelandair via Reykjavik from $550. Book 3-4 weeks out for best deals. Flexible dates? Mid-week flights save $100+. Want me to check specific dates?',
            'verbose': 'Ah, planning a December trip to London! December is a magical time to visit with Christmas markets and festive decorations throughout the city. When it comes to finding affordable flights, there are several factors to consider. The cheapest options typically involve one stopover and departures on less popular days. Budget carriers and those with layovers in their hub cities offer the best rates, usually in the $450-600 range.'
        },
        
        # Healthcare responses
        'Can I schedule a dental cleaning appointment for next week?': {
            'helpful assistant': 'Yes, we have availability for dental cleaning appointments next week. We have openings on Tuesday at 2:00 PM, Wednesday at 10:00 AM, Thursday at 3:30 PM, and Friday at 9:00 AM. A routine cleaning typically takes 45-60 minutes. Which day and time works best for your schedule?',
            'default': 'Yes! Available slots next week: Tue 2PM, Wed 10AM, Thu 3:30PM, Fri 9AM. Cleaning takes ~45-60min. Prefer morning or afternoon?'
        },
        
        # Restaurant responses
        'Do you have vegan options on your menu?': {
            'helpful assistant': 'Yes, we offer several delicious vegan options! Our vegan menu includes: Vegan Buddha Bowl with quinoa and roasted vegetables ($14), Mediterranean Falafel Wrap ($12), Asian Veggie Stir-Fry with tofu ($15), Vegan Margherita Pizza ($16), and Garden Fresh Salad ($11). All dishes are 100% plant-based. Would you like to know more about any specific dish?',
            'default': 'Yes! Vegan options: Buddha Bowl ($14), Falafel Wrap ($12), Veggie Stir-Fry ($15), Vegan Pizza ($16), Garden Salad ($11). All 100% plant-based!'
        },
        
        # Banking responses
        'What is the difference between a checking and savings account?': {
            'helpful assistant': 'The main differences are: Checking accounts are designed for frequent transactions - unlimited deposits/withdrawals, debit cards, checks, minimal interest (0.01-0.05%). Savings accounts are for storing money long-term - limited withdrawals (6/month), higher interest (0.5-4% APY), no checks. Checking is for daily spending, savings is for building funds. Many people maintain both. Need help choosing?',
            'banking assistant': 'Key differences: CHECKING - daily transactions, unlimited access, debit card, checks, low/no interest. SAVINGS - long-term storage, withdrawal limits, higher interest rates (0.5-4% APY), no checks. Think: checking for spending, savings for goals. Which fits your needs?',
            'default': 'Checking: frequent use, unlimited transactions, debit card, little interest. Savings: long-term, limited withdrawals, earns interest (0.5-4%). Use checking for daily spending, savings for building funds.'
        },
        
        # Education responses
        'What are the prerequisites for Introduction to Python Programming?': {
            'helpful assistant': 'Introduction to Python Programming (CS 101) has the following prerequisites: 1) Completion of Introduction to Computer Science (CS 100) with grade C or better, OR 2) Passing the programming placement test. No prior programming experience required, but basic computer skills expected (file management, text editing). The course meets twice weekly with a mandatory lab session. Would you like information about the placement test?',
            'default': 'Prerequisites: CS 100 (grade C+) OR placement test. No programming experience needed. Basic computer skills required. Class: 2x/week + mandatory lab.'
        },
        
        # Technical support
        'How do I reset my password?': {
            'tech support': 'To reset your password: 1) Go to the login page and click "Forgot Password", 2) Enter your registered email address, 3) Check your inbox for a reset link (arrives within 5 minutes), 4) Click the link and create a new password (min 8 characters, include uppercase, number, and symbol), 5) Log in with your new password. Link expires in 24 hours. Need help with any step?',
            'default': 'Password reset: Login page → "Forgot Password" → Enter email → Check inbox (5min) → Click reset link → Create new password (8+ chars, uppercase, number, symbol) → Login. Link valid 24hrs.'
        },
        
        # Travel agent
        'What documents do I need to travel to Italy?': {
            'travel agent': 'For Italy travel, US citizens need: 1) Valid passport (must be valid 3+ months beyond your stay), 2) No visa required for stays under 90 days, 3) Proof of accommodation, 4) Return/onward ticket, 5) Travel insurance recommended. COVID requirements change - check current health rules before departure. When are you planning to travel?',
            'default': 'Italy requirements: Valid passport (3mo+ validity), no visa (under 90 days), proof of accommodation, return ticket. Travel insurance recommended. Check COVID rules before departure.'
        },
        
        # Fitness
        'What\'s the best workout routine for beginners?': {
            'fitness coach': 'Great starting point! For beginners, I recommend: 3 days/week full-body routine - Day 1: Upper body (push-ups, rows, shoulder press), Day 2: Lower body (squats, lunges, deadlifts), Day 3: Core+cardio (planks, bicycle crunches, 20min walk/jog). Start with bodyweight or light weights, 2-3 sets of 10-12 reps. Focus on form over weight. Rest days are crucial! Ready to start this week?',
            'default': '3x/week full-body: Upper (push-ups, rows), Lower (squats, lunges), Core+cardio (planks, 20min cardio). 2-3 sets, 10-12 reps, focus on form. Rest between days!'
        }
    }
    
    # Try to find response based on prompt and context
    context_key = 'helpful assistant' if 'assistant' in context.lower() else \
                  'e-commerce' if 'e-commerce' in context.lower() or 'support' in context.lower() else \
                  'verbose' if 'verbose' in context.lower() or 'storyteller' in context.lower() else \
                  'travel agent' if 'travel' in context.lower() else \
                  'banking assistant' if 'banking' in context.lower() else \
                  'tech support' if 'technical support' in context.lower() else \
                  'fitness coach' if 'fitness' in context.lower() else \
                  'default'
    
    if prompt in responses and context_key in responses[prompt]:
        return responses[prompt][context_key]
    elif prompt in responses and 'default' in responses[prompt]:
        return responses[prompt]['default']
    elif prompt in responses:
        # Return first available response
        return list(responses[prompt].values())[0]
    else:
        return f"I understand your question about: {prompt}. Let me help you with that information."


def test_ecommerce_with_simulated_response():
    """
    E-commerce scenario with simulated LLM response
    Tests response relevancy using simulated output
    """
    
    metric = AnswerRelevancyMetric(
        threshold=0.7,
        include_reason=True
    )
    
    # User's question
    user_question = "Do you have the Samsung Galaxy S24 in stock?"
    
    # Get simulated response
    print(f"\n📝 Question: {user_question}")
    print("⏳ Getting simulated response...")
    
    actual_response = simulate_LLM_response(user_question)
    
    print(f"💬 LLM Response: {actual_response}\n")
    
    # Create test case with simulated response
    test_case = LLMTestCase(
        input=user_question,
        actual_output=actual_response
    )
    
    print("=" * 80)
    print("EVALUATING SIMULATED LLM RESPONSE")
    print("=" * 80)
    
    evaluate(
        test_cases=[test_case],
        metrics=[metric]
    )


def test_multiple_questions_with_simulated_responses():
    """
    Test multiple questions with simulated responses
    Demonstrates batch evaluation of simulated LLM outputs
    """
    
    metric = AnswerRelevancyMetric(
        threshold=0.7,
        include_reason=True
    )
    
    # Multiple user questions
    questions = [
        "Do you have the Samsung Galaxy S24 in stock?",
        "What are the cheapest flights from New York to London in December?",
        "Can I schedule a dental cleaning appointment for next week?",
        "Do you have vegan options on your menu?",
        "What is the difference between a checking and savings account?"
    ]
    
    test_cases = []
    
    print("\n" + "=" * 80)
    print("GETTING SIMULATED RESPONSES FOR MULTIPLE QUESTIONS")
    print("=" * 80 + "\n")
    
    for i, question in enumerate(questions, 1):
        print(f"{i}. Question: {question}")
        print("   ⏳ Getting simulated response...")
        
        # Get simulated LLM response
        response = simulate_LLM_response(question)
        
        print(f"   💬 Response: {response[:100]}...")
        print()
        
        # Create test case with simulated response
        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=response
            )
        )
    
    print("=" * 80)
    print("EVALUATING ALL SIMULATED RESPONSES")
    print("=" * 80)
    
    evaluate(
        test_cases=test_cases,
        metrics=[metric]
    )


def test_with_different_contexts():
    """
    Compare relevancy of responses with different system contexts
    Shows how context/prompt engineering affects answer relevancy
    """
    
    metric = AnswerRelevancyMetric(
        threshold=0.7,
        include_reason=True
    )
    
    question = "Do you have the Samsung Galaxy S24 in stock?"
    
    # Test with different contexts
    contexts = [
        "helpful assistant",
        "e-commerce customer support agent",
        "verbose storyteller"
    ]
    
    test_cases = []
    
    print("\n" + "=" * 80)
    print("COMPARING RESPONSES WITH DIFFERENT CONTEXTS")
    print("=" * 80 + "\n")
    
    for context in contexts:
        print(f"🎯 Testing with: {context}")
        print("   ⏳ Getting simulated response...")
        
        # Get response with specific context
        actual_response = simulate_LLM_response(question, context)
        
        print(f"   💬 Response: {actual_response[:100]}...")
        print()
        
        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=actual_response,
                additional_metadata={"context_type": context}
            )
        )
    
    print("=" * 80)
    print("EVALUATING RELEVANCY ACROSS DIFFERENT CONTEXTS")
    print("=" * 80)
    
    evaluate(
        test_cases=test_cases,
        metrics=[metric]
    )


def test_with_response_variations():
    """
    Compare how different response styles affect relevancy
    Simulates focused vs creative responses
    """
    
    metric = AnswerRelevancyMetric(
        threshold=0.7,
        include_reason=True
    )
    
    question = "What is the difference between a checking and savings account?"
    
    contexts = ["banking assistant", "helpful assistant", "default"]
    context_labels = ["Focused Banking", "Standard Assistant", "Generic Response"]
    
    test_cases = []
    
    print("\n" + "=" * 80)
    print("TESTING RESPONSE STYLE IMPACT ON ANSWER RELEVANCY")
    print("=" * 80 + "\n")
    
    for context, label in zip(contexts, context_labels):
        print(f"🌡️  Style: {label}")
        print("   ⏳ Getting simulated response...")
        
        actual_response = simulate_LLM_response(question, context)
        
        print(f"   💬 Response: {actual_response[:100]}...")
        print()
        
        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=actual_response,
                additional_metadata={"style": label}
            )
        )
    
    print("=" * 80)
    print("EVALUATING RELEVANCY AT DIFFERENT RESPONSE STYLES")
    print("=" * 80)
    
    evaluate(
        test_cases=test_cases,
        metrics=[metric]
    )


def test_with_custom_response_source():
    """
    Example of integrating with custom response source
    Shows flexibility of the evaluation framework
    Replace simulate_LLM_response() with your actual API/LLM call
    """
    
    def get_custom_llm_response(prompt: str) -> str:
        """
        Replace this with your actual API call
        Could be Claude, Gemini, local model, custom endpoint, etc.
        
        Example patterns:
        - REST API: requests.post(url, json={"prompt": prompt})
        - gRPC: stub.Generate(GenerateRequest(prompt=prompt))
        - Local model: model.generate(prompt)
        - Queue-based: queue.enqueue(prompt); return queue.dequeue()
        """
        # For demo purposes, using simulated response
        return simulate_LLM_response(prompt)
    
    metric = AnswerRelevancyMetric(
        threshold=0.7,
        include_reason=True
    )
    
    question = "What are the prerequisites for Introduction to Python Programming?"
    
    print("\n" + "=" * 80)
    print("EVALUATING RESPONSE FROM CUSTOM SOURCE")
    print("=" * 80 + "\n")
    
    print(f"📝 Question: {question}")
    print("⏳ Getting response from custom source...")
    
    # Get response from your custom source
    response = get_custom_llm_response(question)
    
    print(f"💬 Response: {response[:100]}...")
    print()
    
    test_case = LLMTestCase(
        input=question,
        actual_output=response
    )
    
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    evaluate(
        test_cases=[test_case],
        metrics=[metric]
    )


def test_parameterized_questions_and_contexts():
    """
    Advanced: Parameterized testing with various question-context pairs
    Useful for testing chatbots, RAG systems, or contextual responses
    """
    
    metric = AnswerRelevancyMetric(
        threshold=0.7,
        include_reason=True
    )
    
    # Test data with context
    test_scenarios = [
        {
            "context": "technical support agent for a software company",
            "question": "How do I reset my password?",
        },
        {
            "context": "travel agent specializing in European vacations",
            "question": "What documents do I need to travel to Italy?",
        },
        {
            "context": "fitness coach at a gym",
            "question": "What's the best workout routine for beginners?",
        }
    ]
    
    test_cases = []
    
    print("\n" + "=" * 80)
    print("PARAMETERIZED TESTING WITH DIFFERENT CONTEXTS")
    print("=" * 80 + "\n")
    
    for scenario in test_scenarios:
        print(f"🎭 Context: {scenario['context'][:50]}...")
        print(f"❓ Question: {scenario['question']}")
        print("   ⏳ Getting simulated response...")
        
        # Get response with specific context
        actual_response = simulate_LLM_response(scenario['question'], scenario['context'])
        
        print(f"   💬 Response: {actual_response[:80]}...")
        print()
        
        test_cases.append(
            LLMTestCase(
                input=scenario['question'],
                actual_output=actual_response,
                context=[scenario['context']]  # Optional: include context in evaluation
            )
        )
    
    print("=" * 80)
    print("EVALUATING ALL PARAMETERIZED RESPONSES")
    print("=" * 80)
    
    evaluate(
        test_cases=test_cases,
        metrics=[metric]
    )


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ANSWER RELEVANCY EVALUATION WITH SIMULATED LLM RESPONSES")
    print("="*80)
    print("\nThis demonstrates evaluating LLM responses using simulated outputs.")
    print("Replace simulate_LLM_response() with actual API calls for production use.")
    print("\n" + "="*80)
    
    # Check OpenAI API key (required for evaluation, not for LLM responses)
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not found!")
        print("\n⚠️  NOTE: Even though we're using SIMULATED LLM responses,")
        print("   DeepEval's AnswerRelevancyMetric requires OpenAI API access")
        print("   to evaluate the relevancy of those responses.")
        print("\n   OpenAI API is used as the 'judge' to evaluate your LLM's responses.")
        print("\n💡 To fix: Add OPENAI_API_KEY to your .env file")
        exit(1)
    
    # Run tests
    print("\n1️⃣  Testing single question with simulated response...")
    test_ecommerce_with_simulated_response()
    
    print("\n2️⃣  Testing multiple questions with simulated responses...")
    test_multiple_questions_with_simulated_responses()
    
    print("\n3️⃣  Comparing different contexts...")
    test_with_different_contexts()
    
    print("\n4️⃣  Testing response style variations...")
    test_with_response_variations()
    
    print("\n5️⃣  Testing with custom response source...")
    test_with_custom_response_source()
    
    print("\n6️⃣  Parameterized testing with contexts...")
    test_parameterized_questions_and_contexts()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
    print("\n💡 TIP: Replace simulate_LLM_response() with your actual LLM API calls:")
    print("   - OpenAI: openai.chat.completions.create()")
    print("   - Anthropic: anthropic.messages.create()")
    print("   - Google: genai.GenerativeModel().generate_content()")
    print("   - Custom: your_api_client.generate()")
    print("\n📊 HOW IT WORKS:")
    print("   Your LLM → Generates response (can be ANY LLM)")
    print("   OpenAI GPT-4 → Evaluates relevancy (DeepEval's judge)")
    print("="*80)

