"""
RAGAS Metric Examples - HR Leave Credit Policies
Comprehensive RAG pipeline evaluation for HR chatbot scenarios.

⚠️ PREREQUISITE: Install ragas library first
   pip install ragas

RAGAS evaluates 4 aspects of your RAG system:
1. Answer Relevancy - Is the response relevant to the question?
2. Faithfulness - Is the response grounded in retrieved context?
3. Contextual Precision - Are relevant chunks ranked higher?
4. Contextual Recall - Does context contain all info needed for expected output?

Perfect for testing HR chatbots, knowledge base Q&A, and RAG pipelines!
"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics.ragas import RagasMetric
from deepeval.metrics.ragas import (
    RAGASAnswerRelevancyMetric,
    RAGASFaithfulnessMetric,
    RAGASContextualPrecisionMetric,
    RAGASContextualRecallMetric
)


# ============================================================================
# HR POLICY DOCUMENTS (Retrieval Context)
# ============================================================================

LEAVE_POLICY_DOC = """
Company Leave Credit Policy - Effective January 2024

ANNUAL LEAVE (Vacation Days):
- All full-time employees receive 15 days of paid annual leave per year
- Leave accrues at 1.25 days per month
- Maximum carryover: 5 days to the next year
- Unused days beyond 5 are forfeited
- Must be requested at least 2 weeks in advance

SICK LEAVE:
- 10 days of paid sick leave per year
- No carryover to the next year
- Medical certificate required for absences of 3+ consecutive days
- Can be used for personal illness or to care for immediate family members

PARENTAL LEAVE:
- Primary caregiver: 16 weeks paid leave
- Secondary caregiver: 8 weeks paid leave
- Must be taken within 12 months of child's birth or adoption
- Requires 6 months of employment to be eligible

BEREAVEMENT LEAVE:
- Immediate family (parent, spouse, child): 5 days paid
- Extended family (sibling, grandparent): 3 days paid
- No documentation required for up to 3 days

PERSONAL DAYS:
- 3 paid personal days per year
- Can be used for any reason without explanation
- Cannot be carried over to next year
- No advance notice required (but appreciated)
"""

PTO_ACCRUAL_DOC = """
Leave Accrual and Calculation Details

ACCRUAL RATES BY TENURE:
Years 0-2: 15 annual leave days per year (1.25 days/month)
Years 3-5: 18 annual leave days per year (1.5 days/month)
Years 6+: 20 annual leave days per year (1.67 days/month)

IMPORTANT NOTES:
- Leave accrues from your first day of employment
- Part-time employees receive prorated leave (based on hours worked)
- Leave can only be used after the probation period (90 days)
- Negative leave balances are not permitted
"""

LEAVE_REQUEST_PROCESS_DOC = """
How to Request Leave

REQUESTING ANNUAL LEAVE:
1. Submit request via HR portal at least 2 weeks in advance
2. Manager approval required
3. Confirmation email sent within 3 business days
4. For emergencies, contact manager directly

REPORTING SICK LEAVE:
1. Notify manager as soon as possible (preferably before shift start)
2. Submit sick leave form in HR portal when you return
3. Medical certificate required if absent 3+ consecutive days

PARENTAL LEAVE APPLICATION:
1. Submit application at least 30 days before leave start date
2. Provide birth certificate or adoption papers
3. HR will review and confirm eligibility
4. Leave can be taken in one continuous period or split (with approval)
"""


# ============================================================================
# SIMULATED LLM RESPONSES (Replace with actual RAG system outputs)
# ============================================================================

def simulate_RAG_response(question: str, scenario: str = "good") -> str:
    """
    Simulates RAG system responses.
    In production, replace with actual calls to your RAG pipeline.
    """
    responses = {
        # Annual Leave Questions
        "annual_leave_days_good": "All full-time employees receive 15 days of paid annual leave per year, which accrues at 1.25 days per month.",
        "annual_leave_days_hallucinated": "All full-time employees receive 20 days of paid annual leave per year with unlimited carryover.",
        
        "annual_leave_carryover_good": "You can carry over a maximum of 5 annual leave days to the next year. Any unused days beyond 5 will be forfeited.",
        "annual_leave_carryover_partial": "Annual leave can be carried over to the next year, but there are limitations. Check with HR for specific details.",
        
        # Sick Leave Questions
        "sick_leave_good": "You receive 10 days of paid sick leave per year. A medical certificate is required for absences of 3 or more consecutive days.",
        "sick_leave_incorrect": "You receive 10 days of paid sick leave per year, which can be carried over to the next year if unused.",
        
        # Parental Leave Questions
        "parental_leave_good": "Primary caregivers are eligible for 16 weeks of paid parental leave, while secondary caregivers receive 8 weeks. You must have 6 months of employment to be eligible.",
        "parental_leave_wrong_eligibility": "Primary caregivers receive 16 weeks of paid parental leave. There is no minimum employment requirement to be eligible.",
        
        # Bereavement Leave Questions
        "bereavement_immediate_good": "For immediate family members (parent, spouse, or child), you are entitled to 5 days of paid bereavement leave.",
        "bereavement_extended_good": "For extended family members such as siblings or grandparents, you receive 3 days of paid bereavement leave.",
        
        # Personal Days Questions
        "personal_days_good": "You receive 3 paid personal days per year. These can be used for any reason without explanation and cannot be carried over to the next year.",
        
        # Accrual Questions
        "accrual_rate_good": "Annual leave accrues at different rates based on tenure: 1.25 days per month for years 0-2, 1.5 days per month for years 3-5, and 1.67 days per month for 6+ years.",
        
        # Leave Request Process
        "request_process_good": "To request annual leave, submit your request via the HR portal at least 2 weeks in advance. Your manager will need to approve it, and you'll receive confirmation within 3 business days.",
        "request_process_irrelevant": "You can request leave by talking to your manager. We have a great HR team that handles many types of requests including benefits enrollment and payroll questions.",
    }
    
    return responses.get(f"{question}_{scenario}", "I don't have that information available.")


# ============================================================================
# TEST SCENARIOS
# ============================================================================

def test_ragas_annual_leave_inquiry():
    """
    Test 1: Basic Annual Leave Days Inquiry
    Tests all 4 RAGAS components with a straightforward policy question.
    """
    
    metric = RagasMetric(threshold=0.5, model="gpt-4")
    
    # Good response - accurate and grounded in policy
    test_case_good = LLMTestCase(
        input="How many annual leave days do full-time employees get per year?",
        actual_output=simulate_RAG_response("annual_leave_days", "good"),
        expected_output="Full-time employees receive 15 days of paid annual leave per year.",
        retrieval_context=[LEAVE_POLICY_DOC]
    )
    
    # Bad response - hallucinated information
    test_case_bad = LLMTestCase(
        input="How many annual leave days do full-time employees get per year?",
        actual_output=simulate_RAG_response("annual_leave_days", "hallucinated"),
        expected_output="Full-time employees receive 15 days of paid annual leave per year.",
        retrieval_context=[LEAVE_POLICY_DOC]
    )
    
    print("\n" + "="*80)
    print("TEST 1: ANNUAL LEAVE DAYS INQUIRY (RAGAS COMBINED METRIC)")
    print("="*80)
    print("Question: How many annual leave days do full-time employees get?")
    print("Expected: 15 days per year")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_good, test_case_bad],
        metrics=[metric]
    )


def test_ragas_leave_carryover_policy():
    """
    Test 2: Leave Carryover Rules
    Tests understanding of carryover limitations.
    """
    
    metric = RagasMetric(threshold=0.5, model="gpt-4")
    
    test_case_good = LLMTestCase(
        input="How many annual leave days can I carry over to next year?",
        actual_output=simulate_RAG_response("annual_leave_carryover", "good"),
        expected_output="You can carry over a maximum of 5 annual leave days to the next year. Days beyond 5 are forfeited.",
        retrieval_context=[LEAVE_POLICY_DOC]
    )
    
    test_case_partial = LLMTestCase(
        input="How many annual leave days can I carry over to next year?",
        actual_output=simulate_RAG_response("annual_leave_carryover", "partial"),
        expected_output="You can carry over a maximum of 5 annual leave days to the next year. Days beyond 5 are forfeited.",
        retrieval_context=[LEAVE_POLICY_DOC]
    )
    
    print("\n" + "="*80)
    print("TEST 2: LEAVE CARRYOVER POLICY")
    print("="*80)
    print("Question: How many days can I carry over?")
    print("Expected: Maximum 5 days")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_good, test_case_partial],
        metrics=[metric]
    )


def test_ragas_sick_leave_policy():
    """
    Test 3: Sick Leave with Medical Certificate Requirement
    Tests if RAG system correctly retrieves medical cert requirements.
    """
    
    metric = RagasMetric(threshold=0.5, model="gpt-4")
    
    test_case_good = LLMTestCase(
        input="How much sick leave do I get and when do I need a medical certificate?",
        actual_output=simulate_RAG_response("sick_leave", "good"),
        expected_output="You get 10 days of paid sick leave per year. A medical certificate is required for absences of 3 or more consecutive days.",
        retrieval_context=[LEAVE_POLICY_DOC]
    )
    
    test_case_incorrect = LLMTestCase(
        input="How much sick leave do I get and when do I need a medical certificate?",
        actual_output=simulate_RAG_response("sick_leave", "incorrect"),
        expected_output="You get 10 days of paid sick leave per year. A medical certificate is required for absences of 3 or more consecutive days.",
        retrieval_context=[LEAVE_POLICY_DOC]
    )
    
    print("\n" + "="*80)
    print("TEST 3: SICK LEAVE POLICY")
    print("="*80)
    print("Question: Sick leave days and medical certificate requirements?")
    print("Expected: 10 days, cert needed for 3+ consecutive days")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_good, test_case_incorrect],
        metrics=[metric]
    )


def test_ragas_parental_leave_eligibility():
    """
    Test 4: Parental Leave Eligibility Requirements
    Tests retrieval of complex eligibility criteria.
    """
    
    metric = RagasMetric(threshold=0.5, model="gpt-4")
    
    test_case_good = LLMTestCase(
        input="What are the eligibility requirements for parental leave as a primary caregiver?",
        actual_output=simulate_RAG_response("parental_leave", "good"),
        expected_output="Primary caregivers are eligible for 16 weeks of paid parental leave. You must have completed 6 months of employment to be eligible.",
        retrieval_context=[LEAVE_POLICY_DOC]
    )
    
    test_case_wrong = LLMTestCase(
        input="What are the eligibility requirements for parental leave as a primary caregiver?",
        actual_output=simulate_RAG_response("parental_leave", "wrong_eligibility"),
        expected_output="Primary caregivers are eligible for 16 weeks of paid parental leave. You must have completed 6 months of employment to be eligible.",
        retrieval_context=[LEAVE_POLICY_DOC]
    )
    
    print("\n" + "="*80)
    print("TEST 4: PARENTAL LEAVE ELIGIBILITY")
    print("="*80)
    print("Question: Eligibility for parental leave as primary caregiver?")
    print("Expected: 16 weeks, need 6 months employment")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_good, test_case_wrong],
        metrics=[metric]
    )


def test_ragas_individual_metrics():
    """
    Test 5: Using Individual RAGAS Metrics
    Demonstrates how to test each component separately for detailed analysis.
    """
    
    # Create individual metrics
    relevancy_metric = RAGASAnswerRelevancyMetric(threshold=0.5, model="gpt-4")
    faithfulness_metric = RAGASFaithfulnessMetric(threshold=0.5, model="gpt-4")
    precision_metric = RAGASContextualPrecisionMetric(threshold=0.5, model="gpt-4")
    recall_metric = RAGASContextualRecallMetric(threshold=0.5, model="gpt-4")
    
    test_case = LLMTestCase(
        input="How do I request annual leave?",
        actual_output=simulate_RAG_response("request_process", "good"),
        expected_output="Submit your annual leave request via the HR portal at least 2 weeks in advance. Your manager must approve it, and you'll receive confirmation within 3 business days.",
        retrieval_context=[LEAVE_REQUEST_PROCESS_DOC]
    )
    
    print("\n" + "="*80)
    print("TEST 5: INDIVIDUAL RAGAS METRICS BREAKDOWN")
    print("="*80)
    print("Question: How do I request annual leave?")
    print("Testing each RAGAS component separately...")
    print("="*80)
    
    # Evaluate each metric individually
    print("\n1️⃣  Testing Answer Relevancy...")
    evaluate([test_case], [relevancy_metric])
    
    print("\n2️⃣  Testing Faithfulness...")
    evaluate([test_case], [faithfulness_metric])
    
    print("\n3️⃣  Testing Contextual Precision...")
    evaluate([test_case], [precision_metric])
    
    print("\n4️⃣  Testing Contextual Recall...")
    evaluate([test_case], [recall_metric])


def test_ragas_multi_document_retrieval():
    """
    Test 6: Multiple Documents Retrieved
    Tests RAG system with multiple relevant documents in context.
    """
    
    metric = RagasMetric(threshold=0.5, model="gpt-4")
    
    test_case = LLMTestCase(
        input="How does annual leave accrual work for employees with 4 years of service?",
        actual_output=simulate_RAG_response("accrual_rate", "good"),
        expected_output="For employees with 4 years of service, annual leave accrues at 1.5 days per month, totaling 18 days per year.",
        retrieval_context=[
            LEAVE_POLICY_DOC,
            PTO_ACCRUAL_DOC,  # Multiple documents retrieved
        ]
    )
    
    print("\n" + "="*80)
    print("TEST 6: MULTI-DOCUMENT RETRIEVAL")
    print("="*80)
    print("Question: Leave accrual for 4-year employee?")
    print("Expected: 1.5 days/month, 18 days/year")
    print("Context: Multiple policy documents retrieved")
    print("="*80)
    
    evaluate([test_case], [metric])


def test_ragas_irrelevant_response():
    """
    Test 7: Detecting Irrelevant Responses
    Tests RAGAS ability to catch off-topic responses.
    """
    
    metric = RagasMetric(threshold=0.5, model="gpt-4")
    
    test_case_irrelevant = LLMTestCase(
        input="How do I request annual leave?",
        actual_output=simulate_RAG_response("request_process", "irrelevant"),
        expected_output="Submit your annual leave request via the HR portal at least 2 weeks in advance.",
        retrieval_context=[LEAVE_REQUEST_PROCESS_DOC]
    )
    
    print("\n" + "="*80)
    print("TEST 7: IRRELEVANT RESPONSE DETECTION")
    print("="*80)
    print("Question: How do I request annual leave?")
    print("Actual: Rambling about HR team and unrelated topics")
    print("Expected: RAGAS should score this LOW")
    print("="*80)
    
    evaluate([test_case_irrelevant], [metric])


def test_ragas_comprehensive_hr_scenarios():
    """
    Test 8: Comprehensive HR Leave Policy Scenarios
    Tests multiple scenarios in batch for regression testing.
    """
    
    metric = RagasMetric(threshold=0.5, model="gpt-4")
    
    test_cases = [
        # Scenario 1: Bereavement leave for immediate family
        LLMTestCase(
            input="How much bereavement leave do I get if my parent passes away?",
            actual_output=simulate_RAG_response("bereavement_immediate", "good"),
            expected_output="You receive 5 days of paid bereavement leave for immediate family members such as a parent.",
            retrieval_context=[LEAVE_POLICY_DOC]
        ),
        
        # Scenario 2: Bereavement leave for extended family
        LLMTestCase(
            input="How much bereavement leave do I get for my grandmother?",
            actual_output=simulate_RAG_response("bereavement_extended", "good"),
            expected_output="You receive 3 days of paid bereavement leave for extended family members such as grandparents.",
            retrieval_context=[LEAVE_POLICY_DOC]
        ),
        
        # Scenario 3: Personal days policy
        LLMTestCase(
            input="How many personal days do I get and can I carry them over?",
            actual_output=simulate_RAG_response("personal_days", "good"),
            expected_output="You receive 3 paid personal days per year. These cannot be carried over to the next year.",
            retrieval_context=[LEAVE_POLICY_DOC]
        ),
    ]
    
    print("\n" + "="*80)
    print("TEST 8: COMPREHENSIVE HR SCENARIOS (BATCH EVALUATION)")
    print("="*80)
    print("Testing 3 different leave policy scenarios:")
    print("  1. Bereavement (immediate family)")
    print("  2. Bereavement (extended family)")
    print("  3. Personal days")
    print("="*80)
    
    evaluate(test_cases, [metric])


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RAGAS METRIC EVALUATION - HR LEAVE CREDIT POLICIES")
    print("="*80)
    print("\n📚 WHAT IS RAGAS?")
    print("   RAGAS comprehensively evaluates RAG pipelines with 4 metrics:")
    print("   1. Answer Relevancy - Is response relevant to question?")
    print("   2. Faithfulness - Is response grounded in retrieved docs?")
    print("   3. Contextual Precision - Are relevant chunks ranked higher?")
    print("   4. Contextual Recall - Does context have all needed info?")
    print("\n🎯 PERFECT FOR:")
    print("   - HR policy chatbots")
    print("   - Knowledge base Q&A systems")
    print("   - RAG pipeline evaluation")
    print("   - Document retrieval systems")
    print("\n⚠️  PREREQUISITE: pip install ragas")
    print("="*80)
    
    try:
        import ragas
        print("\n✅ ragas library installed! Running tests...\n")
    except ImportError:
        print("\n❌ ERROR: ragas library not installed!")
        print("   Please run: pip install ragas")
        print("   Then run this script again.")
        exit(1)
    
    print("\n🧪 Running RAGAS Tests...\n")
    
    # Run all test scenarios
    test_ragas_annual_leave_inquiry()
    test_ragas_leave_carryover_policy()
    test_ragas_sick_leave_policy()
    test_ragas_parental_leave_eligibility()
    test_ragas_individual_metrics()
    test_ragas_multi_document_retrieval()
    test_ragas_irrelevant_response()
    test_ragas_comprehensive_hr_scenarios()
    
    print("\n" + "="*80)
    print("ALL RAGAS TESTS COMPLETED")
    print("="*80)
    print("\n💡 KEY INSIGHTS:")
    print("   - RAGAS provides holistic RAG pipeline evaluation")
    print("   - Scores range from 0-1 (higher is better)")
    print("   - Threshold of 0.5 means 50% minimum to pass")
    print("   - Use individual metrics for detailed debugging")
    print("\n🚀 NEXT STEPS:")
    print("   1. Replace simulate_RAG_response() with your actual RAG system")
    print("   2. Add your own HR policy documents as retrieval_context")
    print("   3. Collect real user questions for test cases")
    print("   4. Run in CI/CD to catch RAG quality regressions")
    print("="*80)

