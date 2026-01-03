"""
Golden Dataset Examples - HR Leave Credit Policies
This demonstrates using predefined Golden datasets to evaluate LLM responses
for HR leave policy inquiries.

Golden datasets are perfect for:
- Regression testing (ensure new model versions don't break existing answers)
- Model comparison (compare GPT-4 vs GPT-3.5 vs Claude on same questions)
- Performance tracking (track accuracy over time)
"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset, Golden


# Simulated LLM response function (replace with actual LLM API call)
def simulate_LLM_answer(prompt):
    """
    Simulates LLM responses for HR leave policy queries.
    In production, replace with actual API call to your LLM.
    """
    responses = {
        # Annual Leave Questions
        'How many annual leave days do full-time employees receive per year?': 
            'Full-time employees receive 15 days of paid annual leave per year, which accrues at 1.25 days per month.',
        
        'What is the maximum number of annual leave days I can carry over to the next year?': 
            'You can carry over a maximum of 5 annual leave days to the next year. Any unused days beyond 5 will be forfeited.',
        
        'How far in advance do I need to request annual leave?': 
            'Annual leave requests must be submitted at least 2 weeks in advance through the HR portal. Manager approval is required.',
        
        # Sick Leave Questions
        'How much sick leave do employees get annually?': 
            'Employees receive 10 days of paid sick leave per year. Sick leave does not carry over to the next year.',
        
        'When do I need to provide a medical certificate for sick leave?': 
            'A medical certificate is required for sick leave absences of 3 or more consecutive days.',
        
        'Can I use sick leave to care for a family member?': 
            'Yes, sick leave can be used to care for immediate family members (spouse, children, parents).',
        
        # Parental Leave Questions
        'How many weeks of parental leave do primary caregivers get?': 
            'Primary caregivers are eligible for 16 weeks of paid parental leave. You must have 6 months of employment to qualify.',
        
        'What is the parental leave policy for secondary caregivers?': 
            'Secondary caregivers receive 8 weeks of paid parental leave. The same 6-month employment requirement applies.',
        
        'When must parental leave be taken?': 
            'Parental leave must be taken within 12 months of the child\'s birth or adoption. Applications should be submitted 30 days before the leave start date.',
        
        # Bereavement Leave Questions
        'How much bereavement leave do I get if an immediate family member passes away?': 
            'You receive 5 days of paid bereavement leave for immediate family members (parent, spouse, child).',
        
        'What is the bereavement leave policy for extended family?': 
            'You receive 3 days of paid bereavement leave for extended family members (siblings, grandparents).',
        
        # Personal Days Questions
        'How many personal days do employees receive?': 
            'Employees receive 3 paid personal days per year. These can be used for any reason without explanation.',
        
        'Can I carry over unused personal days to the next year?': 
            'No, personal days cannot be carried over to the next year. They must be used within the calendar year.',
        
        # Accrual and Tenure Questions
        'How does leave accrual change with tenure?': 
            'Leave accrual increases with tenure: 15 days/year (0-2 years), 18 days/year (3-5 years), 20 days/year (6+ years).',
        
        'Do part-time employees receive the same leave benefits?': 
            'Part-time employees receive prorated leave benefits based on their scheduled hours worked.',
        
        # Leave Request Process
        'What is the process for requesting annual leave?': 
            'Submit your request via the HR portal at least 2 weeks in advance. Your manager will approve it, and you\'ll receive confirmation within 3 business days.',
        
        'How do I report sick leave?': 
            'Notify your manager as soon as possible before your shift starts. Submit the sick leave form in the HR portal when you return to work.',
        
        # Special Scenarios
        'What happens if I have a negative leave balance?': 
            'Negative leave balances are not permitted. You cannot take leave if you have insufficient accrued balance.',
        
        'Can I take leave during my probation period?': 
            'Leave accrues from your first day but can only be used after completing the 90-day probation period.',
    }
    
    return responses.get(prompt, "I don't have specific information about that leave policy. Please consult with HR or refer to the employee handbook.")


def test_annual_leave_policies_golden():
    """
    Test 1: Annual Leave Policies
    Using Golden dataset to evaluate responses about annual leave rules
    """
    
    # Define metric for evaluation
    correctness_metric = GEval(
        name="HR Policy Correctness",
        criteria=(
            "Determine if the actual output correctly states the HR leave policy, "
            "provides accurate numbers and timeframes, and matches the company policy "
            "as shown in the expected output. The information must be precise and compliant."
        ),
        evaluation_params=["actual_output", "expected_output"],
        threshold=0.7
    )
    
    # Define Golden dataset for annual leave scenarios
    goldens = [
        Golden(
            input='How many annual leave days do full-time employees receive per year?',
            expected_output='Full-time employees receive 15 days of paid annual leave per year, accruing at 1.25 days per month.'
        ),
        Golden(
            input='What is the maximum number of annual leave days I can carry over to the next year?',
            expected_output='Maximum 5 annual leave days can be carried over. Days beyond 5 are forfeited.'
        ),
        Golden(
            input='How far in advance do I need to request annual leave?',
            expected_output='Annual leave must be requested at least 2 weeks in advance through the HR portal with manager approval.'
        ),
    ]
    
    # Create evaluation dataset
    dataset = EvaluationDataset(goldens=goldens)
    
    # Generate test cases with simulated LLM responses
    for golden in dataset.goldens:
        dataset.add_test_case(
            LLMTestCase(
                input=golden.input,
                expected_output=golden.expected_output,
                actual_output=simulate_LLM_answer(golden.input)
            )
        )
    
    print("\n" + "="*80)
    print("TEST 1: ANNUAL LEAVE POLICIES")
    print("="*80)
    
    # Evaluate
    evaluate(dataset, [correctness_metric])


def test_sick_leave_policies_golden():
    """
    Test 2: Sick Leave Policies
    Using Golden dataset to evaluate responses about sick leave rules
    """
    
    sick_leave_metric = GEval(
        name="Sick Leave Policy Accuracy",
        criteria=(
            "Evaluate if the actual output accurately states sick leave entitlements, "
            "medical certificate requirements, and proper usage. The response should "
            "help employees understand when and how to use sick leave correctly."
        ),
        evaluation_params=["actual_output", "expected_output"],
        threshold=0.7
    )
    
    goldens = [
        Golden(
            input='How much sick leave do employees get annually?',
            expected_output='10 days of paid sick leave per year. Does not carry over to next year.'
        ),
        Golden(
            input='When do I need to provide a medical certificate for sick leave?',
            expected_output='Medical certificate required for 3 or more consecutive days of sick leave.'
        ),
        Golden(
            input='Can I use sick leave to care for a family member?',
            expected_output='Yes, sick leave can be used to care for immediate family members (spouse, children, parents).'
        ),
    ]
    
    dataset = EvaluationDataset(goldens=goldens)
    
    for golden in dataset.goldens:
        dataset.add_test_case(
            LLMTestCase(
                input=golden.input,
                expected_output=golden.expected_output,
                actual_output=simulate_LLM_answer(golden.input)
            )
        )
    
    print("\n" + "="*80)
    print("TEST 2: SICK LEAVE POLICIES")
    print("="*80)
    
    evaluate(dataset, [sick_leave_metric])


def test_parental_leave_policies_golden():
    """
    Test 3: Parental Leave Policies
    Using Golden dataset to evaluate parental leave eligibility and duration
    """
    
    parental_leave_metric = GEval(
        name="Parental Leave Policy Accuracy",
        criteria=(
            "Assess if the actual output correctly identifies parental leave duration, "
            "eligibility requirements, and timing constraints. Critical for ensuring "
            "employees receive accurate information about family leave benefits."
        ),
        evaluation_params=["actual_output", "expected_output"],
        threshold=0.7
    )
    
    goldens = [
        Golden(
            input='How many weeks of parental leave do primary caregivers get?',
            expected_output='Primary caregivers: 16 weeks paid leave. Requires 6 months employment.'
        ),
        Golden(
            input='What is the parental leave policy for secondary caregivers?',
            expected_output='Secondary caregivers: 8 weeks paid leave. Requires 6 months employment.'
        ),
        Golden(
            input='When must parental leave be taken?',
            expected_output='Within 12 months of child\'s birth/adoption. Apply 30 days before leave starts.'
        ),
    ]
    
    dataset = EvaluationDataset(goldens=goldens)
    
    for golden in dataset.goldens:
        dataset.add_test_case(
            LLMTestCase(
                input=golden.input,
                expected_output=golden.expected_output,
                actual_output=simulate_LLM_answer(golden.input)
            )
        )
    
    print("\n" + "="*80)
    print("TEST 3: PARENTAL LEAVE POLICIES")
    print("="*80)
    
    evaluate(dataset, [parental_leave_metric])


def test_bereavement_and_personal_leave_golden():
    """
    Test 4: Bereavement and Personal Leave
    Using Golden dataset for bereavement and personal day policies
    """
    
    special_leave_metric = GEval(
        name="Special Leave Policy Accuracy",
        criteria=(
            "Evaluate if the actual output correctly states bereavement leave entitlements "
            "by family relationship and personal day policies. Response should be clear "
            "about what employees are entitled to during difficult times."
        ),
        evaluation_params=["actual_output", "expected_output"],
        threshold=0.7
    )
    
    goldens = [
        Golden(
            input='How much bereavement leave do I get if an immediate family member passes away?',
            expected_output='5 days of paid bereavement leave for immediate family (parent, spouse, child).'
        ),
        Golden(
            input='What is the bereavement leave policy for extended family?',
            expected_output='3 days of paid bereavement leave for extended family (siblings, grandparents).'
        ),
        Golden(
            input='How many personal days do employees receive?',
            expected_output='3 paid personal days per year. Can be used for any reason without explanation.'
        ),
        Golden(
            input='Can I carry over unused personal days to the next year?',
            expected_output='No, personal days cannot be carried over. Must be used within the calendar year.'
        ),
    ]
    
    dataset = EvaluationDataset(goldens=goldens)
    
    for golden in dataset.goldens:
        dataset.add_test_case(
            LLMTestCase(
                input=golden.input,
                expected_output=golden.expected_output,
                actual_output=simulate_LLM_answer(golden.input)
            )
        )
    
    print("\n" + "="*80)
    print("TEST 4: BEREAVEMENT AND PERSONAL LEAVE")
    print("="*80)
    
    evaluate(dataset, [special_leave_metric])


def test_comprehensive_hr_leave_golden():
    """
    Test 5: Comprehensive HR Leave Policy Evaluation
    Tests multiple leave policy scenarios in a single evaluation
    """
    
    hr_policy_metric = GEval(
        name="HR Leave Policy Comprehensive Assessment",
        criteria=(
            "Evaluate if the response provides accurate HR leave policy information including: "
            "1) Correct leave entitlements and numbers, "
            "2) Accurate eligibility requirements and timeframes, "
            "3) Clear explanation of rules and procedures, "
            "4) Compliance with company leave policy. "
            "The response should help employees make informed decisions about leave."
        ),
        evaluation_params=["actual_output", "expected_output"],
        threshold=0.7
    )
    
    # Comprehensive golden dataset covering various leave policy scenarios
    goldens = [
        # Annual Leave
        Golden(
            input='How many annual leave days do full-time employees receive per year?',
            expected_output='Full-time employees receive 15 days of paid annual leave per year, accruing at 1.25 days per month.'
        ),
        Golden(
            input='What is the maximum number of annual leave days I can carry over to the next year?',
            expected_output='Maximum 5 annual leave days can be carried over. Days beyond 5 are forfeited.'
        ),
        
        # Sick Leave
        Golden(
            input='How much sick leave do employees get annually?',
            expected_output='10 days of paid sick leave per year. Does not carry over to next year.'
        ),
        Golden(
            input='When do I need to provide a medical certificate for sick leave?',
            expected_output='Medical certificate required for 3 or more consecutive days of sick leave.'
        ),
        
        # Parental Leave
        Golden(
            input='How many weeks of parental leave do primary caregivers get?',
            expected_output='Primary caregivers: 16 weeks paid leave. Requires 6 months employment.'
        ),
        Golden(
            input='What is the parental leave policy for secondary caregivers?',
            expected_output='Secondary caregivers: 8 weeks paid leave. Requires 6 months employment.'
        ),
        
        # Bereavement Leave
        Golden(
            input='How much bereavement leave do I get if an immediate family member passes away?',
            expected_output='5 days of paid bereavement leave for immediate family (parent, spouse, child).'
        ),
        Golden(
            input='What is the bereavement leave policy for extended family?',
            expected_output='3 days of paid bereavement leave for extended family (siblings, grandparents).'
        ),
        
        # Personal Days
        Golden(
            input='How many personal days do employees receive?',
            expected_output='3 paid personal days per year. Can be used for any reason without explanation.'
        ),
        
        # Accrual Rules
        Golden(
            input='How does leave accrual change with tenure?',
            expected_output='Accrual rates: 15 days/year (0-2 years), 18 days/year (3-5 years), 20 days/year (6+ years).'
        ),
        
        # Leave Request Process
        Golden(
            input='What is the process for requesting annual leave?',
            expected_output='Submit via HR portal 2 weeks in advance. Manager approval required. Confirmation within 3 business days.'
        ),
        Golden(
            input='How do I report sick leave?',
            expected_output='Notify manager ASAP before shift. Submit sick leave form in HR portal upon return.'
        ),
        
        # Special Cases
        Golden(
            input='Can I take leave during my probation period?',
            expected_output='Leave accrues from day 1 but can only be used after completing 90-day probation.'
        ),
        Golden(
            input='Do part-time employees receive the same leave benefits?',
            expected_output='Part-time employees receive prorated leave based on scheduled hours worked.'
        ),
    ]
    
    dataset = EvaluationDataset(goldens=goldens)
    
    print("\n" + "="*80)
    print("GENERATING TEST CASES FROM GOLDEN DATASET")
    print("="*80)
    
    for i, golden in enumerate(dataset.goldens, 1):
        actual = simulate_LLM_answer(golden.input)
        print(f"\n{i}. Input: {golden.input}")
        print(f"   Expected: {golden.expected_output[:80]}...")
        print(f"   Actual: {actual[:80]}...")
        
        dataset.add_test_case(
            LLMTestCase(
                input=golden.input,
                expected_output=golden.expected_output,
                actual_output=actual
            )
        )
    
    print("\n" + "="*80)
    print("TEST 5: COMPREHENSIVE HR LEAVE POLICY EVALUATION")
    print("="*80)
    
    evaluate(dataset, [hr_policy_metric])


def test_hr_policy_with_real_api_integration():
    """
    Test 6: Integration with Real LLM API
    Shows how to use Golden dataset with actual API calls
    """
    
    # You would replace simulate_LLM_answer with real API call
    def get_hr_response_from_api(prompt):
        """
        Replace this with actual LLM API call
        Example: OpenAI, Anthropic, custom HR chatbot, etc.
        """
        # import openai
        # response = openai.chat.completions.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": "You are an HR assistant helping employees understand company leave policies."},
        #         {"role": "user", "content": prompt}
        #     ]
        # )
        # return response.choices[0].message.content
        
        # For demo, using simulated responses
        return simulate_LLM_answer(prompt)
    
    hr_api_metric = GEval(
        name="HR API Response Quality",
        criteria="Evaluate accuracy and completeness of HR policy responses from API. Must be policy-compliant.",
        evaluation_params=["actual_output", "expected_output"],
        threshold=0.7
    )
    
    goldens = [
        Golden(
            input='I need to take time off for a funeral. My grandmother passed away. How many days can I take?',
            expected_output='You can take 3 days of paid bereavement leave for grandparents. No documentation required for up to 3 days. We\'re sorry for your loss.'
        ),
        Golden(
            input='I\'ve been here for 4 years. How much vacation time do I get now?',
            expected_output='With 4 years of service, you receive 18 days of annual leave per year (1.5 days per month). This is the 3-5 year tenure tier.'
        ),
        Golden(
            input='My child is being born next month. What leave am I entitled to as the father?',
            expected_output='As secondary caregiver, you receive 8 weeks of paid parental leave. Must apply 30 days before leave starts. Requires 6 months employment.'
        ),
    ]
    
    dataset = EvaluationDataset(goldens=goldens)
    
    print("\n" + "="*80)
    print("TEST 6: REAL API INTEGRATION EXAMPLE")
    print("="*80)
    
    for golden in dataset.goldens:
        api_response = get_hr_response_from_api(golden.input)
        dataset.add_test_case(
            LLMTestCase(
                input=golden.input,
                expected_output=golden.expected_output,
                actual_output=api_response  # From real API
            )
        )
    
    evaluate(dataset, [hr_api_metric])


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GOLDEN DATASET EXAMPLES - HR LEAVE CREDIT POLICIES")
    print("="*80)
    print("\n📚 WHAT ARE GOLDEN DATASETS?")
    print("   Golden datasets contain input-output pairs that represent")
    print("   the 'gold standard' for correct responses.")
    print("\n🎯 PERFECT FOR:")
    print("   - Regression testing (ensure new models don't break existing answers)")
    print("   - Model comparison (compare GPT-4 vs GPT-3.5 vs Claude)")
    print("   - Performance tracking (measure accuracy over time)")
    print("   - CI/CD integration (automated testing on every deployment)")
    print("\n💼 HR POLICY COVERAGE:")
    print("   ✓ Annual Leave (15 days, carryover rules)")
    print("   ✓ Sick Leave (10 days, medical cert requirements)")
    print("   ✓ Parental Leave (16 weeks primary, 8 weeks secondary)")
    print("   ✓ Bereavement Leave (5 days immediate, 3 days extended)")
    print("   ✓ Personal Days (3 days, no carryover)")
    print("   ✓ Accrual Rules (tenure-based increases)")
    print("   ✓ Request Process (timing, approval, documentation)")
    print("="*80)
    
    # Run all test scenarios
    test_annual_leave_policies_golden()
    test_sick_leave_policies_golden()
    test_parental_leave_policies_golden()
    test_bereavement_and_personal_leave_golden()
    test_comprehensive_hr_leave_golden()
    test_hr_policy_with_real_api_integration()
    
    print("\n" + "="*80)
    print("ALL HR LEAVE POLICY GOLDEN DATASET TESTS COMPLETED")
    print("="*80)
    print("\n💡 NEXT STEPS:")
    print("   1. Replace simulate_LLM_answer() with your actual HR chatbot")
    print("   2. Expand golden dataset with edge cases from real queries")
    print("   3. Run tests after every policy update or model change")
    print("   4. Track scores over time to measure quality trends")
    print("   5. Use for A/B testing different prompt engineering approaches")
    print("="*80)
