"""
Faithfulness Metric Examples - HR Scenarios
This demonstrates how to evaluate if LLM responses are faithful to (grounded in) 
the provided context/documents. Essential for RAG systems and HR chatbots.
"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric


def test_employee_benefits_policy():
    """
    HR Benefits: Testing if responses are faithful to company policy documents
    """
    
    metric = FaithfulnessMetric(
        threshold=0.7,
        include_reason=True
    )
    
    # Actual company policy document (retrieval context)
    policy_context = """
    Company Parental Leave Policy:
    - Primary caregivers are eligible for 16 weeks of paid parental leave
    - Secondary caregivers are eligible for 8 weeks of paid parental leave
    - Leave must be taken within 12 months of the child's birth or adoption
    - Employees must have completed at least 6 months of employment to be eligible
    - Leave can be taken continuously or intermittently with manager approval
    - Benefits continue during parental leave period
    """
    
    # Example 1: Faithful response - accurately reflects the policy
    test_case_faithful = LLMTestCase(
        input='How many weeks of parental leave do I get as a primary caregiver?',
        actual_output='As a primary caregiver, you are eligible for 16 weeks of paid parental leave. This leave must be taken within 12 months of your child\'s birth or adoption, and you need to have completed at least 6 months of employment to be eligible.',
        retrieval_context=[policy_context]
    )
    
    # Example 2: Partially faithful - adds unsupported information
    test_case_partial = LLMTestCase(
        input='How many weeks of parental leave do I get as a primary caregiver?',
        actual_output='You get 16 weeks of paid parental leave. You can also apply for an additional 4 weeks of unpaid leave if needed. The company also provides a $1,000 baby bonus.',
        retrieval_context=[policy_context]
    )
    
    # Example 3: Unfaithful - contradicts the policy
    test_case_unfaithful = LLMTestCase(
        input='How many weeks of parental leave do I get as a primary caregiver?',
        actual_output='Primary caregivers get 12 weeks of paid leave. You must take it all at once and cannot split it up. There is no minimum employment requirement.',
        retrieval_context=[policy_context]
    )
    
    print("\n" + "="*80)
    print("HR BENEFITS: Parental Leave Policy")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_faithful, test_case_partial, test_case_unfaithful],
        metrics=[metric]
    )


def test_pto_accrual_information():
    """
    HR PTO: Testing faithfulness to vacation/PTO accrual rules
    """
    
    metric = FaithfulnessMetric(
        threshold=0.7,
        include_reason=True
    )
    
    pto_policy = """
    Paid Time Off (PTO) Accrual Policy:
    - Employees with 0-2 years of service: 15 days per year (1.25 days per month)
    - Employees with 3-5 years of service: 20 days per year (1.67 days per month)
    - Employees with 6+ years of service: 25 days per year (2.08 days per month)
    - PTO accrues on the first day of each month
    - Maximum carryover: 40 days (days above this are forfeited)
    - PTO requests must be submitted at least 2 weeks in advance
    - Unused PTO is paid out upon termination
    """
    
    # Faithful response
    test_case_faithful = LLMTestCase(
        input='I have been with the company for 4 years. How much PTO do I get?',
        actual_output='With 4 years of service, you fall into the 3-5 years category and receive 20 days of PTO per year, which accrues at 1.67 days per month. PTO accrues on the first day of each month.',
        retrieval_context=[pto_policy]
    )
    
    # Contains hallucination
    test_case_unfaithful = LLMTestCase(
        input='I have been with the company for 4 years. How much PTO do I get?',
        actual_output='With 4 years of service, you receive 18 days of PTO per year. You can also take 5 additional personal days. PTO cannot be carried over to the next year.',
        retrieval_context=[pto_policy]
    )
    
    print("\n" + "="*80)
    print("HR PTO: Vacation Accrual Policy")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_faithful, test_case_unfaithful],
        metrics=[metric]
    )


def test_performance_review_process():
    """
    HR Performance: Testing faithfulness to performance review procedures
    """
    
    metric = FaithfulnessMetric(
        threshold=0.7,
        include_reason=True
    )
    
    review_process = """
    Annual Performance Review Process:
    - Reviews are conducted annually in January
    - Employees complete a self-assessment by January 10th
    - Managers complete evaluations by January 25th
    - Review meetings occur in the first week of February
    - Ratings are on a 5-point scale: Outstanding (5), Exceeds Expectations (4), 
      Meets Expectations (3), Needs Improvement (2), Unsatisfactory (1)
    - Merit increases are based on performance rating and budget availability
    - Typical merit increases: Outstanding 5-7%, Exceeds 3-5%, Meets 0-3%
    - Promotions are considered separately during mid-year review cycle
    """
    
    # Faithful response
    test_case_faithful = LLMTestCase(
        input='When do performance reviews happen and what are the rating levels?',
        actual_output='Performance reviews are conducted annually in January. You will complete a self-assessment by January 10th, your manager completes the evaluation by January 25th, and review meetings occur in the first week of February. The rating scale is: Outstanding (5), Exceeds Expectations (4), Meets Expectations (3), Needs Improvement (2), and Unsatisfactory (1).',
        retrieval_context=[review_process]
    )
    
    # Partially faithful with added incorrect info
    test_case_partial = LLMTestCase(
        input='When do performance reviews happen and what are the rating levels?',
        actual_output='Reviews happen in January and June. The ratings are Excellent, Good, Average, and Poor. You can also request a review at any time if you feel your performance has significantly improved.',
        retrieval_context=[review_process]
    )
    
    print("\n" + "="*80)
    print("HR PERFORMANCE: Annual Review Process")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_faithful, test_case_partial],
        metrics=[metric]
    )


def test_remote_work_policy():
    """
    HR Remote Work: Testing faithfulness to remote work guidelines
    """
    
    metric = FaithfulnessMetric(
        threshold=0.7,
        include_reason=True
    )
    
    remote_policy = """
    Remote Work Policy:
    - Eligible employees can work remotely up to 3 days per week
    - Tuesdays and Thursdays are designated in-office days for all teams
    - Remote work requires manager approval and a signed remote work agreement
    - Employees must maintain their primary residence within 50 miles of the office
    - Home office must have reliable internet (minimum 25 Mbps download, 5 Mbps upload)
    - Company provides a $500 annual stipend for home office equipment
    - Employees must be available during core business hours (9 AM - 3 PM local time)
    - Remote work privileges can be revoked if performance or communication issues arise
    """
    
    # Faithful response
    test_case_faithful = LLMTestCase(
        input='What are the requirements for working remotely?',
        actual_output='To work remotely, you need manager approval and a signed remote work agreement. You can work remotely up to 3 days per week, but must be in the office on Tuesdays and Thursdays. You must live within 50 miles of the office, have reliable internet (minimum 25 Mbps download, 5 Mbps upload), and be available during core hours of 9 AM to 3 PM.',
        retrieval_context=[remote_policy]
    )
    
    # Unfaithful with contradictions
    test_case_unfaithful = LLMTestCase(
        input='What are the requirements for working remotely?',
        actual_output='You can work fully remote from anywhere in the country. No manager approval is needed. The company provides a $1,000 equipment stipend annually. You just need to attend weekly team meetings on Mondays.',
        retrieval_context=[remote_policy]
    )
    
    print("\n" + "="*80)
    print("HR REMOTE WORK: Policy Guidelines")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_faithful, test_case_unfaithful],
        metrics=[metric]
    )


def test_health_insurance_enrollment():
    """
    HR Benefits: Testing faithfulness to health insurance information
    """
    
    metric = FaithfulnessMetric(
        threshold=0.7,
        include_reason=True
    )
    
    insurance_info = """
    Health Insurance Enrollment Information:
    - New employees can enroll within 30 days of hire date
    - Coverage begins on the first day of the month following enrollment
    - Three plan options: Basic PPO ($100/month), Standard PPO ($200/month), Premium PPO ($300/month)
    - All plans include dental and vision coverage at no additional cost
    - Deductibles: Basic $3,000, Standard $1,500, Premium $500
    - Company contributes 75% of Basic plan premium
    - Open enrollment period: November 1-30 annually
    - Qualifying life events allow mid-year changes: marriage, birth, adoption, divorce
    - Dependents can be added: spouse and children under 26
    """
    
    # Faithful response
    test_case_faithful = LLMTestCase(
        input='When can I enroll in health insurance and what plans are available?',
        actual_output='New employees can enroll within 30 days of their hire date, and coverage begins on the first day of the month following enrollment. There are three PPO plan options: Basic ($100/month), Standard ($200/month), and Premium ($300/month). All plans include dental and vision coverage. The company contributes 75% of the Basic plan premium.',
        retrieval_context=[insurance_info]
    )
    
    # Contains fabricated details
    test_case_unfaithful = LLMTestCase(
        input='When can I enroll in health insurance and what plans are available?',
        actual_output='You can enroll at any time during the year. We offer four plans including an HMO option starting at $50/month. Dental and vision are separate add-ons at $25 each. The company pays 100% of your premium. Coverage starts immediately upon enrollment.',
        retrieval_context=[insurance_info]
    )
    
    print("\n" + "="*80)
    print("HR BENEFITS: Health Insurance Enrollment")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_faithful, test_case_unfaithful],
        metrics=[metric]
    )


def test_employee_referral_program():
    """
    HR Recruiting: Testing faithfulness to employee referral bonus details
    """
    
    metric = FaithfulnessMetric(
        threshold=0.7,
        include_reason=True
    )
    
    referral_program = """
    Employee Referral Program:
    - Employees can refer candidates for any open position
    - Referral bonus: $2,000 for engineering roles, $1,000 for all other roles
    - Bonus is paid 50% after candidate completes 90 days, 50% after 6 months
    - Referred candidate must list employee as referral source on application
    - Referral must be submitted before candidate applies through other channels
    - Managers cannot refer candidates for their own team
    - Maximum 5 referral bonuses per employee per calendar year
    - Employees on performance improvement plans are not eligible to participate
    - If multiple employees refer same candidate, first referral receives bonus
    """
    
    # Faithful response
    test_case_faithful = LLMTestCase(
        input='How much is the referral bonus and when do I get paid?',
        actual_output='The referral bonus is $2,000 for engineering roles and $1,000 for all other roles. The bonus is paid in two installments: 50% after the candidate completes 90 days of employment, and the remaining 50% after they complete 6 months.',
        retrieval_context=[referral_program]
    )
    
    # Incorrect information
    test_case_unfaithful = LLMTestCase(
        input='How much is the referral bonus and when do I get paid?',
        actual_output='All referrals receive a $5,000 bonus paid in full after the candidate completes 30 days. You can also get an additional $1,000 if your referral stays for a full year.',
        retrieval_context=[referral_program]
    )
    
    print("\n" + "="*80)
    print("HR RECRUITING: Employee Referral Program")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_faithful, test_case_unfaithful],
        metrics=[metric]
    )


def test_professional_development_budget():
    """
    HR Learning: Testing faithfulness to training and development budget policy
    """
    
    metric = FaithfulnessMetric(
        threshold=0.7,
        include_reason=True
    )
    
    development_policy = """
    Professional Development Budget Policy:
    - Each employee receives $2,000 annual budget for professional development
    - Budget can be used for: conferences, courses, certifications, books, online training
    - Expenses require pre-approval from manager
    - Budget resets January 1st and does not roll over
    - For expenses over $500, submit request at least 30 days in advance
    - Certifications directly related to role may qualify for additional budget
    - Employees must share learnings with team upon completion
    - If employee leaves within 12 months of training over $1,000, may need to reimburse
    - Managers have discretion to approve exceeding budget based on business need
    """
    
    # Faithful response
    test_case_faithful = LLMTestCase(
        input='How much professional development budget do I have and what can I use it for?',
        actual_output='You have a $2,000 annual budget for professional development. You can use it for conferences, courses, certifications, books, and online training. All expenses require pre-approval from your manager. For expenses over $500, you need to submit a request at least 30 days in advance. The budget resets January 1st and doesn\'t roll over.',
        retrieval_context=[development_policy]
    )
    
    # Adds unsupported information
    test_case_partial = LLMTestCase(
        input='How much professional development budget do I have and what can I use it for?',
        actual_output='You have $3,000 for professional development which can be used for training, plus an additional $1,000 for conferences. Unused budget can be rolled over to next year. Manager approval is only needed for expenses over $1,000.',
        retrieval_context=[development_policy]
    )
    
    print("\n" + "="*80)
    print("HR LEARNING: Professional Development Budget")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_faithful, test_case_partial],
        metrics=[metric]
    )


def test_bereavement_leave_policy():
    """
    HR Leave: Testing faithfulness to bereavement leave entitlement
    """
    
    metric = FaithfulnessMetric(
        threshold=0.7,
        include_reason=True
    )
    
    bereavement_policy = """
    Bereavement Leave Policy:
    - Immediate family (spouse, children, parents, siblings): 5 days paid leave
    - Extended family (grandparents, in-laws, aunts, uncles): 3 days paid leave
    - Other relatives or close friends: 1 day paid leave, manager discretion for additional unpaid
    - Leave must be taken within 30 days of the death
    - Additional unpaid leave can be requested through HR
    - Employee may use PTO for additional time if needed
    - No documentation required for immediate family
    - Documentation may be requested for extended family leave
    - Part-time employees receive prorated bereavement leave
    """
    
    # Faithful response
    test_case_faithful = LLMTestCase(
        input='How much bereavement leave do I get if my father passes away?',
        actual_output='For the loss of an immediate family member like your father, you are entitled to 5 days of paid bereavement leave. This leave must be taken within 30 days of the death. No documentation is required for immediate family. If you need additional time, you can request unpaid leave through HR or use your PTO.',
        retrieval_context=[bereavement_policy]
    )
    
    # Incorrect information
    test_case_unfaithful = LLMTestCase(
        input='How much bereavement leave do I get if my father passes away?',
        actual_output='You get 10 days of paid leave for the loss of a parent. You must provide a death certificate to HR within 5 days. The leave must be taken immediately and cannot be split up.',
        retrieval_context=[bereavement_policy]
    )
    
    print("\n" + "="*80)
    print("HR LEAVE: Bereavement Leave Policy")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_faithful, test_case_unfaithful],
        metrics=[metric]
    )


if __name__ == "__main__":
    print("\n" + "="*80)
    print("FAITHFULNESS METRIC - HR POLICY SCENARIOS")
    print("="*80)
    print("\nThis test suite evaluates if LLM responses are faithful to (grounded in)")
    print("the provided HR policy documents and context.")
    print("\nFaithfulness ensures the model doesn't hallucinate or contradict")
    print("the source information - critical for HR chatbots and RAG systems.")
    print("\n" + "="*80)
    
    # Run all test scenarios
    test_employee_benefits_policy()
    test_pto_accrual_information()
    test_performance_review_process()
    test_remote_work_policy()
    test_health_insurance_enrollment()
    test_employee_referral_program()
    test_professional_development_budget()
    test_bereavement_leave_policy()
    
    print("\n" + "="*80)
    print("ALL FAITHFULNESS TESTS COMPLETED")
    print("="*80)

