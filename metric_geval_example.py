from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval


def test_math_calculation_correctness():
    """Test LLM's ability to perform and explain mathematical calculations."""
    correctness_metric = GEval(
        name='Math Correctness',
        criteria=(
            "Evaluate if the actual output contains the correct numerical answer "
            "and reasoning compared to the expected output. The format can vary, "
            "but the mathematical result and logic must be accurate."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.7
    )
    
    test_case = LLMTestCase(
        input='What is 5 divided by 2?',
        expected_output='2.5',
        actual_output='The result is 2.5'
    )
    
    assert_test(test_case, [correctness_metric])


def test_factual_knowledge_correctness():
    """Test LLM's factual knowledge about world events and history."""
    correctness_metric = GEval(
        name='Factual Correctness',
        criteria=(
            "Verify if the actual output provides factually accurate information "
            "that matches the expected output. Key facts, dates, and names must be correct. "
            "Minor wording differences are acceptable as long as the core facts are accurate."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.8
    )
    
    test_case = LLMTestCase(
        input='What is the capital of France?',
        expected_output='The capital of France is Paris.',
        actual_output='Paris is the capital city of France, located in the north-central part of the country.'
    )
    
    assert_test(test_case, [correctness_metric])


def test_technical_code_explanation_correctness():
    """Test LLM's ability to explain technical concepts and code."""
    correctness_metric = GEval(
        name='Technical Explanation Correctness',
        criteria=(
            "Assess if the actual output correctly explains the technical concept "
            "with the same key points as the expected output. The explanation must be "
            "technically accurate, though it may include additional helpful context."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.75
    )
    
    test_case = LLMTestCase(
        input='What does the HTTP status code 404 mean?',
        expected_output='404 means "Not Found" - the requested resource could not be found on the server.',
        actual_output=(
            'HTTP status code 404 indicates that the server cannot find the requested resource. '
            'This is commonly known as a "Not Found" error and typically means the URL is incorrect '
            'or the resource has been moved or deleted.'
        )
    )
    
    assert_test(test_case, [correctness_metric])


def test_summarization_correctness():
    """Test LLM's ability to summarize information accurately."""
    correctness_metric = GEval(
        name='Summarization Correctness',
        criteria=(
            "Determine if the actual output captures the essential information "
            "and main points from the expected output. The summary should be accurate "
            "and complete, even if phrased differently."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.7
    )
    
    test_case = LLMTestCase(
        input='Summarize the benefits of using automated testing in software development.',
        expected_output=(
            'Automated testing provides faster feedback, improves code quality, '
            'reduces manual testing effort, enables continuous integration, and catches bugs early.'
        ),
        actual_output=(
            'The main benefits of automated testing include: rapid feedback on code changes, '
            'higher code quality through consistent testing, significant reduction in manual QA work, '
            'seamless integration with CI/CD pipelines, and early detection of defects before production.'
        )
    )
    
    assert_test(test_case, [correctness_metric])


def test_data_interpretation_correctness():
    """Test LLM's ability to interpret and analyze data correctly."""
    correctness_metric = GEval(
        name='Data Analysis Correctness',
        criteria=(
            "Evaluate if the actual output provides the correct interpretation and analysis "
            "of the data as shown in the expected output. The conclusion and key insights "
            "must be accurate, though the explanation style may vary."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.8
    )
    
    test_case = LLMTestCase(
        input=(
            'Given the following data: Q1 sales=$100K, Q2 sales=$120K, Q3 sales=$150K, Q4 sales=$180K. '
            'What is the total annual sales and the growth trend?'
        ),
        expected_output=(
            'Total annual sales: $550K. The growth trend is positive and accelerating, '
            'with sales increasing each quarter (20% Q1-Q2, 25% Q2-Q3, 20% Q3-Q4).'
        ),
        actual_output=(
            'The total annual sales amount to $550,000. There is a consistent upward trend '
            'throughout the year, with quarterly growth rates of 20%, 25%, and 20% respectively, '
            'indicating strong and accelerating business performance.'
        )
    )
    
    assert_test(test_case, [correctness_metric])

