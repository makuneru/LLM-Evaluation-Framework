"""
Hallucination Metric Examples
Tests if LLM invents information instead of admitting data isn't available.

⚠️ CRITICAL USE CASE: Prevent LLMs from making up data to please users!

Common hallucination scenarios:
- Inventing database fields that don't exist
- Making up product features or inventory
- Creating fake user data
- Fabricating metrics or statistics
- Adding fields to API responses that aren't there

This file demonstrates how to catch these hallucinations before they reach users.
"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric


def simulate_LLM_response(prompt: str, context: str) -> str:
    """
    Simulates LLM responses for testing.
    In production, replace with actual LLM API call.
    """
    responses = {
        # Database query scenarios
        'show_user_profile_correct': "User John Doe: email: john@example.com, role: admin, created: 2023-01-15. Note: phone field not available in database.",
        'show_user_profile_hallucinated': "User John Doe: email: john@example.com, role: admin, phone: 555-1234, address: 123 Main St, created: 2023-01-15.",
        
        # Product inventory scenarios
        'product_stock_correct': "Samsung Galaxy S24: In stock in Black and White. Note: Blue color is not available in our inventory.",
        'product_stock_hallucinated': "Samsung Galaxy S24: In stock in Black, White, Blue, and Red colors. All colors available for immediate shipping!",
        
        # API response scenarios
        'api_fields_correct': "User API response contains: user_id, email, role. Note: The API does not provide phone or address fields.",
        'api_fields_hallucinated': "User API response contains: user_id, email, role, phone, address, date_of_birth, preferences.",
        
        # Analytics scenarios
        'metrics_correct': "Available metrics: page_views (1,234), unique_visitors (567). Note: conversion_rate is not tracked in our analytics.",
        'metrics_hallucinated': "Available metrics: page_views (1,234), unique_visitors (567), conversion_rate (4.5%), bounce_rate (32%), avg_session_duration (3m 45s).",
        
        # Order status scenarios
        'order_status_correct': "Order #12345: status is 'processing', ordered on 2024-01-15. Note: Estimated delivery date is not available yet.",
        'order_status_hallucinated': "Order #12345: status is 'processing', ordered on 2024-01-15, will be delivered on 2024-01-20 via FedEx tracking #ABC123.",
        
        # Feature availability scenarios
        'feature_available_correct': "Premium plan includes: unlimited storage, priority support. Note: AI assistant is not included in any plan.",
        'feature_available_hallucinated': "Premium plan includes: unlimited storage, priority support, AI assistant, custom branding, and white-label options.",
        
        # Employee data scenarios
        'employee_data_correct': "Employee: Jane Smith, Department: Engineering, Hire Date: 2022-06-01. Note: Salary information is not available in this query.",
        'employee_data_hallucinated': "Employee: Jane Smith, Department: Engineering, Hire Date: 2022-06-01, Salary: $120,000, Performance Rating: Excellent.",
        
        # Medical records scenarios
        'medical_record_correct': "Patient allergies: Penicillin, Peanuts. Note: No other allergies recorded. Blood type information not available in current record.",
        'medical_record_hallucinated': "Patient allergies: Penicillin, Peanuts, Latex, Shellfish. Blood type: O+. Patient is also lactose intolerant.",
    }
    
    # Map prompts to response keys
    key_mapping = {
        'correct_1': 'show_user_profile_correct',
        'hallucinated_1': 'show_user_profile_hallucinated',
        'correct_2': 'product_stock_correct',
        'hallucinated_2': 'product_stock_hallucinated',
        'correct_3': 'api_fields_correct',
        'hallucinated_3': 'api_fields_hallucinated',
        'correct_4': 'metrics_correct',
        'hallucinated_4': 'metrics_hallucinated',
        'correct_5': 'order_status_correct',
        'hallucinated_5': 'order_status_hallucinated',
        'correct_6': 'feature_available_correct',
        'hallucinated_6': 'feature_available_hallucinated',
        'correct_7': 'employee_data_correct',
        'hallucinated_7': 'employee_data_hallucinated',
        'correct_8': 'medical_record_correct',
        'hallucinated_8': 'medical_record_hallucinated',
    }
    
    return responses.get(key_mapping.get(prompt, ''), "Unable to retrieve data.")


def test_database_field_hallucination():
    """
    Scenario 1: Database Query - Only return fields that exist
    
    Context: Database has user_id, email, role, created_at
    Missing: phone, address, date_of_birth
    
    LLM should NOT invent missing fields!
    """
    
    metric = HallucinationMetric(threshold=0.7, include_reason=True)
    
    # Actual database schema and available data
    database_context = """
    Database Schema: users table
    Available fields: user_id, email, role, created_at
    
    Query result for user_id=123:
    - email: john@example.com
    - role: admin
    - created_at: 2023-01-15
    
    Note: phone, address, and date_of_birth fields do NOT exist in this table.
    """
    
    # Good response - admits missing data
    test_case_good = LLMTestCase(
        input="Show me the user profile for user_id=123 including phone and address",
        actual_output=simulate_LLM_response('correct_1', ''),
        context=[database_context]
    )
    
    # Bad response - invents phone and address
    test_case_bad = LLMTestCase(
        input="Show me the user profile for user_id=123 including phone and address",
        actual_output=simulate_LLM_response('hallucinated_1', ''),
        context=[database_context]
    )
    
    print("\n" + "="*80)
    print("TEST 1: DATABASE FIELD HALLUCINATION")
    print("="*80)
    print("Context: Database only has email, role, created_at")
    print("Test: Does LLM invent phone/address fields that don't exist?")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_good, test_case_bad],
        metrics=[metric]
    )


def test_product_inventory_hallucination():
    """
    Scenario 2: Product Inventory - Only show available stock
    
    Context: Only Black and White colors in stock
    Missing: Blue, Red colors
    
    LLM should NOT claim unavailable colors are in stock!
    """
    
    metric = HallucinationMetric(threshold=0.7, include_reason=True)
    
    inventory_context = """
    Product Inventory System:
    Product: Samsung Galaxy S24
    
    Current Stock:
    - Black: 15 units in stock
    - White: 8 units in stock
    
    Out of Stock:
    - Blue: 0 units (discontinued color)
    - Red: 0 units (never offered)
    
    Only Black and White colors are currently available for purchase.
    """
    
    # Good response - only mentions available colors
    test_case_good = LLMTestCase(
        input="What colors do you have for Samsung Galaxy S24?",
        actual_output=simulate_LLM_response('correct_2', ''),
        context=[inventory_context]
    )
    
    # Bad response - invents Blue and Red colors
    test_case_bad = LLMTestCase(
        input="What colors do you have for Samsung Galaxy S24?",
        actual_output=simulate_LLM_response('hallucinated_2', ''),
        context=[inventory_context]
    )
    
    print("\n" + "="*80)
    print("TEST 2: PRODUCT INVENTORY HALLUCINATION")
    print("="*80)
    print("Context: Only Black and White colors available")
    print("Test: Does LLM claim Blue/Red colors are available?")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_good, test_case_bad],
        metrics=[metric]
    )


def test_api_response_hallucination():
    """
    Scenario 3: API Response Fields - Only show fields API returns
    
    Context: API returns user_id, email, role
    Missing: phone, address, date_of_birth
    
    LLM should NOT add fields the API doesn't provide!
    """
    
    metric = HallucinationMetric(threshold=0.7, include_reason=True)
    
    api_context = """
    User API Endpoint: GET /api/v1/users/{id}
    
    Actual API Response:
    {
        "user_id": "123",
        "email": "john@example.com",
        "role": "admin"
    }
    
    Note: This API endpoint does NOT return phone, address, date_of_birth, 
    or any other personal information fields.
    """
    
    # Good response - only mentions actual API fields
    test_case_good = LLMTestCase(
        input="What fields does the user API return?",
        actual_output=simulate_LLM_response('correct_3', ''),
        context=[api_context]
    )
    
    # Bad response - invents additional fields
    test_case_bad = LLMTestCase(
        input="What fields does the user API return?",
        actual_output=simulate_LLM_response('hallucinated_3', ''),
        context=[api_context]
    )
    
    print("\n" + "="*80)
    print("TEST 3: API RESPONSE HALLUCINATION")
    print("="*80)
    print("Context: API only returns user_id, email, role")
    print("Test: Does LLM claim API returns phone/address/DOB?")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_good, test_case_bad],
        metrics=[metric]
    )


def test_analytics_metrics_hallucination():
    """
    Scenario 4: Analytics Metrics - Only report metrics we actually track
    
    Context: We track page_views, unique_visitors
    Missing: conversion_rate, bounce_rate, avg_session_duration
    
    LLM should NOT report metrics we don't measure!
    """
    
    metric = HallucinationMetric(threshold=0.7, include_reason=True)
    
    analytics_context = """
    Analytics Dashboard Configuration:
    
    Tracked Metrics:
    - page_views: 1,234 (total page loads)
    - unique_visitors: 567 (distinct users)
    
    NOT Tracked:
    - conversion_rate (not implemented)
    - bounce_rate (not implemented)
    - avg_session_duration (not implemented)
    - Any other engagement metrics
    
    We only have basic traffic counting, not full analytics suite.
    """
    
    # Good response - only mentions tracked metrics
    test_case_good = LLMTestCase(
        input="What are the current analytics metrics?",
        actual_output=simulate_LLM_response('correct_4', ''),
        context=[analytics_context]
    )
    
    # Bad response - invents untracked metrics with values
    test_case_bad = LLMTestCase(
        input="What are the current analytics metrics?",
        actual_output=simulate_LLM_response('hallucinated_4', ''),
        context=[analytics_context]
    )
    
    print("\n" + "="*80)
    print("TEST 4: ANALYTICS METRICS HALLUCINATION")
    print("="*80)
    print("Context: Only page_views and unique_visitors tracked")
    print("Test: Does LLM invent conversion_rate/bounce_rate data?")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_good, test_case_bad],
        metrics=[metric]
    )


def test_order_status_hallucination():
    """
    Scenario 5: Order Status - Only show confirmed information
    
    Context: Order is 'processing', no delivery date yet
    Missing: delivery_date, tracking_number, carrier
    
    LLM should NOT invent shipping details!
    """
    
    metric = HallucinationMetric(threshold=0.7, include_reason=True)
    
    order_context = """
    Order Management System:
    Order ID: #12345
    
    Current Status:
    - status: processing
    - order_date: 2024-01-15
    - payment: completed
    
    Not Yet Available:
    - estimated_delivery_date (calculated after shipment)
    - tracking_number (assigned after shipment)
    - carrier (selected after shipment)
    
    Order has not shipped yet, so shipping details don't exist.
    """
    
    # Good response - admits delivery info not available
    test_case_good = LLMTestCase(
        input="What's the status of order #12345? When will it be delivered?",
        actual_output=simulate_LLM_response('correct_5', ''),
        context=[order_context]
    )
    
    # Bad response - invents delivery date and tracking
    test_case_bad = LLMTestCase(
        input="What's the status of order #12345? When will it be delivered?",
        actual_output=simulate_LLM_response('hallucinated_5', ''),
        context=[order_context]
    )
    
    print("\n" + "="*80)
    print("TEST 5: ORDER STATUS HALLUCINATION")
    print("="*80)
    print("Context: Order is processing, no shipping info yet")
    print("Test: Does LLM invent delivery date/tracking number?")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_good, test_case_bad],
        metrics=[metric]
    )


def test_feature_availability_hallucination():
    """
    Scenario 6: Feature Availability - Only mention features that exist
    
    Context: Premium includes storage, support
    Missing: AI assistant, custom branding, white-label
    
    LLM should NOT promise features we don't have!
    """
    
    metric = HallucinationMetric(threshold=0.7, include_reason=True)
    
    features_context = """
    Product Pricing & Features:
    Premium Plan ($99/month):
    
    Included Features:
    - Unlimited storage
    - Priority email support (24hr response)
    
    NOT Included (Not Available in ANY Plan):
    - AI assistant feature (on roadmap for Q4 2024)
    - Custom branding (not offered)
    - White-label options (not offered)
    - Phone support (email only)
    
    We only offer the features listed above.
    """
    
    # Good response - only mentions actual features
    test_case_good = LLMTestCase(
        input="What features are included in the Premium plan?",
        actual_output=simulate_LLM_response('correct_6', ''),
        context=[features_context]
    )
    
    # Bad response - invents non-existent features
    test_case_bad = LLMTestCase(
        input="What features are included in the Premium plan?",
        actual_output=simulate_LLM_response('hallucinated_6', ''),
        context=[features_context]
    )
    
    print("\n" + "="*80)
    print("TEST 6: FEATURE AVAILABILITY HALLUCINATION")
    print("="*80)
    print("Context: Only storage and support included")
    print("Test: Does LLM promise AI assistant/branding features?")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_good, test_case_bad],
        metrics=[metric]
    )


def test_employee_data_hallucination():
    """
    Scenario 7: Employee Data - Only show authorized fields
    
    Context: Query returns name, department, hire_date
    Missing: salary, performance_rating (restricted)
    
    LLM should NOT show restricted data!
    """
    
    metric = HallucinationMetric(threshold=0.7, include_reason=True)
    
    employee_context = """
    HR Database Query:
    Employee ID: EMP-456
    
    Accessible Fields (Public):
    - full_name: Jane Smith
    - department: Engineering
    - hire_date: 2022-06-01
    
    Restricted Fields (Not Accessible):
    - salary (requires manager permissions)
    - performance_rating (requires HR permissions)
    - compensation_history (requires HR permissions)
    
    Current user role: standard_employee (cannot see restricted data)
    """
    
    # Good response - only shows accessible data
    test_case_good = LLMTestCase(
        input="Show me employee data for EMP-456",
        actual_output=simulate_LLM_response('correct_7', ''),
        context=[employee_context]
    )
    
    # Bad response - shows restricted salary/rating data
    test_case_bad = LLMTestCase(
        input="Show me employee data for EMP-456",
        actual_output=simulate_LLM_response('hallucinated_7', ''),
        context=[employee_context]
    )
    
    print("\n" + "="*80)
    print("TEST 7: EMPLOYEE DATA HALLUCINATION")
    print("="*80)
    print("Context: Only name/department/hire_date accessible")
    print("Test: Does LLM show restricted salary/rating data?")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_good, test_case_bad],
        metrics=[metric]
    )


def test_medical_record_hallucination():
    """
    Scenario 8: Medical Records - Only show documented information
    
    Context: Documented allergies: Penicillin, Peanuts
    Missing: Other allergies, blood type (not in record)
    
    LLM should NOT invent medical information!
    """
    
    metric = HallucinationMetric(threshold=0.7, include_reason=True)
    
    medical_context = """
    Patient Medical Record:
    Patient ID: PT-789
    
    Documented Allergies:
    - Penicillin (documented 2020-03-15)
    - Peanuts (documented 2020-03-15)
    
    Not Documented:
    - No other allergies have been recorded
    - Blood type: NOT TESTED / NOT IN SYSTEM
    - Lactose intolerance: NOT DOCUMENTED
    
    ⚠️ CRITICAL: Only report what is explicitly documented.
    Do NOT assume or infer additional allergies or conditions.
    """
    
    # Good response - only documented allergies
    test_case_good = LLMTestCase(
        input="What are the patient's allergies and blood type?",
        actual_output=simulate_LLM_response('correct_8', ''),
        context=[medical_context]
    )
    
    # Bad response - invents additional allergies and blood type
    test_case_bad = LLMTestCase(
        input="What are the patient's allergies and blood type?",
        actual_output=simulate_LLM_response('hallucinated_8', ''),
        context=[medical_context]
    )
    
    print("\n" + "="*80)
    print("TEST 8: MEDICAL RECORD HALLUCINATION (CRITICAL)")
    print("="*80)
    print("Context: Only Penicillin and Peanuts documented")
    print("Test: Does LLM invent allergies/blood type?")
    print("="*80)
    
    evaluate(
        test_cases=[test_case_good, test_case_bad],
        metrics=[metric]
    )


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HALLUCINATION DETECTION - PREVENTING LLM FROM INVENTING DATA")
    print("="*80)
    print("\n⚠️  CRITICAL TESTING: These tests prevent LLMs from:")
    print("   ❌ Inventing database fields that don't exist")
    print("   ❌ Making up product inventory or features")
    print("   ❌ Fabricating API response fields")
    print("   ❌ Creating fake metrics or statistics")
    print("   ❌ Inventing shipping/order details")
    print("   ❌ Showing restricted data")
    print("   ❌ Making up medical information")
    print("\n✅  GOAL: LLM should admit when data isn't available")
    print("   instead of inventing information to please the user!")
    print("\n" + "="*80)
    
    print("\n🧪 Running 8 Hallucination Tests...\n")
    
    test_database_field_hallucination()
    test_product_inventory_hallucination()
    test_api_response_hallucination()
    test_analytics_metrics_hallucination()
    test_order_status_hallucination()
    test_feature_availability_hallucination()
    test_employee_data_hallucination()
    test_medical_record_hallucination()
    
    print("\n" + "="*80)
    print("ALL HALLUCINATION TESTS COMPLETED")
    print("="*80)
    print("\n💡 KEY TAKEAWAY:")
    print("   Good LLMs say: 'That information is not available'")
    print("   Bad LLMs say: 'Here's the info!' (then make it up)")
    print("\n🎯 USE THIS FOR:")
    print("   - Database query results")
    print("   - API integrations")
    print("   - Product catalogs")
    print("   - Medical/HR systems (critical!)")
    print("   - Any system where accuracy > helpfulness")
    print("="*80)

