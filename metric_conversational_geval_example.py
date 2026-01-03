"""
Conversational LLM Evaluation using DeepEval's Conversational Metrics
This file demonstrates multi-turn conversation testing for real-world business scenarios.
"""

from deepeval import evaluate
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics import ConversationalGEval


def test_professionalism():
    """Test if the assistant maintains professionalism across different business contexts."""
    
    professionalism_metric = ConversationalGEval(
        name='Professionalism',
        criteria="Determine whether the assistant answered the questions of the user in a professional and polite manner."
    )
    
    # HR Scenario 1: Employee Benefits Inquiry
    hr_benefits_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "What's our company's parental leave policy?"},
            {"role": "assistant", "content": "Our parental leave policy provides 12 weeks of paid leave for primary caregivers and 6 weeks for secondary caregivers. Would you like more details about eligibility?"},
            {"role": "user", "content": "Do I need to apply in advance?"},
            {"role": "assistant", "content": "Yes, we recommend submitting your parental leave request at least 30 days before your intended start date. I can help you with the application process if you'd like."},
            {"role": "user", "content": "What if there are complications during pregnancy?"},
            {"role": "assistant", "content": "In case of medical complications, additional medical leave may be available through short-term disability. I recommend speaking with HR to discuss your specific situation confidentially."}
        ]
    )
    
    # HR Scenario 2: Performance Review Discussion
    hr_performance_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "I don't understand why my performance rating was lower this quarter."},
            {"role": "assistant", "content": "I understand your concern. Performance ratings are based on multiple factors including goal achievement, collaboration, and project outcomes. Would you like to review the specific areas that impacted your rating?"},
            {"role": "user", "content": "Yes, I thought I did everything that was asked of me."},
            {"role": "assistant", "content": "Let's review your goals together. While you completed the assigned tasks, the rating also considers initiative, quality of work, and alignment with team objectives. I can help you create an improvement plan for next quarter."},
            {"role": "user", "content": "This seems unfair. Others got better ratings for less work."},
            {"role": "assistant", "content": "I appreciate you sharing your perspective. Each employee's rating is confidential and based on their individual goals. If you'd like to discuss this further, I encourage you to schedule a meeting with your manager or HR for a detailed review."}
        ]
    )
    
    # HR Scenario 3: Workplace Conflict Resolution
    hr_conflict_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "My coworker keeps taking credit for my ideas in meetings."},
            {"role": "assistant", "content": "Thank you for bringing this to my attention. Workplace collaboration and proper attribution are important. Can you provide specific examples of when this occurred?"},
            {"role": "user", "content": "Last week's presentation had my entire proposal but they presented it as their own."},
            {"role": "assistant", "content": "I understand how frustrating that must be. Have you had a chance to speak directly with your coworker about this? Sometimes these situations can be resolved through open communication."},
            {"role": "user", "content": "I tried but they denied it. What should I do?"},
            {"role": "assistant", "content": "I recommend documenting your contributions and scheduling a meeting with your manager to discuss this professionally. HR is also available for mediation if needed. Would you like guidance on how to approach this conversation?"}
        ]
    )
    
    # ERP Scenario 1: Inventory Management Query
    erp_inventory_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "Why is product SKU-2847 showing out of stock when we received a shipment yesterday?"},
            {"role": "assistant", "content": "Thank you for bringing this to my attention. Let me check the system. It appears the receiving process was completed but the inventory wasn't updated in real-time. This can happen if the goods receipt wasn't posted properly."},
            {"role": "user", "content": "How do I fix this? We need to ship orders today."},
            {"role": "assistant", "content": "I can help you resolve this immediately. First, verify that the goods receipt document (MIGO transaction) was saved. Then, check if the material document was created. If everything is confirmed, we may need to run the inventory reconciliation batch job."},
            {"role": "user", "content": "I found the material document. What's next?"},
            {"role": "assistant", "content": "Excellent. Please navigate to the Stock Overview (MMBE) and refresh the data. If the stock still doesn't reflect correctly, I'll create a ticket for the ERP support team to investigate potential posting errors. In the meantime, you can manually verify available quantity."}
        ]
    )
    
    # ERP Scenario 2: Purchase Order Approval
    erp_purchase_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "My purchase order PO-89234 is stuck in approval for 3 days. We need these parts urgently."},
            {"role": "assistant", "content": "I understand the urgency. Let me check the approval workflow for PO-89234. The order is currently pending approval from the Finance Director due to the amount exceeding $50,000. Would you like me to send a reminder to the approver?"},
            {"role": "user", "content": "Yes, and is there a way to expedite this? Production is delayed."},
            {"role": "assistant", "content": "I'll send an urgent notification to the Finance Director immediately. For future reference, purchase orders over $50,000 typically require 5 business days for approval. You might also contact the Finance Director directly given the production impact."},
            {"role": "user", "content": "Can we split the PO to get partial approval faster?"},
            {"role": "assistant", "content": "That's a good thought. However, splitting an existing PO requires canceling and recreating orders, which could cause further delays. I recommend proceeding with the escalation. I can also help you set up a standing PO for recurring purchases to avoid this in the future."}
        ]
    )
    
    # ERP Scenario 3: System Access Issue
    erp_access_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "I can't access the Sales Order module anymore. I could yesterday!"},
            {"role": "assistant", "content": "I apologize for the inconvenience. Let me investigate this access issue for you. Can you tell me which specific transaction code or menu path you're trying to access?"},
            {"role": "user", "content": "VA01 for creating sales orders. It says I don't have authorization."},
            {"role": "assistant", "content": "Thank you for that information. I can see there was a role update deployed last night for security compliance. Your access profile may have been affected. I'll need to submit a request to restore your authorization. Can you confirm your employee ID and department?"},
            {"role": "user", "content": "Employee ID: E-2847, Sales Department. How long will this take? I have orders waiting!"},
            {"role": "assistant", "content": "I understand the urgency. I'm submitting an emergency access request now - these typically take 2-4 hours. As a temporary workaround, you can ask a colleague with sales order access to help, or I can escalate this to the security administrator for immediate attention."}
        ]
    )
    
    # Finance Scenario 1: Invoice Discrepancy
    finance_invoice_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "We received an invoice for $15,000 but our PO was for $12,500. What's going on?"},
            {"role": "assistant", "content": "Thank you for catching this discrepancy. Let me review the details. Can you provide the invoice number and PO number so I can investigate the difference?"},
            {"role": "user", "content": "Invoice INV-8392, PO number PO-7821. The vendor says it's for additional services."},
            {"role": "assistant", "content": "I've pulled up both documents. The original PO was indeed for $12,500 for consulting services. It appears the vendor added $2,500 for additional hours not covered in the original scope. Do you have documentation approving these additional services?"},
            {"role": "user", "content": "Our project manager might have approved it verbally. Should we pay it?"},
            {"role": "assistant", "content": "I recommend obtaining written confirmation of the approved scope change before processing payment. This protects both parties and ensures proper documentation. I can help you draft a request to the vendor for a revised invoice or a separate invoice for the additional services with proper approval."}
        ]
    )
    
    # Finance Scenario 2: Budget Variance Analysis
    finance_budget_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "Why is my department 15% over budget this month?"},
            {"role": "assistant", "content": "Let me pull up your department's budget variance report. I can see several factors contributing to the overage. The largest variance is in operational expenses, specifically travel and training costs. Would you like a detailed breakdown?"},
            {"role": "user", "content": "Yes, we only sent two people to the conference. How is that 15% over?"},
            {"role": "assistant", "content": "Looking at the details, the conference costs were budgeted at $3,000 but came in at $4,200 due to higher hotel rates. Additionally, there were three unplanned software license renewals totaling $2,800 that weren't in the original budget. These combined with other smaller variances created the 15% overage."},
            {"role": "user", "content": "What should I do to fix this for next month?"},
            {"role": "assistant", "content": "I recommend three actions: First, review and adjust your Q4 forecast to reflect actual spending patterns. Second, identify any discretionary expenses that can be deferred. Third, consider submitting a budget revision request if these increased costs are ongoing. I can help you prepare the documentation needed."}
        ]
    )
    
    # Finance Scenario 3: Expense Report Rejection
    finance_expense_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "My expense report was rejected again! What's wrong with it now?"},
            {"role": "assistant", "content": "I apologize for the frustration. Let me review your expense report to identify the issue. Could you provide the report number so I can look into the specific rejection reason?"},
            {"role": "user", "content": "Report ER-3847. I attached all my receipts this time!"},
            {"role": "assistant", "content": "Thank you. I can see the receipts are attached now, which is great. However, the rejection note indicates that three meal expenses exceed the company's $75 per-person policy, and one hotel stay wasn't pre-approved despite being over $200 per night."},
            {"role": "user", "content": "The client insisted on those restaurants! And I had to book the hotel last minute for an emergency trip."},
            {"role": "assistant", "content": "I understand these were business necessities. For the meals over policy, you'll need to add a business justification note explaining the client entertainment. For the hotel, please attach the email or documentation showing the emergency nature of the trip. Once you add these explanations, resubmit and I'll expedite the review."}
        ]
    )
    
    # Evaluate all conversation scenarios
    evaluate(
        test_cases=[
            hr_benefits_conversation,
            hr_performance_conversation,
            hr_conflict_conversation,
            erp_inventory_conversation,
            erp_purchase_conversation,
            erp_access_conversation,
            finance_invoice_conversation,
            finance_budget_conversation,
            finance_expense_conversation
        ],
        metrics=[professionalism_metric]
    )


def test_accuracy_and_helpfulness():
    """Test if the assistant provides accurate and helpful information in specialized domains."""
    
    accuracy_metric = ConversationalGEval(
        name='Accuracy and Helpfulness',
        criteria=(
            "Evaluate whether the assistant provides accurate, relevant, and helpful information "
            "that properly addresses the user's questions. The assistant should demonstrate domain "
            "knowledge and guide users toward solutions while maintaining professional standards."
        )
    )
    
    # HR Scenario: Compensation Inquiry
    hr_compensation_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "How is overtime calculated for salaried employees?"},
            {"role": "assistant", "content": "For salaried employees classified as non-exempt, overtime is calculated at 1.5x your regular hourly rate for hours worked over 40 in a week. Your hourly rate is determined by dividing your annual salary by 2,080 hours (52 weeks × 40 hours)."},
            {"role": "user", "content": "What if I'm classified as exempt?"},
            {"role": "assistant", "content": "Exempt employees are not eligible for overtime pay under FLSA regulations. Exempt status is determined by salary level, salary basis, and job duties (executive, administrative, or professional). If you believe your classification might be incorrect, I recommend discussing this with HR."},
            {"role": "user", "content": "How do I know which classification I have?"},
            {"role": "assistant", "content": "Your classification is listed in your employment contract and offer letter. You can also find it in the HR portal under 'Employment Details' or by contacting HR directly. If you'd like, I can guide you to the specific location in the portal."}
        ]
    )
    
    # ERP Scenario: Materials Planning
    erp_mrp_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "Why did MRP create a purchase requisition for 500 units when we only need 200?"},
            {"role": "assistant", "content": "MRP considers several factors when calculating requirements: current demand, safety stock levels, minimum order quantities, and lead times. The 500-unit quantity likely reflects either a minimum order quantity set in the material master or a lot-sizing rule optimizing order costs."},
            {"role": "user", "content": "How can I change this? We don't have storage space for 500 units."},
            {"role": "assistant", "content": "You have two options: First, you can modify the lot-sizing procedure in the material master (MRP2 view) from 'EX' (lot-for-lot with rounding) to 'FX' (fixed lot size) with your preferred quantity. Second, adjust the minimum order quantity if that's driving the calculation. Would you like step-by-step guidance for either option?"},
            {"role": "user", "content": "Yes, walk me through changing the lot-sizing."},
            {"role": "assistant", "content": "Navigate to MM02 (Change Material Master), enter the material number, select MRP2 view, and organization levels. Find the 'Lot size' field and change it to 'EX'. Set the 'Maximum Lot Size' to 200. Save the changes. Run MRP again (MD02) and the system will now plan for your desired quantity."}
        ]
    )
    
    # Finance Scenario: Accounts Payable Process
    finance_ap_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "What's the difference between a 2-way and 3-way match in AP?"},
            {"role": "assistant", "content": "A 2-way match compares the purchase order and invoice (verifying price and quantity). A 3-way match adds the goods receipt, confirming that items were physically received before payment. Most companies use 3-way matching for better control and to prevent payment for undelivered goods."},
            {"role": "user", "content": "We're getting a lot of invoice blocks. Why?"},
            {"role": "assistant", "content": "Invoice blocks typically occur when there are variances between the PO, goods receipt, and invoice. Common causes include: quantity differences, price discrepancies exceeding tolerance, or goods receipt not posted before invoice. Can you tell me what type of block code you're seeing?"},
            {"role": "user", "content": "It says 'price variance exceeds tolerance'. What tolerance?"},
            {"role": "assistant", "content": "Your company has set tolerance limits for price variances (typically 5-10%). If the invoice price differs from the PO price by more than this percentage, the system blocks payment for review. You can view specific tolerances in configuration or ask AP to review and approve the variance if it's legitimate, such as approved price changes."}
        ]
    )
    
    evaluate(
        test_cases=[
            hr_compensation_conversation,
            erp_mrp_conversation,
            finance_ap_conversation
        ],
        metrics=[accuracy_metric]
    )


def test_problem_solving():
    """Test the assistant's ability to guide users through complex problem resolution."""
    
    problem_solving_metric = ConversationalGEval(
        name='Problem Solving',
        criteria=(
            "Assess whether the assistant effectively identifies problems, provides actionable solutions, "
            "and guides users through resolution steps. The assistant should demonstrate logical thinking, "
            "offer alternatives when appropriate, and help users understand the underlying issues."
        )
    )
    
    # HR Scenario: FMLA Application Process
    hr_fmla_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "I need to take time off for a family medical situation but don't know where to start."},
            {"role": "assistant", "content": "I'm sorry to hear about your family situation. You may be eligible for Family and Medical Leave Act (FMLA) protection. First, let me confirm: Have you worked here for at least 12 months and worked at least 1,250 hours in the past year?"},
            {"role": "user", "content": "Yes, I've been here for 3 years. What do I need to do?"},
            {"role": "assistant", "content": "Great, here's the process: 1) Complete the FMLA request form available on the HR portal. 2) Have your family member's healthcare provider complete the medical certification form. 3) Submit both forms to HR at least 30 days in advance if possible. 4) HR will review and provide approval within 5 business days. Would you like me to email you these forms?"},
            {"role": "user", "content": "Yes, please. How much time can I take?"},
            {"role": "assistant", "content": "FMLA provides up to 12 weeks of unpaid, job-protected leave within a 12-month period. You can take this continuously or intermittently depending on medical necessity. During this time, your health benefits continue. I'm sending the forms now and including a guide explaining your rights and the process."}
        ]
    )
    
    # ERP Scenario: Production Order Delay
    erp_production_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "Production order 1000234 shows 'material shortage' but warehouse says we have the parts!"},
            {"role": "assistant", "content": "This is a common issue. Let's troubleshoot systematically. First, check the material availability in MD04 for each component. The system reserves materials based on storage location and batch requirements. Can you confirm which material numbers are showing as short?"},
            {"role": "user", "content": "Materials 5847 and 5848. Warehouse confirmed 500 pieces of each in stock."},
            {"role": "assistant", "content": "Excellent. Now check these three things: 1) Is the stock in the correct storage location specified in the production order? 2) Is the stock unrestricted (quality status available for use)? 3) Are there other reservations consuming that stock? Let's start with MMBE to check stock type and location."},
            {"role": "user", "content": "Oh! I see the problem. The stock is in quality inspection status."},
            {"role": "assistant", "content": "Perfect, you found it! Quality inspection stock can't be used for production. You have two options: 1) Use stock from another storage location if available, or 2) Contact Quality Assurance to complete the inspection and release the stock. For immediate needs, check MMBE for unrestricted stock in other locations and manually assign it to your production order using MB1A."}
        ]
    )
    
    # Finance Scenario: Month-End Close Issue
    finance_close_conversation = ConversationalTestCase(
        turns=[
            {"role": "user", "content": "We can't close the month! The system says there are unposted transactions in AP."},
            {"role": "assistant", "content": "Let's identify and resolve these unposted transactions quickly. Run transaction F.13 to see the list of parked documents awaiting posting. These are likely invoices that were entered but not completed. Can you run that report and tell me how many documents appear?"},
            {"role": "user", "content": "There are 47 parked documents! We need to close by 5 PM today."},
            {"role": "assistant", "content": "Understood. Here's your action plan: 1) Sort by document date to prioritize current period items. 2) Assign different team members to review and post documents. 3) For documents with issues, park them in the next period if they're not material to current period close. 4) Run F.13 again after each batch to track progress. I can help you delegate these if you need."},
            {"role": "user", "content": "Some have blocking errors. What should we do with those?"},
            {"role": "assistant", "content": "For blocked documents: 1) If it's a minor variance within tolerance, management can release the block using MRBR. 2) If it's a significant issue requiring vendor resolution, park it in next period. 3) Document any unusual items for your close notes. Focus on getting clean documents posted first. Would you like me to prepare a status report template for management?"}
        ]
    )
    
    evaluate(
        test_cases=[
            hr_fmla_conversation,
            erp_production_conversation,
            finance_close_conversation
        ],
        metrics=[problem_solving_metric]
    )


if __name__ == "__main__":
    print("Running Conversational LLM Evaluation Tests...")
    print("=" * 80)
    
    print("\n1. Testing Professionalism across HR, ERP, and Finance scenarios...")
    test_professionalism()
    
    print("\n2. Testing Accuracy and Helpfulness...")
    test_accuracy_and_helpfulness()
    
    print("\n3. Testing Problem Solving capabilities...")
    test_problem_solving()
    
    print("\n" + "=" * 80)
    print("All conversational tests completed!")

