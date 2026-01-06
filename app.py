import streamlit as st
import pandas as pd
import tempfile
import os
import json
from dotenv import load_dotenv

# --- UPDATED IMPORTS ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# 1. Load Environment Variables
load_dotenv()

# 2. Initialize the Brain (LLM)
if not os.getenv("OPENAI_API_KEY"):
    st.error("Missing OpenAI API Key. Please ensure OPENAI_API_KEY is set in your .env file.")
    st.stop()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- HELPER FUNCTIONS ---

def extract_text_from_pdf(uploaded_file):
    """Save uploaded file temporarily and extract text."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    
    loader = PyPDFLoader(temp_path)
    pages = loader.load_and_split()
    os.remove(temp_path) # Clean up
    return " ".join([p.page_content for p in pages])

def extract_benchmark_criteria(text):
    """Extracts parameters from the Manager's RFP document."""
    prompt = ChatPromptTemplate.from_template("""
        You are a Procurement Expert. Analyze this RFP document text and extract the key evaluation criteria.
        Return the output ONLY as a JSON object with this structure:
        {{
            "budget_cap": "The maximum budget allowed",
            "timeline": "The required delivery timeline",
            "must_haves": ["List of mandatory technical requirements"],
            "nice_to_haves": ["List of optional requirements"],
            "payment_terms": "Preferred payment terms"
        }}
        
        RFP Text: {text}
    """)
    chain = prompt | llm
    response = chain.invoke({"text": text})
    clean_json = response.content.replace('```json', '').replace('```', '')
    return json.loads(clean_json)

def analyze_vendor_proposal(criteria, vendor_text, vendor_name):
    """Compares a vendor proposal against the extracted criteria."""
    prompt = ChatPromptTemplate.from_template("""
        You are a Scoring Engine. Compare the Vendor Proposal against the Benchmark Criteria.
        
        Benchmark Criteria: {criteria}
        Vendor Proposal: {vendor_text}
        
        Analyze strictly. Return output ONLY as a JSON object:
        {{
            "vendor_name": "{vendor_name}",
            "cost_score": 0,
            "technical_score": 0,
            "delivery_score": 0,
            "overall_score": 0,
            "flags": ["List specific issues like 'Over budget by $5k' or 'Missing ISO cert'"],
            "recommendation": "Approve, Reject, or Negotiate"
        }}
        
        Note: Scores must be integers (0-100).
    """)
    chain = prompt | llm
    response = chain.invoke({
        "criteria": json.dumps(criteria),
        "vendor_text": vendor_text,
        "vendor_name": vendor_name
    })
    clean_json = response.content.replace('```json', '').replace('```', '')
    return json.loads(clean_json)

def generate_negotiation_email(vendor_name, flags, recommendation, criteria):
    """Generates a specific email based on the analysis."""
    prompt = ChatPromptTemplate.from_template("""
        You are a Senior Procurement Manager. Write a professional email to the vendor "{vendor_name}".
        
        Context:
        - We analyzed their proposal.
        - The internal recommendation is: {recommendation}.
        - The identified issues/flags are: {flags}.
        - Our Benchmark Requirements: {criteria}
        
        Instructions:
        1. If Recommendation is "Reject": Write a polite but firm rejection email citing the flags.
        2. If Recommendation is "Negotiate": Ask them to revise their quote to address the flags (e.g., lower price, faster delivery).
        3. If Recommendation is "Approve": Write a next-steps email to finalize the contract.
        
        Tone: Professional, Direct, and Courteous.
        Output: ONLY the email subject and body.
    """)
    chain = prompt | llm
    response = chain.invoke({
        "vendor_name": vendor_name,
        "flags": flags,
        "recommendation": recommendation,
        "criteria": json.dumps(criteria)
    })
    return response.content

# --- THE UI (Streamlit) ---

st.set_page_config(page_title="AI Procurement Copilot", layout="wide")

st.title("🤖 RFP & RFQ Automation System")
st.markdown("---")

# Session State Initialization
if 'criteria' not in st.session_state:
    st.session_state['criteria'] = None
if 'vendor_results' not in st.session_state:
    st.session_state['vendor_results'] = []

# --- STEP 1: UPLOAD RFP ---
st.header("Step 1: Benchmark Definition")
rfp_file = st.file_uploader("Upload RFP/RFQ Document (PDF)", type="pdf", key="rfp")

if rfp_file and st.button("Analyze RFP Requirements"):
    with st.spinner("Extracting parameters..."):
        try:
            text = extract_text_from_pdf(rfp_file)
            st.session_state['criteria'] = extract_benchmark_criteria(text)
            st.success("Parameters Extracted!")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

# --- STEP 2: HUMAN VALIDATION ---
if st.session_state['criteria']:
    st.subheader("✅ Validate Evaluation Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        new_budget = st.text_input("Budget Cap", st.session_state['criteria'].get('budget_cap', 'N/A'))
        st.session_state['criteria']['budget_cap'] = new_budget
    with col2:
        new_time = st.text_input("Timeline", st.session_state['criteria'].get('timeline', 'N/A'))
        st.session_state['criteria']['timeline'] = new_time
        
    with st.expander("View Technical Requirements"):
        st.write(st.session_state['criteria'].get('must_haves', []))
    
    st.markdown("---")

    # --- STEP 3: VENDOR SCORING ---
    st.header("Step 2: Vendor Proposal Analysis")
    
    vendor_files = st.file_uploader("Upload Vendor Proposals (PDFs)", type="pdf", accept_multiple_files=True, key="vendors")
    
    if vendor_files and st.button("Run Scoring Engine"):
        st.session_state['vendor_results'] = []
        progress_bar = st.progress(0)
        
        for idx, v_file in enumerate(vendor_files):
            with st.spinner(f"Analyzing {v_file.name}..."):
                try:
                    v_text = extract_text_from_pdf(v_file)
                    score_data = analyze_vendor_proposal(st.session_state['criteria'], v_text, v_file.name)
                    st.session_state['vendor_results'].append(score_data)
                except Exception as e:
                    st.error(f"Error analyzing {v_file.name}: {e}")
                
                progress_bar.progress((idx + 1) / len(vendor_files))
                
        st.success("Scoring Complete!")

# --- STEP 4: GOVERNANCE & RESULTS ---
if st.session_state['vendor_results']:
    st.header("Step 3: Governance & Comparison")
    
    df = pd.DataFrame(st.session_state['vendor_results'])
    
    # 4.1 Comparison Matrix (Side-by-Side View)
    st.subheader("📊 Detailed Side-by-Side Comparison")
    
    comparison_data = {}
    
    # Add Benchmark Column
    comparison_data['Benchmark'] = {
        "Overall Score": "100",
        "Cost": st.session_state['criteria'].get('budget_cap'),
        "Timeline": st.session_state['criteria'].get('timeline'),
        "Status": "Target"
    }
    
    # Add Vendor Columns
    for res in st.session_state['vendor_results']:
        comparison_data[res['vendor_name']] = {
            "Overall Score": res['overall_score'],
            "Cost": f"Score: {res['cost_score']}",
            "Timeline": f"Score: {res['delivery_score']}",
            "Status": res['recommendation']
        }
    
    # Display Matrix
    compare_df = pd.DataFrame(comparison_data)
    st.table(compare_df)
    
    # 4.2 Main Scoring Table
    st.subheader("🏆 Scoring Table")
    st.dataframe(
        df[['vendor_name', 'overall_score', 'cost_score', 'technical_score', 'recommendation', 'flags']],
        use_container_width=True
    )
    
    # 4.3 Supervisor Bot
    st.subheader("🤖 Supervisor Bot Insights")
    for res in st.session_state['vendor_results']:
        score = int(res.get('overall_score', 0))
        if score < 60:
            st.warning(f"⚠️ **{res['vendor_name']}**: Low score detected ({score}). Recommendation: {res['recommendation']}")
        if res.get('flags'):
            st.error(f"🚩 **{res['vendor_name']} Risk Flags**: {res['flags']}")

    # --- STEP 5: NEGOTIATOR & EXPORT ---
    st.markdown("---")
    st.header("Step 4: Actions")
    
    col_a, col_b = st.columns(2)
    
    # Export Report
    with col_a:
        st.subheader("📥 Download Report")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Analysis (CSV)",
            data=csv,
            file_name="procurement_analysis.csv",
            mime="text/csv",
        )

    # Email Negotiator
    with col_b:
        st.subheader("📧 Negotiation Assistant")
        selected_vendor = st.selectbox("Select Vendor to Email:", [v['vendor_name'] for v in st.session_state['vendor_results']])
        
        if st.button("Draft Email"):
            # Find the data for the selected vendor
            vendor_data = next(item for item in st.session_state['vendor_results'] if item["vendor_name"] == selected_vendor)
            
            with st.spinner("Drafting email..."):
                email_draft = generate_negotiation_email(
                    vendor_name=vendor_data['vendor_name'],
                    flags=vendor_data['flags'],
                    recommendation=vendor_data['recommendation'],
                    criteria=st.session_state['criteria']
                )
                
            st.text_area("Draft Email", value=email_draft, height=300)
            st.info("You can copy this text into Outlook/Gmail.")