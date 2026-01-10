import streamlit as st
import pandas as pd
import tempfile
import os
import json
import mysql.connector
import urllib.parse
from dotenv import load_dotenv

# --- IMPORTS FOR AI ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# 1. Load Environment Variables
load_dotenv()

# 2. Check OpenAI Key
if not os.getenv("OPENAI_API_KEY"):
    st.error("Missing OpenAI API Key. Please add it to your .env file.")
    st.stop()

# 3. Initialize AI (Using GPT-4o for maximum context)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- DATABASE FUNCTIONS ---

def get_db_connection():
    """Connect to MySQL using .env credentials"""
    try:
        return mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=int(os.getenv("DB_PORT", 3306))
        )
    except mysql.connector.Error as err:
        st.error(f"Database Connection Failed: {err}")
        return None

def save_rfp_to_db(criteria):
    """Saves the Benchmark Rules to MySQL"""
    conn = get_db_connection()
    if not conn: return None
    
    cursor = conn.cursor()
    
    # Pack complex fields
    tech_str = " | ".join(criteria.get('technical_must_haves', []))
    payment_str = str(criteria.get('payment_terms', 'N/A'))
    penalty_str = str(criteria.get('penalty_clauses', 'N/A'))
    
    combined_criteria = f"TECH: {tech_str} || PAYMENT: {payment_str} || PENALTY: {penalty_str}"
    
    query = """
        INSERT INTO rfp_benchmarks (project_name, budget_cap, timeline, must_haves)
        VALUES (%s, %s, %s, %s)
    """
    values = (
        criteria.get('project_name', 'New RFP Project'), 
        criteria.get('budget_cap'), 
        criteria.get('timeline'), 
        combined_criteria
    )
    
    try:
        cursor.execute(query, values)
        conn.commit()
        rfp_id = cursor.lastrowid
        return rfp_id
    except mysql.connector.Error as err:
        st.error(f"Error saving RFP: {err}")
        return None
    finally:
        cursor.close()
        conn.close()

def save_vendor_result_to_db(rfp_id, data):
    """Saves the Vendor Score to MySQL"""
    conn = get_db_connection()
    if not conn: return
    
    cursor = conn.cursor()
    flags_str = ", ".join(data.get('flags', []))
    
    query = """
        INSERT INTO vendor_results 
        (rfp_id, vendor_name, overall_score, cost_score, technical_score, flags, recommendation)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        rfp_id,
        data.get('vendor_name', 'Unknown'),
        data.get('overall_score', 0),
        data.get('cost_score', 0),
        data.get('technical_score', 0),
        flags_str,
        data.get('recommendation', 'Review')
    )
    
    try:
        cursor.execute(query, values)
        conn.commit()
    except mysql.connector.Error as err:
        st.error(f"Error saving Vendor: {err}")
    finally:
        cursor.close()
        conn.close()

def fetch_history():
    """Reads past data from MySQL"""
    conn = get_db_connection()
    if not conn: return pd.DataFrame()
    
    query = """
        SELECT v.vendor_name, v.overall_score, v.recommendation, r.project_name, v.created_at 
        FROM vendor_results v
        JOIN rfp_benchmarks r ON v.rfp_id = r.id
        ORDER BY v.created_at DESC
    """
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# --- AI HELPER FUNCTIONS ---

def clean_and_parse_json(ai_response_text):
    text = ai_response_text.replace('```json', '').replace('```', '')
    start_idx = text.find('{')
    end_idx = text.rfind('}') + 1
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    try:
        return json.loads(text)
    except:
        return {}

def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    
    loader = PyPDFLoader(temp_path)
    pages = loader.load_and_split()
    os.remove(temp_path)
    
    return " ".join([p.page_content for p in pages])

def extract_benchmark_criteria(text):
    prompt = ChatPromptTemplate.from_template("""
        You are a Senior Technical Procurement Officer. 
        Analyze this RFP document deeply (Text length: {length}).
        
        Do NOT summarize. Be EXHAUSTIVE. Extract specific details.
        
        1. **Project Name**: Title of the RFP.
        2. **Budget**: 
           - Look for "Estimated Cost" or "Tender Value".
           - IF NOT FOUND: Look for "EMD" (Earnest Money Deposit).
           - CALCULATE: Estimated Budget = EMD Amount * 100.
           - Return the calculated amount labeled as "Estimated based on EMD".
        3. **Timeline**: Extract specific duration (e.g., "1 year implementation + 3 years ATS").
        4. **Payment Terms**: Extract the exact milestones.
        5. **Penalty Clauses**: Extract Liquidated Damages (LD) percentages.
        6. **Technical Must-Haves**: Extract at least 20 distinct technical requirements.
        
        Output valid JSON:
        {{
            "project_name": "Title",
            "budget_cap": "Calculated or Specific Budget",
            "timeline": "Duration",
            "payment_terms": "Milestone details...",
            "penalty_clauses": "LD details...",
            "technical_must_haves": ["Req 1", "Req 2", "...", "Req 20"]
        }}
        
        RFP Text:
        {text}
    """)
    chain = prompt | llm
    response = chain.invoke({"text": text[:200000], "length": len(text)})
    return clean_and_parse_json(response.content)

def analyze_vendor_proposal(criteria, vendor_text, vendor_name):
    """
    UPDATED: Now asks for 'scoring_reasoning' to explain why a score (e.g. 25) was given.
    """
    prompt = ChatPromptTemplate.from_template("""
        You are a Proposal Evaluator. Compare this Vendor Proposal against the RFP Requirements.
        
        RFP Requirements: {criteria}
        
        Vendor Text (Truncated): {vendor_text}
        
        Task:
        1. Search for the vendor's email.
        2. Score them based on the RFP criteria.
        3. **CRITICAL:** Provide a short "scoring_reasoning" explaining exactly why you gave the Overall Score.
           (e.g., "Score is 25 because vendor missed all technical requirements but accepted payment terms.")
        
        Output JSON:
        {{
            "vendor_name": "{vendor_name}",
            "contact_email": "Email address found",
            "cost_score": 0-100,
            "technical_score": 0-100,
            "compliance_score": 0-100,
            "overall_score": 0-100,
            "scoring_reasoning": "EXPLAIN WHY THE SCORE IS LOW OR HIGH HERE",
            "flags": ["List specific missing requirements", "Deviations"],
            "recommendation": "Strongly Approve / Negotiate / Reject"
        }}
    """)
    
    chain = prompt | llm
    response = chain.invoke({
        "criteria": json.dumps(criteria),
        "vendor_text": vendor_text[:100000], 
        "vendor_name": vendor_name
    })
    return clean_and_parse_json(response.content)

def generate_negotiation_email(vendor_name, flags, recommendation, criteria):
    prompt = ChatPromptTemplate.from_template("""
        Write a professional procurement email to "{vendor_name}".
        Recommendation: {recommendation}.
        Issues identified: {flags}.
        
        Context:
        Project: "{project}".
        
        Output ONLY the email body text.
    """)
    chain = prompt | llm
    response = chain.invoke({
        "vendor_name": vendor_name,
        "flags": flags,
        "recommendation": recommendation,
        "project": criteria.get('project_name', 'The Project')
    })
    return response.content

# --- THE UI (Streamlit) ---

st.set_page_config(page_title="AI Procurement System", layout="wide")

# 1. CHANGED: Removed "Deep Dive Edition" text
st.title("🤖 RFP Analyzer 3.0")

# Initialize Session State
if 'criteria' not in st.session_state:
    st.session_state['criteria'] = None
if 'rfp_db_id' not in st.session_state:
    st.session_state['rfp_db_id'] = None
if 'vendor_results' not in st.session_state:
    st.session_state['vendor_results'] = []

# Tabs
tab1, tab2 = st.tabs(["📂 Analysis", "🗄️ Database History"])

with tab1:
    # --- STEP 1: UPLOAD RFP ---
    st.subheader("Step 1: Analyze RFP Document")
    rfp_file = st.file_uploader("Upload RFP Document", type="pdf", key="rfp")

    if rfp_file and st.button("Analyze RFP"):
        with st.spinner("Processing RFP..."):
            text = extract_text_from_pdf(rfp_file)
            st.session_state['criteria'] = extract_benchmark_criteria(text)
            
            # Save to DB
            rfp_id = save_rfp_to_db(st.session_state['criteria'])
            if rfp_id:
                st.session_state['rfp_db_id'] = rfp_id
                st.success("✅ RFP Analyzed & Saved!")

    # Display Data
    if st.session_state['criteria']:
        data = st.session_state['criteria']
        st.markdown(f"### 📋 Project: {data.get('project_name', 'Unknown')}")
        
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.info(f"💰 **Budget:**\n{data.get('budget_cap')}")
        with k2: st.info(f"⏳ **Timeline:**\n{data.get('timeline')}")
        with k3: st.warning(f"⚠️ **Penalties:**\n{data.get('penalty_clauses')}")
        with k4: st.success(f"💳 **Payment:**\n{data.get('payment_terms')}")

        with st.expander("🔍 Technical Requirements", expanded=False):
            st.write(data.get('technical_must_haves', []))

    # --- STEP 2: VENDOR SCORING ---
    st.markdown("---")
    st.subheader("Step 2: Score Vendors")
    vendor_files = st.file_uploader("Upload Vendor Proposals", type="pdf", accept_multiple_files=True, key="vendors")

    if vendor_files and st.button("Score Vendors"):
        if not st.session_state['rfp_db_id']:
            st.error("Please analyze an RFP first!")
        else:
            st.session_state['vendor_results'] = []
            progress_bar = st.progress(0)
            
            for idx, v_file in enumerate(vendor_files):
                with st.spinner(f"Analyzing {v_file.name}..."):
                    v_text = extract_text_from_pdf(v_file)
                    score_data = analyze_vendor_proposal(st.session_state['criteria'], v_text, v_file.name)
                    if score_data:
                        st.session_state['vendor_results'].append(score_data)
                        save_vendor_result_to_db(st.session_state['rfp_db_id'], score_data)
                    progress_bar.progress((idx + 1) / len(vendor_files))
            
            st.success("Analysis Complete!")

    # --- STEP 3: RESULTS TABLE (UPDATED) ---
    if st.session_state['vendor_results']:
        st.markdown("---")
        st.subheader("🏆 Vendor Comparison Matrix")
        
        comparison_data = []
        for res in st.session_state['vendor_results']:
            # 2. ADDED: Logic to show RFP Name and Score Reasoning
            comparison_data.append({
                "RFP Project": st.session_state['criteria'].get('project_name'), # Added RFP Name
                "Vendor": res['vendor_name'],
                "Overall Score": res.get('overall_score', 0),
                "Score Reasoning": res.get('scoring_reasoning', 'N/A'), # Added Score Reasoning
                "Tech Score": res.get('technical_score', 0),
                "Compliance Score": res.get('compliance_score', 0),
                "Recommendation": res.get('recommendation', 'N/A'),
                "Red Flags": str(res.get('flags', [])[:3]) # Limit flags length
            })
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

        # --- STEP 4: ACTIONS ---
        st.markdown("---")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("📥 Export Report")
            df = pd.DataFrame(st.session_state['vendor_results'])
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "final_report.csv", "text/csv")
            
        with col_b:
            st.subheader("📧 Smart Negotiation")
            vendor_names = [v['vendor_name'] for v in st.session_state['vendor_results']]
            selected_vendor = st.selectbox("Select Vendor:", vendor_names)
            
            if st.button("Draft Email"):
                v_data = next(v for v in st.session_state['vendor_results'] if v['vendor_name'] == selected_vendor)
                with st.spinner("Drafting..."):
                    email_content = generate_negotiation_email(
                        v_data['vendor_name'], 
                        v_data.get('flags', []), 
                        v_data.get('recommendation', 'N/A'), 
                        st.session_state['criteria']
                    )
                st.text_area("AI Draft", value=email_content, height=200)

with tab2:
    st.header("🗄️ Database History")
    if st.button("Refresh"):
        df_hist = fetch_history()
        st.dataframe(df_hist, use_container_width=True)