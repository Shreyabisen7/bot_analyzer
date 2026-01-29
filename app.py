import streamlit as st
import pandas as pd
import tempfile
import os
import json
import mysql.connector
import urllib.parse
from dotenv import load_dotenv

# IMPORTS FOR NETWORK FIX
import httpx

# IMPORTS FOR AI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# 1. Load Environment Variables
load_dotenv()

# NETWORK FIX: UNSET PROXIES & BYPASS SSL
for env_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if env_var in os.environ:
        del os.environ[env_var]

# 2. Check OpenAI Key
if not os.getenv("OPENAI_API_KEY"):
    st.error("Missing OpenAI API Key. Please add it to your .env file.")
    st.stop()

# 3. Initialize AI (With SSL Verification Disabled & Increased Timeout)
custom_http_client = httpx.Client(verify=False, timeout=60.0)

llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0,
    http_client=custom_http_client
)

# --- DATABASE FUNCTIONS (NO SCHEMA CHANGES) ---

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
    """Saves the Vendor Score to MySQL. 
       NOTE: This ignores the 'compliance_breakdown' list, so no DB changes are needed."""
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
        Analyze this RFP document.
        
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
    response = chain.invoke({"text": text[:200000]})
    return clean_and_parse_json(response.content)

def analyze_vendor_proposal(criteria, vendor_text, vendor_name):
    # Convert list of technical requirements to a string for the prompt
    tech_reqs_str = "\n".join(criteria.get('technical_must_haves', []))

    prompt = ChatPromptTemplate.from_template("""
        You are a Proposal Evaluator. Compare this Vendor Proposal against the RFP Requirements.
        
        RFP Technical Requirements: 
        {tech_reqs}
        
        Vendor Text (Truncated): 
        {vendor_text}
        
        Task:
        1. Score the vendor (0-100) on Cost, Technical, and Compliance.
        2. **CRITICAL:** Create a "compliance_breakdown" list. 
           - Go through every single item in the "RFP Technical Requirements" list.
           - Check if the vendor mentions it.
           - Output fields: "RFP_Parameter", "Vendor_Status" (Match/Partial/Mismatch), and "Vendor_Evidence" (Short quote).
        
        Output JSON:
        {{
            "vendor_name": "{vendor_name}",
            "cost_score": 0-100,
            "technical_score": 0-100,
            "overall_score": 0-100,
            "scoring_reasoning": "Explain score...",
            "flags": ["List red flags"],
            "recommendation": "Approve/Reject",
            "compliance_breakdown": [
                {{
                    "RFP_Parameter": "Requirement text...",
                    "Vendor_Status": "Match",
                    "Vendor_Evidence": "Vendor quote..."
                }}
            ]
        }}
    """)
    
    chain = prompt | llm
    response = chain.invoke({
        "tech_reqs": tech_reqs_str,
        "vendor_text": vendor_text[:100000], 
        "vendor_name": vendor_name
    })
    return clean_and_parse_json(response.content)

def generate_negotiation_email(vendor_name, flags, recommendation, criteria):
    prompt = ChatPromptTemplate.from_template("""
        Write a procurement email to "{vendor_name}".
        Recommendation: {recommendation}.
        Issues: {flags}.
        Project: "{project}".
        Output ONLY the email body.
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

st.title(" BOT Analyzer")

# Initialize Session State
if 'criteria' not in st.session_state:
    st.session_state['criteria'] = None
if 'rfp_db_id' not in st.session_state:
    st.session_state['rfp_db_id'] = None
if 'vendor_results' not in st.session_state:
    st.session_state['vendor_results'] = []

# Tabs
tab1, tab2 = st.tabs([" Analysis", " Database History"])

with tab1:
    # STEP 1: UPLOAD RFP
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
                st.success(" RFP Analyzed & Saved!")

    # Display Data
    if st.session_state['criteria']:
        data = st.session_state['criteria']
        st.markdown(f"###  Project: {data.get('project_name', 'Unknown')}")
        
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.info(f" **Budget:**\n{data.get('budget_cap')}")
        with k2: st.info(f" **Timeline:**\n{data.get('timeline')}")
        with k3: st.warning(f" **Penalties:**\n{data.get('penalty_clauses')}")
        with k4: st.success(f" **Payment:**\n{data.get('payment_terms')}")

        with st.expander(" Technical Requirements", expanded=False):
            st.write(data.get('technical_must_haves', []))

    # STEP 2: VENDOR SCORING
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
                        # Save summary to DB (ignores the detailed list automatically)
                        save_vendor_result_to_db(st.session_state['rfp_db_id'], score_data)
                    progress_bar.progress((idx + 1) / len(vendor_files))
            
            st.success("Analysis Complete!")

    # STEP 3: RESULTS & DETAILED TABLES
    if st.session_state['vendor_results']:
        st.markdown("---")
        st.subheader(" High-Level Comparison Matrix")
        
        # Summary Table
        comparison_data = []
        for res in st.session_state['vendor_results']:
            comparison_data.append({
                "Vendor": res['vendor_name'],
                "Overall Score": res.get('overall_score', 0),
                "Recommendation": res.get('recommendation', 'N/A'),
                "Red Flags": str(res.get('flags', [])[:3])
            })
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

        
        # NEW FEATURE: DETAILED VENDOR TABLES (Requested Update)
        
        st.markdown("---")
        st.subheader(" Detailed Compliance Audit (Vendor vs RFP)")
        st.info("Select a vendor below to see exactly which RFP parameters they matched or missed.")

        # 1. Vendor Selector
        vendor_names = [v['vendor_name'] for v in st.session_state['vendor_results']]
        selected_vendor_name = st.selectbox("Select Vendor to Audit:", vendor_names)

        # 2. Get Data for Selected Vendor
        selected_data = next((item for item in st.session_state['vendor_results'] if item["vendor_name"] == selected_vendor_name), None)

        # 3. Display Detailed Table
        if selected_data and 'compliance_breakdown' in selected_data:
            df_details = pd.DataFrame(selected_data['compliance_breakdown'])
            
            # Color Styling Function
            def highlight_status(val):
                color = 'white'
                val_str = str(val).lower()
                if 'match' in val_str:
                    color = '#d4edda' # Light Green
                elif 'miss' in val_str or 'not' in val_str:
                    color = '#f8d7da' # Light Red
                elif 'partial' in val_str:
                    color = '#fff3cd' # Light Yellow
                return f'background-color: {color}'

            try:
                st.write(f"###  {selected_vendor_name} - Parameter Compliance Table")
                st.dataframe(
                    df_details.style.map(highlight_status, subset=['Vendor_Status']),
                    use_container_width=True,
                    height=500
                )
            except Exception as e:
                # Fallback if pandas styling fails
                st.dataframe(df_details, use_container_width=True)
        else:
            st.warning("No detailed compliance data available for this vendor.")
        
        
        # END NEW FEATURE
        

        # STEP 4: ACTIONS
        st.markdown("---")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader(" Export Report")
            df = pd.DataFrame(st.session_state['vendor_results'])
            # Convert list columns to strings for CSV export
            df['flags'] = df['flags'].apply(lambda x: str(x))
            df['compliance_breakdown'] = df.get('compliance_breakdown', []).apply(lambda x: "See App for Details")
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "final_report.csv", "text/csv")
            
        with col_b:
            st.subheader(" Smart Negotiation")
            selected_vendor_email = st.selectbox("Select Vendor for Email:", vendor_names, key="email_sel")
            
            if st.button("Draft Email"):
                v_data = next(v for v in st.session_state['vendor_results'] if v['vendor_name'] == selected_vendor_email)
                with st.spinner("Drafting..."):
                    email_content = generate_negotiation_email(
                        v_data['vendor_name'], 
                        v_data.get('flags', []), 
                        v_data.get('recommendation', 'N/A'), 
                        st.session_state['criteria']
                    )
                st.text_area("AI Draft", value=email_content, height=200)

with tab2:
    st.header(" Database History")
    if st.button("Refresh History"):
        df_hist = fetch_history()
        st.dataframe(df_hist, use_container_width=True)