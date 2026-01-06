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

# 3. Initialize AI
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- DATABASE FUNCTIONS ---

def get_db_connection():
    """Connect to MySQL using .env credentials including PORT"""
    try:
        return mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=int(os.getenv("DB_PORT", 3306))
        )
    except mysql.connector.Error as err:
        st.error(f" Database Connection Failed: {err}")
        return None

def save_rfp_to_db(criteria):
    """Saves the Benchmark Rules to MySQL"""
    conn = get_db_connection()
    if not conn: return None
    
    cursor = conn.cursor()
    must_haves_str = ", ".join(criteria.get('must_haves', []))
    
    query = """
        INSERT INTO rfp_benchmarks (project_name, budget_cap, timeline, must_haves)
        VALUES (%s, %s, %s, %s)
    """
    values = ("New RFP Project", criteria.get('budget_cap'), criteria.get('timeline'), must_haves_str)
    
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
        SELECT v.vendor_name, v.overall_score, v.recommendation, r.budget_cap, v.created_at 
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
    """Helper to clean JSON output from AI"""
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
        Extract key RFP criteria as JSON:
        {{
            "budget_cap": "Max budget",
            "timeline": "Delivery time",
            "must_haves": ["List of requirements"]
        }}
        Text: {text}
    """)
    chain = prompt | llm
    response = chain.invoke({"text": text})
    return clean_and_parse_json(response.content)

def analyze_vendor_proposal(criteria, vendor_text, vendor_name):
    # UPDATED: Now asks for 'contact_email'
    prompt = ChatPromptTemplate.from_template("""
        Compare Vendor Proposal vs Benchmark: {criteria}
        Vendor Text: {vendor_text}
        
        Output JSON:
        {{
            "vendor_name": "{vendor_name}",
            "contact_email": "Extract email address from text (if none found, return '')",
            "cost_score": 0-100,
            "technical_score": 0-100,
            "delivery_score": 0-100,
            "overall_score": 0-100,
            "flags": ["List issues"],
            "recommendation": "Approve/Reject"
        }}
    """)
    chain = prompt | llm
    response = chain.invoke({
        "criteria": json.dumps(criteria),
        "vendor_text": vendor_text,
        "vendor_name": vendor_name
    })
    return clean_and_parse_json(response.content)

def generate_negotiation_email(vendor_name, flags, recommendation, criteria):
    prompt = ChatPromptTemplate.from_template("""
        Write a professional procurement email to "{vendor_name}".
        Recommendation: {recommendation}.
        Issues: {flags}.
        Requirements: {criteria}
        
        If Reject: Polite rejection citing issues.
        If Negotiate: Ask to fix issues.
        If Approve: Next steps.
        
        Output ONLY the email body text.
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

st.set_page_config(page_title="AI Procurement System", layout="wide")
st.title("🤖 RFP Automation (Ultimate Edition)")

# Initialize Session State
if 'criteria' not in st.session_state:
    st.session_state['criteria'] = None
if 'rfp_db_id' not in st.session_state:
    st.session_state['rfp_db_id'] = None
if 'vendor_results' not in st.session_state:
    st.session_state['vendor_results'] = []

# Tabs
tab1, tab2 = st.tabs([" New Analysis", " Database History"])

with tab1:
    # --- STEP 1: UPLOAD RFP ---
    st.subheader("Step 1: Define Benchmarks")
    rfp_file = st.file_uploader("Upload RFP Document", type="pdf", key="rfp")

    if rfp_file and st.button("Analyze & Save RFP"):
        with st.spinner("Extracting & Saving to Database..."):
            text = extract_text_from_pdf(rfp_file)
            st.session_state['criteria'] = extract_benchmark_criteria(text)
            
            st.success("Parameters Extracted!")
            
            # Save to DB
            rfp_id = save_rfp_to_db(st.session_state['criteria'])
            if rfp_id:
                st.session_state['rfp_db_id'] = rfp_id
                st.toast(f"Saved to DB ID: {rfp_id}")

    # Human Validation Section
    if st.session_state['criteria']:
        st.markdown("###  Extracted Requirements")
        c1, c2 = st.columns(2)
        with c1:
            st.info(f" **Budget Cap:** {st.session_state['criteria'].get('budget_cap')}")
        with c2:
            st.info(f" **Timeline:** {st.session_state['criteria'].get('timeline')}")
        
        with st.expander("See Technical Must-Haves"):
            st.write(st.session_state['criteria'].get('must_haves'))

    # --- STEP 2: VENDOR SCORING ---
    st.markdown("---")
    st.subheader("Step 2: Score Vendors")
    vendor_files = st.file_uploader("Upload Vendor PDFs", type="pdf", accept_multiple_files=True, key="vendors")

    if vendor_files and st.button("Score & Save Results"):
        if not st.session_state['rfp_db_id']:
            st.error("Please analyze an RFP first!")
        else:
            st.session_state['vendor_results'] = []
            progress_bar = st.progress(0)
            
            for idx, v_file in enumerate(vendor_files):
                with st.spinner(f"Processing {v_file.name}..."):
                    v_text = extract_text_from_pdf(v_file)
                    score_data = analyze_vendor_proposal(st.session_state['criteria'], v_text, v_file.name)
                    if score_data:
                        st.session_state['vendor_results'].append(score_data)
                        save_vendor_result_to_db(st.session_state['rfp_db_id'], score_data)
                    progress_bar.progress((idx + 1) / len(vendor_files))
            
            st.success("Analysis Complete & Saved!")

    # --- STEP 3: COMPARISON MATRIX ---
    if st.session_state['vendor_results']:
        st.markdown("---")
        st.subheader(" Side-by-Side Comparison")
        
        # Build the Matrix
        comparison_data = {}
        comparison_data['Benchmark'] = {
            "Budget": st.session_state['criteria'].get('budget_cap'),
            "Timeline": st.session_state['criteria'].get('timeline'),
            "Overall Score": "TARGET",
            "Recommendation": "-"
        }
        
        for res in st.session_state['vendor_results']:
            comparison_data[res['vendor_name']] = {
                "Budget": f"Score: {res.get('cost_score', 0)}",
                "Timeline": f"Score: {res.get('delivery_score', 0)}",
                "Overall Score": res.get('overall_score', 0),
                "Recommendation": res.get('recommendation', 'N/A')
            }
        
        st.table(pd.DataFrame(comparison_data))

        # --- STEP 4: ACTIONS ---
        st.markdown("---")
        st.header("Step 4: Actions")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader(" Export Report")
            df = pd.DataFrame(st.session_state['vendor_results'])
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "rfp_report.csv", "text/csv")
            
        with col_b:
            st.subheader(" Negotiation Assistant")
            
            vendor_names = [v['vendor_name'] for v in st.session_state['vendor_results']]
            selected_vendor = st.selectbox("Select Vendor to Email:", vendor_names)
            
            if st.button("Draft Negotiation Email"):
                v_data = next(v for v in st.session_state['vendor_results'] if v['vendor_name'] == selected_vendor)
                
                with st.spinner("Drafting email..."):
                    email_content = generate_negotiation_email(
                        v_data['vendor_name'], 
                        v_data.get('flags', []), 
                        v_data.get('recommendation', 'N/A'), 
                        st.session_state['criteria']
                    )
                
                st.text_area("AI Draft", value=email_content, height=250)
                
                # --- AUTO FILL EMAIL BUTTON ---
                recipient = v_data.get('contact_email', '') # Get extracted email
                subject = f"Regarding Proposal: {selected_vendor}"
                body_encoded = urllib.parse.quote(email_content)
                mailto_link = f"mailto:{recipient}?subject={subject}&body={body_encoded}"
                
                st.markdown(f'''
                    <a href="{mailto_link}" target="_blank" style="
                        background-color: #0078D4; color: white; padding: 10px 20px; 
                        text-decoration: none; border-radius: 5px; font-weight: bold;
                        display: inline-block; margin-top: 10px;
                    ">
                     Open in Outlook (Send to: {recipient if recipient else 'Unknown'})
                    </a>
                ''', unsafe_allow_html=True)

with tab2:
    st.header(" Database History")
    if st.button("Refresh History"):
        df_hist = fetch_history()
        st.dataframe(df_hist, use_container_width=True)