# app.py
"""
IntegrityOS - The CA Super App
------------------------------
A single-file Streamlit application for Chartered Accountants involving:
- Data Collection (Excel, PDF, Tally ODBC, XML)
- Financial Fraud Detection (Rule-based Audit)
- ESG Compliance (Carbon Math)
- PBC (Prepared By Client) Workflow
- Role-based Access (CA vs Client)
- AI Summarization & Reporting (Google Gemini)
- Email Notifications (SMTP)

Dependencies:
-------------
Create a virtual environment and install the following:
# pip install streamlit pandas numpy pyodbc PyPDF2 google-generativeai plotly openpyxl

Configuration:
--------------
- This app creates a local SQLite database 'users.db' automatically.
- It creates an 'uploads' folder automatically.
- Users must provide a Google Gemini API Key in the sidebar for AI features.
- Tally ODBC requires Tally Prime running locally and configured (usually port 9000).
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import io
import datetime
import time
import smtplib
import ssl
import logging
import re
from email.message import EmailMessage
from typing import Optional, List, Dict, Any

# Security & Hashing
import hashlib

# Visualization
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# AI & PDF
import google.generativeai as genai
import PyPDF2

# Database Connectivity
import pyodbc
import xml.etree.ElementTree as ET

# --- CONSTANTS & CONFIGURATION ---
DB_FILE = "users.db"
UPLOAD_FOLDER = "uploads"
APP_TITLE = "IntegrityOS – CA Super App"
APP_ICON = "🛡️"

# Setup Logging
logging.basicConfig(filename='app.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(layout="wide", page_title=APP_TITLE, page_icon=APP_ICON)

# --- CUSTOM CSS (DARK OCEAN THEME) ---
st.markdown("""
<style>
    /* 1. Main Background - Deep Ocean Navy */
    .stApp {
        background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
        color: #e2e8f0; /* Light Gray/White Text */
    }

    /* 2. Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1e293b;
    }

    /* 3. Headers (H1, H2, H3) - Bright Cyan/Teal */
    h1, h2, h3, h4, h5 {
        color: #22d3ee !important; /* Cyan-400 */
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }

    /* 4. Cards / Metrics - Dark Blue with Glow */
    div[data-testid="metric-container"] {
        background-color: #1e293b; /* Slate-800 */
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #22d3ee; /* Cyan Glow on Hover */
    }
    
    /* Label color inside metrics */
    div[data-testid="metric-container"] > label {
        color: #94a3b8 !important; /* Muted Blue-Gray */
    }
    
    /* Value color inside metrics */
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important; /* Bright White */
    }

    /* 5. Buttons - Bright Gradient Cyan */
    .stButton>button {
        background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(8, 145, 178, 0.39);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(8, 145, 178, 0.23);
        background: linear-gradient(135deg, #06b6d4 0%, #22d3ee 100%);
        color: white;
    }

    /* 6. Inputs (Text Input, Number Input) */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1e293b; /* Dark Slate */
        color: #ffffff;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    .stTextInput>div>div>input:focus {
        border-color: #22d3ee;
        box-shadow: 0 0 0 1px #22d3ee;
    }
    
    /* 7. Selectbox & Dropdowns */
    .stSelectbox>div>div>div {
        background-color: #1e293b;
        color: white;
    }

    /* 8. Dataframes/Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 10px;
        background-color: #0f172a;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b;
        color: #22d3ee;
        border-bottom: 2px solid #22d3ee;
    }

</style>
""", unsafe_allow_html=True)

# --- SECURITY UTILS (HASHLIB REPLACEMENT) ---

def generate_password_hash(password: str) -> str:
    """Generates a SHA256 hash for the password (Demo purpose)."""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password_hash(stored_hash: str, password: str) -> bool:
    """Verifies a password against the stored SHA256 hash."""
    return stored_hash == hashlib.sha256(password.encode()).hexdigest()

# --- DATABASE & AUTHENTICATION UTILS ---

def create_db_and_seed():
    """
    Creates the SQLite database and tables if they don't exist.
    Seeds two default users: CA Admin and Client User.
    """
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Create Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')

    # Create Files Table (Metadata)
    c.execute('''CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename_on_disk TEXT,
                    original_filename TEXT,
                    uploader_username TEXT,
                    target_client TEXT,
                    role_uploaded_as TEXT,
                    file_type TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )''')

    # Create Reports Table
    c.execute('''CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    body TEXT,
                    author TEXT,
                    client TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    filepath TEXT
                )''')

    # Check and Seed Users
    try:
        # Seed CA Admin
        c.execute("SELECT * FROM users WHERE username = ?", ("ca_admin",))
        if not c.fetchone():
            pw_hash = generate_password_hash("CApass123!")
            c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                      ("ca_admin", pw_hash, "CA"))
            print("Seeded CA Admin.")

        # Seed Client User
        c.execute("SELECT * FROM users WHERE username = ?", ("client_user",))
        if not c.fetchone():
            pw_hash = generate_password_hash("Clientpass123!")
            c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                      ("client_user", pw_hash, "Client"))
            print("Seeded Client User.")
        
        conn.commit()
    except Exception as e:
        logging.error(f"DB Seeding Error: {e}")
    finally:
        conn.close()

def login_user(username, password):
    """Verifies credentials against SQLite."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
    data = c.fetchone()
    conn.close()
    
    if data:
        stored_hash, role = data
        if check_password_hash(stored_hash, password):
            return role
    return None

def signup_user(username, password, role):
    """Registers a new user (Simple implementation)."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        pw_hash = generate_password_hash(password)
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                  (username, pw_hash, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def log_file_upload(filename_disk, original_name, uploader, role, ftype, note=""):
    """Records file metadata to DB."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""INSERT INTO files 
                 (filename_on_disk, original_filename, uploader_username, role_uploaded_as, file_type, notes)
                 VALUES (?, ?, ?, ?, ?, ?)""",
              (filename_disk, original_name, uploader, role, ftype, note))
    conn.commit()
    conn.close()

# --- EXTERNAL SERVICES & LOGIC (AI, EMAIL, ODBC) ---

# --- FIX: AUTO DETECT MODEL TO PREVENT 404 ERRORS ---
def get_best_available_model():
    """
    Dynamically lists models available to the API Key and picks one that supports generation.
    This solves the '404 not found' error on older libraries.
    """
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'gemini' in m.name:
                    return m.name
        return 'models/gemini-pro' # Fallback
    except Exception as e:
        # If list_models fails (e.g. auth error), return a safe default
        return 'gemini-pro'

def call_gemini_summary(text: str) -> str:
    """
    Interacts with Google Gemini API for summarization.
    """
    api_key = st.session_state.get("gemini_api_key")
    if not api_key:
        return "⚠️ AI features disabled. Please enter Google Gemini API Key in the sidebar."

    try:
        genai.configure(api_key=api_key)
        
        # SMART FIX: Automatically find the correct model name
        model_name = get_best_available_model()
        model = genai.GenerativeModel(model_name)
        
        # PROMPT ENGINEERING:
        prompt = f"""
        You are an expert Chartered Accountant and AI assistant. 
        Please analyze the following text extracted from a financial document.
        Summarize the key financial figures, dates, and any potential red flags or compliance issues.
        Keep it concise and professional.
        
        Text content:
        {text[:8000]} 
        """ 
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini API Error: {e}")
        return f"Error contacting AI service: {str(e)}"

def send_email_smtp(to_email, subject, body):
    """
    Sends email via Gmail SMTP.
    """
    smtp_server = st.session_state.get("smtp_host", "smtp.gmail.com")
    smtp_port = 587
    sender_email = st.session_state.get("smtp_user")
    password = st.session_state.get("smtp_pass")

    if not sender_email or not password:
        st.error("SMTP Credentials missing in Sidebar.")
        return False

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = to_email

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(sender_email, password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        logging.error(f"SMTP Error: {e}")
        return False

# --- DATA LOADING & PARSING ---

def load_data(source_type: str, uploaded_file=None) -> pd.DataFrame:
    """
    Central function to ingest data from Excel, XML, or Tally ODBC.
    """
    df = pd.DataFrame()
    
    # 1. EXCEL UPLOAD
    if source_type == "Upload Excel" and uploaded_file is not None:
        try:
            raw_df = pd.read_excel(uploaded_file)
            col_map = {}
            for col in raw_df.columns:
                l_col = col.lower()
                if "date" in l_col: col_map[col] = "Date"
                elif "ledger" in l_col or "particulars" in l_col: col_map[col] = "Ledger_Name"
                elif "voucher" in l_col or "type" in l_col: col_map[col] = "Voucher_Type"
                elif "amount" in l_col or "debit" in l_col or "credit" in l_col: col_map[col] = "Amount"
                elif "narration" in l_col or "description" in l_col: col_map[col] = "Narration"
            
            df = raw_df.rename(columns=col_map)
            
            required = ["Date", "Ledger_Name", "Voucher_Type", "Amount", "Narration"]
            for r in required:
                if r not in df.columns:
                    df[r] = np.nan if r != "Amount" else 0.0

            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            return df
            
        except Exception as e:
            st.error(f"Error reading Excel: {e}")
            return None

    # 2. TALLY XML PARSING
    elif source_type == "Tally XML" and uploaded_file is not None:
        try:
            tree = ET.parse(uploaded_file)
            root = tree.getroot()
            rows = []
            
            for message in root.findall('.//TALLYMESSAGE'):
                voucher = message.find('VOUCHER')
                if voucher is not None:
                    date_str = voucher.find('DATE').text if voucher.find('DATE') is not None else ""
                    v_type = voucher.find('VOUCHERTYPENAME').text if voucher.find('VOUCHERTYPENAME') is not None else "Unknown"
                    narration = voucher.find('NARRATION').text if voucher.find('NARRATION') is not None else ""
                    
                    for entry in voucher.findall('.//LEDGERENTRIES.LIST'):
                        l_name = entry.find('LEDGERNAME').text if entry.find('LEDGERNAME') is not None else "Unknown"
                        amt = entry.find('AMOUNT').text if entry.find('AMOUNT') is not None else "0"
                        
                        rows.append({
                            "Date": date_str,
                            "Ledger_Name": l_name,
                            "Voucher_Type": v_type,
                            "Amount": float(amt),
                            "Narration": narration
                        })
            
            if rows:
                df = pd.DataFrame(rows)
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
                return df
            else:
                st.warning("No voucher data found in XML.")
                return pd.DataFrame(columns=["Date", "Ledger_Name", "Voucher_Type", "Amount", "Narration"])

        except Exception as e:
            st.error(f"Error parsing Tally XML: {e}")
            return None

    # 3. LIVE TALLY ODBC
    elif source_type == "Live Tally ODBC":
        st.info("Attempting connection to Tally Prime via ODBC...")
        # Note: Will likely fail on Cloud, but logic is sound for local
        conn_str = "DRIVER={Tally ODBC Driver64};Server=localhost;PORT=9000;"
        
        try:
            conn = pyodbc.connect(conn_str, timeout=2)
            query = "SELECT $Date, $LedgerName, $VoucherTypeName, $Amount, $Narration FROM LedgerDetails"
            df = pd.read_sql(query, conn)
            df.rename(columns={
                "$Date": "Date", 
                "$LedgerName": "Ledger_Name", 
                "$VoucherTypeName": "Voucher_Type",
                "$Amount": "Amount",
                "$Narration": "Narration"
            }, inplace=True)
            conn.close()
            return df
        except Exception as e:
            st.error("Tally Connection Failed. Ensure Tally Prime is open and ODBC port is 9000.")
            st.caption(f"Technical Error: {e}")
            return None

    # 4. AI SCAN PDF
    elif source_type == "AI Scan PDF" and uploaded_file is not None:
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return pd.DataFrame([{"Raw_Text": text}])
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None
            
    return pd.DataFrame()

# --- ANALYSIS & AUDIT FUNCTIONS ---

def run_integrity_scan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audit Rules: Round Numbers, Weekends, High Value.
    """
    if df.empty or 'Amount' not in df.columns:
        return pd.DataFrame(), 0

    df = df.copy()
    df['risk_reasons'] = ""
    df['is_risky'] = False
    
    df['Amount_Abs'] = df['Amount'].abs()
    if 'Date' in df.columns:
        df['DayOfWeek'] = df['Date'].dt.dayofweek 

    # Rule 1: Round Numbers
    mask_round = (df['Amount_Abs'] > 0) & ((df['Amount_Abs'] % 1000 == 0) | (df['Amount_Abs'] % 500 == 0))
    df.loc[mask_round, 'risk_reasons'] += "Round Number; "
    df.loc[mask_round, 'is_risky'] = True

    # Rule 2: Weekend Entries
    if 'Date' in df.columns:
        mask_weekend = df['DayOfWeek'].isin([5, 6])
        df.loc[mask_weekend, 'risk_reasons'] += "Weekend Entry; "
        df.loc[mask_weekend, 'is_risky'] = True

    # Rule 3: High Value
    mask_high = df['Amount_Abs'] > 50000
    df.loc[mask_high, 'risk_reasons'] += "High Value (>50k); "
    df.loc[mask_high, 'is_risky'] = True

    high_risk_df = df[df['is_risky'] == True].copy()
    
    # Calculate Risk Score
    total_tx = len(df)
    risky_tx = len(high_risk_df)
    if total_tx > 0:
        ratio_count = risky_tx / total_tx
        total_vol = df['Amount_Abs'].sum()
        risky_vol = high_risk_df['Amount_Abs'].sum()
        ratio_vol = risky_vol / total_vol if total_vol > 0 else 0
        risk_score = int((ratio_count * 50) + (ratio_vol * 50))
        risk_score = min(100, risk_score)
    else:
        risk_score = 0
        
    return high_risk_df, risk_score

def generate_pbc_list(df: pd.DataFrame) -> List[str]:
    """Generates document requests based on ledger names."""
    if df.empty or "Ledger_Name" not in df.columns:
        return []

    ledgers = df['Ledger_Name'].astype(str).str.lower().unique()
    requirements = set()

    for l in ledgers:
        if "rent" in l: requirements.add("Rent Agreement")
        if any(x in l for x in ["electricity", "power", "fuel"]): requirements.add("Utility Bills")
        if any(x in l for x in ["legal", "advocate", "court"]): requirements.add("Case Files / Legal Notices")
        if "salary" in l: requirements.add("Payroll / PT Challans")
            
    return list(requirements)

def esg_analysis(df: pd.DataFrame) -> dict:
    """Calculates carbon footprint."""
    results = {"estimated_units": 0, "co2_tons": 0, "explanation": "No data"}
    
    if df.empty or "Ledger_Name" not in df.columns:
        return results

    mask = df['Ledger_Name'].astype(str).str.contains('Electricity|Power', case=False, regex=True)
    subset = df[mask]
    total_spend = subset['Amount'].abs().sum()
    
    if total_spend > 0:
        avg_cost_per_unit = 8.0
        estimated_units = total_spend / avg_cost_per_unit
        co2_kg = estimated_units * 0.82
        co2_tons = co2_kg / 1000.0
        results = {
            "estimated_units": round(estimated_units, 2),
            "co2_tons": round(co2_tons, 4),
            "explanation": f"Based on total spend of ₹{total_spend:,.2f} @ ₹8/unit."
        }
    
    return results

# --- MAIN APP LAYOUT & LOGIC ---

def main():
    # 0. Initialize System
    create_db_and_seed()
    
    # 1. Sidebar Logic
    st.sidebar.title(f"{APP_ICON} IntegrityOS")
    
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.session_state["role"] = None
    
    if st.session_state["logged_in"]:
        st.sidebar.divider()
        st.session_state["gemini_api_key"] = st.sidebar.text_input("Gemini API Key", type="password", help="Required for AI features")
        
        with st.sidebar.expander("SMTP Settings (Gmail)"):
            st.session_state["smtp_user"] = st.text_input("Email", placeholder="you@gmail.com")
            st.session_state["smtp_pass"] = st.text_input("App Password", type="password")
            st.session_state["smtp_host"] = "smtp.gmail.com"

        st.sidebar.divider()
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()

    # 2. Main Page Routing
    if not st.session_state["logged_in"]:
        show_login_page()
    else:
        if st.session_state["role"] == "CA":
            show_ca_dashboard()
        elif st.session_state["role"] == "Client":
            show_client_dashboard()

def show_login_page():
    st.title("Login to IntegrityOS")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Password", type="password", key="login_p")
        if st.button("Log In"):
            role = login_user(u, p)
            if role:
                st.session_state["logged_in"] = True
                st.session_state["username"] = u
                st.session_state["role"] = role
                st.success(f"Welcome back, {u} ({role})")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid credentials.")
        
        st.info("Demo Accounts:\n\nCA: `ca_admin` / `CApass123!`\n\nClient: `client_user` / `Clientpass123!`")

    with tab2:
        new_u = st.text_input("New Username")
        new_p = st.text_input("New Password", type="password")
        new_role = st.selectbox("Role", ["Client", "CA"])
        if st.button("Sign Up"):
            if signup_user(new_u, new_p, new_role):
                st.success("Account created! Please log in.")
            else:
                st.error("Username already exists.")

def show_ca_dashboard():
    st.title("CA Dashboard - Audit Command Center")
    
    st.sidebar.subheader("Data Input")
    data_source = st.sidebar.radio("Select Source", 
                                   ["Upload Excel", "Tally XML", "AI Scan PDF", "Live Tally ODBC"])
    
    module = st.sidebar.radio("Select Module", ["Smart Audit", "ESG Copilot", "Report Generator"])

    df = None
    pdf_text = None
    
    if data_source in ["Upload Excel", "Tally XML", "AI Scan PDF"]:
        uploaded_file = st.file_uploader(f"Upload {data_source} File")
        if uploaded_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{uploaded_file.name}")
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            log_file_upload(save_path, uploaded_file.name, st.session_state["username"], "CA", data_source)
            
            if data_source == "AI Scan PDF":
                raw_data = load_data(data_source, uploaded_file)
                if raw_data is not None and not raw_data.empty:
                    pdf_text = raw_data.iloc[0]["Raw_Text"]
                    st.success("PDF Text Extracted.")
            else:
                df = load_data(data_source, uploaded_file)
                if df is not None:
                    st.success(f"Loaded {len(df)} transactions.")
    
    elif data_source == "Live Tally ODBC":
        if st.button("Connect to Tally Prime"):
            df = load_data(data_source)
            if df is not None:
                st.success("Connected to Tally Live Data.")

    if module == "Smart Audit":
        st.header("🕵️ Smart Audit Detective")
        
        if df is not None and not df.empty:
            high_risk_df, risk_score = run_integrity_scan(df)
            c1, c2, c3, c4 = st.columns(4)
            rev = df[df['Amount'] > 0]['Amount'].sum()
            exp = df[df['Amount'] < 0]['Amount'].sum()
            pbc_list = generate_pbc_list(df)
            
            c1.metric("Total Credit", f"₹{rev:,.0f}")
            c2.metric("Total Debit", f"₹{exp:,.0f}")
            c3.metric("Integrity Risk Score", f"{risk_score}/100", delta="-Low Risk" if risk_score<30 else "+High Risk", delta_color="inverse")
            c4.metric("Pending Docs", len(pbc_list))
            
            st.divider()
            tab_viz, tab_risk, tab_pbc = st.tabs(["Trends", "High Risk Tx", "PBC Checklist"])
            
            with tab_viz:
                if 'Date' in df.columns:
                    daily_trend = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
                    daily_trend['Date'] = daily_trend['Date'].astype(str)
                    
                    if PLOTLY_AVAILABLE:
                        fig = px.bar(daily_trend, x='Date', y='Amount', title='Monthly Net Flow', template="plotly_dark")
                        fig.update_traces(marker_color='#22d3ee') # Bright Cyan Bars
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                            paper_bgcolor='rgba(0,0,0,0)', # Transparent paper
                            font_color='#e2e8f0' # Light text
                        )
                    else:
                        st.warning("Plotly not installed. Using basic Streamlit chart.")
                        st.bar_chart(daily_trend.set_index('Date')['Amount'])

            
            with tab_risk:
                st.subheader("🚩 Red Flag Transactions")
                if not high_risk_df.empty:
                    st.dataframe(high_risk_df[['Date', 'Ledger_Name', 'Amount', 'risk_reasons']])
                    tx_to_review = st.selectbox("Select Transaction to Mark Reviewed", high_risk_df.index)
                    if st.button("Mark as Reviewed"):
                        st.success(f"Transaction {tx_to_review} marked reviewed.")
                else:
                    st.success("No High Risk transactions detected.")
            
            with tab_pbc:
                st.subheader("Required Documents (PBC)")
                for item in pbc_list:
                    st.checkbox(item, value=False)
                    
        elif pdf_text:
            st.subheader("PDF AI Analysis")
            if st.button("Analyze PDF with Gemini"):
                summary = call_gemini_summary(pdf_text)
                st.markdown(summary)
        else:
            st.info("Please upload data or connect Tally to begin Audit.")

    elif module == "ESG Copilot":
        st.header("🌱 ESG & Sustainability Copilot")
        if df is not None:
            esg_data = esg_analysis(df)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Estimated CO2 Emissions", f"{esg_data['co2_tons']} Tons")
                st.caption(esg_data['explanation'])
            with col2:
                st.info("Emission Factor Source: Indian Grid Avg (0.82 kg/kWh)")
            st.divider()
            
            st.subheader("Vendor ESG Screening")
            vendors = df['Ledger_Name'].unique()
            vendor_data = []
            for v in vendors:
                v_str = str(v)
                status = "⚪ Neutral"
                if "Ltd" in v_str and "Green" not in v_str: status = "🟠 Moderate Risk"
                if any(x in v_str for x in ["Green", "Eco", "Solar"]): status = "🟢 Low Risk"
                if "Private" in v_str: status = "🔴 High Risk"
                vendor_data.append({"Vendor": v, "ESG Status": status})
            
            v_df = pd.DataFrame(vendor_data)
            st.dataframe(v_df, use_container_width=True)
            csv = v_df.to_csv(index=False)
            st.download_button("Download ESG Summary", csv, "esg_summary.csv", "text/csv")
        else:
            st.info("Upload financial data to calculate Carbon Footprint.")

    elif module == "Report Generator":
        st.header("📝 AI Report Studio")
        tabs = st.tabs(["BRSR Questionnaire", "CEO Memo Drafter"])
        
        with tabs[0]:
            st.subheader("BRSR (Business Responsibility) Generator")
            with st.form("brsr_form"):
                turnover = st.text_input("Annual Turnover")
                employees = st.number_input("Total Employees", min_value=1)
                csr_activity = st.text_area("CSR Activities conducted")
                
                if st.form_submit_button("Generate Report Draft"):
                    prompt = f"""
                    You are an experienced Indian CA. 
                    Create a Section A and Section B outline for a BRSR report based on:
                    Turnover: {turnover}, Employees: {employees}, CSR: {csr_activity}
                    """
                    if not st.session_state.get("gemini_api_key"):
                        st.error("AI Key missing.")
                    else:
                        try:
                            genai.configure(api_key=st.session_state["gemini_api_key"])
                            # FIX: AUTO DETECT
                            model_name = get_best_available_model()
                            model = genai.GenerativeModel(model_name)
                            response = model.generate_content(prompt)
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"AI Error: {e}")

        with tabs[1]:
            st.subheader("CEO Memo Drafter")
            memo_points = st.text_area("Key Observations", height=100)
            if st.button("Draft Memo with AI"):
                prompt = f"""
                You are a senior partner at an audit firm. 
                Draft a polite but firm memo to the CEO. Subject: Internal Audit Observations.
                Points: {memo_points}
                """
                if not st.session_state.get("gemini_api_key"):
                    st.error("AI Key missing.")
                else:
                    try:
                        genai.configure(api_key=st.session_state["gemini_api_key"])
                        # FIX: AUTO DETECT
                        model_name = get_best_available_model()
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(prompt)
                        st.session_state['generated_memo'] = response.text
                    except Exception as e:
                        st.error(f"AI Error: {e}")

            if 'generated_memo' in st.session_state:
                final_memo = st.text_area("Edit Memo", st.session_state['generated_memo'], height=300)
                client_email = st.text_input("Client Email")
                if st.button("Send via SMTP"):
                    if send_email_smtp(client_email, "Audit Memo", final_memo):
                        st.success("Email sent!")

def show_client_dashboard():
    st.title("Client Portal")
    st.info(f"Logged in as: {st.session_state['username']}")
    tab1, tab2 = st.tabs(["Upload Documents", "My Files"])
    
    with tab1:
        st.subheader("Pending Documents (PBC)")
        st.write("Please upload: 1. Rent Agreement, 2. Electricity Bill")
        uploaded_file = st.file_uploader("Select File", type=['xlsx', 'pdf', 'jpg', 'docx'])
        notes = st.text_input("Notes for Auditor")
        if uploaded_file and st.button("Submit to CA"):
            user_folder = os.path.join(UPLOAD_FOLDER, st.session_state['username'])
            if not os.path.exists(user_folder): os.makedirs(user_folder)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(user_folder, f"{timestamp}_{uploaded_file.name}")
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            log_file_upload(save_path, uploaded_file.name, st.session_state["username"], "Client", "PBC", notes)
            st.success("File uploaded successfully.")

    with tab2:
        st.subheader("Upload History")
        conn = sqlite3.connect(DB_FILE)
        my_files = pd.read_sql("SELECT original_filename, timestamp, notes FROM files WHERE uploader_username = ?", 
                               conn, params=(st.session_state['username'],))
        conn.close()
        if not my_files.empty:
            st.dataframe(my_files)
        else:
            st.write("No files uploaded yet.")

if __name__ == "__main__":
    main()
