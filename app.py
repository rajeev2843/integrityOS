# app.py
"""
IntegrityOS - The CA Super App
------------------------------
Dependencies:
# pip install streamlit pandas numpy pyodbc PyPDF2 google-generativeai plotly openpyxl
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
import hashlib
from email.message import EmailMessage

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

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #020617 0%, #0f172a 100%); color: #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #020617; border-right: 1px solid #1e293b; }
    h1, h2, h3, h4, h5 { color: #22d3ee !important; font-family: 'Segoe UI', sans-serif; }
    div[data-testid="metric-container"] { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; }
    .stButton>button { background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%); color: white; border: none; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stTextArea>div>div>textarea { background-color: #1e293b; color: #ffffff; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

# --- SECURITY UTILS ---

def generate_password_hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def check_password_hash(stored_hash: str, password: str) -> bool:
    return stored_hash == hashlib.sha256(password.encode()).hexdigest()

# --- DATABASE & AUTHENTICATION UTILS ---

def create_db_and_seed():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')

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

    try:
        c.execute("SELECT * FROM users WHERE username = ?", ("ca_admin",))
        if not c.fetchone():
            pw_hash = generate_password_hash("CApass123!")
            c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                      ("ca_admin", pw_hash, "CA"))

        c.execute("SELECT * FROM users WHERE username = ?", ("client_user",))
        if not c.fetchone():
            pw_hash = generate_password_hash("Clientpass123!")
            c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                      ("client_user", pw_hash, "Client"))
        conn.commit()
    except Exception as e:
        logging.error(f"DB Seeding Error: {e}")
    finally:
        conn.close()

def login_user(username, password):
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
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""INSERT INTO files 
                 (filename_on_disk, original_filename, uploader_username, role_uploaded_as, file_type, notes)
                 VALUES (?, ?, ?, ?, ?, ?)""",
              (filename_disk, original_name, uploader, role, ftype, note))
    conn.commit()
    conn.close()

# --- EXTERNAL SERVICES (AI, EMAIL) ---

def call_gemini_summary(text: str) -> str:
    api_key = st.session_state.get("gemini_api_key")
    if not api_key:
        return "⚠️ Please enter Google Gemini API Key in the sidebar."

    try:
        genai.configure(api_key=api_key)
        # FIX: Using 'gemini-pro' which is standard
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        You are an expert Chartered Accountant. Analyze this text:
        {text[:8000]} 
        Summarize key financial figures and red flags.
        """ 
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Fallback debug info
        error_msg = str(e)
        if "404" in error_msg:
             return f"Error: Model not found. Please check your API Key permissions. (Details: {e})"
        return f"AI Service Error: {e}"

def send_email_smtp(to_email, subject, body):
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
        return False

# --- DATA LOADING ---

def load_data(source_type: str, uploaded_file=None) -> pd.DataFrame:
    df = pd.DataFrame()
    
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
                        rows.append({"Date": date_str, "Ledger_Name": l_name, "Voucher_Type": v_type, "Amount": float(amt), "Narration": narration})
            
            if rows:
                df = pd.DataFrame(rows)
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
                return df
            return pd.DataFrame(columns=["Date", "Ledger_Name", "Voucher_Type", "Amount", "Narration"])
        except Exception as e:
            st.error(f"Error parsing XML: {e}")
            return None

    elif source_type == "Live Tally ODBC":
        st.info("Attempting connection to Tally Prime via ODBC...")
        try:
            conn_str = "DRIVER={Tally ODBC Driver64};Server=localhost;PORT=9000;"
            conn = pyodbc.connect(conn_str, timeout=2)
            query = "SELECT $Date, $LedgerName, $VoucherTypeName, $Amount, $Narration FROM LedgerDetails"
            df = pd.read_sql(query, conn)
            df.rename(columns={"$Date": "Date", "$LedgerName": "Ledger_Name", "$VoucherTypeName": "Voucher_Type", "$Amount": "Amount", "$Narration": "Narration"}, inplace=True)
            conn.close()
            return df
        except Exception as e:
            st.error("Tally Connection Failed. Check if Tally is running on Port 9000.")
            return None

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

# --- ANALYSIS FUNCTIONS ---

def run_integrity_scan(df: pd.DataFrame):
    if df.empty or 'Amount' not in df.columns:
        return pd.DataFrame(), 0

    df = df.copy()
    df['risk_reasons'] = ""
    df['is_risky'] = False
    df['Amount_Abs'] = df['Amount'].abs()
    if 'Date' in df.columns:
        df['DayOfWeek'] = df['Date'].dt.dayofweek 

    mask_round = (df['Amount_Abs'] > 0) & ((df['Amount_Abs'] % 1000 == 0) | (df['Amount_Abs'] % 500 == 0))
    df.loc[mask_round, 'risk_reasons'] += "Round Number; "
    df.loc[mask_round, 'is_risky'] = True

    if 'Date' in df.columns:
        mask_weekend = df['DayOfWeek'].isin([5, 6])
        df.loc[mask_weekend, 'risk_reasons'] += "Weekend Entry; "
        df.loc[mask_weekend, 'is_risky'] = True

    mask_high = df['Amount_Abs'] > 50000
    df.loc[mask_high, 'risk_reasons'] += "High Value (>50k); "
    df.loc[mask_high, 'is_risky'] = True

    high_risk_df = df[df['is_risky'] == True].copy()
    
    total_tx = len(df)
    risky_tx = len(high_risk_df)
    risk_score = 0
    if total_tx > 0:
        ratio_count = risky_tx / total_tx
        total_vol = df['Amount_Abs'].sum()
        risky_vol = high_risk_df['Amount_Abs'].sum()
        ratio_vol = risky_vol / total_vol if total_vol > 0 else 0
        risk_score = int((ratio_count * 50) + (ratio_vol * 50))
        
    return high_risk_df, min(100, risk_score)

def generate_pbc_list(df: pd.DataFrame):
    if df.empty or "Ledger_Name" not in df.columns: return []
    ledgers = df['Ledger_Name'].astype(str).str.lower().unique()
    requirements = set()
    for l in ledgers:
        if "rent" in l: requirements.add("Rent Agreement")
        if any(x in l for x in ["electricity", "power"]): requirements.add("Utility Bills")
        if any(x in l for x in ["legal", "advocate"]): requirements.add("Case Files / Legal Notices")
        if "salary" in l: requirements.add("Payroll / PT Challans")
    return list(requirements)

def esg_analysis(df: pd.DataFrame):
    results = {"estimated_units": 0, "co2_tons": 0, "explanation": "No data"}
    if df.empty or "Ledger_Name" not in df.columns: return results
    mask = df['Ledger_Name'].astype(str).str.contains('Electricity|Power', case=False, regex=True)
    subset = df[mask]
    total_spend = subset['Amount'].abs().sum()
    if total_spend > 0:
        estimated_units = total_spend / 8.0
        co2_tons = (estimated_units * 0.82) / 1000.0
        results = {"estimated_units": round(estimated_units, 2), "co2_tons": round(co2_tons, 4), "explanation": f"Spent ₹{total_spend:,.2f}"}
    return results

# --- MAIN APP ---

def main():
    create_db_and_seed()
    st.sidebar.title(f"{APP_ICON} IntegrityOS")
    
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.session_state["role"] = None
    
    if st.session_state["logged_in"]:
        st.sidebar.divider()
        st.session_state["gemini_api_key"] = st.sidebar.text_input("Gemini API Key", type="password")
        
        with st.sidebar.expander("SMTP Settings"):
            st.session_state["smtp_user"] = st.text_input("Email")
            st.session_state["smtp_pass"] = st.text_input("App Password", type="password")
            st.session_state["smtp_host"] = "smtp.gmail.com"

        st.sidebar.divider()
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()

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
                st.session_state.update({"logged_in": True, "username": u, "role": role})
                st.rerun()
            else:
                st.error("Invalid credentials.")
        st.info("Demo: `ca_admin` / `CApass123!` or `client_user` / `Clientpass123!`")
    
    with tab2:
        nu, np_ = st.text_input("New Username"), st.text_input("New Password", type="password")
        nr = st.selectbox("Role", ["Client", "CA"])
        if st.button("Sign Up") and signup_user(nu, np_, nr):
            st.success("Account created!")

def show_ca_dashboard():
    st.title("CA Dashboard - Audit Command Center")
    st.sidebar.subheader("Data Input")
    ds = st.sidebar.radio("Source", ["Upload Excel", "Tally XML", "AI Scan PDF", "Live Tally ODBC"])
    mod = st.sidebar.radio("Module", ["Smart Audit", "ESG Copilot", "Report Generator"])
    df, pdf_text = None, None
    
    if ds in ["Upload Excel", "Tally XML", "AI Scan PDF"]:
        uf = st.file_uploader(f"Upload {ds}")
        if uf:
            save_path = os.path.join(UPLOAD_FOLDER, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uf.name}")
            with open(save_path, "wb") as f: f.write(uf.getbuffer())
            log_file_upload(save_path, uf.name, st.session_state["username"], "CA", ds)
            if ds == "AI Scan PDF":
                raw = load_data(ds, uf)
                if raw is not None: pdf_text = raw.iloc[0]["Raw_Text"]
                st.success("PDF Extracted.")
            else:
                df = load_data(ds, uf)
                if df is not None: st.success(f"Loaded {len(df)} tx.")
    elif ds == "Live Tally ODBC":
        if st.button("Connect Tally"):
            df = load_data(ds)
            if df is not None: st.success("Connected.")

    if mod == "Smart Audit":
        st.header("🕵️ Smart Audit")
        if df is not None:
            high_risk, score = run_integrity_scan(df)
            c1, c2, c3 = st.columns(3)
            c1.metric("Credits", f"₹{df[df['Amount']>0]['Amount'].sum():,.0f}")
            c2.metric("Debits", f"₹{df[df['Amount']<0]['Amount'].sum():,.0f}")
            c3.metric("Risk Score", f"{score}/100")
            
            t1, t2 = st.tabs(["Trends", "Red Flags"])
            with t1:
                if 'Date' in df.columns:
                    trend = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
                    trend['Date'] = trend['Date'].astype(str)
                    if PLOTLY_AVAILABLE:
                        st.plotly_chart(px.bar(trend, x='Date', y='Amount', template="plotly_dark"))
                    else:
                        st.bar_chart(trend.set_index('Date'))
            with t2:
                st.dataframe(high_risk)
        elif pdf_text:
            if st.button("Analyze PDF"): st.markdown(call_gemini_summary(pdf_text))

    elif mod == "ESG Copilot":
        st.header("🌱 ESG Copilot")
        if df is not None:
            res = esg_analysis(df)
            st.metric("CO2 Emissions", f"{res['co2_tons']} Tons", res['explanation'])

    elif mod == "Report Generator":
        st.header("📝 AI Reports")
        t1, t2 = st.tabs(["BRSR", "CEO Memo"])
        
        with t1:
            with st.form("brsr"):
                turnover = st.text_input("Turnover")
                if st.form_submit_button("Generate"):
                    try:
                        genai.configure(api_key=st.session_state.get("gemini_api_key"))
                        model = genai.GenerativeModel('gemini-pro')
                        st.markdown(model.generate_content(f"Draft BRSR report for turnover {turnover}").text)
                    except Exception as e:
                        st.error(f"AI Error: {e}")
        
        with t2:
            pts = st.text_area("Observations")
            if st.button("Draft Memo"):
                try:
                    genai.configure(api_key=st.session_state.get("gemini_api_key"))
                    model = genai.GenerativeModel('gemini-pro')
                    st.session_state['memo'] = model.generate_content(f"Draft CEO memo on: {pts}").text
                except Exception as e:
                    st.error(f"AI Error: {e}")
            
            if 'memo' in st.session_state:
                final = st.text_area("Edit", st.session_state['memo'])
                if st.button("Send Email") and send_email_smtp(st.text_input("To"), "Audit Memo", final):
                    st.success("Sent.")

    
def show_client_dashboard():
    st.title("Client Portal")
    uploaded_file = st.file_uploader("Upload PBC Doc")
    if uploaded_file and st.button("Submit"):
        save_path = os.path.join(UPLOAD_FOLDER, st.session_state['username'], uploaded_file.name)
        # Ensure dir exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f: f.write(uploaded_file.getbuffer())
        log_file_upload(save_path, uploaded_file.name, st.session_state["username"], "Client", "PBC")
        st.success("Uploaded.")

if __name__ == "__main__":
    main()

