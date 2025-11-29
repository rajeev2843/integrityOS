import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import sqlite3
import hashlib
import datetime
import pyodbc
import io

# --- 1. CONFIGURATION & VISUAL THEME ---
st.set_page_config(
    page_title="IntegrityOS | The Glass Audit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# THEME: "Cyber-Professional" (Deep Navy & Teal)
# Matching the high-fidelity visual benchmark provided.
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #0A1128; /* Deep Navy */
        color: #E2E8F0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1E2749;
    }
    
    /* Metric Cards (Glassmorphism) */
    div[data-testid="metric-container"] {
        background-color: #1E2749;
        border: 1px solid #2C3E50;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #00D9C0; /* Teal Glow */
        transform: translateY(-2px);
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #00D9C0 !important;
        font-family: 'Inter', sans-serif;
    }
    p, label {
        color: #94A3B8 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00D9C0 0%, #00B4D8 100%);
        color: #0A1128;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        height: 45px;
    }
    .stButton>button:hover {
        box-shadow: 0 0 10px #00D9C0;
        color: white;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E2749;
        border-radius: 5px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00D9C0 !important;
        color: #0A1128 !important;
        font-weight: bold;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #1E2749;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. AUTHENTICATION & DATABASE ---

def init_db():
    """Initializes in-memory SQLite for demo purposes."""
    # FIX: check_same_thread=False is required for Streamlit's environment
    conn = sqlite3.connect(':memory:', check_same_thread=False) 
    c = conn.cursor()
    c.execute('''CREATE TABLE users (username TEXT, password TEXT, role TEXT)''')
    # Seed Data
    c.execute("INSERT INTO users VALUES ('admin', '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918', 'CA')") # admin123
    c.execute("INSERT INTO users VALUES ('client', '62608e08adc29a8d6dbc9754e659f125514049680b7f6ce5c0c91176968bc56f', 'Client')") # client123
    conn.commit()
    return conn

def hash_pass(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login(conn, username, password):
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username=? AND password=?", (username, hash_pass(password)))
    return c.fetchone()

# --- 3. THE UNIVERSAL DATA ADAPTER (The Bridge) ---

class DataIngestor:
    @staticmethod
    def normalize_columns(df):
        """Maps varied ERP column names to IntegrityOS Standard Schema."""
        # Standard Schema: [Date, Ledger, Voucher_Type, Amount, Narration]
        col_map = {
            'Date': 'Date', 'Voucher Date': 'Date', 'Txn Date': 'Date',
            'Particulars': 'Ledger', 'Ledger Name': 'Ledger', 'Account': 'Ledger',
            'Vch Type': 'Voucher_Type', 'Voucher Type': 'Voucher_Type',
            'Debit': 'Amount', 'Credit': 'Amount', 'Amount': 'Amount', 'Value': 'Amount',
            'Narration': 'Narration', 'Description': 'Narration', 'Remarks': 'Narration'
        }
        
        df = df.rename(columns=col_map)
        
        # Ensure required columns exist
        required = ['Date', 'Ledger', 'Voucher_Type', 'Amount', 'Narration']
        for col in required:
            if col not in df.columns:
                df[col] = "" if col == 'Narration' else 0
                
        # Fill NaNs
        df['Narration'] = df['Narration'].fillna('')
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        return df[required]

    @staticmethod
    def get_tally_data():
        """Attempts live Tally ODBC connection."""
        try:
            conn_str = "DRIVER={Tally ODBC Driver64};SERVER=localhost;PORT=9000"
            conn = pyodbc.connect(conn_str, timeout=3)
            query = "SELECT $Date, $LedgerName, $VoucherTypeName, $Amount, $Narration FROM LedgerDetails"
            df = pd.read_sql(query, conn)
            df.columns = ['Date', 'Ledger', 'Voucher_Type', 'Amount', 'Narration']
            return df, "Success"
        except Exception:
            return None, "Connection Failed"

# --- 4. THE FORENSIC DETECTIVE (Layer 1) ---

class ForensicEngine:
    def __init__(self, df):
        self.df = df
        self.findings = []

    def run_scans(self):
        self._check_greenwashing()
        self._check_benford()
        self._check_weekend_entries()
        return pd.DataFrame(self.findings)

    def _check_greenwashing(self):
        """
        The 'Gotcha' Feature: Checks for Green Ledgers with Dirty Narrations.
        """
        green_terms = ['Solar', 'Green', 'Renewable', 'Waste Mgmt', 'Eco']
        dirty_terms = ['Diesel', 'Coal', 'Fuel', 'Generator', 'Petrol']
        
        # Logic: Ledger says "Green", Narration says "Dirty"
        for index, row in self.df.iterrows():
            ledger = str(row['Ledger'])
            narration = str(row['Narration'])
            
            if any(g in ledger for g in green_terms) and any(d in narration for d in dirty_terms):
                self.findings.append({
                    "Txn_ID": f"TXN-{index}",
                    "Date": row['Date'],
                    "Risk_Score": 95,
                    "Category": "Greenwashing",
                    "Description": f"Suspicious: Ledger '{ledger}' contains '{narration}'",
                    "Amount": row['Amount']
                })

    def _check_benford(self):
        """Simplified Benford's Law Check on Amount leading digits."""
        # Logic: Flag amounts starting with '9' if they appear too often
        # (Simplified for demo speed)
        amounts = self.df[self.df['Amount'] > 0]['Amount'].astype(str)
        leading_digits = [s[0] for s in amounts if s[0].isdigit()]
        if leading_digits:
            nines = leading_digits.count('9')
            ratio = nines / len(leading_digits)
            if ratio > 0.10: # Expected is ~4.6%
                self.findings.append({
                    "Txn_ID": "STAT-001",
                    "Date": datetime.date.today(),
                    "Risk_Score": 60,
                    "Category": "Statistical Anomaly",
                    "Description": f"Benford Violation: Unusual frequency of amounts starting with 9 ({ratio:.1%})",
                    "Amount": 0
                })

    def _check_weekend_entries(self):
        """Flags high value transactions on Sundays."""
        for index, row in self.df.iterrows():
            if row['Date'].weekday() == 6 and row['Amount'] > 50000:
                 self.findings.append({
                    "Txn_ID": f"TXN-{index}",
                    "Date": row['Date'],
                    "Risk_Score": 75,
                    "Category": "Temporal Anomaly",
                    "Description": "High value transaction posted on a Sunday",
                    "Amount": row['Amount']
                })

# --- 5. THE BRSR ESTIMATOR (Layer 2) ---

class BRSRBuilder:
    def __init__(self, df):
        self.df = df
        
    def estimate_energy(self, tariff_rate=8.5):
        """Converts Financial Data to Physical Units."""
        # Filter for Electricity
        energy_df = self.df[self.df['Ledger'].str.contains('Electric|Power|Utility', case=False, na=False)]
        total_spend = energy_df['Amount'].sum()
        
        # Estimation Logic
        estimated_kwh = total_spend / tariff_rate
        scope2_emissions = estimated_kwh * 0.82 / 1000 # 0.82 kgCO2/kWh (Grid Factor) -> Tons
        
        return {
            "spend": total_spend,
            "kwh": estimated_kwh,
            "emissions": scope2_emissions,
            "source_ledgers": energy_df['Ledger'].unique().tolist()
        }

    def generate_ai_draft(self, metrics, api_key):
        """Uses Gemini to draft the report."""
        if not api_key:
            return "⚠️ Please enter Gemini API Key in the Sidebar to generate the narrative."
            
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            You are an ESG Auditor. Based on this financial data, write the 'Principle 6: Environment' section of the BRSR report.
            
            Data:
            - Electricity Spend: ₹{metrics['spend']:,.2f}
            - Estimated Consumption: {metrics['kwh']:,.2f} kWh
            - Scope 2 Emissions: {metrics['emissions']:.2f} Tons CO2e
            - Source Ledgers: {metrics['source_ledgers']}
            
            Format: Professional, Regulatory tone. Mention that data is estimated from financial records.
            Output as Markdown.
            """
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI Error: {str(e)}"

# --- 6. MAIN APPLICATION LOGIC ---

def main():
    # Initialize Session
    if 'db' not in st.session_state:
        st.session_state.db = init_db()
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # --- LOGIN SCREEN ---
    if not st.session_state.logged_in:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.title("🛡️ IntegrityOS")
            st.markdown("### The Glass Audit Protocol")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Authenticate", use_container_width=True):
                user = login(st.session_state.db, username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.role = user[0]
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Access Denied. Try admin/admin123")
        return

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("IntegrityOS")
        st.caption(f"User: {st.session_state.username} | Role: {st.session_state.role}")
        st.divider()
        
        nav = st.radio("Navigation", ["Dashboard", "Detective", "BRSR Builder", "Smart PBC"])
        
        st.divider()
        api_key = st.text_input("Gemini API Key", type="password", placeholder="Required for AI")
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # --- DATA INGESTION (Global) ---
    # We load data here so it's available across tabs
    if 'data' not in st.session_state:
        st.session_state.data = None

    # --- DASHBOARD TAB ---
    if nav == "Dashboard":
        st.header(f"👋 Welcome, {st.session_state.username}")
        
        # 1. Ingestion Widget
        with st.expander("📂 Data Ingestion (Universal Adapter)", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                # File Upload
                uploaded_file = st.file_uploader("Upload Trial Balance (Excel)", type=['xlsx'])
                if uploaded_file:
                    raw_df = pd.read_excel(uploaded_file)
                    st.session_state.data = DataIngestor.normalize_columns(raw_df)
                    st.success(f"Ingested {len(st.session_state.data)} rows via Universal Adapter")
            with col_b:
                # Live Tally
                st.info("Live Connection")
                if st.button("Sync with Tally Prime (ODBC)"):
                    df, status = DataIngestor.get_tally_data()
                    if df is not None:
                        st.session_state.data = df
                        st.success(f"Synced {len(df)} rows from Tally")
                    else:
                        st.warning("Tally not found on localhost:9000. Using demo mode?")

        # 2. Key Metrics (Glassmorphism Cards)
        if st.session_state.data is not None:
            df = st.session_state.data
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Revenue Velocity", f"₹{df[df['Amount']>0]['Amount'].sum()/1e5:.1f}L", "+12%")
            m2.metric("Audit Readiness", "67%", "+5%")
            m3.metric("Risk Alerts", "3 Critical", "High Priority", delta_color="inverse")
            m4.metric("Est. Carbon", "124 Tons", "Scope 2")
            
            # 3. Visuals
            st.subheader("Transaction Velocity")
            # Simple aggregation for chart
            if 'Date' in df.columns:
                chart_data = df.groupby('Date')['Amount'].sum().reset_index()
                fig = go.Figure(data=go.Scatter(x=chart_data['Date'], y=chart_data['Amount'], 
                                              mode='lines+markers', line=dict(color='#00D9C0', width=3)))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                font=dict(color='white'), height=300)
                st.plotly_chart(fig, use_container_width=True)

    # --- DETECTIVE TAB ---
    elif nav == "Detective":
        st.header("🔍 Forensic Detective (Layer 1)")
        
        if st.session_state.data is not None:
            engine = ForensicEngine(st.session_state.data)
            findings_df = engine.run_scans()
            
            if not findings_df.empty:
                st.warning(f"Detected {len(findings_df)} Anomalies")
                
                # Filter
                severity = st.selectbox("Filter Risk", ["All", "Critical (Greenwashing)", "High", "Medium"])
                
                # Findings Table
                st.dataframe(
                    findings_df.style.apply(lambda x: ['background-color: #3d1010' if x.Risk_Score > 80 else '' for i in x], axis=1),
                    use_container_width=True
                )
                
                # The "Gotcha" Evidence Panel
                st.divider()
                st.subheader("Evidence Vault")
                selected_txn = st.selectbox("Inspect Transaction", findings_df['Txn_ID'].unique())
                details = findings_df[findings_df['Txn_ID'] == selected_txn].iloc[0]
                
                e1, e2 = st.columns(2)
                with e1:
                    st.markdown(f"**Description:** {details['Description']}")
                    st.markdown(f"**Amount:** ₹{details['Amount']:,.2f}")
                with e2:
                    st.error("Risk Assessment: High Probability of Material Misstatement")
            else:
                st.success("No anomalies detected in current dataset.")
        else:
            st.info("Please ingest data in Dashboard first.")

    # --- BRSR BUILDER TAB ---
    elif nav == "BRSR Builder":
        st.header("🌱 BRSR Builder (Layer 2)")
        
        if st.session_state.data is not None:
            builder = BRSRBuilder(st.session_state.data)
            
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.subheader("1. Estimator Settings")
                tariff = st.slider("Avg Electricity Tariff (₹/kWh)", 4.0, 15.0, 8.5)
                metrics = builder.estimate_energy(tariff)
                
                st.markdown("### Calculated Physical Units")
                st.metric("Energy Cost", f"₹{metrics['spend']:,.0f}")
                st.metric("Est. Consumption", f"{metrics['kwh']:,.0f} kWh")
                st.metric("Scope 2 Emissions", f"{metrics['emissions']:.2f} Tons")
                
                st.caption("Source Ledgers:")
                st.code(metrics['source_ledgers'])
                
            with c2:
                st.subheader("2. AI Report Drafter")
                if st.button("Generate Principle 6 Narrative"):
                    with st.spinner("Gemini is drafting regulatory text..."):
                        draft = builder.generate_ai_draft(metrics, api_key)
                        st.markdown(draft)
                        st.success("Draft Generated. Hover over numbers to see 'Audit Pins' (Simulated).")
                        
                        # Simulated Audit Pin
                        st.info("📌 Audit Pin: The value '120.58 Tons' is traced to GL-405 'Factory Power'.")
        else:
            st.info("Please ingest data first.")

    # --- SMART PBC TAB ---
    elif nav == "Smart PBC":
        st.header("📋 Smart Audit Room")
        
        if st.session_state.data is not None:
            # Auto-generate checklist based on triggers
            df = st.session_state.data
            checklist = []
            
            # Trigger Logic
            if df['Ledger'].str.contains('Rent').any():
                checklist.append({"Item": "Rent Agreement", "Reason": "Rent Expenses found in TB", "Status": "Pending"})
            if df['Amount'].sum() > 10000000: # 1 Cr
                checklist.append({"Item": "Tax Audit Report (3CD)", "Reason": "Turnover > 1Cr", "Status": "Pending"})
            if df['Ledger'].str.contains('Loan').any():
                checklist.append({"Item": "Bank Sanction Letter", "Reason": "Secured Loans found", "Status": "Uploaded"})
                
            check_df = pd.DataFrame(checklist)
            
            col_list, col_client = st.columns(2)
            
            with col_list:
                st.subheader("Auto-Generated Requirements")
                if not check_df.empty:
                    st.data_editor(check_df, num_rows="dynamic", use_container_width=True)
                
                st.button("Send Magic Link via WhatsApp 📱")
            
            with col_client:
                st.subheader("Client Portal Simulation")
                st.file_uploader("Client Upload Zone (Drag & Drop)", accept_multiple_files=True)
                st.success("Sanction_Letter_HDFC.pdf analyzed by AI: 'Limit ₹5Cr, Interest 9.5%'")

if __name__ == "__main__":
    main()
                
