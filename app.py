import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="IntegrityOS", layout="wide", page_icon="🏛️")

st.markdown("""
    <style>
    /* Main Background */
    .main { background-color: #F0F2F6; }
    
    /* Headings */
    h1, h2, h3 { color: #002B36; }
    
    /* The Metric Cards - FORCE TEXT COLOR */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #008080;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricLabel"] p { color: #002B36 !important; font-weight: 600; }
    div[data-testid="stMetricValue"] div { color: #002B36 !important; }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: 600;
        color: #002B36;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. THE LOGIC ENGINE (The Brain)
# ==========================================
class IntegrityEngine:
    def __init__(self):
        self.config = {
            "electricity_rate": 8.5,  # INR per kWh
            "scrap_steel_rate": 35.0, # INR per kg
            "co2_factor": 0.82        # kgCO2 per kWh
        }

    def generate_dummy_tb(self):
        """Generates a mock Trial Balance if no file is uploaded."""
        data = {
            "Ledger_Name": ["Sales Account", "MSEDCL Electricity Exp (Factory)", "Scrap Sales (Metal)", 
                            "Staff Welfare & PF", "Regulatory Fines (GST)", "Office Rent", "Donation to Local NGO"],
            "Group": ["Revenue", "Expenses", "Revenue", "Expenses", "Expenses", "Expenses", "Expenses"],
            "Amount_INR": [5000000, 125000, 45000, 800000, 5000, 600000, 25000],
        }
        return pd.DataFrame(data)

    def normalize_columns(self, df, map_dict):
        """Renames user columns to System Standard columns."""
        # map_dict looks like: {'User_Ledger_Col': 'Ledger_Name', ...}
        # Invert the dict to use in pandas rename
        rename_map = {v: k for k, v in map_dict.items()}
        
        # We need to ensure the user selected columns exist
        final_df = pd.DataFrame()
        final_df['Ledger_Name'] = df[map_dict['Ledger_Name']]
        final_df['Group'] = df[map_dict['Group']]
        final_df['Amount_INR'] = pd.to_numeric(df[map_dict['Amount_INR']], errors='coerce').fillna(0)
        
        return final_df

    def map_ledgers(self, df):
        """Zone 1: Tagging Ledgers (The Mapper)"""
        def get_tag(name):
            name = str(name).lower()
            if "electricity" in name or "power" in name: return "ENERGY_SCOPE2"
            if "scrap" in name: return "WASTE_GENERATED"
            if "welfare" in name or "pf" in name or "salary" in name: return "EMPLOYEE_BENEFITS"
            if "fine" in name or "penalty" in name: return "GOVERNANCE_RISK"
            if "donation" in name: return "CSR_ACTIVITY"
            if "rent" in name: return "INFRASTRUCTURE"
            return "GENERAL"

        df['System_Tag'] = df['Ledger_Name'].apply(get_tag)
        return df

    def run_estimation(self, df, elec_rate, scrap_rate):
        """Zone 3: The Calculator"""
        # 1. Energy
        energy_rows = df[df['System_Tag'] == 'ENERGY_SCOPE2']
        total_energy_spend = energy_rows['Amount_INR'].sum()
        estimated_kwh = total_energy_spend / elec_rate
        estimated_emissions = (estimated_kwh * self.config['co2_factor']) / 1000 

        # 2. Waste
        waste_rows = df[df['System_Tag'] == 'WASTE_GENERATED']
        total_scrap_revenue = waste_rows['Amount_INR'].sum()
        estimated_waste_tons = (total_scrap_revenue / scrap_rate) / 1000

        return {
            "energy_spend": total_energy_spend,
            "kwh": estimated_kwh,
            "tco2": estimated_emissions,
            "waste_revenue": total_scrap_revenue,
            "waste_tons": estimated_waste_tons,
            "energy_audit_trail": energy_rows['Ledger_Name'].tolist()
        }

    def generate_pbc_checklist(self, df):
        """Zone 2: The PBC Automator"""
        checklist = []
        for index, row in df.iterrows():
            if "rent" in str(row['Ledger_Name']).lower() and row['Amount_INR'] > 50000:
                checklist.append({"Risk": "Medium", "Ledger": row['Ledger_Name'], 
                                  "Doc": "Lease Agreement", "Reason": "Rent > 50k"})
            if row['System_Tag'] == "CSR_ACTIVITY":
                checklist.append({"Risk": "High", "Ledger": row['Ledger_Name'], 
                                  "Doc": "NGO 80G Cert", "Reason": "CSR Activity"})
            if row['System_Tag'] == "GOVERNANCE_RISK":
                 checklist.append({"Risk": "Critical", "Ledger": row['Ledger_Name'], 
                                   "Doc": "Challan Receipt", "Reason": "Regulatory Fine"})
        return pd.DataFrame(checklist)

# ==========================================
# 3. UI LAYOUT (The Command Center)
# ==========================================
def main():
    engine = IntegrityEngine()
    
    # --- SIDEBAR: CONTROLS & INGESTION ---
    with st.sidebar:
        st.title("🏛️ IntegrityOS")
        st.caption("Financial-to-Non-Financial Bridge")
        st.markdown("---")
        
        # A. File Upload
        st.subheader("📂 1. Data Source")
        uploaded_file = st.file_uploader("Upload Trial Balance", type=['xlsx', 'csv'])
        
        # B. Configuration
        st.subheader("⚙️ 2. Calibration")
        elec_rate = st.number_input("Avg Electricity Rate (₹/kWh)", value=8.5, step=0.1)
        scrap_rate = st.number_input("Scrap Metal Rate (₹/kg)", value=35.0, step=1.0)
        st.info("Status: System Ready 🟢")

    # --- DATA LOADING & MAPPING LOGIC ---
    if uploaded_file:
        try:
            # Load file based on extension
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            st.success(f"Loaded: {uploaded_file.name}")
            
            # THE MAPPER UI
            with st.expander("🛠️ Column Mapping (Standardize Data)", expanded=True):
                st.write("Map your file columns to IntegrityOS standards:")
                cols = raw_df.columns.tolist()
                
                c1, c2, c3 = st.columns(3)
                user_ledger = c1.selectbox("Select 'Ledger Name' Column", cols, index=0)
                user_amount = c2.selectbox("Select 'Amount' Column", cols, index=1 if len(cols)>1 else 0)
                user_group = c3.selectbox("Select 'Group/Type' Column", cols, index=2 if len(cols)>2 else 0)
                
                # Normalize Data
                mapping = {
                    'Ledger_Name': user_ledger,
                    'Amount_INR': user_amount,
                    'Group': user_group
                }
                
                if st.button("Apply Mapping & Run Integrity Engine"):
                    processed_df = engine.normalize_columns(raw_df, mapping)
                    processed_df = engine.map_ledgers(processed_df)
                    st.session_state['processed_df'] = processed_df
                    st.session_state['data_active'] = True
                
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        # Use Dummy Data if no upload
        raw_df = engine.generate_dummy_tb()
        processed_df = engine.map_ledgers(raw_df)
        st.session_state['processed_df'] = processed_df
        st.session_state['data_active'] = False # Just dummy mode

    # Check if we have valid data to show
    if 'processed_df' in st.session_state:
        df_display = st.session_state['processed_df']
        
        # Run Calculation Engine
        metrics = engine.run_estimation(df_display, elec_rate, scrap_rate)
        pbc_df = engine.generate_pbc_checklist(df_display)

        # --- MAIN DASHBOARD ---
        st.title("IntegrityOS Dashboard")
        if not st.session_state.get('data_active', False):
            st.caption("⚠️ Using DEMO DATA. Upload a file in the sidebar to analyze real client data.")

        tab1, tab2, tab3 = st.tabs(["📊 Command Center", "📋 Smart PBC", "🌱 BRSR Builder"])

        with tab1:
            st.subheader("Auditor's Integrity Scan")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Est. Scope 2 Emissions", f"{metrics['tco2']:.2f} Tons", delta="Derived from GL")
            with col2: st.metric("Waste Generated", f"{metrics['waste_tons']:.2f} Tons", delta="Derived from Sales")
            with col3: 
                risk_count = len(pbc_df[pbc_df['Risk'] == 'Critical'])
                st.metric("Governance Risks", f"{risk_count} Flags", delta_color="inverse")
            
            st.markdown("---")
            st.subheader("Traceability Matrix (The Audit Pin)")
            st.dataframe(df_display, use_container_width=True)

        with tab2:
            st.subheader("Automated Requirement Checklist")
            if not pbc_df.empty:
                st.table(pbc_df)
            else:
                st.success("No document requirements triggered.")

        with tab3:
            st.subheader("Principle 6 (Environment) - Draft")
            st.info(f"💡 **Logic:** Found ₹{metrics['energy_spend']:,} in Ledgers '{metrics['energy_audit_trail']}'. Rate: ₹{elec_rate}/unit.")
            st.text_area("Draft Narrative", value=f"Estimated consumption is {metrics['kwh']:.2f} kWh, resulting in {metrics['tco2']:.2f} Tons CO2.", height=150)

if __name__ == "__main__":
    main()
    
