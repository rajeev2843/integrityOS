import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURATION & STYLING (The Look)
# ==========================================
st.set_page_config(page_title="IntegrityOS", layout="wide", page_icon="🏛️")

# Custom CSS for "Teal & Navy" Professional Theme
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #F0F2F6;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #002B36; /* Navy */
    }
    
    /* The Metric Cards - FORCE TEXT COLOR */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #008080; /* Teal */
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Force the Label (Top text) to be Dark Navy */
    div[data-testid="stMetricLabel"] p {
        color: #002B36 !important;
        font-weight: 600;
    }
    
    /* Force the Value (Big Number) to be Dark Navy */
    div[data-testid="stMetricValue"] div {
        color: #002B36 !important;
    }

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
        # Configuration for "The Calculator"
        self.config = {
            "electricity_rate": 8.5,  # INR per kWh
            "scrap_steel_rate": 35.0, # INR per kg
            "co2_factor": 0.82        # kgCO2 per kWh
        }

    def generate_dummy_tb(self):
        """Generates a mock Trial Balance for demonstration."""
        data = {
            "GL_Code": ["GL001", "GL002", "GL003", "GL004", "GL005", "GL006", "GL007"],
            "Ledger_Name": [
                "Sales Account", 
                "MSEDCL Electricity Exp (Factory)", 
                "Scrap Sales (Metal)", 
                "Staff Welfare & PF", 
                "Regulatory Fines (GST)", 
                "Office Rent",
                "Donation to Local NGO"
            ],
            "Group": ["Revenue", "Expenses", "Revenue", "Expenses", "Expenses", "Expenses", "Expenses"],
            "Amount_INR": [5000000, 125000, 45000, 800000, 5000, 600000, 25000],
            "Type": ["Cr", "Dr", "Cr", "Dr", "Dr", "Dr", "Dr"]
        }
        return pd.DataFrame(data)

    def map_ledgers(self, df):
        """Zone 1: Tagging Ledgers (The Mapper)"""
        # Simple Keyword Mapping Logic (In production, use Regex or AI)
        def get_tag(name):
            name = name.lower()
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
        """Zone 3: The Calculator (Financial -> Non-Financial)"""
        
        # 1. Energy Calculation
        energy_rows = df[df['System_Tag'] == 'ENERGY_SCOPE2']
        total_energy_spend = energy_rows['Amount_INR'].sum()
        estimated_kwh = total_energy_spend / elec_rate
        estimated_emissions = (estimated_kwh * self.config['co2_factor']) / 1000 # Tons

        # 2. Waste Calculation
        waste_rows = df[df['System_Tag'] == 'WASTE_GENERATED']
        total_scrap_revenue = waste_rows['Amount_INR'].sum()
        estimated_waste_tons = (total_scrap_revenue / scrap_rate) / 1000

        return {
            "energy_spend": total_energy_spend,
            "kwh": estimated_kwh,
            "tco2": estimated_emissions,
            "waste_revenue": total_scrap_revenue,
            "waste_tons": estimated_waste_tons,
            "energy_audit_trail": energy_rows['Ledger_Name'].tolist() # The Audit Pin
        }

    def generate_pbc_checklist(self, df):
        """Zone 2: The PBC Automator"""
        checklist = []
        
        for index, row in df.iterrows():
            # Rule: High Value Rent
            if "rent" in row['Ledger_Name'].lower() and row['Amount_INR'] > 50000:
                checklist.append({
                    "Risk Level": "Medium",
                    "Ledger": row['Ledger_Name'],
                    "Required Document": "Lease/Rent Agreement (Valid FY24-25)",
                    "Trigger": f"Rent Expense > 50k (Actual: {row['Amount_INR']})"
                })
            
            # Rule: Donations (CSR)
            if row['System_Tag'] == "CSR_ACTIVITY":
                checklist.append({
                    "Risk Level": "High",
                    "Ledger": row['Ledger_Name'],
                    "Required Document": "NGO 80G Certificate & Board Resolution",
                    "Trigger": "CSR Classification"
                })

            # Rule: Fines
            if row['System_Tag'] == "GOVERNANCE_RISK":
                 checklist.append({
                    "Risk Level": "Critical",
                    "Ledger": row['Ledger_Name'],
                    "Required Document": "Notice Copy & Challan Payment Receipt",
                    "Trigger": "Regulatory Fine Detected"
                })
                
        return pd.DataFrame(checklist)

# ==========================================
# 3. UI LAYOUT (The Command Center)
# ==========================================

def main():
    engine = IntegrityEngine()

    # sidebar
    with st.sidebar:
        st.title("🏛️ IntegrityOS")
        st.caption("Financial-to-Non-Financial Bridge")
        st.markdown("---")
        
        st.header("1. Configuration")
        # Creating dynamic inputs for the 'Calculator'
        elec_rate = st.number_input("Avg Electricity Rate (₹/kWh)", value=8.5, step=0.1)
        scrap_rate = st.number_input("Scrap Metal Rate (₹/kg)", value=35.0, step=1.0)
        
        st.markdown("---")
        st.info("Status: System Ready 🟢")

    # Header
    st.title("Client: Acme Corp | FY 2024-25")
    
    # Init Data
    raw_df = engine.generate_dummy_tb()
    processed_df = engine.map_ledgers(raw_df)
    metrics = engine.run_estimation(processed_df, elec_rate, scrap_rate)
    pbc_df = engine.generate_pbc_checklist(processed_df)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Command Center", "📋 Smart PBC", "🌱 BRSR Builder"])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        st.subheader("Auditor's Integrity Scan")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Est. Scope 2 Emissions", f"{metrics['tco2']:.2f} Tons", delta="Derived from GL")
        with col2:
            st.metric("Waste Generated", f"{metrics['waste_tons']:.2f} Tons", delta="Derived from Sales")
        with col3:
            risk_count = len(pbc_df[pbc_df['Risk Level'] == 'Critical'])
            st.metric("Governance Risks", f"{risk_count} Flags", delta_color="inverse")

        st.markdown("---")
        st.subheader("Traceability Matrix (The Audit Pin)")
        st.markdown("This table shows how Financial Ledgers are mapped to ESG Categories.")
        st.dataframe(processed_df, use_container_width=True)

    # --- TAB 2: SMART PBC ---
    with tab2:
        st.subheader("Automated Requirement Checklist")
        st.markdown("IntegrityOS scanned the Trial Balance and identified these missing documents:")
        
        if not pbc_df.empty:
            st.table(pbc_df)
        else:
            st.success("No document requirements triggered.")

    # --- TAB 3: BRSR BUILDER ---
    with tab3:
        st.subheader("Principle 6 (Environment) - Draft")
        
        # Display the Calculation Logic (Transparency)
        st.info(f"💡 **Calculation Logic:** We found **₹{metrics['energy_spend']:,}** in Ledger(s) '{metrics['energy_audit_trail']}'. Using a tariff of **₹{elec_rate}/unit**, we estimated consumption.")

        # Simulate AI Narrative (Placeholder for Gemini)
        st.markdown("### 🤖 AI Narrative Generation (Gemini 1.5)")
        st.text_area("Draft Narrative", 
                     value=f"""
Essential Indicator 1: Energy Consumption
Based on financial records, the company's total electricity expenditure for FY 24-25 stood at INR {metrics['energy_spend']:,}. 
Applying the state industrial tariff rate, the estimated total energy consumption is {metrics['kwh']:.2f} kWh. 
This translates to approximately {metrics['tco2']:.2f} Tons of CO2 equivalent (Scope 2). 

Note: This data is derived from the 'MSEDCL Electricity Exp' ledger and is subject to verification against physical utility bills.
                     """, height=200)
        
        st.button("Regenerate Narrative with AI")

if __name__ == "__main__":
    main()
    
