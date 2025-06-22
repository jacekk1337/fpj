import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import PyPDF2
import pdfplumber
import camelot
from datetime import datetime
from io import BytesIO
import numpy as np
from typing import Dict, List, Union
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

#formatting data
def safe_convert(value):
#turn numbers to float, erase unnecessary extras, 0.0 is the standard
    if pd.isna(value) or value == '':
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace('$', '').replace(',', '').replace('%', '').strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0

def clean_dataframe(df):
#seperate dates from fiscal data
    for col in df.columns:
        if any(term in str(col).lower() for term in ['date', 'year', 'period']):
            continue
        df[col] = df[col].apply(safe_convert)
    return df

def generate_sample_pdf():
    """Generate a proper PDF file with sample financial data"""
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)

    # Add financial statement content
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, 750, "ABC COMPANY FINANCIAL STATEMENTS")

    p.setFont("Helvetica", 12)
    y_position = 720  # Starting Y position

    # Income Statement
    p.drawString(100, y_position, "INCOME STATEMENT (2023)")
    y_position -= 20
    for item, value in [
        ("Revenue:", "1,250,000"),
        ("Cost of Goods Sold:", "750,000"),
        ("Gross Profit:", "500,000"),
        ("Operating Expenses:", "300,000"),
        ("Net Income:", "200,000")
        ]:
        p.drawString(120, y_position, f"{item:25} ${value}")
        y_position -= 20

    # Balance Sheet
    p.showPage()  # New page
    y_position = 750
    p.drawString(100, y_position, "BALANCE SHEET (2023)")
    y_position -= 30

    p.drawString(100, y_position, "ASSETS")
    y_position -= 20
    for item, value in [
        ("Cash:", "150,000"),
        ("Accounts Receivable:", "75,000"),
        ("Inventory:", "125,000"),
        ("Total Current Assets:", "350,000"),
        ("Fixed Assets:", "450,000"),
        ("TOTAL ASSETS:", "800,000")
        ]:
        p.drawString(120, y_position, f"{item:25} ${value}")
        y_position -= 20

    p.save()
    buffer.seek(0)
    return buffer

def extract_financial_data_from_pdf(pdf_file):
    """Improved PDF extraction with better error handling"""
    try:
        # First verify it's a valid PDF
        try:
            PyPDF2.PdfReader(pdf_file)
        except Exception as e:
            st.error(f"Invalid PDF file: {str(e)}")
            return None

        extracted_data = {'combined_data': {}}

        # Try text extraction first
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    for line in text.split('\n'):
                        line_lower = line.lower()
                        for category in FINANCIAL_TERMS.values():
                            for terms in category.values():
                                for term in terms:
                                    if term in line_lower:
                                        # Extract numbers from line
                                        numbers = re.findall(r'[-+]?[\d,]+\.?\d*', line)
                                        if numbers:
                                            try:
                                                value = float(numbers[-1].replace(',', ''))
                                                extracted_data['combined_data'][term] = value
                                            except ValueError:
                                                continue
        return extracted_data if extracted_data['combined_data'] else None

    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return None

def pdf_data_to_dataframe(pdf_data: Dict) -> pd.DataFrame:
    """
    Improved DataFrame conversion that handles the financial terms data
    """
    if not pdf_data:
        return pd.DataFrame()

    # Prefer the combined data (now includes financial_terms)
    if pdf_data.get('combined_data', {}):
        # Create single-period DataFrame
        data = {}
        for key, value in pdf_data['combined_data'].items():
            if isinstance(value, list):
                data[key] = value
            else:
                data[key] = [value]  # Wrap single values in list

        df = pd.DataFrame(data)
        return clean_dataframe(df)

    # Fallback to tables if no combined data
    for table in pdf_data.get('tables', []):
        try:
            df = pd.DataFrame(table['data'])
            return clean_dataframe(df)
        except:
            continue

    return pd.DataFrame()

# --- Session State Initialization ---
if 'current_section' not in st.session_state:
    st.session_state.current_section = 'main'
if 'cashflow_data' not in st.session_state:
    st.session_state.cashflow_data = {}
if 'balancesheet_data' not in st.session_state:
    st.session_state.balancesheet_data = {}
if 'ratio_data' not in st.session_state:
    st.session_state.ratio_data = {
        'liquidity': {},
        'profitability': {},
        'leverage': {},
        'efficiency': {}
    }
if 'history' not in st.session_state:
    st.session_state.history = []

#financial terms to be used (for future reference, this could be expanded even further)
FINANCIAL_TERMS = {
    'balance_sheet': {
        'assets': ['assets', 'current assets', 'fixed assets', 'property', 'equipment', 'inventory'],
        'liabilities': ['liabilities', 'current liabilities', 'long-term debt', 'accounts payable'],
        'equity': ['equity', 'shareholders equity', 'retained earnings']
        },
    'income_statement': {
        'revenue': ['revenue', 'sales', 'income'],
        'expenses': ['expenses', 'cost of goods sold', 'cogs', 'operating expenses'],
        'profit': ['net income', 'profit', 'ebitda', 'earnings']
        },
    'cash_flow': {
        'operating': ['operating activities', 'cash from operations'],
        'investing': ['investing activities', 'capital expenditures'],
        'financing': ['financing activities', 'dividends', 'debt issued']
    }
}

#basic general benchmarks
INDUSTRY_BENCHMARKS = {
    'Retail': {
        'current_ratio': 1.5,
        'quick_ratio': 0.8,
        'profit_margin': 0.03,
        'debt_equity': 1.2,
        'inventory_turnover': 8
        },
    'Manufacturing': {
        'current_ratio': 2.0,
        'quick_ratio': 1.2,
        'profit_margin': 0.08,
        'debt_equity': 0.8,
        'inventory_turnover': 5
        },
    'Technology': {
        'current_ratio': 2.5,
        'quick_ratio': 2.0,
        'profit_margin': 0.15,
        'debt_equity': 0.5,
        'inventory_turnover': 12
    }
}

# --- Financial Analysis Functions ---
def detect_financial_statement(df):
#assessment of the uploaded document, what statement and its contents are implemented
    df = clean_dataframe(df.copy())

    detected = {
        'balance_sheet': {},
        'income_statement': {},
        'cash_flow': {},
        'periods': [],
        'is_time_series': False
    }

#is the app able to make a trend line based on available dates or not
    date_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['date', 'year', 'period'])]
    if date_cols:
        detected['is_time_series'] = True
        try:
            detected['periods'] = pd.to_datetime(df[date_cols[0]]).dt.strftime('%Y-%m-%d').tolist()
        except:
            detected['periods'] = df[date_cols[0]].astype(str).tolist()

    #scan for financial compoments
    for col in df.columns:
        if col in date_cols:
            continue

        col_lower = str(col).lower()

        # Balance sheet items (gathered from businessmadeeasy.xyz, my high school notes for business management :))
        for category, terms in FINANCIAL_TERMS['balance_sheet'].items():
            if any(term in col_lower for term in terms):
                detected['balance_sheet'][col] = df[col].values

        # Income statement items
        for category, terms in FINANCIAL_TERMS['income_statement'].items():
            if any(term in col_lower for term in terms):
                detected['income_statement'][col] = df[col].values

        # Cash flow items
        for category, terms in FINANCIAL_TERMS['cash_flow'].items():
            if any(term in col_lower for term in terms):
                detected['cash_flow'][col] = df[col].values

    return detected

def generate_visualizations(detected_data):
#generate visualizations, this one was really difficult, streamlit api is a life saver
    figures = []

    if detected_data['is_time_series']:
        periods = detected_data['periods']

        if detected_data['balance_sheet']:
            bs_df = pd.DataFrame({
                'Period': periods,
                **{k: v for k, v in detected_data['balance_sheet'].items()}
            })

            fig = px.area(bs_df, x='Period', y=list(detected_data['balance_sheet'].keys()),
                          title="Balance Sheet Composition Over Time")
            figures.append(fig)

            if any('current assets' in k.lower() for k in detected_data['balance_sheet']) and \
               any('current liabilities' in k.lower() for k in detected_data['balance_sheet']):
                current_assets_col = [k for k in detected_data['balance_sheet'] if 'current assets' in k.lower()][0]
                current_liabilities_col = [k for k in detected_data['balance_sheet'] if 'current liabilities' in k.lower()][0]

                working_capital = bs_df[current_assets_col] - bs_df[current_liabilities_col]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=periods, y=working_capital, name='Working Capital'))
                fig.update_layout(title="Working Capital Trend")
                figures.append(fig)

    if detected_data['income_statement']:
        is_df = pd.DataFrame({
            'Category': list(detected_data['income_statement'].keys()),
            'Amount': [values[0] for values in detected_data['income_statement'].values()]
        })

        fig = px.pie(is_df, names='Category', values='Amount', 
                     title="Income Statement Composition")
        figures.append(fig)

        if any('revenue' in k.lower() for k in detected_data['income_statement']) and \
           any('net income' in k.lower() for k in detected_data['income_statement']):
            revenue_col = [k for k in detected_data['income_statement'] if 'revenue' in k.lower()][0]
            net_income_col = [k for k in detected_data['income_statement'] if 'net income' in k.lower()][0]

            profit_margin = detected_data['income_statement'][net_income_col][0] / detected_data['income_statement'][revenue_col][0]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=profit_margin*100,
                title={'text': "Profit Margin (%)"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            figures.append(fig)

    return figures
def generate_enhanced_visualizations(detected_data, source_type="pdf"):
    """
    Generate enhanced visualizations specifically for PDF-extracted data
    Includes financial health gauges and term-specific charts
    """
    figures = []
    
    # 1. Financial Health Gauge
    if detected_data.get('income_statement') and detected_data.get('balance_sheet'):
        ratios = calculate_ratios(detected_data)
        if ratios:
            # Profit Margin Gauge
            if 'profit_margin' in ratios:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ratios['profit_margin']*100,
                    title={'text': "Profit Margin (%)"},
                    gauge={'axis': {'range': [0, 100]}}
                ))
                figures.append(fig)
            
            # Current Ratio Gauge
            if 'current_ratio' in ratios:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ratios['current_ratio'],
                    title={'text': "Current Ratio"},
                    gauge={'axis': {'range': [0, 5]}}
                ))
                figures.append(fig)
    
    # 2. Financial Term Value Bars
    if detected_data.get('balance_sheet'):
        bs_terms = list(detected_data['balance_sheet'].keys())
        bs_values = [detected_data['balance_sheet'][k][0] for k in bs_terms]
        fig = px.bar(x=bs_terms, y=bs_values, title="Balance Sheet Components")
        figures.append(fig)
    
    if detected_data.get('income_statement'):
        is_terms = list(detected_data['income_statement'].keys())
        is_values = [detected_data['income_statement'][k][0] for k in is_terms]
        fig = px.bar(x=is_terms, y=is_values, title="Income Statement Components")
        figures.append(fig)
    
    return figures
def calculate_ratios(detected_data):
#automatic ratio assessment (would need upgrading in the future)
    ratios = {}

    # Liquidity ratios
    current_assets_cols = [k for k in detected_data['balance_sheet'] if 'current assets' in k.lower()]
    current_liab_cols = [k for k in detected_data['balance_sheet'] if 'current liabilities' in k.lower()]

    if current_assets_cols and current_liab_cols:
        current_assets = safe_convert(detected_data['balance_sheet'][current_assets_cols[0]][0])
        current_liabilities = safe_convert(detected_data['balance_sheet'][current_liab_cols[0]][0])

        if current_liabilities != 0:
            ratios['current_ratio'] = current_assets / current_liabilities

        # Quick ratio
        cash_cols = [k for k in detected_data['balance_sheet'] if 'cash' in k.lower()]
        receivables_cols = [k for k in detected_data['balance_sheet'] if 'receivable' in k.lower()]

        if cash_cols and receivables_cols:
            cash = safe_convert(detected_data['balance_sheet'][cash_cols[0]][0])
            receivables = safe_convert(detected_data['balance_sheet'][receivables_cols[0]][0])
            if current_liabilities != 0:
                ratios['quick_ratio'] = (cash + receivables) / current_liabilities

    # Profitability ratios
    revenue_cols = [k for k in detected_data['income_statement'] if 'revenue' in k.lower()]
    net_income_cols = [k for k in detected_data['income_statement'] if 'net income' in k.lower()]

    if revenue_cols and net_income_cols:
        revenue = safe_convert(detected_data['income_statement'][revenue_cols[0]][0])
        net_income = safe_convert(detected_data['income_statement'][net_income_cols[0]][0])

        if revenue != 0:
            ratios['profit_margin'] = net_income / revenue

    # Leverage ratios
    total_liab_cols = [k for k in detected_data['balance_sheet'] if 'total liabilities' in k.lower()]
    total_equity_cols = [k for k in detected_data['balance_sheet'] if 'total equity' in k.lower()]

    if total_liab_cols and total_equity_cols:
        total_liabilities = safe_convert(detected_data['balance_sheet'][total_liab_cols[0]][0])
        total_equity = safe_convert(detected_data['balance_sheet'][total_equity_cols[0]][0])

        if total_equity != 0:
            ratios['debt_to_equity'] = total_liabilities / total_equity

    return ratios
#health score, just a broad analysis based on standard points of reference
def generate_financial_health_score(ratios):
    weights = {
        'current_ratio': 0.25,
        'quick_ratio': 0.25,
        'profit_margin': 0.20,
        'debt_equity': 0.15,
        'inventory_turnover': 0.15
    }

    score = 0
    if ratios.get('current_ratio', 0) > 2: score += weights['current_ratio'] * 100
    elif ratios.get('current_ratio', 0) > 1.5: score += weights['current_ratio'] * 80
    elif ratios.get('current_ratio', 0) > 1: score += weights['current_ratio'] * 60
    else: score += weights['current_ratio'] * 30

    if ratios.get('quick_ratio', 0) > 1.5: score += weights['quick_ratio'] * 100
    elif ratios.get('quick_ratio', 0) > 1: score += weights['quick_ratio'] * 80
    elif ratios.get('quick_ratio', 0) > 0.5: score += weights['quick_ratio'] * 60
    else: score += weights['quick_ratio'] * 30

    if ratios.get('profit_margin', 0) > 0.15: score += weights['profit_margin'] * 100
    elif ratios.get('profit_margin', 0) > 0.1: score += weights['profit_margin'] * 80
    elif ratios.get('profit_margin', 0) > 0.05: score += weights['profit_margin'] * 60
    else: score += weights['profit_margin'] * 30

    if ratios.get('debt_equity', 0) < 0.5: score += weights['debt_equity'] * 100
    elif ratios.get('debt_equity', 0) < 1: score += weights['debt_equity'] * 80
    elif ratios.get('debt_equity', 0) < 1.5: score += weights['debt_equity'] * 60
    else: score += weights['debt_equity'] * 30

    if ratios.get('inventory_turnover', 0) > 10: score += weights['inventory_turnover'] * 100
    elif ratios.get('inventory_turnover', 0) > 6: score += weights['inventory_turnover'] * 80
    elif ratios.get('inventory_turnover', 0) > 3: score += weights['inventory_turnover'] * 60
    else: score += weights['inventory_turnover'] * 30

    return min(100, int(score))
#in the future i would like to use the openAI api but that would cost money since prompts are limited in the free version.
def generate_insights(ratios, industry='Retail'):
    insights = []

    cr = ratios.get('current_ratio', 0)
    if cr > 2:
        insights.append("Strong current ratio indicates good short-term financial health")
    elif cr > 1:
        insights.append("Current ratio is acceptable but could be improved")
    else:
        insights.append("Low current ratio indicates potential liquidity problems")

    qr = ratios.get('quick_ratio', 0)
    if qr > 1:
        insights.append("Healthy quick ratio shows good ability to meet short-term obligations")
    else:
        insights.append("Quick ratio suggests reliance on inventory to meet short-term debts")

    pm = ratios.get('profit_margin', 0)
    benchmark_pm = INDUSTRY_BENCHMARKS.get(industry, {}).get('profit_margin', 0)
    if pm > benchmark_pm * 1.5:
        insights.append(f"Excellent profit margin ({pm:.1%}) compared to industry average ({benchmark_pm:.1%})")
    elif pm > benchmark_pm:
        insights.append(f"Profit margin ({pm:.1%}) is above industry average ({benchmark_pm:.1%}) but could improve")
    else:
        insights.append(f"Profit margin ({pm:.1%}) below industry average ({benchmark_pm:.1%}) - consider cost reduction")

    de = ratios.get('debt_equity', 0)
    benchmark_de = INDUSTRY_BENCHMARKS.get(industry, {}).get('debt_equity', 1)
    if de < benchmark_de * 0.7:
        insights.append("Conservative debt levels provide financial flexibility")
    elif de < benchmark_de:
        insights.append("Debt levels are reasonable but monitor closely")
    else:
        insights.append("High debt-to-equity ratio increases financial risk")

    return insights

def convert_dict_to_excel(data_dict, sheet_name='Sheet1'):
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if any(isinstance(v, dict) for v in data_dict.values()):
                flat_data = {}
                for k, v in data_dict.items():
                    if isinstance(v, dict):
                        for subk, subv in v.items():
                            flat_data[f"{k} - {subk}"] = [subv]
                    else:
                        flat_data[k] = [v]
                df = pd.DataFrame(flat_data)
            else:
                df = pd.DataFrame([data_dict])
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        output.seek(0)
        return output.getvalue()
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None

# --- Main Page ---
if st.session_state.current_section == 'main':
    st.title("Jacek's App")

    with st.expander("How to use this app"):
        st.write("""
        1. Choose a financial statement to create or analyze
        2. Enter your financial data in the provided fields
        3. Save your data to generate insights and visualizations
        4. Compare against industry benchmarks
        5. Download reports in multiple formats
        """)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Cash Flow Statement"):
            st.session_state.current_section = 'cashflow'
            st.rerun()
    with col2:
        if st.button("Balance Sheet"):
            st.session_state.current_section = 'balancesheet'
            st.rerun()
    with col3:
        if st.button("Ratio Analysis"):
            st.session_state.current_section = 'ratios'
            st.rerun()

    st.divider()

    with st.expander("📋 File Format Guidelines"):
        st.write("""
        **For best results, please ensure your file:**
        - Uses standard financial terms in headers
        - Contains numeric values without symbols ($1,000 → 1000)
        - Has one row per period for time series analysis
        - Avoids merged cells or special formatting
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Existing Excel template
            template_data = {
                'Period': ['2023-12-31', '2022-12-31'],
                'Revenue': [1500000, 1200000],
                'Cost of Goods Sold': [900000, 750000],
                'Net Income': [200000, 150000],
                'Current Assets': [500000, 450000],
                'Total Liabilities': [300000, 350000]
            }
            template_df = pd.DataFrame(template_data)
            
            st.download_button(
                label="Download Excel Template",
                data=template_df.to_csv(index=False),
                file_name="financial_template.csv",
                mime="text/csv"
            )
            st.caption("For CSV/Excel uploads")
    
        with col2:
            # Generate a proper PDF file
            pdf_buffer = generate_sample_pdf()
            st.download_button(
                label="Download PDF Template",
                data=pdf_buffer,
                file_name="financial_template.pdf",
                mime="application/pdf"
            )
            st.caption("For PDF uploads")

    # Move the file uploader inside the main section
    st.subheader("Upload Financial Data")
    uploaded_file = st.file_uploader("Choose CSV/Excel/PDF file", 
                                     type=["csv", "xlsx", "pdf"], 
                                   key="main_uploader")

    # Now process the uploaded file
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.pdf'):
                # Process PDF file
                pdf_data = extract_financial_data_from_pdf(uploaded_file)
                if pdf_data is None:
                    st.error("Failed to extract data from PDF. Please try another file.")
                    st.stop()

                df = pdf_data_to_dataframe(pdf_data)
                if df.empty:
                    st.error("No financial data found in PDF. Please check the file format.")
                    st.stop()

                st.write("Extracted PDF data preview:", df.head())

                detected_data = detect_financial_statement(df)

                # Store the data in session state
                if detected_data['balance_sheet']:
                    st.session_state.balancesheet_data = detected_data['balance_sheet']
                if detected_data['income_statement']:
                    st.session_state.ratio_data['profitability'] = {
                        'Profit Margin': calculate_ratios(detected_data).get('profit_margin', 0)
                    }
                if detected_data['cash_flow']:
                    st.session_state.cashflow_data = detected_data['cash_flow']

                st.header("Automated Financial Analysis from PDF")
                st.subheader("Detected Components")
                st.write(detected_data)

                st.subheader("Enhanced Visualizations")
                figures = generate_enhanced_visualizations(detected_data, source_type="pdf")
                for fig in figures:
                    st.plotly_chart(fig, use_container_width=True)

            else:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                df = clean_dataframe(df)
                st.write("Data preview:", df.head())

                detected_data = detect_financial_statement(df)

                if detected_data['balance_sheet']:
                    st.session_state.balancesheet_data = detected_data['balance_sheet']
                if detected_data['income_statement']:
                    st.session_state.ratio_data['profitability'] = {
                        'Profit Margin': calculate_ratios(detected_data).get('profit_margin', 0)
                    }
                if detected_data['cash_flow']:
                    st.session_state.cashflow_data = detected_data['cash_flow']

                st.header("Automated Financial Analysis")
                st.subheader("Detected Components")
                st.write(detected_data)

                st.subheader("Generated Visualizations")
                figures = generate_visualizations(detected_data)
                for fig in figures:
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Key Financial Ratios")
                ratios = calculate_ratios(detected_data)
                if ratios:
                    ratio_df = pd.DataFrame({
                        'Ratio': list(ratios.keys()),
                        'Value': list(ratios.values())
                    })
                    st.dataframe(ratio_df)

                    st.subheader("Financial Insights")
                    insights = generate_insights(ratios)
                    for insight in insights:
                        if "Excellent" in insight or "Strong" in insight:
                            st.success(insight)
                        elif "Potential" in insight or "Low" in insight:
                            st.error(insight)
                        else:
                            st.warning(insight)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# --- Cash Flow Section ---
elif st.session_state.current_section == 'cashflow':
    st.title("Cash Flow Statement")

    with st.form("cashflow_form"):
        st.subheader("Operating Activities")
        op_activities = st.number_input("Net Cash from Operations", step=1000.0, key='cf_op_activities')

        st.subheader("Investing Activities")
        inv_activities = st.number_input("Net Cash from Investing", step=1000.0, key='cf_inv_activities')

        st.subheader("Financing Activities")
        fin_activities = st.number_input("Net Cash from Financing", step=1000.0, key='cf_fin_activities')

        net_change = op_activities + inv_activities + fin_activities
        st.metric("Net Cash Flow", value=f"${net_change:,.2f}")

        if st.form_submit_button("Save"):
            st.session_state.cashflow_data = {
                'Operating Activities': op_activities,
                'Investing Activities': inv_activities,
                'Financing Activities': fin_activities,
                'Net Change in Cash': net_change
            }
            st.session_state.history.append({
                'type': 'cashflow',
                'data': st.session_state.cashflow_data,
                'timestamp': datetime.now()
            })
            st.success("Cash Flow data saved!")

    if st.session_state.cashflow_data:
        st.subheader("Cash Flow Visualization")
        cashflow_df = pd.DataFrame({
            'Activity': ['Operating', 'Investing', 'Financing'],
            'Amount': [
                st.session_state.cashflow_data['Operating Activities'],
                st.session_state.cashflow_data['Investing Activities'],
                st.session_state.cashflow_data['Financing Activities']
            ]
        })

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(cashflow_df, x='Activity', y='Amount', 
                         title="Cash Flow by Activity")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(cashflow_df, names='Activity', values='Amount',
                         title="Cash Flow Composition")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Export Options")
        excel_data = convert_dict_to_excel(st.session_state.cashflow_data, "CashFlow")
        if excel_data:
            st.download_button(
                label="Download as Excel",
                data=excel_data,
                file_name=f"cash_flow_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    if st.button("Back to Main Menu"):
        st.session_state.current_section = 'main'
        st.rerun()

# --- Balance Sheet Section ---
elif st.session_state.current_section == 'balancesheet':
    st.title("Balance Sheet")

    with st.form("balancesheet_form"):
        industry = st.selectbox("Select your industry", 
                                list(INDUSTRY_BENCHMARKS.keys()),
                              key='bs_industry')

        st.subheader("Enter Title & Caption")
        title = st.text_input("Statement Title", "Balance Sheet for ABC Corp.", key='bs_title')
        caption = st.text_input("Year Ended", "for year ended 2013", key='bs_caption')

        st.subheader("Current Assets")
        cash = st.number_input("Cash", min_value=0.0, step=1000.0, key='bs_cash')
        stocks = st.number_input("Stocks", min_value=0.0, step=1000.0, key='bs_stocks')
        debtors = st.number_input("Debtors", min_value=0.0, step=1000.0, key='bs_debtors')
        current_assets_total = cash + stocks + debtors

        st.subheader("Current Liabilities")
        creditors = st.number_input("Creditors", min_value=0.0, step=1000.0, key='bs_creditors')
        tax = st.number_input("Tax", min_value=0.0, step=1000.0, key='bs_tax')
        dividends = st.number_input("Dividends", min_value=0.0, step=1000.0, key='bs_dividends')
        current_liabilities_total = creditors + tax + dividends

        net_current_assets = current_assets_total - current_liabilities_total

        st.subheader("Fixed Assets")
        fixed_assets = st.number_input("Fixed Assets", min_value=0.0, step=1000.0, key='bs_fixed_assets')

        net_assets = net_current_assets + fixed_assets

        st.subheader("Financed By")
        loan_capital = st.number_input("Loan Capital", min_value=0.0, step=1000.0, key='bs_loan_capital')
        share_capital = st.number_input("Share Capital", min_value=0.0, step=1000.0, key='bs_share_capital')
        retained_earnings = st.number_input("Retained Earnings", min_value=0.0, step=1000.0, key='bs_retained_earnings')
        capital_employed = loan_capital + share_capital + retained_earnings

        if st.form_submit_button("Save"):
            st.session_state.balancesheet_data = {
                'Title': title,
                'Period': caption,
                'Industry': industry,
                'Fixed Assets': fixed_assets,
                'Current Assets': {
                    'Cash': cash,
                    'Stocks': stocks,
                    'Debtors': debtors,
                    'Total': current_assets_total
                    },
                'Current Liabilities': {
                    'Creditors': creditors,
                    'Tax': tax,
                    'Dividends': dividends,
                    'Total': current_liabilities_total
                    },
                'Net Current Assets': net_current_assets,
                'Net Assets': net_assets,
                'Financed By': {
                    'Loan Capital': loan_capital,
                    'Share Capital': share_capital,
                    'Retained Earnings': retained_earnings
                    },
                'Capital Employed': capital_employed
            }
            st.session_state.history.append({
                'type': 'balancesheet',
                'data': st.session_state.balancesheet_data,
                'timestamp': datetime.now()
            })
            st.success("Balance Sheet data saved!")

    if st.session_state.balancesheet_data:
        st.subheader("Balance Sheet Visualization")

        assets_data = {
            'Category': ['Cash', 'Stocks', 'Debtors', 'Fixed Assets'],
            'Amount': [
                st.session_state.balancesheet_data['Current Assets']['Cash'],
                st.session_state.balancesheet_data['Current Assets']['Stocks'],
                st.session_state.balancesheet_data['Current Assets']['Debtors'],
                st.session_state.balancesheet_data['Fixed Assets']
            ]
        }
        fig = px.pie(pd.DataFrame(assets_data), names='Category', values='Amount',
                     title="Assets Composition")
        st.plotly_chart(fig, use_container_width=True)

        position_data = {
            'Category': ['Total Assets', 'Total Liabilities', 'Capital'],
            'Amount': [
                st.session_state.balancesheet_data['Net Assets'],
                st.session_state.balancesheet_data['Current Liabilities']['Total'],
                st.session_state.balancesheet_data['Capital Employed']
            ]
        }
        fig = px.bar(pd.DataFrame(position_data), x='Category', y='Amount',
                     title="Financial Position")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Export Options")
        excel_data = convert_dict_to_excel(st.session_state.balancesheet_data, "BalanceSheet")
        if excel_data:
            st.download_button(
                label="Download as Excel",
                data=excel_data,
                file_name=f"balance_sheet_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    if st.button("Back to Main Menu"):
        st.session_state.current_section = 'main'
        st.rerun()

# --- Ratio Analysis Section ---
elif st.session_state.current_section == 'ratios':
    st.title("Financial Ratio Analysis")

    industry = st.selectbox("Select your industry", 
                            list(INDUSTRY_BENCHMARKS.keys()),
                          key='ratio_industry')

    with st.form("ratio_form"):
        st.header("Liquidity Ratios")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Current Ratio")
            current_assets = st.number_input("Current Assets", min_value=0.0, key='rat_current_assets')
            current_liabilities = st.number_input("Current Liabilities", min_value=0.01, key='rat_current_liabilities')
            current_ratio = current_assets / current_liabilities if current_liabilities else 0
            st.metric("Current Ratio", 
                      value=f"{current_ratio:.2f}",
                     delta=f"Industry avg: {INDUSTRY_BENCHMARKS[industry]['current_ratio']:.2f}")

        with col2:
            st.subheader("Quick Ratio")
            quick_assets = st.number_input("Quick Assets", min_value=0.0, key='rat_quick_assets')
            quick_ratio = quick_assets / current_liabilities if current_liabilities else 0
            st.metric("Quick Ratio", 
                      value=f"{quick_ratio:.2f}",
                     delta=f"Industry avg: {INDUSTRY_BENCHMARKS[industry]['quick_ratio']:.2f}")

        st.header("Profitability Ratios")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Profit Margin")
            net_income = st.number_input("Net Income", step=1000.0, key='rat_net_income')
            revenue = st.number_input("Revenue", min_value=0.01, step=1000.0, key='rat_revenue')
            profit_margin = net_income / revenue if revenue else 0
            st.metric("Profit Margin", 
                      value=f"{profit_margin:.1%}",
                     delta=f"Industry avg: {INDUSTRY_BENCHMARKS[industry]['profit_margin']:.1%}")

        with col4:
            st.subheader("Return on Equity (ROE)")
            shareholders_equity = st.number_input("Shareholders' Equity", min_value=0.01, step=1000.0, key='rat_shareholders_equity')
            roe = net_income / shareholders_equity if shareholders_equity else 0
            st.metric("ROE", 
                      value=f"{roe:.1%}",
                     delta="Higher is better")

        st.header("Leverage Ratios")
        st.subheader("Debt-to-Equity Ratio")
        total_debt = st.number_input("Total Liabilities", min_value=0.0, step=1000.0, key='rat_total_debt')
        debt_equity = total_debt / shareholders_equity if shareholders_equity else 0
        st.metric("D/E Ratio", 
                  value=f"{debt_equity:.2f}",
                 delta=f"Industry avg: {INDUSTRY_BENCHMARKS[industry]['debt_equity']:.2f}")

        st.header("Efficiency Ratios")
        st.subheader("Inventory Turnover")
        cogs = st.number_input("Cost of Goods Sold", min_value=0.0, step=1000.0, key='rat_cogs')
        avg_inventory = st.number_input("Average Inventory", min_value=0.01, step=1000.0, key='rat_avg_inventory')
        inventory_turnover = cogs / avg_inventory if avg_inventory else 0
        st.metric("Inventory Turnover", 
                  value=f"{inventory_turnover:.1f}",
                 delta=f"Industry avg: {INDUSTRY_BENCHMARKS[industry]['inventory_turnover']:.1f}")

        if st.form_submit_button("Save Ratios"):
            st.session_state.ratio_data = {
                'liquidity': {
                    'Current Ratio': current_ratio,
                    'Quick Ratio': quick_ratio
                    },
                'profitability': {
                    'Profit Margin': profit_margin,
                    'ROE': roe
                    },
                'leverage': {
                    'Debt/Equity': debt_equity
                    },
                'efficiency': {
                    'Inventory Turnover': inventory_turnover
                }
            }
            st.session_state.history.append({
                'type': 'ratios',
                'data': st.session_state.ratio_data,
                'timestamp': datetime.now()
            })
            st.success("All ratios saved!")

    if st.session_state.ratio_data['liquidity']:
        health_score = generate_financial_health_score({
            'current_ratio': st.session_state.ratio_data['liquidity']['Current Ratio'],
            'quick_ratio': st.session_state.ratio_data['liquidity']['Quick Ratio'],
            'profit_margin': st.session_state.ratio_data['profitability']['Profit Margin'],
            'debt_equity': st.session_state.ratio_data['leverage']['Debt/Equity'],
            'inventory_turnover': st.session_state.ratio_data['efficiency']['Inventory Turnover']
        })

        st.subheader("Financial Health Score")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Score", f"{health_score}/100")
        with col2:
            st.progress(health_score)

        if health_score >= 80:
            st.success("Excellent financial health!")
        elif health_score >= 60:
            st.warning("Moderate financial health - some areas need improvement")
        else:
            st.error("Poor financial health - significant improvements needed")

        st.subheader("Financial Insights")
        insights = generate_insights(st.session_state.ratio_data, industry)
        for insight in insights:
            if "Excellent" in insight or "Strong" in insight:
                st.success(insight)
            elif "Potential" in insight or "Low" in insight:
                st.error(insight)
            else:
                st.warning(insight)

        st.subheader("Benchmark Comparison")
        ratios = st.session_state.ratio_data
        benchmark_data = {
            'Ratio': ['Current Ratio', 'Quick Ratio', 'Profit Margin', 'Debt/Equity', 'Inventory Turnover'],
            'Your Value': [
                ratios['liquidity']['Current Ratio'],
                ratios['liquidity']['Quick Ratio'],
                ratios['profitability']['Profit Margin'],
                ratios['leverage']['Debt/Equity'],
                ratios['efficiency']['Inventory Turnover']
                ],
            'Industry Average': [
                INDUSTRY_BENCHMARKS[industry]['current_ratio'],
                INDUSTRY_BENCHMARKS[industry]['quick_ratio'],
                INDUSTRY_BENCHMARKS[industry]['profit_margin'],
                INDUSTRY_BENCHMARKS[industry]['debt_equity'],
                INDUSTRY_BENCHMARKS[industry]['inventory_turnover']
            ]
        }
        fig = px.bar(pd.DataFrame(benchmark_data).melt(id_vars='Ratio'), 
                     x='Ratio', y='value', color='variable',
                    barmode='group', title="Your Ratios vs Industry Average")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Back to Main Menu"):
        st.session_state.current_section = 'main'
        st.rerun()