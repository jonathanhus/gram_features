import streamlit as st
import os
import pandas as pd

# Set page layout
st.set_page_config(layout="wide")
st.title("GramBank Feature Identification")
st.markdown("This page allows you to review the contents of the data for the files.")
st.markdown("""
- Value: GramBank value
- Comment: Comment from GramBank coder
- code: Value determined via LLM
- commment: Comment determined via LLM
            
*Note: A number of columns have been hidden from the table, but they are available in the Selected Row Details section*
            """)
st.divider()

# Sidebar for filtering options
st.sidebar.header("Experiment Results")

# Language selection in sidebar
language = st.sidebar.selectbox(
    'Pick language',
    ('iloko', 'kalamang', 'minangkabau', 'mizo', 'southern_jinghpaw')
)

base_dir = f"../outputs/{language}"

# Function to select a file
def file_selector(folder_path=base_dir):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()

# Load baseline values
df_values = pd.read_csv('../resources/grambank/values.csv')

# Handle cases for 4 or 5 columns
four_cols = ['ID', 'code', 'comment', 'source']
five_cols = ['ID', 'code', 'comment', 'source', 'prompt_info']

df_exp = pd.read_json(filename, orient='index').reset_index()
if df_exp.shape[1] == len(four_cols):
    df_exp.columns = four_cols
elif df_exp.shape[1] == len(five_cols):
    df_exp.columns = five_cols

df_exp['code'] = df_exp['code'].astype(str)

# Merge data
merged_results = pd.merge(df_values, df_exp, on="ID")
merged_results['Value'] = merged_results['Value'].astype(str)
merged_results['code'] = merged_results['code'].astype(str)

# Remove unwanted columns
# merged_results = merged_results.drop(columns=['source'])

st.sidebar.markdown(f"**Number of Features:** {len(merged_results)}")

# Sidebar filters
st.sidebar.header("Filters")
if merged_results['Value'].str.isnumeric().all():
    merged_results['Value'] = merged_results['Value'].astype(float)
    min_value, max_value = merged_results['Value'].min(), merged_results['Value'].max()
    value_range = st.sidebar.slider('Filter by Value (GramBank Baseline)', min_value, max_value, (min_value, max_value))
    merged_results = merged_results[merged_results['Value'].between(value_range[0], value_range[1])]
else:
    unique_values = merged_results['Value'].unique().tolist()
    selected_values = st.sidebar.multiselect('Filter by Value (GramBank Baseline)', unique_values, default=unique_values)
    merged_results = merged_results[merged_results['Value'].isin(selected_values)]

codes = merged_results['code'].unique().tolist()
selected_codes = st.sidebar.multiselect('Filter by Code (Experiment)', codes, default=codes)

# Apply filters
filtered_results = merged_results[merged_results['code'].isin(selected_codes)].reset_index(drop=True)

# Add count of results in sidebar
st.sidebar.markdown(f"**Number of Results:** {len(filtered_results)}")

# Display table with row selection
st.subheader("Filtered Data")
columns_to_hide = ['Language_ID', 'Parameter_ID', 'Source', 'Source_comment', 'source']
visible_columns = [col for col in filtered_results.columns if col not in columns_to_hide]
event = st.dataframe(filtered_results[visible_columns], selection_mode='single-row', on_select='rerun')

# Display selected row details
st.subheader("Selected Row Details")
st.markdown("**Select row in table above to get details**")
details = event.selection.rows

if details:
    selected_data = filtered_results.iloc[details[0]].to_dict()  # Get first selected row
    st.json(selected_data)  # Show row details in JSON format

# Add bar charts
st.sidebar.subheader("Value Distribution")
value_counts = filtered_results['Value'].value_counts()  # Count occurrences of unique 'Value'
st.sidebar.bar_chart(value_counts)

st.sidebar.subheader("Code Distribution")
value_counts = filtered_results['code'].value_counts()  # Count occurrences of unique 'Value'
st.sidebar.bar_chart(value_counts)