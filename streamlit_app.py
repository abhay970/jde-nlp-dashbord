# Import packages
import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSQLException
import pandas as pd
import re
import yaml
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import uuid
import time

# Page configuration
st.set_page_config(
    page_title="JDE Analytics Platform",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'show_table_manager' not in st.session_state:
    st.session_state.show_table_manager = False

# Constants
JDE_STAGE = "JDE.TESTDTA.JDE_STAGE"
YAML_CONFIG = "JDE.TESTDTA.jde_stage/jde.yaml"
DATA_SOURCE_NAME = "JDE.TESTDTA"

# User-friendly error messages for NLP Dashboard
ERROR_MESSAGES = {
    "no_data": "📞 **Data not available** - Please contact your administrator for assistance.",
    "access_denied": "📞 **Access restricted** - Please contact your administrator for permissions.",
    "config_error": "📞 **System configuration issue** - Please contact your administrator.",
    "general_error": "📞 **Technical issue encountered** - Please contact your administrator for support."
}

# Sidebar Navigation
st.sidebar.title("🏠 Navigation")
st.sidebar.markdown("---")

if st.sidebar.button("🧠 NLP to Dashboard", use_container_width=True, type="primary" if not st.session_state.show_table_manager else "secondary"):
    st.session_state.show_table_manager = False

if st.sidebar.button("🛠️ JDE Table Manager & YAML Generator", use_container_width=True, type="primary" if st.session_state.show_table_manager else "secondary"):
    st.session_state.show_table_manager = True

st.sidebar.markdown("---")
st.sidebar.info("Switch between NLP Dashboard and Table Manager using the buttons above")

# =============================================================================
# TABLE MANAGER FUNCTIONS (Code 1)
# =============================================================================

def get_jde_tables(session):
    """Get list of available tables from JDE on page load"""
    try:
        query = 'SHOW TABLES IN jde.TESTDTA'
        result = session.call('JDE.TESTDTA.DREMIO_DATA_PROCEDURE11', query)
        
        if hasattr(result, 'to_pandas'):
            tables_df = result.to_pandas()
        else:
            tables_df = pd.DataFrame(result)
        
        if not tables_df.empty:
            # Find the table name column
            possible_name_columns = ['name', 'table_name', 'TABLE_NAME', 'NAME', 'Table']
            name_column = None
            
            for col in possible_name_columns:
                if col in tables_df.columns:
                    name_column = col
                    break
            
            if name_column:
                return sorted(tables_df[name_column].tolist()), None
        
        return [], "No tables found meta"
    except Exception as e:
        return [], f"Error loading JDE tables: {str(e)}"

def get_table_metadata_jde(session, table_name):
    """Get metadata for a specific JDE table"""
    try:
        describe_query = f'DESCRIBE jde.{table_name}'
        result = session.call('JDE.TESTDTA.DREMIO_DATA_PROCEDURE11', describe_query)
        
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()
        else:
            df = pd.DataFrame(result)
        
        return df, None
    except Exception as e:
        return None, str(e)

def convert_jde_to_snowflake_type(jde_type):
    """Convert JDE data types to Snowflake data types"""
    type_mapping = {
        'STRING': 'VARCHAR(16777216)',
        'TEXT': 'VARCHAR(16777216)',
        'VARCHAR': 'VARCHAR(16777216)',
        'CHAR': 'VARCHAR(255)',
        'INTEGER': 'NUMBER(38,0)',
        'INT': 'NUMBER(38,0)',
        'BIGINT': 'NUMBER(38,0)',
        'DOUBLE': 'NUMBER(38,10)',
        'FLOAT': 'FLOAT',
        'DECIMAL': 'NUMBER(38,10)',
        'NUMERIC': 'NUMBER(38,10)',
        'NUMBER': 'NUMBER(38,10)',
        'BOOLEAN': 'BOOLEAN',
        'DATE': 'DATE',
        'DATETIME': 'TIMESTAMP_NTZ',
        'TIMESTAMP': 'TIMESTAMP_NTZ',
        'TIME': 'TIME'
    }
    
    jde_type_upper = str(jde_type).upper().strip()
    
    # Handle VARCHAR with size specifications like VARCHAR(100)
    if 'VARCHAR' in jde_type_upper:
        return 'VARCHAR(16777216)'
    
    return type_mapping.get(jde_type_upper, 'VARCHAR(16777216)')

def create_table_sql(table_name, metadata_df):
    """Generate CREATE TABLE SQL from JDE metadata"""
    try:
        # Debug: Show the metadata structure
        st.write(f"Debug - Metadata columns for {table_name}:", list(metadata_df.columns))
        st.write(f"Debug - First few rows:")
        st.dataframe(metadata_df.head())
        
        sql_parts = []
        sql_parts.append(f"CREATE OR REPLACE TABLE {table_name} (")
        
        columns = []
        
        # More comprehensive column name mapping for JDE
        possible_name_columns = [
            'name', 'column_name', 'COLUMN_NAME', 'NAME', 'field_name', 
            'FIELD_NAME', 'col_name', 'COL_NAME', 'Field', 'FIELD'
        ]
        
        possible_type_columns = [
            'type', 'data_type', 'DATA_TYPE', 'TYPE', 'field_type', 
            'FIELD_TYPE', 'col_type', 'COL_TYPE', 'Type', 'DataType'
        ]
        
        # Find the correct column names
        name_column = None
        type_column = None
        
        for col in possible_name_columns:
            if col in metadata_df.columns:
                name_column = col
                break
        
        for col in possible_type_columns:
            if col in metadata_df.columns:
                type_column = col
                break
        
        if not name_column:
            st.error(f"Could not find name column in metadata. Available columns: {list(metadata_df.columns)}")
            return None
        
        if not type_column:
            st.error(f"Could not find type column in metadata. Available columns: {list(metadata_df.columns)}")
            return None
        
        st.write(f"Debug - Using name column: {name_column}, type column: {type_column}")
        
        # Generate columns
        for _, row in metadata_df.iterrows():
            col_name = str(row[name_column]).strip()
            col_type = str(row[type_column]).strip()
            
            if col_name and col_name.lower() != 'nan' and col_type and col_type.lower() != 'nan':
                # Clean column name (remove special characters, spaces)
                clean_col_name = col_name.replace(' ', '_').replace('-', '_')
                # Remove any characters that aren't alphanumeric or underscore
                clean_col_name = ''.join(c for c in clean_col_name if c.isalnum() or c == '_')
                
                snowflake_type = convert_jde_to_snowflake_type(col_type)
                columns.append(f"  {clean_col_name} {snowflake_type}")
        
        if columns:
            sql_parts.append(",\n".join(columns))
            sql_parts.append(");")
            sql_result = "\n".join(sql_parts)
            st.write(f"Debug - Generated SQL for {table_name}:")
            st.code(sql_result, language="sql")
            return sql_result
        else:
            st.error(f"No valid columns found for table {table_name}")
            return None
            
    except Exception as e:
        st.error(f"Error generating SQL for {table_name}: {str(e)}")
        st.write(f"Exception details: {type(e).__name__}")
        return None

def create_tables_in_snowflake(session, table_sqls):
    """Execute CREATE TABLE statements in Snowflake"""
    results = []
    for table_name, sql in table_sqls.items():
        try:
            session.sql(sql).collect()
            results.append({"table": table_name, "status": "✅ Created", "error": None})
        except Exception as e:
            results.append({"table": table_name, "status": "❌ Failed", "error": str(e)})
    
    return results

# New functions for semantic model generation
def generate_ai_description(session, prompt, model='llama3.1-8b'):
    """Generate AI-powered descriptions using Snowflake Cortex"""
    try:
        # Escape single quotes in the prompt
        escaped_prompt = prompt.replace("'", "''")
        
        cortex_query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            '{escaped_prompt}'
        ) AS description
        """
        
        result = session.sql(cortex_query).collect()
        if result:
            return result[0]['DESCRIPTION'].strip()
        return ""
    except Exception as e:
        st.warning(f"AI description generation failed: {str(e)}")
        return ""

def categorize_column_type(column_name, data_type, use_ai_categorization=False, session=None):
    """Categorize column into dimension, time_dimension, or fact"""
    column_name_lower = column_name.lower()
    data_type_upper = data_type.upper()
    
    # AI-powered categorization (more dynamic)
    if use_ai_categorization and session:
        try:
            prompt = f"""
            Categorize this database column for semantic modeling:
            Column: {column_name}
            Data Type: {data_type}
            
            Choose ONE category:
            - dimension: Descriptive attributes (names, categories, IDs, codes, text)
            - time_dimension: Date/time related fields
            - fact: Numeric measures that can be aggregated (amounts, quantities, counts, metrics)
            
            Respond with only: dimension, time_dimension, or fact
            """
            
            ai_response = generate_ai_description(session, prompt)
            if ai_response and any(cat in ai_response.lower() for cat in ['dimension', 'time_dimension', 'fact']):
                if 'time_dimension' in ai_response.lower():
                    return 'time_dimension'
                elif 'fact' in ai_response.lower():
                    return 'fact'
                else:
                    return 'dimension'
        except Exception as e:
            st.warning(f"AI categorization failed for {column_name}: {str(e)}")
    
    # Fallback to rule-based categorization
    # Time dimension patterns
    time_patterns = ['date', 'time', 'timestamp', 'dt', 'tm', 'year', 'month', 'day', 'created', 'updated']
    if any(pattern in column_name_lower for pattern in time_patterns) or 'DATE' in data_type_upper or 'TIMESTAMP' in data_type_upper:
        return 'time_dimension'
    
    # Fact patterns (numeric fields that could be measures)
    fact_patterns = ['amount', 'amt', 'qty', 'quantity', 'price', 'cost', 'rate', 'total', 'sum', 'count', 'value', 'balance', 'units']
    if ('NUMBER' in data_type_upper or 'DECIMAL' in data_type_upper or 'FLOAT' in data_type_upper or 'INTEGER' in data_type_upper) and \
       any(pattern in column_name_lower for pattern in fact_patterns):
        return 'fact'
    
    # Default to dimension
    return 'dimension'

def map_to_semantic_type(data_type):
    """Map data types to semantic model data types"""
    data_type = data_type.upper()
    
    if any(t in data_type for t in ['VARCHAR', 'TEXT', 'STRING', 'CHAR', 'CLOB']):
        return 'VARCHAR(16777216)'
    elif 'NUMBER(38,10)' in data_type or 'DECIMAL' in data_type or 'DOUBLE' in data_type or 'NUMERIC' in data_type:
        return 'NUMBER(38,10)'
    elif 'NUMBER(38,0)' in data_type or 'INTEGER' in data_type or 'BIGINT' in data_type or 'INT' in data_type:
        return 'NUMBER(38,0)'
    elif 'FLOAT' in data_type:
        return 'FLOAT'
    elif 'DATE' in data_type:
        return 'DATE'
    elif 'TIMESTAMP' in data_type or 'DATETIME' in data_type:
        return 'TIMESTAMP_NTZ(9)'
    elif 'BOOLEAN' in data_type:
        return 'BOOLEAN'
    else:
        return 'VARCHAR(16777216)'

def generate_dynamic_table_description(session, table_name, column_names, use_ai_descriptions=True):
    """Generate dynamic table description"""
    if use_ai_descriptions and session:
        prompt = f"Generate a concise business description for a database table named '{table_name}' with columns: {', '.join(column_names[:10])}. Focus on what business entity or process this table represents. Keep it under 100 words and business-friendly."
        
        ai_description = generate_ai_description(session, prompt)
        if ai_description:
            return ai_description
    
    # Fallback dynamic description
    return f"Database table {table_name.upper()} containing operational information with {len(column_names)} fields for business analysis and reporting."

def generate_dynamic_column_description(session, column_name, table_name, data_type, use_ai_descriptions=True):
    """Generate dynamic column description"""
    if use_ai_descriptions and session:
        prompt = f"Generate a brief business description for a database column named '{column_name}' of type '{data_type}' in table '{table_name}'. Keep it concise (under 50 words) and focus on what this data represents in business terms."
        
        ai_description = generate_ai_description(session, prompt)
        if ai_description:
            return ai_description
    
    # Fallback dynamic description
    col_lower = column_name.lower()
    if any(pattern in col_lower for pattern in ['id', 'key', 'code']):
        return f"Identifier field {column_name.upper()} for unique record identification and referential integrity"
    elif any(pattern in col_lower for pattern in ['name', 'desc', 'title']):
        return f"Descriptive field {column_name.upper()} containing textual information for record identification"
    elif any(pattern in col_lower for pattern in ['date', 'time']):
        return f"Temporal field {column_name.upper()} for tracking timing and scheduling information"
    elif any(pattern in col_lower for pattern in ['amount', 'qty', 'count']):
        return f"Numeric field {column_name.upper()} containing quantitative data for analysis and reporting"
    else:
        return f"Data field {column_name.upper()} containing business information for operational analysis"

def generate_semantic_model(session, selected_tables, model_name, model_description, use_ai_descriptions=True, use_ai_categorization=False):
    """Generate complete semantic model following Snowflake specification"""
    
    semantic_model = {
        'name': model_name,
        'description': model_description,
        'tables': []
    }
    
    progress_bar = st.progress(0)
    total_tables = len(selected_tables)
    
    for idx, table_name in enumerate(selected_tables):
        st.write(f"Processing table: {table_name}")
        
        # Get table metadata
        metadata_df, error = get_table_metadata_jde(session, table_name)
        
        if error or metadata_df is None or metadata_df.empty:
            st.warning(f"Could not get metadata for {table_name}")
            continue
        
        # Find column name and type columns
        possible_name_columns = ['name', 'column_name', 'COLUMN_NAME', 'NAME', 'field_name', 'FIELD_NAME']
        possible_type_columns = ['type', 'data_type', 'DATA_TYPE', 'TYPE', 'field_type', 'FIELD_TYPE']
        
        name_column = None
        type_column = None
        
        for col in possible_name_columns:
            if col in metadata_df.columns:
                name_column = col
                break
        
        for col in possible_type_columns:
            if col in metadata_df.columns:
                type_column = col
                break
        
        if not name_column or not type_column:
            st.warning(f"Could not find proper columns for {table_name}")
            continue
        
        # Get column names for context
        column_names = metadata_df[name_column].tolist()
        
        # Generate table description
        table_description = generate_dynamic_table_description(session, table_name, column_names, use_ai_descriptions)
        
        # Build table definition
        table_def = {
            'name': table_name.lower(),
            'description': table_description,
            'base_table': {
                'database': 'JDE',
                'schema': 'TESTDTA',
                'table': table_name
            }
        }
        
        # Process columns by category
        dimensions = []
        time_dimensions = []
        facts = []
        primary_key_columns = []
        
        for _, row in metadata_df.iterrows():
            col_name = str(row[name_column]).strip()
            col_type = str(row[type_column]).strip()
            
            if not col_name or col_name.lower() == 'nan' or not col_type or col_type.lower() == 'nan':
                continue
            
            # Clean column name
            clean_col_name = col_name.replace(' ', '_').replace('-', '_')
            clean_col_name = ''.join(c for c in clean_col_name if c.isalnum() or c == '_')
            
            # Categorize column
            category = categorize_column_type(col_name, col_type, use_ai_categorization, session)
            
            # Generate column description
            col_description = generate_dynamic_column_description(session, col_name, table_name, col_type, use_ai_descriptions)
            
            # Build column definition based on category
            if category == 'time_dimension':
                time_dimensions.append({
                    'name': clean_col_name.lower(),
                    'synonyms': [col_name.replace('_', ' '), clean_col_name.lower()],
                    'description': col_description,
                    'expr': f"{table_name.lower()}.{clean_col_name}",
                    'data_type': map_to_semantic_type(col_type)
                })
            elif category == 'fact':
                facts.append({
                    'name': clean_col_name.lower(),
                    'synonyms': [col_name.replace('_', ' '), clean_col_name.lower()],
                    'description': col_description,
                    'expr': f"{table_name.lower()}.{clean_col_name}",
                    'data_type': map_to_semantic_type(col_type)
                })
            else:  # dimension
                dimensions.append({
                    'name': clean_col_name.lower(),
                    'synonyms': [col_name.replace('_', ' '), clean_col_name.lower()],
                    'description': col_description,
                    'expr': f"{table_name.lower()}.{clean_col_name}",
                    'data_type': map_to_semantic_type(col_type),
                    'unique': col_name.lower().endswith('_id') or 'id' in col_name.lower()
                })
            
            # Check for potential primary key
            if col_name.lower().endswith('_id') or col_name.lower() == 'id' or 'key' in col_name.lower():
                primary_key_columns.append(clean_col_name.lower())
        
        # Add columns to table definition
        if dimensions:
            table_def['dimensions'] = dimensions
        if time_dimensions:
            table_def['time_dimensions'] = time_dimensions
        if facts:
            table_def['facts'] = facts
        if primary_key_columns:
            table_def['primary_key'] = {'columns': primary_key_columns}
        
        # Add synonyms for the table
        table_def['synonyms'] = [table_name.replace('_', ' ').title(), table_name.lower(), table_name.upper()]
        
        semantic_model['tables'].append(table_def)
        
        # Update progress
        progress_bar.progress((idx + 1) / total_tables)
    
    progress_bar.empty()
    return semantic_model

def export_to_yaml(semantic_model):
    """Export semantic model to YAML format"""
    yaml.add_representer(dict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
    
    return yaml.dump(
        semantic_model, 
        default_flow_style=False, 
        indent=2, 
        sort_keys=False,
        allow_unicode=True,
        width=1000
    )

# =============================================================================
# NLP DASHBOARD FUNCTIONS (Code 2)
# =============================================================================

def initialize_nlp_session_state():
    """Initialize session state with default values for NLP Dashboard."""
    defaults = {
        "messages": [],
        "active_suggestion": None,
        "warnings": [],
        "fire_API_error_notify": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_analyst_response(messages: List[Dict]) -> Tuple[Dict, Optional[str]]:
    """
    Get response from Cortex Analyst API with improved error handling.
    """
    semantic_model_file = f"@{YAML_CONFIG}"
    
    try:
        result = session.call(
            "JDE.TESTDTA.CORTEX_ANALYST_API_PROCEDURE",
            messages,
            semantic_model_file
        )
        
        if result is None:
            return {"request_id": "error"}, ERROR_MESSAGES["general_error"]
        
        # Parse response
        response_data = json.loads(result) if isinstance(result, str) else result
        
        # Handle successful response
        if response_data.get("success", False):
            return {
                "message": response_data.get("analyst_response", {}),
                "request_id": response_data.get("request_id", "N/A"),
                "warnings": response_data.get("warnings", [])
            }, None
        
        # Handle error response
        error_details = response_data.get("error_details", {})
        error_code = error_details.get('error_code', '').lower()
        
        # Map specific errors to user-friendly messages
        if 'access' in error_code or 'permission' in error_code:
            error_msg = ERROR_MESSAGES["access_denied"]
        elif 'config' in error_code or 'not found' in error_code:
            error_msg = ERROR_MESSAGES["config_error"]
        else:
            error_msg = ERROR_MESSAGES["general_error"]
        
        return {
            "request_id": response_data.get("request_id", "error"),
            "warnings": response_data.get("warnings", [])
        }, error_msg
        
    except SnowparkSQLException as e:
        error_str = str(e).lower()
        if 'access denied' in error_str or 'insufficient privileges' in error_str:
            return {"request_id": "error"}, ERROR_MESSAGES["access_denied"]
        elif 'does not exist' in error_str:
            return {"request_id": "error"}, ERROR_MESSAGES["config_error"]
        else:
            return {"request_id": "error"}, ERROR_MESSAGES["general_error"]
        
    except Exception:
        return {"request_id": "error"}, ERROR_MESSAGES["general_error"]

@st.cache_data(show_spinner=False, ttl=300)
def execute_data_procedure(query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute data procedure with enhanced error handling and user-friendly messages.
    """
    try:
        # Clean the query
        clean_query = query.replace("'", "''")
        
        # Execute procedure
        result = session.call("JDE.TESTDTA.DREMIO_DATA_PROCEDURE11", clean_query)
        
        if result is None:
            return None, ERROR_MESSAGES["no_data"]
        
        # Convert to pandas DataFrame
        try:
            df = result.to_pandas()
            
            if df.empty:
                return None, ERROR_MESSAGES["no_data"]
                
            # Check for error indicators in the result
            if has_data_errors(df):
                return None, ERROR_MESSAGES["no_data"]
            
            return df, None
            
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['unexpected data format', 'no data', 'empty', 'invalid']):
                return None, ERROR_MESSAGES["no_data"]
            else:
                return None, ERROR_MESSAGES["general_error"]
        
    except SnowparkSQLException as e:
        error_str = str(e).lower()
        
        if any(keyword in error_str for keyword in ['no data', 'no records', 'empty', 'not found', 'zero rows']):
            return None, ERROR_MESSAGES["no_data"]
        elif 'does not exist' in error_str:
            return None, ERROR_MESSAGES["config_error"]
        elif 'access denied' in error_str or 'insufficient privileges' in error_str:
            return None, ERROR_MESSAGES["access_denied"]
        else:
            return None, ERROR_MESSAGES["general_error"]
            
    except Exception as e:
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['unexpected data format', 'no data', 'empty', 'invalid']):
            return None, ERROR_MESSAGES["no_data"]
        else:
            return None, ERROR_MESSAGES["general_error"]

def has_data_errors(df: pd.DataFrame) -> bool:
    """Check if DataFrame contains error indicators."""
    # Check for error columns
    if "ERROR" in df.columns or "error" in df.columns:
        error_col = "ERROR" if "ERROR" in df.columns else "error"
        error_rows = df[df[error_col].notna()]
        if not error_rows.empty:
            # Check for specific error patterns
            for error_val in error_rows[error_col]:
                error_str = str(error_val).lower()
                if any(keyword in error_str for keyword in [
                    'unexpected data format', 
                    'error while executing', 
                    'no data', 
                    'not found', 
                    'empty',
                    'sql error',
                    'execution error'
                ]):
                    return True
    
    # Check for DATA column with error messages
    if "DATA" in df.columns:
        for data_val in df["DATA"]:
            data_str = str(data_val).lower()
            if any(keyword in data_str for keyword in [
                'error while executing sql',
                'sql error',
                'execution failed',
                'query failed'
            ]):
                return True
    
    # Check for message-only results indicating no data
    if len(df.columns) == 1 and "message" in df.columns:
        message = str(df["message"].iloc[0]).lower()
        if any(keyword in message for keyword in ['no data', 'not found', 'empty', 'error']):
            return True
    
    # Check if DataFrame only contains error-related columns
    error_related_cols = ['ERROR', 'error', 'RECEIVED_TYPE', 'DATA']
    if all(col in error_related_cols for col in df.columns):
        return True
    
    return False

def process_user_input(prompt: str):
    """Process user input and generate analyst response."""
    # Clear previous warnings
    st.session_state.warnings = []

    # Add user message (hidden from UI)
    user_message = {
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
        "hidden": True
    }
    st.session_state.messages.append(user_message)
    
    # Prepare messages for API
    api_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]

    # Generate analyst response
    with st.chat_message("analyst"):
        with st.spinner("🤔 Analyzing your question..."):
            response, error_msg = get_analyst_response(api_messages)
            
            # Create analyst message
            if error_msg is None:
                analyst_message = {
                    "role": "analyst",
                    "content": response["message"]["content"],
                    "request_id": response["request_id"],
                }
            else:
                analyst_message = {
                    "role": "analyst",
                    "content": [{"type": "text", "text": error_msg}],
                    "request_id": response.get("request_id", "error"),
                }
                st.session_state.fire_API_error_notify = True

            # Handle warnings
            if "warnings" in response:
                st.session_state.warnings = response["warnings"]

            st.session_state.messages.append(analyst_message)
            st.rerun()

def display_conversation():
    """Display conversation history excluding hidden messages."""
    for idx, message in enumerate(st.session_state.messages):
        if message.get("hidden", False):
            continue
            
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            display_message_content(content, idx)

def display_message_content(content: List[Dict], message_index: int):
    """Display message content with improved handling."""
    for item in content:
        if item["type"] == "text":
            st.markdown(item["text"])
            
        elif item["type"] == "suggestions":
            st.markdown("**💡 Suggested questions:**")
            for suggestion_index, suggestion in enumerate(item["suggestions"]):
                if st.button(
                    suggestion, 
                    key=f"suggestion_{message_index}_{suggestion_index}",
                    type="secondary"
                ):
                    st.session_state.active_suggestion = suggestion
                    
        elif item["type"] == "sql":
            # Execute SQL without displaying the query
            execute_and_display_results(item["statement"], message_index, item.get("confidence"))

def execute_and_display_results(sql: str, message_index: int, confidence: dict):
    """
    Execute SQL query and display results without showing the SQL query.
    """
    # Display confidence info if available
    if confidence:
        display_confidence_info(confidence, message_index)

    # Execute and display results
    with st.spinner(f"⚡ Fetching data from {DATA_SOURCE_NAME}..."):
        df, error_msg = execute_data_procedure(sql)
        
        if df is None or error_msg:
            st.error(error_msg)
            return
        
        if df.empty:
            st.error(ERROR_MESSAGES["no_data"])
            return
        
        # Check for runtime errors in data
        if has_data_errors(df):
            st.error(ERROR_MESSAGES["no_data"])
            return
        
        # Display successful results
        display_data_results(df, message_index)

def display_confidence_info(confidence: dict, message_index: int):
    """Display confidence information in a collapsible section."""
    verified_query = confidence.get("verified_query_used")
    
    if verified_query:
        with st.expander("🔍 Query Information"):
            st.write(f"**Name:** {verified_query.get('name', 'N/A')}")
            st.write(f"**Question:** {verified_query.get('question', 'N/A')}")
            st.write(f"**Verified by:** {verified_query.get('verified_by', 'N/A')}")
            
            if 'verified_at' in verified_query:
                st.write(f"**Verified at:** {datetime.fromtimestamp(verified_query['verified_at'])}")

def display_data_results(df: pd.DataFrame, message_index: int):
    """Display data results in organized tabs."""
    # Create tabs for different views
    data_tab, chart_tab = st.tabs(["📄 Data View", "📈 Chart View"])
    
    with data_tab:
        st.dataframe(df, use_container_width=True)
        st.caption(f"📊 Showing {len(df):,} rows × {len(df.columns)} columns")

    with chart_tab:
        display_chart_options(df, message_index)

def display_chart_options(df: pd.DataFrame, message_index: int):
    """Display chart creation options with improved UX."""
    # Filter suitable columns for charting
    chart_columns = [col for col in df.columns if col not in ['error', 'message']]
    
    if len(chart_columns) < 2:
        st.info("📊 Need at least 2 data columns to create charts")
        return
    
    # Get column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Chart configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_col = st.selectbox(
            "X-axis", 
            chart_columns, 
            key=f"x_axis_{message_index}"
        )
    
    with col2:
        y_options = [col for col in chart_columns if col != x_col]
        y_col = st.selectbox(
            "Y-axis", 
            y_options,
            key=f"y_axis_{message_index}"
        )
    
    with col3:
        chart_types = ["📊 Bar Chart", "📈 Line Chart", "🔢 Area Chart", "🎯 Scatter Plot"]
        chart_type = st.selectbox(
            "Chart Type",
            chart_types,
            key=f"chart_type_{message_index}"
        )
    
    # Generate chart
    try:
        create_chart(df, x_col, y_col, chart_type, numeric_cols)
    except Exception as e:
        st.error("❌ Unable to create chart with selected options")
        st.info("💡 Try different column combinations or chart types")

def create_chart(df: pd.DataFrame, x_col: str, y_col: str, chart_type: str, numeric_cols: list):
    """Create and display chart based on selections."""
    if y_col in numeric_cols:
        # Numeric data charting
        chart_data = df.set_index(x_col)[y_col] if x_col != y_col else df[y_col]
        
        if chart_type == "📊 Bar Chart":
            st.bar_chart(chart_data)
        elif chart_type == "📈 Line Chart":
            st.line_chart(chart_data)
        elif chart_type == "🔢 Area Chart":
            st.area_chart(chart_data)
        elif chart_type == "🎯 Scatter Plot":
            st.scatter_chart(df, x=x_col, y=y_col)
    else:
        # Categorical data - show frequency/count
        if chart_type in ["📊 Bar Chart", "📈 Line Chart"]:
            grouped = df.groupby([x_col, y_col]).size().reset_index(name='count')
            chart_data = grouped.set_index(x_col)['count']
            
            if chart_type == "📊 Bar Chart":
                st.bar_chart(chart_data)
            else:
                st.line_chart(chart_data)
        else:
            st.info("📊 For categorical data, use Bar or Line charts")

def handle_user_inputs():
    """Handle user inputs from the chat interface."""
    user_input = st.chat_input("What is your question?")
    
    if user_input:
        process_user_input(user_input)
    elif st.session_state.active_suggestion:
        suggestion = st.session_state.active_suggestion
        st.session_state.active_suggestion = None
        process_user_input(suggestion)

def handle_error_notifications():
    """Handle error toast notifications."""
    if st.session_state.get("fire_API_error_notify"):
        st.toast("An API error has occurred!", icon="🚨")
        st.session_state.fire_API_error_notify = False

# =============================================================================
# TABLE MANAGER SESSION STATE INITIALIZATION
# =============================================================================

# Initialize session state for Table Manager
if 'jde_tables_loaded' not in st.session_state:
    st.session_state.jde_tables_loaded = False
if 'jde_tables' not in st.session_state:
    st.session_state.jde_tables = []
if 'jde_tables_error' not in st.session_state:
    st.session_state.jde_tables_error = None
if 'generated_yaml' not in st.session_state:
    st.session_state.generated_yaml = None
if 'semantic_model' not in st.session_state:
    st.session_state.semantic_model = None
if 'tables_created_successfully' not in st.session_state:
    st.session_state.tables_created_successfully = []

# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================

try:
    # Initialize Snowflake session
    session = get_active_session()
    st.success("✅ Connected to Snowflake successfully!")
    
    # Display content based on selected mode
    if st.session_state.show_table_manager:
        # =============================================================================
        # TABLE MANAGER INTERFACE
        # =============================================================================
        
        # Page Title for Table Manager
        st.title("📊 JDE Tables Manager")
        st.markdown("Select JDE tables to view metadata, create in Snowflake, and generate semantic models")
        st.markdown("---")
        
        # Load JDE tables on page initialization
        if not st.session_state.jde_tables_loaded:
            with st.spinner("🔄 Loading JDE tables metadata..."):
                tables_list, error = get_jde_tables(session)
                st.session_state.jde_tables = tables_list
                st.session_state.jde_tables_error = error
                st.session_state.jde_tables_loaded = True
                
                if st.session_state.jde_tables:
                    st.success(f"✅ Loaded {len(st.session_state.jde_tables)} JDE tables")
                else:
                    st.warning(f"⚠️ {error or 'No JDE tables found'}")
        
        # Show number of available tables
        if st.session_state.jde_tables:
            st.info(f"📋 Found {len(st.session_state.jde_tables)} available JDE tables")
            
            # Searchable multiselect for tables
            st.subheader("🔍 Select Tables")
            selected_tables = st.multiselect(
                "Choose tables to describe and create:",
                options=st.session_state.jde_tables,
                default=[],
                help="Start typing to search for tables"
            )
            
            if selected_tables:
                st.success(f"Selected {len(selected_tables)} table(s): {', '.join(selected_tables)}")
                
                # Show metadata for selected tables (optional preview)
                if st.checkbox("🔍 Preview metadata before creating tables"):
                    for table in selected_tables:
                        with st.expander(f"Metadata for {table}"):
                            with st.spinner(f"Loading metadata for {table}..."):
                                metadata_df, error = get_table_metadata_jde(session, table)
                                if error:
                                    st.error(f"Error: {error}")
                                elif metadata_df is not None and not metadata_df.empty:
                                    st.dataframe(metadata_df, use_container_width=True)
                                else:
                                    st.warning("No metadata found")
                
                # Create Tables button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    if st.button("🚀 Create Selected Tables in Snowflake", type="primary", use_container_width=True):
                        table_sqls = {}
                        
                        # Get metadata and generate SQL for each table
                        for table in selected_tables:
                            with st.spinner(f"Processing {table}..."):
                                st.write(f"🔄 Getting metadata for {table}")
                                metadata_df, error = get_table_metadata_jde(session, table)
                                
                                if error:
                                    st.error(f"❌ Error loading metadata for {table}: {error}")
                                    continue
                                elif metadata_df is not None and not metadata_df.empty:
                                    st.write(f"✅ Metadata loaded for {table} ({len(metadata_df)} columns)")
                                    sql = create_table_sql(table, metadata_df)
                                    if sql:
                                        table_sqls[table] = sql
                                        st.success(f"✅ SQL generated for {table}")
                                    else:
                                        st.error(f"❌ Could not generate SQL for {table}")
                                else:
                                    st.warning(f"⚠️ No metadata found for {table}")
                        
                        # Create tables if we have SQL
                        if table_sqls:
                            st.write("🚀 Creating tables in Snowflake...")
                            results = create_tables_in_snowflake(session, table_sqls)
                            
                            # Show results
                            st.subheader("📊 Creation Results")
                            
                            successful_tables = []
                            for result in results:
                                if result['error']:
                                    st.error(f"{result['status']} {result['table']}: {result['error']}")
                                else:
                                    st.success(f"{result['status']} {result['table']}")
                                    successful_tables.append(result['table'])
                            
                            # Store successfully created tables
                            st.session_state.tables_created_successfully = successful_tables
                            
                            # Summary
                            success_count = len(successful_tables)
                            st.info(f"📈 Summary: {success_count}/{len(results)} tables created successfully")
                            
                        else:
                            st.error("❌ Could not generate SQL for any selected tables")

                # Generate Semantic Model button (only show if tables were created successfully)
                if st.session_state.tables_created_successfully:
                    st.markdown("---")
                    st.header("📝 Generate Semantic Model")
                    
                    # Semantic Model Configuration
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        model_name = st.text_input(
                            "Model Name *", 
                            value="jde_semantic_model",
                            placeholder="e.g., my_analytics_model",
                            help="Must be unique and follow Snowflake identifier requirements"
                        )
                        
                        use_ai_descriptions = st.checkbox(
                            "🤖 Use AI-Generated Descriptions", 
                            value=True,
                            help="Use Snowflake Cortex to generate intelligent descriptions"
                        )
                        
                        use_ai_categorization = st.checkbox(
                            "🧠 Use AI Column Categorization", 
                            value=False,
                            help="Use AI to intelligently categorize columns (slower but more accurate)"
                        )
                    
                    with col2:
                        model_description = st.text_area(
                            "Model Description *", 
                            value="Comprehensive semantic model for database analytics enabling business intelligence and reporting.",
                            placeholder="Describe the purpose and scope of this semantic model...",
                            help="Provide context about what kind of analysis this model supports"
                        )
                    
                    # Generate semantic model button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        can_generate = all([model_name, model_description, st.session_state.tables_created_successfully])
                        
                        if st.button("🎯 Generate Semantic Model", type="primary", use_container_width=True, disabled=not can_generate):
                            if can_generate:
                                st.subheader("📝 Generating Semantic Model")
                                with st.spinner("Generating semantic model... This may take a few minutes."):
                                    try:
                                        semantic_model = generate_semantic_model(
                                            session,
                                            st.session_state.tables_created_successfully,
                                            model_name,
                                            model_description,
                                            use_ai_descriptions,
                                            use_ai_categorization
                                        )
                                        
                                        yaml_content = export_to_yaml(semantic_model)
                                        
                                        st.session_state.semantic_model = semantic_model
                                        st.session_state.generated_yaml = yaml_content
                                        
                                        st.success("✅ Semantic model generated successfully!")
                                        st.rerun()
                                        
                                    except Exception as e:
                                        st.error(f"❌ Error generating semantic model: {str(e)}")
                            else:
                                st.warning("⚠️ Please fill in all required fields.")

                    # Display results if YAML was generated
                    if st.session_state.generated_yaml:
                        st.markdown("---")
                        st.header("📄 Generated Semantic Model")
                        
                        # Model summary
                        if st.session_state.semantic_model:
                            model = st.session_state.semantic_model
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Tables", len(model.get('tables', [])))
                            with col2:
                                total_columns = sum(
                                    len(table.get('dimensions', [])) + 
                                    len(table.get('time_dimensions', [])) + 
                                    len(table.get('facts', []))
                                    for table in model.get('tables', [])
                                )
                                st.metric("Total Columns", total_columns)
                            with col3:
                                ai_used = "Yes" if use_ai_descriptions else "No"
                                st.metric("AI Descriptions", ai_used)
                            with col4:
                                ai_cat_used = "Yes" if use_ai_categorization else "No"
                                st.metric("AI Categorization", ai_cat_used)
                        
                        # Model details
                        with st.expander("📊 Model Details", expanded=False):
                            if st.session_state.semantic_model:
                                for table in st.session_state.semantic_model.get('tables', []):
                                    st.subheader(f"Table: {table['name'].title()}")
                                    st.write(f"**Description:** {table['description']}")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        if table.get('dimensions'):
                                            st.write(f"**Dimensions ({len(table['dimensions'])}):**")
                                            for dim in table['dimensions']:
                                                st.write(f"- {dim['name']}: {dim['description']}")
                                    
                                    with col2:
                                        if table.get('time_dimensions'):
                                            st.write(f"**Time Dimensions ({len(table['time_dimensions'])}):**")
                                            for td in table['time_dimensions']:
                                                st.write(f"- {td['name']}: {td['description']}")
                                    
                                    with col3:
                                        if table.get('facts'):
                                            st.write(f"**Facts ({len(table['facts'])}):**")
                                            for fact in table['facts']:
                                                st.write(f"- {fact['name']}: {fact['description']}")
                                    
                                    st.markdown("---")
                        
                        # YAML content
                        st.subheader("📝 YAML Content")
                        st.code(st.session_state.generated_yaml, language='yaml')
                        
                        # Download and Clear buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📥 Download YAML File",
                                data=st.session_state.generated_yaml.encode('utf-8'),
                                file_name=f"{model_name}.yaml",
                                mime="text/yaml",
                                type="primary",
                                use_container_width=True
                            )
                        
                        with col2:
                            if st.button("🗑️ Clear Generated Model", type="secondary", use_container_width=True):
                                st.session_state.generated_yaml = None
                                st.session_state.semantic_model = None
                                st.rerun()
            
            else:
                st.info("👆 Please select one or more tables to view their metadata and create them in Snowflake")
        
        else:
            if st.session_state.jde_tables_error:
                st.error(f"❌ {st.session_state.jde_tables_error}")
            else:
                st.warning("⚠️ No JDE tables available")
            
            # Option to manually enter table name
            st.subheader("Manual Table Entry")
            manual_table = st.text_input("Enter JDE table name manually:", placeholder="e.g., F0012")
            
            if manual_table:
                if st.button("🔍 Get Metadata and Create Table"):
                    with st.spinner(f"Processing {manual_table}..."):
                        metadata_df, error = get_table_metadata_jde(session, manual_table)
                        
                        if error:
                            st.error(f"❌ Error loading metadata for {manual_table}: {error}")
                        elif metadata_df is not None and not metadata_df.empty:
                            st.success(f"✅ Metadata loaded for {manual_table} ({len(metadata_df)} columns)")
                            
                            # Show metadata
                            with st.expander(f"Metadata for {manual_table}", expanded=True):
                                st.dataframe(metadata_df, use_container_width=True)
                            
                            # Generate and execute SQL
                            sql = create_table_sql(manual_table, metadata_df)
                            if sql:
                                try:
                                    session.sql(sql).collect()
                                    st.success(f"✅ Table {manual_table} created successfully in Snowflake")
                                    
                                    # Store this table as successfully created for semantic model generation
                                    st.session_state.tables_created_successfully = [manual_table]
                                    
                                    # Show semantic model generation option
                                    st.markdown("---")
                                    st.subheader("📝 Generate Semantic Model for Created Table")
                                    
                                    with st.expander("Semantic Model Configuration", expanded=True):
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            model_name_manual = st.text_input(
                                                "Model Name", 
                                                value=f"{manual_table.lower()}_semantic_model",
                                                key="manual_model_name"
                                            )
                                            
                                            use_ai_desc_manual = st.checkbox(
                                                "Use AI Descriptions", 
                                                value=True,
                                                key="manual_ai_desc"
                                            )
                                            
                                            use_ai_cat_manual = st.checkbox(
                                                "Use AI Categorization", 
                                                value=False,
                                                key="manual_ai_cat"
                                            )
                                        
                                        with col2:
                                            model_desc_manual = st.text_area(
                                                "Model Description", 
                                                value=f"Semantic model for {manual_table} table enabling business intelligence and analytics.",
                                                key="manual_model_desc"
                                            )
                                        
                                        if st.button("Generate Semantic Model for Manual Table", key="manual_generate_button"):
                                            with st.spinner("Generating semantic model..."):
                                                try:
                                                    semantic_model = generate_semantic_model(
                                                        session,
                                                        [manual_table],
                                                        model_name_manual,
                                                        model_desc_manual,
                                                        use_ai_desc_manual,
                                                        use_ai_cat_manual
                                                    )
                                                    
                                                    yaml_content = export_to_yaml(semantic_model)
                                                    
                                                    st.success("✅ Semantic model generated!")
                                                    st.code(yaml_content, language='yaml')
                                                    
                                                    # Download button for manual table
                                                    st.download_button(
                                                        label="📥 Download YAML File",
                                                        data=yaml_content.encode('utf-8'),
                                                        file_name=f"{model_name_manual}.yaml",
                                                        mime="text/yaml",
                                                        use_container_width=True
                                                    )
                                                    
                                                except Exception as e:
                                                    st.error(f"❌ Error generating semantic model: {str(e)}")
                                    
                                except Exception as e:
                                    st.error(f"❌ Failed to create table {manual_table}: {str(e)}")
                            else:
                                st.error(f"❌ Could not generate SQL for {manual_table}")
                        else:
                            st.warning(f"⚠️ No metadata found for {manual_table}")

    else:
        # =============================================================================
        # NLP DASHBOARD INTERFACE
        # =============================================================================
        
        # Initialize NLP session state
        initialize_nlp_session_state()
        
        # Page Title for NLP Dashboard
        st.title("🧠 NLP to Dashboard")
        st.markdown("Welcome to Analyst! Ask questions about your JDE data.")
        st.divider()
        
        # Show initial question only once
        if len(st.session_state.messages) == 0 and not st.session_state.get("initial_question_asked", False):
            st.session_state.initial_question_asked = True
            process_user_input("What questions can I ask?")
        
        display_conversation()
        handle_user_inputs()
        handle_error_notifications()

except Exception as e:
    st.error(f"❌ Failed to connect to Snowflake: {str(e)}")
    st.info("Please ensure you're running this in a Snowflake environment with proper session context.")

# Footer
st.markdown("---")
if st.session_state.show_table_manager:
    st.markdown("*📊 JDE Tables Manager with Semantic Model Generation - Enhanced for comprehensive analytics*")
else:
    st.markdown("*🧠 NLP to Dashboard - Ask questions in natural language to analyze your JDE data*")
