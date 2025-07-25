import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import uuid

import pandas as pd
import streamlit as st
from snowflake.snowpark.exceptions import SnowparkSQLException

# Fixed configuration for JDE.TESTDTA only
YAML_CONFIG = "JDE.TESTDTA.jde_stage/jde.yaml"
DATA_SOURCE_NAME = "JDE.TESTDTA"
cnx = st.connection("snowflake")
session = cnx.session()

# User-friendly error messages
ERROR_MESSAGES = {
    "no_data": "📞 **Data not available** - Please contact your administrator for assistance.",
    "access_denied": "📞 **Access restricted** - Please contact your administrator for permissions.",
    "config_error": "📞 **System configuration issue** - Please contact your administrator.",
    "general_error": "📞 **Technical issue encountered** - Please contact your administrator for support."
}


def main():
    """Main application entry point."""
    initialize_session_state()
    show_header()
    
    # Show initial question only once
    if len(st.session_state.messages) == 0 and not st.session_state.get("initial_question_asked", False):
        st.session_state.initial_question_asked = True
        process_user_input("What questions can I ask?")
    
    display_conversation()
    handle_user_inputs()
    handle_error_notifications()


def initialize_session_state():
    """Initialize session state with default values."""
    defaults = {
        "messages": [],
        "active_suggestion": None,
        "warnings": [],
        "fire_API_error_notify": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_header():
    """Display the application header."""
    st.title("🧠 NLP to Dashboard")
    st.markdown("Welcome to Analyst! Ask questions about your JDE data.")
    st.divider()


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


def execute_and_display_results(sql: str, message_index: int, confidence: dict):
    """
    Execute SQL query and display results without showing the SQL query.
    """
    # Display confidence info if available
    if confidence:
        display_confidence_info(confidence, message_index)

    # Execute and display results
    st.markdown("**📊 Query Results**")
    
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


if __name__ == "__main__":
    main()
