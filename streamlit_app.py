import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import uuid

import pandas as pd
import streamlit as st
# from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSQLException

# Fixed configuration for JDE.TESTDTA only
YAML_CONFIG = "JDE.TESTDTA.jde_stage/jde.yaml"
DATA_SOURCE_NAME = "JDE.TESTDTA"
cnx = st.connection("snowflake")
session = cnx.session()
# Initialize session once and reuse
@st.cache_resource
# def get_snowpark_session():
#     """Get and cache the Snowpark session to avoid recreation"""
#     return get_active_session()

# session = get_snowpark_session()


def main():
    # Initialize session state
    if "messages" not in st.session_state:
        reset_session_state()
    
    show_header()
    
    # Show initial question only once
    if len(st.session_state.messages) == 0 and "initial_question_asked" not in st.session_state:
        st.session_state.initial_question_asked = True
        process_user_input("What questions can I ask?")
    
    display_conversation()
    handle_user_inputs()
    handle_error_notifications()


def reset_session_state():
    """Reset important session state elements."""
    st.session_state.messages = []
    st.session_state.active_suggestion = None
    st.session_state.warnings = []
    if "initial_question_asked" in st.session_state:
        del st.session_state.initial_question_asked


def show_header():
    """Display the header of the app."""
    st.title("🧠 NLP to Dashboard")
    
    st.markdown("Welcome to  Analyst! Ask questions about your JDE  data.")
    # st.info(f"📊 **{DATA_SOURCE_NAME}** data source")
    st.divider()


def handle_user_inputs():
    """Handle user inputs from the chat interface."""
    user_input = st.chat_input("What is your question?")
    if user_input:
        process_user_input(user_input)
    elif st.session_state.active_suggestion is not None:
        suggestion = st.session_state.active_suggestion
        st.session_state.active_suggestion = None
        process_user_input(suggestion)


def handle_error_notifications():
    """Handle error notifications."""
    if st.session_state.get("fire_API_error_notify"):
        st.toast("An API error has occurred!", icon="🚨")
        st.session_state["fire_API_error_notify"] = False


def process_user_input(prompt: str):
    """Process user input and update the conversation history."""
    # Clear previous warnings
    st.session_state.warnings = []

    # Create user message (hidden from UI)
    new_user_message = {
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
        "hidden": True
    }
    st.session_state.messages.append(new_user_message)
    
    # Prepare messages for API
    messages_for_api = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]

    # Show analyst response with progress
    with st.chat_message("analyst"):
        with st.spinner("🤔 Analyzing your question..."):
            response, error_msg = get_analyst_response(messages_for_api)
            
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
                st.session_state["fire_API_error_notify"] = True

            if "warnings" in response:
                st.session_state.warnings = response["warnings"]

            st.session_state.messages.append(analyst_message)
            st.rerun()


def get_analyst_response(messages: List[Dict]) -> Tuple[Dict, Optional[str]]:
    """
    Send chat history to the Cortex Analyst API via stored procedure.
    OPTIMIZED: Improved error handling and response processing.
    """
    semantic_model_file = f"@{YAML_CONFIG}"
    
    try:
        # Call stored procedure with timeout handling
        result = session.call(
            "JDE.TESTDTA.CORTEX_ANALYST_API_PROCEDURE",
            messages,
            semantic_model_file
        )
        
        if result is None:
            return {"request_id": "error"}, "❌ No response from Cortex Analyst procedure"
        
        # Parse response
        if isinstance(result, str):
            response_data = json.loads(result)
        else:
            response_data = result
        
        # Handle successful response
        if response_data.get("success", False):
            return_data = {
                "message": response_data.get("analyst_response", {}),
                "request_id": response_data.get("request_id", "N/A"),
                "warnings": response_data.get("warnings", [])
            }
            return return_data, None
        
        # Handle error response
        error_details = response_data.get("error_details", {})
        error_msg = f"""
❌ **Cortex Analyst Error**

**Error Code:** `{error_details.get('error_code', 'N/A')}`  
**Request ID:** `{error_details.get('request_id', 'N/A')}`  
**Status:** `{error_details.get('response_code', 'N/A')}`

**Message:** {error_details.get('error_message', 'No error message provided')}

💡 **Troubleshooting:**
- Verify your jde.yaml file exists in the stage
- Check database and schema permissions
- Ensure Cortex Analyst is properly configured
        """
        
        return_data = {
            "request_id": response_data.get("request_id", "error"),
            "warnings": response_data.get("warnings", [])
        }
        return return_data, error_msg
        
    except SnowparkSQLException as e:
        error_msg = f"""
❌ **Database Error**

{str(e)}

💡 **Check:**
- Procedure exists: `JDE.TESTDTA.CORTEX_ANALYST_API_PROCEDURE`
- You have EXECUTE permissions
- YAML file exists in stage
        """
        return {"request_id": "error"}, error_msg
        
    except Exception as e:
        error_msg = f"❌ **Unexpected Error:** {str(e)}"
        return {"request_id": "error"}, error_msg


def display_conversation():
    """Display the conversation history (excluding hidden messages)."""
    for idx, message in enumerate(st.session_state.messages):
        if message.get("hidden", False):
            continue
            
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            display_message(content, idx)


def display_message(content: List[Dict[str, Union[str, Dict]]], message_index: int):
    """Display a single message content."""
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
            display_sql_query(
                item["statement"], message_index, item.get("confidence")
            )


@st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes
@st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes
def execute_data_procedure(query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute data procedure for JDE.TESTDTA with enhanced error handling.
    FIXED: Better DataFrame handling and error reporting.
    """
    try:
        # Clean the query to prevent SQL injection
        clean_query = query.replace("'", "''")  # Escape single quotes
        
        # Use JDE.TESTDTA specific procedure call
        result = session.call("JDE.TESTDTA.DREMIO_DATA_PROCEDURE11", clean_query)
        
        # Convert to pandas DataFrame
        if result is None:
            return None, "📞 **Please contact your administrator - no data available for this query.**"
        
        # Handle Snowpark DataFrame result
        try:
            df = result.to_pandas()
            
            # Check if DataFrame is empty
            if df.empty:
                return None, "📞 **Please contact your administrator - we do not have this data available.**"
                
            # Check for error columns in the result
            if "error" in df.columns:
                error_rows = df[df["error"].notna()]
                if not error_rows.empty:
                    error_msg = error_rows["error"].iloc[0]
                    # Check if it's a "no data" type error
                    if any(keyword in error_msg.lower() for keyword in ['no data', 'no records', 'empty', 'not found']):
                        return None, "📞 **Please contact your administrator - we do not have this data available.**"
                    else:
                        return None, f"📞 **Please contact your administrator - data source error: {error_msg}**"
            
            return df, None
            
        except Exception as conversion_error:
            error_str = str(conversion_error).lower()
            
            # Handle specific "unexpected data format" errors
            if any(keyword in error_str for keyword in ['unexpected data format', 'no data', 'empty result', 'invalid format']):
                return None, "📞 **Please contact your administrator - we do not have this data available.**"
            else:
                return None, f"📞 **Please contact your administrator - data processing error occurred.**"
        
    except SnowparkSQLException as e:
        error_str = str(e).lower()
        
        # Handle "no data" related SQL errors
        if any(keyword in error_str for keyword in ['no data', 'no records', 'empty', 'not found', 'zero rows']):
            return None, "📞 **Please contact your administrator - we do not have this data available.**"
        elif "does not exist" in error_str:
            return None, "📞 **Please contact your administrator - data source configuration issue.**"
        elif "access denied" in error_str or "insufficient privileges" in error_str:
            return None, "📞 **Please contact your administrator - access permission required.**"
        else:
            return None, "📞 **Please contact your administrator - database error occurred.**"
            
    except Exception as e:
        error_str = str(e).lower()
        
        # Handle general "no data" scenarios
        if any(keyword in error_str for keyword in ['unexpected data format', 'no data', 'empty', 'invalid']):
            return None, "📞 **Please contact your administrator - we do not have this data available.**"
        else:
            return None, "📞 **Please contact your administrator - an unexpected error occurred.**"


def display_sql_query(sql: str, message_index: int, confidence: dict):
    """
    Display SQL query and execute it via JDE.TESTDTA data procedure.
    FIXED: Enhanced error handling for better user experience.
    """
    # Display SQL query in a simple container (not nested expander)
    st.markdown("**📝 SQL Query**")
    st.code(sql, language="sql")
    
    # Display confidence info separately with unique key
    display_sql_confidence(confidence, message_index)

    # Execute and display results
    st.markdown("**📊 Results**")
    with st.spinner(f"⚡ Executing via {DATA_SOURCE_NAME}..."):
        df, err_msg = execute_data_procedure(sql)
        
        if df is None:
            # Display user-friendly error message
            st.error(err_msg)
            return
        
        if df.empty:
            st.error("📞 **Please contact your administrator - we do not have this data available.**")
            return
        
        # Check for error messages in the DataFrame
        if "error" in df.columns:
            error_rows = df[df["error"].notna()]
            if not error_rows.empty:
                error_msg = error_rows["error"].iloc[0]
                # Check if it's a "no data" type error
                if any(keyword in error_msg.lower() for keyword in ['no data', 'no records', 'empty', 'not found']):
                    st.error("📞 **Please contact your administrator - we do not have this data available.**")
                else:
                    st.error("📞 **Please contact your administrator - data source error occurred.**")
                return
        
        # Check for message-only results
        if len(df.columns) == 1 and "message" in df.columns:
            message = df["message"].iloc[0]
            if "No data returned" in str(message):
                st.error("📞 **Please contact your administrator - we do not have this data available.**")
                return
        
        # Display results in tabs
        data_tab, chart_tab = st.tabs(["📄 Data", "📈 Chart"])
        
        with data_tab:
            st.dataframe(df, use_container_width=True)
            st.caption(f"📊 {len(df)} rows × {len(df.columns)} columns")

        with chart_tab:
            display_charts_tab(df, message_index)
def display_sql_confidence(confidence: dict, message_index: int):
    """Display SQL confidence information with unique keys."""
    if confidence is None:
        return
        
    verified_query_used = confidence.get("verified_query_used")
    
    # Create unique key using message index and timestamp
    unique_key = f"confidence_{message_index}_{int(time.time() * 1000)}"
    
    # Use a simple button with unique key
    if st.button("🔍 Show Query Details", key=unique_key):
        if verified_query_used is None:
            st.info("No verified query used for this response")
        else:
            st.write(f"**Name:** {verified_query_used.get('name', 'N/A')}")
            st.write(f"**Question:** {verified_query_used.get('question', 'N/A')}")
            st.write(f"**Verified by:** {verified_query_used.get('verified_by', 'N/A')}")
            
            if 'verified_at' in verified_query_used:
                st.write(f"**Verified at:** {datetime.fromtimestamp(verified_query_used['verified_at'])}")
            
            st.markdown("**SQL Query:**")
            st.code(verified_query_used.get("sql", "N/A"), language="sql")




def display_charts_tab(df: pd.DataFrame, message_index: int) -> None:
    """
    Display charts tab with improved performance and error handling.
    FIXED: Better handling of different data types and empty data.
    """
    if len(df.columns) < 2:
        st.info("📊 At least 2 columns required for charts")
        return
    
    # Filter out error/message columns for charting
    chart_columns = [col for col in df.columns if col not in ['error', 'message']]
    
    if len(chart_columns) < 2:
        st.info("📊 No suitable columns found for charting")
        return
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_col = st.selectbox(
            "X-axis", chart_columns, 
            key=f"x_col_select_{message_index}"
        )
    
    with col2:
        available_y_cols = [col for col in chart_columns if col != x_col]
        y_col = st.selectbox(
            "Y-axis", available_y_cols,
            key=f"y_col_select_{message_index}"
        )
    
    with col3:
        chart_type = st.selectbox(
            "Chart type",
            ["📈 Line", "📊 Bar", "🔢 Area", "🎯 Scatter"],
            key=f"chart_type_{message_index}"
        )
    
    # Create chart based on selection
    try:
        # Handle different data types
        if y_col in numeric_cols:
            # Numeric Y-axis
            if chart_type == "📈 Line":
                chart_data = df.set_index(x_col)[y_col]
                st.line_chart(chart_data)
            elif chart_type == "📊 Bar":
                chart_data = df.set_index(x_col)[y_col]
                st.bar_chart(chart_data)
            elif chart_type == "🔢 Area":
                chart_data = df.set_index(x_col)[y_col]
                st.area_chart(chart_data)
            elif chart_type == "🎯 Scatter":
                st.scatter_chart(df, x=x_col, y=y_col)
        else:
            # Non-numeric Y-axis - show count/frequency
            if chart_type in ["📊 Bar", "📈 Line"]:
                grouped = df.groupby([x_col, y_col]).size().reset_index(name='count')
                chart_data = grouped.set_index(x_col)['count']
                if chart_type == "📊 Bar":
                    st.bar_chart(chart_data)
                else:
                    st.line_chart(chart_data)
            else:
                st.info("📊 Selected chart type not suitable for categorical Y-axis. Try Bar or Line chart.")
                
    except Exception as e:
        st.error(f"❌ Chart error: {str(e)}")
        st.info("💡 Try selecting different columns or chart type")


if __name__ == "__main__":
    main()
