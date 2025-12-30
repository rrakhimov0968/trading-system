"""Streamlit monitoring dashboard for AI Trading System."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import get_config
from core.orchestrator import TradingSystemOrchestrator
from core.async_orchestrator import AsyncTradingSystemOrchestrator
import asyncio
import threading
import time

# Page config
st.set_page_config(
    page_title="AI Trading System Monitor",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "running" not in st.session_state:
    st.session_state.running = False

# Auto-initialize orchestrator for metrics (read-only, doesn't start trading loop)
if st.session_state.orchestrator is None:
    try:
        config = get_config()
        use_async = os.getenv("USE_ASYNC_ORCHESTRATOR", "true").lower() == "true"
        
        if use_async:
            # Create async orchestrator instance (for metrics only)
            st.session_state.orchestrator = AsyncTradingSystemOrchestrator(config=config)
        else:
            # Create sync orchestrator instance (for metrics only)
            st.session_state.orchestrator = TradingSystemOrchestrator(config=config)
        
        # Note: We don't call start() - this is just for reading metrics
        # The actual trading system runs separately via main.py
    except Exception as e:
        st.warning(f"Could not initialize orchestrator: {str(e)}. Metrics will not be available.")

# Title
st.title("📈 AI Trading System Monitor")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("System Controls")
    
    # Initialize orchestrator if not already initialized
    if st.session_state.orchestrator is None:
        if st.button("🚀 Connect to Trading System", use_container_width=True, type="primary"):
            try:
                config = get_config()
                use_async = os.getenv("USE_ASYNC_ORCHESTRATOR", "true").lower() == "true"
                
                if use_async:
                    st.session_state.orchestrator = AsyncTradingSystemOrchestrator(config=config)
                    st.success("✅ Connected to Async Orchestrator")
                else:
                    st.session_state.orchestrator = TradingSystemOrchestrator(config=config)
                    st.success("✅ Connected to Sync Orchestrator")
                
                st.rerun()
            except Exception as e:
                st.error(f"Failed to connect: {str(e)}")
    
    if st.session_state.orchestrator and st.button("🔄 Refresh Metrics", use_container_width=True):
        st.rerun()
    
    if st.session_state.orchestrator and st.button("🔌 Disconnect", use_container_width=True):
        try:
            if hasattr(st.session_state.orchestrator, 'stop'):
                asyncio.run(st.session_state.orchestrator.stop())
            st.session_state.orchestrator = None
            st.session_state.metrics = {}
            st.rerun()
        except Exception as e:
            st.warning(f"Error disconnecting: {str(e)}")
    
    st.markdown("---")
    st.header("Circuit Breaker Status")
    
    if st.session_state.orchestrator:
        circuit_metrics = st.session_state.orchestrator.circuit_breaker.get_metrics()
        
        # Circuit state indicator
        state = circuit_metrics.get("state", "unknown")
        if state == "closed":
            st.success("✅ Circuit CLOSED - Normal Operation")
        elif state == "open":
            st.error("🔴 Circuit OPEN - Trading Stopped")
        elif state == "half_open":
            st.warning("🟡 Circuit HALF-OPEN - Testing Recovery")
        
        # Circuit breaker details
        with st.expander("Circuit Breaker Details"):
            st.metric("Consecutive Failures", circuit_metrics.get("consecutive_failures", 0))
            st.metric("LLM Failures", circuit_metrics.get("llm_failures", 0))
            st.metric("Data Quality Failures", circuit_metrics.get("data_quality_failures", 0))
            
            if circuit_metrics.get("equity_drop_detected"):
                st.error("⚠️ Equity Drop Detected")
            
            if circuit_metrics.get("opened_at"):
                st.info(f"Opened at: {circuit_metrics['opened_at']}")
    
    st.markdown("---")
    st.header("Configuration")
    st.caption("System uses async/event-driven and persistence by default")

# Main metrics row
col1, col2, col3, col4 = st.columns(4)

if st.session_state.orchestrator:
    metrics = st.session_state.orchestrator.get_monitoring_metrics()
    
    with col1:
        st.metric(
            "Iterations",
            metrics.get("iteration_count", 0),
            delta=metrics.get("total_iterations", 0)
        )
    
    with col2:
        llm_success_rate = metrics.get("llm_success_rate", 0.0)
        st.metric(
            "LLM Success Rate",
            f"{llm_success_rate:.1f}%",
            delta=f"{metrics.get('llm_success_count', 0)}/{metrics.get('llm_success_count', 0) + metrics.get('llm_failure_count', 0)}"
        )
    
    with col3:
        total = metrics.get("total_iterations", 0)
        successful = metrics.get("successful_iterations", 0)
        success_rate = (successful / total * 100) if total > 0 else 0
        st.metric(
            "Iteration Success Rate",
            f"{success_rate:.1f}%",
            delta=f"{successful}/{total}"
        )
    
    with col4:
        circuit_state = metrics.get("circuit_breaker", {}).get("state", "unknown")
        st.metric(
            "Circuit State",
            circuit_state.upper(),
            delta="OPEN" if circuit_state == "open" else None
        )
else:
    with col1:
        st.metric("Iterations", 0)
    with col2:
        st.metric("LLM Success Rate", "0%")
    with col3:
        st.metric("Iteration Success Rate", "0%")
    with col4:
        st.metric("Circuit State", "N/A")

st.markdown("---")

# Recent signals
st.header("📊 Recent Trading Signals")
if st.session_state.orchestrator:
    signals = metrics.get("recent_signals", [])
    if signals:
        signals_df = pd.DataFrame(signals)
        
        # Format timestamp
        if "timestamp" in signals_df.columns:
            signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])
            signals_df["time"] = signals_df["timestamp"].dt.strftime("%H:%M:%S")
        
        # Display table
        st.dataframe(
            signals_df[["time", "symbol", "action", "strategy", "confidence"]].tail(20),
            use_container_width=True,
            hide_index=True
        )
        
        # Signal distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            if "action" in signals_df.columns:
                action_counts = signals_df["action"].value_counts()
                fig_actions = px.pie(
                    values=action_counts.values,
                    names=action_counts.index,
                    title="Signal Actions Distribution"
                )
                st.plotly_chart(fig_actions, use_container_width=True)
        
        with col2:
            if "strategy" in signals_df.columns:
                strategy_counts = signals_df["strategy"].value_counts().head(5)
                fig_strategies = px.bar(
                    x=strategy_counts.index,
                    y=strategy_counts.values,
                    title="Top 5 Strategies Used",
                    labels={"x": "Strategy", "y": "Count"}
                )
                st.plotly_chart(fig_strategies, use_container_width=True)
    else:
        st.info("No signals generated yet. System may be starting up or waiting for data.")
else:
    st.info("Orchestrator not initialized. Start the system to see metrics.")

st.markdown("---")

# Equity curve
st.header("💰 Equity Curve")
if st.session_state.orchestrator:
    equity_values = metrics.get("recent_equity_values", [])
    if equity_values:
        equity_df = pd.DataFrame(equity_values)
        if "timestamp" in equity_df.columns:
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
            
            fig_equity = px.line(
                equity_df,
                x="timestamp",
                y="equity",
                title="Account Equity Over Time",
                labels={"equity": "Equity ($)", "timestamp": "Time"}
            )
            fig_equity.update_traces(line_color="#1f77b4")
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # Equity stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Equity", f"${equity_df['equity'].iloc[-1]:,.2f}")
            with col2:
                st.metric("Max Equity", f"${equity_df['equity'].max():,.2f}")
            with col3:
                change = equity_df['equity'].iloc[-1] - equity_df['equity'].iloc[0]
                st.metric("Total Change", f"${change:,.2f}", delta=f"{(change/equity_df['equity'].iloc[0]*100):.2f}%")
    else:
        st.info("No equity data available yet.")
else:
    st.info("Orchestrator not initialized.")

st.markdown("---")

# LLM Performance
st.header("🤖 LLM Performance")
if st.session_state.orchestrator:
    col1, col2 = st.columns(2)
    
    with col1:
        llm_data = {
            "Success": metrics.get("llm_success_count", 0),
            "Failure": metrics.get("llm_failure_count", 0)
        }
        fig_llm = px.bar(
            x=list(llm_data.keys()),
            y=list(llm_data.values()),
            title="LLM Calls: Success vs Failure",
            labels={"x": "Status", "y": "Count"},
            color=list(llm_data.keys()),
            color_discrete_map={"Success": "#28a745", "Failure": "#dc3545"}
        )
        st.plotly_chart(fig_llm, use_container_width=True)
    
    with col2:
        st.metric("Total LLM Calls", metrics.get("llm_success_count", 0) + metrics.get("llm_failure_count", 0))
        st.metric("Success Rate", f"{metrics.get('llm_success_rate', 0):.1f}%")
        st.metric("Success Count", metrics.get("llm_success_count", 0))
        st.metric("Failure Count", metrics.get("llm_failure_count", 0))

# Auto-refresh
st.markdown("---")
auto_refresh = st.checkbox("🔄 Auto-refresh (every 5 seconds)", value=False)

if auto_refresh:
    time.sleep(5)
    st.rerun()

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Info about how metrics work
if st.session_state.orchestrator:
    st.info("""
    ℹ️ **Note:** This dashboard creates its own orchestrator instance to read metrics. 
    The trading system (`python main.py`) runs separately and writes to the database.
    Metrics shown here reflect the orchestrator instance in this dashboard session.
    For complete historical data, check the database directly.
    """)
else:
    st.warning("⚠️ Orchestrator not initialized. Metrics will not be available.")

