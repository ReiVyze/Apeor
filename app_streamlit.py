import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import random
import re
import os
from collections import Counter
from model_inference import analyze_sentiment, analyze_batch, calculate_metrics, get_actionable_category, check_relevance
from database import db_handler  # Added for MongoDB integration

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Processed Data State
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

if 'ingest_method' not in st.session_state:
    st.session_state.ingest_method = None

if 'metadata' not in st.session_state:
    st.session_state.metadata = {}

if 'history' not in st.session_state:
    st.session_state.history = []

STOPWORDS = {'the','a','an','is','it','in','of','to','and','for','i','my','this','that','was','be','are','with','on','at','have','not','but', 'the', 'their', 'they', 'our', 'will', 'all', 'would', 'the', 'more'}

def get_word_freq(df, top_n=9):
    text_col = 'comment' if 'comment' in df.columns else 'text'
    if text_col not in df.columns:
        return []
    all_text = ' '.join(df[text_col].dropna().astype(str).str.lower())
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    filtered = [w for w in words if w not in STOPWORDS]
    return Counter(filtered).most_common(top_n)


# Theme Colors
COLORS = {'Positive': '#22c55e', 'Neutral': '#f59e0b', 'Negative': '#ef4444'}

# --- PAGE CONFIG ---
st.set_page_config(page_title="OracleE Sentiment Dashboard", page_icon="〇", layout="wide", initial_sidebar_state="expanded")

# --- LOAD CSS/HTML ---
def load_html(relative_path):
    full_path = os.path.join(BASE_DIR, relative_path)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

st.markdown(f"<style>{load_html('static/style.css')}</style>", unsafe_allow_html=True)

# --- SIDEBAR LOGIC ---
if 'active_page' not in st.session_state:
    st.session_state.active_page = "Sentiment Analysis"

st.sidebar.markdown('''
<div class="sidebar-logo">
    <div class="logo-circle" style="background:white; color:#111010;">C</div><div class="logo-text">ORACLEE</div>
</div>
''', unsafe_allow_html=True)

user_role = st.sidebar.selectbox("Perspective:", ["👤 Active Citizen", "💼 Official Policymaker"], index=1)

# --- NAVIGATION ---
if user_role == "👤 Active Citizen":
    nav_items = [
        {"label": "Submit Feedback", "icon": "✍️"}
    ]
    if st.session_state.active_page not in ["Submit Feedback"]:
        st.session_state.active_page = "Submit Feedback"
else:
    nav_items = [
        {"label": "Overview", "icon": "⊞"},
        {"label": "Sentiment Analysis", "icon": "☻"},
        {"label": "Feedback Channels", "icon": "💬"}
    ]
    if st.session_state.active_page == "Submit Feedback":
        st.session_state.active_page = "Sentiment Analysis"

for item in nav_items:
    cls = "nav-active" if st.session_state.active_page == item["label"] else "nav-item"
    st.sidebar.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    if st.sidebar.button(f"{item['icon']}   {item['label']}"):
        st.session_state.active_page = item["label"]
        st.rerun()
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# --- AI MODEL (pinned via CSS) ---
if user_role == "👤 Active Citizen":
    model_type = st.sidebar.selectbox("AI Model:", ["Vader (Lexicon)"])
else:
    model_type = st.sidebar.selectbox("AI Model:", ["RoBERTa (HF)", "Local DistilBert", "Vader (Lexicon)"])

# --- TOP ROW (PROFILE) ---
_, col_profile = st.columns([6, 1])
with col_profile:
    st.markdown('<div class="profile-avatar">👤</div>', unsafe_allow_html=True)

# --- MAIN CONTENT ---
if st.session_state.active_page in ["Sentiment Analysis", "Submit Feedback"]:
    
    if user_role == "👤 Active Citizen":
        # FULL HEADER FOR CITIZEN
        st.markdown('<div class="action-row"><div class="page-title" style="flex:1;">Submit Feedback</div></div>', unsafe_allow_html=True)
        col_f, col_n = st.columns([1.5, 1], gap="large")
        with col_f:
            pc = st.selectbox("Policy Clause:", ["Clause 1: Economics", "Clause 2: Healthcare", "Clause 3: Environment"])
            rg = st.selectbox("Region:", ["North", "South", "East", "West", "Central"])
            cmt = st.text_area("Feedback:", height=150)
            if st.button("Submit Feedback", use_container_width=True):
                 if cmt.strip():
                     with st.spinner("Analyzing your feedback..."):
                         if not check_relevance(cmt, pc):
                             st.error(f"❌ Feedback irrelevant. Please ensure your feedback is related to **{pc.split(': ')[-1]}**.")
                         else:
                             res = analyze_sentiment(cmt, model_type, pc)
                             
                             new_entry = {
                                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                 "text": cmt, "label": res['label'], "score": res['score'],
                                 "theme": pc, "region": rg, "channel": "Direct Portal", "model": model_type
                             }
                             
                             # Save to MongoDB
                             db_handler.save_feedback(new_entry.copy())
                             
                             # Save to history
                             st.session_state.history.insert(0, new_entry)
                             
                             # Pipe directly into the live dashboard if it's open
                             if st.session_state.processed_df is not None:
                                 new_df = pd.DataFrame([new_entry])
                                 st.session_state.processed_df = pd.concat([st.session_state.processed_df, new_df], ignore_index=True)
    
                             st.success(f"✅ Feedback submitted successfully! The AI analyzed your feedback as: **{res['label']}**")
                 else:
                     st.warning("Please enter some feedback before submitting.")
        with col_n:
            st.markdown(load_html("templates/privacy.html"), unsafe_allow_html=True)
            
        # --- CITIZEN SUBMISSION HISTORY ---
        if st.session_state.history:
            st.markdown('<br><div class="page-title" style="font-size: 1.2rem;">Your Recent Submissions</div>', unsafe_allow_html=True)
            for entry in st.session_state.history[:5]: # Show last 5
                st.markdown(f'''
                <div class="mc" style="margin-bottom: 0.75rem; border-left: 4px solid {COLORS.get(entry['label'], "#6b7280")};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size: 0.7rem; color: #6b7280; font-weight:600;">{entry['timestamp']}</span>
                        <span class="channel-badge" style="background: #f3f4f6; color: #111827;">{entry['theme']}</span>
                    </div>
                    <div style="font-size: 0.85rem; margin-top: 0.5rem; color: #111827;">"{entry['text']}"</div>
                    <div style="font-size: 0.75rem; margin-top: 0.4rem; font-weight: 700; color: {COLORS.get(entry['label'], "#6b7280")};">Analysis: {entry['label']} ({int(entry['score']*100)}% confidence)</div>
                </div>
                ''', unsafe_allow_html=True)

    else:
        if st.session_state.processed_df is None:
            st.markdown('<div class="page-title">Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
            
            db_data = db_handler.get_all_feedback()
            if db_data:
                df_db = pd.DataFrame(db_data)
            else:
                df_db = pd.DataFrame(columns=['timestamp', 'text', 'label', 'score', 'theme', 'region', 'channel', 'model'])
            
            st.session_state.processed_df = df_db
            st.rerun()


        else:
            # --- FULL HEADER & DASHBOARD FILTERS ---
            st.markdown('<div class="page-title">Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
            
            df = st.session_state.processed_df.copy()
            # Ensure timestamp for Date Filter
            if 'timestamp' in df.columns:
                df['date_dt'] = pd.to_datetime(df['timestamp']).dt.date
            else:
                df['date_dt'] = datetime.now().date()

            # FILTER ROW
            with st.expander("⚙️ Dashboard Filters", expanded=False):
                col_d, col_c, col_t, col_s = st.columns([1.2, 1, 1, 1])
                
                with col_d:
                    min_d, max_d = df['date_dt'].min(), df['date_dt'].max()
                    if pd.isna(min_d) or pd.isna(max_d):
                        min_d = max_d = datetime.now().date()
                    date_range = st.date_input("📅 Date Range", (min_d, max_d))
                
                with col_c:
                    if 'channel' in df.columns and len(df) > 0:
                        ch_options = sorted(df['channel'].unique().tolist())
                    else:
                        ch_options = ["Direct Portal", "Email", "Chat", "Social Media"]
                    
                    if not ch_options:
                        ch_options = ["Direct Portal", "Email", "Chat", "Social Media"]
                        
                    sel_channels = st.multiselect("✉ Channels", ch_options, default=ch_options)
                
                with col_t:
                    th_options = sorted(df['theme'].unique().tolist()) if ('theme' in df.columns and len(df) > 0) else ["General"]
                    sel_themes = st.multiselect("🏷 Themes", th_options, default=th_options)
                    
                with col_s:
                    sent_options = ["Positive", "Neutral", "Negative"]
                    sel_sent = st.multiselect("☻ Sentiment", sent_options, default=sent_options)

            # APPLY FILTERS
            if len(date_range) == 2:
                df = df[(df['date_dt'] >= date_range[0]) & (df['date_dt'] <= date_range[1])]
            
            if 'channel' in df.columns:
                df = df[df['channel'].isin(sel_channels)]
            
            if 'theme' in df.columns:
                df = df[df['theme'].isin(sel_themes)]
            
            if 'label' in df.columns and len(df) > 0:
                df = df[df['label'].isin(sel_sent)]
            
            # --- ACTION BAR (EXPORT) ---
            col_exp_1, col_exp_2 = st.columns([6, 1.2])
            with col_exp_2:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"⎘ Export {len(df)} Records",
                    data=csv_data,
                    file_name=f"sentiment_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            metrics = calculate_metrics(df)
            total = metrics['total']
            
            # METRICS ROW
            st.markdown(f'''
            <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                <div class="mc" style="width: 700px; display: flex; align-items: center; justify-content: space-between; padding: 20px 40px;">
                    <div>
                        <div class="mc-label">Total Feedback</div>
                        <div class="mc-val">{total}</div>
                    </div>
                    <div style="border-left: 1px solid #e4e8ef; height: 50px; margin: 0 30px;"></div>
                    <div style="display: flex; gap: 20px;">
                        <div style="background: #f0fdf4; padding: 10px 15px; border-radius: 8px; display:flex; align-items:center; gap:12px;"><span style="font-size:0.85rem; color:#6b7280;">Positive</span> <span style="font-size: 1.3rem; font-weight:700; color:#111827;">{metrics['pos_pct']}%</span></div>
                        <div style="background: #fefce8; padding: 10px 15px; border-radius: 8px; display:flex; align-items:center; gap:12px;"><span style="font-size:0.85rem; color:#6b7280;">Neutral</span> <span style="font-size: 1.3rem; font-weight:700; color:#111827;">{metrics['neu_pct']}%</span></div>
                        <div style="background: #f8fafc; padding: 10px 15px; border-radius: 8px; display:flex; align-items:center; gap:12px;"><span style="font-size:0.85rem; color:#6b7280;">Negative</span> <span style="font-size: 1.3rem; font-weight:700; color:#111827;">{metrics['neg_pct']}%</span></div>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            # CHARTS ROW 1 (Centered Line Chart)
            # REAL DATA TREND
            if len(df) == 0:
                days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']
                pos, neu, neg = [0]*7, [0]*7, [0]*7
            else:
                df['_idx'] = range(len(df))
                df['_bin'] = pd.cut(df['_idx'], bins=min(len(df), 7), labels=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][:min(len(df), 7)])
                
                pos, neu, neg = [], [], []
                days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][:min(len(df), 7)]
                for day in days:
                    chunk = df[df['_bin'] == day]
                    total_chunk = len(chunk) or 1
                    pos.append(round(len(chunk[chunk['label']=='Positive']) / total_chunk * 100, 1))
                    neu.append(round(len(chunk[chunk['label']=='Neutral']) / total_chunk * 100, 1))
                    neg.append(round(len(chunk[chunk['label']=='Negative']) / total_chunk * 100, 1))
            
            # Calculate daily changes for hover tooltips
            pos_diff = [0] + [round(pos[i] - pos[i-1], 1) for i in range(1, len(pos))]
            neu_diff = [0] + [round(neu[i] - neu[i-1], 1) for i in range(1, len(neu))]
            neg_diff = [0] + [round(neg[i] - neg[i-1], 1) for i in range(1, len(neg))]

            fig_line = go.Figure()
            
            # Common hover template
            h_temp = "<b>%{x}</b><br>Sentiment: %{y}%<br>Change: %{customdata:+.1f}%<extra></extra>"

            fig_line.add_trace(go.Scatter(
                x=days, y=pos, mode='lines+markers', 
                line=dict(color='#22c55e', width=3, shape='spline'), 
                name='Positive', customdata=pos_diff, hovertemplate=h_temp
            ))
            fig_line.add_trace(go.Scatter(
                x=days, y=neu, mode='lines+markers', 
                line=dict(color='#f59e0b', width=3, shape='spline'), 
                name='Neutral', customdata=neu_diff, hovertemplate=h_temp
            ))
            fig_line.add_trace(go.Scatter(
                x=days, y=neg, mode='lines+markers', 
                line=dict(color='#ef4444', width=3, shape='spline'), 
                name='Negative', customdata=neg_diff, hovertemplate=h_temp
            ))
            
            # Dynamic Annotation
            if len(pos) >= 3:
                tue_val = pos[2] # Tuesday logic
                tue_diff = pos_diff[2]
                fig_line.add_trace(go.Scatter(x=[days[2]], y=[tue_val], mode='markers', marker=dict(color='#111827', size=10), hoverinfo='none', showlegend=False))
                fig_line.add_annotation(x=days[2], y=tue_val, text=f"<span style='color:#6b7280;font-size:11px;'>Trend Point</span><br><b>{tue_val:.1f}%</b> ({tue_diff:+.1f}%)", showarrow=True, arrowhead=0, arrowcolor='white', bordercolor='#e4e8ef', borderwidth=1, borderpad=6, bgcolor='white', font=dict(color='#111827', size=14), ax=40, ay=-40)

            fig_line.update_layout(
                title=dict(text="<b>Sentiment Trend Overview (Daily Shift)</b>", font=dict(color='#111827', size=16), y=0.95, x=0.5, xanchor='center'),
                showlegend=True, 
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", y=1.05, 
                    xanchor="center", x=0.5, 
                    font=dict(size=12, color='#6b7280')
                ),
                hovermode="x unified",
                margin=dict(l=80, r=80, t=100, b=60), height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#f3f4f6', tickfont=dict(color='#9ca3af', size=12)),
                yaxis=dict(showgrid=True, gridcolor='#f3f4f6', tickfont=dict(color='#9ca3af', size=12), title="Sentiment Percentage", title_font=dict(color='#9ca3af', size=12)),
                transition={'duration': 500, 'easing': 'cubic-in-out'},
                uirevision='true'
            )
            
            # Use columns to center the line chart
            c_left, c_mid, c_right = st.columns([0.1, 0.8, 0.1])
            with c_mid:
                st.plotly_chart(fig_line, use_container_width=True, config={
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['toImage', 'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'],
                    'displaylogo': False
                })

            # CHARTS ROW 2
            col_bar, col_horiz = st.columns(2)
            
            with col_bar:
                if 'theme' in df.columns and len(df) > 0:
                    top3 = df['theme'].value_counts().head(3).index.tolist()
                    themes = top3
                    vals = [len(df[df['theme'] == t]) for t in top3]
                    colors = []
                    texts = []
                    for t in top3:
                        sub = df[df['theme'] == t]
                        dominant = sub['label'].value_counts().idxmax()
                        colors.append(COLORS.get(dominant, '#6b7280'))
                        texts.append(f"Sentiment: {dominant}")
                else:
                    themes = ['Positive', 'Neutral', 'Negative']
                    vals = [metrics['pos_count'], metrics['neu_count'], metrics['neg_count']]
                    colors = ['#22c55e', '#f59e0b', '#ef4444']
                    texts = ['', '', '']
                
                fig_bar = go.Figure(go.Bar(
                    x=vals, y=themes, orientation='h',
                    marker_color=colors, width=0.6,
                    text=texts, textposition='outside', textfont=dict(color='#9ca3af', size=11), hoverinfo='none'
                ))
                fig_bar.update_layout(
                    title=dict(text="<b>Top 3 Feedback Themes/Topics</b>", font=dict(color='#111827', size=14)),
                    margin=dict(l=60, r=100, t=70, b=30), height=310, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(title="Themes/Topics", title_font=dict(color='#9ca3af', size=12), tickfont=dict(color='#6b7280', size=11)),
                    transition={'duration': 500, 'easing': 'cubic-in-out'},
                    uirevision='true'
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['toImage', 'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'],
                    'displaylogo': False
                })
                
            with col_horiz:
                word_freq = get_word_freq(df)
                words = [w for w, _ in word_freq]
                freqs = [c for _, c in word_freq]
                
                # Fallback if no words found
                if not words:
                    words = ['No Data']
                    freqs = [0]
                
                fig_wf = go.Figure(go.Bar(
                    x=freqs, y=words, orientation='h',
                    marker_color='#111827', width=0.5,
                    text=freqs, textposition='outside', textfont=dict(color='#9ca3af', size=10), hoverinfo='none'
                ))
                fig_wf.update_layout(
                    title=dict(text="<b>Word Frequency by Sentiment</b>", font=dict(color='#111827', size=14)),
                    margin=dict(l=60, r=80, t=70, b=30), height=310, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(tickfont=dict(color='#6b7280', size=10)),
                    transition={'duration': 500, 'easing': 'cubic-in-out'},
                    uirevision='true'
                )
                st.plotly_chart(fig_wf, use_container_width=True, config={
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['toImage', 'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'],
                    'displaylogo': False
                })
elif st.session_state.active_page == "Overview":
    st.markdown('<div class="action-row"><div class="page-title" style="flex:1;">Platform Overview</div></div>', unsafe_allow_html=True)
    st.markdown(load_html("templates/overview.html"), unsafe_allow_html=True)
elif st.session_state.active_page == "Feedback Channels":
    st.markdown('<div class="action-row"><div class="page-title" style="flex:1;">Feedback Channels Analysis</div></div>', unsafe_allow_html=True)
    
    # --- LOAD REAL DATA FROM SESSION STATE ---
    df_src = st.session_state.processed_df.copy() if st.session_state.processed_df is not None else pd.DataFrame()
    
    # Standardize columns for the Channel view
    if not df_src.empty:
        # Create columns needed for the template/table if they don't exist
        if 'timestamp' in df_src.columns:
            df_src['Timestamp'] = df_src['timestamp']
        else:
            df_src['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        if 'theme' in df_src.columns:
            df_src['Theme'] = df_src['theme']
            
        if 'text' in df_src.columns:
            df_src['Feedback'] = df_src['text']
            
        if 'channel' in df_src.columns:
            df_src['Channel'] = df_src['channel']
            
        df_src['Customer ID'] = [f"Citizen-{100 + i}" for i in range(len(df_src))]
        
        # Ensure ratings columns exist for calculation (even if None)
        if 'nps_rating' not in df_src.columns:
            df_src['nps_rating'] = None
        if 'csat_rating' not in df_src.columns:
            df_src['csat_rating'] = None
    else:
        df_src = pd.DataFrame(columns=["Timestamp", "Customer ID", "Channel", "Theme", "Feedback", "label", "nps_rating", "csat_rating"])

    cdf = df_src
    
    # Render HTML Template
    html_content = load_html("templates/feedback_channels.html")
    st.markdown(html_content, unsafe_allow_html=True)

    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed Data Table
    st.markdown('<div class="page-title" style="font-size: 1.2rem;">Live Feedback Stream</div>', unsafe_allow_html=True)
    
    # Add 'Actionable Category' column on the fly
    cdf['Actionable Category'] = cdf['Feedback'].apply(lambda x: get_actionable_category(x))
    
    display_df = cdf[["Timestamp", "Customer ID", "Channel", "Theme", "Actionable Category", "label"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

else:
    st.markdown('<div class="action-row"><div class="page-title" style="flex:1;">' + st.session_state.active_page + '</div></div>', unsafe_allow_html=True)
    st.info("This section is under construction. Please use the 'Sentiment Analysis' tab for the demo.")
