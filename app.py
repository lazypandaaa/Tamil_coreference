
import streamlit as st
import json
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple
from pydantic import BaseModel
import numpy as np
from google import genai





# Hide GitHub button, footer, and optionally main menu
hide_github_icon = """
    <style>
    /* Hide Streamlit GitHub icon (top-right) */
    .stApp a[data-testid="stAppGithubLink"] {
        display: none !important;
    }

    /* Hide Streamlit footer */
    footer {visibility: hidden;}

    /* Hide Streamlit hamburger menu (optional) */
    #MainMenu {visibility: hidden;}
    </style>
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)






# Configure Streamlit page
st.set_page_config(
    page_title="Tamil Coreference Resolution",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .cluster-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .word-mapping {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Pydantic models
class Mention(BaseModel):
    text: str
    start_index: int
    end_index: int
    sentence_index: int

class CoreferenceCluster(BaseModel):
    cluster_id: int
    entity_name: str
    mentions: List[Mention]
    entity_type: str

class CoreferenceResult(BaseModel):
    text: str
    clusters: List[CoreferenceCluster]
    total_mentions: int

# Tamil Text Processor
class TamilTextProcessor:
    def __init__(self):
        self.word_to_number = {}
        self.number_to_word = {}
        self.word_counter = 1

    def tokenize_tamil(self, text: str) -> List[str]:
        return re.findall(r'\S+', text)

    def map_words_to_numbers(self, text: str) -> Dict[str, int]:
        words = self.tokenize_tamil(text)
        for word in words:
            if word not in self.word_to_number:
                self.word_to_number[word] = self.word_counter
                self.number_to_word[self.word_counter] = word
                self.word_counter += 1
        return self.word_to_number

# Gemini API Integration
class GeminiCoreferenceResolver:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def resolve_coreferences(self, tamil_text: str) -> CoreferenceResult:
        prompt = f"""
        Analyze the following Tamil text and identify coreference clusters. A coreference cluster contains all mentions that refer to the same entity.

        Tamil Text: "{tamil_text}"

        Instructions:
        1. Identify all mentions (nouns, pronouns, proper names) that refer to entities
        2. Group mentions that refer to the same entity into clusters
        3. For each mention, provide exact text and character positions
        4. Classify entity types as: PERSON, PLACE, ORGANIZATION, or OTHER
        5. Give each cluster a meaningful English entity name (e.g., "Ram", "Teacher", "Student")

        Be precise with character positions and ensure all referring expressions are captured.
        """

        try:
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": CoreferenceResult,
                }
            )

            result = json.loads(response.text)
            return CoreferenceResult(**result)

        except Exception as e:
            st.error(f"Error with Gemini API: {e}")
            return CoreferenceResult(text=tamil_text, clusters=[], total_mentions=0)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = TamilTextProcessor()
if 'resolver' not in st.session_state:
    st.session_state.resolver = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

    
def main():
    # Header
    st.markdown('<h1 class="main-header">🔗 Tamil Coreference Resolution System</h1>', unsafe_allow_html=True)

    # API Key input (hardcoded here, ideally should be from secrets or sidebar)
    api_key = "AIzaSyDy9M-SIfT1vAwndfObo7xKOOvz6-Hxxxs"

    if api_key:
        if "resolver" not in st.session_state or st.session_state.resolver is None:
            st.session_state.resolver = GeminiCoreferenceResolver(api_key)

    st.markdown("---")

    # Sample texts
    st.subheader("📝 Sample Texts")
    sample_options = {
        "Example 1": "ராம் ஒரு மாணவன். அவன் தனது வீட்டை விட்டு கிளம்பினான். சீதா ஒரு ஆசிரியை. அவள் அவன் தன்னை கண்ட போது.",
        "Example 2": "ராஜா ஒரு டாக்டர். அவர் மருத்துவமனையில் வேலை செய்கிறார். அவருடைய மனைவி ஒரு செவிலியர்.",
        "Example 3": "மீனா பள்ளிக்கு போனாள். அவள் தன் நண்பர்களை சந்தித்தாள். அவர்கள் ஒன்றாக படித்தனர்."
    }

    selected_sample = st.selectbox("Choose a sample:", ["Custom"] + list(sample_options.keys()))

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📄 Input Tamil Text")

        if selected_sample != "Custom":
            default_text = sample_options[selected_sample]
        else:
            default_text = ""


        tamil_text = st.text_area(
            "Enter Tamil text for coreference analysis:",
            value=default_text,
            height=150,
            help="Enter Tamil text with multiple sentences containing entities and references"
        )

        # Analysis button
        if st.button("🔍 Analyze Coreferences", disabled=not api_key or not tamil_text):
            with st.spinner("Analyzing Tamil text ..."):
                # Process text
                word_mapping = st.session_state.processor.map_words_to_numbers(tamil_text)

                # Resolve coreferences
                result = st.session_state.resolver.resolve_coreferences(tamil_text)
                st.session_state.analysis_result = result

                if result.clusters:
                    st.success(f"✅ Analysis complete! Found {len(result.clusters)} coreference clusters.")
                else:
                    st.warning("⚠️ No coreference clusters detected in the text.")

    with col2:
        st.subheader("📊 Quick Stats")
        if st.session_state.analysis_result:
            result = st.session_state.analysis_result

            # Metrics
            st.metric("Total Clusters", len(result.clusters))
            st.metric("Total Mentions", result.total_mentions)
            st.metric("Unique Words", len(st.session_state.processor.word_to_number))

            # Entity types breakdown
            if result.clusters:
                entity_types = [cluster.entity_type for cluster in result.clusters]
                entity_counts = pd.Series(entity_types).value_counts()

                st.subheader("Entity Types")
                for entity_type, count in entity_counts.items():
                    st.write(f"• {entity_type}: {count}")

    # Results section
    if st.session_state.analysis_result and st.session_state.analysis_result.clusters:
        st.markdown("---")
        display_results(st.session_state.analysis_result)

def display_results(result: CoreferenceResult):
    """Display analysis results with visualizations"""

    st.header("📈 Analysis Results")

    # Word mapping display
    with st.expander("🔢 Word-to-Number Mapping", expanded=False):
        word_mapping = st.session_state.processor.word_to_number

        # Create mapping dataframe
        mapping_df = pd.DataFrame([
            {"Word Number": num, "Tamil Word": word}
            for word, num in word_mapping.items()
        ])

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(mapping_df, use_container_width=True)

        with col2:
            # Text with numbers
            words = st.session_state.processor.tokenize_tamil(result.text)
            number_text = " ".join([str(word_mapping.get(word, 0)) for word in words])
            st.subheader("Text as Numbers:")
            st.code(number_text)

    # Cluster information
    st.subheader("🔗 Detected Coreference Clusters")

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

    for i, cluster in enumerate(result.clusters):
        color = colors[i % len(colors)]

        with st.container():
            # Cluster header
            st.markdown(f"""
            <div class="cluster-header" style="background: {color};">
                <h4>Cluster {cluster.cluster_id}: {cluster.entity_name} ({cluster.entity_type})</h4>
            </div>
            """, unsafe_allow_html=True)

            # Mentions
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Tamil Mentions:**")
                mentions_text = " → ".join([mention.text for mention in cluster.mentions])
                st.write(mentions_text)

            with col2:
                st.write("**Number Mappings:**")
                mentions_numbers = " → ".join([
                    str(st.session_state.processor.word_to_number.get(mention.text, 0))
                    for mention in cluster.mentions
                ])
                st.write(mentions_numbers)

    # Visualizations
    st.markdown("---")
    st.header("📊 Visualizations")

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Word Distribution", "🌐 Arc Visualization", "🕸️ Network Graph", "📈 Cluster Analysis"])

    with tab1:
        create_word_distribution_chart(result)

    with tab2:
        create_arc_visualization(result)

    with tab3:
        create_network_visualization(result)

    with tab4:
        create_cluster_analysis(result)

def create_word_distribution_chart(result: CoreferenceResult):
    """Create word distribution bar chart"""
    words = st.session_state.processor.tokenize_tamil(result.text)
    word_numbers = [st.session_state.processor.word_to_number.get(word, 0) for word in words]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(words))),
        y=word_numbers,
        text=[f"Word {i+1}" for i in range(len(words))],
        textposition='auto',
        marker_color='lightblue',
        name='Word Numbers'
    ))

    fig.update_layout(
        title="Word Position vs Mapped Numbers",
        xaxis_title="Word Position in Text",
        yaxis_title="Assigned Number",
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

def create_arc_visualization(result: CoreferenceResult):
    """Create arc-based coreference visualization"""
    words = st.session_state.processor.tokenize_tamil(result.text)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    fig = go.Figure()

    # Add word positions
    fig.add_trace(go.Scatter(
        x=list(range(len(words))),
        y=[0] * len(words),
        mode='markers+text',
        text=[str(st.session_state.processor.word_to_number.get(word, 0)) for word in words],
        textposition="bottom center",
        marker=dict(size=12, color='lightblue'),
        name='Words (as numbers)',
        showlegend=False
    ))

    # Add arcs for each cluster
    for cluster_idx, cluster in enumerate(result.clusters):
        color = colors[cluster_idx % len(colors)]
        mention_positions = []

        # Find positions of mentions
        for mention in cluster.mentions:
            for pos, word in enumerate(words):
                if word == mention.text:
                    mention_positions.append(pos)
                    break

        # Draw arcs between consecutive mentions
        for i in range(len(mention_positions) - 1):
            start_pos = mention_positions[i]
            end_pos = mention_positions[i + 1]

            # Create arc
            mid_x = (start_pos + end_pos) / 2
            height = 0.5 + cluster_idx * 0.3

            arc_x = np.linspace(start_pos, end_pos, 50)
            arc_y = [height * (1 - 4 * (x - mid_x) ** 2 / (end_pos - start_pos) ** 2)
                    if end_pos != start_pos else height for x in arc_x]

            fig.add_trace(go.Scatter(
                x=arc_x,
                y=arc_y,
                mode='lines',
                line=dict(color=color, width=3),
                name=f'{cluster.entity_name}',
                showlegend=i == 0  # Only show legend for first arc of each cluster
            ))

    fig.update_layout(
        title="Coreference Arcs (Words shown as numbers)",
        xaxis_title="Word Position",
        yaxis_title="Arc Height",
        height=500,
        yaxis=dict(showticklabels=False)
    )

    st.plotly_chart(fig, use_container_width=True)

def create_network_visualization(result: CoreferenceResult):
    """Create network graph visualization"""

    # Create network data
    nodes = []
    edges = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    for cluster_idx, cluster in enumerate(result.clusters):
        cluster_color = colors[cluster_idx % len(colors)]
        cluster_mentions = []

        # Add nodes for each mention
        for i, mention in enumerate(cluster.mentions):
            word_num = st.session_state.processor.word_to_number.get(mention.text, 0)
            node_id = f"{cluster.entity_name}_{i}"

            nodes.append({
                'id': node_id,
                'label': f'{cluster.entity_name}\n({word_num})',
                'color': cluster_color,
                'cluster': cluster.entity_name
            })
            cluster_mentions.append(node_id)

        # Add edges within cluster
        for i in range(len(cluster_mentions)):
            for j in range(i + 1, len(cluster_mentions)):
                edges.append({
                    'source': cluster_mentions[i],
                    'target': cluster_mentions[j],
                    'color': cluster_color
                })

    if not nodes:
        st.info("No network to display - no coreference clusters found.")
        return

    # Create network layout using networkx
    G = nx.Graph()
    for node in nodes:
        G.add_node(node['id'], **node)
    for edge in edges:
        G.add_edge(edge['source'], edge['target'])

    pos = nx.spring_layout(G, k=2, iterations=50)

    # Create Plotly network visualization
    fig = go.Figure()

    # Add edges
    for edge in edges:
        x0, y0 = pos[edge['source']]
        x1, y1 = pos[edge['target']]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=2, color=edge['color']),
            showlegend=False,
            hoverinfo='none'
        ))

    # Add nodes by cluster
    for cluster_name in set(node['cluster'] for node in nodes):
        cluster_nodes = [node for node in nodes if node['cluster'] == cluster_name]
        x_coords = [pos[node['id']][0] for node in cluster_nodes]
        y_coords = [pos[node['id']][1] for node in cluster_nodes]
        labels = [node['label'] for node in cluster_nodes]

        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='markers+text',
            marker=dict(size=20, color=cluster_nodes[0]['color']),
            text=labels,
            textposition="middle center",
            name=cluster_name,
            textfont=dict(size=10, color='white')
        ))

    fig.update_layout(
        title="Coreference Network Graph",
        showlegend=True,
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    st.plotly_chart(fig, use_container_width=True)

def create_cluster_analysis(result: CoreferenceResult):
    """Create cluster analysis charts"""

    col1, col2 = st.columns(2)

    with col1:
        # Cluster size distribution
        cluster_sizes = [len(cluster.mentions) for cluster in result.clusters]
        cluster_names = [cluster.entity_name for cluster in result.clusters]

        fig1 = px.bar(
            x=cluster_names,
            y=cluster_sizes,
            title="Mentions per Cluster",
            labels={'x': 'Entity', 'y': 'Number of Mentions'},
            color=cluster_sizes,
            color_continuous_scale='viridis'
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Entity type distribution
        entity_types = [cluster.entity_type for cluster in result.clusters]
        entity_type_counts = pd.Series(entity_types).value_counts()

        fig2 = px.pie(
            values=entity_type_counts.values,
            names=entity_type_counts.index,
            title="Entity Type Distribution"
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Detailed cluster information table
    st.subheader("📋 Detailed Cluster Information")

    cluster_data = []
    for cluster in result.clusters:
        cluster_data.append({
            'Cluster ID': cluster.cluster_id,
            'Entity Name': cluster.entity_name,
            'Entity Type': cluster.entity_type,
            'Mention Count': len(cluster.mentions),
            'Tamil Mentions': ' | '.join([mention.text for mention in cluster.mentions]),
            'Number Mappings': ' | '.join([
                str(st.session_state.processor.word_to_number.get(mention.text, 0))
                for mention in cluster.mentions
            ])
        })

    cluster_df = pd.DataFrame(cluster_data)
    st.dataframe(cluster_df, use_container_width=True)

    # Download results
    if st.button("💾 Download Results as JSON"):
        results_json = result.dict()
        st.download_button(
            label="Download JSON",
            data=json.dumps(results_json, indent=2, ensure_ascii=False),
            file_name="tamil_coreference_results.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
