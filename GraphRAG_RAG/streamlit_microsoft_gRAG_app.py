import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import json
import os
from pathlib import Path
import numpy as np

# Graph RAG imports
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# Configure Streamlit page
st.set_page_config(
    page_title="Graph RAG Knowledge Explorer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GraphRAGVisualizer:
    def __init__(self):
        self.entities = None
        self.relationships = None
        self.communities = None
        self.reports = None
        self.text_units = None
        self.search_engine = None
        self.global_search = None
        
    @st.cache_data
    def load_data(_self, artifacts_path="./output/artifacts"):
        """Load Graph RAG artifacts"""
        try:
            entities = read_indexer_entities(artifacts_path, entity_table="create_final_entities")
            relationships = read_indexer_relationships(artifacts_path)
            communities = read_indexer_communities(artifacts_path, community_table="create_final_communities")
            reports = read_indexer_reports(artifacts_path, community_table="create_final_communities")
            text_units = read_indexer_text_units(artifacts_path)
            
            return entities, relationships, communities, reports, text_units
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None, None, None, None
    
    def setup_search_engines(self, api_key):
        """Setup local and global search engines"""
        try:
            llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-4",
                api_type=OpenaiApiType.OpenAI,
                max_retries=20,
            )
            
            # Setup vector store
            vector_store = LanceDBVectorStore(collection_name="default-entity-description")
            vector_store.connect(db_uri="./output/lancedb")
            
            # Local search setup
            context_builder = LocalSearchMixedContext(
                community_reports=self.reports,
                text_units=self.text_units,
                entities=self.entities,
                relationships=self.relationships,
                entity_text_embeddings=vector_store,
            )
            
            self.search_engine = LocalSearch(
                llm=llm,
                context_builder=context_builder,
                response_type="multiple paragraphs",
            )
            
            return True
        except Exception as e:
            st.error(f"Error setting up search engines: {e}")
            return False
    
    def create_network_graph(self, max_entities=100):
        """Create interactive network graph using Plotly"""
        if self.entities is None or self.relationships is None:
            return None
            
        # Sample entities if too many
        entities_sample = self.entities.head(max_entities)
        entity_titles = set(entities_sample['title'].tolist())
        
        # Filter relationships to sampled entities
        relationships_filtered = self.relationships[
            (self.relationships['source'].isin(entity_titles)) & 
            (self.relationships['target'].isin(entity_titles))
        ]
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for _, entity in entities_sample.iterrows():
            G.add_node(
                entity['title'],
                type=entity.get('type', 'unknown'),
                description=entity.get('description', '')[:100] + "...",
                degree=entity.get('degree', 0)
            )
        
        # Add edges
        for _, rel in relationships_filtered.iterrows():
            if rel['source'] in entity_titles and rel['target'] in entity_titles:
                G.add_edge(
                    rel['source'], 
                    rel['target'],
                    weight=rel.get('weight', 1),
                    description=rel.get('description', '')
                )
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Prepare data for Plotly
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{edge[0]} → {edge[1]}")
        
        node_x = []
        node_y = []
        node_info = []
        node_colors = []
        node_sizes = []
        
        # Color mapping for different entity types
        type_colors = {
            'person': '#FF6B6B',
            'organization': '#4ECDC4',
            'location': '#45B7D1',
            'event': '#96CEB4',
            'concept': '#FFEAA7',
            'technology': '#DDA0DD',
            'unknown': '#95A5A6'
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            node_info.append(f"<b>{node}</b><br>"
                           f"Type: {node_data.get('type', 'unknown')}<br>"
                           f"Degree: {node_data.get('degree', 0)}<br>"
                           f"Description: {node_data.get('description', 'No description')}")
            
            node_type = node_data.get('type', 'unknown')
            node_colors.append(type_colors.get(node_type, '#95A5A6'))
            node_sizes.append(max(10, min(50, node_data.get('degree', 1) * 3)))
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_info,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_colors,
                size=node_sizes,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    xanchor="left",
                    title="Node Type"
                ),
                line=dict(width=2)
            ),
            showlegend=False
        ))
        
        fig.update_layout(
            title='Knowledge Graph Network',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Interactive Knowledge Graph - Hover over nodes for details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='#888', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_entity_analytics(self):
        """Create analytics dashboard for entities"""
        if self.entities is None:
            return None, None, None
        
        # Entity type distribution
        type_counts = self.entities['type'].value_counts()
        fig_types = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Entity Type Distribution"
        )
        
        # Entity degree distribution
        if 'degree' in self.entities.columns:
            fig_degrees = px.histogram(
                self.entities,
                x='degree',
                title="Entity Degree Distribution",
                labels={'degree': 'Number of Connections', 'count': 'Number of Entities'}
            )
        else:
            fig_degrees = None
        
        # Top entities by degree
        if 'degree' in self.entities.columns:
            top_entities = self.entities.nlargest(20, 'degree')[['title', 'type', 'degree']]
            fig_top = px.bar(
                top_entities,
                x='degree',
                y='title',
                color='type',
                title="Top 20 Most Connected Entities",
                orientation='h'
            )
            fig_top.update_layout(height=600)
        else:
            fig_top = None
        
        return fig_types, fig_degrees, fig_top
    
    def create_community_analysis(self):
        """Create community analysis visualizations"""
        if self.communities is None:
            return None, None
        
        # Community size distribution
        if 'size' in self.communities.columns:
            fig_sizes = px.histogram(
                self.communities,
                x='size',
                title="Community Size Distribution",
                labels={'size': 'Community Size', 'count': 'Number of Communities'}
            )
        else:
            fig_sizes = None
        
        # Community details table
        community_summary = self.communities[['title', 'size', 'level']].head(20) if 'size' in self.communities.columns else self.communities[['title']].head(20)
        
        return fig_sizes, community_summary

def main():
    st.title("🕸️ Graph RAG Knowledge Explorer")
    st.markdown("Interactive visualization and querying of your PDF knowledge graph")
    
    # Initialize visualizer
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = GraphRAGVisualizer()
    
    visualizer = st.session_state.visualizer
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    # Data loading
    artifacts_path = st.sidebar.text_input("Artifacts Path", value="./output/artifacts")
    
    if st.sidebar.button("Load Data"):
        with st.spinner("Loading Graph RAG data..."):
            entities, relationships, communities, reports, text_units = visualizer.load_data(artifacts_path)
            if entities is not None:
                visualizer.entities = entities
                visualizer.relationships = relationships
                visualizer.communities = communities
                visualizer.reports = reports
                visualizer.text_units = text_units
                st.sidebar.success("Data loaded successfully!")
                
                # Setup search engines
                if api_key:
                    visualizer.setup_search_engines(api_key)
            else:
                st.sidebar.error("Failed to load data")
    
    # Main content
    if visualizer.entities is not None:
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query", "🕸️ Network Graph", "📊 Analytics", "🏘️ Communities"])
        
        with tab1:
            st.header("Query Knowledge Graph")
            
            # Search type selection
            search_type = st.radio("Search Type", ["Local Search", "Global Search"])
            
            # Query input
            query = st.text_area("Enter your question:", 
                               placeholder="e.g., What are the main topics discussed in the documents?")
            
            if st.button("Search") and query:
                if visualizer.search_engine is not None:
                    with st.spinner("Searching..."):
                        try:
                            if search_type == "Local Search":
                                result = asyncio.run(visualizer.search_engine.asearch(query))
                            else:
                                if visualizer.global_search:
                                    result = asyncio.run(visualizer.global_search.asearch(query))
                                else:
                                    st.error("Global search not configured")
                                    result = None
                            
                            if result:
                                st.subheader("Search Results")
                                st.write(result.response)
                                
                                # Show context if available
                                if hasattr(result, 'context_data') and result.context_data:
                                    with st.expander("View Context"):
                                        st.json(result.context_data)
                        except Exception as e:
                            st.error(f"Search error: {e}")
                else:
                    st.error("Search engine not configured. Please provide API key and load data.")
        
        with tab2:
            st.header("Interactive Network Graph")
            
            # Graph controls
            col1, col2 = st.columns([3, 1])
            
            with col2:
                max_entities = st.slider("Max Entities to Display", 50, 500, 100)
                
            with col1:
                # Create and display network graph
                network_fig = visualizer.create_network_graph(max_entities)
                if network_fig:
                    st.plotly_chart(network_fig, use_container_width=True)
                else:
                    st.error("Could not create network graph")
        
        with tab3:
            st.header("Knowledge Graph Analytics")
            
            # Entity analytics
            fig_types, fig_degrees, fig_top = visualizer.create_entity_analytics()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if fig_types:
                    st.plotly_chart(fig_types, use_container_width=True)
                
            with col2:
                if fig_degrees:
                    st.plotly_chart(fig_degrees, use_container_width=True)
            
            if fig_top:
                st.plotly_chart(fig_top, use_container_width=True)
            
            # Entity details table
            st.subheader("Entity Details")
            if visualizer.entities is not None:
                # Add filters
                entity_types = visualizer.entities['type'].unique()
                selected_types = st.multiselect("Filter by Entity Type", entity_types, default=entity_types)
                
                filtered_entities = visualizer.entities[visualizer.entities['type'].isin(selected_types)]
                st.dataframe(filtered_entities[['title', 'type', 'description']].head(100))
        
        with tab4:
            st.header("Community Analysis")
            
            fig_sizes, community_summary = visualizer.create_community_analysis()
            
            if fig_sizes:
                st.plotly_chart(fig_sizes, use_container_width=True)
            
            if community_summary is not None:
                st.subheader("Community Summary")
                st.dataframe(community_summary)
            
            # Community reports
            if visualizer.reports is not None:
                st.subheader("Community Reports")
                selected_community = st.selectbox("Select Community", visualizer.reports['title'].tolist())
                if selected_community:
                    report = visualizer.reports[visualizer.reports['title'] == selected_community]['full_content'].iloc[0]
                    st.markdown(report)
    
    else:
        st.info("Please load your Graph RAG data using the sidebar to start exploring!")
        
        # Show sample data structure
        st.subheader("Expected Data Structure")
        st.markdown("""
        Make sure you have:
        1. Completed the Graph RAG indexing process
        2. Generated artifacts in the output directory
        3. Provided your OpenAI API key
        
        The app will visualize:
        - **Entities**: People, organizations, concepts, etc.
        - **Relationships**: Connections between entities
        - **Communities**: Clusters of related entities
        - **Search Results**: AI-powered query responses
        """)

if __name__ == "__main__":
    main()
