import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import PyPDF2
import io
import os
import json
import time
from pathlib import Path
import re
from collections import defaultdict, Counter

# LlamaIndex imports
from llama_index.core import (
    Document, 
    KnowledgeGraphIndex, 
    ServiceContext,
    StorageContext,
    Settings
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)

# Configure Streamlit page
st.set_page_config(
    page_title="Novel Knowledge Graph Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class NovelKnowledgeGraphAnalyzer:
    def __init__(self):
        self.kg_index = None
        self.graph_store = None
        self.documents = []
        self.entities = []
        self.relationships = []
        self.characters = []
        self.themes = []
        self.custom_features = {}
        self.feature_relationships = []
        self.entity_categories = {}
        
    def setup_ollama_models(self, llm_model="llama3.1:8b", embed_model="nomic-embed-text"):
        """Setup Ollama LLM and embedding models"""
        try:
            # Configure LlamaIndex settings
            Settings.llm = Ollama(
                model=llm_model, 
                request_timeout=120.0,
                temperature=0.1
            )
            Settings.embed_model = OllamaEmbedding(
                model_name=embed_model,
                request_timeout=60.0
            )
            
            # Test connection
            test_response = Settings.llm.complete("Hello")
            return True, "Models configured successfully!"
            
        except Exception as e:
            return False, f"Error setting up models: {str(e)}"
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            total_pages = len(pdf_reader.pages)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n\n"
                
                # Update progress
                progress = (i + 1) / total_pages
                progress_bar.progress(progress)
                status_text.text(f"Processing page {i + 1} of {total_pages}")
            
            progress_bar.empty()
            status_text.empty()
            
            return text, None
            
        except Exception as e:
            return None, f"Error extracting PDF: {str(e)}"
    
    def preprocess_novel_text(self, text):
        """Preprocess novel text for better entity extraction"""
        # Clean text
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'\n+', '\n', text)  # Remove extra newlines
        
        # Split into chapters if possible
        chapter_pattern = r'(CHAPTER\s+\w+|Chapter\s+\d+|PART\s+\w+)'
        chapters = re.split(chapter_pattern, text, flags=re.IGNORECASE)
        
        if len(chapters) > 1:
            # Process each chapter as a separate document
            documents = []
            current_chapter = ""
            for i, chunk in enumerate(chapters):
                if re.match(chapter_pattern, chunk, re.IGNORECASE):
                    if current_chapter:
                        documents.append(Document(
                            text=current_chapter,
                            metadata={"chapter": f"Chapter {len(documents) + 1}"}
                        ))
                    current_chapter = chunk + "\n"
                else:
                    current_chapter += chunk
            
            if current_chapter:
                documents.append(Document(
                    text=current_chapter,
                    metadata={"chapter": f"Chapter {len(documents) + 1}"}
                ))
                
            return documents
        else:
            # Split into smaller chunks if no chapters found
            splitter = SentenceSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )
            documents = [Document(text=text)]
            return splitter.get_nodes_from_documents(documents)
    
    def create_knowledge_graph(self, text, chunk_size=1500):
        """Create knowledge graph from novel text"""
        try:
            # Preprocess text into documents
            documents = self.preprocess_novel_text(text)
            
            # Setup graph store
            self.graph_store = SimpleGraphStore()
            storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
            
            # Configure extractors for better novel analysis
            extractors = [
                TitleExtractor(nodes=5),
                QuestionsAnsweredExtractor(questions=3),
                SummaryExtractor(summaries=["prev", "self"]),
                KeywordExtractor(keywords=10),
            ]
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Creating knowledge graph...")
            
            # Create knowledge graph index
            self.kg_index = KnowledgeGraphIndex.from_documents(
                documents,
                storage_context=storage_context,
                max_triplets_per_chunk=10,  # Increase for novels
                include_embeddings=True,
                transformations=extractors,
                show_progress=True,
            )
            
            progress_bar.progress(1.0)
            status_text.text("Knowledge graph created successfully!")
            
            # Extract entities and relationships
            self._extract_graph_data()
            
            return True, "Knowledge graph created successfully!"
            
        except Exception as e:
            return False, f"Error creating knowledge graph: {str(e)}"
    
    def _extract_graph_data(self):
        """Extract entities and relationships from the graph store"""
        try:
            if self.graph_store is None:
                return
            
            # Get all triplets from the graph store
            triplets = self.graph_store.get_triplets()
            
            # Extract unique entities and relationships
            entities_set = set()
            relationships_list = []
            
            for triplet in triplets:
                subject, relation, obj = triplet
                entities_set.add(subject)
                entities_set.add(obj)
                relationships_list.append({
                    'source': subject,
                    'target': obj,
                    'relation': relation
                })
            
            self.entities = list(entities_set)
            self.relationships = relationships_list
            
            # Categorize entities (simple heuristics for novels)
            self._categorize_entities()
            
        except Exception as e:
            st.error(f"Error extracting graph data: {str(e)}")
    
    def _categorize_entities(self):
        """Categorize entities into characters, locations, themes, etc."""
        characters = []
        locations = []
        themes = []
        objects = []
        emotions = []
        events = []
        organizations = []
        other = []
        
        # Enhanced categorization patterns
        character_indicators = ['Mr.', 'Mrs.', 'Dr.', 'Sir', 'Lady', 'Captain', 'Professor', 'Lord', 'Miss']
        location_indicators = ['Castle', 'House', 'City', 'Town', 'Street', 'Road', 'Park', 'Hospital', 'School', 'Church', 'Palace', 'Forest', 'Mountain', 'River', 'Ocean', 'Country', 'Kingdom']
        theme_indicators = ['love', 'death', 'war', 'peace', 'friendship', 'betrayal', 'revenge', 'justice', 'honor', 'sacrifice', 'redemption', 'power', 'corruption', 'innocence', 'guilt']
        emotion_indicators = ['anger', 'joy', 'fear', 'sadness', 'surprise', 'disgust', 'anxiety', 'happiness', 'melancholy', 'despair', 'hope', 'hatred', 'compassion']
        object_indicators = ['sword', 'ring', 'crown', 'book', 'letter', 'treasure', 'weapon', 'artifact', 'jewel', 'mirror', 'portrait']
        event_indicators = ['battle', 'wedding', 'funeral', 'celebration', 'meeting', 'journey', 'quest', 'ceremony', 'trial', 'escape']
        organization_indicators = ['army', 'guild', 'council', 'court', 'church', 'society', 'brotherhood', 'sisterhood', 'order', 'clan']
        
        for entity in self.entities:
            entity_lower = entity.lower()
            categorized = False
            
            # Check against custom feature definitions first
            for custom_category, keywords in self.custom_features.items():
                if any(keyword.lower() in entity_lower for keyword in keywords):
                    if custom_category not in self.entity_categories:
                        self.entity_categories[custom_category] = []
                    self.entity_categories[custom_category].append(entity)
                    categorized = True
                    break
            
            if categorized:
                continue
            
            # Standard categorization
            if (any(indicator in entity for indicator in character_indicators) or
                (entity[0].isupper() and len(entity.split()) <= 3 and 
                 not any(indicator.lower() in entity_lower for indicator in location_indicators + object_indicators))):
                characters.append(entity)
                self.entity_categories['Characters'] = characters
            
            elif any(indicator.lower() in entity_lower for indicator in location_indicators):
                locations.append(entity)
                self.entity_categories['Locations'] = locations
                
            elif any(theme in entity_lower for theme in theme_indicators):
                themes.append(entity)
                self.entity_categories['Themes'] = themes
                
            elif any(emotion in entity_lower for emotion in emotion_indicators):
                emotions.append(entity)
                self.entity_categories['Emotions'] = emotions
                
            elif any(obj in entity_lower for obj in object_indicators):
                objects.append(entity)
                self.entity_categories['Objects'] = objects
                
            elif any(event in entity_lower for event in event_indicators):
                events.append(entity)
                self.entity_categories['Events'] = events
                
            elif any(org in entity_lower for org in organization_indicators):
                organizations.append(entity)
                self.entity_categories['Organizations'] = organizations
                
            else:
                other.append(entity)
                self.entity_categories['Other'] = other
        
        # Update legacy attributes for compatibility
        self.characters = characters
        self.locations = locations
        self.themes = themes
        self.objects = objects
        self.emotions = emotions
        self.events = events
        self.organizations = organizations
        self.other_entities = other
    
    def add_custom_feature(self, feature_name, keywords):
        """Add a custom feature category with associated keywords"""
        self.custom_features[feature_name] = keywords
        
        # Re-categorize entities if they exist
        if self.entities:
            self._categorize_entities()
    
    def get_feature_relationships(self, feature1_name, feature2_name):
        """Get relationships between two specific feature categories"""
        if feature1_name not in self.entity_categories or feature2_name not in self.entity_categories:
            return []
        
        feature1_entities = set(self.entity_categories[feature1_name])
        feature2_entities = set(self.entity_categories[feature2_name])
        
        cross_relationships = []
        for rel in self.relationships:
            if ((rel['source'] in feature1_entities and rel['target'] in feature2_entities) or
                (rel['source'] in feature2_entities and rel['target'] in feature1_entities)):
                cross_relationships.append(rel)
        
        return cross_relationships
    
    def analyze_feature_interactions(self):
        """Analyze interactions between different feature categories"""
        feature_interactions = {}
        categories = list(self.entity_categories.keys())
        
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if i <= j:  # Avoid duplicate pairs
                    relationships = self.get_feature_relationships(cat1, cat2)
                    if relationships:
                        key = f"{cat1} ↔ {cat2}" if cat1 != cat2 else f"{cat1} (Internal)"
                        feature_interactions[key] = len(relationships)
        
        return feature_interactions
    
    def create_feature_relationship_matrix(self):
        """Create a matrix showing relationships between different feature types"""
        if not self.entity_categories:
            return None
        
        categories = list(self.entity_categories.keys())
        matrix_data = []
        
        for cat1 in categories:
            row = []
            for cat2 in categories:
                relationships = self.get_feature_relationships(cat1, cat2)
                row.append(len(relationships))
            matrix_data.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=categories,
            y=categories,
            colorscale='Viridis',
            showscale=True,
            hoverongaps=False,
            text=matrix_data,
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        
        fig.update_layout(
            title='Feature Relationship Matrix',
            xaxis_title='Target Features',
            yaxis_title='Source Features',
            width=600,
            height=600
        )
        
        return fig
    
    def create_custom_feature_network(self, selected_features, max_relationships=100):
        """Create network visualization for specific feature categories"""
        if not selected_features or not self.entity_categories:
            return None
        
        # Get entities from selected features
        selected_entities = set()
        for feature in selected_features:
            if feature in self.entity_categories:
                selected_entities.update(self.entity_categories[feature])
        
        if not selected_entities:
            return None
        
        # Filter relationships to only include selected entities
        filtered_relationships = [
            rel for rel in self.relationships[:max_relationships]
            if rel['source'] in selected_entities and rel['target'] in selected_entities
        ]
        
        if not filtered_relationships:
            return None
        
        # Create NetworkX graph
        G = nx.Graph()
        for rel in filtered_relationships:
            G.add_edge(rel['source'], rel['target'], relation=rel['relation'])
        
        if len(G.nodes()) == 0:
            return None
        
        # Calculate layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Color mapping for selected features
        colors = px.colors.qualitative.Set3
        feature_colors = {feature: colors[i % len(colors)] for i, feature in enumerate(selected_features)}
        
        # Prepare traces
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_traces = []
        
        # Create separate trace for each feature type
        for feature in selected_features:
            if feature not in self.entity_categories:
                continue
                
            feature_entities = [e for e in self.entity_categories[feature] if e in G.nodes()]
            if not feature_entities:
                continue
            
            node_x = [pos[node][0] for node in feature_entities]
            node_y = [pos[node][1] for node in feature_entities]
            node_text = []
            node_sizes = []
            
            for node in feature_entities:
                degree = G.degree(node)
                connections = list(G.neighbors(node))
                node_text.append(f"<b>{node}</b><br>"
                               f"Category: {feature}<br>"
                               f"Connections: {degree}<br>"
                               f"Connected to: {', '.join(connections[:3])}{'...' if len(connections) > 3 else ''}")
                node_sizes.append(max(15, min(50, degree * 5)))
            
            node_traces.append(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                name=feature,
                marker=dict(
                    size=node_sizes,
                    color=feature_colors[feature],
                    line=dict(width=2)
                )
            ))
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add node traces
        for trace in node_traces:
            fig.add_trace(trace)
        
        fig.update_layout(
            title=f'Relationships Between: {", ".join(selected_features)}',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600
        )
        
        return fig
    
    def query_feature_relationships(self, feature_type, query_text):
        """Query relationships for a specific feature type"""
        if self.kg_index is None or feature_type not in self.entity_categories:
            return "Please create a knowledge graph first or check the feature type."
        
        # Get entities of the specified feature type
        entities = self.entity_categories[feature_type]
        entity_list = ", ".join(entities[:10])  # Limit for context
        
        # Create targeted query
        enhanced_query = f"""
        Focus on {feature_type.lower()} in the novel, specifically: {entity_list}
        
        Original question: {query_text}
        
        Please provide detailed analysis about how these {feature_type.lower()} relate to other elements in the story.
        """
        
        try:
            query_engine = self.kg_index.as_query_engine(
                include_text=True,
                response_mode="tree_summarize",
                embedding_mode="hybrid",
                similarity_top_k=8,
            )
            
            response = query_engine.query(enhanced_query)
            return str(response)
            
        except Exception as e:
            return f"Error querying feature relationships: {str(e)}"
    
    def query_knowledge_graph(self, query, mode="tree_summarize"):
        """Query the knowledge graph"""
        if self.kg_index is None:
            return "Please create a knowledge graph first."
        
        try:
            query_engine = self.kg_index.as_query_engine(
                include_text=True,
                response_mode=mode,
                embedding_mode="hybrid",
                similarity_top_k=5,
            )
            
            response = query_engine.query(query)
            return str(response)
            
        except Exception as e:
            return f"Error querying graph: {str(e)}"
    
    def create_network_visualization(self, max_nodes=50):
        """Create interactive network visualization of the knowledge graph"""
        if not self.relationships:
            return None
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes and edges (limit for visualization)
        edge_count = min(len(self.relationships), max_nodes * 2)
        for rel in self.relationships[:edge_count]:
            G.add_edge(rel['source'], rel['target'], relation=rel['relation'])
        
        if len(G.nodes()) == 0:
            return None
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Get relationship info
            relation = G[edge[0]][edge[1]].get('relation', 'related to')
            edge_info.append(f"{edge[0]} → {relation} → {edge[1]}")
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        # Color mapping for different entity types
        color_map = {
            'character': '#FF6B6B',
            'location': '#4ECDC4', 
            'theme': '#45B7D1',
            'other': '#95A5A6'
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Determine node type and color
            if node in self.characters:
                node_type = 'character'
                color = color_map['character']
            elif node in getattr(self, 'locations', []):
                node_type = 'location' 
                color = color_map['location']
            elif node in getattr(self, 'themes', []):
                node_type = 'theme'
                color = color_map['theme']
            else:
                node_type = 'other'
                color = color_map['other']
            
            node_colors.append(color)
            
            # Node size based on degree
            degree = G.degree(node)
            node_sizes.append(max(10, min(50, degree * 5)))
            
            # Node hover text
            connections = list(G.neighbors(node))
            node_text.append(f"<b>{node}</b><br>"
                           f"Type: {node_type}<br>"
                           f"Connections: {degree}<br>"
                           f"Connected to: {', '.join(connections[:5])}{'...' if len(connections) > 5 else ''}")
        
        # Create figure
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
            text=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    xanchor="left",
                    title="Entity Types"
                )
            ),
            showlegend=False
        ))
        
        fig.update_layout(
            title='Novel Knowledge Graph - Character and Entity Relationships',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Interactive Knowledge Graph - Hover over nodes for details",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='#888', size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600
        )
        
        return fig
    
    def create_entity_analytics(self):
        """Create analytics charts for entities"""
        if not self.entities:
            return None, None, None, None
        
        # Entity type distribution (now includes custom features)
        entity_counts = {}
        for category, entities in self.entity_categories.items():
            if entities:  # Only include non-empty categories
                entity_counts[category] = len(entities)
        
        if entity_counts:
            fig_pie = px.pie(
                values=list(entity_counts.values()),
                names=list(entity_counts.keys()),
                title="Entity Type Distribution (Including Custom Features)"
            )
        else:
            fig_pie = None
        
        # Relationship frequency
        if self.relationships:
            relations = [rel['relation'] for rel in self.relationships]
            relation_counts = Counter(relations)
            
            fig_relations = px.bar(
                x=list(relation_counts.keys())[:15],
                y=list(relation_counts.values())[:15],
                title="Top 15 Relationship Types",
                labels={'x': 'Relationship Type', 'y': 'Frequency'}
            )
            fig_relations.update_layout(xaxis_tickangle=-45)
        else:
            fig_relations = None
        
        # Most connected entities across all categories
        if self.relationships:
            entity_connections = defaultdict(int)
            for rel in self.relationships:
                entity_connections[rel['source']] += 1
                entity_connections[rel['target']] += 1
            
            if entity_connections:
                top_entities = dict(sorted(entity_connections.items(), 
                                         key=lambda x: x[1], reverse=True)[:15])
                
                # Add category information
                entity_categories_for_chart = []
                for entity in top_entities.keys():
                    category = "Unknown"
                    for cat, entities in self.entity_categories.items():
                        if entity in entities:
                            category = cat
                            break
                    entity_categories_for_chart.append(category)
                
                fig_top_entities = go.Figure()
                fig_top_entities.add_trace(go.Bar(
                    x=list(top_entities.values()),
                    y=list(top_entities.keys()),
                    orientation='h',
                    text=entity_categories_for_chart,
                    textposition='auto',
                    marker_color=px.colors.qualitative.Set3[:len(top_entities)]
                ))
                
                fig_top_entities.update_layout(
                    title="Most Connected Entities (All Categories)",
                    xaxis_title="Number of Connections",
                    yaxis_title="Entity",
                    height=500
                )
            else:
                fig_top_entities = None
        else:
            fig_top_entities = None
        
        # Feature interaction analysis
        feature_interactions = self.analyze_feature_interactions()
        if feature_interactions:
            fig_interactions = px.bar(
                x=list(feature_interactions.values()),
                y=list(feature_interactions.keys()),
                orientation='h',
                title="Feature Category Interactions",
                labels={'x': 'Number of Relationships', 'y': 'Feature Interaction'}
            )
            fig_interactions.update_layout(height=400)
        else:
            fig_interactions = None
        
        return fig_pie, fig_relations, fig_top_entities, fig_interactions

def main():
    st.title("📚 Novel Knowledge Graph Analyzer")
    st.markdown("Extract characters, relationships, and themes from your novel using AI-powered knowledge graphs")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = NovelKnowledgeGraphAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    llm_model = st.sidebar.selectbox(
        "LLM Model", 
        ["llama3.1:8b", "llama3.1:70b", "mistral:7b", "phi3:medium"],
        help="Make sure the model is installed in Ollama"
    )
    
    embed_model = st.sidebar.selectbox(
        "Embedding Model",
        ["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
        help="Embedding model for semantic understanding"
    )
    
    # Setup models button
    if st.sidebar.button("Setup Models"):
        with st.spinner("Setting up Ollama models..."):
            success, message = analyzer.setup_ollama_models(llm_model, embed_model)
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📖 Upload & Process", 
        "🎯 Custom Features",
        "🕸️ Knowledge Graph", 
        "🔍 Query & Analysis", 
        "🔗 Feature Relationships",
        "📊 Analytics", 
        "👥 Characters & Entities"
    ])
    
    with tab1:
        st.header("Upload and Process Your Novel")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your novel (PDF format)",
            type=['pdf'],
            help="Upload a PDF file of your novel to analyze"
        )
        
        if uploaded_file is not None:
            st.success(f"Uploaded: {uploaded_file.name}")
            
            # Extract text button
            if st.button("Extract Text from PDF", type="primary"):
                with st.spinner("Extracting text from PDF..."):
                    text, error = analyzer.extract_text_from_pdf(uploaded_file)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Extracted {len(text)} characters from PDF")
                        
                        # Show preview
                        with st.expander("Text Preview"):
                            st.text_area("First 1000 characters:", text[:1000], height=200)
                        
                        # Store text in session state
                        st.session_state.novel_text = text
            
            # Create knowledge graph button
            if hasattr(st.session_state, 'novel_text'):
                if st.button("Create Knowledge Graph", type="primary"):
                    with st.spinner("Creating knowledge graph... This may take several minutes."):
                        success, message = analyzer.create_knowledge_graph(st.session_state.novel_text)
                        
                        if success:
                            st.success(message)
                            st.balloons()
                            
                            # Show basic stats
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Entities", len(analyzer.entities))
                            with col2:
                                st.metric("Total Relationships", len(analyzer.relationships))
                            with col3:
                                st.metric("Feature Categories", len(analyzer.entity_categories))
                            with col4:
                                st.metric("Characters Found", len(analyzer.characters))
                        else:
                            st.error(message)
    
    with tab2:
        st.header("🎯 Custom Feature Definition")
        st.markdown("Define your own custom features to extract from the novel (e.g., magical objects, political themes, etc.)")
        
        # Custom feature creation interface
        st.subheader("Add New Custom Feature")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            new_feature_name = st.text_input("Feature Name", placeholder="e.g., Magical Objects")
        
        with col2:
            new_feature_keywords = st.text_area(
                "Keywords (comma-separated)", 
                placeholder="e.g., wand, spell, potion, magic, enchantment",
                help="Enter keywords that help identify this feature type"
            )
        
        if st.button("Add Custom Feature") and new_feature_name and new_feature_keywords:
            keywords = [kw.strip() for kw in new_feature_keywords.split(',') if kw.strip()]
            analyzer.add_custom_feature(new_feature_name, keywords)
            st.success(f"Added custom feature: {new_feature_name}")
            st.rerun()
        
        # Display existing custom features
        if analyzer.custom_features:
            st.subheader("Current Custom Features")
            
            for feature_name, keywords in analyzer.custom_features.items():
                with st.expander(f"📋 {feature_name}"):
                    st.write(f"**Keywords:** {', '.join(keywords)}")
                    
                    # Show entities found for this feature
                    if feature_name in analyzer.entity_categories:
                        entities = analyzer.entity_categories[feature_name]
                        st.write(f"**Found Entities ({len(entities)}):** {', '.join(entities[:10])}")
                        if len(entities) > 10:
                            st.write(f"... and {len(entities) - 10} more")
                    else:
                        st.write("*No entities found yet. Create knowledge graph to see results.*")
                    
                    # Remove feature button
                    if st.button(f"Remove {feature_name}", key=f"remove_{feature_name}"):
                        del analyzer.custom_features[feature_name]
                        if feature_name in analyzer.entity_categories:
                            del analyzer.entity_categories[feature_name]
                        st.success(f"Removed {feature_name}")
                        st.rerun()
        
        # Predefined feature templates
        st.subheader("📚 Predefined Feature Templates")
        st.markdown("Quick-add common literary features:")
        
        templates = {
            "Magical Elements": ["magic", "spell", "wand", "potion", "curse", "enchantment", "wizard", "witch"],
            "Political Themes": ["power", "government", "politics", "revolution", "democracy", "tyranny", "freedom", "oppression"],
            "Religious Elements": ["god", "prayer", "faith", "church", "temple", "divine", "sacred", "holy"],
            "Scientific Concepts": ["science", "experiment", "discovery", "technology", "invention", "research", "laboratory"],
            "Psychological States": ["memory", "dream", "nightmare", "consciousness", "madness", "sanity", "identity", "soul"],
            "Social Classes": ["nobility", "peasant", "aristocrat", "commoner", "wealthy", "poor", "elite", "working class"],
            "Natural Elements": ["forest", "mountain", "river", "ocean", "storm", "fire", "earth", "wind", "water"]
        }
        
        cols = st.columns(3)
        for i, (template_name, keywords) in enumerate(templates.items()):
            with cols[i % 3]:
                if st.button(f"Add {template_name}", key=f"template_{template_name}"):
                    analyzer.add_custom_feature(template_name, keywords)
                    st.success(f"Added {template_name}")
                    st.rerun()
    
    with tab3:
        st.header("Interactive Knowledge Graph")
        
        if analyzer.kg_index is not None:
            # Graph controls
            col1, col2 = st.columns([3, 1])
            
            with col2:
                max_nodes = st.slider("Max Nodes to Display", 20, 100, 50)
                
                # Feature filter
                st.subheader("Filter by Features")
                available_features = list(analyzer.entity_categories.keys())
                selected_features = st.multiselect(
                    "Show only these features:",
                    available_features,
                    default=available_features[:4] if len(available_features) > 4 else available_features
                )
                
            with col1:
                # Choose between full graph or feature-specific graph
                if selected_features and len(selected_features) < len(available_features):
                    st.info(f"Showing relationships between: {', '.join(selected_features)}")
                    network_fig = analyzer.create_custom_feature_network(selected_features, max_nodes)
                else:
                    # Create and display full network graph
                    network_fig = analyzer.create_network_visualization(max_nodes)
                
                if network_fig:
                    st.plotly_chart(network_fig, use_container_width=True)
                else:
                    st.warning("No relationships found to visualize for selected features")
        else:
            st.info("Please upload a novel and create a knowledge graph first.")
    
    with tab4:
        st.header("Query and Analysis")
        
        if analyzer.kg_index is not None:
            # Query interface with feature-specific options
            st.subheader("Ask Questions About Your Novel")
            
            # Feature-specific query option
            col1, col2 = st.columns([1, 2])
            
            with col1:
                query_mode = st.radio("Query Mode", ["General Query", "Feature-Specific Query"])
                
                if query_mode == "Feature-Specific Query":
                    available_features = list(analyzer.entity_categories.keys())
                    selected_feature = st.selectbox("Focus on Feature:", available_features)
            
            with col2:
                # Predefined queries
                if query_mode == "General Query":
                    query_examples = [
                        "Who are the main characters in this novel?",
                        "What are the key relationships between characters?",
                        "What are the main themes explored in this story?",
                        "Describe the setting and locations in the novel",
                        "What conflicts occur in the story?",
                        "How do the characters develop throughout the novel?"
                    ]
                else:
                    query_examples = [
                        f"How do {selected_feature if 'selected_feature' in locals() else 'these features'} influence the plot?",
                        f"What relationships exist between {selected_feature if 'selected_feature' in locals() else 'these features'} and characters?",
                        f"How are {selected_feature if 'selected_feature' in locals() else 'these features'} described in the novel?",
                        f"What role do {selected_feature if 'selected_feature' in locals() else 'these features'} play in conflicts?",
                        f"How do {selected_feature if 'selected_feature' in locals() else 'these features'} change throughout the story?"
                    ]
                
                selected_query = st.selectbox("Choose a predefined query:", ["Custom query"] + query_examples)
                
                if selected_query == "Custom query":
                    query = st.text_area("Enter your custom question about the novel:")
                else:
                    query = selected_query
                    st.text_area("Selected query:", value=query, disabled=True)
            
            response_mode = st.selectbox(
                "Response Mode", 
                ["tree_summarize", "compact", "simple_summarize"],
                help="How to generate the response"
            )
            
            if st.button("Query Knowledge Graph") and query:
                with st.spinner("Analyzing your question..."):
                    if query_mode == "Feature-Specific Query" and 'selected_feature' in locals():
                        response = analyzer.query_feature_relationships(selected_feature, query)
                    else:
                        response = analyzer.query_knowledge_graph(query, response_mode)
                    
                    st.subheader("Answer:")
                    st.write(response)
        else:
            st.info("Please create a knowledge graph first to enable querying.")
    
    with tab5:
        st.header("🔗 Feature Relationships Analysis")
        
        if analyzer.entity_categories:
            # Feature relationship matrix
            st.subheader("Feature Interaction Heatmap")
            st.markdown("This heatmap shows how many relationships exist between different feature categories.")
            
            matrix_fig = analyzer.create_feature_relationship_matrix()
            if matrix_fig:
                st.plotly_chart(matrix_fig, use_container_width=True)
            
            # Pairwise feature analysis
            st.subheader("Pairwise Feature Analysis")
            
            available_features = list(analyzer.entity_categories.keys())
            
            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.selectbox("Feature Category 1:", available_features, key="feat1")
            with col2:
                feature2 = st.selectbox("Feature Category 2:", available_features, key="feat2")
            
            if st.button("Analyze Relationship"):
                relationships = analyzer.get_feature_relationships(feature1, feature2)
                
                st.subheader(f"Relationships between {feature1} and {feature2}")
                
                if relationships:
                    st.success(f"Found {len(relationships)} relationships")
                    
                    # Display relationships in a table
                    rel_df = pd.DataFrame(relationships)
                    st.dataframe(rel_df, use_container_width=True)
                    
                    # Relationship summary
                    relation_types = Counter([rel['relation'] for rel in relationships])
                    
                    if len(relation_types) > 1:
                        fig_rel_types = px.bar(
                            x=list(relation_types.keys()),
                            y=list(relation_types.values()),
                            title=f"Relationship Types between {feature1} and {feature2}",
                            labels={'x': 'Relationship Type', 'y': 'Frequency'}
                        )
                        st.plotly_chart(fig_rel_types, use_container_width=True)
                    
                    # Show specific examples
                    st.subheader("Relationship Examples:")
                    for i, rel in enumerate(relationships[:5]):
                        st.write(f"**{i+1}.** {rel['source']} *{rel['relation']}* {rel['target']}")
                    
                    if len(relationships) > 5:
                        st.write(f"... and {len(relationships) - 5} more relationships")
                
                else:
                    st.warning(f"No direct relationships found between {feature1} and {feature2}")
            
            # Cross-feature entity network
            st.subheader("Cross-Feature Network Visualization")
            
            multi_features = st.multiselect(
                "Select multiple features to visualize their relationships:",
                available_features,
                default=available_features[:3] if len(available_features) >= 3 else available_features
            )
            
            if multi_features and len(multi_features) >= 2:
                max_rels = st.slider("Max Relationships to Show", 20, 200, 100, key="cross_feature_rels")
                
                cross_network_fig = analyzer.create_custom_feature_network(multi_features, max_rels)
                if cross_network_fig:
                    st.plotly_chart(cross_network_fig, use_container_width=True)
                else:
                    st.warning("No relationships found between selected features")
            
        else:
            st.info("Please create a knowledge graph first to analyze feature relationships.")
    
    with tab6:
        st.header("Novel Analytics Dashboard")
        
        if analyzer.entities:
            # Create analytics
            fig_pie, fig_relations, fig_top_entities, fig_interactions = analyzer.create_entity_analytics()
            
            # Display charts
            col1, col2 = st.columns(2)
            
            with col1:
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)
                
            with col2:
                if fig_relations:
                    st.plotly_chart(fig_relations, use_container_width=True)
            
            if fig_top_entities:
                st.plotly_chart(fig_top_entities, use_container_width=True)
            
            if fig_interactions:
                st.plotly_chart(fig_interactions, use_container_width=True)
            
            # Summary statistics
            st.subheader("Novel Statistics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Entities", len(analyzer.entities))
            with col2:
                st.metric("Feature Categories", len(analyzer.entity_categories))
            with col3:
                st.metric("Custom Features", len(analyzer.custom_features))
            with col4:
                st.metric("Relationships", len(analyzer.relationships))
            with col5:
                total_interactions = sum(analyzer.analyze_feature_interactions().values())
                st.metric("Feature Interactions", total_interactions)
        else:
            st.info("Please create a knowledge graph first to see analytics.")
    
    with tab7:
        st.header("Characters and Entities by Category")
        
        if analyzer.entity_categories:
            # Display all entity categories including custom ones
            for category, entities in analyzer.entity_categories.items():
                if entities:  # Only show non-empty categories
                    st.subheader(f"📋 {category} ({len(entities)} items)")
                    
                    # Show entities in expandable section
                    with st.expander(f"View all {category.lower()}", expanded=len(entities) <= 10):
                        # Create dataframe
                        entity_df = pd.DataFrame({'Entity': entities})
                        
                        # Add connection counts if relationships exist
                        if analyzer.relationships:
                            connections = []
                            for entity in entities:
                                count = sum(1 for rel in analyzer.relationships 
                                          if rel['source'] == entity or rel['target'] == entity)
                                connections.append(count)
                            entity_df['Connections'] = connections
                            
                            # Sort by connections
                            entity_df = entity_df.sort_values('Connections', ascending=False)
                        
                        st.dataframe(entity_df, use_container_width=True)
                        
                        # Download button for this category
                        csv = entity_df.to_csv(index=False)
                        st.download_button(
                            label=f"Download {category} as CSV",
                            data=csv,
                            file_name=f"novel_{category.lower().replace(' ', '_')}.csv",
                            mime="text/csv",
                            key=f"download_{category}"
                        )
            
            # Relationships table
            st.subheader("🔗 All Relationships")
            if analyzer.relationships:
                # Enhanced relationships table with categories
                enhanced_relationships = []
                for rel in analyzer.relationships:
                    # Find categories for source and target
                    source_category = "Unknown"
                    target_category = "Unknown"
                    
                    for category, entities in analyzer.entity_categories.items():
                        if rel['source'] in entities:
                            source_category = category
                        if rel['target'] in entities:
                            target_category = category
                    
                    enhanced_relationships.append({
                        'Source': rel['source'],
                        'Source Category': source_category,
                        'Relation': rel['relation'],
                        'Target': rel['target'],
                        'Target Category': target_category
                    })
                
                rel_df = pd.DataFrame(enhanced_relationships)
                
                # Filter options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    source_filter = st.multiselect(
                        "Filter by Source Category",
                        options=list(analyzer.entity_categories.keys()),
                        default=list(analyzer.entity_categories.keys())
                    )
                
                with col2:
                    target_filter = st.multiselect(
                        "Filter by Target Category", 
                        options=list(analyzer.entity_categories.keys()),
                        default=list(analyzer.entity_categories.keys())
                    )
                
                with col3:
                    relation_filter = st.multiselect(
                        "Filter by Relation Type",
                        options=sorted(rel_df['Relation'].unique()),
                        default=sorted(rel_df['Relation'].unique())[:10]  # Show top 10 by default
                    )
                
                # Apply filters
                filtered_rel_df = rel_df[
                    (rel_df['Source Category'].isin(source_filter)) &
                    (rel_df['Target Category'].isin(target_filter)) &
                    (rel_df['Relation'].isin(relation_filter))
                ]
                
                st.dataframe(filtered_rel_df, use_container_width=True)
                
                # Download button for relationships
                csv = filtered_rel_df.to_csv(index=False)
                st.download_button(
                    label="Download Filtered Relationships as CSV",
                    data=csv,
                    file_name="novel_relationships_enhanced.csv",
                    mime="text/csv"
                )
                
                # Relationship statistics
                st.subheader("📊 Relationship Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Relationships", len(filtered_rel_df))
                
                with col2:
                    unique_relations = len(filtered_rel_df['Relation'].unique())
                    st.metric("Unique Relation Types", unique_relations)
                
                with col3:
                    cross_category = len(filtered_rel_df[
                        filtered_rel_df['Source Category'] != filtered_rel_df['Target Category']
                    ])
                    st.metric("Cross-Category Relations", cross_category)
            else:
                st.info("No relationships found yet.")
        else:
            st.info("Please create a knowledge graph first to see entities and categories.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **📋 Setup Requirements:**
    
    1. **Install Ollama:** `curl -fsSL https://ollama.com/install.sh | sh`
    2. **Pull Models:** `ollama pull llama3.1:8b` and `ollama pull nomic-embed-text`
    3. **Install Python Packages:** `pip install streamlit llama-index llama-index-llms-ollama llama-index-embeddings-ollama PyPDF2 plotly networkx pandas`
    
    **🎯 Enhanced Features:**
    - ✅ Custom feature definition and extraction
    - ✅ Cross-feature relationship analysis  
    - ✅ Interactive feature filtering
    - ✅ Enhanced entity categorization
    - ✅ Feature interaction heatmaps
    - ✅ Downloadable analysis results
    """)

if __name__ == "__main__":
    main()

