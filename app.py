import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import gzip
import pickle
from ast import literal_eval
from utils.helpers import DataVisualizer, TextProcessor, ResultsFormatter, Config
from utils.search import SearchEngine


class TopicVisualizationApp:
    def __init__(self):
        self.load_data()
        self.search_engine = SearchEngine()
        
    def initialize_search_engine(self):
        """Initialize the search engine"""
        self.search_engine = SearchEngine()
        
    def load_data(self):
        """Load and prepare all necessary data"""
        try:
            self.df = pd.read_csv('data/processed/processed_articles.csv.gz')
                
            print("Available columns:", self.df.columns)
                
            required_columns = ['title', 'summary', 'date', 'entities', 'themes', 'topic', 'topic_name']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                st.error(f"Colonnes manquantes : {', '.join(missing_columns)}")
                return
                
            for col in ['entities', 'key_phrases', 'main_subjects', 'themes']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].apply(literal_eval)
                
            if 'sentiment' in self.df.columns:
                try:
                    self.df['sentiment'] = self.df['sentiment'].apply(literal_eval)
                except:
                    pass
                
            if 'date' in self.df.columns:
                # Convertir en datetime et forcer en UTC
                self.df['date'] = pd.to_datetime(self.df['date']).dt.tz_convert('UTC')
                self.df = self.df.sort_values('date', ascending=False)
                
            try:
                with gzip.open('data/processed/semantic_index.pkl.gz', 'rb') as f:
                    self.semantic_index = pickle.load(f)
            except Exception as e:
                print(f"Error loading semantic index: {str(e)}")
                self.semantic_index = None
                
            try:
                with gzip.open('data/processed/topic_model.pkl.gz', 'rb') as f:
                    self.model = pickle.load(f)
            except Exception as e:
                print(f"Error loading topic model: {str(e)}")
                self.model = None
                
            if 'topic' in self.df.columns:
                self.df['topic'] = self.df['topic'].astype(int)
                
            if 'cluster' in self.df.columns:
                self.df['cluster'] = self.df['cluster'].astype(int)
                
            if 'cluster_summary' in self.df.columns:
                try:
                    self.df['cluster_summary'] = self.df['cluster_summary'].apply(literal_eval)
                except:
                    pass
                    
            if 'topic_summary' in self.df.columns:
                try:
                    self.df['topic_summary'] = self.df['topic_summary'].apply(literal_eval)
                except:
                    pass
                
            print("Data loading completed successfully")
            print(f"Total number of articles: {len(self.df)}")
            print(f"Period covered: from {self.df['date'].min()} to {self.df['date'].max()}")
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            print(f"Detailed error: {str(e)}")
            raise
    
    def run(self):
        """Streamlit application entry point"""
        st.set_page_config(
            page_title=Config.APP_TITLE,
            page_icon="📰",
            layout="wide"
        )
        
        st.markdown("""
            <style>
            [data-testid="stSidebarNav"] img {
                filter: invert(1);
            }
            .stAlert {
                padding: 1rem;
                margin-bottom: 1rem;
                border-radius: 0.5rem;
            }
            .article-title {
                font-size: 1.2rem;
                font-weight: bold;
                margin-bottom: 0.5rem;
            }
            .metadata {
                font-size: 0.9rem;
                color: #666;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.title(Config.APP_TITLE)
        
        # Sidebar
        st.sidebar.image(Config.LOGO_URL, use_column_width=True)
        
        page = st.sidebar.selectbox(
            'Navigation',
            ['Vue d\'ensemble', 'Explorer les Topics', 'Analyse des Clusters', 'Recherche Avancée']
        )
        
        if page == 'Vue d\'ensemble':
            self.show_overview()
        elif page == 'Explorer les Topics':
            self.explore_topics()
        elif page == 'Analyse des Clusters':
            self.analyze_clusters()
        else:
            self.advanced_search()
    
    def show_overview(self):
        """Enhanced overview page"""
        st.header('Vue d\'ensemble des Articles')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Articles", len(self.df), "total")
        with col2:
            st.metric("Topics", len(self.df['topic'].unique()))
        with col3:
            st.metric("Clusters", len(self.df['cluster'].unique()))
        with col4:
            if 'date' in self.df.columns:
                days_covered = (self.df['date'].max() - self.df['date'].min()).days
                st.metric("Période", f"{days_covered} jours")
        
        if 'date' in self.df.columns:
            st.subheader('Distribution temporelle')
            fig_timeline = DataVisualizer.create_timeline_plot(self.df)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.subheader('Distribution des Topics')
        fig_topics = DataVisualizer.create_topic_distribution(self.df)
        st.plotly_chart(fig_topics, use_container_width=True)
        
        if 'sentiment' in self.df.columns:
            st.subheader('Analyse des Sentiments')
            if isinstance(self.df['sentiment'][0], dict):
                sentiments = pd.DataFrame([
                    {'sentiment': s.get('label', 'unknown'), 'count': 1}
                    for s in self.df['sentiment']
                ])
            else:
                sentiments = pd.Series(self.df['sentiment']).value_counts().reset_index()
                sentiments.columns = ['sentiment', 'count']
            
            fig_sentiments = px.pie(
                sentiments, 
                names='sentiment',
                title='Distribution des Sentiments',
                color='sentiment',
                color_discrete_map={
                    'positive': 'green',
                    'negative': 'red',
                    'neutral': 'gray',
                    'unknown': 'gray'
                }
            )
            st.plotly_chart(fig_sentiments)
        
        st.subheader('Entités Principales')
        col1, col2 = st.columns(2)
        
        with col1:
            entities_po = [
                ent['text'] for ents in self.df['entities']
                for ent in ents if ent['type'] in ['PERSON', 'ORGANIZATION']
            ]
            top_po = Counter(entities_po).most_common(10)
            fig_po = px.bar(
                x=[e[0] for e in top_po],
                y=[e[1] for e in top_po],
                title='Top Personnes et Organisations'
            )
            st.plotly_chart(fig_po)
        
        with col2:
            locations = [
                ent['text'] for ents in self.df['entities']
                for ent in ents if ent['type'] == 'LOCATION'
            ]
            top_locations = Counter(locations).most_common(10)
            fig_locations = px.bar(
                x=[e[0] for e in top_locations],
                y=[e[1] for e in top_locations],
                title='Top Lieux Mentionnés'
            )
            st.plotly_chart(fig_locations)
        
    def explore_topics(self):
        """Enhanced topic exploration page"""
        st.header('Explorer les Topics')
        
        col1, col2 = st.columns([2, 1])
        with col1:
            topic_options = sorted(self.df['topic_name'].unique())
            selected_topic_name = st.selectbox('Sélectionnez un topic', topic_options)
        
        with col2:
            time_range = st.selectbox(
                'Période',
                list(Config.TIME_RANGES.keys()),
                index=1
            )
            days_filter = Config.TIME_RANGES[time_range]
        
        topic_id = self.df[self.df['topic_name'] == selected_topic_name]['topic'].iloc[0]
        topic_articles = self.df[self.df['topic'] == topic_id].copy()
        
        if days_filter:
            cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days_filter)
            topic_articles = topic_articles[topic_articles['date'] > cutoff_date]
        
        st.subheader(f'Analyse du Topic: {selected_topic_name}')
        
        col1, col2 = st.columns([2, 1])
        with col1:
            topic_words = self.model.get_topic(topic_id)
            words, weights = zip(*topic_words[:10])
            fig_words = px.bar(
                x=list(words),
                y=list(weights),
                labels={'x': 'Mots', 'y': 'Importance'},
                title='Mots-clés du Topic'
            )
            st.plotly_chart(fig_words)
        
        with col2:
            st.metric("Nombre d'articles", len(topic_articles))
            if 'sentiment' in topic_articles.columns and not topic_articles.empty:
                try:
                    if isinstance(topic_articles['sentiment'].iloc[0], dict):
                        sentiments = pd.DataFrame([
                            {'sentiment': s.get('label', 'unknown'), 'count': 1}
                            for s in topic_articles['sentiment'] if isinstance(s, dict)
                        ])
                        if not sentiments.empty:
                            dominant_sentiment = sentiments['sentiment'].mode().iloc[0]
                            st.metric("Sentiment dominant", dominant_sentiment)
                    else:
                        sentiments = topic_articles['sentiment'].value_counts()
                        if not sentiments.empty:
                            dominant_sentiment = sentiments.index[0]
                            st.metric("Sentiment dominant", dominant_sentiment)
                except Exception as e:
                    print(f"Error processing sentiments: {str(e)}")
                    st.warning("Unable to analyze sentiments for this topic")
        
        if 'date' in topic_articles.columns and not topic_articles.empty:
            st.subheader('Évolution temporelle')
            fig_evolution = DataVisualizer.create_timeline_plot(
                topic_articles,
                f'Articles par jour - {selected_topic_name}'
            )
            st.plotly_chart(fig_evolution)
        
        st.subheader(f'Articles ({len(topic_articles)} articles)')
        
        if topic_articles.empty:
            st.info("Aucun article trouvé pour ce topic dans la période sélectionnée.")
        else:
            for date in sorted(topic_articles['date'].dt.date.unique(), reverse=True):
                date_articles = topic_articles[topic_articles['date'].dt.date == date]
                
                st.markdown(f"### {date.strftime('%d %B %Y')}")
                
                for _, article in date_articles.iterrows():
                    with st.expander(article['title']):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(article['summary'])
                            if pd.notna(article.get('link')):
                                st.markdown(f"[Lire l'article complet]({article['link']})")
                        
                        with col2:
                            st.write("**Heure:**", article['date'].strftime('%H:%M'))
                            if pd.notna(article.get('category')):
                                st.write("**Catégorie:**", article['category'])
                            
                            if article['entities']:
                                st.markdown("**Entités:**")
                                st.markdown(ResultsFormatter.format_entities(article['entities']))
                            
                            if 'sentiment' in article:
                                st.markdown("**Sentiment:**")
                                if isinstance(article['sentiment'], dict):
                                    st.markdown(ResultsFormatter.format_sentiment(article['sentiment']))
                                else:
                                    st.markdown(f"{article['sentiment']}")
                            
                            if article['themes']:
                                st.write("**Thèmes:**", ', '.join(article['themes'][:3]))
    
    def analyze_clusters(self):
        """Enhanced cluster analysis"""
        st.header('Analyse des Clusters')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            cluster_id = st.selectbox(
                'Sélectionnez un cluster',
                sorted(self.df['cluster'].unique())
            )
        
        with col2:
            time_range = st.selectbox(
                'Période',
                list(Config.TIME_RANGES.keys()),
                index=1
            )
            days_filter = Config.TIME_RANGES[time_range]
        
        cluster_articles = self.df[self.df['cluster'] == cluster_id].copy()
        
        if days_filter:
            cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days_filter)
            cluster_articles = cluster_articles[cluster_articles['date'] > cutoff_date]
        
        if not cluster_articles.empty and 'is_representative' in cluster_articles.columns:
            representative = cluster_articles[cluster_articles['is_representative']]
            if len(representative) > 0:
                st.subheader('Article représentatif')
                with st.expander(representative.iloc[0]['title'], expanded=True):
                    st.write(representative.iloc[0]['summary'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Distribution des Topics')
            topic_dist = cluster_articles['topic_name'].value_counts()
            if not topic_dist.empty:
                fig_topics = px.pie(
                    values=topic_dist.values,
                    names=topic_dist.index,
                    title='Répartition des Topics'
                )
                st.plotly_chart(fig_topics)
        
        with col2:
            st.subheader('Entités Principales')
            if not cluster_articles.empty:
                entities = [
                    ent['text'] for arts in cluster_articles['entities']
                    for ent in arts if ent['type'] in ['PERSON', 'ORGANIZATION', 'LOCATION']
                ]
                if entities:
                    top_entities = Counter(entities).most_common(10)
                    fig_entities = px.bar(
                        x=[e[0] for e in top_entities],
                        y=[e[1] for e in top_entities],
                        title='Top 10 Entités'
                    )
                    st.plotly_chart(fig_entities)
        
        if not cluster_articles.empty and 'date' in cluster_articles.columns:
            st.subheader('Évolution temporelle')
            fig_evolution = DataVisualizer.create_timeline_plot(
                cluster_articles,
                f'Articles par jour - Cluster {cluster_id}'
            )
            st.plotly_chart(fig_evolution)
        
        st.subheader(f'Articles du cluster ({len(cluster_articles)} articles)')
        
        if cluster_articles.empty:
            st.info("Aucun article trouvé pour ce cluster dans la période sélectionnée.")
        else:
            for date in sorted(cluster_articles['date'].dt.date.unique(), reverse=True):
                st.write(f"### {date.strftime('%d %B %Y')}")
                
                date_articles = cluster_articles[cluster_articles['date'].dt.date == date]
                for _, article in date_articles.iterrows():
                    with st.expander(article['title']):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(article['summary'])
                            if pd.notna(article.get('link')):
                                st.markdown(f"[Lire l'article complet]({article['link']})")
                        
                        with col2:
                            st.write("**Topic:**", article['topic_name'])
                            if pd.notna(article.get('category')):
                                st.write("**Catégorie:**", article['category'])
                            
                            if article['entities']:
                                st.markdown("**Entités:**")
                                st.markdown(ResultsFormatter.format_entities(article['entities']))
                            
                            if 'sentiment' in article:
                                st.markdown("**Sentiment:**")
                                if isinstance(article['sentiment'], dict):
                                    st.markdown(ResultsFormatter.format_sentiment(article['sentiment']))
                                else:
                                    st.markdown(f"{article['sentiment']}")
                            
                            if article['themes']:
                                st.write("**Thèmes:**", ', '.join(article['themes'][:3]))
    
    def advanced_search(self):
        """Advanced search interface with natural language processing"""
        st.header('Recherche Avancée')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input(
                'Posez votre question naturellement',
                placeholder="Ex: Que s'est-il passé à Bruxelles cette semaine ?"
            )
            
            if not search_query:
                st.info("""
                    Exemples de questions :
                    - Que s'est-il passé à Bruxelles cette semaine ?
                    - Montre-moi les derniers articles sur le climat
                    - Quels sont les événements importants du mois dernier ?
                    - Parle-moi des actualités politiques récentes
                    - Quelles sont les nouvelles concernant l'économie ?
                """)
        
        with col2:
            sentiment_filter = st.selectbox(
                'Sentiment',
                ['Tous', 'Positif', 'Négatif', 'Neutre']
            )
            
            categories = ['Toutes'] + sorted(self.df['category'].dropna().unique().tolist())
            selected_category = st.selectbox('Catégorie', categories)
        
        if search_query:
            query_info = self.search_engine.parse_query(search_query)
            base_mask = self.search_engine.create_search_filters(query_info, self.df)
            
            if sentiment_filter != 'Tous':
                sentiment_label = sentiment_filter.lower()
                base_mask &= self.df['sentiment'].apply(lambda x: x['label'] == sentiment_label)
            
            if selected_category != 'Toutes':
                base_mask &= (self.df['category'] == selected_category)
            
            results = self.search_engine.rank_results(self.df, query_info, base_mask)
            
            if len(results) > 0:
                st.subheader(f'Résultats ({len(results)} articles trouvés)')
                search_summary = self.search_engine.format_search_results(results, query_info)
                st.markdown(f"*{search_summary}*")
                
                for date in sorted(results['date'].dt.date.unique(), reverse=True):
                    st.write(f"### {date.strftime('%d %B %Y')}")
                    
                    date_results = results[results['date'].dt.date == date]
                    for _, article in date_results.iterrows():
                        relevance = f"{article['relevance']:.0f}% pertinent"
                        
                        title = TextProcessor.highlight_text(
                            article['title'],
                            query_info['keywords']  
                        )
                        
                        with st.expander(f"{title} - {relevance}"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                summary = TextProcessor.highlight_text(
                                    article['summary'],
                                    query_info['keywords']  
                                )
                                st.markdown(summary)
                                
                                if pd.notna(article.get('link')):
                                    st.markdown(f"[Lire l'article complet]({article['link']})")
                            
                            with col2:
                                st.write("**Heure:**", article['date'].strftime('%H:%M'))
                                
                                if pd.notna(article.get('category')):
                                    st.write("**Catégorie:**", article['category'])
                                
                                st.write("**Topic:**", article['topic_name'])
                                
                                if article['entities']:
                                    st.markdown("**Entités:**")
                                    st.markdown(ResultsFormatter.format_entities(article['entities']))
                                
                                if 'sentiment' in article:
                                    st.markdown("**Sentiment:**")
                                    st.markdown(ResultsFormatter.format_sentiment(article['sentiment']))
                                
                                if article['themes']:
                                    st.write("**Thèmes:**", ', '.join(article['themes'][:3]))
            else:
                st.warning('No articles found for these criteria.')
                st.info("""
                    Suggestions :
                    - Essayez des termes plus généraux
                    - Vérifiez l'orthographe
                    - Élargissez la période de recherche
                    - Essayez sans filtres supplémentaires
                """)

def main():
    app = TopicVisualizationApp()
    app.run()

if __name__ == "__main__":
    main()

