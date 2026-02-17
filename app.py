import os, json, uuid, re, time
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import Counter
import hashlib
import numpy as np

from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate

# -----------------------
# Setup with Enhanced Config
# -----------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
PROFILES_DIR = os.path.join(DATA_DIR, "profiles")
FAISS_DIR = os.path.join(DATA_DIR, "faiss")
NOTES_DIR = os.path.join(DATA_DIR, "notes")
PROGRESS_DIR = os.path.join(DATA_DIR, "progress")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

for d in [DATA_DIR, UPLOADS_DIR, PROFILES_DIR, FAISS_DIR, NOTES_DIR, PROGRESS_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# Page config
st.set_page_config(
    page_title="Study Brain AI", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        margin-bottom: 2rem;
    }
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-badge {
        background: #48bb78;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        display: inline-block;
    }
    .warning-badge {
        background: #ed8936;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        display: inline-block;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background: #e2e8f0;
        margin-left: 20%;
    }
    .bot-message {
        background: #667eea;
        color: white;
        margin-right: 20%;
    }
    .feature-card {
        background: #f7fafc;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.5rem;
    }
    .progress-bar-container {
        background: #e2e8f0;
        border-radius: 10px;
        height: 10px;
        width: 100%;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        height: 10px;
    }
</style>
""", unsafe_allow_html=True)

if not GROQ_API_KEY:
    st.error("🚨 Missing GROQ_API_KEY. Add it to .env or environment variables.")
    st.stop()

# Initialize LLM with different models
llm_fast = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    groq_api_key=GROQ_API_KEY
)

llm_advanced = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    groq_api_key=GROQ_API_KEY
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Enhanced Base System Prompt
BASE_SYSTEM = """
You are Study Brain AI, an advanced adaptive tutoring system.

CORE PRINCIPLES:
1) Use ONLY the provided CONTEXT - no hallucinations
2) Adapt to student's profile (level, goals, learning style)
3) Focus on understanding, not memorization
4) Encourage active learning and critical thinking
5) Provide actionable feedback and next steps
6) Reference sources clearly

LEARNING STYLES ADAPTATION:
- Visual: Use diagrams, spatial explanations, visual analogies
- Auditory: Use rhythm, repetition, verbal explanations
- Reading/Writing: Provide structured notes, summaries
- Kinesthetic: Use examples, exercises, real-world applications

COGNITIVE LOAD MANAGEMENT:
- Break complex topics into chunks
- Build from known to unknown
- Use analogies and metaphors
- Provide retrieval practice
"""

PROFILE_BLOCK = "STUDENT PROFILE:\n{profile}\n"
CONTEXT_BLOCK  = "DOCUMENT CONTEXT:\n{context}\n"

def make_prompt(task: str):
    return ChatPromptTemplate.from_messages([
        ("system", BASE_SYSTEM),
        ("human", PROFILE_BLOCK + "\n" + CONTEXT_BLOCK + "\n" + task)
    ])

# -----------------------
# Enhanced Prompts Library - FIXED VERSION with escaped curly braces
# -----------------------
PROMPTS = {
    "explain": make_prompt("""
TASK: Explain {topic} comprehensively from the CONTEXT.

ADAPT TO: {learning_style} learning style

OUTPUT FORMAT:
## 🎯 Learning Objectives
3-4 clear objectives

## 📚 Core Concepts
- Main idea (simple explanation)
- Key components (with analogies if helpful)
- How it connects to other topics

## 🔍 Deep Dive
Step-by-step breakdown with examples

## 💡 Tips for {learning_style} Learners
Specific strategies for this learning style

## ✅ Check Your Understanding
3 questions with answers

## 🚀 Next Steps
What to study next
"""),
    
    "summarize": make_prompt("""
TASK: Create an executive summary of {topic}

OUTPUT:
## 📋 One-Page Summary
## 🔑 Key Takeaways (7 bullets)
## 📖 Vocabulary (term - definition)
## ❌ Common Pitfalls
## 🎯 Must Remember (3 things)
"""),
    
    "quiz": make_prompt("""
TASK: Create an adaptive quiz on {topic}

DIFFICULTY: {level}

OUTPUT:
## 📝 Quiz ({num_questions} questions)
Each question includes:
- Question
- Options (for MCQ)
- Correct answer
- Explanation with source
- Why students often get this wrong

## 📊 Difficulty Analysis
- Easy questions (concept checking)
- Medium questions (application)
- Hard questions (analysis/synthesis)

## 💪 Study Recommendations
Based on performance in each area
"""),
    
    "flashcards": make_prompt("""
TASK: Create spaced-repetition flashcards on {topic}

FORMAT for each card:
---
**Front:** Clear, concise question
**Back:** Complete answer with key points
**Hint:** One-word clue if needed
**Source:** [filename]
**Difficulty:** {level}
**Tags:** [topic, subtopic]
---

Create {num_cards} cards
"""),
    
    "study_plan": make_prompt("""
TASK: Design a personalized study plan

STUDENT: {level} level, {daily_time} min/day
EXAM DATE: {exam_date}
LEARNING STYLE: {learning_style}

OUTPUT:
## 🗓️ {days}-Day Sprint Plan

Day by day breakdown:
- Focus topics
- Activities (time allocation)
- Resources to use
- Success criteria

## 🎯 Milestones
Day 3, 7, 14 checkpoints

## 🔄 Revision Schedule
Using spaced repetition

## 📈 Progress Tracking
How to measure improvement
"""),
    
    "mindmap": make_prompt("""
TASK: Create a hierarchical mind map from CONTEXT

TOPIC: {topic}

RULES:
- Max 3 levels depth
- Use short, memorable labels
- Show relationships clearly

OUTPUT JSON (use exactly this format, replace with actual content):
{{"root": "Main Topic", "children": [{{"label": "Branch 1", "children": [{{"label": "Leaf 1.1"}}, {{"label": "Leaf 1.2"}}]}}, {{"label": "Branch 2", "children": [{{"label": "Leaf 2.1"}}]}}]}}

Make sure to output ONLY the JSON, no other text.
"""),
    
    "practice_problems": make_prompt("""
TASK: Generate practice problems on {topic}

TYPES:
- Conceptual (understanding)
- Applied (real-world)
- Analytical (critical thinking)

For each problem:
## Problem 
**Type:** [conceptual/applied/analytical]
**Difficulty:** {level}

**Question:**

**Solution:**

**Common Mistakes:**

**Learning Point:**
"""),
    
    "knowledge_gaps": make_prompt("""
TASK: Analyze knowledge gaps based on quiz performance

QUIZ RESULTS: {quiz_results}

OUTPUT:
## 🔍 Gap Analysis
- Strong areas (>=80%)
- Developing (60-79%)
- Weak (<60%)

## 🎯 Priority Topics to Review
1. [Topic] - Why it's foundational
2. [Topic] - Common prerequisite

## 📚 Targeted Resources
Specific sections from context to re-read

## ✅ Fix Plan
Next 30 minutes focused practice
"""),
    
    "compare_topics": make_prompt("""
TASK: Compare and contrast {topic1} and {topic2}

OUTPUT:
## 🤔 Similarities
- Point 1
- Point 2

## 🔄 Differences
| Aspect | {topic1} | {topic2} |
|--------|----------|----------|
| Key idea | | |
| When to use | | |
| Example | | |

## 💡 Decision Framework
How to choose which one to use

## 🎯 Practice
Scenario-based question to apply both
"""),
    
    "teach_me": make_prompt("""
TASK: Teach {topic} as if I'm a complete beginner

STYLE: Use simple analogies, avoid jargon initially

OUTPUT:
## 🎯 The Big Picture (1 paragraph)

## 📖 Simple Analogy
Real-world comparison everyone understands

## 🧩 Building Blocks
Break it down into tiny pieces

## 🔨 Build It Up
Show how pieces fit together

## ✅ Quick Check
3 simple questions

## 📚 If You Want to Go Deeper
Advanced concepts (with warnings they're advanced)
"""),
    
    "exam_strategy": make_prompt("""
TASK: Create exam preparation strategy for {topic}

EXAM TYPE: {exam_type}
TIME LEFT: {time_left}

OUTPUT:
## 🎯 High-Yield Topics (Pareto 80/20)
What gives most points for time invested

## ⏱️ Time Management
- How to allocate time during exam
- Which questions to attempt first

## 📝 Common Question Patterns
- Typical MCQ traps
- Short answer keywords
- Essay structures

## 🚨 Last-Minute Review
What to review the night before

## 💪 Exam Day Checklist
Mental and physical preparation
"""),
    
    "project_ideas": make_prompt("""
TASK: Suggest hands-on projects to apply {topic}

LEVEL: {level}

OUTPUT:
## 🚀 Beginner Project (2-3 hours)
- Description
- Learning objectives
- Step-by-step guide
- Expected outcome

## ⚡ Intermediate Project (1-2 days)
- Description
- Skills practiced
- Challenges to expect
- Success criteria

## 🔥 Advanced Project (1 week)
- Real-world application
- Integration with other topics
- Portfolio-worthy features
- Extension ideas
"""),
    
    "study_buddy": make_prompt("""
TASK: Act as a study buddy discussing {topic}

INTERACTION STYLE:
- Friendly and encouraging
- Ask probing questions
- Share "aha moments"
- Point out interesting connections

Current conversation:
{chat_history}

Respond naturally as a study partner would.
"""),
    
    "topic_graph": make_prompt("""
TASK: Extract topic relationships from CONTEXT

OUTPUT JSON (use exactly this format, replace with actual content):
{{"nodes": [{{"id": "topic1", "importance": 5, "prerequisites": []}}], "edges": [{{"from": "topic1", "to": "topic2", "relationship": "related"}}]}}

Make sure to output ONLY the JSON, no other text.
"""),
    
    "memory_techniques": make_prompt("""
TASK: Create memory aids for {topic}

TECHNIQUES TO USE:
- Mnemonics
- Visual associations
- Stories
- Chunking
- Method of Loci

OUTPUT:
## 🎨 Visual Memory Aids
Describe mental images

## 📝 Mnemonics
Acronyms and phrases

## 🏰 Memory Palace
How to place concepts in locations

## 🔄 Review Schedule
When to review for long-term retention
""")
}

# -----------------------
# Enhanced Utils
# -----------------------
class StudyBrainUtils:
    @staticmethod
    def read_pdf_text(path: str) -> str:
        try:
            reader = PdfReader(path)
            parts = []
            for p in reader.pages:
                parts.append(p.extract_text() or "")
            return "\n".join(parts).strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    @staticmethod
    def format_profile(p: dict) -> str:
        learning_styles = p.get('learning_style', [])
        if isinstance(learning_styles, list):
            learning_styles_str = ', '.join(learning_styles)
        else:
            learning_styles_str = str(learning_styles)
            
        return (
            f"🎓 Domain: {p.get('domain','')}\n"
            f"🎯 Goal: {p.get('goal','')}\n"
            f"📊 Level: {p.get('level','')}\n"
            f"⏱️ Daily time: {p.get('daily_time','')} min\n"
            f"📅 Exam date: {p.get('exam_date','')}\n"
            f"🧠 Learning style: {learning_styles_str}\n"
            f"💪 Motivation: {p.get('motivation','')}\n"
            f"⚠️ Challenges: {p.get('challenges','')}"
        )
    
    @staticmethod
    def save_profile(session_id: str, profile: dict):
        with open(os.path.join(PROFILES_DIR, f"{session_id}.json"), "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_profile(session_id: str):
        p = os.path.join(PROFILES_DIR, f"{session_id}.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    @staticmethod
    def build_vectorstore(session_id: str, docs: list[Document]):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(os.path.join(FAISS_DIR, session_id))
        
        # Save metadata
        metadata = {
            "num_docs": len(docs),
            "num_chunks": len(chunks),
            "sources": [d.metadata.get("source") for d in docs],
            "created_at": datetime.now().isoformat()
        }
        with open(os.path.join(FAISS_DIR, f"{session_id}_meta.json"), "w") as f:
            json.dump(metadata, f)
            
        return vs, chunks
    
    @staticmethod
    def load_vectorstore(session_id: str):
        path = os.path.join(FAISS_DIR, session_id)
        if not os.path.exists(path):
            return None
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    
    @staticmethod
    def retrieve_context(vs, query: str, k: int = 6, diversify: bool = False) -> str:
        if diversify:
            # Get more chunks and deduplicate by content
            hits = vs.similarity_search(query, k=k*2)
            seen = set()
            unique_hits = []
            for h in hits:
                content_hash = hashlib.md5(h.page_content[:100].encode()).hexdigest()
                if content_hash not in seen:
                    seen.add(content_hash)
                    unique_hits.append(h)
            hits = unique_hits[:k]
        else:
            hits = vs.similarity_search(query, k=k)
        
        ctx = []
        for i, d in enumerate(hits, 1):
            src = d.metadata.get("source", "unknown")
            ctx.append(f"📄 [Source {i}: {src}]\n{d.page_content}")
        return "\n\n---\n\n".join(ctx)
    
    @staticmethod
    def extract_json(text: str) -> dict:
        import re
        # Try to find JSON object in the text
        json_pattern = r'\{.*\}|\[.*\]'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue
        
        # If no valid JSON found, try to clean the text and parse
        try:
            # Remove any markdown formatting
            cleaned = re.sub(r'```json\n|```\n|```', '', text)
            return json.loads(cleaned)
        except:
            raise ValueError("No valid JSON found in response")
    
    @staticmethod
    def save_note(session_id: str, title: str, content: str, tags: list):
        note = {
            "id": str(uuid.uuid4()),
            "title": title,
            "content": content,
            "tags": tags,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        notes_file = os.path.join(NOTES_DIR, f"{session_id}.json")
        if os.path.exists(notes_file):
            with open(notes_file, "r") as f:
                notes = json.load(f)
        else:
            notes = []
        notes.append(note)
        with open(notes_file, "w") as f:
            json.dump(notes, f, indent=2)
        return note
    
    @staticmethod
    def load_notes(session_id: str):
        notes_file = os.path.join(NOTES_DIR, f"{session_id}.json")
        if os.path.exists(notes_file):
            with open(notes_file, "r") as f:
                return json.load(f)
        return []
    
    @staticmethod
    def track_progress(session_id: str, activity: str, score: float = None):
        progress_file = os.path.join(PROGRESS_DIR, f"{session_id}.json")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "activity": activity,
            "score": score
        }
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                progress = json.load(f)
        else:
            progress = []
        progress.append(entry)
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

# -----------------------
# Visualization Functions (Matplotlib only)
# -----------------------
class StudyVisualizations:
    @staticmethod
    def create_progress_chart(session_id: str):
        """Create progress chart using matplotlib"""
        progress_file = os.path.join(PROGRESS_DIR, f"{session_id}.json")
        if not os.path.exists(progress_file):
            return None
        
        with open(progress_file, "r") as f:
            progress = json.load(f)
        
        if not progress:
            return None
            
        df = pd.DataFrame(progress)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create daily activity counts
        df['date'] = df['timestamp'].dt.date
        daily_activity = df.groupby('date').size().reset_index()
        daily_activity.columns = ['date', 'count']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Daily activity
        ax1.plot(daily_activity['date'], daily_activity['count'], 
                marker='o', linestyle='-', linewidth=2, color='#667eea')
        ax1.set_title('Daily Study Activity', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Activities')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Scores if available
        if 'score' in df.columns and df['score'].notna().any():
            scores = df[df['score'].notna()]
            ax2.scatter(scores['timestamp'], scores['score'], 
                       color='#48bb78', s=100, alpha=0.6)
            ax2.set_title('Quiz Scores Over Time', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Score (%)')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_activity_pie_chart(session_id: str):
        """Create pie chart of activity types"""
        progress_file = os.path.join(PROGRESS_DIR, f"{session_id}.json")
        if not os.path.exists(progress_file):
            return None
        
        with open(progress_file, "r") as f:
            progress = json.load(f)
        
        if not progress:
            return None
        
        # Count activity types
        activity_types = [p.get('activity', 'unknown') for p in progress]
        activity_counts = Counter(activity_types)
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(activity_counts)))
        
        wedges, texts, autotexts = ax.pie(
            activity_counts.values(), 
            labels=activity_counts.keys(),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        
        # Style
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Study Activities Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def draw_mind_map(mm_data: dict):
        """Draw mind map using networkx and matplotlib"""
        G = nx.DiGraph()
        
        def add_nodes(parent, children, parent_id="root"):
            for i, child in enumerate(children):
                node_id = f"{parent_id}_{i}"
                G.add_node(node_id, label=child['label'])
                G.add_edge(parent_id, node_id)
                if 'children' in child:
                    add_nodes(node_id, child['children'])
        
        # Build graph
        root_label = mm_data.get('root', 'Root')
        G.add_node('root', label=root_label)
        add_nodes('root', mm_data.get('children', []))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Use hierarchical layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='#888', width=1, arrows=True, arrowsize=10)
        
        # Draw nodes with different colors by level
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            depth = len(node.split('_')) - 1 if node != 'root' else 0
            # Color gradient based on depth
            colors = ['#4299e1', '#667eea', '#9f7aea', '#ed64a6']
            node_colors.append(colors[min(depth, len(colors)-1)])
            # Size based on depth
            node_sizes.append(3000 - depth * 200)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        # Draw labels
        labels = {n: G.nodes[n]['label'] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
        
        ax.set_title(f"🧠 Mind Map: {root_label}", fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#4299e1', label='Level 1 (Main Branches)'),
            mpatches.Patch(color='#667eea', label='Level 2'),
            mpatches.Patch(color='#9f7aea', label='Level 3'),
            mpatches.Patch(color='#ed64a6', label='Level 4')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_study_plan_gantt(study_plan_text: str):
        """Create a simple Gantt chart for study plan"""
        # This is a simplified version - in practice you'd parse the study plan
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sample data structure - would be parsed from AI response
        days = list(range(1, 8))
        topics = ['Topic A', 'Topic B', 'Topic C', 'Topic D', 'Topic E']
        
        # Create a simple heatmap of study intensity
        data = np.random.rand(len(topics), len(days))
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # Customize
        ax.set_xticks(np.arange(len(days)))
        ax.set_yticks(np.arange(len(topics)))
        ax.set_xticklabels([f'Day {d}' for d in days])
        ax.set_yticklabels(topics)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        ax.set_title('Study Plan Intensity Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Days')
        ax.set_ylabel('Topics')
        
        # Colorbar
        plt.colorbar(im, ax=ax, label='Study Intensity')
        
        plt.tight_layout()
        return fig

# -----------------------
# Main UI
# -----------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = []

session_id = st.session_state.session_id
utils = StudyBrainUtils()
viz = StudyVisualizations()

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🧠 Study Brain AI</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'><span class='warning-badge'>Session: {session_id}</span></p>", unsafe_allow_html=True)
    
    # Quick Stats
    prof = utils.load_profile(session_id)
    if prof:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Level", prof.get('level', 'N/A'))
        with col2:
            st.metric("Daily Time", f"{prof.get('daily_time', 0)}m")
    
    # Navigation with icons
    st.markdown("---")
    pages = {
        "📝 Profile": "Profile",
        "📚 Upload & Index": "Upload",
        "🎯 Study Modes": "Study",
        "📊 Progress": "Progress",
        "📔 My Notes": "Notes",
        "🤖 Study Buddy": "Chat"
    }
    
    page_icons = list(pages.keys())
    selected_page = st.radio("Navigate", page_icons, index=0)
    page = pages[selected_page]
    
    # Quick actions
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    if st.button("📋 Topic Index", use_container_width=True):
        st.session_state['quick_mode'] = "Topic Index"
    
    if st.button("🎴 Flashcards", use_container_width=True):
        st.session_state['quick_mode'] = "Flashcards"

# Main content area
if page == "Profile":
    st.markdown("<h1 class='main-header'>📝 Your Learning Profile</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Tell me about yourself so I can personalize your learning</p>", unsafe_allow_html=True)
    
    existing = utils.load_profile(session_id) or {}
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Academic Info")
            domain = st.text_input("Domain / Subject", value=existing.get("domain", "General"))
            goal = st.selectbox("Main Goal", 
                ["Exam Preparation", "Deep Understanding", "Quick Overview", "Research", "Teaching Others"],
                index=["Exam Preparation", "Deep Understanding", "Quick Overview", "Research", "Teaching Others"].index(
                    existing.get("goal", "Deep Understanding")) if existing.get("goal") in ["Exam Preparation", "Deep Understanding", "Quick Overview", "Research", "Teaching Others"] else 1)
            level = st.select_slider("Current Level", 
                options=["Beginner", "Basic", "Intermediate", "Advanced", "Expert"],
                value=existing.get("level", "Intermediate"))
        
        with col2:
            st.subheader("⏰ Time & Schedule")
            daily_time = st.slider("Daily study time (minutes)", 15, 480, 
                value=int(existing.get("daily_time", 90)))
            
            # Handle exam date
            exam_date_str = existing.get("exam_date", "")
            if exam_date_str:
                try:
                    exam_date_default = datetime.fromisoformat(exam_date_str).date()
                except:
                    exam_date_default = datetime.now().date() + timedelta(days=30)
            else:
                exam_date_default = datetime.now().date() + timedelta(days=30)
                
            exam_date = st.date_input("Exam date (if any)", 
                value=exam_date_default,
                min_value=datetime.now().date())
            days_left = (exam_date - datetime.now().date()).days if exam_date else None
            if days_left and days_left > 0:
                st.info(f"📅 {days_left} days until exam")
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🧠 Learning Style")
            learning_style_options = ["Visual (diagrams, charts)", "Auditory (discussions, explanations)", 
                 "Reading/Writing (notes, text)", "Kinesthetic (examples, practice)"]
            
            existing_styles = existing.get("learning_style", ["Reading/Writing (notes, text)"])
            # Ensure existing_styles is a list
            if isinstance(existing_styles, str):
                existing_styles = [existing_styles]
            
            learning_style = st.multiselect("How do you learn best?",
                learning_style_options,
                default=existing_styles)
        
        with col4:
            st.subheader("🎯 Motivation & Challenges")
            motivation = st.text_area("What motivates you?", value=existing.get("motivation", ""))
            challenges = st.text_area("What challenges do you face?", value=existing.get("challenges", ""))
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("💾 Save Profile", use_container_width=True):
            prof = {
                "domain": domain.strip(),
                "goal": goal,
                "level": level,
                "daily_time": daily_time,
                "exam_date": exam_date.isoformat() if exam_date else "",
                "learning_style": learning_style,
                "motivation": motivation.strip(),
                "challenges": challenges.strip(),
                "updated_at": datetime.now().isoformat()
            }
            utils.save_profile(session_id, prof)
            utils.track_progress(session_id, "profile_created")
            st.success("✅ Profile saved successfully!")
            st.balloons()
    
    # Display current profile
    if existing:
        st.markdown("---")
        st.subheader("📋 Current Profile Summary")
        st.code(utils.format_profile(utils.load_profile(session_id) or {}))

elif page == "Upload":
    st.markdown("<h1 class='main-header'>📚 Upload & Process Materials</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload PDF files", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Upload your study materials (lecture notes, textbooks, articles)"
        )
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Quick Stats")
        meta_path = os.path.join(FAISS_DIR, f"{session_id}_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            st.metric("Documents", meta['num_docs'])
            st.metric("Chunks", meta['num_chunks'])
            st.caption(f"Created: {meta['created_at'][:10]}")
        else:
            st.info("No documents processed yet")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_files and st.button("🚀 Process Documents", use_container_width=True):
        with st.spinner("Processing your documents..."):
            progress_bar = st.progress(0)
            
            docs = []
            for i, f in enumerate(uploaded_files):
                # Save file
                save_path = os.path.join(UPLOADS_DIR, f"{session_id}__{f.name}")
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())
                
                # Extract text
                text = utils.read_pdf_text(save_path)
                if text:
                    docs.append(Document(
                        page_content=text, 
                        metadata={"source": f.name, "uploaded": datetime.now().isoformat()}
                    ))
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if docs:
                vs, chunks = utils.build_vectorstore(session_id, docs)
                utils.track_progress(session_id, f"uploaded_{len(docs)}_docs")
                
                st.success(f"✅ Successfully processed {len(docs)} documents into {len(chunks)} chunks!")
                
                # Show preview
                with st.expander("📄 Document Preview"):
                    for doc in docs[:2]:  # Show first 2 docs
                        st.markdown(f"**{doc.metadata['source']}**")
                        st.text(doc.page_content[:500] + "...")
            else:
                st.error("Couldn't extract text. PDFs might be scanned images.")

elif page == "Study":
    st.markdown("<h1 class='main-header'>🎯 Study Modes</h1>", unsafe_allow_html=True)
    
    prof = utils.load_profile(session_id)
    if not prof:
        st.warning("⚠️ Please complete your profile first!")
        st.stop()
    
    vs = utils.load_vectorstore(session_id)
    if not vs:
        st.warning("⚠️ Please upload and process documents first!")
        st.stop()
    
    # Check for quick mode from sidebar
    quick_mode = st.session_state.get('quick_mode', None)
    
    # Study mode selection with icons
    modes = {
        "📖 Explain Topic": "explain",
        "📋 Summarize": "summarize", 
        "📝 Practice Quiz": "quiz",
        "🎴 Flashcards": "flashcards",
        "🗓️ Study Plan": "study_plan",
        "🧠 Mind Map": "mindmap",
        "✏️ Practice Problems": "practice_problems",
        "🔍 Knowledge Gaps": "knowledge_gaps",
        "🔄 Compare Topics": "compare_topics",
        "👶 Teach Me Like I'm 5": "teach_me",
        "📝 Exam Strategy": "exam_strategy",
        "💡 Project Ideas": "project_ideas",
        "🧮 Memory Techniques": "memory_techniques"
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Set default index based on quick mode
        default_index = 0
        if quick_mode == "Topic Index":
            # Find index for Topic Index - we don't have it in this list, so default to 0
            default_index = 0
        elif quick_mode == "Flashcards":
            # Find index for Flashcards
            for i, (display, mode_val) in enumerate(modes.items()):
                if mode_val == "flashcards":
                    default_index = i
                    break
        
        selected_mode_display = st.selectbox("Select Study Mode", list(modes.keys()), index=default_index)
        mode = modes[selected_mode_display]
        
        # Clear quick mode
        if 'quick_mode' in st.session_state:
            del st.session_state['quick_mode']
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        difficulty = st.select_slider("Difficulty", ["Easy", "Medium", "Hard", "Expert"], value="Medium")
    
    # Topic input based on mode
    topic = ""
    topic2 = ""
    topic_required = mode not in ["study_plan", "topic_graph"]
    
    if mode == "compare_topics":
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            topic = st.text_input("🔍 Topic 1", 
                placeholder="First topic...",
                value=st.session_state.current_topic)
        with col_t2:
            topic2 = st.text_input("🔍 Topic 2", 
                placeholder="Second topic...")
        st.session_state.current_topic = topic
    elif topic_required:
        topic = st.text_input("🔍 Topic / Question", 
            placeholder="e.g., Neural Networks, Photosynthesis, World War II...",
            value=st.session_state.current_topic)
        st.session_state.current_topic = topic
    
    # Mode-specific inputs
    extra_params = {}
    
    if mode == "quiz":
        num_questions = st.slider("Number of questions", 5, 20, 10)
        extra_params['num_questions'] = num_questions
    
    elif mode == "flashcards":
        num_cards = st.slider("Number of flashcards", 5, 30, 12)
        extra_params['num_cards'] = num_cards
    
    elif mode == "study_plan":
        days = st.slider("Plan duration (days)", 3, 30, 7)
        extra_params['days'] = days
    
    elif mode == "compare_topics":
        if topic2:
            extra_params['topic1'] = topic
            extra_params['topic2'] = topic2
    
    elif mode == "exam_strategy":
        exam_type = st.selectbox("Exam type", ["MCQ", "Essay", "Mixed", "Open book"])
        time_left = st.selectbox("Time until exam", ["1 day", "1 week", "1 month", "3 months"])
        extra_params['exam_type'] = exam_type
        extra_params['time_left'] = time_left
    
    elif mode == "knowledge_gaps":
        if st.session_state.quiz_results:
            extra_params['quiz_results'] = json.dumps(st.session_state.quiz_results[-1])
        else:
            st.warning("⚠️ Take a quiz first to analyze gaps! Using sample data.")
            extra_params['quiz_results'] = json.dumps({"score": 65, "weak_areas": ["concept1", "concept2"]})
    
    # Additional options
    with st.expander("⚙️ Advanced Options"):
        col_a, col_b = st.columns(2)
        with col_a:
            k_chunks = st.slider("Number of context chunks", 3, 20, 6)
            diversify = st.checkbox("Diversify sources", True)
        with col_b:
            use_advanced_model = st.checkbox("Use advanced model", True)
            include_examples = st.checkbox("Include examples", True)
    
    # Run button
    if st.button("🚀 Generate", use_container_width=True):
        # Validate inputs
        if mode == "compare_topics":
            if not topic or not topic2:
                st.warning("Please enter both topics to compare")
                st.stop()
        elif topic_required and not topic:
            st.warning("Please enter a topic")
            st.stop()
        
        with st.spinner(f"Generating {selected_mode_display}..."):
            try:
                # Get context
                if mode == "compare_topics":
                    search_query = f"{topic} {topic2}"
                else:
                    search_query = topic if topic else "main topics"
                    
                ctx = utils.retrieve_context(vs, search_query, k=k_chunks, diversify=diversify)
                
                # Select model
                llm = llm_advanced if use_advanced_model else llm_fast
                
                # Prepare prompt parameters
                learning_style_str = ", ".join(prof.get('learning_style', ['Reading/Writing (notes, text)']))
                
                # Base parameters
                prompt_params = {
                    "profile": utils.format_profile(prof),
                    "context": ctx,
                    "topic": topic,
                    "level": difficulty.lower(),
                    "learning_style": learning_style_str,
                    "daily_time": prof.get('daily_time', 60),
                    "exam_date": prof.get('exam_date', 'Not set'),
                }
                
                # Add mode-specific parameters
                prompt_params.update(extra_params)
                
                # For compare_topics, ensure we have both topics
                if mode == "compare_topics":
                    prompt_params["topic1"] = topic
                    prompt_params["topic2"] = topic2
                
                # For mindmap, ensure we have the right parameters
                if mode == "mindmap":
                    # Make sure topic is set
                    if not topic:
                        prompt_params["topic"] = "Document Overview"
                
                # Get response
                if mode in PROMPTS:
                    response = llm.invoke(PROMPTS[mode].format(**prompt_params)).content
                else:
                    # Fallback to generic prompt
                    generic_prompt = make_prompt(f"TASK: {selected_mode_display} about {topic}")
                    response = llm.invoke(generic_prompt.format(**prompt_params)).content
                
                # Display response
                st.markdown("---")
                st.markdown(response)
                
                # Track progress
                utils.track_progress(session_id, f"used_{mode}")
                
                # Save option
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("💾 Save to Notes"):
                        title = f"{selected_mode_display}"
                        if topic:
                            title += f": {topic}"
                        if mode == "compare_topics" and topic2:
                            title += f" vs {topic2}"
                        note = utils.save_note(session_id, title, response, [mode, topic])
                        st.success("✅ Saved to notes!")
                
                with col2:
                    if st.button("📋 Copy to Clipboard"):
                        st.code(response, language="markdown")
                        st.info("Copy the text above (Ctrl+C)")
                
                with col3:
                    if mode == "mindmap":
                        try:
                            mm_data = utils.extract_json(response)
                            fig = viz.draw_mind_map(mm_data)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Could not create mind map visualization: {e}")
                            st.write("Raw response:")
                            st.code(response)
                    
                    elif mode == "knowledge_gaps":
                        # Store quiz results
                        st.session_state.quiz_results.append({"analysis": response[:100], "timestamp": datetime.now().isoformat()})
                        
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.exception(e)

elif page == "Progress":
    st.markdown("<h1 class='main-header'>📊 Your Learning Progress</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Load progress data
    progress_file = os.path.join(PROGRESS_DIR, f"{session_id}.json")
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)
        
        if progress:
            # Calculate stats
            total_activities = len(progress)
            
            # Count unique days
            dates = set()
            for p in progress:
                if 'timestamp' in p:
                    dates.add(p['timestamp'][:10])
            unique_days = len(dates)
            
            # Activity types
            activity_types = [p.get('activity', 'unknown') for p in progress if 'activity' in p]
            activity_counts = Counter(activity_types)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Total Activities", total_activities)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Days Active", unique_days)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Study Modes Used", len(activity_counts))
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Progress chart
            st.markdown("---")
            st.subheader("📈 Activity Over Time")
            fig = viz.create_progress_chart(session_id)
            if fig:
                st.pyplot(fig)
            else:
                st.info("Not enough data for chart")
            
            # Activity breakdown
            st.markdown("---")
            st.subheader("📊 Activity Breakdown")
            
            # Create pie chart
            if activity_counts:
                fig = viz.create_activity_pie_chart(session_id)
                if fig:
                    st.pyplot(fig)
            
            # Progress bar for today
            st.markdown("---")
            st.subheader("📅 Today's Progress")
            
            today = datetime.now().date().isoformat()
            today_activities = [p for p in progress if p.get('timestamp', '').startswith(today)]
            
            # Create custom progress bar
            target_activities = 5  # Daily goal
            today_count = len(today_activities)
            progress_percent = min(100, (today_count / target_activities) * 100)
            
            st.markdown(f"**Today's activities:** {today_count}/{target_activities}")
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width: {progress_percent}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Recent activity table
            st.markdown("---")
            st.subheader("📋 Recent Activity")
            recent = progress[-10:][::-1]  # Last 10, newest first
            
            for act in recent:
                with st.expander(f"📅 {act.get('timestamp', 'Unknown')[:16]} - {act.get('activity', 'Unknown')}"):
                    if act.get('score'):
                        st.metric("Score", f"{act['score']}%")
                    st.json(act)
        else:
            st.info("No progress data yet. Start studying to track your progress!")
    else:
        st.info("No progress data yet. Start studying to track your progress!")

elif page == "Notes":
    st.markdown("<h1 class='main-header'>📔 My Study Notes</h1>", unsafe_allow_html=True)
    
    notes = utils.load_notes(session_id)
    
    if notes:
        st.subheader(f"📚 Your Notes ({len(notes)})")
        
        # Search/filter
        search = st.text_input("🔍 Search notes", placeholder="Enter keyword...")
        
        filtered_notes = notes
        if search:
            filtered_notes = [n for n in notes if search.lower() in n['title'].lower() or search.lower() in n['content'].lower()]
        
        # Display notes
        for note in filtered_notes[-20:]:  # Show recent 20
            with st.expander(f"📝 {note['title']} - {note.get('created_at', 'Unknown')[:10]}"):
                st.markdown(note['content'])
                if note.get('tags'):
                    st.caption(f"🏷️ Tags: {', '.join(note['tags'])}")
                
                # Delete button
                if st.button(f"🗑️ Delete", key=f"delete_{note['id']}"):
                    notes.remove(note)
                    with open(os.path.join(NOTES_DIR, f"{session_id}.json"), "w") as f:
                        json.dump(notes, f, indent=2)
                    st.rerun()
    else:
        st.info("No notes yet. Save study sessions to create notes!")
        
        # Quick note creator
        st.markdown("---")
        st.subheader("➕ Quick Note")
        note_title = st.text_input("Title")
        note_content = st.text_area("Content")
        note_tags = st.text_input("Tags (comma separated)", placeholder="e.g., important, review, concept")
        
        if st.button("Create Note"):
            if note_title and note_content:
                tags_list = [t.strip() for t in note_tags.split(",")] if note_tags else []
                note = utils.save_note(session_id, note_title, note_content, tags_list)
                st.success("Note saved!")
                st.rerun()
            else:
                st.warning("Please enter title and content")

elif page == "Chat":
    st.markdown("<h1 class='main-header'>🤖 Study Buddy Chat</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Chat with me about your studies!</p>", unsafe_allow_html=True)
    
    prof = utils.load_profile(session_id)
    vs = utils.load_vectorstore(session_id)
    
    if not prof:
        st.warning("⚠️ Please complete your profile first!")
        st.stop()
    
    if not vs:
        st.warning("⚠️ Please upload and process documents first!")
        st.stop()
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your studies..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get relevant context
                    ctx = utils.retrieve_context(vs, prompt, k=4)
                    
                    # Prepare chat history
                    chat_history_str = "\n".join([
                        f"{msg['role']}: {msg['content']}" 
                        for msg in st.session_state.chat_history[-5:]  # Last 5 messages
                    ])
                    
                    # Generate response
                    prompt_params = {
                        "profile": utils.format_profile(prof),
                        "context": ctx,
                        "topic": prompt,
                        "chat_history": chat_history_str
                    }
                    
                    response = llm_fast.invoke(PROMPTS["study_buddy"].format(**prompt_params)).content
                    
                except Exception as e:
                    response = f"I'm having trouble responding right now. Error: {str(e)}"
                
                st.markdown(response)
                
                # Add to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Track
                utils.track_progress(session_id, "chat_interaction")
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🧠 Study Brain AI - Personalized Learning Assistant")
with col2:
    st.caption(f"Session: {session_id}")
with col3:
    st.caption(f"v2.0 | {datetime.now().strftime('%Y-%m-%d')}")