import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./collab.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
models.py
python
Copy code
# backend/app/models.py
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from .db import Base
from datetime import datetime

class Author(Base):
    __tablename__ = "authors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    affiliation = Column(String, nullable=True)
    # store precomputed embedding (as text; you can store as blob in production)
    embedding = Column(Text, nullable=True)

    papers = relationship("PaperAuthor", back_populates="author")

class Paper(Base):
    __tablename__ = "papers"
    id = Column(Integer, primary_key=True)
    title = Column(String, index=True)
    year = Column(Integer)
    venue = Column(String, nullable=True)

    authors = relationship("PaperAuthor", back_populates="paper")

class PaperAuthor(Base):
    __tablename__ = "paper_author"
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey("papers.id"))
    author_id = Column(Integer, ForeignKey("authors.id"))
    position = Column(Integer, nullable=True)

    paper = relationship("Paper", back_populates="authors")
    author = relationship("Author", back_populates="papers")
    __table_args__ = (UniqueConstraint('paper_id', 'author_id', name='_paper_author_uc'),)

class TrainedModel(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
schemas.py
python
Copy code
# backend/app/schemas.py
from pydantic import BaseModel
from typing import Optional, List, Dict

class AuthorCreate(BaseModel):
    name: str
    affiliation: Optional[str] = None

class PredictRequest(BaseModel):
    author_a: str
    author_b: str

class PredictResponse(BaseModel):
    author_a: str
    author_b: str
    probability: float
    features: Dict[str, float]
utils.py
python
Copy code
# backend/app/utils.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

def vec_to_text(vec: np.ndarray) -> str:
    return json.dumps(vec.tolist())

def text_to_vec(s: str):
    import json
    return np.array(json.loads(s))
ml_pipeline.py
python
Copy code
# backend/app/ml_pipeline.py
import os
import joblib
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
from .db import SessionLocal
from .models import Author, Paper, PaperAuthor
from .utils import text_to_vec, vec_to_text
from dotenv import load_dotenv
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "./model/joblib_rf.pkl")

# Feature functions
def common_neighbors(G, u, v):
    return len(list(nx.common_neighbors(G, u, v)))

def jaccard_coeff(G, u, v):
    # networkx.jaccard_coefficient returns generator; compute manually
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    if len(nu | nv) == 0:
        return 0.0
    return len(nu & nv) / len(nu | nv)

def pref_attachment(G, u, v):
    return G.degree[u] * G.degree[v]

def adamic_adar(G, u, v):
    return sum(1.0 / np.log(G.degree[w]) for w in nx.common_neighbors(G, u, v) if G.degree[w] > 1)

def research_similarity(emb_a, emb_b):
    if emb_a is None or emb_b is None:
        return 0.0
    emb_a = np.array(emb_a)
    emb_b = np.array(emb_b)
    denom = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
    if denom == 0:
        return 0.0
    return float(np.dot(emb_a, emb_b) / denom)

def build_graph_from_db(session):
    G = nx.Graph()
    authors = session.query(Author).all()
    for a in authors:
        G.add_node(a.name)
    # add edges from papers
    papers = session.query(Paper).all()
    for p in papers:
        pats = [pa.author for pa in p.authors]
        names = [a.name for a in pats]
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                if G.has_edge(names[i], names[j]):
                    G[names[i]][names[j]]['weight'] += 1
                else:
                    G.add_edge(names[i], names[j], weight=1)
    return G

def generate_training_data(session, positive_year_cutoff=None):
    """
    Create positive and negative pairs from DB.
    positive_year_cutoff: treat collaborations <= this year as existing.
    For simplicity: we treat all existing edges as positive; sample random non-edges as negative.
    """
    G = build_graph_from_db(session)
    nodes = list(G.nodes())
    X = []
    y = []
    # positive samples from existing edges
    for u, v, data in G.edges(data=True):
        ua = session.query(Author).filter(Author.name==u).first()
        va = session.query(Author).filter(Author.name==v).first()
        emb_u = text_to_vec(ua.embedding) if ua and ua.embedding else None
        emb_v = text_to_vec(va.embedding) if va and va.embedding else None
        feat = [
            common_neighbors(G, u, v),
            jaccard_coeff(G, u, v),
            pref_attachment(G, u, v),
            adamic_adar(G, u, v),
            research_similarity(emb_u, emb_v)
        ]
        X.append(feat)
        y.append(1)
    # negative samples: random pairs not edges
    import random
    non_edges = []
    attempts = 0
    desired_neg = len(X)
    while len(non_edges) < desired_neg and attempts < desired_neg * 10:
        a, b = random.sample(nodes, 2)
        if not G.has_edge(a,b):
            non_edges.append((a,b))
        attempts += 1
    for u, v in non_edges:
        ua = session.query(Author).filter(Author.name==u).first()
        va = session.query(Author).filter(Author.name==v).first()
        emb_u = text_to_vec(ua.embedding) if ua and ua.embedding else None
        emb_v = text_to_vec(va.embedding) if va and va.embedding else None
        feat = [
            common_neighbors(G, u, v),
            jaccard_coeff(G, u, v),
            pref_attachment(G, u, v),
            adamic_adar(G, u, v),
            research_similarity(emb_u, emb_v)
        ]
        X.append(feat)
        y.append(0)
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_and_save(session, save_path=MODEL_PATH):
    X, y = generate_training_data(session)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs))
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(clf, save_path)
    return metrics, save_path

def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None

def predict_pair(session, author_a_name, author_b_name, model):
    G = build_graph_from_db(session)
    u, v = author_a_name, author_b_name
    if u not in G:
        return None, f"Author {u} not found"
    if v not in G:
        return None, f"Author {v} not found"
    ua = session.query(Author).filter(Author.name==u).first()
    va = session.query(Author).filter(Author.name==v).first()
    emb_u = text_to_vec(ua.embedding) if ua and ua.embedding else None
    emb_v = text_to_vec(va.embedding) if va and va.embedding else None
    feat = [
        common_neighbors(G, u, v),
        jaccard_coeff(G, u, v),
        pref_attachment(G, u, v),
        adamic_adar(G, u, v),
        research_similarity(emb_u, emb_v)
    ]
    prob = float(model.predict_proba([feat])[0][1])
    feature_dict = {
        "common_neighbors": feat[0],
        "jaccard": feat[1],
        "preferential_attachment": feat[2],
        "adamic_adar": feat[3],
        "research_similarity": feat[4]
    }
    return prob, feature_dict
