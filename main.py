from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import wikipediaapi
import re
from enum import Enum
from difflib import SequenceMatcher
from collections import Counter
import time
import json
from datetime import datetime, timedelta
import uuid
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
import sqlite3
import hashlib
from contextlib import contextmanager
import jwt
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Database configuration
DATABASE_URL = "echolens.db"
SECRET_KEY = "your-secret-key-change-in-production"  # Change this in production!
ALGORITHM = "HS256"

# Initialize FastAPI app
app = FastAPI(
    title="EchoLens - AI Content Fact Checker",
    description="Advanced AI content verification against Wikipedia with user sessions and analytics",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class ClaimType(str, Enum):
    TEMPORAL = "temporal"
    QUANTITATIVE = "quantitative"
    RELATIONAL = "relational"
    ATTRIBUTIVE = "attributive"
    COMPARATIVE = "comparative"
    EXISTENTIAL = "existential"
    BIOGRAPHICAL = "biographical"
    GEOGRAPHICAL = "geographical"
    SCIENTIFIC = "scientific"

class Verdict(str, Enum):
    STRONGLY_SUPPORTED = "STRONGLY_SUPPORTED"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    STRONGLY_CONTRADICTED = "STRONGLY_CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"
    MISLEADING = "MISLEADING"

class FactCheckRequest(BaseModel):
    content: str
    language: str = "en"
    session_id: Optional[str] = None

class UserSession(BaseModel):
    session_id: str
    created_at: str
    last_activity: str
    analysis_count: int

class AnalysisHistory(BaseModel):
    analysis_id: str
    session_id: str
    content_preview: str
    accuracy_score: float
    misinformation_risk: str
    total_claims: int
    created_at: str

class VerificationResult(BaseModel):
    claim: str
    verdict: Verdict
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    wikipedia_source: str
    similarity_score: float
    claim_type: ClaimType
    explanation: str
    entity_matches: List[str]
    confidence_factors: List[str]

class FactCheckResponse(BaseModel):
    analysis_id: str
    session_id: str
    timestamp: str
    content: str
    total_claims: int
    verified_claims: int
    accuracy_score: float
    average_confidence: float
    misinformation_risk: str
    detailed_results: List[VerificationResult]
    summary: str
    processing_time: float
    claim_breakdown: Dict[str, Any]

class AnalyticsResponse(BaseModel):
    total_analyses: int
    average_accuracy: float
    risk_distribution: Dict[str, int]
    popular_claim_types: Dict[str, int]
    recent_activity: List[AnalysisHistory]

# Database setup
def init_database():
    """Initialize the SQLite database with required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analysis_count INTEGER DEFAULT 0
            )
        ''')
        
        # Analysis history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                analysis_id TEXT PRIMARY KEY,
                session_id TEXT,
                content TEXT,
                content_preview TEXT,
                accuracy_score REAL,
                misinformation_risk TEXT,
                total_claims INTEGER,
                verified_claims INTEGER,
                average_confidence REAL,
                processing_time REAL,
                claim_breakdown TEXT,
                detailed_results TEXT,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
            )
        ''')
        
        # Analytics table for aggregated data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE DEFAULT CURRENT_DATE,
                total_analyses INTEGER DEFAULT 0,
                average_accuracy REAL DEFAULT 0,
                high_risk_count INTEGER DEFAULT 0,
                medium_risk_count INTEGER DEFAULT 0,
                low_risk_count INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")

@contextmanager
def get_db_connection():
    """Database connection context manager"""
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

class DatabaseManager:
    def __init__(self):
        init_database()
    
    def create_user_session(self) -> str:
        """Create a new user session and return session ID"""
        session_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO user_sessions (session_id) VALUES (?)',
                (session_id,)
            )
            conn.commit()
        logger.info(f"Created new user session: {session_id}")
        return session_id
    
    def get_user_session(self, session_id: str) -> Optional[Dict]:
        """Get user session data"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM user_sessions WHERE session_id = ?',
                (session_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_session_activity(self, session_id: str):
        """Update last activity timestamp for session"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE user_sessions SET last_activity = CURRENT_TIMESTAMP, analysis_count = analysis_count + 1 WHERE session_id = ?',
                (session_id,)
            )
            conn.commit()
    
    def save_analysis(self, analysis_data: Dict, session_id: str):
        """Save analysis results to database"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare data for storage
            content_preview = analysis_data['content'][:100] + '...' if len(analysis_data['content']) > 100 else analysis_data['content']
            
            cursor.execute('''
                INSERT INTO analysis_history (
                    analysis_id, session_id, content, content_preview, accuracy_score,
                    misinformation_risk, total_claims, verified_claims, average_confidence,
                    processing_time, claim_breakdown, detailed_results, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_data['analysis_id'],
                session_id,
                analysis_data['content'],
                content_preview,
                analysis_data['accuracy_score'],
                analysis_data['misinformation_risk'],
                analysis_data['total_claims'],
                analysis_data['verified_claims'],
                analysis_data['average_confidence'],
                analysis_data['processing_time'],
                json.dumps(analysis_data['claim_breakdown']),
                json.dumps(analysis_data['detailed_results']),
                analysis_data['summary']
            ))
            
            # Update analytics
            self._update_analytics(analysis_data)
            
            conn.commit()
        
        logger.info(f"Saved analysis {analysis_data['analysis_id']} for session {session_id}")
    
    def _update_analytics(self, analysis_data: Dict):
        """Update analytics aggregation"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            today = datetime.now().date()
            
            # Check if analytics entry exists for today
            cursor.execute('SELECT * FROM analytics WHERE date = ?', (today,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing entry
                cursor.execute('''
                    UPDATE analytics SET 
                    total_analyses = total_analyses + 1,
                    average_accuracy = (average_accuracy * total_analyses + ?) / (total_analyses + 1),
                    high_risk_count = high_risk_count + ?,
                    medium_risk_count = medium_risk_count + ?,
                    low_risk_count = low_risk_count + ?
                    WHERE date = ?
                ''', (
                    analysis_data['accuracy_score'],
                    1 if analysis_data['misinformation_risk'] in ['HIGH', 'VERY_HIGH'] else 0,
                    1 if analysis_data['misinformation_risk'] == 'MEDIUM' else 0,
                    1 if analysis_data['misinformation_risk'] in ['LOW', 'VERY_LOW'] else 0,
                    today
                ))
            else:
                # Create new entry
                cursor.execute('''
                    INSERT INTO analytics (
                        date, total_analyses, average_accuracy,
                        high_risk_count, medium_risk_count, low_risk_count
                    ) VALUES (?, 1, ?, ?, ?, ?)
                ''', (
                    today,
                    analysis_data['accuracy_score'],
                    1 if analysis_data['misinformation_risk'] in ['HIGH', 'VERY_HIGH'] else 0,
                    1 if analysis_data['misinformation_risk'] == 'MEDIUM' else 0,
                    1 if analysis_data['misinformation_risk'] in ['LOW', 'VERY_LOW'] else 0
                ))
    
    def get_analysis_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get analysis history for a session"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT analysis_id, session_id, content_preview, accuracy_score,
                       misinformation_risk, total_claims, created_at
                FROM analysis_history
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (session_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict]:
        """Get full analysis data by ID"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM analysis_history WHERE analysis_id = ?', (analysis_id,))
            row = cursor.fetchone()
            
            if row:
                data = dict(row)
                # Parse JSON fields
                data['claim_breakdown'] = json.loads(data['claim_breakdown'])
                data['detailed_results'] = json.loads(data['detailed_results'])
                return data
            return None
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get aggregated analytics data"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Total analyses
            cursor.execute('SELECT COUNT(*) as total FROM analysis_history')
            total_analyses = cursor.fetchone()['total']
            
            # Average accuracy
            cursor.execute('SELECT AVG(accuracy_score) as avg_accuracy FROM analysis_history')
            avg_accuracy = cursor.fetchone()['avg_accuracy'] or 0
            
            # Risk distribution
            cursor.execute('''
                SELECT misinformation_risk, COUNT(*) as count 
                FROM analysis_history 
                GROUP BY misinformation_risk
            ''')
            risk_distribution = {row['misinformation_risk']: row['count'] for row in cursor.fetchall()}
            
            # Popular claim types (from recent analyses)
            cursor.execute('''
                SELECT claim_breakdown FROM analysis_history 
                ORDER BY created_at DESC LIMIT 50
            ''')
            claim_type_counts = Counter()
            for row in cursor.fetchall():
                breakdown = json.loads(row['claim_breakdown'])
                if 'claim_types' in breakdown:
                    claim_type_counts.update(breakdown['claim_types'])
            
            # Recent activity
            cursor.execute('''
                SELECT analysis_id, session_id, content_preview, accuracy_score,
                       misinformation_risk, total_claims, created_at
                FROM analysis_history
                ORDER BY created_at DESC
                LIMIT 10
            ''')
            recent_activity = [dict(row) for row in cursor.fetchall()]
            
            return {
                'total_analyses': total_analyses,
                'average_accuracy': round(avg_accuracy, 3),
                'risk_distribution': risk_distribution,
                'popular_claim_types': dict(claim_type_counts),
                'recent_activity': recent_activity
            }
    
    def cleanup_old_sessions(self, days: int = 7):
        """Clean up sessions older than specified days"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
            
            # Delete old analyses first (foreign key constraint)
            cursor.execute('''
                DELETE FROM analysis_history 
                WHERE session_id IN (
                    SELECT session_id FROM user_sessions 
                    WHERE last_activity < ?
                )
            ''', (cutoff_date,))
            
            # Delete old sessions
            cursor.execute('DELETE FROM user_sessions WHERE last_activity < ?', (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old sessions")
            
            return deleted_count

# Initialize database manager
db_manager = DatabaseManager()

# Enhanced Fact Checking Engine (same as before, but updated for EchoLens)
class RobustWikipediaFactChecker:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='EchoLens/3.0 (https://echolens.ai)',
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'with', 'by'])
        
        # Simple entity patterns
        self.entity_patterns = {
            'person': r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',
            'organization': r'\b([A-Z][a-zA-Z&]+(?: [A-Z][a-zA-Z]+)*)\b',
            'year': r'\b(19|20)\d{2}\b',
            'number': r'\b\d+(?:\.\d+)? (?:million|billion|thousand|percent|%)\b'
        }

    # ... (All the existing RobustWikipediaFactChecker methods remain the same)
    # [Include all the existing methods from your current implementation]
    def extract_claims_from_ai_content(self, ai_content: str) -> List[Dict]:
        """Robust claim extraction with error handling"""
        try:
            claims = []
            sentences = self._simple_sentence_split(ai_content)
            
            for sentence in sentences:
                if self._is_factual_claim(sentence):
                    claim_type = self._classify_claim_type(sentence)
                    entities = self._simple_entity_extraction(sentence)
                    confidence = self._assess_claim_confidence(sentence)
                    
                    claims.append({
                        'text': sentence.strip(),
                        'claim_type': claim_type,
                        'confidence': confidence,
                        'entities': entities,
                        'keywords': self._extract_simple_keywords(sentence),
                        'context': 'general'
                    })
            
            return claims
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            return []

    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]

    def _simple_entity_extraction(self, sentence: str) -> List[str]:
        """Simple but robust entity extraction"""
        entities = []
        
        # Extract capitalized sequences
        words = sentence.split()
        current_entity = []
        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                current_entity.append(clean_word)
            else:
                if len(current_entity) > 0:
                    entity_text = ' '.join(current_entity)
                    if len(entity_text) > 3:
                        entities.append(entity_text)
                    current_entity = []
        
        if len(current_entity) > 0:
            entity_text = ' '.join(current_entity)
            if len(entity_text) > 3:
                entities.append(entity_text)
        
        # Extract years and numbers
        years = re.findall(r'\b(19|20)\d{2}\b', sentence)
        entities.extend(years)
        
        numbers = re.findall(r'\b\d+\.?\d* (?:million|billion|thousand)\b', sentence)
        entities.extend(numbers)
        
        return list(set(entities))[:5]  # Return top 5 entities

    def _extract_simple_keywords(self, sentence: str) -> List[str]:
        """Extract simple keywords"""
        words = sentence.lower().split()
        keywords = [word for word in words if word not in self.stop_words and len(word) > 3]
        return keywords[:3]

    def _is_factual_claim(self, sentence: str) -> bool:
        """Check if sentence contains factual claim"""
        sentence_lower = sentence.lower()
        
        # Factual indicators
        factual_indicators = [
            r'\b(is|was|are|were)\b',
            r'\b(has|have|had)\b',
            r'\b(contains|includes)\b',
            r'\b(located in|based in|found in)\b',
            r'\b(created|founded|established|built)\b',
            r'\b(known for|famous for)\b',
            r'\b(population|area|size)\b',
            r'\b(born|died)\b',
            r'\b(developed|invented|discovered)\b',
            r'\d{4}',
            r'\d+%'
        ]
        
        # Exclusions
        exclusion_patterns = [
            r'^\s*(how|what|why|when|where)',
            r'\?\s*$',
            r'\b(maybe|perhaps|possibly|might)\b',
            r'\b(I think|I believe|in my opinion)\b'
        ]
        
        has_factual = any(re.search(pattern, sentence_lower) for pattern in factual_indicators)
        has_exclusion = any(re.search(pattern, sentence_lower) for pattern in exclusion_patterns)
        
        return has_factual and not has_exclusion and len(sentence.split()) > 3

    def _classify_claim_type(self, sentence: str) -> ClaimType:
        """Classify claim type"""
        sentence_lower = sentence.lower()
        
        if re.search(r'\b(born|died|graduated|married)\b', sentence_lower):
            return ClaimType.BIOGRAPHICAL
        elif re.search(r'\b(19|20)\d{2}\b', sentence):
            return ClaimType.TEMPORAL
        elif re.search(r'\b(located|situated|based) in\b', sentence_lower):
            return ClaimType.GEOGRAPHICAL
        elif re.search(r'\d+%', sentence) or re.search(r'\b(population|area|size)\b', sentence_lower):
            return ClaimType.QUANTITATIVE
        else:
            return ClaimType.ATTRIBUTIVE

    def _assess_claim_confidence(self, sentence: str) -> float:
        """Assess claim confidence"""
        sentence_lower = sentence.lower()
        confidence = 0.5
        
        if re.search(r'\b(always|never|certainly|definitely)\b', sentence_lower):
            confidence += 0.3
        elif re.search(r'\b(probably|likely|possibly|maybe)\b', sentence_lower):
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))

    def find_relevant_wikipedia_pages(self, claim: Dict) -> List[Dict]:
        """Find relevant Wikipedia pages with robust error handling"""
        potential_pages = []
        
        # Generate search queries from entities
        search_queries = []
        for entity in claim['entities']:
            if len(entity) > 3:
                search_queries.append(entity)
        
        # Add some fallback queries based on claim type
        if claim['claim_type'] == ClaimType.BIOGRAPHICAL:
            search_queries.append("biography")
        elif claim['claim_type'] == ClaimType.GEOGRAPHICAL:
            search_queries.append("geography")
        
        for query in search_queries[:3]:  # Limit to 3 queries
            try:
                page = self.wiki.page(query)
                if page.exists() and page.summary:
                    page_data = {
                        'title': page.title,
                        'summary': page.summary,
                        'content': page.text[:5000] if page.text else page.summary,  # Limit content size
                        'url': page.fullurl,
                        'categories': list(page.categories.keys()) if hasattr(page, 'categories') else [],
                        'sections': [s.title for s in page.sections] if hasattr(page, 'sections') else []
                    }
                    potential_pages.append(page_data)
                    logger.info(f"Found Wikipedia page: {page.title}")
            except Exception as e:
                logger.warning(f"Error fetching page for '{query}': {e}")
                continue
        
        return potential_pages[:2]  # Return top 2 pages

    def verify_claim_against_wikipedia(self, claim: Dict, wiki_pages: List[Dict]) -> Dict:
        """Verify claim against Wikipedia pages"""
        if not wiki_pages:
            return self._create_unverifiable_result(claim)
        
        best_result = None
        best_score = -1
        
        for wiki_page in wiki_pages:
            try:
                result = self._verify_single_page(claim, wiki_page)
                score = self._calculate_verification_score(result)
                
                if score > best_score:
                    best_score = score
                    best_result = result
            except Exception as e:
                logger.warning(f"Error verifying against page: {e}")
                continue
        
        return best_result if best_result else self._create_unverifiable_result(claim)

    def _verify_single_page(self, claim: Dict, wiki_page: Dict) -> Dict:
        """Verify claim against single Wikipedia page"""
        claim_text = claim['text']
        wiki_content = (wiki_page.get('content', '') or wiki_page.get('summary', '')).lower()
        
        # Calculate similarity
        similarity_score = SequenceMatcher(None, claim_text.lower(), wiki_content).ratio()
        
        # Find evidence
        supporting_evidence = self._find_supporting_evidence(claim, wiki_content)
        contradicting_evidence = self._find_contradicting_evidence(claim, wiki_content)
        
        # Determine verdict
        verdict, confidence, explanation = self._determine_verdict(
            claim, supporting_evidence, contradicting_evidence, similarity_score
        )
        
        return {
            'claim': claim_text,
            'verdict': verdict,
            'confidence': confidence,
            'supporting_evidence': supporting_evidence,
            'contradicting_evidence': contradicting_evidence,
            'wikipedia_source': wiki_page.get('url', ''),
            'similarity_score': similarity_score,
            'claim_type': claim['claim_type'],
            'explanation': explanation,
            'entity_matches': self._find_entity_matches(claim, wiki_content),
            'confidence_factors': self._get_confidence_factors(supporting_evidence, contradicting_evidence, similarity_score)
        }

    def _find_supporting_evidence(self, claim: Dict, wiki_content: str) -> List[str]:
        """Find supporting evidence"""
        evidence = []
        claim_lower = claim['text'].lower()
        
        # Check for entity matches
        for entity in claim['entities']:
            if entity.lower() in wiki_content:
                evidence.append(f"Entity '{entity}' found in Wikipedia")
        
        # Check for year matches
        years_in_claim = re.findall(r'\b(19|20)\d{2}\b', claim['text'])
        for year in years_in_claim:
            if year in wiki_content:
                evidence.append(f"Year {year} verified")
        
        # Check for keyword matches
        for keyword in claim['keywords']:
            if keyword in wiki_content:
                evidence.append(f"Keyword '{keyword}' found")
        
        return evidence[:3]

    def _find_contradicting_evidence(self, claim: Dict, wiki_content: str) -> List[str]:
        """Find contradicting evidence"""
        contradictions = []
        
        # Check for year contradictions
        claim_years = set(re.findall(r'\b(19|20)\d{2}\b', claim['text']))
        wiki_years = set(re.findall(r'\b(19|20)\d{2}\b', wiki_content))
        
        for claim_year in claim_years:
            if wiki_years:
                closest_year = min(wiki_years, key=lambda x: abs(int(x) - int(claim_year)))
                difference = abs(int(closest_year) - int(claim_year))
                if difference > 5:
                    contradictions.append(f"Year {claim_year} differs from {closest_year} ({difference} years)")
        
        return contradictions[:2]

    def _find_entity_matches(self, claim: Dict, wiki_content: str) -> List[str]:
        """Find entity matches"""
        matches = []
        for entity in claim['entities']:
            if entity.lower() in wiki_content:
                matches.append(f"'{entity}' verified")
        return matches

    def _get_confidence_factors(self, supporting: List[str], contradicting: List[str], similarity: float) -> List[str]:
        """Get confidence factors"""
        factors = []
        
        if supporting:
            factors.append(f"{len(supporting)} supporting evidences")
        if contradicting:
            factors.append(f"{len(contradicting)} contradictions")
        if similarity > 0.7:
            factors.append("High text similarity")
        elif similarity > 0.4:
            factors.append("Moderate text similarity")
        else:
            factors.append("Low text similarity")
            
        return factors

    def _determine_verdict(self, claim: Dict, supporting: List[str], 
                          contradicting: List[str], similarity: float) -> tuple:
        """Determine verdict"""
        support_count = len(supporting)
        contradict_count = len(contradicting)
        
        if support_count >= 2 and contradict_count == 0:
            return Verdict.STRONGLY_SUPPORTED, min(1.0, claim['confidence'] + 0.3), "Strongly supported by multiple evidences"
        elif support_count >= 1 and contradict_count == 0:
            return Verdict.SUPPORTED, min(1.0, claim['confidence'] + 0.2), "Supported by evidence"
        elif contradict_count >= 2:
            return Verdict.STRONGLY_CONTRADICTED, max(0.1, claim['confidence'] - 0.4), "Strongly contradicted by evidence"
        elif contradict_count >= 1 and support_count == 0:
            return Verdict.CONTRADICTED, max(0.1, claim['confidence'] - 0.3), "Contradicted by evidence"
        elif support_count > 0 and contradict_count > 0:
            return Verdict.PARTIALLY_SUPPORTED, claim['confidence'], "Mixed evidence found"
        elif similarity > 0.6:
            return Verdict.SUPPORTED, claim['confidence'] * 0.8, "No direct evidence but high similarity"
        else:
            return Verdict.UNVERIFIABLE, 0.3, "Insufficient evidence for verification"

    def _calculate_verification_score(self, result: Dict) -> float:
        """Calculate verification score"""
        score = result['confidence']
        if result['supporting_evidence']:
            score += len(result['supporting_evidence']) * 0.1
        if result['contradicting_evidence']:
            score -= len(result['contradicting_evidence']) * 0.15
        score += result['similarity_score'] * 0.2
        return max(0.0, min(1.0, score))

    def _create_unverifiable_result(self, claim: Dict) -> Dict:
        """Create unverifiable result"""
        return {
            'claim': claim['text'],
            'verdict': Verdict.UNVERIFIABLE,
            'confidence': 0.1,
            'supporting_evidence': [],
            'contradicting_evidence': [],
            'wikipedia_source': "",
            'similarity_score': 0.0,
            'claim_type': claim['claim_type'],
            'explanation': "No relevant Wikipedia information found",
            'entity_matches': [],
            'confidence_factors': ["No Wikipedia reference available"]
        }

class RobustAIContentFactChecker:
    def __init__(self):
        self.wiki_checker = RobustWikipediaFactChecker()
    
    def analyze_ai_content(self, ai_content: str) -> Dict[str, Any]:
        """Analyze AI content with comprehensive error handling"""
        start_time = time.time()
        
        try:
            logger.info("Starting content analysis...")
            claims = self.wiki_checker.extract_claims_from_ai_content(ai_content)
            logger.info(f"Extracted {len(claims)} claims")
            
            results = []
            total_confidence = 0.0
            
            for i, claim in enumerate(claims, 1):
                logger.info(f"Processing claim {i}/{len(claims)}")
                
                try:
                    wiki_pages = self.wiki_checker.find_relevant_wikipedia_pages(claim)
                    result = self.wiki_checker.verify_claim_against_wikipedia(claim, wiki_pages)
                    results.append(result)
                    total_confidence += result['confidence']
                    
                    # Small delay to be respectful to Wikipedia API
                    time.sleep(0.3)
                    
                except Exception as e:
                    logger.error(f"Error processing claim {i}: {e}")
                    # Create unverifiable result for failed claims
                    results.append(self.wiki_checker._create_unverifiable_result(claim))
                    total_confidence += 0.1
            
            # Calculate metrics
            total_claims = len(claims)
            verified_claims = sum(1 for r in results if r['verdict'] in [Verdict.SUPPORTED, Verdict.STRONGLY_SUPPORTED])
            accuracy_score = verified_claims / total_claims if total_claims > 0 else 0
            avg_confidence = total_confidence / total_claims if total_claims > 0 else 0
            misinformation_risk = self._assess_misinformation_risk(results)
            processing_time = time.time() - start_time
            
            response = {
                'analysis_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'content': ai_content,
                'total_claims': total_claims,
                'verified_claims': verified_claims,
                'accuracy_score': accuracy_score,
                'average_confidence': avg_confidence,
                'misinformation_risk': misinformation_risk,
                'detailed_results': results,
                'summary': self._generate_summary(results, accuracy_score, misinformation_risk),
                'processing_time': processing_time,
                'claim_breakdown': self._generate_claim_breakdown(results)
            }
            
            logger.info("Analysis completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Critical error in analysis: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    def _assess_misinformation_risk(self, results: List[Dict]) -> str:
        """Assess misinformation risk"""
        if not results:
            return "UNKNOWN"
            
        risk_score = 0
        total = len(results)
        
        for result in results:
            if result['verdict'] in [Verdict.STRONGLY_CONTRADICTED, Verdict.CONTRADICTED]:
                risk_score += 2
            elif result['verdict'] == Verdict.PARTIALLY_SUPPORTED:
                risk_score += 1
            elif result['verdict'] in [Verdict.STRONGLY_SUPPORTED, Verdict.SUPPORTED]:
                risk_score -= 1
        
        normalized_risk = risk_score / total
        
        if normalized_risk >= 1.0:
            return "VERY_HIGH"
        elif normalized_risk >= 0.5:
            return "HIGH"
        elif normalized_risk >= 0.0:
            return "MEDIUM"
        elif normalized_risk >= -0.5:
            return "LOW"
        else:
            return "VERY_LOW"

    def _generate_summary(self, results: List[Dict], accuracy: float, risk: str) -> str:
        """Generate summary"""
        verdict_counts = Counter([r['verdict'] for r in results])
        
        summary = [
            "ECHOLENS FACT-CHECKING ANALYSIS REPORT",
            "=" * 45,
            f"Overall Accuracy: {accuracy:.1%}",
            f"Misinformation Risk: {risk}",
            f"Total Claims Analyzed: {len(results)}",
            "",
            "Verdict Distribution:"
        ]
        
        for verdict, count in verdict_counts.most_common():
            percentage = (count / len(results)) * 100
            icon = "✅" if verdict in [Verdict.SUPPORTED, Verdict.STRONGLY_SUPPORTED] else "❌" if verdict in [Verdict.CONTRADICTED, Verdict.STRONGLY_CONTRADICTED] else "⚠️"
            summary.append(f"  {icon} {verdict.value}: {count} ({percentage:.1f}%)")
        
        return "\n".join(summary)

    def _generate_claim_breakdown(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate claim breakdown"""
        type_counts = Counter([r['claim_type'] for r in results])
        return {
            'claim_types': {k.value: v for k, v in type_counts.items()},
            'total_claims': len(results)
        }

# Initialize the fact checker
fact_checker = RobustAIContentFactChecker()

# Session management
def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create new one"""
    if session_id:
        session = db_manager.get_user_session(session_id)
        if session:
            return session_id
    
    return db_manager.create_user_session()

# FastAPI routes
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/api/fact-check", response_model=FactCheckResponse)
async def fact_check_content(request: FactCheckRequest):
    try:
        logger.info("Received fact-check request")
        
        # Get or create session
        session_id = get_or_create_session(request.session_id)
        
        # Perform analysis
        analysis = fact_checker.analyze_ai_content(request.content)
        
        # Add session ID to response
        analysis['session_id'] = session_id
        
        # Save to database
        db_manager.save_analysis(analysis, session_id)
        db_manager.update_session_activity(session_id)
        
        return analysis
    except Exception as e:
        logger.error(f"Fact-check error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/session/new")
async def create_new_session():
    """Create a new user session"""
    session_id = db_manager.create_user_session()
    return {"session_id": session_id, "message": "New session created"}

@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get analysis history for a session"""
    history = db_manager.get_analysis_history(session_id)
    return {"session_id": session_id, "history": history}

@app.get("/api/analysis/{analysis_id}")
async def get_analysis_by_id(analysis_id: str):
    """Get specific analysis by ID"""
    analysis = db_manager.get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis

@app.get("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get aggregated analytics data"""
    analytics = db_manager.get_analytics()
    return analytics

@app.post("/api/cleanup")
async def cleanup_old_data():
    """Clean up old sessions and data (admin endpoint)"""
    deleted_count = db_manager.cleanup_old_sessions()
    return {"message": f"Cleaned up {deleted_count} old sessions"}

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "service": "EchoLens AI Fact Checker",
        "version": "3.0.0"
    }

@app.get("/api/stats")
async def get_stats():
    analytics = db_manager.get_analytics()
    return {
        "service": "EchoLens AI Fact Checker",
        "version": "3.0.0",
        "status": "operational",
        "total_analyses": analytics['total_analyses'],
        "average_accuracy": f"{analytics['average_accuracy']:.1%}",
        "database": "SQLite"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database and cleanup old data on startup"""
    logger.info("EchoLens starting up...")
    db_manager.cleanup_old_sessions()
    logger.info("Startup completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)