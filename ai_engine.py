import wikipediaapi
import requests
import re
import spacy
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from difflib import SequenceMatcher
from collections import Counter
import time

# Try to load spaCy model, fallback to simpler NLP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️  spaCy model not found. Installing simpler NLTK fallback...")
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nlp = None

class ClaimType(Enum):
    TEMPORAL = "temporal"  # Dates, years, timelines
    QUANTITATIVE = "quantitative"  # Numbers, statistics
    RELATIONAL = "relational"  # Relationships, connections
    ATTRIBUTIVE = "attributive"  # Properties, characteristics
    COMPARATIVE = "comparative"  # Comparisons
    EXISTENTIAL = "existential"  # Existence claims

class Verdict(Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNVERIFIABLE = "UNVERIFIABLE"
    MISLEADING = "MISLEADING"
    OUTDATED = "OUTDATED"

@dataclass
class FactClaim:
    text: str
    claim_type: ClaimType
    confidence: float
    entities: List[str]
    source_text: str

@dataclass
class VerificationResult:
    claim: str
    verdict: Verdict
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    wikipedia_source: str
    similarity_score: float
    claim_type: ClaimType
    explanation: str

class WikipediaFactChecker:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='AIFactChecker/3.0 (https://example.com)',
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        
    def extract_claims_from_ai_content(self, ai_content: str) -> List[FactClaim]:
        """Extract factual claims from AI-generated content"""
        claims = []
        
        # Split into sentences for analysis
        sentences = self._split_into_sentences(ai_content)
        
        for sentence in sentences:
            if self._is_factual_claim(sentence):
                claim_type = self._classify_claim_type(sentence)
                entities = self._extract_entities(sentence)
                confidence = self._assess_claim_confidence(sentence)
                
                claims.append(FactClaim(
                    text=sentence.strip(),
                    claim_type=claim_type,
                    confidence=confidence,
                    entities=entities,
                    source_text=ai_content
                ))
        
        return claims
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy or fallback"""
        if nlp:
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            # Simple regex-based sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """Determine if a sentence contains a factual claim"""
        sentence_lower = sentence.lower()
        
        # Patterns that indicate factual claims
        factual_indicators = [
            r'\b(is|was|are|were)\b',
            r'\b(has|have|had)\b',
            r'\b(contains|includes)\b',
            r'\b(located in|based in|found in)\b',
            r'\b(created|founded|established|built)\b',
            r'\b(known for|famous for)\b',
            r'\b(population of|area of|size of)\b',
            r'\b(born in|died in)\b',
            r'\b(developed|invented|discovered)\b',
            r'\d{4}',  # Years
            r'\d+%',   # Percentages
            r'\b(over|more than|less than) \d+',  # Quantitative comparisons
        ]
        
        # Exclude questions and speculative language
        exclusion_patterns = [
            r'^how',
            r'^what',
            r'^why',
            r'^when',
            r'^where',
            r'\?$',
            r'\b(maybe|perhaps|possibly|might)\b',
            r'\b(I think|I believe|in my opinion)\b'
        ]
        
        # Check for factual indicators
        has_factual = any(re.search(pattern, sentence_lower) for pattern in factual_indicators)
        
        # Check for exclusions
        has_exclusion = any(re.search(pattern, sentence_lower) for pattern in exclusion_patterns)
        
        return has_factual and not has_exclusion and len(sentence.split()) > 3
    
    def _classify_claim_type(self, sentence: str) -> ClaimType:
        """Classify the type of factual claim"""
        sentence_lower = sentence.lower()
        
        # Temporal claims (dates, years, timelines)
        if (re.search(r'\b(19|20)\d{2}\b', sentence) or 
            re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', sentence_lower) or
            re.search(r'\b(created|founded|established|born|died|invented)\b', sentence_lower)):
            return ClaimType.TEMPORAL
        
        # Quantitative claims (numbers, statistics)
        if (re.search(r'\d+%', sentence) or
            re.search(r'\b(population|area|size|height|weight|distance)\b', sentence_lower) or
            re.search(r'\b(\d+ million|\d+ billion|\d+ thousand)\b', sentence_lower)):
            return ClaimType.QUANTITATIVE
        
        # Relational claims (relationships, connections)
        if (re.search(r'\b(son of|daughter of|father of|mother of|wife of|husband of)\b', sentence_lower) or
            re.search(r'\b(worked for|employed by|studied at|graduated from)\b', sentence_lower)):
            return ClaimType.RELATIONAL
        
        # Comparative claims
        if (re.search(r'\b(larger than|smaller than|older than|younger than|more than|less than)\b', sentence_lower) or
            re.search(r'\b(better|worse|faster|slower)\b', sentence_lower)):
            return ClaimType.COMPARATIVE
        
        # Existential claims
        if re.search(r'\b(exists|located in|based in|found in)\b', sentence_lower):
            return ClaimType.EXISTENTIAL
        
        # Default to attributive (properties, characteristics)
        return ClaimType.ATTRIBUTIVE
    
    def _extract_entities(self, sentence: str) -> List[str]:
        """Extract main entities from sentence"""
        if nlp:
            doc = nlp(sentence)
            entities = [ent.text for ent in doc.ents]
            return entities
        else:
            # Simple noun phrase extraction as fallback
            words = sentence.split()
            # Heuristic: capitalize proper nouns and longer words
            entities = [word for word in words if word.istitle() and len(word) > 3]
            return entities[:3]  # Return top 3
    
    def _assess_claim_confidence(self, sentence: str) -> float:
        """Assess initial confidence in the claim based on language"""
        sentence_lower = sentence.lower()
        
        confidence = 0.5  # Base confidence
        
        # High confidence indicators
        high_confidence_indicators = [
            r'\b(always|never|certainly|definitely)\b',
            r'\d{4}',  # Specific years
            r'\b(exactly|precisely)\b',
        ]
        
        # Low confidence indicators
        low_confidence_indicators = [
            r'\b(probably|likely|possibly|maybe)\b',
            r'\b(approximately|about|around)\b',
            r'\b(some|many|few)\b',
        ]
        
        for pattern in high_confidence_indicators:
            if re.search(pattern, sentence_lower):
                confidence += 0.3
                break
        
        for pattern in low_confidence_indicators:
            if re.search(pattern, sentence_lower):
                confidence -= 0.2
                break
        
        return max(0.1, min(1.0, confidence))
    
    def find_relevant_wikipedia_page(self, claim: FactClaim) -> Optional[Dict]:
        """Find the most relevant Wikipedia page for a claim"""
        entities = claim.entities
        
        for entity in entities:
            if len(entity) > 3:  # Avoid very short entities
                page = self.wiki.page(entity)
                if page.exists():
                    return {
                        'title': page.title,
                        'summary': page.summary,
                        'content': page.text,
                        'url': page.fullurl,
                        'categories': list(page.categories.keys()),
                        'sections': [s.title for s in page.sections]
                    }
        
        # Fallback: search using the entire claim
        search_terms = ' '.join(entities) if entities else claim.text[:50]
        page = self.wiki.page(search_terms)
        if page.exists():
            return {
                'title': page.title,
                'summary': page.summary,
                'content': page.text,
                'url': page.fullurl,
                'categories': list(page.categories.keys()),
                'sections': [s.title for s in page.sections]
            }
        
        return None
    
    def verify_claim_against_wikipedia(self, claim: FactClaim, wiki_content: Dict) -> VerificationResult:
        """Verify a single claim against Wikipedia content"""
        claim_text = claim.text
        wiki_text = wiki_content['content'].lower()
        wiki_summary = wiki_content['summary'].lower()
        
        # Calculate text similarity
        similarity = self._calculate_similarity(claim_text.lower(), wiki_text)
        
        # Check for supporting evidence
        supporting_evidence = self._find_supporting_evidence(claim_text, wiki_text, wiki_content)
        
        # Check for contradicting evidence
        contradicting_evidence = self._find_contradicting_evidence(claim_text, wiki_text, wiki_content)
        
        # Determine verdict
        verdict, confidence, explanation = self._determine_verdict(
            claim, supporting_evidence, contradicting_evidence, similarity
        )
        
        return VerificationResult(
            claim=claim_text,
            verdict=verdict,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            wikipedia_source=wiki_content['url'],
            similarity_score=similarity,
            claim_type=claim.claim_type,
            explanation=explanation
        )
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _find_supporting_evidence(self, claim: str, wiki_text: str, wiki_content: Dict) -> List[str]:
        """Find evidence in Wikipedia that supports the claim"""
        evidence = []
        claim_lower = claim.lower()
        claim_words = set(claim_lower.split())
        
        # Look for exact matches of key phrases
        key_phrases = self._extract_key_phrases(claim)
        for phrase in key_phrases:
            if phrase.lower() in wiki_text:
                evidence.append(f"Found supporting phrase: '{phrase}'")
        
        # Check for semantic matches in important sections
        important_sections = ['history', 'biography', 'overview', 'early life', 'career']
        for section_title in important_sections:
            if section_title in wiki_text:
                # In a real implementation, you'd extract section content
                evidence.append(f"Related information found in '{section_title}' section")
        
        # Check for numerical consistency
        numbers_in_claim = re.findall(r'\d+\.?\d*', claim)
        numbers_in_wiki = re.findall(r'\d+\.?\d*', wiki_text)
        
        for num in numbers_in_claim:
            if num in numbers_in_wiki:
                evidence.append(f"Numerical value {num} is consistent")
        
        return evidence[:5]  # Return top 5 evidence points
    
    def _find_contradicting_evidence(self, claim: str, wiki_text: str, wiki_content: Dict) -> List[str]:
        """Find evidence in Wikipedia that contradicts the claim"""
        contradictions = []
        claim_lower = claim.lower()
        
        # Check for temporal contradictions
        temporal_contradictions = self._check_temporal_contradictions(claim, wiki_text)
        contradictions.extend(temporal_contradictions)
        
        # Check for numerical contradictions
        numerical_contradictions = self._check_numerical_contradictions(claim, wiki_text)
        contradictions.extend(numerical_contradictions)
        
        # Check for relational contradictions
        relational_contradictions = self._check_relational_contradictions(claim, wiki_text)
        contradictions.extend(relational_contradictions)
        
        return contradictions[:5]  # Return top 5 contradictions
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        if nlp:
            doc = nlp(text)
            phrases = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) > 1:  # Only multi-word phrases
                    phrases.append(chunk.text)
            return phrases
        else:
            # Simple fallback: extract phrases around verbs
            words = text.split()
            phrases = []
            for i, word in enumerate(words):
                if i < len(words) - 1 and len(word) > 3 and len(words[i+1]) > 3:
                    phrases.append(f"{word} {words[i+1]}")
            return phrases[:3]
    
    def _check_temporal_contradictions(self, claim: str, wiki_text: str) -> List[str]:
        """Check for temporal contradictions"""
        contradictions = []
        
        # Extract years from claim and wiki
        claim_years = set(re.findall(r'\b(19|20)\d{2}\b', claim))
        wiki_years = set(re.findall(r'\b(19|20)\d{2}\b', wiki_text))
        
        for claim_year in claim_years:
            if wiki_years:
                closest_year = min(wiki_years, key=lambda x: abs(int(x) - int(claim_year)))
                difference = abs(int(closest_year) - int(claim_year))
                if difference > 5:  # Significant temporal difference
                    contradictions.append(f"Year {claim_year} differs from Wikipedia's {closest_year} (difference: {difference} years)")
        
        return contradictions
    
    def _check_numerical_contradictions(self, claim: str, wiki_text: str) -> List[str]:
        """Check for numerical contradictions"""
        contradictions = []
        
        # Extract numbers from claim and wiki
        claim_numbers = set(re.findall(r'\b(\d+\.?\d*)\b', claim))
        wiki_numbers = set(re.findall(r'\b(\d+\.?\d*)\b', wiki_text))
        
        for num in claim_numbers:
            if float(num) > 100:  # Only check significant numbers
                if wiki_numbers:
                    closest = min(wiki_numbers, key=lambda x: abs(float(x) - float(num)))
                    difference = abs(float(closest) - float(num))
                    if difference > float(num) * 0.5 > 50% difference:
                        contradictions.append(f"Number {num} significantly differs from Wikipedia's {closest}")
        
        return contradictions
    
    def _check_relational_contradictions(self, claim: str, wiki_text: str) -> List[str]:
        """Check for relational contradictions"""
        contradictions = []
        claim_lower = claim.lower()
        wiki_lower = wiki_text.lower()
        
        # Check for relationship contradictions
        relationships = {
            'son of': 'daughter of',
            'father of': 'mother of', 
            'husband of': 'wife of',
            'born in': 'died in',
            'created by': 'created',
        }
        
        for rel1, rel2 in relationships.items():
            if rel1 in claim_lower and rel2 in wiki_lower:
                contradictions.append(f"Relationship '{rel1}' in claim contradicts Wikipedia's '{rel2}'")
        
        return contradictions
    
    def _determine_verdict(self, claim: FactClaim, supporting: List[str], 
                          contradicting: List[str], similarity: float) -> Tuple[Verdict, float, str]:
        """Determine final verdict based on evidence"""
        
        support_strength = len(supporting)
        contradiction_strength = len(contradicting)
        
        if support_strength > 0 and contradiction_strength == 0:
            explanation = f"Claim is supported by {support_strength} evidence points"
            return Verdict.SUPPORTED, min(1.0, claim.confidence + 0.3), explanation
            
        elif contradiction_strength > 0 and support_strength == 0:
            explanation = f"Claim contradicted by {contradiction_strength} evidence points"
            return Verdict.CONTRADICTED, max(0.1, claim.confidence - 0.4), explanation
            
        elif support_strength > 0 and contradiction_strength > 0:
            explanation = f"Mixed evidence: {support_strength} supporting, {contradiction_strength} contradicting"
            return Verdict.PARTIALLY_SUPPORTED, claim.confidence, explanation
            
        elif similarity > 0.7:
            explanation = "No direct evidence found, but high semantic similarity"
            return Verdict.SUPPORTED, claim.confidence * 0.8, explanation
            
        else:
            explanation = "Insufficient evidence to verify this claim"
            return Verdict.UNVERIFIABLE, 0.3, explanation

class AIContentFactChecker:
    def __init__(self):
        self.wiki_checker = WikipediaFactChecker()
    
    def analyze_ai_content(self, ai_content: str) -> Dict[str, Any]:
        """Comprehensive analysis of AI-generated content against Wikipedia"""
        print("🔍 Extracting factual claims from AI content...")
        claims = self.wiki_checker.extract_claims_from_ai_content(ai_content)
        
        print(f"📋 Found {len(claims)} factual claims to verify...")
        
        results = []
        overall_confidence = 0.0
        verified_claims = 0
        
        for i, claim in enumerate(claims, 1):
            print(f"  Verifying claim {i}/{len(claims)}: {claim.text[:80]}...")
            
            wiki_page = self.wiki_checker.find_relevant_wikipedia_page(claim)
            
            if wiki_page:
                result = self.wiki_checker.verify_claim_against_wikipedia(claim, wiki_page)
                results.append(result)
                
                if result.verdict == Verdict.SUPPORTED:
                    verified_claims += 1
                overall_confidence += result.confidence
            else:
                # No Wikipedia page found
                results.append(VerificationResult(
                    claim=claim.text,
                    verdict=Verdict.UNVERIFIABLE,
                    confidence=0.1,
                    supporting_evidence=[],
                    contradicting_evidence=[],
                    wikipedia_source="",
                    similarity_score=0.0,
                    claim_type=claim.claim_type,
                    explanation="No relevant Wikipedia page found for verification"
                ))
            
            # Be respectful to Wikipedia API
            time.sleep(0.5)
        
        # Calculate overall metrics
        total_claims = len(claims)
        accuracy_score = verified_claims / total_claims if total_claims > 0 else 0
        avg_confidence = overall_confidence / total_claims if total_claims > 0 else 0
        
        # Assess misinformation risk
        misinformation_risk = self._assess_misinformation_risk(results)
        
        return {
            'ai_content': ai_content,
            'total_claims': total_claims,
            'verified_claims': verified_claims,
            'accuracy_score': accuracy_score,
            'average_confidence': avg_confidence,
            'misinformation_risk': misinformation_risk,
            'detailed_results': results,
            'summary': self._generate_summary(results, accuracy_score, misinformation_risk)
        }
    
    def _assess_misinformation_risk(self, results: List[VerificationResult]) -> str:
        """Assess overall risk of misinformation"""
        contradicted = sum(1 for r in results if r.verdict == Verdict.CONTRADICTED)
        total = len(results)
        
        contradiction_ratio = contradicted / total if total > 0 else 0
        
        if contradiction_ratio > 0.5:
            return "HIGH"
        elif contradiction_ratio > 0.2:
            return "MEDIUM"
        elif contradiction_ratio > 0:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _generate_summary(self, results: List[VerificationResult], accuracy: float, risk: str) -> str:
        """Generate human-readable summary"""
        verdict_counts = Counter([r.verdict for r in results])
        
        summary_parts = [
            f"Fact-Checking Summary:",
            f"• Total claims analyzed: {len(results)}",
            f"• Accuracy score: {accuracy:.1%}",
            f"• Misinformation risk: {risk}",
            f"• Verdict distribution:"
        ]
        
        for verdict, count in verdict_counts.most_common():
            summary_parts.append(f"  - {verdict.value}: {count} claims")
        
        if accuracy < 0.5:
            summary_parts.append("\n⚠️  Warning: This AI content contains significant factual inaccuracies")
        elif accuracy > 0.8:
            summary_parts.append("\n✅ This AI content appears mostly factually accurate")
        else:
            summary_parts.append("\n⚠️  This AI content contains some factual inaccuracies")
        
        return "\n".join(summary_parts)
    
    def print_detailed_report(self, analysis: Dict[str, Any]):
        """Print a detailed fact-checking report"""
        print("\n" + "="*80)
        print("🤖 AI CONTENT FACT-CHECKING REPORT")
        print("="*80)
        print(f"📊 Overall Accuracy: {analysis['accuracy_score']:.1%}")
        print(f"🚨 Misinformation Risk: {analysis['misinformation_risk']}")
        print(f"📝 Total Claims Analyzed: {analysis['total_claims']}")
        print(f"✅ Verified Claims: {analysis['verified_claims']}")
        print("\n" + "-"*80)
        
        # Print detailed results for each claim
        for i, result in enumerate(analysis['detailed_results'], 1):
            print(f"\n{i}. {result.claim}")
            print(f"   Type: {result.claim_type.value}")
            print(f"   Verdict: {result.verdict.value}")
            print(f"   Confidence: {result.confidence:.1%}")
            print(f"   Explanation: {result.explanation}")
            
            if result.supporting_evidence:
                print("   📈 Supporting Evidence:")
                for evidence in result.supporting_evidence[:2]:
                    print(f"     ✓ {evidence}")
            
            if result.contradicting_evidence:
                print("   📉 Contradicting Evidence:")
                for evidence in result.contradicting_evidence[:2]:
                    print(f"     ✗ {evidence}")
            
            if result.wikipedia_source:
                print(f"   🔗 Source: {result.wikipedia_source}")
            
            print("   " + "-"*40)
        
        print(f"\n{analysis['summary']}")
        print("="*80)

def main():
    checker = AIContentFactChecker()
    
    print("🚀 AI CONTENT FACT-CHECKER")
    print("Eliminating misinformation by verifying against Wikipedia")
    print("="*70)
    
    # Example AI-generated content for demonstration
    sample_ai_content = """
    Barack Obama was the 44th president of the United States. He was born in 1961 in Hawaii. 
    Obama served as president from 2009 to 2017. He graduated from Harvard Law School and 
    previously worked as a community organizer in Chicago. Obama was awarded the Nobel Peace 
    Prize in 2009. He is married to Michelle Obama and they have two daughters. 
    Before becoming president, Obama was a United States Senator from Illinois.
    """
    
    print("Sample AI content to check:")
    print('"' + sample_ai_content + '"')
    print("\n" + "="*70)
    
    while True:
        print("\nOptions:")
        print("1. Use sample content above")
        print("2. Enter your own AI-generated content")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            content = sample_ai_content
        elif choice == '2':
            print("\nEnter/Paste the AI-generated content to fact-check:")
            content = ""
            while True:
                line = input()
                if line.strip() == "END":
                    break
                content += line + "\n"
        elif choice == '3':
            print("👋 Thank you for using the AI Content Fact-Checker!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
            continue
        
        if content.strip():
            print("\n" + "⏳ Fact-checking in progress..." + "\n")
            analysis = checker.analyze_ai_content(content)
            checker.print_detailed_report(analysis)
        else:
            print("❌ No content provided. Please try again.")

if __name__ == "__main__":
    main()