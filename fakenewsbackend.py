import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
from collections import Counter
from typing import Dict, List
import math

nltk.download('vader_lexicon')
nltk.download('stopwords')

class LinguisticAnalyzer:
    """Analyze linguistic patterns common in fake news"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        
        # Fake news linguistic markers
        self.exaggeration_words = [
            'unbelievable', 'incredible', 'shocking', 'amazing', 'mind-blowing',
            'explosive', 'devastating', 'sensational', 'unprecedented'
        ]
        
        self.uncertainty_words = [
            'maybe', 'perhaps', 'possibly', 'allegedly', 'reportedly',
            'rumor', 'sources say', 'insiders claim'
        ]
        
        self.emotional_words = [
            'outrage', 'furious', 'terrified', 'heartbreaking', 'tragic',
            'horrifying', 'devastated', 'disgusting'
        ]
        
        self.fake_indicators = [
            'you won\'t believe', 'doctors hate', 'one weird trick',
            'miracle cure', 'secret revealed', 'they don\'t want you to know'
        ]
    
    def analyze(self, text: str) -> Dict:
        """Complete linguistic analysis"""
        doc = self.nlp(text)
        
        return {
            'sentiment_analysis': self.analyze_sentiment(text),
            'exaggeration_score': self.analyze_exaggeration(text),
            'subjectivity_score': self.analyze_subjectivity(text),
            'readability_score': self.analyze_readability(text),
            'emotion_intensity': self.analyze_emotional_intensity(text),
            'linguistic_complexity': self.analyze_complexity(doc),
            'fake_markers': self.detect_fake_markers(text),
            'overall_linguistic_risk': 0
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment patterns"""
        scores = self.sia.polarity_scores(text)
        blob = TextBlob(text)
        
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'subjectivity': blob.sentiment.subjectivity,
            'polarity': blob.sentiment.polarity,
            'sentiment_risk': self._calculate_sentiment_risk(scores)
        }
    
    def analyze_exaggeration(self, text: str) -> Dict:
        """Detect exaggeration and hyperbolic language"""
        text_lower = text.lower()
        
        # Count exaggeration words
        exaggeration_count = sum(1 for word in self.exaggeration_words 
                                if word in text_lower)
        
        # Detect excessive punctuation
        exclamation_count = text.count('!')
        question_marks = text.count('?')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Detect ALL CAPS words (emphasis)
        all_caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
        
        score = (
            (exaggeration_count * 0.3) +
            (min(exclamation_count / 10, 0.3)) +
            (caps_ratio * 0.2) +
            (len(all_caps_words) * 0.1)
        )
        
        return {
            'exaggeration_score': min(score, 1.0),
            'exaggeration_count': exaggeration_count,
            'exclamation_count': exclamation_count,
            'caps_ratio': caps_ratio,
            'all_caps_words': all_caps_words[:5]
        }
    
    def analyze_subjectivity(self, text: str) -> float:
        """Measure subjectivity vs objectivity"""
        blob = TextBlob(text)
        return blob.sentiment.subjectivity
    
    def analyze_readability(self, text: str) -> Dict:
        """Calculate readability scores"""
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        if not sentences or not words:
            return {'flesch_kincaid': 0, 'reading_level': 'Unknown'}
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Average syllables per word (simplified)
        syllables = sum(self._count_syllables(word) for word in words)
        avg_syllables = syllables / len(words)
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid = 0.39 * (len(words) / len(sentences)) + \
                        11.8 * (syllables / len(words)) - 15.59
        
        return {
            'flesch_kincaid': round(flesch_kincaid, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_syllables_per_word': round(avg_syllables, 2),
            'reading_level': self._get_reading_level(flesch_kincaid)
        }
    
    def analyze_emotional_intensity(self, text: str) -> float:
        """Measure emotional intensity of the text"""
        text_lower = text.lower()
        
        # Count emotional words
        emotional_count = sum(1 for word in self.emotional_words 
                            if word in text_lower)
        
        # Check for emotional punctuation
        emotional_punctuation = text.count('!') + text.count('?') * 0.5
        
        # Calculate intensity
        intensity = (emotional_count / max(len(text.split()), 1)) * 100 + \
                   (emotional_punctuation / max(len(text), 1)) * 100
        
        return min(intensity / 10, 1.0)
    
    def analyze_complexity(self, doc) -> Dict:
        """Analyze linguistic complexity"""
        # Part-of-speech distribution
        pos_counts = Counter(token.pos_ for token in doc)
        
        # Named entity recognition
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Lexical diversity (Type-Token Ratio)
        unique_words = set(token.text.lower() for token in doc if not token.is_punct)
        total_words = len([token for token in doc if not token.is_punct])
        lexical_diversity = len(unique_words) / max(total_words, 1)
        
        return {
            'pos_distribution': dict(pos_counts),
            'named_entities': entities[:10],
            'lexical_diversity': lexical_diversity,
            'unique_entities': len(set(entities))
        }
    
    def detect_fake_markers(self, text: str) -> List[str]:
        """Detect common fake news linguistic markers"""
        text_lower = text.lower()
        detected_markers = []
        
        for marker in self.fake_indicators:
            if marker in text_lower:
                detected_markers.append(marker)
        
        return detected_markers
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counter"""
        word = word.lower()
        count = 0
        vowels = 'aeiou'
        
        if not word:
            return 0
        
        if word[0] in vowels:
            count += 1
        
        for index in range(1, len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count += 1
        
        if word.endswith('e'):
            count -= 1
        
        if count == 0:
            count = 1
        
        return count
    
    def _get_reading_level(self, score: float) -> str:
        """Interpret readability score"""
        if score < 6:
            return "Very Easy (5th grade)"
        elif score < 8:
            return "Easy (6-7th grade)"
        elif score < 10:
            return "Fairly Easy (8-9th grade)"
        elif score < 12:
            return "Standard (10-12th grade)"
        else:
            return "Difficult (College level)"
    
    def _calculate_sentiment_risk(self, scores: Dict) -> float:
        """Calculate risk based on sentiment"""
        # High negative or high positive can be risky
        if scores['compound'] > 0.8 or scores['compound'] < -0.8:
            return 0.8
        elif scores['compound'] > 0.6 or scores['compound'] < -0.6:
            return 0.6
        elif scores['compound'] > 0.4 or scores['compound'] < -0.4:
            return 0.4
        else:
            return 0.2