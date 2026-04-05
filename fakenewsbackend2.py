import requests
from urllib.parse import urlparse
import whois
from datetime import datetime
import dns.resolver
from typing import Dict, List, Optional
import hashlib
import json

class SourceVerifier:
    """Verify news source credibility"""
    
    def __init__(self):
        self.trusted_domains = self._load_trusted_domains()
        self.fake_domains = self._load_fake_domains()
        self.fact_check_apis = [
            'https://api.factchecktools.googleapis.com/v1alpha1/claims:search',
            'https://api.politifact.com/api/factchecks/'
        ]
    
    def _load_trusted_domains(self) -> set:
        """Load list of trusted news domains"""
        return {
            'reuters.com', 'apnews.com', 'bbc.com', 'nytimes.com', 
            'wsj.com', 'washingtonpost.com', 'economist.com', 
            'bloomberg.com', 'npr.org', 'theguardian.com'
        }
    
    def _load_fake_domains(self) -> set:
        """Load list of known fake news domains"""
        return {
            'theonion.com',  # Satire
            'dailycurrant.com',
            'worldnewsdailyreport.com'
        }
    
    def verify_source(self, url: str) -> Dict:
        """Comprehensive source verification"""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower().replace('www.', '')
        
        # Domain age check
        domain_age = self._check_domain_age(domain)
        
        # SSL certificate check
        ssl_info = self._check_ssl(url)
        
        # DNS records check
        dns_info = self._check_dns(domain)
        
        # Trust score calculation
        trust_score = self._calculate_trust_score(domain, domain_age, ssl_info)
        
        return {
            'domain': domain,
            'is_trusted': domain in self.trusted_domains,
            'is_fake_source': domain in self.fake_domains,
            'domain_age_days': domain_age,
            'ssl_valid': ssl_info.get('valid', False),
            'ssl_issuer': ssl_info.get('issuer', ''),
            'dns_records': dns_info,
            'trust_score': trust_score,
            'recommendation': self._get_recommendation(trust_score)
        }
    
    def _check_domain_age(self, domain: str) -> int:
        """Check domain age in days"""
        try:
            w = whois.whois(domain)
            if w.creation_date:
                if isinstance(w.creation_date, list):
                    creation_date = w.creation_date[0]
                else:
                    creation_date = w.creation_date
                
                age = (datetime.now() - creation_date).days
                return age
        except Exception as e:
            print(f"Whois lookup failed: {e}")
        return 0
    
    def _check_ssl(self, url: str) -> Dict:
        """Check SSL certificate validity"""
        import ssl
        import socket
        
        try:
            hostname = urlparse(url).netloc
            context = ssl.create_default_context()
            with socket.create_connection((hostname, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    return {
                        'valid': True,
                        'issuer': dict(x[0] for x in cert['issuer']).get('organizationName', ''),
                        'expiry': cert['notAfter']
                    }
        except Exception:
            return {'valid': False, 'issuer': 'Unknown'}
    
    def _check_dns(self, domain: str) -> Dict:
        """Check DNS records"""
        records = {}
        record_types = ['A', 'MX', 'TXT']
        
        for record_type in record_types:
            try:
                answers = dns.resolver.resolve(domain, record_type)
                records[record_type] = [str(answer) for answer in answers]
            except Exception:
                records[record_type] = []
        
        return records
    
    def _calculate_trust_score(self, domain: str, domain_age: int, ssl_info: Dict) -> float:
        """Calculate overall trust score"""
        score = 0.5  # Neutral starting point
        
        # Domain in trusted list
        if domain in self.trusted_domains:
            score += 0.4
        
        # Domain in fake list
        if domain in self.fake_domains:
            score -= 0.5
        
        # Domain age factor
        if domain_age > 3650:  # 10+ years
            score += 0.2
        elif domain_age > 1825:  # 5+ years
            score += 0.1
        elif domain_age < 365:  # Less than 1 year
            score -= 0.2
        
        # SSL factor
        if ssl_info.get('valid', False):
            score += 0.1
        
        return max(0, min(score, 1.0))
    
    def _get_recommendation(self, trust_score: float) -> str:
        """Get recommendation based on trust score"""
        if trust_score > 0.8:
            return "Highly Trustworthy Source"
        elif trust_score > 0.6:
            return "Generally Trustworthy"
        elif trust_score > 0.4:
            return "Needs Verification"
        elif trust_score > 0.2:
            return "Suspicious Source"
        else:
            return "Unreliable Source - Avoid"
    
    def check_fact_check_apis(self, claim: str) -> List[Dict]:
        """Query fact-checking APIs"""
        results = []
        
        # Google Fact Check API example
        api_key = "YOUR_API_KEY"
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {'query': claim, 'key': api_key}
        
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                for claim in data.get('claims', []):
                    results.append({
                        'text': claim.get('text', ''),
                        'claimant': claim.get('claimant', ''),
                        'rating': claim.get('claimReview', [{}])[0].get('textualRating', ''),
                        'url': claim.get('claimReview', [{}])[0].get('url', '')
                    })
        except Exception as e:
            print(f"Fact-check API error: {e}")
        
        return results