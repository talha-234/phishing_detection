import re
from urllib.parse import urlparse
import numpy as np
import tldextract
from typing import Dict, Any, List
import pandas as pd


class URLFeatureExtractor:
    def __init__(self):
        self.tld_extractor = tldextract.TLDExtract(suffix_list_urls=[])

    def extract(self, url: str) -> Dict[str, Any]:
        if not isinstance(url, str) or not url.strip():
            return {name: 0 for name in self.get_feature_names()}

        url = url.strip()
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            parsed = urlparse(url)
            ext = self.tld_extractor(url)

            features = {
                'url_length': len(url),
                'num_dots': url.count('.'),
                'num_slashes': url.count('/'),
                'num_query_params': len(parsed.query.split('&')) if parsed.query else 0,
                'has_ip_address': int(bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', ext.domain))),
                'url_entropy': self._shannon_entropy(url),
                'num_subdomains': max(0, len(ext.subdomain.split('.')) - 1 if ext.subdomain else 0),
                'is_https': int(parsed.scheme.lower() == 'https'),
                'domain_length': len(ext.domain),
                'has_suspicious_tld': int(ext.suffix.lower() in {'tk', 'ml', 'ga', 'cf', 'gq', 'xyz'}),
                'path_length': len(parsed.path),
                'has_login_keyword': int(bool(re.search(r'(?:login|sign-in|signin|account|password|verify|update|secure)', url.lower()))),
                'percent_encoded_chars': len(re.findall(r'%[0-9a-fA-F]{2}', url)),
            }
            return features
        except Exception:
            return {name: 0 for name in self.get_feature_names()}

    def _shannon_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        prob = [float(text.count(c)) / len(text) for c in set(text)]
        return -sum(p * np.log2(p) for p in prob if p > 0)

    @staticmethod
    def get_feature_names() -> List[str]:
        return [
            'url_length', 'num_dots', 'num_slashes', 'num_query_params',
            'has_ip_address', 'url_entropy', 'num_subdomains', 'is_https',
            'domain_length', 'has_suspicious_tld', 'path_length',
            'has_login_keyword', 'percent_encoded_chars'
        ]

    def transform(self, urls: List[str]) -> pd.DataFrame:
        rows = [self.extract(u) for u in urls]
        return pd.DataFrame(rows)