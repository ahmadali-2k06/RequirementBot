import os
import json
import re
import logging
import spacy
import torch
from typing import Dict, Any, List
from sentence_transformers import util, SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration, \
    pipeline

# Configure Logging
logger = logging.getLogger("PreciselyExtensions")


# ==========================================
# 1. DATA MANAGER
# ==========================================
class DataManager:
    def __init__(self, storage_path=None):
        if storage_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.storage_path = os.path.join(base_dir, "..", "data", "projects.json")
        else:
            self.storage_path = storage_path
        self.ensure_directory()

    def ensure_directory(self):
        directory = os.path.dirname(self.storage_path)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError:
                pass

    def save_db(self, db: Dict[str, Any]):
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(db, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save DB: {e}")

    def load_db(self) -> Dict[str, Any]:
        if not os.path.exists(self.storage_path):
            return {}
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load DB: {e}")
            return {}


# ==========================================
# 2. IEEE 830 PARSER
# ==========================================
class IEEE830Parser:
    def __init__(self):
        self.nlp = self._load_spacy()
        self.system_categories = {
            'security': ['login', 'password', 'encrypt', 'access'],
            'performance': ['response', 'latency', 'speed', 'fast'],
            'availability': ['uptime', 'recovery', 'backup'],
            'scalability': ['scale', 'users', 'load'],
            'compliance': ['gdpr', 'hipaa', 'iso'],
            'usability': ['ui', 'ux', 'user-friendly'],
        }
        self.requirement_keywords = {
            'absolute': ['shall', 'must', 'will'],
            'desirable': ['should', 'could', 'may'],
        }

    def _load_spacy(self):
        try:
            return spacy.load("en_core_web_sm")
        except:
            return None

    def parse_requirement(self, text: str) -> Dict[str, Any]:
        result = {
            'valid': True, 'issues': [], 'warnings': [],
            'components': {'req_id': None, 'priority': None, 'status': None, 'keyword': None},
            'cleaned_text': text
        }

        # Regex for ID/Priority/Status
        id_match = re.match(r'(REQ-\d+)\s+(HIGH|MEDIUM|LOW)\s+(ACTIVE|DRAFT|APPROVED)', text, re.IGNORECASE)
        if id_match:
            result['components']['req_id'] = id_match.group(1)
            result['components']['priority'] = id_match.group(2).upper()
            result['components']['status'] = id_match.group(3).upper()
            text = text[id_match.end():].strip()
        else:
            prio_match = re.match(r'(HIGH|MEDIUM|LOW)\s+(ACTIVE|DRAFT|APPROVED)\s+(.*)', text, re.IGNORECASE)
            if prio_match:
                result['components']['priority'] = prio_match.group(1).upper()
                result['components']['status'] = prio_match.group(2).upper()
                text = prio_match.group(3).strip()

        result['cleaned_text'] = text

        if not result['components']['priority']: result['components']['priority'] = 'MEDIUM'
        if not result['components']['status']: result['components']['status'] = 'DRAFT'

        found = False
        for k_type, k_list in self.requirement_keywords.items():
            if any(k in text.lower() for k in k_list):
                result['components']['keyword'] = k_type.upper()
                found = True
                break
        if not found:
            result['warnings'].append("Missing keyword (SHALL, MUST, SHOULD)")

        if len(text.split()) < 3:
            result['issues'].append("Requirement too short")
            result['valid'] = False

        return result


# ==========================================
# 3. REQUIREMENT CLASSIFIER (ALL 4 MODELS)
# ==========================================
class RequirementClassifier:
    """Loads Intent, FR/NFR, QA, and Ambiguity models"""

    def __init__(self, models_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_dir = models_dir
        self.models = {}
        self.load_models()

    def load_models(self):
        # 1. Intent Model (Manual)
        try:
            path = os.path.join(self.models_dir, "intent_classifier")
            if os.path.exists(path):
                self.models['intent'] = {
                    'tokenizer': AutoTokenizer.from_pretrained(path),
                    'model': AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
                }
        except Exception as e:
            logger.warning(f"Intent model failed: {e}")

        # 2. FR/NFR Model (Pipeline)
        try:
            path = os.path.join(self.models_dir, "fr_nfr_classifier/final_model")
            if os.path.exists(path):
                # Use -1 for CPU, 0 for CUDA
                device_id = 0 if self.device == "cuda" else -1
                self.models['fr_nfr'] = pipeline("text-classification", model=path, tokenizer=path, device=device_id)
        except Exception as e:
            logger.warning(f"FR/NFR model failed: {e}")

        # 3. QA Model (Manual)
        try:
            path = os.path.join(self.models_dir, "quality_attribute_classifier/final_model")
            if os.path.exists(path):
                self.models['qa'] = {
                    'tokenizer': AutoTokenizer.from_pretrained(path),
                    'model': AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
                }
        except Exception as e:
            logger.warning(f"QA model failed: {e}")

        # 4. Ambiguity Model (T5)
        try:
            path = os.path.join(self.models_dir, "ambiguity_detector")
            if os.path.exists(path):
                self.models['ambiguity'] = {
                    'tokenizer': T5Tokenizer.from_pretrained(path, legacy=False),
                    'model': T5ForConditionalGeneration.from_pretrained(path).to(self.device)
                }
        except Exception as e:
            logger.warning(f"Ambiguity model failed: {e}")

    def predict_helper(self, text, model_key):
        """Helper for manual models (Intent, QA)"""
        if model_key not in self.models: return None, "0.00"
        try:
            data = self.models[model_key]
            inputs = data['tokenizer'](text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                logits = data['model'](**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                pred_id = torch.argmax(logits, dim=1).item()
                label = data['model'].config.id2label[pred_id]
                score = probs[0][pred_id].item()
            return label, f"{score:.2f}"
        except:
            return None, "0.00"

    def detect_fr_nfr(self, text):
        """Helper for FR/NFR Pipeline"""
        if 'fr_nfr' not in self.models: return "FR", "0.00"
        try:
            res = self.models['fr_nfr'](text)[0]
            label = "NFR" if "LABEL_0" in res['label'] else "FR"
            return label, f"{res['score']:.2f}"
        except:
            return "FR", "0.00"

    def detect_ambiguity(self, text):
        """Helper for T5 Ambiguity"""
        if 'ambiguity' not in self.models:
            return {'is_ambiguous': False, 'score': 0, 'correction': text}
        try:
            data = self.models['ambiguity']
            input_text = "analyze requirement: " + text
            inputs = data['tokenizer'](input_text, return_tensors="pt", max_length=128, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = data['model'].generate(inputs.input_ids, max_length=256, num_beams=5, early_stopping=True)

            result = data['tokenizer'].decode(outputs[0], skip_special_tokens=True)
            is_ambig = "Ambiguous: YES" in result
            correction = text
            if is_ambig and "Correction:" in result:
                correction = result.split("Correction:")[1].strip()

            return {
                'is_ambiguous': is_ambig,
                'score': 0.95 if is_ambig else 0.10,
                'correction': correction
            }
        except Exception as e:
            logger.error(f"Ambiguity error: {e}")
            return {'is_ambiguous': False, 'score': 0, 'correction': text}

    def classify_all(self, text):
        """Run ALL 4 models"""
        return {
            'intent': self.predict_helper(text, 'intent'),
            'fr_nfr': self.detect_fr_nfr(text),
            'qa': self.predict_helper(text, 'qa'),
            'ambiguity': self.detect_ambiguity(text)
        }


# ==========================================
# 4. SCOPE ANALYZER
# ==========================================
class ScopeAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            self.model = None

    def check_alignment(self, text: str, scope: Dict):
        for excl in scope.get('excluded_features', []):
            if excl.lower() in text.lower():
                return {'aligned': False, 'reason': f"Matches excluded: '{excl}'"}

        if self.model and scope.get('main_features'):
            feats = scope['main_features']
            emb1 = self.model.encode(text, convert_to_tensor=True, device=self.device)
            emb2 = self.model.encode(feats, convert_to_tensor=True, device=self.device)
            scores = util.cos_sim(emb1, emb2)[0]
            if torch.max(scores) < 0.2:
                return {'aligned': True, 'warning': "Low relevance to main features."}
        return {'aligned': True, 'reason': "Aligned"}