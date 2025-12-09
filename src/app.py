import difflib
import os
import sys
import torch
import re
import spacy
import json
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from datetime import datetime


# ==========================================
# COMPREHENSIVE GUIDED REQUIREMENT SYSTEM
# ==========================================

class GuidedTutorial:
    """Interactive tutorial to help users understand requirements"""

    @staticmethod
    def show_requirement_basics():
        """Teach users what a good requirement is"""
        print("\n" + "=" * 80)
        print("📚 WHAT IS A GOOD REQUIREMENT? (Quick Tutorial)")
        print("=" * 80 + "\n")

        print("A good requirement should follow the SMART criteria:\n")
        print("  S - SPECIFIC: Clear and well-defined")
        print("    ❌ BAD: 'The system should be fast'")
        print("    ✅ GOOD: 'The system SHALL process login requests within 2 seconds'\n")

        print("  M - MEASURABLE: Can be tested/verified")
        print("    ❌ BAD: 'The system should be easy to use'")
        print("    ✅ GOOD: 'Users SHALL complete registration in less than 5 steps'\n")

        print("  A - ACHIEVABLE: Realistic and feasible")
        print("    ❌ BAD: 'The system SHALL process 1 million requests per second'")
        print("    ✅ GOOD: 'The system SHALL process 10,000 requests per second'\n")

        print("  R - RELEVANT: Matches your system's purpose")
        print("    ❌ BAD: For a Hospital System: 'The system SHALL manage inventory'")
        print("    ✅ GOOD: 'The system SHALL record patient medical history'\n")

        print("  T - TESTABLE: You can verify it works")
        print("    ❌ BAD: 'The system should be reliable'")
        print("    ✅ GOOD: 'The system SHALL have 99.5% uptime measured monthly'\n")

        print("=" * 80 + "\n")
        input("Press Enter to continue...")


class IEEE830Parser:
    """Parse requirements in IEEE 830 format with detailed guidance"""

    def __init__(self):
        print("   - Loading IEEE 830 Parser...")
        self.nlp = self._load_spacy()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Comprehensive system-wide categories (domain-agnostic)
        self.system_categories = {
            'authentication_security': ['login', 'signin', 'logout', 'session', 'authenticate', 'password',
                                        'credentials', 'auth'],
            'data_security': ['encrypt', 'secure', 'privacy', 'confidential', 'protect', 'access control', 'permission',
                              'authorization'],
            'performance': ['response time', 'latency', 'throughput', 'speed', 'fast', 'slow', 'delay', 'timeout'],
            'availability': ['uptime', 'availability', 'downtime', 'backup', 'recovery', 'disaster', 'failover',
                             'redundancy'],
            'scalability': ['scale', 'concurrent', 'load', 'growth', 'users', 'capacity', 'performance under load'],
            'compliance': ['gdpr', 'hipaa', 'iso', 'pci', 'wcag', 'soc', 'compliance', 'regulatory', 'standard'],
            'usability': ['user interface', 'ui', 'ux', 'intuitive', 'user-friendly', 'navigation', 'accessibility',
                          'usable'],
            'maintenance': ['log', 'monitor', 'alert', 'debug', 'trace', 'metric', 'maintenance', 'support'],
            'integration': ['integrate', 'api', 'interface', 'connect', 'sync', 'export', 'import'],
            'error_handling': ['error', 'exception', 'failure', 'recovery', 'retry', 'validation', 'constraint'],
        }

        self.requirement_keywords = {
            'absolute': ['shall', 'must', 'will'],
            'desirable': ['should', 'could', 'may'],
            'negative': ['shall not', 'must not', 'cannot']
        }

    def _load_spacy(self):
        try:
            return spacy.load("en_core_web_sm")
        except:
            return None

    def get_requirement_category(self, text):
        """Identify if requirement is system-wide or domain-specific"""
        text_lower = text.lower()
        matched_categories = []

        for category, keywords in self.system_categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matched_categories.append(category)
                    break

        return {
            'is_system_wide': len(matched_categories) > 0,
            'categories': list(set(matched_categories)),
            'requirement_type': 'Non-Functional' if len(matched_categories) > 0 else 'Functional'
        }

    def parse_requirement(self, text):
        """Parse requirement with detailed feedback"""
        result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'components': {
                'req_id': None,
                'priority': None,
                'status': None,
                'keyword': None,
                'keyword_type': None,
                'actor': None,
                'action': None,
                'object': None,
                'constraints': []
            },
            'original_text': text,
            'cleaned_text': text  # Default to original
        }

        # --- PARSE: Requirement ID, Priority, Status ---
        id_match = re.match(r'(REQ-\d+)\s+(HIGH|MEDIUM|LOW)\s+(ACTIVE|DRAFT|APPROVED)', text, re.IGNORECASE)

        if id_match:
            result['components']['req_id'] = id_match.group(1)
            result['components']['priority'] = id_match.group(2).upper()
            result['components']['status'] = id_match.group(3).upper()
            text = text[id_match.end():].strip()
        else:
            priority_match = re.match(r'(HIGH|MEDIUM|LOW)\s+(ACTIVE|DRAFT|APPROVED)\s+(.*)', text, re.IGNORECASE)
            if priority_match:
                result['components']['priority'] = priority_match.group(1).upper()
                result['components']['status'] = priority_match.group(2).upper()
                text = priority_match.group(3).strip()

        # Update the cleaned text (The CRITICAL Fix)
        result['cleaned_text'] = text

        if not result['components']['priority']:
            result['components']['priority'] = 'MEDIUM'
        if not result['components']['status']:
            result['components']['status'] = 'DRAFT'

        # --- PARSE: Requirement Keyword ---
        keyword_found = False
        for keyword_type, keywords in self.requirement_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    result['components']['keyword'] = keyword.upper()
                    result['components']['keyword_type'] = keyword_type
                    keyword_found = True
                    break
            if keyword_found:
                break

        if not keyword_found:
            result['issues'].append("Missing requirement keyword (SHALL, MUST, SHOULD)")
            result['warnings'].append("💡 Add SHALL for mandatory, SHOULD for desirable requirements")

        # --- PARSE: NLP Components ---
        if self.nlp:
            doc = self.nlp(text.lower())

            for token in doc:
                if token.pos_ == "VERB":
                    result['components']['action'] = token.text
                    break

            for token in doc:
                if token.dep_ == "nsubj":
                    result['components']['actor'] = token.text
                elif token.dep_ == "dobj" and not result['components']['object']:
                    result['components']['object'] = token.text

            # Extract constraints
            constraint_patterns = [
                r'within (\d+\s*(?:seconds?|minutes?|hours?|days?))',
                r'(\d+\s*(?:requests?|transactions?|users?|records?).*(?:per|/)\s*(?:second|minute|hour))',
                r'(greater than|less than|more than|at least|maximum|minimum)\s*(\d+%?)',
                r'(?:must be|should be|is|are)\s*(encrypted|secure|password-protected)',
                r'(iso|gdpr|hipaa|pci|wcag|soc)',
            ]

            for pattern in constraint_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                result['components']['constraints'].extend(matches)

        # --- VALIDATION ---
        if not text or len(text.split()) < 5:
            result['issues'].append("Requirement too short (minimum 5 words)")
            result['warnings'].append("💡 Use complete sentences with subject-verb-object structure")
            result['valid'] = False

        if len(text.split()) > 200:
            result['issues'].append("Requirement too long (maximum 200 words)")
            result['warnings'].append("💡 Break into multiple requirements if too long")
            result['valid'] = False

        has_action = result['components']['action'] is not None
        has_object = result['components']['object'] is not None

        if not (has_action and has_object):
            result['issues'].append("Missing clear action and/or object")
            result['warnings'].append("💡 Structure: WHO SHALL ACTION WHAT (CONSTRAINT)")
            result['warnings'].append("   Example: 'The system SHALL validate email addresses'")

        # Check for specificity
        vague_words = ['good', 'bad', 'nice', 'user-friendly', 'robust', 'efficient', 'easy']
        if any(word in text.lower() for word in vague_words):
            result['warnings'].append(
                "💡 Avoid vague terms like 'easy' or 'efficient' - be specific with numbers/metrics")

        return result

class InteractiveScopeBuilder:
    """Build scope with detailed guidance at each step"""

    def __init__(self):
        print("   - Loading Scope Builder...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = self._load_spacy()

        self.scope = {
            'system_name': None,
            'system_description': None,
            'primary_users': [],
            'main_features': [],
            'excluded_features': [],
            'technical_constraints': [],
            'business_constraints': []
        }

    def _load_spacy(self):
        try:
            return spacy.load("en_core_web_sm")
        except:
            return None

    def build_scope(self):
        """Interactive scope building with guidance"""
        print("\n" + "=" * 80)
        print("🎯 STEP 1: DEFINE YOUR SYSTEM SCOPE")
        print("=" * 80)
        print("\nDuring this phase, we'll define WHAT your system will and won't do.")
        print("This helps ensure all requirements stay aligned with your goals.\n")

        # 1. System Name
        print("-" * 80)
        print("📌 STEP 1.1: SYSTEM NAME & DESCRIPTION")
        print("-" * 80 + "\n")
        print("This should be a clear, concise name for your system.")
        print("Examples:")
        print("  • E-Commerce Platform")
        print("  • Hospital Management System")
        print("  • Weather Forecasting Application")
        print("  • Social Media Platform\n")

        while True:
            name = input("📝 Enter system name: ").strip()
            if len(name) >= 3:
                self.scope['system_name'] = name
                break
            print("   ⚠️ Please provide at least 3 characters\n")

        print("\nNow, describe what your system does in 1-2 sentences.")
        print("Examples:")
        print("  • 'Manages online shopping with product catalog, cart, checkout, and order tracking'")
        print("  • 'Enables healthcare providers to manage patient records, appointments, and billing'")
        print("  • 'Allows users to share photos, videos, and messages with their network'\n")

        while True:
            desc = input("📝 Enter system description (min 10 words): ").strip()
            if len(desc.split()) >= 10:
                self.scope['system_description'] = desc
                break
            print("   ⚠️ Please provide at least 10 words\n")

        # 2. Primary Users
        print("\n" + "-" * 80)
        print("👥 STEP 1.2: PRIMARY USERS/STAKEHOLDERS")
        print("-" * 80 + "\n")
        print("Who will use or interact with your system?")
        print("Examples:")
        print("  • End Users / Customers / Employees / Students")
        print("  • Admin / Manager / Supervisor / Doctor")
        print("  • System Administrator / Support Staff / Analyst")
        print("  • Third-party systems / External APIs\n")

        users = []
        while True:
            user = input(f"   User type {len(users) + 1} (or press Enter): ").strip()
            if not user:
                if len(users) >= 1:
                    break
                print("   ⚠️ Please define at least 1 user type\n")
                continue

            if user.lower() not in [u.lower() for u in users]:
                users.append(user)
                print(f"      ✅ Added: {user}")
            else:
                print(f"      ⚠️ Already added")

        self.scope['primary_users'] = users

        # 3. Main Features
        print("\n" + "-" * 80)
        print("⚡ STEP 1.3: MAIN FEATURES/CAPABILITIES")
        print("-" * 80 + "\n")
        print("List the core features your system WILL provide.")
        print("Examples for E-Commerce:")
        print("  • Product browsing and search")
        print("  • Shopping cart management")
        print("  • Secure payment processing")
        print("  • Order tracking")
        print("  • User reviews and ratings\n")

        features = []
        while True:
            feature = input(f"   Feature {len(features) + 1} (or press Enter): ").strip()
            if not feature:
                if len(features) >= 2:
                    break
                print(f"   ⚠️ Please define at least 2 features\n")
                continue

            if feature.lower() not in [f.lower() for f in features]:
                features.append(feature)
                print(f"      ✅ Added: {feature}")
            else:
                print(f"      ⚠️ Already added")

        self.scope['main_features'] = features

        # 4. Excluded Features
        print("\n" + "-" * 80)
        print("🚫 STEP 1.4: WHAT'S OUT OF SCOPE?")
        print("-" * 80 + "\n")
        print("List features or capabilities your system will NOT provide.")
        print("Examples for E-Commerce:")
        print("  • Manufacturing or inventory management")
        print("  • Customer financial auditing")
        print("  • Shipping logistics")
        print("  • Tax calculation and filing\n")

        excluded = []
        while True:
            excl = input(f"   Excluded feature {len(excluded) + 1} (or press Enter): ").strip()
            if not excl:
                if len(excluded) >= 1:
                    break
                print(f"   ⚠️ Please define at least 1 excluded feature\n")
                continue

            if excl.lower() not in [e.lower() for e in excluded]:
                excluded.append(excl)
                print(f"      ✅ Added: {excl}")
            else:
                print(f"      ⚠️ Already added")

        self.scope['excluded_features'] = excluded

        # 5. Technical Constraints
        print("\n" + "-" * 80)
        print("⚙️ STEP 1.5: TECHNICAL CONSTRAINTS")
        print("-" * 80 + "\n")
        print("What technical limitations exist?")
        print("Examples:")
        print("  • Must run on Windows, Mac, Linux")
        print("  • Browser compatibility (Chrome, Firefox, Safari, Edge)")
        print("  • Database: PostgreSQL, MySQL, MongoDB")
        print("  • Cloud platform: AWS, Azure, Google Cloud")
        print("  • Maximum file size: 100MB")
        print("  • Supported languages: English, Spanish, French\n")

        tech_constraints = []
        while True:
            constraint = input(f"   Constraint {len(tech_constraints) + 1} (or press Enter): ").strip()
            if not constraint:
                if len(tech_constraints) >= 1:
                    break
                print(f"   ⚠️ Please define at least 1 technical constraint\n")
                continue

            tech_constraints.append(constraint)
            print(f"      ✅ Added: {constraint}")

        self.scope['technical_constraints'] = tech_constraints

        # 6. Business Constraints
        print("\n" + "-" * 80)
        print("💼 STEP 1.6: BUSINESS/REGULATORY CONSTRAINTS")
        print("-" * 80 + "\n")
        print("What business or regulatory requirements exist?")
        print("Examples:")
        print("  • GDPR compliance required")
        print("  • HIPAA compliance for healthcare")
        print("  • PCI-DSS for payment processing")
        print("  • Budget limit: $100,000")
        print("  • Launch date: Q2 2024")
        print("  • Support 1 million concurrent users")
        print("  • 24/7 availability required\n")

        biz_constraints = []
        while True:
            constraint = input(f"   Constraint {len(biz_constraints) + 1} (or press Enter): ").strip()
            if not constraint:
                if len(biz_constraints) >= 1:
                    break
                print(f"   ⚠️ Please define at least 1 business constraint\n")
                continue

            biz_constraints.append(constraint)
            print(f"      ✅ Added: {constraint}")

        self.scope['business_constraints'] = biz_constraints

        self.display_scope()
        return self.scope

    def display_scope(self):
        """Display scope summary"""
        print("\n" + "=" * 80)
        print("✅ SCOPE STATEMENT COMPLETE")
        print("=" * 80 + "\n")

        print(f"📌 SYSTEM: {self.scope['system_name']}")
        print(f"📝 DESCRIPTION: {self.scope['system_description']}\n")

        print(f"👥 PRIMARY USERS:")
        for user in self.scope['primary_users']:
            print(f"   • {user}")

        print(f"\n⚡ MAIN FEATURES:")
        for feature in self.scope['main_features']:
            print(f"   • {feature}")

        print(f"\n🚫 OUT OF SCOPE:")
        for excluded in self.scope['excluded_features']:
            print(f"   • {excluded}")

        print(f"\n⚙️ TECHNICAL CONSTRAINTS:")
        for constraint in self.scope['technical_constraints']:
            print(f"   • {constraint}")

        print(f"\n💼 BUSINESS CONSTRAINTS:")
        for constraint in self.scope['business_constraints']:
            print(f"   • {constraint}")

        print("\n" + "=" * 80)


class RequirementClassifier:
    """ML-based requirement classification with ALL models"""

    def __init__(self, models_dir):
        print("   - Loading Classifiers...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_dir = models_dir

        # 1. Load Intent Model (Manual)
        self.intent_model_data = self._load_intent_model_manual()

        # 2. Load FR/NFR (Still using pipeline as it seemed to work, but can be changed if needed)
        self.fr_nfr_classifier = self._load_pipeline("fr_nfr_classifier/final_model")

        # 3. Load QA Model (Manual)
        self.qa_model_data = self._load_qa_model_manual()

        # 4. Load Ambiguity Model (Manual)
        self.ambiguity_detector = self._load_ambiguity_model_manual()

    def _load_pipeline(self, model_subpath):
        """Helper to load standard pipelines"""
        try:
            model_path = os.path.join(self.models_dir, model_subpath).replace("\\", "/")
            if os.path.exists(model_path):
                pipe_device = 0 if self.device == "cuda" else -1
                return pipeline("text-classification", model=model_path,
                                tokenizer=model_path, device=pipe_device)
        except Exception as e:
            print(f"      ⚠️ Could not load {model_subpath}: {str(e)[:50]}")
        return None

    def _load_intent_model_manual(self):
        """Manually load Intent model to ensure labels are correct"""
        try:
            model_path = os.path.join(self.models_dir, "intent_classifier").replace("\\", "/")
            if os.path.exists(model_path):
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.to(self.device)
                model.eval()
                return {'tokenizer': tokenizer, 'model': model, 'id2label': model.config.id2label}
        except Exception as e:
            print(f"      ⚠️ Intent Model could not load: {str(e)[:50]}")
        return None

    def _load_qa_model_manual(self):
        """Manually load QA model"""
        try:
            model_path = os.path.join(self.models_dir, "quality_attribute_classifier/final_model").replace("\\", "/")
            if os.path.exists(model_path):
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.to(self.device)
                model.eval()
                return {'tokenizer': tokenizer, 'model': model, 'id2label': model.config.id2label}
        except Exception as e:
            print(f"      ⚠️ QA Model could not load: {str(e)[:50]}")
        return None

    def _load_ambiguity_model_manual(self):
        """Manually load T5 Ambiguity model"""
        try:
            model_path = os.path.join(self.models_dir, "ambiguity_detector").replace("\\", "/")
            if os.path.exists(model_path):
                from transformers import T5Tokenizer, T5ForConditionalGeneration
                tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
                model = T5ForConditionalGeneration.from_pretrained(model_path)
                model.to(self.device)
                model.eval()
                return {'tokenizer': tokenizer, 'model': model}
        except Exception as e:
            print(f"      ⚠️ Ambiguity detector could not load: {str(e)[:50]}")
        return None

    def predict_generic_manual(self, text, model_data):
        """Generic manual prediction helper for Intent and QA"""
        if not model_data:
            return None, "0.00"

        try:
            tokenizer = model_data['tokenizer']
            model = model_data['model']
            id2label = model_data['id2label']

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)

            pred_id = torch.argmax(logits, dim=1).item()
            score = probs[0][pred_id].item()
            label = id2label[pred_id]

            return label, f"{score:.2f}"
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, "0.00"

    def detect_ambiguity(self, text):
        """Ambiguity detection using T5"""
        if not self.ambiguity_detector:
            return {
                'is_ambiguous': False,
                'ambiguity_score': 0,
                'explanation': 'Model not loaded',
                'suggestions': [],
                'corrected_requirement': text
            }

        try:
            tokenizer = self.ambiguity_detector['tokenizer']
            model = self.ambiguity_detector['model']

            # PREFIX MUST MATCH TRAINING
            input_text = "analyze requirement: " + text

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=128,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True
                )

            raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            is_ambiguous = False
            explanation = ""
            corrected_requirement = text
            suggestions = []

            if "Ambiguous: NO" in raw_output:
                is_ambiguous = False
            else:
                is_ambiguous = True

                if "Reason:" in raw_output:
                    try:
                        part1 = raw_output.split("Reason:")[1]
                        reason_text = part1.split("|")[0].strip()
                        explanation = reason_text
                        suggestions.append(explanation)
                    except:
                        explanation = "Could not parse reason."

                if "Correction:" in raw_output:
                    try:
                        corr_text = raw_output.split("Correction:")[1].strip()
                        corrected_requirement = corr_text
                    except:
                        pass

            return {
                'is_ambiguous': is_ambiguous,
                'ambiguity_score': 0.95 if is_ambiguous else 0.1,
                'explanation': explanation,
                'suggestions': suggestions,
                'corrected_requirement': corrected_requirement
            }

        except Exception as e:
            print(f"Error in ambiguity: {e}")
            return {
                'is_ambiguous': False,
                'ambiguity_score': 0,
                'explanation': 'Error',
                'suggestions': [],
                'corrected_requirement': text
            }

    def classify(self, text):
        """Classify requirement using ALL ML models"""
        result = {
            'intent': None,
            'fr_nfr': 'FR',
            'quality_attribute': None,
            'ambiguity': {
                'is_ambiguous': False,
                'explanation': ''
            },
            'confidence': {}
        }

        # 1. Intent Classification (UPDATED TO MANUAL)
        print("      🔄 Running Intent Classifier...", end="", flush=True)
        if self.intent_model_data:
            label, score = self.predict_generic_manual(text, self.intent_model_data)
            if label:
                result['intent'] = label
                result['confidence']['intent'] = score
                print(" ✅")
            else:
                print(" ⚠️ (Prediction failed)")
        else:
            print(" ⏭️ (Not loaded)")

        # 2. FR/NFR Classification
        print("      🔄 Running FR/NFR Classifier...", end="", flush=True)
        if self.fr_nfr_classifier:
            try:
                fr_nfr_result = self.fr_nfr_classifier(text)[0]
                label = fr_nfr_result['label']
                result['fr_nfr'] = "NFR" if "LABEL_0" in label else "FR"
                result['confidence']['fr_nfr'] = f"{fr_nfr_result['score']:.2f}"
                print(" ✅")
            except Exception as e:
                print(f" ⚠️ ({e})")
        else:
            print(" ⏭️ (Not loaded)")

        # 3. Quality Attribute Classification (MANUAL)
        print("      🔄 Running Quality Attribute Classifier...", end="", flush=True)
        if self.qa_model_data:
            label, score = self.predict_generic_manual(text, self.qa_model_data)
            if label:
                result['quality_attribute'] = label
                result['confidence']['qa'] = score
                print(" ✅")
            else:
                print(" ⚠️ (Prediction failed)")
        else:
            print(" ⏭️ (Not loaded)")

        # 4. Ambiguity Detection (MANUAL T5)
        print("      🔄 Running Ambiguity Detector...", end="", flush=True)
        ambiguity_result = self.detect_ambiguity(text)
        result['ambiguity'] = ambiguity_result

        if self.ambiguity_detector:
            if ambiguity_result['is_ambiguous']:
                print(" ⚠️ AMBIGUOUS")
            else:
                print(" ✅ Clear")
        else:
            print(" ⏭️ (Not loaded)")

        return result


class GuidedRequirementSession:
    """Main guided requirement gathering session"""

    def __init__(self, models_dir):
        self.scope_builder = InteractiveScopeBuilder()
        self.parser = IEEE830Parser()
        self.classifier = RequirementClassifier(models_dir)
        self.scope = None
        self.requirements = []
        self.req_counter = 0
        self.models_dir = models_dir

    def run(self):
        print("\n" + "=" * 80)
        print("🤖 INTELLIGENT REQUIREMENT GATHERING SYSTEM v6.0")
        print("=" * 80 + "\n")

        # Show tutorial
        print("Would you like a quick tutorial on what makes a good requirement? (y/n): ", end="")
        if input().strip().lower() == 'y':
            GuidedTutorial.show_requirement_basics()

        # Build scope
        self.scope = self.scope_builder.build_scope()

        # Gather requirements
        self._gather_requirements()

        # Final analysis and report
        self._generate_final_report()

    def _quick_test_mode(self):
        """Quick test mode - skip scope definition and test requirements directly"""
        print("\n" + "=" * 80)
        print("⚡ QUICK TEST MODE - Test Requirements Directly")
        print("=" * 80 + "\n")

        # Create minimal mock scope
        self.scope = {
            'system_name': 'Test System',
            'system_description': 'Testing system for requirement validation',
            'primary_users': ['User'],
            'main_features': ['Testing'],
            'excluded_features': ['None'],
            'technical_constraints': ['None'],
            'business_constraints': ['None']
        }

        print("ℹ️ Scope: Minimal test scope (all features in scope)")
        print("📝 You can now test requirement validation, ambiguity detection, and ML classification\n")
        print("Commands: 'help', 'checklist', 'summary', 'exit'\n")
        print("=" * 80 + "\n")

        # Quick test requirement gathering
        while True:
            try:
                user_input = input("📝 Enter Requirement to Test: ").strip()

                if not user_input:
                    continue

                # Commands
                if user_input.lower() == 'exit':
                    if self.requirements:
                        print("\nWould you like to see summary? (y/n): ", end="")
                        if input().strip().lower() == 'y':
                            self._show_current_summary()
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'checklist':
                    self._show_requirement_checklist()
                    continue
                elif user_input.lower() == 'summary':
                    self._show_current_summary()
                    continue

                # Parse requirement
                parsed = self.parser.parse_requirement(user_input)
                cleaned_text = parsed['cleaned_text']  # <--- EXTRACT CLEAN TEXT

                # Show validation feedback
                if not parsed['valid']:
                    print(f"\n⚠️ VALIDATION ISSUES:")
                    for issue in parsed['issues']:
                        print(f"   ❌ {issue}")
                    if parsed['warnings']:
                        print()
                        for warning in parsed['warnings']:
                            print(f"   {warning}")
                    print()
                    continue

                if parsed['warnings']:
                    print(f"\n💡 SUGGESTIONS:")
                    for warning in parsed['warnings']:
                        print(f"   {warning}")
                    print()

                # Get category
                category = self.parser.get_requirement_category(cleaned_text)

                # Classify using CLEANED text (This fixes the ML confusion)
                print(f"\n   🤖 Running ML Models:")
                classification = self.classifier.classify(cleaned_text)

                # Generate ID
                self.req_counter += 1
                if not parsed['components']['req_id']:
                    parsed['components']['req_id'] = f"REQ-{self.req_counter:03d}"

                # Decide what text to save
                final_text = cleaned_text
                if classification['ambiguity']['is_ambiguous']:
                    suggestion = classification['ambiguity']['corrected_requirement']
                    if suggestion and len(suggestion) > 5:
                        final_text = suggestion

                # Store requirement
                req_entry = {
                    'id': parsed['components']['req_id'],
                    'priority': parsed['components']['priority'],
                    'status': parsed['components']['status'],
                    'keyword': parsed['components']['keyword'],
                    'is_system_wide': category['is_system_wide'],
                    'categories': category['categories'],
                    'requirement_type': category['requirement_type'],
                    'fr_nfr': classification['fr_nfr'],
                    'quality_attribute': classification['quality_attribute'],
                    'intent': classification['intent'],
                    'is_ambiguous': classification['ambiguity']['is_ambiguous'],
                    'ambiguity_details': classification['ambiguity'],
                    'full_text': final_text,  # Save corrected/clean version
                    'original_user_input': user_input,  # Save raw input for reference
                    'timestamp': datetime.now().isoformat()
                }
                self.requirements.append(req_entry)

                # Display acceptance
                self._display_requirement_accepted(req_entry, classification)

            except Exception as e:
                print(f"\n❌ Error: {e}\n")

    def _gather_requirements(self):
        """Main requirement gathering loop with guidance"""
        print("\n" + "=" * 80)
        print("🎯 STEP 2: GATHER REQUIREMENTS")
        print("=" * 80)
        print("\nNow we'll collect detailed requirements for your system.")
        print("Each requirement will be validated and classified.\n")

        print("-" * 80)
        print("REQUIREMENT FORMAT:")
        print("-" * 80)
        print("[Optional: REQ-ID] [PRIORITY] [STATUS] requirement-statement\n")
        print("PRIORITY: HIGH (essential), MEDIUM (important), LOW (nice-to-have)")
        print("STATUS: ACTIVE (ready), DRAFT (being reviewed), APPROVED (finalized)\n")

        print("EXAMPLES:")
        print("  ✅ HIGH ACTIVE The system SHALL validate user email addresses")
        print("  ✅ MEDIUM DRAFT The system SHOULD send confirmation emails")
        print("  ✅ LOW ACTIVE The system MAY allow custom themes\n")

        print("TIPS FOR GOOD REQUIREMENTS:")
        print("  • Use SHALL for mandatory, SHOULD for desirable, MAY for optional")
        print("  • Include specific numbers/metrics when possible")
        print("  • Start with 'The system SHALL', 'Users SHALL', 'The system MUST'")
        print("  • Avoid vague words: good, bad, easy, fast (without metrics)")
        print("  • One requirement per statement (don't use 'and')\n")

        print("COMMANDS: 'help', 'scope', 'checklist', 'summary', 'exit'\n")
        print("-" * 80 + "\n")

        while True:
            try:
                user_input = input("📝 Enter Requirement: ").strip()

                if not user_input:
                    continue

                # Commands
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'scope':
                    self.scope_builder.display_scope()
                    continue
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'checklist':
                    self._show_requirement_checklist()
                    continue
                elif user_input.lower() == 'summary':
                    self._show_current_summary()
                    continue

                # Parse requirement
                parsed = self.parser.parse_requirement(user_input)

                # Show validation feedback
                if not parsed['valid']:
                    print(f"\n⚠️ VALIDATION ISSUES:")
                    for issue in parsed['issues']:
                        print(f"   ❌ {issue}")
                    print()
                    for warning in parsed['warnings']:
                        print(f"   {warning}")
                    print()
                    continue

                if parsed['warnings']:
                    print(f"\n💡 SUGGESTIONS:")
                    for warning in parsed['warnings']:
                        print(f"   {warning}")
                    print()

                # Get category
                category = self.parser.get_requirement_category(user_input)

                # Check scope alignment
                scope_alignment = self._check_scope_alignment(user_input)
                if not scope_alignment['aligned']:
                    print(f"\n⚠️ SCOPE ALERT: {scope_alignment['message']}")
                    confirm = input("   Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        print()
                        continue

                # Classify using ALL ML models
                print(f"\n   🤖 Running ML Models:")
                classification = self.classifier.classify(user_input)

                # Generate ID
                self.req_counter += 1
                if not parsed['components']['req_id']:
                    parsed['components']['req_id'] = f"REQ-{self.req_counter:03d}"

                # Store requirement with ambiguity info
                req_entry = {
                    'id': parsed['components']['req_id'],
                    'priority': parsed['components']['priority'],
                    'status': parsed['components']['status'],
                    'keyword': parsed['components']['keyword'],
                    'is_system_wide': category['is_system_wide'],
                    'categories': category['categories'],
                    'requirement_type': category['requirement_type'],
                    'fr_nfr': classification['fr_nfr'],
                    'quality_attribute': classification['quality_attribute'],
                    'intent': classification['intent'],
                    'is_ambiguous': classification['ambiguity']['is_ambiguous'],
                    'ambiguity_details': classification['ambiguity'],
                    'full_text': user_input,
                    'timestamp': datetime.now().isoformat()
                }

                self.requirements.append(req_entry)

                # Display acceptance with detailed info
                self._display_requirement_accepted(req_entry, classification)

            except Exception as e:
                print(f"\n❌ Error: {e}\n")

    def _check_scope_alignment(self, requirement_text):
        """Check if requirement aligns with defined scope"""
        text_lower = requirement_text.lower()

        # Check against excluded features
        for excluded in self.scope['excluded_features']:
            if excluded.lower() in text_lower:
                return {
                    'aligned': False,
                    'message': f"Seems to relate to '{excluded}' which is out of scope"
                }

        # Check against main features
        matching_features = []
        for feature in self.scope['main_features']:
            if feature.lower() in text_lower:
                matching_features.append(feature)

        if matching_features:
            return {
                'aligned': True,
                'message': f"Aligns with features: {', '.join(matching_features)}"
            }

        # Check if system-wide (always aligned)
        category = self.parser.get_requirement_category(requirement_text)
        if category['is_system_wide']:
            return {
                'aligned': True,
                'message': f"System-wide requirement: {', '.join(category['categories'])}"
            }

        return {
            'aligned': False,
            'message': "Doesn't clearly relate to defined scope. Is it still relevant?"
        }

    def _display_requirement_accepted(self, req, classification):
        """Display detailed acceptance message with corrected requirement option"""
        print(f"\n✅ REQUIREMENT ACCEPTED: {req['id']}")
        print(f"   Priority: {req['priority']} | Status: {req['status']}")
        print(f"   Keyword: {req['keyword']}\n")

        print(f"   🔍 ML CLASSIFICATION RESULTS:")
        print(f"   " + "-" * 76)

        # Intent
        if classification['intent']:
            print(
                f"   📌 Intent: {classification['intent']} (Confidence: {classification['confidence'].get('intent', 'N/A')})")

        # FR/NFR
        print(f"   📊 Type: {req['fr_nfr']} (Confidence: {classification['confidence'].get('fr_nfr', 'N/A')})")

        # Quality Attribute
        if classification['quality_attribute']:
            print(
                f"   ⭐ Quality Attribute: {classification['quality_attribute']} (Confidence: {classification['confidence'].get('qa', 'N/A')})")

        # System-Wide vs Domain-Specific
        if req['is_system_wide']:
            print(f"   🌐 Scope: System-Wide ({', '.join(req['categories'])})")
        else:
            print(f"   🎯 Scope: Domain-Specific")

        # AMBIGUITY DETECTION (UPDATED DISPLAY)
        print(f"   " + "-" * 76)
        if classification['ambiguity']['is_ambiguous']:
            print(f"   ⚠️ AMBIGUITY DETECTED (Score: {classification['ambiguity']['ambiguity_score']:.2f}):")

            # Show what the user actually typed
            print(f"\n   📝 ORIGINAL INPUT:")
            print(f"      {req.get('original_user_input', req['full_text'])}")

            # Show the stored (corrected) text with the requested label
            print(f"\n   Corrected : {req['full_text']}")

            if classification['ambiguity']['suggestions']:
                print(f"\n   💡 Why it needs improvement:")
                for i, suggestion in enumerate(classification['ambiguity']['suggestions'], 1):
                    print(f"      {i}. {suggestion}")
        else:
            print(f"   ✅ Requirement is CLEAR (No ambiguity detected)")
            print(f"   📝 {req['full_text']}")

        print(f"   " + "-" * 76)
        print(f"   Total Requirements: {len(self.requirements)}\n")

    def _show_help(self):
        """Show help menu"""
        print("\n" + "=" * 80)
        print("📖 HELP MENU")
        print("=" * 80 + "\n")
        print("REQUIREMENT STRUCTURE: [ID] [PRIORITY] [STATUS] statement\n")
        print("USE THESE KEYWORDS:")
        print("  SHALL/MUST/WILL  → Mandatory (system MUST do this)")
        print("  SHOULD/COULD     → Desirable (system ideally does this)")
        print("  MAY              → Optional (nice to have)\n")
        print("PRIORITY GUIDE:")
        print("  HIGH   → Critical for system to work")
        print("  MEDIUM → Important but system can function without it")
        print("  LOW    → Nice-to-have enhancement\n")
        print("QUALITY CHECKLIST:")
        print("  ✓ Does it have a clear subject and action?")
        print("  ✓ Is it measurable/testable?")
        print("  ✓ Does it avoid vague terms?")
        print("  ✓ Is it relevant to your scope?")
        print("  ✓ Can you verify it works?\n")
        print("COMMANDS:")
        print("  'scope'     - Show system scope")
        print("  'checklist' - Show requirement quality checklist")
        print("  'summary'   - Show all requirements so far")
        print("  'exit'      - Finish gathering\n")
        print("=" * 80 + "\n")
        input("Press Enter to continue...")

    def _show_requirement_checklist(self):
        """Show quality checklist for requirements"""
        print("\n" + "=" * 80)
        print("✅ REQUIREMENT QUALITY CHECKLIST")
        print("=" * 80 + "\n")

        print("Before entering a requirement, ensure it meets these criteria:\n")

        checklist = [
            ("SPECIFIC", "Is it clear and well-defined?",
             "❌ 'The system should be fast' → ✅ 'Response time < 2 seconds'"),
            ("MEASURABLE", "Can you test if it works?",
             "❌ 'Easy to use' → ✅ 'Complete registration in < 5 steps'"),
            ("ACHIEVABLE", "Is it realistic and feasible?",
             "❌ 'Process 1M requests/sec' → ✅ 'Process 10K requests/sec'"),
            ("RELEVANT", "Does it match your system scope?",
             "❌ Hospital system: 'Manage inventory' → ✅ 'Record patient history'"),
            ("TESTABLE", "Can QA/testers verify it?",
             "❌ 'Be reliable' → ✅ '99.5% uptime measured monthly'"),
        ]

        for criterion, question, example in checklist:
            print(f"{criterion}")
            print(f"  Question: {question}")
            print(f"  Example: {example}\n")

        print("=" * 80 + "\n")
        input("Press Enter to continue...")

    def _show_current_summary(self):
        """Show current requirements summary"""
        print("\n" + "=" * 80)
        print("📊 CURRENT REQUIREMENTS SUMMARY")
        print("=" * 80 + "\n")

        print(f"System: {self.scope['system_name']}")
        print(f"Total Requirements: {len(self.requirements)}\n")

        if self.requirements:
            high = sum(1 for r in self.requirements if r['priority'] == 'HIGH')
            medium = sum(1 for r in self.requirements if r['priority'] == 'MEDIUM')
            low = sum(1 for r in self.requirements if r['priority'] == 'LOW')

            fr_count = sum(1 for r in self.requirements if r['fr_nfr'] == 'FR')
            nfr_count = sum(1 for r in self.requirements if r['fr_nfr'] == 'NFR')

            print(f"📈 STATISTICS:")
            print(f"  Priority: HIGH={high}, MEDIUM={medium}, LOW={low}")
            print(f"  Type: FR (Functional)={fr_count}, NFR (Non-Functional)={nfr_count}\n")

            print(f"📝 REQUIREMENTS:\n")
            for req in self.requirements:
                print(f"  [{req['id']}] {req['priority']:6} {req['fr_nfr']}")
                print(f"      {req['full_text'][:70]}...")
                print()
        else:
            print("  (No requirements added yet)\n")

        print("=" * 80 + "\n")

    def _generate_final_report(self):
        """Generate comprehensive final report with quality analysis"""
        print("\n" + "=" * 80)
        print("📋 FINAL REQUIREMENTS REPORT")
        print("=" * 80 + "\n")

        print(f"PROJECT: {self.scope['system_name']}")
        print(f"DESCRIPTION: {self.scope['system_description']}\n")

        print(f"DOCUMENT GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Statistics
        print("=" * 80)
        print("STATISTICS")
        print("=" * 80 + "\n")

        print(f"Total Requirements: {len(self.requirements)}\n")

        high = sum(1 for r in self.requirements if r['priority'] == 'HIGH')
        medium = sum(1 for r in self.requirements if r['priority'] == 'MEDIUM')
        low = sum(1 for r in self.requirements if r['priority'] == 'LOW')

        print("By Priority:")
        print(f"  🔴 HIGH (Critical):     {high} requirements")
        print(f"  🟡 MEDIUM (Important):  {medium} requirements")
        print(f"  🟢 LOW (Nice-to-have):  {low} requirements\n")

        fr = sum(1 for r in self.requirements if r['fr_nfr'] == 'FR')
        nfr = sum(1 for r in self.requirements if r['fr_nfr'] == 'NFR')

        print("By Type:")
        print(f"  ⚙️ Functional Requirements (FR):      {fr}")
        print(f"  📊 Non-Functional Requirements (NFR): {nfr}\n")

        # Quality metrics
        print("=" * 80)
        print("QUALITY METRICS")
        print("=" * 80 + "\n")

        avg_length = sum(len(r['full_text'].split()) for r in self.requirements) / len(
            self.requirements) if self.requirements else 0
        has_keyword = sum(1 for r in self.requirements if r['keyword']) / len(
            self.requirements) if self.requirements else 0

        print(f"Average Requirement Length: {avg_length:.0f} words")
        print(f"Requirements with Keywords (SHALL/MUST/SHOULD): {has_keyword * 100:.0f}%\n")

        # Ambiguity Analysis
        print("=" * 80)
        print("AMBIGUITY ANALYSIS")
        print("=" * 80 + "\n")

        ambiguous_reqs = [r for r in self.requirements if r.get('is_ambiguous', False)]
        clear_reqs = [r for r in self.requirements if not r.get('is_ambiguous', False)]

        print(f"Clear Requirements:      {len(clear_reqs)} ✅")
        print(f"Ambiguous Requirements:  {len(ambiguous_reqs)} ⚠️\n")

        if ambiguous_reqs:
            print("Ambiguous Requirements to Review:")
            for req in ambiguous_reqs:
                print(f"  • {req['id']}: {req['full_text'][:70]}...")
            print()

        # Requirements list with full details
        print("=" * 80)
        print("DETAILED REQUIREMENTS")
        print("=" * 80 + "\n")

        for req in self.requirements:
            print(f"{req['id']} [{req['priority']}] [{req['fr_nfr']}]")
            print(f"   {req['full_text']}")
            if req['is_system_wide']:
                print(f"   📌 System-Wide: {', '.join(req['categories'])}")
            if req['quality_attribute']:
                print(f"   ⭐ Attribute: {req['quality_attribute']}")
            if req.get('is_ambiguous'):
                print(f"   ⚠️ AMBIGUOUS - Review needed")
            print()

        # Export option
        print("=" * 80)
        print("EXPORT OPTIONS")
        print("=" * 80 + "\n")

        export = input("Save requirements to JSON file? (y/n): ").strip().lower()
        if export == 'y':
            filename = f"requirements_{self.scope['system_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            export_data = {
                'system_name': self.scope['system_name'],
                'system_description': self.scope['system_description'],
                'generated_at': datetime.now().isoformat(),
                'total_requirements': len(self.requirements),
                'scope': self.scope,
                'requirements': self.requirements,
                'statistics': {
                    'by_priority': {'HIGH': high, 'MEDIUM': medium, 'LOW': low},
                    'by_type': {'FR': fr, 'NFR': nfr},
                    'ambiguous_count': len(ambiguous_reqs),
                    'clear_count': len(clear_reqs)
                }
            }

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)

            print(f"\n✅ Exported to: {filename}\n")


def main():
    print("\n" + "=" * 80)
    print("🤖 INTELLIGENT REQUIREMENT GATHERING SYSTEM v6.0")
    print("=" * 80 + "\n")

    try:
        # Ask user if they want quick test or full flow
        print("SELECT MODE:")
        print("  1. Full Scope Definition → Requirement Gathering (complete flow)")
        print("  2. Quick Test Mode (skip scope, test requirements directly)\n")

        raw_input = input("Enter choice (1 or 2): ")
        mode = raw_input.strip()  # Remove spaces

        # DEBUG PRINT
        print(f"\n[DEBUG] You entered: '{mode}'")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.join(current_dir, "..", "models")

        # Initialize Session
        session = GuidedRequirementSession(models_path)

        if mode == '2':
            print("[DEBUG] Starting Quick Test Mode...")
            session._quick_test_mode()
        else:
            print(f"[DEBUG] Mode was not '2' (it was '{mode}'). Starting Full Flow...")
            session.run()

    except KeyboardInterrupt:
        print("\n\n🤖 Session interrupted. Goodbye! 👋")
        sys.exit(0)
    except AttributeError as ae:
        print("\n❌ ERROR: The '_quick_test_mode' function is missing or indented incorrectly inside the class.")
        print("   Make sure 'def _quick_test_mode(self):' is aligned with 'def run(self):'")
        print(f"   Full error: {ae}")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()