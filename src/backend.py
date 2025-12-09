import os
import sys
import logging
import json
import csv
from io import StringIO
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from io import BytesIO
import jwt
from starlette.responses import StreamingResponse
from xhtml2pdf import pisa
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import Depends, Header
# Import helpers from extensions.py
# (Ensure extensions.py contains DataManager, IEEE830Parser, ScopeAnalyzer, RequirementClassifier)
from extensions import DataManager, ScopeAnalyzer, IEEE830Parser, RequirementClassifier

# ==========================================
# CONFIGURATION
# ==========================================
JWT_SECRET = "my_temporary_super_secret_access_key_123"
ALGORITHM = "HS256"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Backend")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "..", "models")  # Points to project_root/models
STATIC_DIR = os.path.join(CURRENT_DIR, "..", "front end")


class AppState:
    db: Dict[str, Any] = {}
    parser: Optional[IEEE830Parser] = None
    scope_analyzer: Optional[ScopeAnalyzer] = None
    classifier: Optional[RequirementClassifier] = None  # The 4-Model Engine
    data_manager: Optional[DataManager] = None


state = AppState()

# ==========================================
# STATIC TEXT (MATCHING YOUR CLI)
# ==========================================
TUTORIAL_TEXT = """
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef;">
    <h3 style="margin-top:0;">📚 WHAT IS A GOOD REQUIREMENT? (Quick Tutorial)</h3>
    <p>A good requirement should follow the <strong>SMART</strong> criteria:</p>
    <ul>
        <li><strong>S - SPECIFIC:</strong> Clear and well-defined<br>
            ❌ BAD: 'The system should be fast'<br>
            ✅ GOOD: 'The system SHALL process login requests within 2 seconds'
        </li>
        <li><strong>M - MEASURABLE:</strong> Can be tested/verified<br>
            ❌ BAD: 'The system should be easy to use'<br>
            ✅ GOOD: 'Users SHALL complete registration in less than 5 steps'
        </li>
        <li><strong>A - ACHIEVABLE:</strong> Realistic and feasible</li>
        <li><strong>R - RELEVANT:</strong> Matches your system's purpose</li>
        <li><strong>T - TESTABLE:</strong> You can verify it works</li>
    </ul>
</div>
"""

CHECKLIST_TEXT = """
<div style="background-color: #f0fdf4; padding: 15px; border-radius: 8px; border: 1px solid #bbf7d0;">
    <h3 style="margin-top:0;">✅ REQUIREMENT QUALITY CHECKLIST</h3>
    <p>Before entering a requirement, ensure it meets these criteria:</p>
    <ul>
        <li><strong>SPECIFIC:</strong> Is it clear? (Avoid 'good', 'easy')</li>
        <li><strong>MEASURABLE:</strong> Can you test if it works?</li>
        <li><strong>ACHIEVABLE:</strong> Is it realistic?</li>
        <li><strong>RELEVANT:</strong> Matches your system scope?</li>
        <li><strong>TESTABLE:</strong> Can QA/testers verify it?</li>
    </ul>
</div>
"""

HELP_TEXT = """
<div style="background-color: #eff6ff; padding: 15px; border-radius: 8px; border: 1px solid #bfdbfe;">
    <h3 style="margin-top:0;">📖 HELP MENU</h3>
    <p><strong>FORMAT:</strong> [Optional: ID] [PRIORITY] [STATUS] statement</p>
    <p><strong>KEYWORDS:</strong></p>
    <ul>
        <li>SHALL/MUST → Mandatory</li>
        <li>SHOULD → Desirable</li>
        <li>MAY → Optional</li>
    </ul>
    <p><strong>COMMANDS:</strong> 'scope', 'checklist', 'summary'</p>
</div>
"""


async def get_current_user(authorization: str = Header(None)):
    """
    Validates the Bearer Token sent from Frontend.
    If valid, allows access. If invalid, blocks request.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authentication Token")

    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer':
            raise HTTPException(status_code=401, detail="Invalid Authentication Scheme")

        # Verify Token signature using the shared Secret
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return payload  # Returns {userId: "...", name: "..."}

    except (ValueError, jwt.ExpiredSignatureError, jwt.DecodeError):
        raise HTTPException(status_code=401, detail="Invalid or Expired Token")

# ==========================================
# LIFECYCLE MANAGEMENT
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Server Starting...")

    # 1. Load Data Manager
    state.data_manager = DataManager()
    state.db = state.data_manager.load_db()

    # 2. Load Parsers
    state.parser = IEEE830Parser()

    # 3. Load Scope Analyzer
    state.scope_analyzer = ScopeAnalyzer()

    # 4. Load ALL ML Models (The heavy lifting)
    logger.info(f"📂 Loading ML Models from: {MODELS_DIR}")
    # This class loads Intent, FR/NFR, QA, and Ambiguity models
    state.classifier = RequirementClassifier(MODELS_DIR)

    yield

    # Shutdown
    state.data_manager.save_db(state.db)
    logger.info("🛑 Server Shutdown")


app = FastAPI(title="Precisely Backend", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Pydantic Models
# ==========================================
class ProjectInit(BaseModel):
    name: str
    description: str = ""


class ChatMessage(BaseModel):
    text: str


def parse_smart_list(text: str) -> List[str]:
    """
    Splits by comma but respects quotes.
    Example: 'Linux, "PostgreSQL, AWS", Python' -> ['Linux', 'PostgreSQL, AWS', 'Python']
    """
    if not text: return []
    try:
        # Use csv reader to handle quoted strings automatically
        reader = csv.reader(StringIO(text), skipinitialspace=True)
        for row in reader:
            return [item.strip() for item in row if item.strip()]
        return []
    except:
        # Fallback to simple split if csv parsing fails
        return [t.strip() for t in text.split(',') if t.strip()]
# ==========================================
# WIZARD LOGIC (Exact CLI Prompts)
# ==========================================
# ==========================================
# WIZARD LOGIC
# ==========================================
def run_scope_wizard(project: Dict, user_text: str) -> str:
    """
    State machine that mimics InteractiveScopeBuilder exactly.
    Uses parse_smart_list to handle quoted inputs (e.g., "Linux, Windows").
    """
    step = project.get('wizard_step')
    scope = project['scope']
    reply = ""

    # --- STEP 1: NAME ---
    if step == 'ASK_NAME':
        if len(user_text) < 3:
            return "⚠️ Please provide at least 3 characters for the System Name."

        scope['name'] = user_text
        project['wizard_step'] = 'ASK_DESC'

        reply = (
            f"✅ Name set: <strong>{user_text}</strong>.<br><br>"
            f"{'-' * 40}<br>"
            f"📌 <strong>STEP 1.1: SYSTEM DESCRIPTION</strong><br>"
            f"{'-' * 40}<br>"
            f"Describe what your system does in 1-2 sentences. Focus on the <strong>Main Goal</strong> and <strong>Primary Users</strong>.<br>"
            f"<em>Try using this format:<br>"
            f"• 'A <strong>[platform/tool]</strong> that allows <strong>[users]</strong> to <strong>[core activity]</strong>.'<br>"
            f"• 'Automates <strong>[process]</strong> for <strong>[target audience]</strong> to improve <strong>[outcome]</strong>.'</em><br><br>"
            f"📝 <strong>Enter system description (min 10 words):</strong>"
        )

    # --- STEP 2: DESCRIPTION ---
    elif step == 'ASK_DESC':
        if len(user_text.split()) < 10:
            return "⚠️ Please provide at least 10 words for the description."

        scope['description'] = user_text
        project['wizard_step'] = 'ASK_USERS'

        reply = (
            f"✅ Description saved.<br><br>"
            f"{'-' * 40}<br>"
            f"👥 <strong>STEP 1.2: PRIMARY USERS/STAKEHOLDERS</strong><br>"
            f"{'-' * 40}<br>"
            f"Who will use or interact with your system?<br>"
            f"<em>Common Examples:</em><br>"
            f"• <strong>General:</strong> End Users, Customers, Employees, Students<br>"
            f"• <strong>Management:</strong> Admin, Manager, Supervisor, Doctor<br>"
            f"• <strong>Technical:</strong> System Administrator, Support Staff, Analyst<br>"
            f"• <strong>External:</strong> Third-party systems, External APIs<br><br>"
            f"📝 <strong>Enter user types (comma separated):</strong><br>"
            f"💡 <em>Tip: Use quotes \"...\" to group user types containing commas.</em>"
        )

    # --- STEP 3: USERS ---
    elif step == 'ASK_USERS':
        users = parse_smart_list(user_text)  # Use Smart Parser
        if not users:
            return "⚠️ Please define at least 1 user type."

        scope['primary_users'] = users
        project['wizard_step'] = 'ASK_FEATURES'

        reply = (
            f"✅ Added users: {', '.join(users)}.<br><br>"
            f"{'-' * 40}<br>"
            f"⚡ <strong>STEP 1.3: MAIN FEATURES/CAPABILITIES</strong><br>"
            f"{'-' * 40}<br>"
            f"List the core features your system WILL provide.<br>"
            f"<em>General Examples:</em><br>"
            f"• <strong>User Mgmt:</strong> Registration, Authentication, Profile Management<br>"
            f"• <strong>Data:</strong> Search, Filtering, Data Import/Export<br>"
            f"• <strong>Action:</strong> Booking, Payment Processing, Messaging/Notifications<br>"
            f"• <strong>Analysis:</strong> Reporting, Dashboard Visualization, Logging<br><br>"
            f"📝 <strong>Enter features (comma separated):</strong><br>"
            f"💡 <em>Tip: Use quotes \"...\" to group items containing commas.</em>"
        )

    # --- STEP 4: FEATURES ---
    elif step == 'ASK_FEATURES':
        feats = parse_smart_list(user_text)  # Use Smart Parser
        if len(feats) < 2:
            return "⚠️ Please define at least 2 features."

        scope['main_features'] = feats
        project['wizard_step'] = 'ASK_EXCLUSIONS'

        reply = (
            f"✅ Added features.<br><br>"
            f"{'-' * 40}<br>"
            f"🚫 <strong>STEP 1.4: WHAT'S OUT OF SCOPE?</strong><br>"
            f"{'-' * 40}<br>"
            f"List features or capabilities your system will NOT provide.<br>"
            f"<em>Common Exclusions:</em><br>"
            f"• <strong>Platform:</strong> Native Mobile App, Offline Mode<br>"
            f"• <strong>Data:</strong> Legacy Data Migration, Real-time Sync<br>"
            f"• <strong>Integrations:</strong> specific third-party tools not needed yet<br>"
            f"• <strong>Localization:</strong> Multi-language Support<br><br>"
            f"📝 <strong>Enter excluded features (comma separated).</strong><br>"
            f"💡 <em>Tip: Use quotes \"...\" for complex items.</em>"
        )

    # --- STEP 5: EXCLUSIONS ---
    elif step == 'ASK_EXCLUSIONS':
        excl = parse_smart_list(user_text)  # Use Smart Parser
        if not excl:
            # Allow user to say 'None' effectively, but list shouldn't be empty if they typed something
            return "⚠️ Please define exclusions (or type 'None')."

        scope['excluded_features'] = excl
        project['wizard_step'] = 'ASK_TECH'

        reply = (
            f"✅ Exclusions noted.<br><br>"
            f"{'-' * 40}<br>"
            f"⚙️ <strong>STEP 1.5: TECHNICAL CONSTRAINTS</strong><br>"
            f"{'-' * 40}<br>"
            f"What technical limitations or requirements exist?<br>"
            f"<em>Consider these categories:</em><br>"
            f"• <strong>Platform/OS:</strong> Windows, macOS, Linux, iOS, Android, Web<br>"
            f"• <strong>Database:</strong> PostgreSQL, MySQL, MongoDB, Oracle<br>"
            f"• <strong>Cloud/Infrastructure:</strong> AWS, Azure, Google Cloud, On-Premise<br>"
            f"• <strong>Language/Framework:</strong> Python, Java, React, .NET<br>"
            f"• <strong>Performance:</strong> Max file size, Response time limits<br><br>"
            f"📝 <strong>Enter technical constraints (comma separated).</strong><br>"
            f"💡 <em>Example: \"Linux, Windows\", \"PostgreSQL, AWS\", \"Python 3.10+\"</em>"
        )

    # --- STEP 6: TECH ---
    elif step == 'ASK_TECH':
        tech = parse_smart_list(user_text)  # Use Smart Parser
        if not tech:
            return "⚠️ Please define at least 1 technical constraint."

        scope['technical_constraints'] = tech
        project['wizard_step'] = 'ASK_BIZ'

        reply = (
            f"✅ Tech constraints saved.<br><br>"
            f"{'-' * 40}<br>"
            f"💼 <strong>STEP 1.6: BUSINESS/REGULATORY CONSTRAINTS</strong><br>"
            f"{'-' * 40}<br>"
            f"What business rules, regulations, or limits apply?<br>"
            f"<em>Consider these factors:</em><br>"
            f"• <strong>Compliance:</strong> GDPR, HIPAA, PCI-DSS, ISO Standards<br>"
            f"• <strong>Resources:</strong> Budget cap, Team size, Hardware limits<br>"
            f"• <strong>Timeline:</strong> Launch deadline, Beta release date, Milestones<br>"
            f"• <strong>Legal/Ops:</strong> Licensing (Open Source/Commercial), 24/7 Support required<br><br>"
            f"📝 <strong>Enter business constraints (comma separated).</strong>"
        )

    # --- STEP 7: BIZ & FINISH ---
    elif step == 'ASK_BIZ':
        biz = parse_smart_list(user_text)  # Use Smart Parser
        if not biz:
            return "⚠️ Please define at least 1 business constraint."

        scope['business_constraints'] = biz
        project['wizard_step'] = 'DONE'
        project['status'] = 'GATHERING'

        reply = (
            f"🎉 <strong>SCOPE DEFINITION COMPLETE</strong><hr>"
            f"<strong>System:</strong> {scope['name']}<br>"
            f"<strong>Description:</strong> {scope['description']}<br><br>"
            f"{'=' * 40}<br>"
            f"🎯 <strong>STEP 2: GATHER REQUIREMENTS</strong><br>"
            f"{'=' * 40}<br>"
            f"Now, please enter your requirements one by one. I will analyze them for quality, ambiguity, and classification.<br><br>"
            f"<em>What to enter:</em><br>"
            f"• <strong>Functional:</strong> What the system MUST do (e.g., 'The system SHALL allow users to register').<br>"
            f"• <strong>Non-Functional:</strong> How the system performs (e.g., 'The system SHALL respond within 2 seconds').<br>"
            f"• <strong>Constraints:</strong> Specific limitations (e.g., 'The system MUST run on port 8080').<br><br>"
            f"👉 <strong>Type your first requirement:</strong><br><br>"
            f"<em>Commands: 'help', 'checklist', 'summary', 'exit'</em>"
        )

    return reply

# ==========================================
# GENERATE FINAL REPORT (UPDATED)
# ==========================================
def generate_final_report(project: Dict) -> str:
    # Need PID to create the link. We find the pid by checking state.db
    # (In a real DB we'd have the ID inside the project object, but here we search)
    pid = next((k for k, v in state.db.items() if v is project), None)

    scope = project['scope']
    reqs = project['requirements']

    count = len(reqs)
    high = sum(1 for r in reqs if r['priority'] == 'HIGH')
    med = sum(1 for r in reqs if r['priority'] == 'MEDIUM')
    low = sum(1 for r in reqs if r['priority'] == 'LOW')
    fr = sum(1 for r in reqs if r['type'] == 'Functional')
    nfr = sum(1 for r in reqs if r['type'] == 'Non-Functional')

    def list_to_html(items):
        if not items: return "<em>None defined</em>"
        return "<ul>" + "".join([f"<li>{i}</li>" for i in items]) + "</ul>"

    report = (
        f"<div style='background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; font-family: sans-serif;'>"
        f"<h2 style='color: #1e40af; margin-top: 0; border-bottom: 2px solid #1e40af; padding-bottom: 10px;'>"
        f"📋 FINAL REQUIREMENTS REPORT</h2>"
        f"<p style='color: #6b7280; font-size: 0.9em;'>"
        f"<strong>Project:</strong> {scope.get('name', 'Untitled')}<br>"
        f"<strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>"

        f"<h3 style='background: #f3f4f6; padding: 8px; border-radius: 4px; margin-top: 20px;'>1. PROJECT SCOPE</h3>"
        f"<p><strong>📝 Description:</strong><br>{scope.get('description', 'N/A')}</p>"
        f"<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>"
        f"  <div><strong>👥 Primary Users:</strong>{list_to_html(scope.get('primary_users', []))}</div>"
        f"  <div><strong>⚡ Key Features:</strong>{list_to_html(scope.get('main_features', []))}</div>"
        f"</div>"
        f"<div style='margin-top: 10px;'><strong>🚫 Out of Scope:</strong>{list_to_html(scope.get('excluded_features', []))}</div>"

        f"<h3 style='background: #f3f4f6; padding: 8px; border-radius: 4px; margin-top: 20px;'>2. CONSTRAINTS</h3>"
        f"<p><strong>⚙️ Technical:</strong> {', '.join(scope.get('technical_constraints', [])) or 'None'}</p>"
        f"<p><strong>💼 Business:</strong> {', '.join(scope.get('business_constraints', [])) or 'None'}</p>"

        f"<h3 style='background: #f3f4f6; padding: 8px; border-radius: 4px; margin-top: 20px;'>3. SUMMARY STATISTICS</h3>"
        f"<div style='display: flex; gap: 15px; background: #eff6ff; padding: 10px; border-radius: 6px;'>"
        f"  <span><strong>Total:</strong> {count}</span> | "
        f"  <span>🔴 <strong>High:</strong> {high}</span> | "
        f"  <span>🟡 <strong>Med:</strong> {med}</span> | "
        f"  <span>🟢 <strong>Low:</strong> {low}</span>"
        f"</div>"
        f"<div style='margin-top: 5px; font-size: 0.9em; color: #555;'>⚙️ Functional: {fr} &nbsp;|&nbsp; 📊 Non-Functional: {nfr}</div>"

        f"<h3 style='background: #f3f4f6; padding: 8px; border-radius: 4px; margin-top: 20px;'>4. DETAILED REQUIREMENTS</h3>"
        f"<table style='width: 100%; border-collapse: collapse; font-size: 0.9em;'>"
        f"  <tr style='background: #e5e7eb; text-align: left;'><th style='padding: 8px;'>ID</th><th style='padding: 8px;'>Priority</th><th style='padding: 8px;'>Requirement</th></tr>"
    )

    for req in reqs:
        prio_color = "#ef4444" if req['priority'] == "HIGH" else "#f59e0b" if req['priority'] == "MEDIUM" else "#10b981"
        report += (
            f"<tr style='border-bottom: 1px solid #eee;'>"
            f"  <td style='padding: 8px; font-weight: bold;'>{req['code']}</td>"
            f"  <td style='padding: 8px;'><span style='color: {prio_color}; font-weight: bold;'>{req['priority']}</span></td>"
            f"  <td style='padding: 8px;'>{req['description']}</td>"
            f"</tr>"
        )

    # ADD THE EXPORT BUTTON AT THE BOTTOM
    report += "</table>"

    if pid:
        # FIX: Use the full URL so it works even if you open the HTML file directly
        export_url = f"http://127.0.0.1:8000/api/projects/{pid}/export"

        report += (
            f"<div style='text-align: center; margin-top: 30px; margin-bottom: 20px;'>"
            f"  <a href='{export_url}' target='_blank' "
            f"     style='background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold; display: inline-flex; align-items: center; gap: 8px;'>"
            f"     📄 Export to PDF"
            f"  </a>"
            f"</div>"
        )
    report += "<hr style='margin-top: 20px; border: 0; border-top: 1px solid #ddd;'><p style='text-align: center; color: #888;'><em>👋 Session Ended. Thank you for using Precisely.</em></p></div>"
    return report
# ==========================================
# ENDPOINTS
# ==========================================

@app.post("/api/projects")
async def create_project(init: ProjectInit, user: dict = Depends(get_current_user)):
    pid = str(len(state.db) + 1)

    # Initialize with the provided Name and Description
    state.db[pid] = {
        "scope": {
            "name": init.name,
            "description": init.description,
            "primary_users": [],
            "main_features": [],
            "excluded_features": [],
            "technical_constraints": [],
            "business_constraints": []
        },
        "requirements": [],
        "messages": [],
        "status": "SETUP",

        # SKIP the first two steps (Name/Desc) since we have them
        "wizard_step": "ASK_USERS",

        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

    # Custom Intro that acknowledges the created project and jumps to Users
    intro = (
        f"{TUTORIAL_TEXT}<hr>"
        f"✅ Project <strong>{init.name}</strong> created.<br>"
        f"📝 Description: <em>{init.description}</em><br><br>"
        f"{'-' * 40}<br>"
        f"👥 <strong>STEP 1.2: PRIMARY USERS/STAKEHOLDERS</strong><br>"
        f"{'-' * 40}<br>"
        f"Who will use or interact with your system?<br>"
        f"<em>Examples: End Users, Admin, Doctor, External APIs</em><br><br>"
        f"📝 <strong>Enter user types (separated by commas):</strong>"
    )

    state.db[pid]['messages'].append({
        "sender": "bot",
        "text": intro,
        "created_at": datetime.now().isoformat()
    })

    state.data_manager.save_db(state.db)

    # Return the ACTUAL name so frontend displays it correctly
    return {"id": pid, "name": init.name}

@app.get("/api/projects/{pid}")
async def get_project(pid: str):
    if pid not in state.db: raise HTTPException(404)
    return state.db[pid]


@app.get("/api/projects/{pid}/messages")
async def get_messages(pid: str):
    if pid not in state.db: raise HTTPException(404)
    return state.db[pid]['messages']


@app.get("/api/projects/{pid}/requirements")
async def get_requirements(pid: str):
    if pid not in state.db: raise HTTPException(404)
    return state.db[pid]['requirements']


@app.post("/api/projects/{pid}/messages")
@app.post("/api/projects/{pid}/messages")
async def send_message(pid: str, msg: ChatMessage):
    """
    THE CORE BRAIN: Handles commands, Wizard logic, and the FULL ML Pipeline.
    """
    if pid not in state.db: raise HTTPException(404, "Project not found")

    # Standardize variable name to 'project'
    project = state.db[pid]
    user_text = msg.text.strip()

    # 1. Save User Message
    project['messages'].append({
        "sender": "user",
        "text": user_text,
        "created_at": datetime.now().isoformat()
    })

    bot_reply = ""
    new_req_added = False

    # --- COMMAND HANDLING ---
    text_lower = user_text.lower()

    if text_lower == 'checklist':
        bot_reply = CHECKLIST_TEXT

    elif text_lower == 'help':
        bot_reply = HELP_TEXT

    elif text_lower == 'scope':
        s = project['scope']
        bot_reply = (f"📌 <strong>CURRENT SCOPE:</strong><br>"
                     f"System: {s.get('name')}<br>"
                     f"Users: {', '.join(s.get('primary_users', []))}<br>"
                     f"Features: {', '.join(s.get('main_features', []))}")

    elif text_lower == 'summary':
        reqs = project['requirements']
        high = sum(1 for r in reqs if r['priority'] == 'HIGH')
        med = sum(1 for r in reqs if r['priority'] == 'MEDIUM')
        low = sum(1 for r in reqs if r['priority'] == 'LOW')
        bot_reply = (f"📊 <strong>SUMMARY:</strong><br>"
                     f"Total: {len(reqs)}<br>"
                     f"Priority: High={high}, Med={med}, Low={low}")

    elif text_lower == 'exit':
        bot_reply = generate_final_report(project)

    # --- WIZARD MODE ---
    elif project['status'] == 'SETUP':
        bot_reply = run_scope_wizard(project, user_text)

    # --- REQUIREMENT GATHERING MODE (The 4 Models) ---
    elif project['status'] == 'GATHERING':

        # A. PARSE REQUIREMENT
        parsed = state.parser.parse_requirement(user_text)
        cleaned_text = parsed['cleaned_text']

        if parsed['valid']:
            # B. SCOPE ALIGNMENT CHECK
            scope_res = state.scope_analyzer.check_alignment(cleaned_text, project['scope'])

            # C. RUN ALL 4 ML MODELS
            ml_results = state.classifier.classify_all(cleaned_text)

            intent_label, intent_conf = ml_results['intent']
            fr_nfr_label, fr_nfr_conf = ml_results['fr_nfr']
            qa_label, qa_conf = ml_results['qa']
            ambiguity_data = ml_results['ambiguity']

            # D. DETERMINE FINAL TEXT (Handle Ambiguity)
            final_text = cleaned_text
            ambiguity_html = ""

            if ambiguity_data['is_ambiguous']:
                final_text = ambiguity_data['correction']  # Use the corrected version
                ambiguity_html = (
                    f"<div style='background: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 0.9em;'>"
                    f"⚠️ <strong>AMBIGUITY DETECTED (Score: {ambiguity_data['score']:.2f})</strong><br>"
                    f"📝 <em>Original:</em> {user_text}<br>"
                    f"💡 <em>Rewritten for clarity:</em> {final_text}"
                    f"</div>"
                )
            else:
                ambiguity_html = (
                    f"<div style='color: #155724; margin-top: 5px; font-size: 0.9em;'>"
                    f"✅ Requirement is CLEAR (No ambiguity detected)"
                    f"</div>"
                )

            # E. CONSTRUCT ID & SAVE
            req_id = parsed['components']['req_id']
            if not req_id:
                req_id = f"REQ-{len(project['requirements']) + 1:03d}"

            new_req = {
                "id": str(len(project['requirements']) + 1),
                "code": req_id,
                "description": final_text,
                "priority": parsed['components']['priority'],
                "status": parsed['components']['status'],
                "type": "Functional" if fr_nfr_label == "FR" else "Non-Functional",
                "original": user_text,
                "analysis": {
                    "intent": intent_label, "intent_conf": intent_conf,
                    "fr_nfr": fr_nfr_label, "fr_nfr_conf": fr_nfr_conf,
                    "qa": qa_label, "qa_conf": qa_conf,
                    "ambiguity": ambiguity_data
                }
            }

            project['requirements'].append(new_req)
            new_req_added = True

            # F. GENERATE DETAILED HTML REPLY
            bot_reply = (
                f"✅ <strong>REQUIREMENT ACCEPTED: {req_id}</strong><br>"
                f"Priority: <span class='badge'>{new_req['priority']}</span> | "
                f"Status: <span class='badge'>{new_req['status']}</span><br><br>"

                f"📝 <strong>{final_text}</strong>"
                f"{ambiguity_html}"

                f"<hr style='margin: 10px 0; border: 0; border-top: 1px solid #eee;'>"
                f"🔍 <strong>ML CLASSIFICATION RESULTS:</strong><br>"

                f"📌 <strong>Intent:</strong> {intent_label} (Conf: {intent_conf})<br>"
                f"📊 <strong>Type:</strong> {fr_nfr_label} (Conf: {fr_nfr_conf})<br>"
            )

            if qa_label:
                bot_reply += f"⭐ <strong>Quality Attribute:</strong> {qa_label} (Conf: {qa_conf})<br>"

            if scope_res.get('warning'):
                bot_reply += f"⚠️ <strong>Scope Warning:</strong> {scope_res['warning']}"
            else:
                bot_reply += f"🎯 <strong>Scope:</strong> Aligned ({scope_res['reason']})"

            # Constant Reminder
            bot_reply += "<br><br><em>Commands: 'checklist', 'summary', 'exit'</em>"

        else:
            # Validation Failed
            issues_html = "".join([f"<li>❌ {i}</li>" for i in parsed['issues']])
            warnings_html = "".join([f"<li>💡 {w}</li>" for w in parsed['warnings']])

            bot_reply = (
                f"⚠️ <strong>VALIDATION ISSUES:</strong>"
                f"<ul>{issues_html}</ul>"
            )
            if parsed['warnings']:
                bot_reply += f"<strong>SUGGESTIONS:</strong><ul>{warnings_html}</ul>"

    # 3. Save Bot Reply
    project['messages'].append({
        "sender": "bot",
        "text": bot_reply,
        "created_at": datetime.now().isoformat()
    })

    state.data_manager.save_db(state.db)

    return {"botReply": bot_reply, "newRequirementsDetected": new_req_added}


# ==========================================
# DELETE REQUIREMENT
# ==========================================
@app.delete("/api/requirements/{req_id}")
async def delete_requirement(req_id: str):
    """Deletes a requirement by ID from ALL projects"""
    for pid, data in state.db.items():
        data['requirements'] = [r for r in data['requirements'] if r['id'] != req_id]
    state.data_manager.save_db(state.db)
    return {"status": "deleted"}


@app.get("/api/projects/{pid}/export")
async def export_pdf(pid: str):
    if pid not in state.db: raise HTTPException(404)

    # 1. Get the HTML Report Logic
    project = state.db[pid]
    # We reuse the logic, but strip the "NEXT STEPS" footer for the PDF version
    report_html = generate_final_report(project)

    # 2. Add Basic Styling for PDF (xhtml2pdf needs simple CSS)
    pdf_template = f"""
    <html>
    <head>
        <style>
            @page {{ size: A4; margin: 2cm; }}
            body {{ font-family: Helvetica, sans-serif; font-size: 12px; color: #333; }}
            h2 {{ color: #1e40af; border-bottom: 1px solid #1e40af; padding-bottom: 5px; }}
            h3 {{ background-color: #f3f4f6; padding: 5px; color: #111; margin-top: 20px; }}
            table {{ width: 100%; border: 1px solid #ddd; border-collapse: collapse; margin-top: 10px; }}
            th {{ background-color: #e5e7eb; padding: 8px; text-align: left; font-weight: bold; }}
            td {{ padding: 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
            .stats {{ background-color: #eff6ff; padding: 10px; border: 1px solid #bfdbfe; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        {report_html}
    </body>
    </html>
    """

    # 3. Convert to PDF
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(pdf_template, dest=pdf_buffer)

    if pisa_status.err:
        raise HTTPException(500, "PDF generation failed")

    pdf_buffer.seek(0)

    # 4. Return as Download
    filename = f"Requirements_Report_{project['scope'].get('name', 'Project').replace(' ', '_')}.pdf"
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    import uvicorn

    # Mount Static for the frontend HTML/CSS/JS
    if os.path.exists(STATIC_DIR):
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    print(f"✅ Backend running. Models loaded from {MODELS_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000)