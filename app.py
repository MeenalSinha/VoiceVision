import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from gtts import gTTS
import os
import tempfile
import re
from datetime import datetime
import time

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="VoiceVision - AI Accessibility Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GLASSMORPHISM + PASTEL UI THEME (Matching RoadGuardian)
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Pastel gradient background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Glassmorphism sidebar */
    [data-testid="stSidebar"] {
        background: rgba(249, 229, 216, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    
    /* Headers with gradient */
    h1, h2, h3, h4, h5, h6 {
        color: #6A5D7B !important;
        font-weight: 700;
    }
    
    h1 {
        background: linear-gradient(135deg, #6A5D7B 0%, #8B7E99 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.4);
        transition: all 0.3s ease;
        animation: fadeIn 0.6s ease-out;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Hero section with glassmorphism */
    .hero-section {
        background: linear-gradient(135deg, rgba(200, 184, 219, 0.6), rgba(163, 201, 168, 0.6));
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 2rem;
        animation: heroFadeIn 1s ease-out;
    }
    
    @keyframes heroFadeIn {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .hero-logo {
        font-size: 4rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .hero-title {
        color: white !important;
        font-size: 3.5rem;
        font-weight: 900;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .hero-subtitle {
        color: white;
        font-size: 1.5rem;
        font-weight: 400;
        opacity: 0.95;
    }
    
    /* Pastel buttons */
    .stButton>button {
        background: linear-gradient(135deg, #A3C9A8 0%, #B8D4BE 100%);
        color: white;
        border-radius: 15px;
        height: 3.5em;
        width: 100%;
        font-size: 1.1em;
        font-weight: 700;
        border: none;
        box-shadow: 0 4px 15px rgba(163, 201, 168, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #9EB5A5 0%, #B0C8B7 100%);
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(163, 201, 168, 0.5);
    }
    
    /* Metric cards with glassmorphism */
    .metric-glass-card {
        background: linear-gradient(135deg, rgba(200, 184, 219, 0.7), rgba(212, 196, 232, 0.7));
        backdrop-filter: blur(15px);
        padding: 1.8rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-glass-card:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 12px 40px rgba(200, 184, 219, 0.4);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 900;
        margin: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.95;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Alert boxes with glassmorphism */
    .glass-alert-success {
        background: rgba(212, 241, 221, 0.7);
        backdrop-filter: blur(10px);
        border-left: 5px solid #A3C9A8;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(163, 201, 168, 0.2);
    }
    
    .glass-alert-warning {
        background: rgba(255, 243, 205, 0.7);
        backdrop-filter: blur(10px);
        border-left: 5px solid #F9C74F;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(249, 199, 79, 0.2);
    }
    
    .glass-alert-danger {
        background: rgba(255, 229, 229, 0.7);
        backdrop-filter: blur(10px);
        border-left: 5px solid #F4978E;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(244, 151, 142, 0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .glass-alert-info {
        background: rgba(227, 242, 253, 0.7);
        backdrop-filter: blur(10px);
        border-left: 5px solid #90CAF9;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(144, 202, 249, 0.2);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #A3C9A8 0%, #C8B8DB 100%);
        animation: progressGlow 2s ease-in-out infinite;
    }
    
    @keyframes progressGlow {
        0%, 100% { box-shadow: 0 0 10px rgba(163, 201, 168, 0.5); }
        50% { box-shadow: 0 0 20px rgba(200, 184, 219, 0.7); }
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(245, 223, 208, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 15px 15px 0 0;
        color: #6A5D7B;
        padding: 12px 24px;
        font-weight: 700;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(245, 223, 208, 0.8);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(200, 184, 219, 0.8), rgba(212, 196, 232, 0.8));
        color: white;
        box-shadow: 0 4px 15px rgba(200, 184, 219, 0.3);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 238, 248, 0.5);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(200, 184, 219, 0.6);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #C8B8DB;
        background: rgba(255, 238, 248, 0.7);
    }
    
    /* Tech badge styling */
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #A3C9A8, #C8B8DB);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Footer with glassmorphism */
    .footer {
        background: rgba(234, 231, 220, 0.6);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 3rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Loader animation */
    .loader {
        border: 4px solid rgba(163, 201, 168, 0.3);
        border-top: 4px solid #A3C9A8;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = True
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'last_message' not in st.session_state:
    st.session_state.last_message = ""
if 'reader' not in st.session_state:
    st.session_state.reader = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en', 'hi'])

def speak_text(text, lang='en'):
    if st.session_state.audio_enabled and text:
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                tts.save(fp.name)
                st.audio(fp.name, format='audio/mp3', autoplay=True)
                st.session_state.last_message = text
        except Exception as e:
            st.error(f"Audio error: {e}")

def detect_currency(image):
    reader = st.session_state.reader
    if reader is None:
        reader = load_ocr_reader()
        st.session_state.reader = reader
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    start_time = time.time()
    results = reader.readtext(image)
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    
    denominations = [2000, 500, 200, 100, 50, 20, 10]
    detected = []
    
    for (bbox, text, confidence) in results:
        clean_text = re.sub(r'[^\d]', '', text)
        for denom in denominations:
            if str(denom) in clean_text and confidence > 0.3:
                detected.append((denom, confidence))
    
    if detected:
        detected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return detected[0][0], detected[0][1], processing_time
    
    return None, 0.0, processing_time

def extract_text(image):
    reader = st.session_state.reader
    if reader is None:
        reader = load_ocr_reader()
        st.session_state.reader = reader
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    start_time = time.time()
    results = reader.readtext(image)
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    
    if not results:
        return None, [], processing_time
    
    text_lines = []
    for (bbox, text, confidence) in results:
        if confidence > 0.3:
            text_lines.append((text, confidence))
    
    full_text = " ".join([text for text, _ in text_lines])
    return full_text, text_lines, processing_time

def detect_objects_simple(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    start_time = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w/2
            img_width = image.shape[1]
            
            if center_x < img_width/3:
                position = "left"
            elif center_x > 2*img_width/3:
                position = "right"
            else:
                position = "center"
            
            objects.append({
                'type': 'obstacle',
                'bbox': (x, y, w, h),
                'position': position,
                'confidence': 0.75
            })
    
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    return objects, processing_time

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # ========================================================================
    # HERO SECTION
    # ========================================================================
    st.markdown("""
    <div class="hero-section">
        <div class="hero-logo">👁️</div>
        <h1 class="hero-title">VoiceVision</h1>
        <p class="hero-subtitle">
            AI Accessibility Assistant for Indian Context
        </p>
        <div style="margin-top: 1.5rem;">
            <span class="tech-badge">🤖 EasyOCR</span>
            <span class="tech-badge">🗣️ Hindi + English</span>
            <span class="tech-badge">💰 Currency Detection</span>
            <span class="tech-badge">♿ Accessibility First</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # IMPACT BANNER
    # ========================================================================
    st.markdown("""
    <div class="glass-alert-info">
        <strong>📊 Mission:</strong> Empowering 8M+ visually impaired Indians with AI-powered assistance 
        for currency recognition, text reading, and spatial awareness.
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ ACCESSIBILITY NOTICE
    st.markdown("""
    <div class="glass-alert-success">
        <strong>♿ Accessibility:</strong> All features are keyboard-navigable and screen-reader compatible. 
        Press <kbd>Tab</kbd> to navigate, <kbd>Enter</kbd> to activate buttons.
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR CONTROLS
    # ========================================================================
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        audio_enabled = st.toggle("🔊 Audio Feedback", value=st.session_state.audio_enabled)
        st.session_state.audio_enabled = audio_enabled
        
        language = st.selectbox(
            "🌐 Voice Language",
            options=['en', 'hi'],
            format_func=lambda x: "English" if x == 'en' else "हिंदी",
            index=0 if st.session_state.language == 'en' else 1
        )
        st.session_state.language = language
        
        st.markdown("---")
        
        # ✅ MODEL STATUS INDICATOR
        st.markdown("### 🔍 Model Status")
        st.markdown("""
        <div style="background: rgba(212, 241, 221, 0.5); padding: 0.5rem; border-radius: 10px; margin: 0.3rem 0;">
            <strong style="color: #A3C9A8;">✅ OCR Engine:</strong> <span style="color: #666;">Active (EasyOCR)</span>
        </div>
        <div style="background: rgba(255, 243, 205, 0.5); padding: 0.5rem; border-radius: 10px; margin: 0.3rem 0;">
            <strong style="color: #F9C74F;">⚠️ Currency Detection:</strong> <span style="color: #666;">Demo Mode (OCR-based)</span>
        </div>
        <div style="background: rgba(255, 243, 205, 0.5); padding: 0.5rem; border-radius: 10px; margin: 0.3rem 0;">
            <strong style="color: #F9C74F;">⚠️ Object Detection:</strong> <span style="color: #666;">Simplified (Edge-based)</span>
        </div>
        <div style="background: rgba(212, 241, 221, 0.5); padding: 0.5rem; border-radius: 10px; margin: 0.3rem 0;">
            <strong style="color: #A3C9A8;">✅ Audio TTS:</strong> <span style="color: #666;">Active (gTTS)</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("💡 **Demo Mode**: Uses OCR pattern matching instead of trained ML models")
        
        st.markdown("---")
        
        st.markdown("### 🎯 Select Mode")
        mode = st.radio(
            "Choose assistance type:",
            options=[
                "💰 Currency Reader",
                "📝 Text Reader (OCR)",
                "👁️ Object Awareness",
                "🧭 Navigation Assist"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        with st.expander("⚠️ Prototype Limitations"):
            st.markdown("""
            **Current Limitations:**
            - Requires good lighting
            - Works best with clear images
            - OCR accuracy depends on text clarity
            - Object detection is simplified
            
            **Future Scope:**
            - Mobile app deployment
            - Offline functionality
            - Advanced YOLO integration
            - GPS-based navigation
            - More regional languages
            """)
        
        if st.session_state.last_message:
            if st.button("🔄 Replay Last Message"):
                speak_text(st.session_state.last_message, st.session_state.language)
    
    # ========================================================================
    # NAVIGATION TABS
    # ========================================================================
    tab1, tab2, tab3 = st.tabs([
        "🏠 Home",
        "🎯 Detect",
        "ℹ️ About"
    ])
    
    # ========================================================================
    # TAB 1: HOME
    # ========================================================================
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h2 style="text-align: center; color: #F4978E !important;">💡</h2>
                <h4 style="text-align: center;">The Problem</h4>
                <p style="text-align: center; color: #666;">
                    8M+ visually impaired Indians struggle with currency identification, 
                    reading signage, and spatial navigation daily.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h2 style="text-align: center; color: #A3C9A8 !important;">🤖</h2>
                <h4 style="text-align: center;">Our Solution</h4>
                <p style="text-align: center; color: #666;">
                    Real-time AI-powered assistance with currency detection, 
                    Hindi+English OCR, and audio feedback.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="glass-card">
                <h2 style="text-align: center; color: #C8B8DB !important;">🎯</h2>
                <h4 style="text-align: center;">Impact</h4>
                <p style="text-align: center; color: #666;">
                    Enables independent daily living, financial autonomy, 
                    and improved mobility for users.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ✅ WHO BENEFITS SECTION
        st.markdown("### 👥 Who Benefits from VoiceVision?")
        
        st.markdown("""
        <div class="glass-card">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div style="background: rgba(163, 201, 168, 0.2); padding: 1rem; border-radius: 10px; text-align: center;">
                    <h3 style="color: #A3C9A8; margin: 0;">Primary Users</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #666; font-weight: 600;">Visually Impaired Individuals</p>
                    <p style="margin: 0.3rem 0 0 0; color: #888; font-size: 0.9rem;">Complete blindness or low vision</p>
                </div>
                <div style="background: rgba(200, 184, 219, 0.2); padding: 1rem; border-radius: 10px; text-align: center;">
                    <h3 style="color: #C8B8DB; margin: 0;">Secondary Users</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #666; font-weight: 600;">Elderly & Low-Vision Users</p>
                    <p style="margin: 0.3rem 0 0 0; color: #888; font-size: 0.9rem;">Age-related vision loss</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 📈 Key Features")
        
        col1, col2, col3, col4 = st.columns(4)
        
        features = [
            ("Currency", "₹10-₹2000", "All Notes"),
            ("Languages", "2+", "EN + HI"),
            ("Audio", "Real-time", "Instant"),
            ("Modes", "4", "Complete")
        ]
        
        for col, (label, value, sublabel) in zip([col1, col2, col3, col4], features):
            with col:
                st.markdown(f"""
                <div class="metric-glass-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label" style="font-size: 0.8rem;">{sublabel}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 2: DETECT (MAIN DETECTION INTERFACE)
    # ========================================================================
    with tab2:
        st.markdown("""
        <div class="glass-alert-success">
            <strong>✅ System Ready:</strong> EasyOCR loaded with Hindi + English support | 
            Audio feedback enabled
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        if "💰 Currency Reader" in mode:
            st.markdown("## 💰 Indian Currency Recognition")
            st.markdown("""
            <div class="glass-alert-info">
                <strong>How it works:</strong> Capture or upload an image of an Indian currency note. 
                The AI will identify the denomination and announce it audibly.
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                input_method = st.radio("Input method:", ["📷 Camera", "📁 Upload"], horizontal=True)
                
                if input_method == "📷 Camera":
                    camera_image = st.camera_input("Capture currency note")
                    if camera_image:
                        image = Image.open(camera_image)
                else:
                    uploaded_file = st.file_uploader("Upload currency image", type=['jpg', 'jpeg', 'png'])
                    if uploaded_file:
                        image = Image.open(uploaded_file)
                    else:
                        image = None
                
                if 'image' in locals() and image:
                    st.image(image, caption="Captured Image", use_container_width=True)
                    
                    if st.button("🔍 Detect Denomination", use_container_width=True):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        with st.spinner("Analyzing currency..."):
                            denomination, confidence, proc_time = detect_currency(image)
                            
                            if denomination:
                                st.markdown(f"""
                                <div class="glass-alert-success">
                                    <h3 style="margin:0; color: #6A5D7B;">✅ Detected: ₹{denomination}</h3>
                                    <p style="margin-top: 0.5rem;">Confidence: {confidence*100:.1f}%</p>
                                    <p style="margin-top: 0.3rem; font-size: 0.9rem; opacity: 0.8;">⚡ Processed in {proc_time:.0f}ms</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if confidence > 0.5:
                                    message = f"Detected: {denomination} rupees note"
                                    if st.session_state.language == 'hi':
                                        message = f"{denomination} रुपये का नोट पहचाना गया"
                                    speak_text(message, st.session_state.language)
                                    st.balloons()
                                else:
                                    st.warning("⚠️ Low confidence detection. Please recapture in better lighting.")
                            else:
                                st.markdown("""
                                <div class="glass-alert-warning">
                                    <strong>⚠️ Unable to detect denomination</strong><br>
                                    Please ensure the note is clearly visible and well-lit.
                                </div>
                                """, unsafe_allow_html=True)
                                speak_text("Unable to read currency note. Please try again.", st.session_state.language)
            
            with col2:
                st.markdown("### 📊 Supported Notes")
                notes = [10, 20, 50, 100, 200, 500, 2000]
                for note in notes:
                    st.markdown(f'<span class="tech-badge">₹{note}</span>', unsafe_allow_html=True)
        
        elif "📝 Text Reader" in mode:
            st.markdown("## 📝 Multilingual Text Reader")
            st.markdown("""
            <div class="glass-alert-info">
                <strong>How it works:</strong> Capture text in English or Hindi. 
                The AI will extract and read it aloud.
            </div>
            """, unsafe_allow_html=True)
            
            input_method = st.radio("Input method:", ["📷 Camera", "📁 Upload"], horizontal=True)
            
            if input_method == "📷 Camera":
                camera_image = st.camera_input("Capture text")
                if camera_image:
                    image = Image.open(camera_image)
            else:
                uploaded_file = st.file_uploader("Upload image with text", type=['jpg', 'jpeg', 'png'])
                if uploaded_file:
                    image = Image.open(uploaded_file)
                else:
                    image = None
            
            if 'image' in locals() and image:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image, caption="Captured Image", use_container_width=True)
                
                with col2:
                    if st.button("📖 Extract & Read Text", use_container_width=True):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        with st.spinner("Extracting text..."):
                            full_text, text_lines, proc_time = extract_text(image)
                            
                            if full_text:
                                st.markdown("""
                                <div class="glass-alert-success">
                                    <strong>✅ Extracted Text:</strong>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div style="background: rgba(255,255,255,0.3); 
                                            padding: 20px; 
                                            border-radius: 12px; 
                                            color: #6A5D7B;
                                            font-size: 1.1rem;
                                            margin: 15px 0;">
                                    {full_text}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div style="opacity: 0.7; font-size: 0.9rem; margin-top: 0.5rem;">
                                    ⚡ Processed in {proc_time:.0f}ms
                                </div>
                                """, unsafe_allow_html=True)
                                
                                speak_text(full_text, st.session_state.language)
                                st.balloons()
                            else:
                                st.markdown("""
                                <div class="glass-alert-warning">
                                    <strong>⚠️ No text detected</strong><br>
                                    Please ensure text is clearly visible.
                                </div>
                                """, unsafe_allow_html=True)
                                speak_text("No text found in image", st.session_state.language)
        
        elif "👁️ Object Awareness" in mode:
            st.markdown("## 👁️ Real-Time Object Detection")
            st.markdown("""
            <div class="glass-alert-info">
                <strong>How it works:</strong> Detects obstacles and objects in the environment 
                to help with spatial awareness.
            </div>
            """, unsafe_allow_html=True)
            
            input_method = st.radio("Input method:", ["📷 Camera", "📁 Upload"], horizontal=True)
            
            if input_method == "📷 Camera":
                camera_image = st.camera_input("Capture surroundings")
                if camera_image:
                    image = Image.open(camera_image)
            else:
                uploaded_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
                if uploaded_file:
                    image = Image.open(uploaded_file)
                else:
                    image = None
            
            if 'image' in locals() and image:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if st.button("🔍 Detect Objects", use_container_width=True):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        with st.spinner("Analyzing environment..."):
                            objects, proc_time = detect_objects_simple(image)
                            
                            img_array = np.array(image)
                            for obj in objects:
                                x, y, w, h = obj['bbox']
                                cv2.rectangle(img_array, (x, y), (x+w, y+h), (163, 201, 168), 2)
                                cv2.putText(img_array, f"{obj['type']} ({obj['position']})", 
                                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (163, 201, 168), 2)
                            
                            st.image(img_array, caption="Detected Objects", use_container_width=True)
                            
                            if objects:
                                st.markdown(f"""
                                <div class="glass-alert-success">
                                    <strong>✅ Detected {len(objects)} object(s)</strong>
                                    <p style="margin-top: 0.3rem; font-size: 0.9rem; opacity: 0.8;">⚡ Processed in {proc_time:.0f}ms</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                positions = [obj['position'] for obj in objects]
                                message = f"Detected {len(objects)} obstacles. "
                                if 'center' in positions:
                                    message += "Obstacle ahead in center. "
                                if 'left' in positions:
                                    message += "Obstacle on left. "
                                if 'right' in positions:
                                    message += "Obstacle on right. "
                                
                                speak_text(message, st.session_state.language)
                                st.balloons()
                            else:
                                st.markdown("""
                                <div class="glass-alert-info">
                                    <strong>ℹ️ Path appears clear</strong>
                                </div>
                                """, unsafe_allow_html=True)
                                speak_text("Path appears clear", st.session_state.language)
                
                with col2:
                    st.image(image, caption="Original Image", use_container_width=True)
        
        else:  # Navigation Assist
            st.markdown("## 🧭 Navigation Assistance")
            st.markdown("""
            <div class="glass-alert-info">
                <strong>How it works:</strong> Provides directional guidance based on detected obstacles.
            </div>
            """, unsafe_allow_html=True)
            
            input_method = st.radio("Input method:", ["📷 Camera", "📁 Upload"], horizontal=True)
            
            if input_method == "📷 Camera":
                camera_image = st.camera_input("Capture path ahead")
                if camera_image:
                    image = Image.open(camera_image)
            else:
                uploaded_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
                if uploaded_file:
                    image = Image.open(uploaded_file)
                else:
                    image = None
            
            if 'image' in locals() and image:
                st.image(image, caption="Path View", use_container_width=True)
                
                if st.button("🧭 Get Navigation Guidance", use_container_width=True):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    with st.spinner("Analyzing path..."):
                        objects, proc_time = detect_objects_simple(image)
                        
                        if objects:
                            positions = [obj['position'] for obj in objects]
                            
                            if 'center' in positions:
                                if 'left' not in positions:
                                    guidance = "Obstacle ahead. Move left."
                                    direction = "⬅️ LEFT"
                                    alert_class = "glass-alert-warning"
                                elif 'right' not in positions:
                                    guidance = "Obstacle ahead. Move right."
                                    direction = "➡️ RIGHT"
                                    alert_class = "glass-alert-warning"
                                else:
                                    guidance = "Multiple obstacles detected. Proceed with caution."
                                    direction = "⚠️ CAUTION"
                                    alert_class = "glass-alert-danger"
                            elif 'left' in positions:
                                guidance = "Obstacle on left. Move right."
                                direction = "➡️ RIGHT"
                                alert_class = "glass-alert-warning"
                            elif 'right' in positions:
                                guidance = "Obstacle on right. Move left."
                                direction = "⬅️ LEFT"
                                alert_class = "glass-alert-warning"
                            else:
                                guidance = "Path appears clear. Proceed carefully."
                                direction = "✅ CLEAR"
                                alert_class = "glass-alert-success"
                            
                            st.markdown(f"""
                            <div class="{alert_class}">
                                <h2 style="margin: 0; color: #6A5D7B;">{direction}</h2>
                                <p style="margin: 10px 0 0 0; font-size: 1.2rem;">{guidance}</p>
                                <p style="margin: 5px 0 0 0; font-size: 0.9rem; opacity: 0.8;">⚡ Processed in {proc_time:.0f}ms</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            speak_text(guidance, st.session_state.language)
                        else:
                            st.markdown("""
                            <div class="glass-alert-success">
                                <h2 style="margin: 0; color: #6A5D7B;">✅ PATH CLEAR</h2>
                                <p style="margin: 10px 0 0 0; font-size: 1.2rem;">No obstacles detected ahead.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            speak_text("Path clear. Proceed carefully.", st.session_state.language)
                            st.balloons()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 3: ABOUT
    # ========================================================================
    with tab3:
        st.markdown("### ℹ️ About VoiceVision")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #6A5D7B !important;">🤖 System Overview</h4>
                <p style="color: #666; line-height: 1.8;">
                VoiceVision is an AI-powered accessibility platform designed specifically for 
                the Indian context. It helps visually impaired users with currency identification, 
                text reading in Hindi and English, object detection, and basic navigation guidance.
                </p>
                <p style="color: #666; line-height: 1.8;">
                The system uses EasyOCR for multilingual text recognition, computer vision for 
                obstacle detection, and text-to-speech for audio feedback. All features work 
                together to promote independent daily living.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-glass-card">
                <h4 style="color: white !important;">📊 Key Stats</h4>
                <div style="margin: 1rem 0;">
                    <div class="metric-value">4</div>
                    <div class="metric-label">Modes</div>
                </div>
                <div style="margin: 1rem 0;">
                    <div class="metric-value">2</div>
                    <div class="metric-label">Languages</div>
                </div>
                <div style="margin: 1rem 0;">
                    <div class="metric-value">7</div>
                    <div class="metric-label">Currency Notes</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 🛠️ Technology Stack")
        
        tech_categories = {
            "AI/ML": ["EasyOCR", "OpenCV", "NumPy", "PIL"],
            "Audio": ["gTTS", "Real-time TTS", "Hindi Support"],
            "Frontend": ["Streamlit", "Custom CSS", "Glassmorphism"],
            "Features": ["Currency Detection", "Multilingual OCR", "Navigation", "Audio Feedback"]
        }
        
        cols = st.columns(4)
        for col, (category, techs) in zip(cols, tech_categories.items()):
            with col:
                tech_list = "".join([f'<span class="tech-badge">{t}</span>' for t in techs])
                st.markdown(f"""
                <div class="glass-card">
                    <h5 style="color: #6A5D7B !important; text-align: center;">{category}</h5>
                    <div style="text-align: center; margin-top: 1rem;">
                        {tech_list}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("#### 📋 Feature Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align: center; color: #A3C9A8 !important;">💰 Currency Reader</h4>
                <ul style="color: #666;">
                    <li>₹10 to ₹2000 detection</li>
                    <li>OCR-based recognition</li>
                    <li>Confidence scoring</li>
                    <li>Audio announcement</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align: center; color: #C8B8DB !important;">📝 Text Reader</h4>
                <ul style="color: #666;">
                    <li>English + Hindi support</li>
                    <li>Signboard reading</li>
                    <li>Document scanning</li>
                    <li>Real-time audio output</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="glass-card">
                <h4 style="text-align: center; color: #F4978E !important;">🧭 Navigation</h4>
                <ul style="color: #666;">
                    <li>Obstacle detection</li>
                    <li>Directional guidance</li>
                    <li>Spatial awareness</li>
                    <li>Safety alerts</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### ⚠️ Known Limitations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="glass-alert-warning">
                <strong>Technical Limitations:</strong>
                <ul style="margin-top: 0.5rem;">
                    <li>Requires good lighting conditions</li>
                    <li>Currency detection via OCR (not trained model)</li>
                    <li>Object detection is rule-based</li>
                    <li>No offline functionality yet</li>
                    <li>Desktop/web only (no mobile app)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-alert-info">
                <strong>Future Enhancements:</strong>
                <ul style="margin-top: 0.5rem;">
                    <li>Trained YOLO model for currency</li>
                    <li>Offline TTS and OCR</li>
                    <li>Mobile app (Android/iOS)</li>
                    <li>GPS-based navigation</li>
                    <li>More regional languages</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 👥 Target Users")
        
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #6A5D7B !important;">Who Benefits from VoiceVision?</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div style="background: rgba(163, 201, 168, 0.2); padding: 1rem; border-radius: 10px;">
                    <strong>👁️ Visually Impaired</strong>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">Complete blindness or low vision users</p>
                </div>
                <div style="background: rgba(200, 184, 219, 0.2); padding: 1rem; border-radius: 10px;">
                    <strong>👴 Elderly</strong>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">Age-related vision loss</p>
                </div>
                <div style="background: rgba(249, 199, 79, 0.2); padding: 1rem; border-radius: 10px;">
                    <strong>🎓 Students</strong>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">Assistive learning tools</p>
                </div>
                <div style="background: rgba(244, 151, 142, 0.2); padding: 1rem; border-radius: 10px;">
                    <strong>🏪 Shopkeepers</strong>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">Currency verification assistance</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🌟 Social Impact")
        
        impact_metrics = [
            ("8M+", "Target Users", "Visually impaired in India"),
            ("4", "Core Features", "Complete assistance suite"),
            ("2", "Languages", "English + Hindi"),
            ("24/7", "Availability", "Always accessible")
        ]
        
        cols = st.columns(4)
        for col, (value, label, desc) in zip(cols, impact_metrics):
            with col:
                st.markdown(f"""
                <div class="metric-glass-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                    <p style="font-size: 0.75rem; margin-top: 0.5rem; opacity: 0.8;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        <div class="hero-logo" style="font-size: 2.5rem;">👁️</div>
        <h3 style="color: #6A5D7B !important; margin: 1rem 0;">VoiceVision</h3>
        <p style="font-size: 1.2rem; color: #8E8D8A; margin-bottom: 1.5rem;">
            AI Accessibility Assistant for Indian Context
        </p>
        <div style="margin: 1.5rem 0;">
            <span class="tech-badge">🤖 EasyOCR</span>
            <span class="tech-badge">🗣️ Multilingual</span>
            <span class="tech-badge">💰 Currency Detection</span>
            <span class="tech-badge">♿ Accessibility First</span>
        </div>
        <p style="opacity: 0.7; font-size: 0.95rem; margin-top: 2rem;">
            Empowering Independence Through AI<br>
            Powered by EasyOCR + Streamlit + OpenCV | © 2024 VoiceVision
        </p>
        <p style="opacity: 0.6; font-size: 0.85rem; margin-top: 1rem;">
            Version 1.0.0 | Prototype Status
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()