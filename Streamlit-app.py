import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime, timedelta
import pickle
import pandas as pd
import time
from PIL import Image
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import hashlib
from typing import List, Tuple, Optional
import logging
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background: #0f1115;
    }
    .content-box {
        background: #181a1f;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #2a2e35;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #181a1f;
        padding: 1.5rem;
        border-radius: 10px;
        color: #e5e7eb;
        text-align: center;
        border: 1px solid #2a2e35;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #9aa3af;
    }
    h1, h2, h3 {
        color: #f3f4f6;
    }
    .stButton>button {
        width: 100%;
        background: #1e3a8a;
        color: #ffffff;
        border: 1px solid #1e3a8a;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }
    .stButton>button:hover {
        background: #2563eb; /* match Download CSV hover */
        border-color: #2563eb;
        color: #ffffff !important;
    }
    .stButton>button:focus {
        outline: none;
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.35);
        color: #ffffff !important;
    }
    .stButton>button:active {
        background: #1e40af;
        border-color: #1e40af;
        color: #ffffff !important;
    }
    .stButton>button:disabled, .stButton>button[disabled] {
        background: #1e3a8a;
        border-color: #1e3a8a;
        color: #ffffff !important;
        opacity: 0.6;
        cursor: not-allowed;
    }
    .stButton>button:focus, .stButton>button:active {
        color: #ffffff !important;
    }
    .success-box {
        background: #111315;
        color: #e5e7eb;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #22c55e;
    }
    .warning-box {
        background: #111315;
        color: #e5e7eb;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #f59e0b;
    }
    .info-box {
        background: #111315;
        color: #e5e7eb;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #2563eb;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    div[data-testid="stSidebar"] {
        background: #111315;
    }
    div[data-testid="stSidebar"] .css-1d391kg {
        color: #e5e7eb;
    }
    /* Inputs and selects focus state to navy */
    input:focus, textarea:focus, select:focus {
        border-color: #1e3a8a !important;
        box-shadow: 0 0 0 1px #1e3a8a inset, 0 0 0 3px rgba(30, 58, 138, 0.25) !important;
    }
    /* Download button styled like primary buttons */
    div[data-testid="stDownloadButton"] > button {
        background: #1e3a8a;
        color: #ffffff;
        border: 1px solid #1e3a8a;
        border-radius: 6px;
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: #2563eb; /* dashboard blue */
        border-color: #2563eb;
        color: #ffffff !important;
    }
    /* Multiselect selected tags (Filter by People) to navy */
    div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background-color: #1e3a8a !important;
        color: #ffffff !important;
        border-color: #1e3a8a !important;
    }
    div[data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    div[data-testid="stMultiSelect"] [data-baseweb="tag"]:hover {
        background-color: #152c6b !important;
        border-color: #152c6b !important;
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_attendance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handle SQLite database operations"""
    
    def __init__(self, db_path: str = "attendance.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    encoding_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    date DATE NOT NULL,
                    time TIME NOT NULL,
                    status TEXT DEFAULT 'Present',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES people (id),
                    UNIQUE(person_id, date)
                )
            """)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def add_person(self, name: str, encoding_hash: str) -> bool:
        """Add a person to the database"""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    "INSERT INTO people (name, encoding_hash) VALUES (?, ?)",
                    (name, encoding_hash)
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False
    
    def get_people(self) -> List[dict]:
        """Get all people from database"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM people ORDER BY name")
            return [dict(row) for row in cursor.fetchall()]
    
    def remove_person(self, person_id: int) -> bool:
        """Remove a person from database"""
        try:
            with self.get_connection() as conn:
                conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
                conn.execute("DELETE FROM attendance WHERE person_id = ?", (person_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error removing person: {e}")
            return False
    
    def mark_attendance(self, person_id: int, name: str) -> bool:
        """Mark attendance for a person"""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H:%M:%S")
            
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO attendance (person_id, date, time, status)
                    VALUES (?, ?, ?, 'Present')
                """, (person_id, current_date, current_time))
                
                if conn.total_changes > 0:
                    conn.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Error marking attendance: {e}")
            return False
    
    def get_attendance_records(self, start_date: str = None, end_date: str = None) -> List[dict]:
        """Get attendance records"""
        query = """
            SELECT p.name, a.date, a.time, a.status, a.created_at
            FROM attendance a
            JOIN people p ON a.person_id = p.id
        """
        params = []
        
        if start_date and end_date:
            query += " WHERE a.date BETWEEN ? AND ?"
            params.extend([start_date, end_date])
        
        query += " ORDER BY a.date DESC, a.time DESC"
        
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting attendance records: {e}")
            return []

class FaceAttendanceSystem:
    def __init__(self):
        # Create necessary directories
        self.data_dir = Path("face_data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.known_faces_dir = self.data_dir / "known_faces"
        self.known_faces_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.encodings_file = self.data_dir / "face_encodings.pkl"
        
        # Face recognition settings
        self.tolerance = 0.6
        self.model = 'hog'
        
        # Cache for face encodings
        self.known_face_encodings = []
        self.known_face_names = []
        self.person_ids = []
        
        self.load_encodings()
    
    def generate_encoding_hash(self, encoding: np.ndarray) -> str:
        """Generate a hash for face encoding"""
        encoding_bytes = encoding.tobytes()
        return hashlib.sha256(encoding_bytes).hexdigest()
    
    def load_encodings(self):
        """Load face encodings from file and database"""
        try:
            if self.encodings_file.exists():
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                    self.person_ids = data.get('person_ids', [])
        except Exception as e:
            logger.error(f"Error loading encodings: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
            self.person_ids = []
    
    def save_encodings(self):
        """Save face encodings to file"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'person_ids': self.person_ids
            }
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving encodings: {e}")
    
    def add_person_from_upload(self, uploaded_file, person_name: str) -> Tuple[bool, str]:
        """Add a person from uploaded image"""
        try:
            # Read uploaded file
            image_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            # Convert to numpy array
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Process the image
            face_encodings = face_recognition.face_encodings(image_np, model=self.model)
            
            if not face_encodings:
                return False, "No face detected in the image"
            
            if len(face_encodings) > 1:
                return False, "Multiple faces detected. Please use an image with only one person"
            
            face_encoding = face_encodings[0]
            encoding_hash = self.generate_encoding_hash(face_encoding)
            
            # Save image to known_faces directory
            target_path = self.known_faces_dir / f"{person_name}.jpg"
            image.save(target_path)
            
            # Add to database
            if self.db_manager.add_person(person_name, encoding_hash):
                # Get the person ID
                people = self.db_manager.get_people()
                person_id = next(p['id'] for p in people if p['name'] == person_name)
                
                # Add to memory
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(person_name)
                self.person_ids.append(person_id)
                
                self.save_encodings()
                return True, f"Successfully added {person_name}"
            else:
                target_path.unlink()
                return False, "Person already exists in database"
                
        except Exception as e:
            logger.error(f"Error adding person: {e}")
            return False, f"Error processing image: {str(e)}"
    
    def remove_person(self, person_name: str) -> bool:
        """Remove a person"""
        try:
            if person_name in self.known_face_names:
                index = self.known_face_names.index(person_name)
                person_id = self.person_ids[index]
                
                # Remove from database
                if self.db_manager.remove_person(person_id):
                    # Remove from memory
                    self.known_face_encodings.pop(index)
                    self.known_face_names.pop(index)
                    self.person_ids.pop(index)
                    
                    # Remove image file
                    image_path = self.known_faces_dir / f"{person_name}.jpg"
                    if image_path.exists():
                        image_path.unlink()
                    
                    self.save_encodings()
                    return True
            return False
        except Exception as e:
            logger.error(f"Error removing person: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[str, Optional[int]]]]:
        """Process frame for face recognition"""
        try:
            # Resize for faster processing
            scale_factor = 0.25
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_small_frame, model=self.model)
            
            if not face_locations:
                return frame, []
            
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model=self.model)
            
            face_names = []
            detected_person_ids = []
            
            for face_encoding in face_encodings:
                name = "Unknown"
                person_id = None
                
                if self.known_face_encodings:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if face_distances[best_match_index] < self.tolerance:
                        name = self.known_face_names[best_match_index]
                        person_id = self.person_ids[best_match_index]
                
                face_names.append(name)
                detected_person_ids.append(person_id)
            
            # Draw rectangles and labels
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up
                top = int(top / scale_factor)
                right = int(right / scale_factor)
                bottom = int(bottom / scale_factor)
                left = int(left / scale_factor)
                
                # Draw rectangle
                # Use green for known, dark navy for unknown (matches dashboard palette)
                color = (0, 255, 0) if name != "Unknown" else (138, 58, 30)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
            return frame, list(zip(face_names, detected_person_ids))
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, []
    
    def mark_attendance_for_detected(self, detected_faces: List[Tuple[str, Optional[int]]]) -> List[str]:
        """Mark attendance for detected faces"""
        messages = []
        for name, person_id in detected_faces:
            if name != "Unknown" and person_id:
                if self.db_manager.mark_attendance(person_id, name):
                    messages.append(f"Attendance marked for {name}")
        return messages

# Initialize session state
def init_session_state():
    if 'attendance_system' not in st.session_state:
        st.session_state.attendance_system = FaceAttendanceSystem()
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'last_attendance_check' not in st.session_state:
        st.session_state.last_attendance_check = time.time()
    # Manage People form reset state
    if 'add_person_reset_pending' not in st.session_state:
        st.session_state.add_person_reset_pending = False
    if 'add_person_nonce' not in st.session_state:
        st.session_state.add_person_nonce = 0

def _safe_rerun():
    try:
        if hasattr(st, 'rerun'):
            st.rerun()
        else:
            st.experimental_rerun()
    except Exception:
        pass

def main():
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "",
            ["Dashboard", "Live Attendance", "Manage People", "Attendance Records", "Statistics", "Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # System status
        st.markdown("### System Status")
        people_count = len(st.session_state.attendance_system.known_face_names)
        today = datetime.now().strftime("%Y-%m-%d")
        today_records = st.session_state.attendance_system.db_manager.get_attendance_records(today, today)
        
        st.metric("Registered People", people_count)
        st.metric("Today's Attendance", len(today_records))
        
        st.markdown("---")
        st.markdown("**Made by Suren Saghatelyan**")
        st.markdown("© 2025 All rights reserved")
    
    # Main content
    if page == "Dashboard":
        show_dashboard()
    elif page == "Live Attendance":
        show_live_attendance()
    elif page == "Manage People":
        show_manage_people()
    elif page == "Attendance Records":
        show_attendance_records()
    elif page == "Statistics":
        show_statistics()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    st.title("Dashboard")
    
    attendance_system = st.session_state.attendance_system
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    people = attendance_system.db_manager.get_people()
    all_records = attendance_system.db_manager.get_attendance_records()
    today = datetime.now().strftime("%Y-%m-%d")
    today_records = [r for r in all_records if r['date'] == today]
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    week_records = [r for r in all_records if r['date'] >= week_ago]
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total People</div>
            <div class="metric-value">{len(people)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Today's Attendance</div>
            <div class="metric-value">{len(today_records)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">This Week</div>
            <div class="metric-value">{len(week_records)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{len(all_records)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Today's Attendance")
        if today_records:
            df = pd.DataFrame(today_records)
            st.dataframe(df[['name', 'time', 'status']], use_container_width=True, hide_index=True)
        else:
            st.info("No attendance records for today yet")
    
    with col2:
        st.markdown("### Recent Activity")
        if all_records:
            recent = all_records[:10]
            df = pd.DataFrame(recent)
            st.dataframe(df[['name', 'date', 'time']], use_container_width=True, hide_index=True)
        else:
            st.info("No attendance records found")
    
    # Quick actions
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start Live Attendance", use_container_width=True):
            st.switch_page("pages/live_attendance.py") if hasattr(st, 'switch_page') else st.session_state.update({'page': 'Live Attendance'})
    
    with col2:
        if st.button("Add New Person", use_container_width=True):
            st.session_state.page = 'Manage People'
            st.rerun()
    
    with col3:
        if st.button("View Records", use_container_width=True):
            st.session_state.page = 'Attendance Records'
            st.rerun()

def show_live_attendance():
    st.title("Live Attendance Tracking")
    
    attendance_system = st.session_state.attendance_system
    
    if len(attendance_system.known_face_names) == 0:
        st.warning("No known faces loaded. Please add people in the 'Manage People' section first.")
        return
    
    st.info(f"Registered people: {', '.join(attendance_system.known_face_names)}")
    
    # Camera controls
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("Start Camera" if not st.session_state.camera_active else "Stop Camera", 
                     type="primary", use_container_width=True):
            st.session_state.camera_active = not st.session_state.camera_active
    
    # Camera feed placeholder
    camera_placeholder = st.empty()
    status_placeholder = st.empty()
    
    if st.session_state.camera_active:
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Cannot access camera. Please check your camera connection.")
                st.session_state.camera_active = False
                return
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            frame_count = 0
            
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                frame_count += 1
                
                # Process every 10th frame
                if frame_count % 10 == 0:
                    processed_frame, detected_faces = attendance_system.process_frame(frame.copy())
                    
                    # Convert to RGB for display
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    camera_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                    
                    # Mark attendance
                    current_time = time.time()
                    if current_time - st.session_state.last_attendance_check > 3.0:
                        messages = attendance_system.mark_attendance_for_detected(detected_faces)
                        if messages:
                            status_placeholder.success("\n".join(messages))
                            st.session_state.last_attendance_check = current_time
                else:
                    # Just display the frame
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                
                time.sleep(0.033)
            
            cap.release()
            
        except Exception as e:
            st.error(f"Error accessing camera: {e}")
            st.session_state.camera_active = False
    
    # Today's attendance summary
    st.markdown("### Today's Attendance Summary")
    today = datetime.now().strftime("%Y-%m-%d")
    today_records = attendance_system.db_manager.get_attendance_records(today, today)
    
    if today_records:
        df = pd.DataFrame(today_records)
        st.dataframe(df[['name', 'time', 'status']], use_container_width=True, hide_index=True)
    else:
        st.info("No attendance records for today yet")

def show_manage_people():
    st.title("Manage People")
    
    attendance_system = st.session_state.attendance_system
    
    # Handle deferred form reset BEFORE rendering widgets
    if st.session_state.get("add_person_reset_pending"):
        # Bump nonce so keyed widgets remount
        st.session_state.add_person_nonce += 1
        st.session_state.add_person_reset_pending = False
    
    # Add new person section
    with st.expander("Add New Person", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            person_name = st.text_input("Person Name", placeholder="Enter full name", key=f"add_person_name_{st.session_state.add_person_nonce}")
        
        with col2:
            source = st.radio("Photo Source", ["Upload", "Camera"], horizontal=True, key=f"add_photo_source_{st.session_state.add_person_nonce}")
            uploaded_file = None
            captured_file = None
            if source == "Upload":
                uploaded_file = st.file_uploader("Upload Photo", type=['png', 'jpg', 'jpeg'], key=f"add_person_upload_{st.session_state.add_person_nonce}")
            else:
                captured_file = st.camera_input("Capture Photo", key=f"add_person_camera_{st.session_state.add_person_nonce}")
        
        # Preview
        preview_file = uploaded_file if uploaded_file else captured_file
        if preview_file:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(preview_file, caption="Preview", width=300)
        
        if st.button("Add Person", type="primary", use_container_width=True):
            input_file = uploaded_file if uploaded_file else captured_file
            if person_name and input_file:
                if person_name.strip():
                    input_file.seek(0)
                    success, message = attendance_system.add_person_from_upload(input_file, person_name.strip())
                    if success:
                        st.success(message)
                        # Defer clearing inputs by remounting keyed widgets via nonce
                        st.session_state.add_person_reset_pending = True
                        time.sleep(1)
                        _safe_rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please enter a valid name")
            else:
                st.error("Please provide both name and a photo (upload or camera)")
    
    # People list
    st.markdown("### Registered People")
    
    people = attendance_system.db_manager.get_people()
    
    if people:
        # Search
        search_term = st.text_input("Search people", placeholder="Type to search...")
        
        filtered_people = [p for p in people if search_term.lower() in p['name'].lower()] if search_term else people
        
        if filtered_people:
            for person in filtered_people:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{person['name']}**")
                
                with col2:
                    # View image
                    image_path = attendance_system.known_faces_dir / f"{person['name']}.jpg"
                    if image_path.exists():
                        if st.button("View", key=f"view_{person['id']}"):
                            st.image(str(image_path), width=200)
                
                with col3:
                    if st.button("Remove", key=f"remove_{person['id']}"):
                        if attendance_system.remove_person(person['name']):
                            st.success(f"Removed {person['name']}")
                            time.sleep(1)
                            _safe_rerun()
                        else:
                            st.error(f"Failed to remove {person['name']}")
                
                st.markdown("---")
        else:
            st.info("No people found matching your search")
    else:
        st.info("No people registered yet. Add some people to get started!")

def show_attendance_records():
    st.title("Attendance Records")
    
    attendance_system = st.session_state.attendance_system
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        from_date = st.date_input("From Date", value=datetime.now().date() - timedelta(days=30))
    
    with col2:
        to_date = st.date_input("To Date", value=datetime.now().date())
    
    with col3:
        people = attendance_system.known_face_names
        selected_people = st.multiselect("Filter by People", options=people, default=people if len(people) <= 5 else [])
    
    # Fetch records
    records = attendance_system.db_manager.get_attendance_records(
        from_date.strftime("%Y-%m-%d"),
        to_date.strftime("%Y-%m-%d")
    )
    
    if records and selected_people:
        records = [r for r in records if r['name'] in selected_people]
    
    # Display
    if records:
        st.markdown(f"### Records ({len(records)} entries)")
        
        df = pd.DataFrame(records)
        st.dataframe(df[['name', 'date', 'time', 'status']], use_container_width=True, hide_index=True)
        
        # Export
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"attendance_records_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No attendance records found for the selected criteria")

def show_statistics():
    st.title("Statistics & Analytics")
    
    attendance_system = st.session_state.attendance_system
    
    records = attendance_system.db_manager.get_attendance_records()
    people = attendance_system.db_manager.get_people()
    
    if not records:
        st.info("No attendance data available for statistics")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = len([r for r in records if r['date'] == today])
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    week_count = len([r for r in records if r['date'] >= week_ago])
    
    with col1:
        st.metric("Total Records", len(records))
    
    with col2:
        st.metric("Unique People", len(people))
    
    with col3:
        st.metric("Today's Attendance", today_count)
    
    with col4:
        st.metric("This Week", week_count)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Attendance by Person")
        df = pd.DataFrame(records)
        if not df.empty:
            person_counts = df['name'].value_counts().head(10)
            st.bar_chart(person_counts)
        else:
            st.info("No data to display")
    
    with col2:
        st.markdown("### Daily Attendance Trend")
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            daily_counts = df.groupby('date').size()
            st.line_chart(daily_counts)
        else:
            st.info("No data to display")
    
    # Top performers
    st.markdown("### Top Performers")
    if not df.empty:
        name_counts = df['name'].value_counts().head(5)
        
        for i, (name, count) in enumerate(name_counts.items(), 1):
            col1, col2, col3 = st.columns([1, 4, 1])
            with col1:
                st.write(f"#{i}")
            with col2:
                st.write(f"**{name}**")
            with col3:
                st.write(f"{count} days")
    
    # Recent activity
    st.markdown("### Recent Activity")
    recent = records[:10] if records else []
    if recent:
        recent_df = pd.DataFrame(recent)
        st.dataframe(recent_df[['name', 'date', 'time']], use_container_width=True, hide_index=True)
    else:
        st.info("No recent activity")

def show_settings():
    st.title("Settings")
    
    attendance_system = st.session_state.attendance_system
    
    # Face Recognition Settings
    with st.expander("Face Recognition Settings", expanded=True):
        st.markdown("### Detection Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_tolerance = st.slider(
                "Face Recognition Tolerance",
                min_value=0.1,
                max_value=1.0,
                value=attendance_system.tolerance,
                step=0.1,
                help="Lower values = more strict matching. Higher values = more lenient matching."
            )
            
            if new_tolerance != attendance_system.tolerance:
                attendance_system.tolerance = new_tolerance
                st.success(f"Tolerance updated to {new_tolerance}")
        
        with col2:
            current_model = attendance_system.model
            model_option = st.selectbox(
                "Detection Model",
                options=['hog', 'cnn'],
                index=0 if current_model == 'hog' else 1,
                help="HOG: Faster, CPU-friendly. CNN: More accurate, requires more processing power."
            )
            
            if model_option != current_model:
                attendance_system.model = model_option
                st.success(f"Model updated to {model_option}")
    
    # Database Management
    with st.expander("Database Management"):
        st.markdown("### Database Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Rebuild Encodings", use_container_width=True):
                try:
                    attendance_system.load_encodings()
                    st.success("Encodings rebuilt successfully")
                except Exception as e:
                    st.error(f"Error rebuilding encodings: {e}")
        
        with col2:
            records = attendance_system.db_manager.get_attendance_records()
            if records:
                df = pd.DataFrame(records)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "Backup Data",
                    data=csv_data,
                    file_name=f"attendance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            if st.button("Clear Old Records", use_container_width=True):
                st.warning("This feature is not yet implemented")
    
    # System Information
    with st.expander("System Information"):
        st.markdown("### Current Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Face Recognition:**")
            st.write(f"- Model: {attendance_system.model}")
            st.write(f"- Tolerance: {attendance_system.tolerance}")
            st.write(f"- Registered People: {len(attendance_system.known_face_names)}")
            st.write(f"- Database: SQLite")
        
        with col2:
            st.write("**Data Storage:**")
            
            # Calculate storage sizes
            data_dir_size = sum(f.stat().st_size for f in attendance_system.data_dir.rglob('*') if f.is_file())
            data_dir_size_mb = data_dir_size / (1024 * 1024)
            
            st.write(f"- Data Directory: {data_dir_size_mb:.2f} MB")
            st.write(f"- Images: {len(list(attendance_system.known_faces_dir.glob('*.jpg')))}")
            st.write(f"- Encodings File: {'Exists' if attendance_system.encodings_file.exists() else 'Missing'}")
    
    # Performance Tips
    with st.expander("Performance Tips"):
        st.markdown("""
        **Optimization Tips:**
        
        - Use HOG model for CPU-only systems (faster but less accurate)
        - Use CNN model if you have GPU support (slower but more accurate)
        - Lower the tolerance for more accurate but slower matching
        - Ensure good lighting when adding new faces
        - Use high-quality, front-facing photos for registration
        - Regularly clean up unused face encodings
        
        **Best Practices:**
        
        - Add clear, well-lit photos with single faces
        - Avoid photos with multiple people or faces at angles
        - Test recognition accuracy after adding new people
        - Back up your database regularly
        - Monitor system performance and adjust settings as needed
        """)

if __name__ == "__main__":
    main()