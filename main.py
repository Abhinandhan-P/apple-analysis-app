import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import plotly.express as px
from fpdf import FPDF
import tempfile
import os

# --- Page Config ---
st.set_page_config(
    page_title="Apple Ripeness by Abhinandhan P",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global Theme */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* Custom Header */
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        background: linear-gradient(90deg, #ff6b6b, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        font-size: 1.1rem;
        color: #dfe6e9;
        margin-top: 10px;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #b2bec3;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #ff7675, #d63031);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(214, 48, 49, 0.4);
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.15);
        color: #ff7675;
    }
    /* Animations */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .metric-card {
        animation: fadeIn 0.5s ease-out forwards;
    }
    .main-header {
        animation: fadeIn 0.8s ease-out forwards;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
        backdrop-filter: blur(5px);
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.info(
    """
    **Apple Ripeness AI** v2.0
    
    This app uses advanced Computer Vision (YOLOv8) to:
    1. Detect Apples 🍎
    2. Analyze Color Ratios 🎨
    3. Classify Ripeness (Ripe, Overripe, Rotten) ✅
    
    Built with Streamlit & Ultralytics.
    """
)

# --- Model Loading ---
@st.cache_resource
def load_models():
    try:
        c_model = YOLO('runs/detect/train/weights/best.pt') # Color Model
        r_model = YOLO('best.pt') # Ripeness Model
        return c_model, r_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

color_model, ripeness_model = load_models()

# --- Helper Functions ---
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def get_color_ratios(image_crop):
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV)
    total_pixels = image_crop.shape[0] * image_crop.shape[1]
    
    if total_pixels == 0:
        return 0.0, 0.0, 0.0

    # 1. Green Range
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = cv2.countNonZero(green_mask) / total_pixels

    # 2. Red Range (wraps around 0/180)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    red_ratio = cv2.countNonZero(red_mask) / total_pixels

    # 3. Dark Red / Overripe Range
    lower_maroon1 = np.array([0, 40, 20])
    upper_maroon1 = np.array([10, 255, 80])
    lower_maroon2 = np.array([170, 40, 20])
    upper_maroon2 = np.array([180, 255, 80])
    maroon_mask = cv2.inRange(hsv, lower_maroon1, upper_maroon1) + cv2.inRange(hsv, lower_maroon2, upper_maroon2)
    maroon_ratio = cv2.countNonZero(maroon_mask) / total_pixels

    return green_ratio, red_ratio, maroon_ratio

def create_pdf(total_stats, detailed_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Apple Ripeness Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', size=10)
    pdf.cell(200, 10, txt="Summary Metrics", ln=True)
    pdf.set_font("Arial", size=10)
    for key, value in total_stats.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', size=10)
    pdf.cell(200, 10, txt="Detailed Breakdown (First 50 Apples)", ln=True)
    pdf.set_font("Arial", size=8)
    
    # Table Header
    pdf.cell(20, 10, "Image", 1)
    pdf.cell(15, 10, "ID", 1)
    pdf.cell(30, 10, "Status", 1)
    pdf.cell(30, 10, "Type", 1)
    pdf.cell(20, 10, "Green %", 1)
    pdf.cell(20, 10, "Red %", 1)
    pdf.cell(20, 10, "Dark %", 1)
    pdf.ln()
    
    for row in detailed_data[:50]: # Limit to 50 rows for PDF simplicity
        pdf.cell(20, 10, str(row.get('Image ID', 'N/A')), 1)
        pdf.cell(15, 10, str(row['ID']), 1)
        pdf.cell(30, 10, str(row['Status']), 1)
        pdf.cell(30, 10, str(row['Type']), 1)
        pdf.cell(20, 10, str(row['Green %']), 1)
        pdf.cell(20, 10, str(row['Red %']), 1)
        pdf.cell(20, 10, str(row['Dark %']), 1)
        pdf.ln()
        
    return pdf.output(dest='S').encode('latin-1')

# Helper function to draw custom bounding boxes
def plot_custom_boxes(image_array, boxes, statuses):
    img = image_array.copy()
    for box, status in zip(boxes, statuses):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
        # Color based on status
        color = (0, 255, 0) # Green for Ripe (Green)
        if "Red" in status or "Overripe" in status: color = (255, 0, 0) # Red
        elif "Rotten" in status: color = (139, 69, 19) # Brown
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"{status}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

# Helper for Metric Cards
def metric_card(label, value, color="#ffffff"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Main App ---
st.markdown("""
<div class="main-header">
    <h1>🍎 Apple Classification, Counting, and Ripeness Analysis</h1>
    <p>Advanced Computer Vision for Precision Agriculture</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🖼️ Single Image Analysis", "📚 Batch Processing"])

# --- TAB 1: Single Image ---
with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="single")

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button("Analyze Image", key="btn_single"):
            if color_model and ripeness_model:
                with st.spinner('Analyzing...'):
                    img_array = np.array(image)
                    
                    # 1. Detection
                    color_results = color_model(img_array, conf=confidence_threshold)
                    color_result = color_results[0]
                    
                    # 2. Hybrid Check
                    ripeness_results = ripeness_model(img_array, conf=0.25)
                    ripeness_result = ripeness_results[0]
                    
                    # 3. Processing
                    combined_data = []
                    counts = {'Ripe (Green)': 0, 'Ripe (Red)': 0, 'Overripe': 0, 'Rotten': 0}
                    statuses = [] # Store statuses for plotting
                    
                    for i, box in enumerate(color_result.boxes):
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        apple_crop = img_array[y1:y2, x1:x2]
                        
                        g_ratio, r_ratio, m_ratio = get_color_ratios(apple_crop)
                        
                        is_rotten = False
                        for r_box in ripeness_result.boxes:
                            r_xyxy = r_box.xyxy[0].cpu().numpy()
                            iou = calculate_iou(xyxy, r_xyxy)
                            if iou > 0.5 and ripeness_result.names[int(r_box.cls[0])] == 'rotten_apple':
                                is_rotten = True; break
                        
                        status = "Unknown"
                        disp_color = "Unknown"
                        if is_rotten: status="Rotten"; disp_color="Rotten Apple"; counts['Rotten']+=1
                        elif m_ratio > 0.15: status="Overripe"; disp_color="Red Apple"; counts['Overripe']+=1
                        elif r_ratio > 0.02: status="Ripe"; disp_color="Red Apple"; counts['Ripe (Red)']+=1 # Any significant red -> Red Apple
                        else: status="Ripe"; disp_color="Green Apple"; counts['Ripe (Green)']+=1
                        
                        statuses.append(status)
                        combined_data.append({
                            "ID": i+1, "Status": status, "Type": disp_color,
                            "Green %": f"{g_ratio*100:.1f}%", "Red %": f"{r_ratio*100:.1f}%", "Dark %": f"{m_ratio*100:.1f}%"
                        })
                    
                    # Display Custom Plot
                    annotated_img = plot_custom_boxes(img_array, color_result.boxes, statuses)
                    st.image(annotated_img, caption='Analyzed Image', use_container_width=True)
                    st.dataframe(combined_data, use_container_width=True)
                    
                    # Custom Metric Cards
                    c1, c2, c3 = st.columns(3)
                    with c1: metric_card("Total Apples", len(combined_data), "#ffffff")
                    with c2: metric_card("Ripe (Good)", counts['Ripe (Green)'] + counts['Ripe (Red)'], "#2ecc71")
                    with c3: metric_card("Overripe/Rotten", counts['Overripe'] + counts['Rotten'], "#e74c3c")

# --- TAB 2: Batch Processing ---
with tab2:
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch")

    if uploaded_files:
        # Initialize session state for batch results if not present
        if 'batch_data' not in st.session_state:
            st.session_state.batch_data = None
            
        if st.button("Analyze Batch", key="btn_batch"):
            if color_model and ripeness_model:
                
                all_combined_data = []
                total_counts = {'Ripe (Green)': 0, 'Ripe (Red)': 0, 'Overripe': 0, 'Rotten': 0}
                processed_images = []
                
                progress_bar = st.progress(0)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    image = Image.open(uploaded_file).convert('RGB')
                    img_array = np.array(image)
                    
                    # Detection & Hybrid Check
                    color_result = color_model(img_array, conf=confidence_threshold)[0]
                    ripeness_result = ripeness_model(img_array, conf=0.25)[0]
                    
                    statuses = []
                    
                    for i, box in enumerate(color_result.boxes):
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        apple_crop = img_array[y1:y2, x1:x2]
                        g_ratio, r_ratio, m_ratio = get_color_ratios(apple_crop)
                        
                        is_rotten = False
                        for r_box in ripeness_result.boxes:
                            iou = calculate_iou(xyxy, r_box.xyxy[0].cpu().numpy())
                            if iou > 0.5 and ripeness_result.names[int(r_box.cls[0])] == 'rotten_apple':
                                is_rotten = True; break
                        
                        status="Unknown"; disp_color="Unknown"
                        if is_rotten: status="Rotten"; disp_color="Rotten Apple"; total_counts['Rotten']+=1
                        elif m_ratio > 0.15: status="Overripe"; disp_color="Red Apple"; total_counts['Overripe']+=1
                        elif r_ratio > 0.02: status="Ripe"; disp_color="Red Apple"; total_counts['Ripe (Red)']+=1 # Any significant red -> Red Apple
                        else: status="Ripe"; disp_color="Green Apple"; total_counts['Ripe (Green)']+=1
                        
                        statuses.append(status)
                        all_combined_data.append({
                            "Image ID": idx+1, "ID": i+1, "Status": status, "Type": disp_color,
                            "Green %": f"{g_ratio*100:.1f}%", "Red %": f"{r_ratio*100:.1f}%", "Dark %": f"{m_ratio*100:.1f}%"
                        })
                    
                    # Store processed image for gallery
                    processed_images.append(plot_custom_boxes(img_array, color_result.boxes, statuses))
                    progress_bar.progress((idx + 1) / len(uploaded_files))

                st.success("Batch Analysis Complete!")
                
                # Store results in session state
                st.session_state.batch_data = {
                    'total_counts': total_counts,
                    'all_combined_data': all_combined_data,
                    'processed_images': processed_images
                }

        # Display Results if available in session state
        if st.session_state.batch_data:
            data = st.session_state.batch_data
            total_counts = data['total_counts']
            all_combined_data = data['all_combined_data']
            processed_images = data['processed_images']
            
            # 1. Metrics (Simplified)
            st.subheader("📊 Aggregate Metrics")
            total_apples = sum(total_counts.values())
            
            c1, c2, c3 = st.columns(3)
            with c1: metric_card("Total Apples", total_apples, "#ffffff")
            with c2: metric_card("Ripe (Good)", total_counts['Ripe (Green)'] + total_counts['Ripe (Red)'], "#2ecc71")
            with c3: metric_card("Overripe/Rotten", total_counts['Overripe'] + total_counts['Rotten'], "#e74c3c")
            
            # 2. Charts (Detailed - Showing Green/Red split)
            st.subheader("📈 Visual Analytics")
            # Prepare data for detailed pie chart
            chart_counts = {
                'Ripe (Green)': total_counts['Ripe (Green)'],
                'Ripe (Red)': total_counts['Ripe (Red)'],
                'Overripe': total_counts['Overripe'] + total_counts['Rotten']
            }
            chart_data = {"Status": list(chart_counts.keys()), "Count": list(chart_counts.values())}
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig_pie = px.pie(chart_data, values='Count', names='Status', title='Distribution (Pie)', hole=0.3,
                                color='Status',
                                color_discrete_map={'Ripe (Green)':'green', 'Ripe (Red)':'red', 'Overripe':'darkred'})
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                fig_bar = px.bar(chart_data, x='Status', y='Count', title='Distribution (Bar)', color='Status',
                                 color_discrete_map={'Ripe (Green)':'green', 'Ripe (Red)':'red', 'Overripe':'darkred'})
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # 3. Image Gallery
            st.subheader("🖼️ Processed Images")
            with st.expander("View All Processed Images"):
                cols = st.columns(3)
                for i, img in enumerate(processed_images):
                    cols[i % 3].image(img, caption=f"Image {i+1}", use_container_width=True)
            
            # 4. Detailed Data & PDF
            st.subheader("📋 Detailed Data")
            st.dataframe(all_combined_data, use_container_width=True)
            
            # Generate PDF Bytes
            pdf_bytes = create_pdf(total_counts, all_combined_data)
            
            c_d1, c_d2 = st.columns([1, 4])
            with c_d1:
                st.download_button(
                    label="📄 Download Report",
                    data=pdf_bytes,
                    file_name="apple_analysis_report.pdf",
                    mime="application/pdf"
                )
            with c_d2:
                if st.button("🔄 Reset Analysis"):
                    st.session_state.batch_data = None
                    st.rerun()

# --- Footer ---
st.markdown("""
<div class="footer">
    <p>Developed by Abhinandhan P | © 2025 Apple Ripeness AI</p>
</div>
""", unsafe_allow_html=True)
