import streamlit as st
import base64
import os
import json
import hashlib
from google import genai
from google.genai import types
from dotenv import load_dotenv

# -------------------------------
# Setup & Config
# -------------------------------
load_dotenv()

APP_PASSWORD = os.getenv("APP_PASSWORD")
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Brand Standards Analyzer", page_icon="🏨", layout="centered")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "pdf_extracted" not in st.session_state:
    st.session_state["pdf_extracted"] = None
if "pdf_extract_raw" not in st.session_state:
    st.session_state["pdf_extract_raw"] = None
if "pdf_hash" not in st.session_state:
    st.session_state["pdf_hash"] = None
if "pdf_system_prompt" not in st.session_state:
    st.session_state["pdf_system_prompt"] = (
        "You are a PDF extraction assistant. Extract the following structured information "
        "from the provided brand standards PDF as JSON. Return JSON only.\n\n"
        "Required keys: BrandName, RequiredColors (list), RequiredFonts (list), RoomRequirements (list), Notes (string)\n\n"
        "If a field is not found, return null or an empty list."
    )
if "show_pdf_extracted" not in st.session_state:
    st.session_state["show_pdf_extracted"] = False

# -------------------------------
# Authentication (no experimental_rerun usage)
# -------------------------------
if not st.session_state["authenticated"]:
    st.markdown(
        """
        <div style='text-align:center;'>
            <h2>🏨 Brand Standards Analyzer Access</h2>
            <p>Please enter your access token to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    token_input = st.text_input("🔑 Enter Access Token:", type="password")
    if st.button("Login"):
        if token_input == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.success("Access Granted ✅")
            st.rerun()
        else:
            st.error("Invalid token. Please try again.")
    # Stop execution until authenticated; after successful login the next rerun continues
    # if not st.session_state["authenticated"]:
    st.stop()

# -------------------------------
# Styling / Header
# -------------------------------
st.markdown(
    """
    <style>
        .stApp { font-family: 'Segoe UI', Roboto, sans-serif; background:#f8fafc; }
        .card { background: #fff; padding: 16px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
        .muted { color: #6b7280; font-size: 14px; }
        .result-card { background: #fff; padding: 14px; border-radius: 10px; margin-top: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        .success-card { border-left: 6px solid #16a34a; }
        .error-card { border-left: 6px solid #dc2626; }
        .warning-card { border-left: 6px solid #f59e0b; }
        pre { white-space: pre-wrap; word-break: break-word; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏨 Brand Standards Analyzer")
st.write("Extract requirements from a brand standards PDF and run image QA using an LLM. Single Submit runs extraction (if needed) and the image check.")
st.markdown("---")

# -------------------------------
# File upload area (side-by-side)
# -------------------------------
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    pdf_file = st.file_uploader("📄 Upload Brand Standards (PDF)", type=["pdf"], key="pdf_upload")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    image_file = st.file_uploader("🖼️ Upload Image for QA Check", type=["jpg", "jpeg", "png"], key="img_upload")
    if image_file:
        st.image(image_file, use_container_width=True, caption="Image preview")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# PDF extraction function
# -------------------------------
def extract_from_pdf(pdf_bytes: bytes, system_prompt: str):
    """Call Gemini to extract required info from a PDF using provided system prompt."""
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")
    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                types.Part.from_text(text="Please extract requested information from the attached PDF and return ONLY JSON or concise text."),
            ],
        ),
    ]

    generate_config = types.GenerateContentConfig(
        system_instruction=[types.Part.from_text(text=system_prompt)],
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_config,
    )
    return response.text.strip()

# -------------------------------
# Gemini QA function (uses extracted PDF context)
# -------------------------------
def call_gemini_api(image_bytes: bytes, image_mime: str, pdf_extracted):
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")
    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"

    base_system = (
         "You are a Hotel QA Specialist. Evaluate the provided image for compliance with cleanliness, "
        "consistency, and maintenance standards. Use the following key categories when classifying findings:\n\n"
        "1. Condition – Issues related to wear, damage, or deterioration of materials or fixtures (e.g., peeling drawers, faded paint, broken parts).\n"
        "2. Cleanliness – Issues involving dirt, stains, or maintenance of hygiene and visual appearance (e.g., dirty surfaces, stains, debris).\n"
        "3. Compliance – Issues involving deviation from brand standards or design specifications (e.g., incorrect fixtures, missing required items).\n\n"
        "Respond STRICTLY in JSON with the following top-level fields only:\n"
        "{\n"
        '  "Issue_Present": true|false,                  // overall boolean\n'
        '  "Category": "Condition"|"Cleanliness"|"Compliance",  // primary category for the issue\n'
        '  "Description": "Concise explanation of the finding",\n'
        '  "Resolution": "Specific words to correct the issue", // like replace, paint, remove, wash, clean etc.\n'
        '  ]\n'
        "}\n\n"
        "Do not include any extra text, commentary, or code blocks. If multiple issues exist, choose the primary Category that best represents the most important issue and list others in Findings.")
    if pdf_extracted:
        pdf_context = json.dumps(pdf_extracted) if isinstance(pdf_extracted, (dict, list)) else str(pdf_extracted)
        system_instruction_text = base_system + "\n\nContext from brand standards (extracted):\n" + pdf_context
    else:
        system_instruction_text = base_system

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
                types.Part.from_text(text="Analyze this single image and return the JSON result described in system instruction."),
            ],
        ),
    ]

    generate_config = types.GenerateContentConfig(
        system_instruction=[types.Part.from_text(text=system_instruction_text)],
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_config,
    )
    return response.text.strip()

# -------------------------------
# PDF extraction settings (editable, hidden by default)
# -------------------------------
# with st.expander("PDF extraction system prompt (used automatically on Submit)", expanded=False):
st.session_state["pdf_system_prompt"] = """
You are an expert **Brand Compliance Auditor** for a hospitality corporation. Your mission is to analyze a comprehensive "Brand Standards Manual" PDF and extract every single quantifiable, measurable, or visually verifiable technical specification required for an image-based compliance check.

**Your output MUST adhere strictly to the format below.** Do not include any introductory text, conversation, or filler outside of the structured headings and bullet points.

**Role Constraints:**
1.  **Extract ONLY visual, structural, or quantifiable data.** This includes dimensions (inches, sq ft), material types (HEPA-filtered, slip-resistant), brand/model numbers, color specifications (Hex, Pantone), minimum/maximum size requirements, and physical accessibility features (ADA, grab bars, etc.).
2.  **Ignore (Negative Constraints):** Do not extract abstract concepts, training protocols, financial requirements, legal dispute processes, internal philosophies, penalties/fines, general prose, or standards that require real-time staff action (e.g., "Must greet guests with a smile").
3.  **Maintain Structure:** If a section of the provided output format is empty because the PDF lacks those specifications, include the heading but leave the bullet points empty.

**Required Output Structure (Strict Adherence):**

### 🎨 Visual Identity & Aesthetics

* **Logo Specifications:** [List specific design, color, and size/spacing requirements.]
* **Color Palette (Mandatory Adherence):** [List colors, including names and quantifiable codes (Hex, Pantone, etc.).]
* **Typography:** [List required font face, weights, and usage context.]

### 🛌 Guestroom Design & FF&E (Furniture, Fixtures, & Equipment)

* **Dimensions & Layout:** [List measurable dimensional requirements (e.g., pathway width, minimum room size).]
* **Materials & Finishes:** [List verifiable material requirements (e.g., flooring type, window treatments).]
* **Lighting:** [List specific lighting types, fixtures, and locations.]
* **Required FF&E (Specific Models/Features):** [List specific product models, features, or required components.]
* **Bedding and Linens:** [List color schemes, material requirements, and mandatory components (e.g., protectors).]
* **Technology:** [List minimum/maximum screen sizes, port types, and required features.]

### 🏟️ Public Areas, Exterior & Safety

* **Public Area Safety & Accessibility (ADA):** [List structural and fixture requirements (e.g., ramps, grab bars, path clearance).]
* **Décor Restriction:** [List any explicitly prohibited visual content.]
* **Exterior Maintenance:** [List visible maintenance requirements (e.g., striping, lighting, repair status).]
* **Family Amenities (Physical):** [List dimensions and prohibited items for children's equipment (e.g., cribs).]
* **Fire Safety (Passive/Active):** [List requirements for detectors, extinguishers, storage, and evacuation maps.]
* **Other Structural/Safety:** [List any other measurable construction or material requirements.]
"""

# Show extracted PDF data (hidden by default)
if st.session_state.get("pdf_extracted") is not None:
    show = st.checkbox(
        "Show extracted PDF information",
        value=st.session_state.get("show_pdf_extracted", False),
        key="show_pdf_extracted_checkbox",
    )
    st.session_state["show_pdf_extracted"] = show
    if show:
        st.markdown("### 📄 Extracted PDF Data")
        extracted = st.session_state["pdf_extracted"]
        if isinstance(extracted, (dict, list)):
            st.json(extracted)
        else:
            st.markdown(st.session_state.get("pdf_extract_raw", extracted))

# -------------------------------
# Utility to clean code block wrappers and parse JSON
# -------------------------------
def _clean_model_json(text: str) -> str:
    # remove triple-backtick fenced blocks if present
    if "```json" in text:
        try:
            return text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        except Exception:
            pass
    if "```" in text:
        try:
            return text.split("```", 1)[1].rsplit("```", 1)[0].strip()
        except Exception:
            pass
    return text.strip()

# -------------------------------
# Single Submit: extract (if needed) + analyze image
# -------------------------------
st.markdown("### 🧪 Run QA (single Submit)")
if st.button("🔍 Submit"):
    # Validate image
    if not image_file:
        st.error("Please upload an image for QA check before submitting.")
    else:
        # Determine image bytes & mime type
        try:
            image_bytes = image_file.read()
            image_mime = getattr(image_file, "type", None) or "image/jpeg"
        except Exception as e:
            st.error(f"Unable to read image: {e}")
            image_bytes = None
            image_mime = "image/jpeg"

        if image_bytes is None:
            st.stop()

        # If PDF present, check hash and extract if needed
        pdf_ctx = st.session_state.get("pdf_extracted")
        if pdf_file:
            try:
                pdf_bytes = pdf_file.getbuffer().tobytes()
                new_hash = hashlib.sha256(pdf_bytes).hexdigest()
            except Exception as e:
                st.warning(f"Unable to read PDF for hashing: {e}")
                pdf_bytes = None
                new_hash = None

            if (not st.session_state.get("pdf_extracted")) or (new_hash and st.session_state.get("pdf_hash") != new_hash):
                # run extraction automatically
                with st.spinner("Extracting information from PDF..."):
                    try:
                        system_prompt = st.session_state.get("pdf_system_prompt")
                        extracted_text = extract_from_pdf(pdf_bytes, system_prompt)
                        st.session_state["pdf_extract_raw"] = extracted_text
                        try:
                            cleaned = _clean_model_json(extracted_text)
                            st.session_state["pdf_extracted"] = json.loads(cleaned)
                        except Exception:
                            # store raw if JSON parse fails
                            st.session_state["pdf_extracted"] = extracted_text
                        st.session_state["pdf_hash"] = new_hash
                        pdf_ctx = st.session_state["pdf_extracted"]
                        # ensure extracted data is shown immediately after first extraction
                        st.session_state["show_pdf_extracted"] = True
                        st.success("PDF extraction complete.")
                        st.markdown("### 📄 Extracted PDF Data (just extracted)")
                        extracted_now = st.session_state["pdf_extracted"]
                        if isinstance(extracted_now, (dict, list)):
                            st.json(extracted_now)
                        else:
                            st.markdown(st.session_state.get("pdf_extract_raw", extracted_now))

                    except Exception as e:
                        st.session_state["pdf_extract_raw"] = None
                        st.session_state["pdf_extracted"] = None
                        st.error(f"PDF extraction failed: {e}")
                        pdf_ctx = None
            else:
                st.info("Using previously extracted PDF data (no change detected).")
                pdf_ctx = st.session_state.get("pdf_extracted")

        # Run image analysis
        with st.spinner("Analyzing image with Gemini... ⏳"):
            try:
                result_text = call_gemini_api(image_bytes, image_mime, pdf_ctx)
            except Exception as e:
                st.error(f"Image analysis failed: {e}")
                result_text = None

        if not result_text:
            st.stop()

        # Display QA result
        st.markdown("### 🧾 QA Analysis Result")
        cleaned = _clean_model_json(result_text)
        try:
            result_json = json.loads(cleaned)
            compliant = result_json.get("Issue_Present", False)
            category = result_json.get("Category", "Unknown")
            description = result_json.get("Description", "No description provided.")
            resolution = result_json.get("Resolution", "No Soultion provided.")
            if compliant == False:
                st.markdown(
                    f"""
                    <div class="result-card success-card">
                        <h4>{category}</h4>
                        <p>{description}</p>
                        <p><strong>{resolution}</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card error-card">
                        <h4>{category}</h4>
                        <p>{description}</p>
                        <p><strong>{resolution}</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except json.JSONDecodeError:
            st.markdown(
                f"""
                <div class="result-card warning-card">
                    <h4>⚠️ Unable to Parse JSON</h4>
                    <p>Raw model output:</p>
                    <pre>{result_text}</pre>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")
st.caption("🌐 Powered by AI • Mock Hotels QA Team • Built with Streamlit")