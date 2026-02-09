"""
AI Resume Analyzer Backend - FastAPI
Single file backend to analyze resumes using Google Gemini AI
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
from PyPDF2 import PdfReader
from docx import Document
import os
import json
import io
from dotenv import load_dotenv

# ============================================================
# CONFIGURATION
# ============================================================

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Resume Analyzer API",
    description="Analyze resumes against job descriptions using AI",
    version="1.0.0"
)

# Configure CORS (allow Streamlit to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY not found in .env file!")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini AI configured successfully")


# ============================================================
# HELPER FUNCTIONS - FILE PROCESSING
# ============================================================

def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text from PDF file
    
    Args:
        file_content: PDF file as bytes
        
    Returns:
        Extracted text as string
    """
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error extracting PDF: {str(e)}"
        )


def extract_text_from_docx(file_content: bytes) -> str:
    """
    Extract text from DOCX file
    
    Args:
        file_content: DOCX file as bytes
        
    Returns:
        Extracted text as string
    """
    try:
        docx_file = io.BytesIO(file_content)
        doc = Document(docx_file)
        
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return text.strip()
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error extracting DOCX: {str(e)}"
        )


def extract_text_from_file(filename: str, file_content: bytes) -> str:
    """
    Extract text from uploaded file (PDF or DOCX)
    
    Args:
        filename: Name of the uploaded file
        file_content: File content as bytes
        
    Returns:
        Extracted text as string
    """
    # Get file extension
    file_extension = filename.lower().split('.')[-1]
    
    # Extract based on file type
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_content)
    
    elif file_extension == 'docx':
        return extract_text_from_docx(file_content)
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_extension}. Only PDF and DOCX are supported."
        )


# ============================================================
# HELPER FUNCTIONS - AI PROCESSING
# ============================================================

def build_analysis_prompt(resume_text: str, job_description: str) -> str:
    """
    Build the prompt for Gemini AI to analyze the resume
    
    Args:
        resume_text: Extracted resume text
        job_description: Job description provided by user
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an expert ATS (Applicant Tracking System) analyzer and career consultant. 
Analyze the following resume against the job description and provide a comprehensive analysis.

RESUME CONTENT:
{resume_text}

JOB DESCRIPTION:
{job_description}

Provide your analysis in the following JSON format (respond ONLY with valid JSON, no markdown, no extra text):

{{
    "atsScore": <number between 0-100>,
    "summary": "<2-3 sentence summary of candidate's fit for the role>",
    "categoryScores": {{
        "hardSkills": <score 0-5>,
        "softSkills": <score 0-5>,
        "experience": <score 0-5>,
        "qualifications": <score 0-5>
    }},
    "strengths": [
        "<strength 1>",
        "<strength 2>",
        "<strength 3>"
    ],
    "weaknesses": [
        "<weakness 1>",
        "<weakness 2>",
        "<weakness 3>"
    ],
    "recommendations": [
        "<actionable recommendation 1>",
        "<actionable recommendation 2>",
        "<actionable recommendation 3>"
    ]
}}

ANALYSIS CRITERIA:
- ATS Score: Overall match percentage (0-100)
- Hard Skills: Technical skills match (0-5)
- Soft Skills: Communication, leadership, teamwork (0-5)
- Experience: Relevant work experience (0-5)
- Qualifications: Education and certifications (0-5)
- Strengths: Top 3 positive highlights
- Weaknesses: Top 3 areas for improvement
- Recommendations: Top 3 actionable suggestions to improve the resume

Be honest, constructive, and specific in your analysis.
"""
    
    return prompt


def analyze_with_gemini(resume_text: str, job_description: str) -> dict:
    """
    Call Gemini AI to analyze the resume
    
    Args:
        resume_text: Extracted resume text
        job_description: Job description
        
    Returns:
        Analysis results as dictionary
    """
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-flash-latest')
        
        # Build prompt
        prompt = build_analysis_prompt(resume_text, job_description)
        
        # Generate response
        print("🤖 Calling Gemini AI...")
        response = model.generate_content(prompt)
        
        # Extract text from response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            # Remove first line (```json)
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1])
        
        # Parse JSON
        analysis_result = json.loads(response_text)
        
        print("✅ Analysis completed successfully")
        return analysis_result
    
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Response text: {response_text}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse AI response as JSON: {str(e)}"
        )
    
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calling Gemini AI: {str(e)}"
        )


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """
    Root endpoint - API health check
    """
    return {
        "message": "AI Resume Analyzer API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/resume/health",
            "analyze": "/api/resume/analyze (POST)"
        }
    }


@app.get("/api/resume/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "message": "Resume Analyzer API is running!",
        "gemini_configured": bool(GEMINI_API_KEY)
    }


@app.post("/api/resume/analyze")
async def analyze_resume(
    resumeFile: UploadFile = File(..., description="Resume file (PDF or DOCX)"),
    jobDescription: str = Form(..., description="Job description text")
):
    """
    Main endpoint to analyze resume against job description
    
    Args:
        resumeFile: Uploaded resume file
        jobDescription: Job description text
        
    Returns:
        Analysis results with ATS score, category scores, strengths, weaknesses, recommendations
    """
    
    # ===== VALIDATION =====
    
    # Check if API key is configured
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key not configured. Please add GEMINI_API_KEY to .env file"
        )
    
    # Validate file upload
    if not resumeFile:
        raise HTTPException(
            status_code=400,
            detail="Resume file is required"
        )
    
    # Validate job description
    if not jobDescription or not jobDescription.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description is required"
        )
    
    # Validate file type
    allowed_extensions = ['pdf', 'docx']
    file_extension = resumeFile.filename.lower().split('.')[-1]
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # ===== STEP 1: READ FILE =====
        print(f"📄 Reading file: {resumeFile.filename}")
        file_content = await resumeFile.read()
        
        # ===== STEP 2: EXTRACT TEXT =====
        print("📝 Extracting text from file...")
        resume_text = extract_text_from_file(resumeFile.filename, file_content)
        
        if not resume_text or len(resume_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Could not extract enough text from resume. Please ensure the file is not corrupted or password-protected."
            )
        
        print(f"✅ Extracted {len(resume_text)} characters")
        
        # ===== STEP 3: ANALYZE WITH AI =====
        print("🤖 Analyzing with Gemini AI...")
        analysis_result = analyze_with_gemini(resume_text, jobDescription)
        
        # ===== STEP 4: RETURN RESULTS =====
        print("✅ Returning analysis results")
        return JSONResponse(
            status_code=200,
            content=analysis_result
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Handle unexpected errors
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    PORT = int(os.getenv("PORT", 8081))
    
    print("\n" + "="*60)
    print("🚀 Starting AI Resume Analyzer Backend")
    print("="*60)
    print(f"📍 Server: http://localhost:{PORT}")
    print(f"📚 API Docs: http://localhost:{PORT}/docs")
    print(f"🔍 Health Check: http://localhost:{PORT}/api/resume/health")
    print("="*60 + "\n")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True  # Auto-reload on code changes
    )
