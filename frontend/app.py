import streamlit as st
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.interpolate import make_interp_spline
import requests

# ---------------- Configuration ----------------
BACKEND_URL = "http://localhost:8081/api/resume"

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# ---------------- Layout ----------------
left, right = st.columns([1, 1.2])

# ---------------- LEFT PANEL ----------------
with left:
    st.title("📄 AI Resume Analyzer")
    st.caption("Analyze your resume against a job description using AI")

    st.subheader("Upload Resume")
    resume_file = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"]
    )

    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the job description here",
        height=220
    )

    analyze_btn = st.button("🔍 Analyze Resume", use_container_width=True)
    
    # ============== ANALYSIS RESULTS ON LEFT ==============
    if analyze_btn:
        if not resume_file:
            st.error("Please upload a resume file")
        elif not job_description.strip():
            st.error("Please provide a job description")
        else:
            with st.spinner("Analyzing resume with AI..."):
                try:
                    # Prepare the request
                    files = {
                        'resumeFile': (resume_file.name, resume_file.getvalue(), resume_file.type)
                    }
                    data = {
                        'jobDescription': job_description
                    }

                    # Call backend API
                    response = requests.post(
                        f"{BACKEND_URL}/analyze",
                        files=files,
                        data=data,
                        timeout=60
                    )

                    if response.status_code == 200:
                        result = response.json()
                        
                        # Extract data
                        ats_score = result['atsScore']
                        summary = result['summary']
                        category_scores = result['categoryScores']
                        strengths = result.get('strengths', [])
                        weaknesses = result.get('weaknesses', [])
                        recommendations = result.get('recommendations', [])

                        # Store in session state to display in right panel
                        st.session_state['analysis_done'] = True
                        st.session_state['ats_score'] = ats_score
                        st.session_state['summary'] = summary
                        st.session_state['category_scores'] = category_scores
                        st.session_state['strengths'] = strengths
                        st.session_state['weaknesses'] = weaknesses
                        st.session_state['recommendations'] = recommendations

                        # ========== DISPLAY RESULTS ON LEFT ==========
                        st.divider()
                        st.subheader("Analysis Results")
                        
                        # Display ATS Score
                        st.metric("ATS Score", f"{ats_score}/100")
                        st.progress(ats_score / 100)

                        # Display Summary
                        if ats_score >= 80:
                            st.success(summary)
                        elif ats_score >= 60:
                            st.warning(summary)
                        else:
                            st.error(summary)

                    else:
                        st.error(f"Error: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend server. Make sure Python backend is running on port 8081")
                except requests.exceptions.Timeout:
                    st.error("❌ Request timeout. The analysis is taking too long.")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

# ---------------- RIGHT PANEL ----------------
with right:
    st.subheader("Detailed Analysis")
    
    # Check if analysis has been done
    if st.session_state.get('analysis_done', False):
        
        # Get data from session state
        ats_score = st.session_state['ats_score']
        category_scores = st.session_state['category_scores']
        strengths = st.session_state['strengths']
        weaknesses = st.session_state['weaknesses']
        recommendations = st.session_state['recommendations']
        
        # ================= PIE CHART =================
        fig1, ax1 = plt.subplots(
            figsize=(3.5, 3.5),
            facecolor="#12061F"
        )

        ax1.set_facecolor("#12061F")

        ax1.pie(
            [ats_score, 100 - ats_score],
            startangle=90,
            radius=0.70,
            colors=["#7B6CF6", "#3A2D66"],
            wedgeprops=dict(width=0.18, edgecolor="#12061F")
        )

        ax1.text(
            0, 0.05,
            f"{ats_score}%",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            color="white"
        )

        ax1.text(
            0, -0.18,
            "ATS Score",
            ha="center",
            va="center",
            fontsize=11,
            color="#B6AEDC"
        )

        ax1.set_aspect("equal")
        plt.tight_layout(pad=0.6)

        # ================= LINE CHART (FIXED) =================
        categories = [
            "Hard Skills",
            "Soft Skills",
            "Experience",
            "Qualifications"
        ]
        scores = [
            category_scores.get('hardSkills', 0),
            category_scores.get('softSkills', 0),
            category_scores.get('experience', 0),
            category_scores.get('qualifications', 0)
        ]

        x = np.arange(len(categories))
        x_smooth = np.linspace(x.min(), x.max(), 300)

        spl = make_interp_spline(x, scores, k=3)
        y_smooth = spl(x_smooth)

        fig2, ax2 = plt.subplots(
            figsize=(4.5, 3.5),  # Larger size
            facecolor="#12061F"
        )

        ax2.set_facecolor("#12061F")

        # Plot line with better visibility
        ax2.plot(
            x_smooth,
            y_smooth,
            color="#9A8CFF",
            linewidth=2.5
        )

        # Plot points
        ax2.scatter(
            x,
            scores,
            color="#7B6CF6",
            s=80,
            zorder=3,
            edgecolors='white',
            linewidths=1.5
        )

        # Add value labels on points
        for i, score in enumerate(scores):
            ax2.text(
                i, 
                score + 0.15, 
                f'{score:.1f}',
                ha='center',
                va='bottom',
                color='white',
                fontsize=10,
                fontweight='bold'
            )

        ax2.set_xticks(x)
        ax2.set_xticklabels(
            categories,
            color="white",
            fontsize=10,
            rotation=15,
            ha='right'
        )

        # Fix: Add padding to y-axis limits
        ax2.set_ylim(-0.3, 5.5)  # Extended range for better visibility
        ax2.set_yticks([0, 1, 2, 3, 4, 5])
        ax2.set_yticklabels(['0', '1', '2', '3', '4', '5'], color="#B6AEDC", fontsize=10)

        # Remove spines
        for spine in ax2.spines.values():
            spine.set_visible(False)

        # Add subtle grid
        ax2.grid(False)
        ax2.set_axisbelow(True)
        
        plt.tight_layout(pad=1.0)

        # ================= DISPLAY CHARTS SIDE BY SIDE =================
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            st.pyplot(fig1, use_container_width=True)

        with chart_col2:
            st.pyplot(fig2, use_container_width=True)

        # ================= DETAILED FEEDBACK =================
        st.divider()
        
        # Strengths
        st.markdown("### ✅ Strengths")
        for i, strength in enumerate(strengths, 1):
            st.markdown(f"**{i}.** {strength}")
        
        st.divider()
        
        # Weaknesses
        st.markdown("### ⚠️ Areas for Improvement")
        for i, weakness in enumerate(weaknesses, 1):
            st.markdown(f"**{i}.** {weakness}")
        
        st.divider()
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")

    else:
        # Show placeholder when no analysis done
        st.info("👈 Upload a resume and provide a job description to see detailed analysis here")
        
        # Show example visualization
        st.markdown("#### Example Analysis Preview")
        st.image("https://via.placeholder.com/600x400/12061F/7B6CF6?text=Charts+will+appear+here", use_container_width=True)
