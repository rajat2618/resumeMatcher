import streamlit as st
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np



# -------------------------
# AI AGENT
# -------------------------


class ResumeAgent:


    def __init__(self):


        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )


        self.jobs = [

        {
        "title":"AI Engineer",
        "description":
        "Python machine learning deep learning NLP transformers PyTorch AI models"
        },


        {
        "title":"Data Scientist",
        "description":
        "Python statistics machine learning data analysis SQL visualization"
        },


        {
        "title":"Backend Developer",
        "description":
        "Python Java APIs databases Django Flask software development"
        },


        {
        "title":"Cloud Engineer",
        "description":
        "AWS cloud DevOps Linux Docker Kubernetes infrastructure"
        },


        {
        "title":"ML Engineer",
        "description":
        "Machine learning models deployment MLOps TensorFlow PyTorch"
        }

        ]


        self.job_vectors = self.model.encode(

            [
             j["description"]

             for j in self.jobs
            ]

        )


        self.index = faiss.IndexFlatL2(

            self.job_vectors.shape[1]

        )


        self.index.add(

            np.array(
                self.job_vectors
            ).astype("float32")

        )




    def read_resume(self,file):


        reader=PdfReader(file)


        text=""


        for page in reader.pages:

            text += page.extract_text()


        return text




    def analyze(self,resume):


        vector=self.model.encode(

            resume

        )


        result=self.index.search(

            np.array(
            [vector]
            ).astype("float32"),

            3

        )


        recommendations=[]


        for i in result[1][0]:


            job=self.jobs[i]


            recommendations.append(

            {
            "job":job["title"],

            "reason":
            "Your resume skills match with "+
            job["description"]

            }

            )


        return recommendations




# -------------------------
# STREAMLIT UI
# -------------------------


st.title(
"🤖 Resume Job Recommendation Agent"
)



st.write(
"Upload your resume and AI will suggest suitable jobs"
)



agent=ResumeAgent()



uploaded = st.file_uploader(

"Upload Resume PDF",

type=["pdf"]

)



if uploaded:


    resume_text = agent.read_resume(
        uploaded
    )


    st.subheader(
    "Resume Extracted"
    )


    st.write(
    resume_text[:1000]
    )



    if st.button(
    "Analyze Resume"
    ):


        results=agent.analyze(

            resume_text

        )


        st.subheader(
        "Best Job Matches"
        )


        for r in results:


            st.success(

                r["job"]

            )


            st.write(

                r["reason"]

            )