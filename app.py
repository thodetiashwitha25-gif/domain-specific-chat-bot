import gradio as gr
from google import genai

from transformers import pipeline
# ============================================================
#                 CONFIGURE GEMINI API
# ============================================================

# Replace with your Gemini API Key
GEMINI_API_KEY = ""

client = genai.Client(
    api_key=GEMINI_API_KEY
)

# ============================================================
#             BERT INTENT CLASSIFIER
# ============================================================

intent_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# ============================================================
#                   CAREER DATABASE
# ============================================================

career_db = {

    "Data Scientist":{

        "skills":[
            "Python",
            "SQL",
            "Statistics",
            "Machine Learning",
            "Deep Learning",
            "Power BI"
        ],

        "roadmap":[
            "Learn Python",
            "Learn Statistics",
            "Learn SQL",
            "Machine Learning",
            "Deep Learning",
            "Build Projects",
            "Apply for Internships"
        ],

        "certifications":[
            "IBM Data Science",
            "Google Data Analytics",
            "Azure AI Fundamentals"
        ],

        "interview":[
            "Python",
            "Statistics",
            "Machine Learning",
            "Pandas",
            "SQL"
        ]
    },

    "AI Engineer":{

        "skills":[
            "Python",
            "Deep Learning",
            "TensorFlow",
            "PyTorch",
            "NLP",
            "Computer Vision"
        ],

        "roadmap":[
            "Python",
            "Machine Learning",
            "Deep Learning",
            "NLP",
            "Projects",
            "Internships"
        ],

        "certifications":[
            "TensorFlow Developer",
            "AWS ML Specialty",
            "Azure AI Engineer"
        ],

        "interview":[
            "Neural Networks",
            "CNN",
            "RNN",
            "Transformers",
            "Python"
        ]
    },

    "Data Analyst":{

        "skills":[
            "Excel",
            "SQL",
            "Power BI",
            "Python",
            "Statistics"
        ],

        "roadmap":[
            "Excel",
            "SQL",
            "Power BI",
            "Python",
            "Projects"
        ],

        "certifications":[
            "Google Data Analytics",
            "Microsoft Power BI"
        ],

        "interview":[
            "SQL",
            "Excel",
            "Power BI",
            "Python"
        ]
    },

    "Python Developer":{

        "skills":[
            "Python",
            "Flask",
            "Django",
            "REST API",
            "SQL"
        ],

        "roadmap":[
            "Python",
            "OOP",
            "Flask",
            "Django",
            "Projects"
        ],

        "certifications":[
            "Python Institute"
        ],

        "interview":[
            "Python",
            "OOP",
            "DSA",
            "APIs"
        ]
    }

}

# ============================================================
#                  INTENT LABELS
# ============================================================

candidate_labels = [

    "Career Guidance",

    "Skill Recommendation",

    "Certification Recommendation",

    "Interview Preparation",

    "Resume Help"

]

# ============================================================
#              ENTITY EXTRACTION
# ============================================================

def extract_role(text):

    for role in career_db.keys():

        if role.lower() in text.lower():

            return role

    return None

# ============================================================
#            INTENT DETECTION
# ============================================================

def detect_intent(text):

    result = intent_classifier(

        text,

        candidate_labels

    )

    return result["labels"][0]

# ============================================================
#          PROMPT CREATION
# ============================================================

def build_prompt(intent, role):

    if role is None:

        return """
You are an AI Career Counselor.

Politely ask the user to mention the job role.
"""

    info = career_db[role]

    prompt = f"""

You are an expert Career Counselor.

Intent:

{intent}

Career:

{role}

Skills:

{', '.join(info['skills'])}

Roadmap:

{', '.join(info['roadmap'])}

Certifications:

{', '.join(info['certifications'])}

Interview Topics:

{', '.join(info['interview'])}

Generate a professional response.

Keep it under 200 words.

"""

    return prompt

# ============================================================
#            RESPONSE GENERATION
# ============================================================

def chatbot(user_query):

    intent = detect_intent(user_query)

    role = extract_role(user_query)

    prompt = build_prompt(

        intent,

        role

    )

    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)


    final_answer = f"""

Intent Detected :

{intent}

Role Detected :

{role}

--------------------------------------

{response.text}

"""

    return final_answer




# Function called by Gradio
def chat(message, history):
    if history is None:
        history = []

    response = chatbot(message)
    history.append((message, response))

    return "", history

with gr.Blocks(title="AI Career Guidance Chatbot") as demo:

    gr.Markdown(
        """
        # 🎓 AI Career Guidance Chatbot
        ### BERT + Entity Extraction + Generative AI
        """
    )

    chatbot_ui = gr.Chatbot(height=450)

    msg = gr.Textbox(
        placeholder="Ask your career question here...",
        label="Your Question"
    )

    with gr.Row():
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")

    send.click(
        chat,
        inputs=[msg, chatbot_ui],
        outputs=[msg, chatbot_ui]
    )

    msg.submit(
        chat,
        inputs=[msg, chatbot_ui],
        outputs=[msg, chatbot_ui]
    )

    clear.click(
        lambda: [],
        outputs=chatbot_ui
    )

    gr.Examples(
        examples=[
            ["How can I become a Data Scientist?"],
            ["What skills are required for AI Engineer?"],
            ["Suggest certifications for Data Analyst"],
            ["How should I prepare for Python Developer interview?"],
            ["Help me improve my resume for Data Scientist"]
        ],
        inputs=msg
    )

demo.launch()
