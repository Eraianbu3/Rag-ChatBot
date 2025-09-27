import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, TypedDict
import warnings
warnings.filterwarnings('ignore')

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()
os.environ['GOOGLE_API_KEY'] = ''

language_mapping = {
    '6': 'Hindi',
    '7': 'Kannada', 
    '11': 'Malayalam',
    '20': 'Tamil',
    '21': 'Telugu',
    '24': 'English'
}

class ChatbotState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    response: str
    language: str
    relevance_score: float
    has_relevant_info: bool

def load_and_process_data():
    # Get absolute path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'bw_courses - Sheet1.csv')
    df = pd.read_csv(data_path)
    
    def process_languages(lang_codes):
        if pd.isna(lang_codes):
            return []
        codes = str(lang_codes).split(',')
        return [language_mapping.get(code.strip(), code.strip()) for code in codes]
    
    df['Languages'] = df['Released Languages'].apply(process_languages)
    df['Language_Names'] = df['Languages'].apply(lambda x: ', '.join(x) if x else 'Not specified')
    
    return df

def create_documents(df):
    documents = []
    
    for _, row in df.iterrows():
        content = f"""Course Title: {row['Course Title']}
Course Description: {row['Course Description']}
Available Languages: {row['Language_Names']}
Target Audience: {row['Who This Course is For']}
Course Number: {row['Course No']}"""
        
        doc = Document(
            page_content=content,
            metadata={
                'course_no': row['Course No'],
                'title': row['Course Title'],
                'languages': row['Languages'],
                'language_codes': row['Released Languages']
            }
        )
        documents.append(doc)
    
    return documents

def setup_rag_system(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    return retriever

def retrieve_documents(state: ChatbotState, retriever) -> ChatbotState:
    query = state['query']
    print(f"DEBUG: retrieve_documents - language is: {state.get('language', 'not set')}")
    state['debug_retrieve_lang'] = state.get('language', 'not set')
    docs = retriever.get_relevant_documents(query)
    state['retrieved_docs'] = docs
    return state

def detect_language(state: ChatbotState) -> ChatbotState:
    # Preserve the user's selected language - don't override it
    print(f"DEBUG: detect_language - language is: {state.get('language', 'not set')}")
    state['debug_detect_lang'] = state.get('language', 'not set')
    return state

def create_relevance_agent(llm):
    def check_course_relevance(query_and_docs):
        parts = query_and_docs.split("|||")
        if len(parts) != 2:
            return "RELEVANT: 0.5"
        
        query, context = parts
        
        if not context.strip():
            return "NOT_RELEVANT: 0.9"
        
        prompt = f"Question: {query}\nCourses: {context[:500]}...\nCan this question be answered from these courses? Reply RELEVANT or NOT_RELEVANT with score 0-1"
        response = llm.invoke(prompt)
        return response
    
    tool = Tool(
        name="course_relevance_checker",
        description="Check if a query can be answered from course data",
        func=check_course_relevance
    )
    
    return tool

def check_relevance(state: ChatbotState, llm) -> ChatbotState:
    query = state['query']
    docs = state['retrieved_docs']
    
    print(f"DEBUG: check_relevance - language is: {state.get('language', 'not set')}")
    state['debug_relevance_lang'] = state.get('language', 'not set')
    
    if not docs:
        state['has_relevant_info'] = False
        state['relevance_score'] = 0.0
        return state
    
    context = "\n".join([doc.page_content[:200] for doc in docs[:3]])
    query_and_context = f"{query}|||{context}"
    
    relevance_tool = create_relevance_agent(llm)
    result = relevance_tool.func(query_and_context)
    
    try:
        if "RELEVANT" in result and "NOT_RELEVANT" not in result:
            state['has_relevant_info'] = True
            score_match = result.split()[-1] if result.split() else "0.8"
            state['relevance_score'] = float(score_match) if score_match.replace('.','').isdigit() else 0.8
        else:
            state['has_relevant_info'] = False
            state['relevance_score'] = 0.3
    except:
        state['has_relevant_info'] = True
        state['relevance_score'] = 0.5
    
    return state

def generate_response(state: ChatbotState, llm) -> ChatbotState:
    query = state['query']
    docs = state['retrieved_docs']
    has_relevant_info = state['has_relevant_info']
    selected_language = state.get('language', 'english')
    
    print(f"DEBUG: Selected language = '{selected_language}'")
    state['debug_language'] = selected_language
    
    if not has_relevant_info:
        state['response'] = generate_no_info_response(query, selected_language)
        return state
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    language_instructions = {
        'hindi': "IMPORTANT: You MUST respond ONLY in Hindi (हिंदी). Do not use English words.",
        'tamil': "IMPORTANT: You MUST respond ONLY in Tamil (தமிழ்). Do not use English words.",
        'telugu': "IMPORTANT: You MUST respond ONLY in Telugu (తెలుగు). Do not use English words.",
        'kannada': "IMPORTANT: You MUST respond ONLY in Kannada (ಕನ್ನಡ). Do not use English words.",
        'malayalam': "IMPORTANT: You MUST respond ONLY in Malayalam (മലയാളം). Do not use English words.",
        'english': "Respond in English."
    }
    
    language_instruction = language_instructions.get(selected_language, "Respond in English.")
    
    language_templates = {
        'hindi': """आपको केवल हिंदी में उत्तर देना है। Boss Wallah कोर्स की जानकारी के आधार पर जवाब दें।

कोर्स जानकारी:
{context}

प्रश्न: {query}

हिंदी में उत्तर:""",
        'tamil': """நீங்கள் தமிழில் மட்டுமே பதிலளிக்க வேண்டும். Boss Wallah பாடநெறி தகவல்களின் அடிப்படையில் பதிலளிக்கவும்।

பாடநெறி தகவல்:
{context}

கேள்வி: {query}

தமிழில் பதில்:""",
        'telugu': """మీరు తెలుగులో మాత్రమే సమాధానం చెప్పాలి. Boss Wallah కోర్స్ సమాచారం ఆధారంగా సమాధానం ఇవ్వండి।

కోర్స్ సమాచారం:
{context}

ప్రశ్న: {query}

తెలుగులో సమాధానం:""",
        'kannada': """ನೀವು ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸಬೇಕು. Boss Wallah ಕೋರ್ಸ್ ಮಾಹಿತಿಯ ಆಧಾರದ ಮೇಲೆ ಉತ್ತರಿಸಿ।

ಕೋರ್ಸ್ ಮಾಹಿತಿ:
{context}

ಪ್ರಶ್ನೆ: {query}

ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರ:""",
        'malayalam': """നിങ്ങൾ മലയാളത്തിൽ മാത്രം ഉത്തരം നൽകണം. Boss Wallah കോഴ്‌സ് വിവരങ്ങളുടെ അടിസ്ഥാനത്തിൽ ഉത്തരം നൽകുക.

കോഴ്‌സ് വിവരങ്ങൾ:
{context}

ചോദ്യം: {query}

മലയാളത്തിൽ ഉത്തര:""",
        'english': """Answer in English based on the Boss Wallah course information.

Course Information:
{context}

Question: {query}

Answer:"""
    }
    
    template = language_templates.get(selected_language, language_templates['english'])
    print(f"DEBUG: Using template for language: {selected_language}")
    print(f"DEBUG: Template starts with: {template[:50]}...")
    state['debug_template'] = template[:100]
    formatted_prompt = template.format(context=context, query=query)
    response = llm.invoke(formatted_prompt)
    
    state['response'] = response
    return state

def generate_no_info_response(query: str, language: str = 'english') -> str:
    query_lower = query.lower()
    
    responses = {
        'english': {
            'location': """I can only provide information about Boss Wallah courses from our dataset. For specific store locations or external services, I recommend:

1. Contacting local suppliers in your area
2. Checking with relevant government departments
3. Consulting with our course mentors who may have local recommendations

However, I'd be happy to help you with information about our related courses that might guide you on what to look for!""",
            'external': """I'm specifically designed to help with Boss Wallah course information. I can't provide information about topics outside our course dataset.

Is there anything about our courses I can help you with instead?""",
            'general': """I apologize, but I don't have relevant information about that in our Boss Wallah course dataset. 

I can help you with:
- Course details and descriptions
- Target audiences for different courses  
- Available languages for courses
- Recommendations based on your interests or background

Is there a specific course topic you'd like to know more about?"""
        },
        'hindi': {
            'location': """मैं केवल हमारे डेटासेट से Boss Wallah कोर्स की जानकारी प्रदान कर सकता हूं। विशिष्ट स्टोर स्थानों या बाहरी सेवाओं के लिए, मैं सुझाता हूं:

1. अपने क्षेत्र के स्थानीय आपूर्तिकर्ताओं से संपर्क करें
2. संबंधित सरकारी विभागों से जांच करें  
3. हमारे कोर्स मेंटर्स से सलाह लें जो स्थानीय सप्लायर की सिफारिश कर सकते हैं

हालांकि, मैं आपको हमारे संबंधित कोर्स की जानकारी देकर मदद कर सकता हूं!""",
            'external': """मैं विशेष रूप से Boss Wallah कोर्स की जानकारी के लिए डिज़ाइन किया गया हूं। मैं हमारे कोर्स डेटासेट के बाहर की जानकारी प्रदान नहीं कर सकता।

क्या आपको हमारे कोर्स के बारे में कुछ और जानना है?""",
            'general': """माफ करें, हमारे Boss Wallah कोर्स डेटासेट में इसकी जानकारी नहीं है।

मैं इन चीजों में आपकी मदद कर सकता हूं:
- कोर्स विवरण और जानकारी
- विभिन्न कोर्स के लक्षित दर्शक
- कोर्स की उपलब्ध भाषाएं
- आपकी रुचि के आधार पर सिफारिशें

कोई विशिष्ट कोर्स टॉपिक के बारे में जानना चाहते हैं?"""
        },
        'malayalam': {
            'location': """ഞാൻ ഞങ്ങളുടെ ഡാറ്റാസെറ്റിൽ നിന്നുള്ള Boss Wallah കോഴ്‌സുകളെ കുറിച്ചുള്ള വിവരങ്ങൾ മാത്രമേ നൽകാൻ കഴിയൂ. നിർദ്ദിഷ്ട സ്റ്റോർ ലൊക്കേഷനുകൾ അല്ലെങ്കിൽ ബാഹ്യ സേവനങ്ങൾക്കായി ഞാൻ ശുപാർശ ചെയ്യുന്നു:

1. നിങ്ങളുടെ പ്രദേശത്തെ പ്രാദേശിക വിതരണക്കാരുമായി ബന്ധപ്പെടുക
2. അനുബന്ധ സർക്കാർ വകുപ്പുകളുമായി പരിശോധിക്കുക
3. പ്രാദേശിക ശുപാർശകൾ ഉണ്ടായിരിക്കാവുന്ന ഞങ്ങളുടെ കോഴ്‌സ് മെന്റർമാരുമായി കൂടിയാലോചിക്കുക

എന്നിരുന്നാലും, എന്തെല്ലാം തിരയണമെന്ന് നിങ്ങളെ സഹായിച്ചേക്കാവുന്ന ഞങ്ങളുടെ അനുബന്ധ കോഴ്‌സുകളെ കുറിച്ചുള്ള വിവരങ്ങൾ നൽകാൻ ഞാൻ സന്തോഷിക്കുന്നു!""",
            'external': """ഞാൻ പ്രത്യേകമായി Boss Wallah കോഴ്‌സ് വിവരങ്ങൾ സഹായിക്കാൻ രൂപകൽപ്പന ചെയ്തിട്ടുള്ളതാണ്. ഞങ്ങളുടെ കോഴ്‌സ് ഡാറ്റാസെറ്റിന് പുറത്തുള്ള വിഷയങ്ങളെ കുറിച്ചുള്ള വിവരങ്ങൾ എനിക്ക് നൽകാൻ കഴിയില്ല.

പകരം ഞങ്ങളുടെ കോഴ്‌സുകളെ കുറിച്ച് എന്തെങ്കിലും സഹായിക്കാൻ കഴിയുമോ?""",
            'general': """ക്ഷമിക്കുക, ഞങ്ങളുടെ Boss Wallah കോഴ്‌സ് ഡാറ്റാസെറ്റിൽ അതിനെ കുറിച്ചുള്ള പ്രസക്ത വിവരങ്ങൾ എനിക്കില്ല.

എനിക്ക് നിങ്ങളെ സഹായിക്കാൻ കഴിയും:
- കോഴ്‌സിന്റെ വിശദാംശങ്ങളും വിവരണങ്ങളും
- വിവിധ കോഴ്‌സുകളുടെ ടാർഗെറ്റ് പ്രേക്ഷകർ
- കോഴ്‌സുകൾക്കായി ലഭ്യമായ ഭാഷകൾ
- നിങ്ങളുടെ താൽപ്പര്യങ്ങളോ പശ്ചാത്തലമോ അടിസ്ഥാനമാക്കിയുള്ള ശുപാർശകൾ

നിങ്ങൾക്ക് അറിയാൻ താൽപ്പര്യമുള്ള ഏതെങ്കിലും നിർദ്ദിഷ്ട കോഴ്‌സ് വിഷയം ഉണ്ടോ?"""
        },
        'kannada': {
            'location': """ನಾನು ನಮ್ಮ ಡೇಟಾಸೆಟ್‌ನಿಂದ Boss Wallah ಕೋರ್ಸ್‌ಗಳ ಬಗ್ಗೆ ಮಾತ್ರ ಮಾಹಿತಿ ನೀಡಬಲ್ಲೆ. ನಿರ್ದಿಷ್ಟ ಅಂಗಡಿ ಸ್ಥಳಗಳು ಅಥವಾ ಬಾಹ್ಯ ಸೇವೆಗಳಿಗಾಗಿ, ನಾನು ಶಿಫಾರಸು ಮಾಡುತ್ತೇನೆ:

1. ನಿಮ್ಮ ಪ್ರದೇಶದ ಸ್ಥಳೀಯ ಪೂರೈಕೆದಾರರನ್ನು ಸಂಪರ್ಕಿಸಿ
2. ಸಂಬಂಧಿತ ಸರ್ಕಾರಿ ಇಲಾಖೆಗಳೊಂದಿಗೆ ಪರಿಶೀಲಿಸಿ
3. ಸ್ಥಳೀಯ ಶಿಫಾರಸುಗಳನ್ನು ಹೊಂದಿರುವ ನಮ್ಮ ಕೋರ್ಸ್ ಮೆಂಟರ್‌ಗಳೊಂದಿಗೆ ಸಲಹೆ ಮಾಡಿ

ಆದಾಗ್ಯೂ, ನೀವು ಏನನ್ನು ಹುಡುಕಬೇಕೆಂದು ನಿಮಗೆ ಮಾರ್ಗದರ್ಶನ ನೀಡುವ ನಮ್ಮ ಸಂಬಂಧಿತ ಕೋರ್ಸ್‌ಗಳ ಬಗ್ಗೆ ಮಾಹಿತಿಯೊಂದಿಗೆ ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ನಾನು ಸಂತೋಷಪಡುತ್ತೇನೆ!""",
            'external': """ನಾನು ನಿರ್ದಿಷ್ಟವಾಗಿ Boss Wallah ಕೋರ್ಸ್ ಮಾಹಿತಿಗೆ ಸಹಾಯ ಮಾಡಲು ವಿನ್ಯಾಸಗೊಳಿಸಲಾಗಿದೆ. ನಮ್ಮ ಕೋರ್ಸ್ ಡೇಟಾಸೆಟ್‌ನ ಹೊರಗಿನ ವಿಷಯಗಳ ಬಗ್ಗೆ ಮಾಹಿತಿ ನೀಡಲು ನನಗೆ ಸಾಧ್ಯವಿಲ್ಲ.

ಬದಲಿಗೆ ನಮ್ಮ ಕೋರ್ಸ್‌ಗಳ ಬಗ್ಗೆ ನಿಮಗೆ ಸಹಾಯ ಮಾಡಬಹುದಾದ ಏನಾದರೂ ಇದೆಯೇ?""",
            'general': """ಕ್ಷಮಿಸಿ, ನಮ್ಮ Boss Wallah ಕೋರ್ಸ್ ಡೇಟಾಸೆಟ್‌ನಲ್ಲಿ ಅದರ ಬಗ್ಗೆ ಸಂಬಂಧಿತ ಮಾಹಿತಿ ನನ್ನ ಬಳಿ ಇಲ್ಲ.

ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ:
- ಕೋರ್ಸ್ ವಿವರಗಳು ಮತ್ತು ವಿವರಣೆಗಳು
- ವಿವಿಧ ಕೋರ್ಸ್‌ಗಳ ಗುರಿ ಪ್ರೇಕ್ಷಕರು
- ಕೋರ್ಸ್‌ಗಳಿಗೆ ಲಭ್ಯವಿರುವ ಭಾಷೆಗಳು
- ನಿಮ್ಮ ಆಸಕ್ತಿಗಳು ಅಥವಾ ಹಿನ್ನೆಲೆಯ ಆಧಾರದ ಮೇಲೆ ಶಿಫಾರಸುಗಳು

ನೀವು ಇನ್ನಷ್ಟು ತಿಳಿದುಕೊಳ್ಳಲು ಬಯಸುವ ಯಾವುದೇ ನಿರ್ದಿಷ್ಟ ಕೋರ್ಸ್ ವಿಷಯ ಇದೆಯೇ?"""
        },
        'tamil': {
            'location': """நாங்கள் எங்கள் டேட்டாசெட்டில் இருந்து Boss Wallah பாடநெறிகளைப் பற்றிய தகவல்களை மட்டுமே வழங்க முடியும். குறிப்பிட்ட கடை இடங்கள் அல்லது வெளிப்புற சேவைகளுக்கு, நான் பரிந்துரைக்கிறேன்:

1. உங்கள் பகுதியில் உள்ள உள்ளூர் சப்ளையர்களைத் தொடர்பு கொள்ளுங்கள்
2. தொடர்புடைய அரசாங்க துறைகளுடன் சரிபார்க்கவும்
3. உள்ளூர் பரிந்துரைகள் இருக்கக்கூடிய எங்கள் பாடநெறி வழிகாட்டிகளுடன் ஆலோசிக்கவும்

இருப்பினும், நீங்கள் எதைத் தேட வேண்டும் என்பதற்கு வழிகாட்டக்கூடிய எங்கள் தொடர்புடைய பாடநெறிகளைப் பற்றிய தகவل்களுடன் உங்களுக்கு உதவ நான் மகிழ்ச்சியடைவேன்!""",
            'external': """நான் குறிப்பாக Boss Wallah பாடநெறி தகவலுக்கு உதவ வடிவமைக்கப்பட்டுள்ளேன். எங்கள் பாடநெறி டேட்டாசெட்டுக்கு வெளியே உள்ள தலைப்புகளைப் பற்றிய தகவல்களை என்னால் வழங்க முடியாது.

மாறாக எங்கள் பாடநெறிகளைப் பற்றி நான் உங்களுக்கு எதையாவது உதவ முடியுமா?""",
            'general': """மன்னிக்கவும், எங்கள் Boss Wallah பாடநெறி டேட்டாசெட்டில் அதைப் பற்றிய தொடர்புடைய தகவல்கள் என்னிடம் இல்லை.

என்னால் உங்களுக்கு உதவ முடியும்:
- பாடநெறி விவரங்கள் மற்றும் விளக்கங்கள்
- பல்வேறு பாடநெறிகளுக்கான இலக்கு பார்வையாளர்கள்
- பாடநெறிகளுக்கு கிடைக்கும் மொழிகள்
- உங்கள் ஆர்வங்கள் அல்லது பின்னணியின் அடிப்படையில் பரிந்துரைகள்

நீங்கள் மேலும் அறிய விரும்பும் ஏதேனும் குறிப்பிட்ட பாடநெறி தலைப்பு உள்ளதா?"""
        },
        'telugu': {
            'location': """నేను మా డేటాసెట్ నుండి Boss Wallah కోర్సుల గురించి మాత్రమే సమాచారం అందించగలను. నిర్దిష్ట స్టోర్ లొకేషన్లు లేదా బాహ్య సేవల కోసం, నేను సిఫార్సు చేస్తున్నాను:

1. మీ ప్రాంతంలోని స్థానిక సప్లైయర్లను సంప్రదించండి
2. సంబంధిత ప్రభుత్వ విభాగాలతో తనిఖీ చేయండి
3. స్థానిక సిఫార్సులను కలిగి ఉండే మా కోర్స్ మెంటర్లతో సంప్రదించండి

అయితే, మీరు దేని కోసం వెతకాలో మీకు మార్గదర్శనం చేసే మా సంబంధిత కోర్సుల గురించిన సమాచారంతో మీకు సహాయం చేయడానికి నేను సంతోషిస్తాను!""",
            'external': """నేను ప్రత్యేకంగా Boss Wallah కోర్స్ సమాచారానికి సహాయం చేయడానికి రూపొందించబడ్డాను. మా కోర్స్ డేటాసెట్ వెలుపలి అంశాల గురించి సమాచారం అందించలేను.

బదులుగా మా కోర్సుల గురించి నేను మీకు ఏదైనా సహాయం చేయగలనా?""",
            'general': """క్షమించండి, మా Boss Wallah కోర్స్ డేటాసెట్లో దాని గురించి సంబంధిత సమాచారం నా దగ్గర లేదు.

నేను మీకు సహాయం చేయగలను:
- కోర్స్ వివరాలు మరియు వర్ణనలు
- వివిధ కోర్సుల లక్ష్య ప్రేక్షకులు
- కోర్సులకు అందుబాటులో ఉన్న భాషలు
- మీ ఆసక్తులు లేదా నేపథ్యం ఆధారంగా సిఫార్సులు

మీరు మరింత తెలుసుకోవాలని అనుకునే ఏదైనా నిర్దిష్ట కోర్స్ అంశం ఉందా?"""
        }
    }
    
    lang_responses = responses.get(language, responses['english'])
    
    if any(keyword in query_lower for keyword in ['store', 'shop', 'where to buy', 'location', 'address']):
        return lang_responses['location']
    elif any(keyword in query_lower for keyword in ['weather', 'news', 'stock price', 'current events']):
        return lang_responses['external']
    else:
        return lang_responses['general']

def route_decision(state: ChatbotState) -> str:
    decision = "generate_response" if state['has_relevant_info'] else "generate_no_info"
    print(f"DEBUG: route_decision - routing to: {decision}")
    state['debug_route'] = decision
    return decision

class BossWallahChatbot:
    def __init__(self):
        self.df = load_and_process_data()
        self.documents = create_documents(self.df)
        self.retriever = setup_rag_system(self.documents)
        self.llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        self.app = self.setup_langgraph()
    
    def setup_langgraph(self):
        workflow = StateGraph(ChatbotState)
        
        workflow.add_node("retrieve", lambda state: retrieve_documents(state, self.retriever))
        workflow.add_node("detect_language", detect_language)
        workflow.add_node("check_relevance", lambda state: check_relevance(state, self.llm))
        workflow.add_node("generate_response", lambda state: generate_response(state, self.llm))
        workflow.add_node("generate_no_info", lambda state: {**state, "response": generate_no_info_response(state['query'], state.get('language', 'english'))})
        
        workflow.set_entry_point("retrieve")
        
        workflow.add_edge("retrieve", "detect_language")
        workflow.add_edge("detect_language", "check_relevance")
        
        workflow.add_conditional_edges(
            "check_relevance",
            route_decision,
            {
                "generate_response": "generate_response",
                "generate_no_info": "generate_no_info"
            }
        )
        
        workflow.add_edge("generate_response", END)
        workflow.add_edge("generate_no_info", END)
        
        return workflow.compile()
    
    def ask(self, question: str, language: str = "english") -> str:
        initial_state = {
            "query": question,
            "retrieved_docs": [],
            "response": "",
            "language": language,
            "relevance_score": 0.0,
            "has_relevant_info": False
        }
        
        result = self.app.invoke(initial_state)
        return result.get('response', 'No response generated')

def main():
    print("Initializing Boss Wallah AI Support Agent...")
    chatbot = BossWallahChatbot()
    
    print("Boss Wallah AI Support Agent")
    print("Ask me anything about our courses! (Type 'quit' to exit)")
    print("="*50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Thank you for using Boss Wallah Support! Have a great day!")
            break
            
        if not user_input:
            continue
            
        try:
            response = chatbot.ask(user_input)
            print(f"\nBot: {response}")
        except Exception as e:
            print(f"\nSorry, I encountered an error: {str(e)}")
            print("Please try asking your question differently.")

if __name__ == "__main__":
    main()