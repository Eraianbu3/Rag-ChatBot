#AI Support Agent

A sophisticated RAG-based chatbot built with LangChain, LangGraph, and Google Gemini LLM that provides intelligent support for courses. The system uses advanced conditional routing to handle dataset queries and beyond-dataset requests appropriately.

## Features

- **RAG-based responses**: Advanced vector similarity search with FAISS
- **Conditional routing**: Intelligently distinguishes dataset vs beyond-dataset queries
- **Multilingual support**: English, Hindi, Tamil, Telugu, Kannada, Malayalam (6 languages)
- **LangGraph workflow**: Structured conversation flow with advanced state management
- **FastAPI backend**: High-performance REST API for programmatic access
- **Streamlit frontend**: Interactive web interface for easy testing


## Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API Key (get from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Step-by-Step Setup Instructions

#### Step 1: 
```bash
cd Rag_chatbot
```

#### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Set Google Gemini API Key
**Option A**: Create a `.env` file in the root directory:
```bash
GOOGLE_API_KEY=your_api_key_here
```

**Option B**: Update the `GOOGLE_API_KEY` directly in `src/main.py` in (line 17) and `src/app.py` in (line 14)

#### Step 5: Verify Dataset
The course dataset (100 courses) is included at:
```
data/bw_courses - Sheet1.csv
```

#### Step 6: Test Installation
```bash
# On Windows (recommended for encoding support)
set PYTHONIOENCODING=utf-8 && python src/main.py

# On Linux/macOS
python src/main.py
```

### Language Support
The system supports 6 languages with proper language code mapping:
- **6**: Hindi | **7**: Kannada | **11**: Malayalam
- **20**: Tamil | **21**: Telugu | **24**: English

## How to Run the Project

### Method 1: Command Line Interface
```bash
# On Windows (recommended for multilingual support)
set PYTHONIOENCODING=utf-8 && python src/main.py

# On Linux/macOS
python src/main.py
```
Simple CLI interface for direct interaction with the chatbot.

### Method 2: Streamlit Web App (Recommended)
```bash
# Default port
streamlit run src/app.py --server.port 8505

# Alternative port (if 8505 is busy - try to run on other port (8504 , 8506 if they are occupied)
streamlit run src/app.py --server.port 8502

# With encoding support (Windows)
set PYTHONIOENCODING=utf-8 && streamlit run src/app.py --server.port 8505
```
**Visit:** http://localhost:8505 (or your specified port)


### Method 3: FastAPI Server
```bash
cd src
python api.py
```

**Alternative start methods:**
```bash
# Method 3a: Using uvicorn module
cd src
python -m uvicorn api:app --host 0.0.0.0 --port 8001 --reload

# Method 3b: Using uvicorn directly
cd src
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

**Available Endpoints:**
- **API Base:** http://localhost:8001
- **Interactive API Docs:** http://localhost:8001/docs
- **Health Check:** http://localhost:8001/health


## Test Questions & Expected Behavior

### Course Information Queries
   - "Tell me about honey bee farming course"
   - "I want to learn how to start a poultry farm"
   - "Do you have any courses in Tamil?"
   - "I am a recent high school graduate, are there any opportunities for me?"
   
   **Expected:** Detailed course information from dataset with multilingual support

### Beyond Dataset Queries
   - "Where can I buy seeds for papaya farming near Whitefield, Bangalore?"
   - "What's the weather like today?"
   - "What is the current stock price of Boss Wallah?"
   
   **Expected:** Acknowledges limitation, provides guidance, suggests course alternatives

### Irrelevant Queries
   - "How to cook pasta?"
   - "Tell me a joke"
   - "What is machine learning?"
   
   **Expected:** Politely redirects to course-related topics





### LangGraph-based Workflow with Conditional Routing

```
User Query → Retrieve Documents → Detect Language → Check Relevance
                                                            ↓
                                                 [Has Info?] [No Info?]
                                                     ↓           ↓
                                               LLM Response  Template Response
                                                     ↓           ↓
                                                 Final Response
```

### Workflow Components
1. **Retrieve Documents**: FAISS vector search finds relevant courses from dataset
2. **Detect Language**: Identifies user's preferred response language (6 languages supported)
3. **Check Relevance**: LLM determines if query can be answered from Boss Wallah dataset
4. **Conditional Routing**: Smart routing to appropriate response generation method
5. **Response Generation**: Creates multilingual responses based on routing decision



## Project Structure

```
AI Support Agent/
├── src/                              # Source code
│   ├── main.py                       # Core chatbot implementation (LangGraph)
│   ├── api.py                        # FastAPI REST API server  
│   └── app.py                        # Streamlit web interface
├── data/                             # Dataset
│   └── bw_courses - Sheet1.csv       # courses (100 courses)
├── docs/                             # Documentation folder                    
├── workflow_visualization.ipynb      # Interactive workflow exploration
├── workflow_diagram.png             # Auto-generated workflow diagram
├── requirements.txt                 # Python dependencies
├── .env.example                    # Environment variables template
├── .env                            # Environment variables (create this)
└── README.md                       # This documentation
```

## Response Guidelines & Behavior

| Query Type | Behavior | Example |
|-----------|----------|---------|
| **Dataset Queries** | Detailed course info from dataset | "Tell me about honey bee farming" |
| **Beyond Dataset** | Acknowledges limitation, offers guidance | "Store locations near me" |
| **Irrelevant** | Politely redirects to course topics | "Tell me a joke" |
| **Multilingual** | Responds in selected language | Works in all 6 supported languages |

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Google Gemini 2.5 Flash | Large language model for responses |
| **Framework** | LangChain | Document processing & LLM integration |
| **Workflow** | LangGraph | Advanced workflow orchestration |
| **Vector Search** | FAISS | High-performance similarity search |
| **API** | FastAPI | Modern, fast REST API framework |
| **Web UI** | Streamlit | Interactive web interface |
| **Data Processing** | Pandas | Data manipulation and analysis |
| **Visualization** | Matplotlib + Jupyter | Workflow diagrams and exploration |
| **Languages** | 6 Languages | Hindi, Tamil, Telugu, Kannada, Malayalam, English |


### Technical Implementation
| Requirement | Status | Details |
|-------------|--------|---------|
| **RAG-based chatbot** | Complete | Uses provided dataset only with FAISS |
| **Conditional routing** | Complete | LangGraph workflow with decision nodes |
| **LLM Integration** | Complete | Google Gemini 2.5 Flash |
| **Project structure** | Complete | Clean, organized, well-documented |
| **Multiple interfaces** | Bonus | CLI, Web UI, API, Jupyter notebook |

## Project Highlights
- **100% Assignment Completion**: All main and bonus tasks implemented
- **Advanced Architecture**: LangGraph workflow with conditional routing  
- **Comprehensive Multilingual**: 6 languages with proper code mapping
- **Interactive Visualization**: Jupyter notebook with workflow diagrams
- **Professional Documentation**: Enhanced README with clear instructions
- **Multiple Deployment Options**: 4 different usage methods
- **Robust Encoding Support**: Handles multilingual content on all platforms
- **Path Resolution**: Automatic dataset detection from any directory

## Troubleshooting

### Common Issues
1. **API Key Error**: Make sure your Google Gemini API key is set correctly
   - Set in `.env` file or `src/main.py`
2. **Module Import Error**: Ensure all dependencies are installed: `pip install -r requirements.txt`
3. **Character Encoding Issues (Windows)**: 
   - **Error**: `'charmap' codec can't encode characters`
   - **Solution**: Run with UTF-8 encoding: `set PYTHONIOENCODING=utf-8 && python src/main.py`
   - **Alternative**: The project now auto-detects encoding and handles multilingual text
4. **Port Already in Use**: 
   - For Streamlit: `streamlit run src/app.py --server.port 8502` (if 8505 is busy)
   - For FastAPI: The API runs on port 8001 by default (changed from 8000 to avoid conflicts)
5. **Dataset Not Found**: Verify the CSV file exists at `data/bw_courses - Sheet1.csv`
   - The project now uses absolute paths and auto-detects the dataset location
6. **FastAPI Server Issues**: Make sure to run from the `src` directory: `cd src && python api.py`




## Contributing & Development

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run in development mode
export GOOGLE_API_KEY=your_key_here  # Linux/macOS
set GOOGLE_API_KEY=your_key_here     # Windows

# Test all components
python src/main.py           # CLI
streamlit run src/app.py     # Web UI
python src/api.py           # FastAPI
```



