# Trip Explorer

Welcome to the Trip Explorer Streamlit app! This app leverages advanced language models and vector databases to deliver personalized travel recommendations based on user queries. It provides a simple and intuitive interface to help you plan your trips by suggesting destinations, activities, and durations tailored to your preferences.

### Streamlit App:  https://trip-explorer.streamlit.app/

### Youtube link: 

## Features

- **Personalized Travel Recommendations**: Get tailored suggestions for destinations, activities, and durations based on your input.
- **Advanced Language Models**: Utilizes ChatGroq and Google Generative AI Embeddings for accurate and insightful responses.
- **Streamlit Interface**: Simple and user-friendly interface to interact with the app.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

### **Clone the Repository**

   ```bash
   https://github.com/Venkata-Bhargavi/Itinerary-Recommender-RAG.git
   ```

### Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install required packages

`pip install -r requirements.txt
`
## Setup Guide

### Set Up Environment Variables

Create a `.env` file in the root directory of the project and add your API keys:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key

```

Running the App
Start the Streamlit App
Run the following command to start the Streamlit app:


`streamlit run app.py
`

### Interact with the App:

Open your web browser and go to http://localhost:8501 to interact with the Trip Explorer app. Enter your travel preferences or questions to receive personalized recommendations.

