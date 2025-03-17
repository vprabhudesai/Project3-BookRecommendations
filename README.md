# AI Book Recommendation System

A semantic search and recommendation system that helps users discover books based on their specific interests, using a dataset from [**GoodReads**](https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k/data), natural language processing and AI-generated summaries.

## Overview

This project combines the power of sentence transformers for semantic understanding with Google's Gemini AI for natural language generation to create an intelligent book recommendation engine. Users can describe their interests in natural language, and the system will find and present relevant books with beautifully formatted summaries.

## Presentation
[Beyond Keywords: Semantic Search for Literary Exploration](https://docs.google.com/presentation/d/1nKm9c0ns3QdyMI1paktGKJS5YdN47fc0lPe3gPaRLRI/edit#slide=id.p1)

## Features

- **Semantic Search**: Understand the meaning behind user queries, not just keywords
- **Two-Step Recommendation Process**: Filter by genre first for better performance and relevance
- **Hybrid Ranking Algorithm**: Combines relevance, quality ratings, and popularity
- **AI-Generated Summaries**: Beautiful, context-aware book recommendations using Google Gemini
- **Intuitive User Interface**: Easy-to-use Gradio web interface
- **Performance Optimized**: Smart caching mechanism for faster responses

## Technology Stack

### Core Technologies

- **Python**: Primary programming language for the entire application
- **Jupyter Notebook**: Development and execution environment

### Data Processing

- **Pandas**: Data manipulation library for dataset handling
- **NumPy**: Numerical computation for vector operations and scoring
- **Regular Expressions**: Pattern matching for text cleaning and normalization

### Natural Language Processing

- **Sentence Transformers**: Neural network model for creating semantic embeddings
  - Model: `all-MiniLM-L6-v2` (384-dimensional vectors)
  - Capabilities: Semantic similarity, cross-lingual matching, document retrieval

- **Scikit-learn**: Implementation of cosine similarity metrics
  - Used for computing vector distances in semantic space
  - Optimized for speed with numerical operations

### Artificial Intelligence

- **Google Generative AI**: Library for interfacing with Google's Gemini models
  - Model: `gemini-1.5-pro`
  - Usage: Natural language generation for book summaries

### User Interface

- **Gradio**: Framework for creating browser-based interfaces
  - Components: Tabs, dropdowns, text inputs, markdown display
  - Features: Event handling, theme customization, responsive design
  - Deployment: Local server with browser access

### Environment Management

- **python-dotenv**: Library for loading environment variables
  - Used for secure API key management
  - Prevents credentials from being stored in the code
 
### Presentation Development

 - **Gamma**: AI powered presentations and images from outline/prompt

## Data Preprocessing

The recommendation engine relies on clean, well-structured data. The system implements several preprocessing steps to ensure optimal performance:

### Dataset Preparation

- **Source**: Uses the GoodReads 100k Books dataset with comprehensive metadata
- **Initial Cleaning**: Removes unnecessary columns like URLs, ISBNs, and image links
- **Null Handling**: Drops rows with missing descriptions or genre information
- **Text Normalization**: Standardizes text formatting and encoding

### Genre Processing

- **Genre Parsing**: Converts comma-separated genre strings into Python lists
- **Genre Extraction**: Creates a comprehensive set of all unique genres in the dataset
- **Genre Filtering**: Enables efficient subsetting of the data by genre

### Text Cleaning

- **Character Normalization**: Handles Unicode and special characters
- **Encoding Fixes**: Replaces problematic character sequences (e.g., 'â€™' → "'")
- **Whitespace Management**: Normalizes spacing and removes redundant whitespace

### Embedding Generation

- **Vectorization**: Converts book descriptions into numerical vectors using SentenceTransformer
- **Model Selection**: Uses 'all-MiniLM-L6-v2' for its balance of performance and accuracy
- **Tensor Conversion**: Stores embeddings as tensors for efficient similarity computation
- **On-Demand Processing**: Computes embeddings only for the selected genre to conserve memory

### Performance Optimization

- **Subset-based Processing**: Works with genre-specific subsets instead of the full dataset
- **Caching Strategy**: Stores computed embeddings in a dictionary keyed by genre
- **Memory Management**: Releases unnecessary data to manage memory consumption
- **Incremental Loading**: Processes data only when needed, improving startup time

### Data Structure

The core data structure is a pandas DataFrame with the following key columns:

- `title`: Book title
- `author`: Book author(s)
- `desc`: Full book description
- `genre`: List of genres associated with the book
- `rating`: Average user rating (0-5 scale)
- `totalratings`: Number of user ratings
- `similarity`: Computed similarity to the query (added during recommendation)
- `final_score`: Combined ranking score (added during recommendation)

## Recommendation Algorithm

The book recommendation system uses a sophisticated multi-step approach to find the most relevant books based on a user's natural language query:

### Semantic Understanding

- **Vector Embeddings**: Book descriptions are converted into high-dimensional vectors (384 dimensions) using the SentenceTransformer model `all-MiniLM-L6-v2`
- **Semantic Space**: These vectors capture the semantic meaning of each book, placing similar books closer together in the vector space
- **Query Encoding**: User queries are encoded using the same model, ensuring they exist in the same semantic space as the book descriptions

### Similarity Computation

- **Cosine Similarity**: Measures the angle between the query vector and each book description vector
- **Mathematical Formula**:
  ```
  similarity = cos(θ) = (A·B)/(||A||·||B||)
  ```
  where A is the query vector and B is a book description vector
- **Normalization**: Similarity scores range from 0 to 1, with 1 indicating perfect similarity

### Hybrid Ranking System

The system employs a weighted ranking algorithm that balances three key factors:

1. **Semantic Relevance (50% weight)**:
   - How well the book's description matches the user's query
   - Directly uses the cosine similarity score (0-1 scale)

2. **Book Quality (30% weight)**:
   - Based on the book's average rating from Goodreads
   - Normalized to a 0-1 scale (dividing by 5, the maximum rating)

3. **Popularity (20% weight)**:
   - Based on the number of ratings a book has received
   - Logarithmically scaled to prevent bestsellers from dominating:
     ```
     normalized_popularity = log(1 + num_ratings) / log(1 + max_ratings)
     ```
   - This approach ensures niche but relevant books can still rank well

### Final Scoring Formula

```python
final_score = (
    (similarity * 0.5) +
    (normalized_rating * 0.3) +
    (normalized_popularity * 0.2)
)
```
## Examples of Scoring Formula

### Genre - Aircraft
### Query - The Invention of Airplanes

1. **BOOK 2463:**
  - Title: Aircraft of World War II
  - Author: Jim Winchester
  - Rating: 4.48 out of 5 (based on 25 ratings)
  - Similarity to query: 0.44

2. **BOOK 75033:**
  - Title: World's Greatest Aircraft
  - Author: Christopher Chant
  - Rating: 4.45 out of 5 (based on 40 ratings)
  - Similarity to query: 0.41

3. **BOOK 63679:**
  - Title: The Complete Encyclopedia of World Aircraft
  - Author: Soph Moeng
  - Rating: 4.7 out of 5 (based on 10 ratings)
  - Similarity to query: 0.43

4. **BOOK 83158:**
  - Title: The Ransom of Black Stealth One
  - Author: Dean Ing
  - Rating: 3.83 out of 5 (based on 180 ratings)
  - Similarity to query: 0.43

### Genre - Basketball
### Query - Michael Jordan Era

1. **BOOK 46035:**
  - Title: Dream Team: How Michael, Magic, Larry, Charles, and the Greatest Team of All Time Conquered the World and Changed the Game of Basketball Forever
  - Author: Jack McCallum
  - Rating: 4.2 out of 5 (based on 11375 ratings)
  - Similarity to query: 0.59

2. **BOOK 92137:**
  - Title: For the Love of the Game: My Story
  - Author: Michael Jordan,Mark Vancil
  - Rating: 4.2 out of 5 (based on 647 ratings)
  - Similarity to query: 0.60

3. **BOOK 92133:**
  - Title: Michael Jordan
  - Author: Coleen Lovitt,Coleen Lovitt
  - Rating: 4.32 out of 5 (based on 276 ratings)
  - Similarity to query: 0.61

4. **BOOK 92129:**
  - Title: I Can't Accept Not Trying: Michael Jordan on the Pursuit of Excellence
  - Author: Michael Jordan,Sandro Miller
  - Rating: 4.18 out of 5 (based on 683 ratings)
  - Similarity to query: 0.55

5. **BOOK 12558:**
  - Title: The Whore of Akron: One Man's Search for the Soul of LeBron James
  - Author: Scott Raab
  - Rating: 3.8 out of 5 (based on 1097 ratings)
  - Similarity to query: 0.53


### Performance Optimization

- **Genre Pre-filtering**: Dramatically improves performance by computing similarities only within the selected genre
- **Caching Mechanism**: Stores computed embeddings by genre to avoid redundant calculations
- **Oversampling and Re-ranking**: Initially selects twice as many candidates as needed, then applies the ranking algorithm to find the best matches

### Handling Edge Cases

- **Low-rating but highly relevant books**: Can still rank well due to the weighted system
- **Popular but less relevant books**: Are appropriately downranked by the similarity component
- **Niche books with few ratings**: Can surface based on strong relevance and rating

## Gemini AI Integration

The project leverages Google's Gemini AI to transform raw book recommendation data into engaging, personalized summaries that enhance the user experience.

### Technical Integration

- **API Configuration**: Uses the Google Generative AI Python library to connect with the Gemini API
- **Model Selection**: Utilizes the `gemini-1.5-pro` model for its strong natural language capabilities
- **Environment Management**: Securely stores API credentials using `.env` file and dotenv library
- **Implementation Pattern**: Follows an async request-response pattern with proper error handling
- **Fallback Mechanism**: Gracefully degrades to basic formatting if the API is unavailable


### Data Preprocessing for Gemini

Before sending data to Gemini, the system:

1. **Cleans Text**: Removes encoding artifacts and normalizes characters
2. **Structures Input**: Formats book data into a consistent structure
3. **Contextualizes Information**: Includes genre and query context
4. **Optimizes Token Usage**: Balances detail with conciseness to stay within model limits

### Response Processing

The Gemini output is:

- **Parsed as Markdown**: Preserving formatting for display in the Gradio interface
- **Directly Rendered**: Without post-processing to maintain AI-generated structure
- **Quality Controlled**: Through careful prompt design rather than output filtering

### Summary Features

The AI-generated summaries include:

- **Thematic Titles**: Capturing the essence of the book collection
- **Contextual Introductions**: Providing background on the genre and query
- **Individual Book Analyses**: Highlighting why each book matches the query
- **Reading Recommendations**: Suggesting where to start in the collection
- **Consistent Voice**: Maintaining an enthusiastic, knowledgeable tone throughout

### Benefits of AI Summaries

- **Reduced Cognitive Load**: Presents information in an easily digestible format
- **Enhanced Context**: Provides background knowledge about the genre
- **Personalization**: Explicitly ties recommendations to the user's specific interests
- **Professional Presentation**: Creates a polished, cohesive reading experience

## User Interface

The project features a polished, user-friendly interface built with Gradio that guides users through the recommendation process:

### Interface Structure

- **Header Section**: Clear project title and description to orient users
- **"How it Works" Accordion**: Collapsible explanation of the system's functionality
- **Tabbed Navigation**: 
  - "Find Books" tab: Main recommendation functionality
  - "Explore Popular Genres" tab: Genre browsing and exploration

### Input Components

- **Genre Selection (Step 1)**:
  - Dropdown menu with all available book genres
  - "Apply Genre Filter" button to initiate genre filtering
  - Status indicator showing the number of books available in the selected genre
  
- **Query Input (Step 2)**:
  - Multi-line text box for natural language queries
  - Helpful placeholder text with example queries
  - Checkbox to toggle AI-powered formatting
  - Prominent "Get Recommendations" button

### Results Display

- **Recommendation Section**:
  - Clean, markdown-formatted presentation of book recommendations
  - When using Gemini AI: 
    - Custom title reflecting the query theme
    - Introductory paragraph with context
    - Individual book sections with title, author, rating, and description
    - Thoughtful conclusion with reading suggestions
  - Without Gemini: 
    - Simple formatted list with essential book information

### Additional Features

- **Example Queries**: Quick-access buttons for popular search types
- **Genre Explorer**: Categorized lists of genres with brief descriptions


### Technical Implementation

- **Event Handling**: 
  - Genre selection triggers embedding computation for that subset
  - Query submission processes user input and displays recommendations
  - Example buttons pre-fill the interface with sample scenarios
  
- **Caching System**:
  - Stores computed embeddings by genre
  - Avoids recomputation when a previously selected genre is chosen again
  - Significantly improves response time for genre-specific queries

- **Error Handling**:
  - Graceful fallback to basic formatting if Gemini API fails
  - Clear status messages when actions are needed
  - Input validation to ensure proper query flow

## Installation and Usage

### Prerequisites

- Python 3.8+
- pip
- Jupyter Notebook

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-book-recommendation.git
cd ai-book-recommendation
```
2. Install required packages:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file in the project root with your Google Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```
4. Download the Goodreads dataset:
```bash
# The dataset can be found at: https://github.com/zygmuntz/goodbooks-10k
# Place the CSV file in a 'Resources' folder in your project directory
```

### Running the Application

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open the main notebook:
```
Project3GoodReadsRecommendation.ipynb
```

3. Run all cells to initialize the models and launch the interface

4. When the Gradio interface appears:
   - Select a genre from the dropdown menu
   - Click "Apply Genre Filter" to pre-filter the data
   - Enter a description of the books you're interested in
   - Toggle the "Use AI for beautiful formatting" option if desired
   - Click "Get Recommendations"
   - Review your personalized book recommendations

### Using Example Queries

For a quick start, click any of the example buttons at the bottom of the interface:
- "Historical fiction about World War II"
- "Space exploration science fiction"
- "Female scientists biographies"
- "Victorian detective stories"

### Troubleshooting

- **API Key Issues**: Ensure your `.env` file is in the correct location and properly formatted
- **Memory Errors**: Try selecting genres with fewer books if you encounter memory limitations
- **Slow Performance**: The first query for each genre will be slower as embeddings are computed

### Requirements

```
pandas==1.5.3
numpy==1.26.0
sentence-transformers==2.5.1
scikit-learn==1.3.0
gradio==4.19.2
google-generativeai==0.3.1
python-dotenv==1.0.0
ipython==8.15.0
jupyter==1.0.0
```

## Project Structure

```
ai-book-recommendation/
├── Project3GoodReadsRecommendation.ipynb   # Main notebook with all code
├── .env                                     # Environment variables (API keys)
├── README.md                                # Project documentation
├── Resources/                              
│   └── GoodReads_100k_books.csv            # Dataset
└── requirements.txt                         # Dependencies
```

## Future Improvements

### Short-Term Enhancements

- **Fine-Tuned Ranking**: Adjust weights in the ranking algorithm based on user feedback
- **Expanded Genre Categories**: Add sub-genres and cross-genre recommendations
- **Book Cover Integration**: Display book covers using the Goodreads image URLs
- **Performance Optimizations**: Further caching and parallel processing improvements
- **Export Functionality**: Allow users to export recommendations as PDF or email

### Medium-Term Developments

- **User Profiles**: Save user preferences and history for personalized recommendations
- **Authentication System**: User accounts with saved recommendations and favorites
- **Advanced Filtering**: Add parameters for book length, reading level, publication date
- **Similar Books Feature**: "More like this" option for individual recommendations
- **Reading Lists**: Allow users to create themed collections from recommendations

### Long-Term Vision

- **Mobile Application**: Native app for iOS and Android
- **API Access**: Public API for integration with other services
- **External Integrations**:
  - Library availability checking
  - Bookstore/e-book price comparison
  - Goodreads/StoryGraph integration for automatic shelving
- **Community Features**: User reviews and recommendation sharing
- **Recommendation Explanations**: AI-generated explanations of why specific books were recommended

### Technical Improvements

- **Custom ML Model**: Train a specialized book recommendation model on broader dataset
- **Multi-Modal Capabilities**: Include book cover analysis in the recommendation process
- **Multilingual Support**: Expand to non-English books and queries
- **Distributed Computing**: Scale to larger datasets with distributed processing
- **A/B Testing Framework**: Test different recommendation algorithms and UI designs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Vassudeo Prabhudesai and Jill Balderson



---

*Created as a project for AI-powered content recommendation systems.*
