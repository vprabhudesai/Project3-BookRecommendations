# Project 3 - Book Recommendation System

A similarity-based book recommendation system using sentence transformers and the Goodreads database.

## Overview

This project creates an interactive book recommendation system that suggests books based on user preferences. By leveraging natural language processing and semantic similarity, the system can understand and match user requests like "sci-fi with complex characters" or "books similar to The Hunger Games" with appropriate titles from a database of 100,000 books.

## Features

- **Natural Language Queries**: Enter preferences in plain English
- **Semantic Understanding**: Captures the meaning behind requests rather than just keywords
- **Interactive Interface**: Simple UI built in ***** for easy interaction
- **Flexible Recommendations**: Request additional suggestions if initial recommendations don't match preferences

## Data

The system uses a preprocessed dataset derived from the Goodreads database containing 100,000 book titles with the following information:
- Title
- Author
- Description
- Genre
- Pages
- Rating
- Number of ratings

## Data Pre-processing

Before implementing our book recommendation system, we performed several crucial data preparation steps:

1. **Data Cleaning**
   - Removed unnecessary columns to focus only on relevant features (title, author, description, genre, pages, rating, number of ratings)
   - Eliminated rows with null values in critical columns (specifically genre and description) to ensure quality recommendations

2. **Genre Processing**
   - Transformed genre data from string format (e.g., "Fantasy, Young Adult, Romance") into structured lists of discrete genres
   - This conversion enables more precise similarity matching and filtering based on specific genre preferences
   - Created a consistent genre vocabulary across the dataset

3. **Quality Assurance**
   - Verified data integrity after transformations
   - Ensured all remaining entries contained sufficient information for meaningful recommendations

This pre-processing pipeline resulted in a clean, structured dataset ready for embedding generation and similarity search implementation.

## Technical Architecture

[TBD]

## Installation

[TBD]

## Usage

[TBD]

## Model Details

[TBD]

## Performance Evaluation

[TBD]

## Future Improvements

[TBD]

## Contributors

Balderson and Prabhudesai

## License

MIT
