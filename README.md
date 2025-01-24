# Vector Image-to-Image Search Using MongoDB Atlas & Node.js

This project demonstrates how to build an image-to-image search application using **MongoDB Atlas** with vector search functionality and **Node.js**. The project uses MongoDB’s new vector search feature to compare image embeddings for efficient similarity-based search.

## Description

The application allows you to search for images similar to a query image by embedding both images and comparing their embeddings using MongoDB Atlas’ vector search. The images are processed using an image embedding model like OpenAI’s CLIP, and then MongoDB Atlas is used to perform a similarity search on the embedded vectors.

## Features
- **Image Embedding**: Convert images to vector embeddings using a pre-trained model like OpenAI CLIP.
- **Vector Search**: Perform similarity-based image search using MongoDB Atlas' vector search feature.
- **Node.js Backend**: Handle requests and interact with MongoDB Atlas using Node.js.

## Technologies Used
- **Node.js**: Backend for handling requests.
- **MongoDB Atlas**: Cloud database with vector search support.
- **OpenAI CLIP (or another embedding model)**: Used for generating image embeddings.
- **Express.js**: Web framework for handling API requests.

## Prerequisites

Before running the application, ensure you have the following installed:
- Node.js (>= 14.x)
- MongoDB Atlas account
- API key or credentials for accessing MongoDB Atlas

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/vector-image-search.git
   cd vector-image-search
