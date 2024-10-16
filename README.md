# AutoMate: Context-Aware Chatbot for Car Manuals

## Overview

**AutoMate** is a context-aware chatbot designed to integrate with car manuals, providing drivers with real-time guidance and support. By leveraging Large Language Models *(LLMs)* and Retrieval Augmented Generation *(RAG)*, AutoMate can effectively answer queries related to vehicle warning messages, their meanings, and recommended actions.
The chatbot aims to enhance the driving experience by delivering information in a user-friendly manner, with potential integration with text-to-speech software to read responses aloud.

## Project Description

As a proof of concept, we utilize pages from the car manual for the MG ZS. This manual contains crucial information about various car warning messages and their associated actions.

## Model Used

AutoMate employs *OpenAI GPT-3.5* for generating responses. The model has been fine-tuned to understand and provide context-aware answers based on the specific content of the car manual.

## Features

**Context-Aware Responses:** AutoMate retrieves relevant information from the car manual using RAG and generates concise answers based on user queries.
**Integration with Text-to-Speech:** Future enhancements will allow the chatbot's responses to be read aloud, improving accessibility for drivers.
**User-Friendly Interface:** The chatbot provides a simple interaction model for users to query warning messages and receive guidance.

## RAG Architecture

The architecture of Retrieval Augmented Generation (RAG) combines information retrieval with language generation. Below is a diagram that illustrates this architecture:

![image](https://github.com/user-attachments/assets/8e8d9635-f004-4748-9513-7ab75adc2243)

## Technologies Used

- **LangChain:** A framework for building applications with LLMs, facilitating easy retrieval and generation of context-aware responses.
- **OpenAI API:** Used for accessing the LLM to generate responses based on the retrieved information from the car manual.
- **Chroma Vectorstore:** A vector store for storing and retrieving documents related to the car manual.
- **Streamlit:** A framework for creating interactive web applications to facilitate user interaction with the chatbot.

