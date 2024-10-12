# Context-Aware Chatbot for Car Manuals

## Overview

This project aims to create a context-aware chatbot that integrates with car manuals to provide drivers with real-time guidance and support. By leveraging Large Language Models (LLMs) and Retrieval Augmented Generation (RAG), we will implement a system capable of answering queries related to vehicle warning messages, along with their meanings and recommended actions.

The chatbot is designed to enhance the driving experience by delivering information in a user-friendly manner, potentially integrating with text-to-speech software to read responses aloud.

## Project Description

As a proof of concept, we will utilize pages from the car manual for the MG ZS, a compact SUV, stored in the HTML file `mg-zs-warning-messages.html`. This manual contains crucial information about various car warning messages and their associated actions.

### Features

- **Context-Aware Responses:** The chatbot retrieves relevant information from the car manual and generates concise answers based on user queries.
- **Integration with Text-to-Speech:** Future enhancements will allow the chatbot's responses to be read aloud, improving accessibility for drivers.
- **User-Friendly Interface:** The chatbot will provide a simple interaction model for users to query warning messages and receive guidance.

## Technologies Used

- **LangChain:** A framework for building applications with LLMs, facilitating easy retrieval and generation of context-aware responses.
- **OpenAI API:** Used for accessing the LLM to generate responses based on the retrieved information from the car manual.
- **Chroma Vectorstore:** A vector store for storing and retrieving documents related to the car manual.

