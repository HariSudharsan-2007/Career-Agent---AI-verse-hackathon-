# AI Career Support Agent

## Overview
This project focuses on building an intelligent AI Agent designed to support career development. In an increasingly complex job market, candidates often struggle with navigating career paths, optimizing resumes, and preparing for interviews. This agent aims to bridge that gap by providing personalized, data-driven, and conversational support using local Large Language Models (LLMs).

## Problem Statement
Navigating a career path today is overwhelming. Job seekers and professionals face several challenges, primarily information overload, where sifting through generic advice to find what applies to a specific domain or experience level becomes a bottleneck. Furthermore, traditional tools often lack the personalization required to give tailored feedback on resumes or portfolios. Finally, identifying exactly which skills are missing for a target role is often difficult without expert mentorship, leading to gaps in skill acquisition.

## Project Goal
To create an autonomous or semi-autonomous AI agent capable of understanding user context, analyzing career data, and providing actionable insights for career growth, interview preparation, and skill acquisition.

## Prerequisites
To run this project, you must have the following software installed and configured:

### 1. Python Version
Python 3.12.5 is strictly required to ensure compatibility with dependencies. Ensure this specific version is installed and added to your system path.

### 2. Ollama
This project uses Ollama to run Large Language Models locally. Download and install Ollama from the official website. Ensure the Ollama service is running in the background before executing the agent.

## Model Configuration
We selected Llama 3.1 8B for this agent because it strikes an optimal balance between local performance and agentic capabilities. According to the Berkeley Function Calling Leaderboard (BFCL), this model achieves a high reliability score (approx. 76.1), outperforming many larger models in tool-use reliability. This is a critical requirement for an agent that effectively interacts with external APIs or parses complex career data.

Additionally, on the Instruction Following Evaluation (IFEval), it demonstrates robust adherence to strict formatting constraints (Score ~0.80). This ensures the agent provides structured, actionable feedback—such as JSON-formatted resume critiques—without hallucinating formats or deviating from system instructions.

Before running the agent, you must pull the required Large Language Model using Ollama. Open your terminal or command prompt and run the following command:

```bash
ollama pull llama3.1:8b

Tool Breakdown & Testing Guide
This project includes standalone Jupyter Notebooks to test individual components before integrating them into the main application.

1. Memory & Context System (memory_test.ipynb / chromadb_test.ipynb)
What it does: Instead of a simple stateless chat, this module stores user facts (e.g., "I know Python", "I live in Chennai") in ChromaDB. It retrieves this context automatically during conversations to personalize answers.

How to Test:

Run memory_test.ipynb.

Simulation: The script simulates a conversation where a user introduces themselves.

Extraction: The LLM extracts entities (Name, Skills, Goals) and saves them to user_profile.json or ChromaDB.

Verification: A fresh agent instance is started to answer questions like "What is my name?" or "Do I know Python?" based only on the saved memory.

2. Market Trend & Job Search (job_trend_search_test.ipynb)
What it does: Uses the DuckDuckGo API to fetch real-time data about job markets, salary trends, and specific role requirements (e.g., "React vs Angular demand in India").

How to Test:

Run job_trend_search_test.ipynb.

Input: Provide a query like "Data Scientist salary India".

Process: The tool fetches 4-5 search results and feeds them into the LLM.

Output: The LLM synthesizes a "Market Report" summarizing salary ranges, demand levels (High/Low), and key skills required.

3. Resume & File Parsing (file_parser_test.ipynb)
What it does: Reads .pdf, .docx, and .txt files to extract raw text. The agent then analyzes this text to summarize skills or compare the resume against a job description.

How to Test:

Run file_parser_test.ipynb.

Setup: Place a sample resume (PDF/DOCX) in the file_samples/ folder.

Execution: Run the parser block.

Validation: Check if the output correctly lists "Key Skills" and "Experience" extracted from your document.

4. Location Mapping (location_test.ipynb)
What it does: Visualizes job locations. If a user asks about "Jobs in Bangalore", the agent identifies the city, fetches coordinates using Geopy, and generates an interactive HTML map using Folium.

How to Test:

Run location_test.ipynb.

Input: Enter a city name (e.g., "Chennai").

Output: The script will generate a file named job_location_map.html in your directory. Open this file in a browser to verify the map pin is accurate.

5. Opportunities Finder (opportunities_test.ipynb)
What it does: A specialized search tool that looks for "Internships", "Certifications", or "Courses" specifically. It filters out generic noise to find actionable links.

How to Test:

Run opportunities_test.ipynb.

Command: Call find_opportunities("AWS Cloud", "certification").

Result: The agent should return a list of top 3 certifications with direct links and key benefits.

Future Scope
LinkedIn Integration: Automate job application via Selenium/API.

Voice Mode: Add Speech-to-Text (Whisper) for mock interviews.

Multi-Agent System: Separate agents for "Resume Critique" vs "Market Analysis".