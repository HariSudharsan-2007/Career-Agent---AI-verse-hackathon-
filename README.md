# AI Career Support Agent

## Overview
This is a local AI agent designed to automate career development tasks. It uses local Large Language Models (LLMs) to provide personalized resume analysis, interview preparation, and skill gap identification without sending data to external cloud providers.

## Problem Statement
Job seekers currently face three main technical and practical bottlenecks:
* **Information Overload:** Filtering generic advice for specific domains is inefficient.
* **Stateless Interactions:** Traditional tools do not retain user context (experience, skills) across sessions.
* **Unstructured Feedback:** Candidates rarely receive data-driven, actionable critiques on resumes or portfolios.

## Project Goal
To build an autonomous agent that retains user context and interacts with external data sources to deliver structured, actionable career insights (e.g., JSON-formatted resume feedback, localized job maps).

## Prerequisites

### 1. Python 3.12.5
**Strictly required.** This specific version is necessary to prevent dependency conflicts with the orchestration libraries.

### 2. Ollama
Required for running the LLM locally. Ensure the Ollama service is active in the background before execution.

## Model Configuration
**Selected Model:** `Llama 3.1 8B`

We chose this model based on specific performance benchmarks relevant to agentic workflows:

* **Tool-Use Reliability (BFCL Score: ~76.1):** It outperforms many larger models in the *Berkeley Function Calling Leaderboard*, ensuring reliable interaction with external APIs (Search, Maps, File Parsers).
* **Instruction Following (IFEval Score: ~0.80):** It demonstrates strict adherence to formatting rules. This is critical for generating clean, machine-readable outputs (like JSON) without hallucination.

**Setup Command:**
```bash
ollama pull llama3.1:8b
```


## Architecture & Tech Stack
The agent operates as a **State-Based Orchestrator** using **LangGraph**, strictly separating the system into three layers: Memory (Context), Tools (Capabilities), and Reasoning (LLM).

* **Core Logic:** Python 3.12.5
* **Orchestration:** LangChain & LangGraph (State machine management)
* **Memory:** ChromaDB (Vector database for long-term user context)
* **Interface:** Chainlit (Chat UI)
* **External Tools:**
    * **DuckDuckGo:** Live web search for real-time market data.
    * **Folium & Geopy:** Geospatial data processing and map visualization.
    * **PyPDF & Python-Docx:** Text extraction from binary document formats.

## Tool Breakdown & Testing
The project is modularized into standalone components. Each module includes a dedicated Jupyter Notebook for isolated testing before integration.

### 1. Memory Module (`memory_test.ipynb`)
* **Function:** Ingests raw conversation, extracts entities (Name, Skills, Goals), and stores them in **ChromaDB**.
* **Test:** Simulates a user introduction -> Verifies if a fresh agent instance can recall the user's name and skills without re-prompting.

### 2. Market Trend Search (`job_trend_search_test.ipynb`)
* **Function:** Queries **DuckDuckGo** for real-time salary and skill demand data (e.g., "React vs Angular demand").
* **Test:** Input a role -> Verifies the LLM returns a structured "Market Report" citing salary ranges and demand levels.

### 3. Resume Parser (`file_parser_test.ipynb`)
* **Function:** Converts `.pdf`, `.docx`, and `.txt` files into raw text strings for LLM analysis.
* **Test:** Load a sample resume -> Verifies correct extraction of "Key Skills" and "Experience" sections.

### 4. Location Mapper (`location_test.ipynb`)
* **Function:** Identifies city entities in prompts, fetches coordinates via **Geopy**, and renders an interactive HTML map using **Folium**.
* **Test:** Input "Jobs in Bangalore" -> Verifies generation of a valid `job_location_map.html` file.

### 5. Opportunities Finder (`opportunities_test.ipynb`)
* **Function:** A strict filter search that targets specific keywords ("Internship", "Certification") to reduce noise.
* **Test:** Query "AWS Certification" -> Verifies output contains direct links to valid courses/exams.

## Future Scope
* **LinkedIn Integration:** Browser automation (Selenium) for direct job applications.
* **Voice Mode:** Speech-to-Text integration (Whisper) for interactive mock interviews.
* **Multi-Agent Architecture:** Decoupling the "Resume Critic" and "Market Analyst" into separate, specialized agent loops.
