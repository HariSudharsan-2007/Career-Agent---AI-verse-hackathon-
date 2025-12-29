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

