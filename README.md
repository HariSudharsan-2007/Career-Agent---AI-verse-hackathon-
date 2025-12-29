# AI Career Support Agent

## Overview
This project focuses on building an intelligent AI Agent designed to support career development. In an increasingly complex job market, candidates often struggle with navigating career paths, optimizing resumes, and preparing for interviews. This agent aims to bridge that gap by providing personalized, data-driven, and conversational support using local Large Language Models (LLMs).

## Problem Statement
Navigating a career path today is overwhelming. Job seekers and professionals face several challenges:
* **Information Overload:** Sifting through generic advice to find what applies to a specific domain or experience level is a bottleneck.
* **Lack of Personalization:** Traditional tools often lack the ability to give tailored feedback on resumes or portfolios based on a user's unique history.
* **Skill Gap Identification:** Identifying exactly which skills are missing for a target role is often difficult without expert mentorship.

## Project Goal
To create an autonomous or semi-autonomous AI agent capable of understanding user context, analyzing career data, and providing actionable insights for career growth, interview preparation, and skill acquisition.

## Prerequisites
To run this project, you must have the following software installed and configured:

### 1. Python Version
**Python 3.12.5** is strictly required to ensure compatibility with dependencies (especially `langgraph` and `chromadb`).

### 2. Ollama Setup
This project uses **Ollama** to run the inference engine locally.
1. Download and install Ollama from the official website.
2. Pull the specific model used in configuration:
   ```bash
   ollama pull llama3.1:8b
