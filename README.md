# Greenwashing Detection in Financial Reports

## Overview
This repository contains the code and resources for a Natural Language Processing (NLP) pipeline designed to quantify "greenwashing" risks in the annual reports of major asset managers. This project was developed as part of the "Applications of Data Science: LLMs" course at WU Vienna and serves as the foundational code for a Bachelor Thesis.

## Project Goal
The objective is to move beyond ESG metrics and audit the *narrative layer* of financial reports. The aim is to detect whether sustainability claims are specific, measurable, and time-bound, or vague and hedging.

## Methodology: The Hybrid Scoring System
The project utilizes a hybrid approach combining rule-based linguistic analysis with a fine-tuned Large Language Model (LLM).

The system calculates three key metrics:
1. **VUI (Vagueness Index):** A rule-based metric normalizing the frequency of hedging terms (e.g., "aim," "strive," "intend").
2. **SPI (Specificity Index):** A combination of rule-based regex matching (detecting numbers, units, and dates) and a Fine-Tuned DistilRoBERTa model classified to distinguish "Specific" claims from "Vague" ones.
3. **GW (Greenwashing Risk) Score:** An aggregated score where a lower value indicates better, more specific reporting.

The aggregation formula used is:
GW = 0.56 * VUI_norm + 0.44 * (1 - SPI_hybrid)

## Data Strategy
### 1. Synthetic Training Data
Due to the scarcity of labeled greenwashing datasets, we generated a synthetic dataset using GPT-4o for the initial training phase.
- **Class 0 (Vague):** Claims containing hedging words without concrete targets.
- **Class 1 (Specific):** Claims containing concrete numbers, dates, and measurable units.

*Note regarding Bachelor Thesis:* While this repository currently uses synthetic data for model training, the final thesis work involves replacing/augmenting this with a dataset annotated by human validators via Prolific to ensure rigorous ground-truth validation.

### 2. Inference Data
The model is applied to real-world annual reports (2021-2024) from major asset managers, including BlackRock, Amundi, DWS, and KKR.

## Repository Structure
- `notebooks/`: Contains the three-stage pipeline (Data Generation -> Fine-Tuning -> Analysis).
- `src/`: Helper scripts for PDF parsing, text cleaning, and scoring logic.
- `models/`: Directory for the fine-tuned DistilRoBERTa model.
- `inputs/`: Raw PDF reports and synthetic training CSVs.
- `outputs/`: Generated scores and visualizations.

## Key Results
The analysis reveals significant variation in reporting quality across issuers. Below is an example of the temporal analysis performed in Notebook 3.

![Greenwashing Analysis](./outputs/greenwashing_analysis.png)

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Navigate to `notebooks/` and run the files in order:
   - `1_Data_Prep_Synthetic.ipynb`: Generates training data.
   - `2_Fine_Tuning_RoBERTa.ipynb`: Fine-tunes the classification model.
   - `3_Greenwashing_Analysis.ipynb`: Parses PDFs and calculates final scores.

## SFDR Comparison
This Notebook compares the Greenwashing Scores (from the previous analysis) to the actual percentage of sustainable assets this asset manager has under management. 
First it uses 2 data points that were researched and found from official sources.
Then it uses 29 more data points that were researched and estimated by ChatGPT.

## RAG Setup
This Notebooks loads the the text of the 31 PDFs (of the asset managers annual reports) and chunks it into smaller blocks. These Blocks are saved into a Vector Data Structure for easy retreival and comparison. In the next Step you can check specific statements, wether they are supported by text in the PDFs and where. 

## simple chatbot & chat rag
This is an unfinished chatbot integration, where you can chat with the Bot and ask it stuff based on the RAG.

   #### Simple Transformer Chatbot
   If you just want to talk to a small Hugging Face model locally, a lightweight console chatbot is included:

   1. Install dependencies (first run will download the model unless it is already cached):  
      `pip install -r requirements.txt`
   2. Start chatting (type `exit` to quit):  
      `python simple_chatbot.py --model-id microsoft/Phi-3-mini-4k-instruct`

   Optional flags: `--local-files-only` (stay offline)

## Authors
**Maxim Gomez Valverde** (GitHub: @builtbymaxim)
*Concept, Pipeline Development, Model Fine-Tuning, and Thesis Authorship.*
**Emanuel Zeder** 
*SFDR Comparison, RAG, chatbots*
