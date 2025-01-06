# Summary the Savior: Harmful Keyword-Query assisted Security-Aware-Summary-based Jailbreak Defense


![Project Banner](https://github.com/shrestho10/SummaryTheSavior/blob/main/for%20figures%20in%20paper/detailed_method.drawio%20(1).png)

---

[![Paper](https://img.shields.io/badge/Paper-Read-blue)](https://link-to-paper.com)
[![Hamrful Keyword Extractor]([https://img.shields.io/badge/1-Link-green)](https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer)
[![Security Aware Summarizer](https://img.shields.io/badge/2-Link-orange)](https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer)

---

## About the Project
Large Language Models (LLMs) are widely used due to their capabilities but face threats from jailbreak attacks, which exploit LLMs to generate illegal information and bypass their defense mechanisms. Existing defenses are specific to jailbreak attacks which necessitates a robust, attack-independent solution to address both Natural Language Processing (NLP) ambiguities and attack variability. In this study, we have introduced, Summary The Savior, a novel jailbreak detection mechanism leveraging harmful keywords and query-controlled security aware summary classification. By analyzing the improper contents of prompts within the summaries, the proposed method remains robust against attack diversity and NLP ambiguities. Two novel datasets for harmful keyword extraction and security aware summaries utilizing GPT-4 and Llama-3.1 70B respectively have been generated in this regard. Moreover, an "ambiguous harmful" class has been introduced to address content and intent ambiguities. Evaluation results demonstrate that Summary The Savior achieves superior defense performance, outperforming state-of-the-art defense mechanisms namely Perplexity Filtering, SmoothLLM, Non-Dictionary Removal with lowest attack success rates across various jailbreak attacks namely PAIR, GCG, JBC and Random Search, on Llama-2, Vicuna and GPT-4.
