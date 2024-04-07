# Voice Verification Toolkit
The advanced analysis of voice data, encompassing voice cloning, embedding extraction, and the use of cosine similarity for voice verification. 
It enables the differentiation between original and cloned voices, as well as the verification of speaker identity, leveraging state-of-the-art machine learning techniques.
## Overview
The toolkit comprises three main components: voice cloning using a Text-to-Speech (TTS) API, extraction of voice embeddings, and analysis of these embeddings through cosine similarity. 
This multi-faceted approach facilitates a thorough examination of voice data, providing essential tools for speaker verification and the study of voice cloning technologies.
### Embedding Extraction
Voice embeddings are obtained by feeding voice recordings into the deep neural network, which outputs a vector in a high-dimensional space. Each dimension represents a learned feature of the voice, capturing unique characteristics of the speaker.
**Model:** We utilize NVIDIA's NeMo toolkit, specifically the **nvidia/speakerverification_en_titanet_large model**, for extracting voice embeddings. This pre-trained model is designed for speaker verification tasks and generates a high-dimensional vector representation for each voice sample.
### Cosine Similarity Analysis
Cosine similarity measures the cosine of the angle between two non-zero vectors in a multi-dimensional space, offering a metric for assessing the similarity between two voice embeddings. It is defined as follows:
