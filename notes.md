## Introduction to AI and Machine Learning on Google Cloud

<ins>Artificial intelligence</ins>, or <ins>AI</ins>, is an umbrella term that includes anything related to computers mimicking human intelligence. Some examples of AI applications include robots and self-driving cars.

<ins>Machine learning</ins> is a subset of AI that allows computers to learn without being explicitly programmed. This is in contrast to traditional programming, where the computer is told explicitly what to do. Machine learning mainly includes supervised and unsupervised learning.

You might also hear the terms <ins>Deep Learning</ins> or <ins>Deep Neural Networks</ins>. This is a subset of Machine Learning that adds layers in between input data and output results to make a machine learn at more depth. You'll learn more about Neural Networks and Deep Learning later in this course.

Also:

- <ins>Predictive AI</ins>, forecasts future outcomes from historical data (e.g., predicting customer churn).

- <ins>Generative AI</ins>, which produces content and performs tasks based on requests. Generative AI relies on training extensive models like Large Language Models. These models are a type of Deep Learning model.

**AI Infrastructure**

Google Cloud’s AI infrastructure has three layers: **networking & security** (foundation), **compute & storage** (scalable processing and data storage, including TPUs and BigQuery), and **data & AI products** (tools like Pub/Sub, Dataflow, and Vertex AI to ingest, analyze, and act on data).

It lets organizations go from raw data to AI insights efficiently without managing hardware.

**AI models**

- <ins>Supervised learning</ins>: Classification, Regression

- <ins>Unsupervised learning</ins>: Clustering, Association, Dimensionality reduction.

<ins>**BigQuery ML Key Phases**</ins>:

        - Extract, transform, and load data into BigQuery.
        - Select and preprocess features.
        - Create the model inside BigQuery.
        - Evaluate the performance of the trained model.
        - Use the model to make predictions.
<br/>

**Generative AI**

Generative AI (GenAI) is a type of artificial intelligence that creates new, original content like text, images, music, code, or video by learning patterns from massive datasets, rather than just analyzing existing information. It works by predicting what comes next in a sequence, allowing it to produce human-like outputs in response to prompts, powering chatbots, content creation tools, and more, but also raising concerns about copyright and misuse.

<p align="center">
<img src="https://github.com/user-attachments/assets/34c34045-d71c-4cf6-b6ad-e81bfbda9e2c" />
</p>

There are three main approaches for developers to interact with foundational models:

        - UI
        - API
        - SDKs

**Vertex AI Studio** is a low-code/no-code platform that lets users—from non-technical business analysts to AI developers—rapidly prototype, customize, and deploy generative AI applications using foundation models like Gemini. Effective AI prompts include **task, context, and examples**, and can use techniques like zero-shot, few-shot, chain-of-thought, or retrieval-augmented generation. The platform also offers AI-assisted prompt creation, a prompt gallery, and support for multimodal inputs and outputs, enabling users to quickly turn ideas into working AI apps.

The second half of the **prompt-to-production lifecycle** in Vertex AI Studio covers **integration, deployment, monitoring, and optimization.** Users can generate code automatically, use SDKs or APIs, and deploy AI apps without worrying about cloud infrastructure. To ensure accurate, up-to-date outputs, models can be **grounded** with trusted data, often via **RAG**. Model quality can be further improved through **prompt design, parameter-efficient tuning**, or **full fine-tuning** using labeled datasets. Vertex AI Studio supports these tuning methods, allowing users to create, monitor, and deploy customized generative AI models for specific tasks, with both technical and low-code approaches.

**AI Agents**

AI agents extend foundation models by enabling them to take actions, access external systems, and automate multi-step workflows. Unlike standalone models, AI agents combine three core components—a **model** for reasoning and decision-making, **tools** for interacting with external applications and data, and an **orchestration layer** that manages actions and feedback loops—to achieve goal-oriented, autonomous behavior. Agents can retrieve information, validate decisions, and execute tasks like sending emails, while agentic AI coordinates multiple agents for more complex, end-to-end processes, making AI increasingly practical for real-world applications.

<p align="center">
<img src="https://github.com/user-attachments/assets/eb0928b4-8592-4089-b1a1-34c3e48b96a3" />
</p><br/><br/>

**AI Development Options**

<p align="center">
<img src="https://github.com/user-attachments/assets/87401b18-a0f0-4c27-9d6b-9182f6fda467" />
</p><br/><br/>

<p align="center">
<img src="https://github.com/user-attachments/assets/67f446e3-5101-4330-a5f1-c4cf9dd58bf8" />
</p><br/><br/>

<ins>Pre-trained APIs</ins>: Here you directly call Google's train models to solve your problems.

    - Speech, text and language APIs.
    - Image and video APIs.
    - Document and data APIs.
    - Conversational AI APIs.
    
<ins>Vertex AI</ins>: Unified platform that supports various technologies and tools on Google Cloud to help you build an ML project from end to end (it simplifies building, deploying, and managing machine learning models at scale).

<ins>AutoML</ins>: AutoML, which stands for Automated Machine Learning, aims to automate the process to develop and deploy an ML model.

**AutoML** in Vertex AI automates the end-to-end machine learning process, from data preparation to model deployment, without requiring code. AutoML works in four phases: (1) **data processing**, converting different data types for modeling; (2) **model search and parameter tuning**, using Neural Architecture Search to find optimal models and Transfer Learning to leverage pre-trained models for faster, more accurate results; (3) **model assembly**, combining top-performing models; and (4) **prediction**, generating outputs from the ensemble. By automating feature engineering, architecture search, hyperparameter tuning, and model ensembling, AutoML enables users to efficiently build high-performing ML models through a simple no-code interface.

<ins>Custom training</ins>: You do it yourself solution to build an ML project.

**AI Development Workflow**

There are three main stages to the ML workflow with Vertex AI.

<p align="center">
<img src="https://github.com/user-attachments/assets/ffd94702-b087-4bc9-a5a2-31a2e89808ec" />
</p>

        - Data preparation:
        
                Two steps: Data uploading and feature engineering.
                Data types: Streaming versus batch data, and structured versus unstructured data.
                
        - Data development: A model needs a tremedous amount of iterative training (Train the model, evaluate it,
        train the model some more).

        - Model serving: Deploy and monitor the model.

Compare this process to serving food in a restaurant. Data preparation is when you prepare the raw ingredients, model development is when you experiment with different recipes, and model serving is when you finalize the menu to serve the meal to customers.

**It's important to note that an ML workflow isn't linear, it's iterative.**

<p align="center">
<img src="https://github.com/user-attachments/assets/a3c27f24-4d45-48b3-9c6e-e9a61036249e" />
</p>

## Build, Train and Deploy ML Models with Keras on Google Cloud

**Introduction to the TensorFlow Ecosystem**

TensorFlow in an open-source, high-performance library for numerical computation that uses directed graphs. **TensorFlow provides optimized tools and libraries** that make building, training, and deploying ML/deep learning models **faster, easier, and more scalable**.<br/><br/>

<p align="center">
<img src="https://github.com/user-attachments/assets/c4acba15-4178-4196-a805-92451251a464" />
</p><br/><br/>

<p align="center">
<img src="https://github.com/user-attachments/assets/80eef629-e583-4ede-82e2-ab4d39c6d321" />
</p><br/><br/>

<ins>TensorFlow API Hierarchy</ins><br/><br/>

<p align="center">
<img src="https://github.com/user-attachments/assets/20e12123-9aaa-49ab-8b6c-cb8db3401eb9" />
</p>

**Design and Build an Input Data Pipeline**

**Building Neural Networks witht he TensorFlow and Keras API**

Keras is a neural network Application Programming Interface (API) for Python that is tightly integrated with TensorFlow, which is used to build machine learning models. Keras' models offer a simple, user-friendly way to define a neural network, which will then be built for you by TensorFlow.















