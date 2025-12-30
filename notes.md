# ML on Google Cloud

   The specialization provides an end-to-end learning path for building, deploying, and managing ML models on Google Cloud. It starts with an introduction to AI and ML, covering foundational concepts, Google Cloud AI tools, and the ML lifecycle. Next, learners build, train, and deploy models using Keras, gaining hands-on experience with deep learning, model training, and production deployment. The Feature Engineering course teaches how to process, transform, and optimize data for better model performance. Finally, Machine Learning in the Enterprise focuses on applying ML at scale, addressing challenges like model monitoring, MLOps, and integrating AI into business workflows. Overall, the specialization equips learners with practical skills to develop scalable, real-world ML solutions using Google Cloud.

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

A **pipeline** in machine learning (ML) refers to a series of automated, interconnected steps or processes that handle the workflow from **data processing** to **model deployment**. It's essentially a way of organizing the tasks involved in building, training, and deploying a machine learning model in a structured, repeatable, and automated manner.

Here's a breakdown of what a typical ML pipeline includes:

1. **Data Ingestion**: Collecting and importing data from various sources (e.g., databases, cloud storage).
2. **Data Preprocessing**: Cleaning, transforming, and preparing the data for training (e.g., handling missing values, scaling features).
3. **Model Training**: Training the machine learning model using the prepared data.
4. **Model Evaluation**: Assessing the model's performance using evaluation metrics (e.g., accuracy, precision).
5. **Model Deployment**: Deploying the trained model to a production environment (e.g., an API endpoint) so it can make real-time predictions.
6. **Monitoring and Retraining**: Continuously monitoring the model's performance in production and retraining it when needed.

In an **automated pipeline**, each of these steps happens in sequence, often without manual intervention. The pipeline can be automated for things like **continuous integration (CI)** and **continuous delivery (CD)**, meaning the model can automatically be retrained and redeployed based on new data or improved performance metrics.

#### Key Points:

* A **pipeline** organizes the entire ML process into a series of tasks or "components."
* It can be **automated**, allowing for **continuous** training, testing, and deployment.
* It ensures consistency and repeatability in your ML workflow.

In summary, an ML pipeline automates the steps from data ingestion to model deployment, making the ML process more efficient, scalable, and maintainable.

#### How Neural Networks Learn

Machine learning models—such as DNNs, CNNs, RNNs, and LLMs—are all built upon the foundation of the **artificial neural network (ANN)**. Understanding how an ANN learns explains how most ML models work.

An ANN consists of three main layers: **input**, **hidden**, and **output** layers. Each neuron is connected by weights, which store what the model learns during training.

#### Learning Process

1. **Weighted Sum**:
   Inputs are multiplied by weights and summed (often with a bias).

2. **Activation Function**:
   The weighted sum is passed through an activation function to introduce **non-linearity**, allowing the network to solve complex problems.

3. **Forward Propagation**:
   This process continues layer by layer until the output layer produces a **predicted value (ŷ)**.

4. **Activation Functions**:

   * **ReLU**: Outputs 0 for negative values, keeps positives.
   * **Sigmoid**: Outputs values between 0 and 1 (binary classification).
   * **Tanh**: Outputs values between −1 and 1.
   * **Softmax**: Produces a probability distribution for **multiclass classification**.

5. **Loss / Cost Function**:
   The prediction is compared to the actual value:

   * **MSE** for regression problems.
   * **Cross-entropy** for classification problems.

6. **Backpropagation**:
   If the error is large, the network adjusts weights and biases to reduce it.

7. **Gradient Descent**:
   Uses derivatives to decide:

   * **Direction** to adjust weights.
   * **Step size**, controlled by the **learning rate**.

8. **Iteration (Epochs)**:
   One full training cycle is an **epoch**. Training continues until the cost stops decreasing.

#### Key Concepts

* **Parameters**: Weights and biases learned during training.
* **Hyperparameters**: Learning rate, number of layers, neurons, activation functions, and epochs—set by humans before training.
* **Optimization**: The goal is to minimize the cost function by iteratively adjusting parameters.

Neural networks learn by repeatedly making predictions, measuring errors, and updating weights to improve performance—much like humans learning from feedback. These principles apply regardless of network size or complexity and form the foundation of modern machine learning.

---


## Build, Train and Deploy ML Models with Keras on Google Cloud

**Introduction to the TensorFlow Ecosystem**

TensorFlow is an open-source, high-performance library for numerical computation that uses directed graphs. **TensorFlow provides optimized tools and libraries** that make building, training, and deploying ML/deep learning models **faster, easier, and more scalable**.<br/><br/>

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

<ins>Tensors and variables in TensorFlow and how they are used in machine learning</ins>

**tf.constant** creates immutable tensors of various ranks—0D (scalars), 1D (vectors), 2D (matrices), and higher dimensions through stacking. You can slice tensors to access parts of data and reshape them to change their dimensions while keeping the data order. **tf.Variable** creates mutable tensors, useful for model weights that are updated during training. TensorFlow also supports automatic differentiation using **tf.GradientTape**, which records operations during a forward pass and computes gradients in reverse, enabling optimization. Custom gradients can be defined for more control in special cases. Overall, the lesson covers tensor creation, manipulation, and how TensorFlow handles variables and gradients for ML training.

**Design and Build an Input Data Pipeline**

**tf.data** is a **TensorFlow API for efficiently loading, preprocessing, and feeding data** into machine learning models during training.

**tf.data.Dataset** is a TensorFlow class that represents a collection of data elements (like images, text, or numerical data). It provides efficient ways to load, shuffle, transform, and batch data for training machine learning models.

The **tf.data API** in TensorFlow provides the **tf.data.Dataset** abstraction to handle large datasets efficiently by loading and processing data in batches. It supports creating datasets from various data sources, such as text files (CSV, TFRecord) or binary files, and enables operations like shuffling, mapping, and batching. This API allows for progressive loading of data, avoiding the need to keep the entire dataset in memory. Through operations like `TFRecordDataset`, datasets can be loaded, shuffled, processed, and batched before training. The dataset elements are iterated over using a loop mechanism, with the iterator resource being properly disposed of when the dataset is no longer needed to prevent memory issues.

Key steps for building efficient input pipelines in TensorFlow with **tf.data.Dataset**, especially when working with large datasets that don't fit into memory, are:

1. **Creating Datasets**: You can create a dataset from in-memory data using `from_tensor` or `from_tensor_slices`. The former returns a single dataset element, while the latter creates separate elements for each row of the input tensor.

2. **Loading Data**: For CSV files, `TextLineDataset` is used to load text data line-by-line, where each line is processed into a dictionary. You can apply transformations (like parsing CSV) via the `map` function.

3. **Shuffling, Batching, and Prefetching**: To improve training efficiency, you can shuffle the data (recommended for training data only), batch it into manageable chunks, and prefetch to allow parallel data preparation while the model is training.

4. **Working with Sharded Files**: For large datasets spread across multiple files, `list_files` can be used to gather filenames, which are then loaded into datasets. The `flat_map` function flattens multiple datasets into one for easier processing.

5. **Performance Optimization**: Prefetching and multi-threading allow efficient parallelization, ensuring the CPU and GPU are fully utilized during training, which improves performance by reducing idle times.

In short, tf.data.Dataset provides a flexible and efficient way to handle large datasets and create optimized input pipelines for training machine learning models.

In this section, we explore how to build a machine learning model to predict property prices, starting with selecting features like square footage (numeric) and property type (categorical). To handle categorical data, we use **feature columns**, such as **numeric_column** for numbers and **categorical_column_with_vocabulary_list** for strings, transforming them into a format suitable for neural networks, like one-hot encoding. We also discuss other feature column types, like **bucketized columns** for discretizing continuous data and **embedding columns** for handling large categorical data efficiently. For complex relationships, we can use **feature crosses** to combine features and help the model learn non-linear patterns. Finally, once features are prepared, we can train the model using **Keras** with **tf.data** for large datasets, utilizing a **dense_features** layer to input the features into the model for training (<ins>It’s common to use **tf.data** to build and manage the input pipeline (loading, preprocessing, batching, etc.), and then use **Keras** for building, training, and evaluating the model. **tf.data** handles the efficient data handling, while **Keras** takes care of model architecture and training, making them work together seamlessly</ins>).

While **tf.data** is responsible for efficiently loading, batching, and transforming data at the pipeline level, **Feature Columns** handle the transformation of individual features before they're fed into the model.

The pipeline would generally follow this order: **tf.data → Feature Columns → Keras**.

**Scaling data processing using Keras preprocessing layers and tf.data**

Data is scaled to build end-to-end models that can handle raw input data like images or structured data while also performing feature normalization and encoding. **Keras preprocessing layers** include tools for working with text, numerical features, categorical data, and images. For example, the **TextVectorization** layer converts raw text into token indices, the **Normalization** layer scales numerical features to a mean of 0 and a standard deviation of 1, and **StringLookup** or **IntegerLookup** layers encode categorical values. You can use the **adapt method** on these layers to compute necessary statistics, like the mean and variance, based on the training data.

There are two main ways to apply preprocessing layers: as part of the model's computation graph (which runs synchronously on the GPU) or asynchronously through the **tf.data** pipeline, which utilizes multiple threads on the CPU. The second method is especially useful for data augmentation and other asynchronous tasks. By including preprocessing in the model, you can export it as a fully self-contained solution, making it easier for inference deployment (e.g., in TensorFlow.js) without requiring users to handle preprocessing separately.

**Building Neural Networks with the TensorFlow and Keras API**

Keras is a neural network Application Programming Interface (API) for Python that is tightly integrated with TensorFlow, which is used to build machine learning models. Keras' models offer a simple, user-friendly way to define a neural network, which will then be built for you by TensorFlow.

Deep learning models are built by stacking layers of neurons, where each neuron applies a weighted sum to its inputs followed by a non-linear activation function. Linear layers alone, no matter how many are stacked, cannot increase the model’s expressive power—they collapse into a single linear transformation. Non-linear activation functions such as ReLU, sigmoid, tanh, and their variants are what allow deep neural networks to learn complex patterns. ReLU is especially popular due to its simplicity and fast training, though variants like Leaky ReLU, ELU, and GELU exist to mitigate issues such as the dying ReLU problem.

High-level APIs like **tf.keras** provide an accessible, modular, and flexible interface for defining, training, and evaluating deep learning models. Models can be built using the **Sequential API**, which is a simple way to build neural networks by stacking layers in a linear fashion, where each layer has exactly one input and one output. It’s ideal for models that don’t require complex architectures, like those with a single input and a single output. You define a **Sequential model** by adding layers one by one in the order they should process data. This makes it easy to use for straightforward problems, but it's limited when you need multiple inputs, outputs, or shared layers. Then, models can also be built using the **Functional API**, which supports more complex architectures with multiple inputs, outputs, or shared layers. Users can specify activation functions, compile models with optimizers and loss functions, and fit them to data. Optimizers such as SGD, Adam, and FTRL iteratively update the model’s weights to minimize the loss function. Deep networks can achieve more powerful learning but require care to avoid overfitting, using techniques like regularization, and their training progress can be monitored with metrics, callbacks, and visualization tools.

Once a model is trained and performing well, it can be saved and exported using the **SavedModel** format, which is portable, language-neutral, and compatible with TensorFlow Serving. Exported models can then be deployed to cloud platforms, such as Google Cloud AI Platform, where they can be versioned and served for scalable predictions. This allows client applications—web, mobile, or other code—to access the trained model for inference without needing the original in-memory model object. Overall, this workflow—from feature selection, model design, training, and deployment—enables deep learning models to move from experimentation to real-world applications.

<ins>Keras Functional API</ins>

Wide and Deep Learning is a model architecture that combines the strengths of both **memorization** (wide models) and **generalization** (deep models). It’s particularly useful for large-scale regression and classification problems with sparse, high-dimensional data, such as in recommendation systems or ranking tasks. The wide component captures simple linear relationships, while the deep component leverages a neural network to discover more complex patterns by decorrelating the input features. This approach mimics the human ability to both memorize specific instances and generalize to new situations, like understanding that most birds can fly, but with exceptions.

The **Functional API** in Keras offers flexibility by allowing the creation of complex models that may have multiple inputs, outputs, or shared layers. It supports non-linear architectures and enables reusing layers across different parts of the model, helping with data efficiency and reducing the need for large datasets. This approach is useful for tasks like multi-input/multi-output models or shared layers in models processing similar data (e.g., text with overlapping vocabulary). Although it offers strong debugging and validation features, the functional API has limitations when it comes to dynamic architectures like recursive networks.

Overall, Wide and Deep models, built using the Functional API, are powerful tools for scalable and flexible model architectures. They combine the simplicity of linear models with the complexity of deep learning, making them effective for handling diverse and large-scale machine learning problems. However, more complex tasks may require subclassing to go beyond the directed acyclic graph (DAG) structure that the Functional API uses: **Model Subclassing** offers complete flexibility, allowing full control over layers, custom arguments, and the training loop, making it ideal for advanced, research-focused tasks.

Model training aims to minimize loss, but when training loss keeps decreasing while test loss starts increasing, the model is overfitting—memorizing training data instead of generalizing. Early stopping can help, but it doesn’t address the root problem when the model itself is too complex. Regularization tackles this by penalizing model complexity, encouraging simpler models that generalize better. Visual tools like TensorFlow Playground show how overly complex features can create arbitrary decision boundaries and worsen test performance. This aligns with Occam’s razor: prefer simpler models with fewer assumptions. The key is to balance fitting the data well with keeping the model simple, controlled through a regularization term weighted by a tunable hyperparameter (lambda), which is data-dependent and must be carefully chosen.

Regularization includes many techniques that improve a model’s ability to generalize to unseen data, with L1 and L2 regularization being two common methods based on penalizing parameter magnitudes. Both measure model complexity using the norm of the weight vector: **L2 regularization** uses the Euclidean norm, encouraging weights to stay small and smooth (weight decay), while **L1 regularization** uses the sum of absolute values, often driving some weights exactly to zero. A tunable hyperparameter, **lambda**, controls the trade-off between minimizing training loss and keeping the model simple, and its optimal value is data-dependent. L2 helps prevent overly large weights, whereas L1 promotes sparsity and is widely used for feature selection by effectively removing less important features.

**Training at scale with Vertex AI**

Distributed TensorFlow on **Vertex AI** allows you to train models at scale by using cloud infrastructure: Distributed TensorFlow is a way to train machine learning models across multiple machines or devices simultaneously. It helps speed up training by splitting the workload, allowing large models and datasets to be processed more efficiently. Instead of running on a single machine, the work is distributed across many, which can significantly reduce training time. Before training, prepare and upload your data to **Google Cloud Storage**. The training code is typically split into `task.py` (handles job details) and `model.py` (core ML logic). For Vertex AI, package the code using `setup.py` and upload it to Google Cloud Storage.

When submitting a job, specify details like the region, job name, and Python package URIs. For distributed training, configure multiple worker pools (for multiple replicas). Use a prebuilt container or a custom one, and include parameters like machine type, number of replicas, and arguments like data path and batch size.

For efficient ML training, ensure the Cloud Storage bucket is in a single region. Monitor jobs via the **Google Cloud Console** and use **TensorBoard** to track performance and debug. Once the training is done, Vertex AI helps deploy the model for predictions via scalable, secure REST APIs for both real-time and batch inferences.

      A job is a task you send to the cloud to run, like training your machine learning model. It includes everything the system
      needs to know, such as your code, the data, and any settings or options for training the model.
      
      A prebuilt container is a ready-to-use package that has everything needed to run your model, like TensorFlow and
      other software. It saves you the trouble of setting everything up yourself, so you can just focus on training your
      model without worrying about the environment.


---


## Feature Engineering

**Vertex AI Feature Store**

Vertex AI Feature Store helps solve common feature management challenges by providing a centralized, fully managed repository for creating, storing, sharing, and serving machine learning features. It eliminates redundant feature development, reduces dependence on ops teams, enables low-latency online serving and high-throughput batch serving, and prevents training–serving skew by allowing features to be computed once and reused consistently for both training and prediction. Feature Store organizes data using entity types, entities, features, and time-stamped feature values, supports batch and streaming ingestion from BigQuery and Cloud Storage, and provides monitoring for feature quality and drift. By handling infrastructure, scaling, and lifecycle management, Vertex AI Feature Store allows data scientists, ML engineers, developers, and DevOps teams to collaborate efficiently and accelerate the development and deployment of ML applications.

**Raw Data to Features**

Feature engineering is the foundation of successful supervised machine learning. The goal is to transform raw, messy, and heterogeneous data into **numeric feature vectors** that make learning easier, faster, and more accurate for predictive models.

<ins>1. Purpose and Nature of Feature Engineering</ins>

Predictive models do not learn directly from raw data—they learn from **features**, which are carefully designed representations of that data. Feature engineering is the process of transforming, creating, selecting, and refining features to reduce modeling error and improve predictive power. It is a **manual, iterative, and domain-driven process**, often consuming the majority of time in an ML project. There is no universal recipe: different problems, even within the same domain, require different features.

Raw data may come from many sources and formats, and features can be numerical, categorical, bucketized, crossed, embedded, or hashed. Feature engineering often involves both **manual design** and **automatic methods** (e.g., PCA or learned embeddings), and models are typically improved by iteratively refining features after building a baseline model.

<ins>2. What Makes a Good Feature</ins>

Not all features are useful. A good feature must satisfy four key criteria:

1. **Relevance to the objective**
   Features must be plausibly related to the target, supported by a reasonable hypothesis. Including unrelated or arbitrary data leads to spurious correlations and poor generalization.

2. **Availability at prediction time**
   Features must be known when predictions are made. Using future data, delayed data, or unstable feature definitions causes data leakage and model failure in production. Training data must reflect the same data freshness and latency as real-time prediction data.

3. **Numeric representation with meaningful magnitude**
   Machine learning models operate on numbers. Features must be numeric, and the numeric relationships must make sense. Categorical variables require careful transformation (e.g., one-hot encoding or embeddings) rather than arbitrary numeric labels.

4. **Sufficient examples for each feature value**
   Each feature value should appear often enough in the dataset (rule of thumb: at least five examples). Rare or overly specific values must be grouped, discretized, or removed to avoid overfitting and misleading patterns.

<ins>3. How Features Are Represented in Practice</ins>

Once good features are identified, they must be represented correctly:

* **Real-valued features** (e.g., price, wait time) can be used directly.
* **Identifiers** (e.g., transaction IDs) should be excluded, as they provide no predictive signal.
* **Categorical features** (e.g., employee ID, item type) are converted into numeric form using:

  * **One-hot encoding (sparse columns)** when categories are known
  * **Vocabularies** built during preprocessing, which must remain consistent at prediction time
  * **Integralized features** for naturally indexed categories (e.g., hour of day)
  * **Hashed features** when categories are large or unknown ahead of time

Special care must be taken to handle **new or unseen categories**, and to ensure feature definitions do not change over time.

Missing values must also be handled explicitly—typically with **indicator columns** or dedicated “missing” categories—to avoid mixing real values with absent data.

<ins>Key Takeaway</ins>

Feature engineering is an **iterative, insight-driven process** that bridges raw data and machine learning models. By selecting features that are relevant, available at prediction time, numeric with meaningful magnitude, and well-supported by data—and by representing them consistently and carefully—you make the learning problem easier for the model and dramatically improve real-world performance.

**Feature Engineering**

<ins>1. Machine Learning vs. Statistics</ins>:

   * In **statistics**, missing values are often imputed (e.g., replaced with the average), and outliers are typically removed.
   * In **machine learning (ML)**, the approach is different. Missing data might lead to creating separate models for data with or without missing values. Outliers are considered useful, and you might even use them to train models.
   * ML can handle large datasets, so it can work with missing values or outliers by creating specific features or models to accommodate them. On the other hand, statistics focuses on maximizing the value from the data you already have.

<ins>2. Feature Representation</ins>:

   * In ML, **features** (input data for models) can be transformed. For example, instead of using latitude as a continuous number, you can split it into distinct categories (bins), like "LatitudeBin1", "LatitudeBin2", etc., to better represent the data.
   * You might also use techniques like **one-hot encoding** for categorical features (e.g., converting words into binary features) or **sparse feature embeddings** for high-cardinality categorical data.

<ins>3. Feature Construction</ins>:

   * **Feature construction** involves creating new features based on existing data using techniques like **polynomial expansion**, **feature crossing** (combining features), or even domain-specific logic (e.g., using business knowledge).

<ins>4. BigQuery ML</ins>:

   * **BigQuery ML** supports both **automatic** and **manual feature pre-processing**. This includes converting data types, handling missing values, and using SQL functions to manipulate data (e.g., extracting date parts, filtering out invalid records).
   * You can use BigQuery ML's **TRANSFORM clause** for custom pre-processing, and it supports operations like filtering out bogus data (e.g., rides with zero miles), extracting time-based features (like day of the week, hour, etc.), and converting categorical data with one-hot encoding.

<ins>5. New York Taxi Fare Prediction</ins>:

   * The module uses the **New York City taxi ride dataset** for a regression problem: predicting taxi fare prices.
   * The **baseline model** starts without feature engineering, and a **final model** with feature engineering aims to improve the model's predictions.
   * The evaluation metric for the model’s performance is **Root Mean Squared Error (RMSE)**, where lower values indicate better predictions. In this case, the baseline model's RMSE is used as a comparison point to gauge improvements.

<ins>6. Feature Engineering for Better Model Accuracy</ins>:

   * By applying feature engineering (like date extraction, filtering, and transforming categorical data), the **final model** can improve prediction accuracy, reducing RMSE and making more accurate fare predictions.

<ins>Key Takeaway</ins>

In summary, the module focuses on how ML, particularly in BigQuery, handles data preprocessing (like feature transformation and construction) and how it differs from traditional statistical approaches. It emphasizes how feature engineering can significantly improve model performance in practical applications like predicting taxi fares.

<ins>Some advanced feature engineering and preprocessing techniques in BigQuery ML</ins>: Feature crosses, bucketization, and the use of the `TRANSFORM` clause for efficient model training, prediction, and evaluation.

Here’s a summary of the key points:

1. **Feature Crosses**:

   * A feature cross combines different features (e.g., `hour of day` and `day of week`) to create new features, which is useful for capturing complex relationships in the data.
   * This technique works best with large datasets, where it becomes feasible to memorize patterns (e.g., learning the mean for every combination of features).
   * Feature crosses lead to sparse data (with many zeros) as only specific combinations of features are activated (e.g., `3 PM on Wednesday`).

2. **Sparsity in Models**:

   * Sparse models are easier to train and less prone to overfitting because they involve fewer features.
   * Sparse models are more interpretable as only the most meaningful features remain.
   * The sparse representation, though large in terms of features, helps in reducing complexity and improving model performance.

3. **Geospatial Functions**:

   * BigQuery ML includes geospatial functions like `ST_distance`, which helps with spatial data (e.g., calculating distances between latitude and longitude points).
   * These functions can be used for deriving new features like geographical distances for predictive models.

4. **Transforming Features**:

   * BigQuery ML allows you to define preprocessing steps during model creation using the `TRANSFORM` clause.
   * This clause applies feature transformations (e.g., normalization, encoding) automatically during model training and prediction, ensuring the preprocessing pipeline is consistent.
   * By using `TRANSFORM`, you can separate raw input data from transformed features and avoid the need to manually preprocess data during prediction.

5. **Bucketizing**:

   * The `BUCKETIZE` function is used to group continuous numerical features into discrete buckets (e.g., latitude and longitude values can be bucketized).
   * This is useful for creating categorical features from continuous ones, making them easier to handle in the model.

6. **Data Preprocessing Automation**:

   * When using the `TRANSFORM` clause, transformations like standardization for numeric features and one-hot encoding for categorical features are applied automatically.
   * This helps in automating the data preprocessing pipeline and ensures consistency across training, evaluation, and prediction phases.

7. **Time-Windowed Features**:

   * Some features, like the number of products sold in the last hour, require statistics over a time window.
   * To handle such time-based features, tools like Apache Beam and Google Cloud’s Dataflow can be used to process data in real-time or batch mode.

In essence, the passage highlights the importance of feature engineering (such as feature crosses and bucketization) and the advantages of automating preprocessing steps through BigQuery ML's `TRANSFORM` clause, allowing for consistent and efficient model training, evaluation, and prediction.

**Preprocessing and Feature Creation**

**Apache Beam** and **Google Cloud Dataflow** are powerful tools for building and running scalable data processing pipelines. Dataflow is a fully managed, serverless platform that allows developers to process large-scale data without managing infrastructure. It works by defining a pipeline in **Apache Beam**, where the data goes through a series of transformations (called **transforms**) on **PCollections** (the main data structure used in **Apache Beam**. They represent a **distributed collection of data** that can be processed in parallel across multiple machines. Think of them like a list or array, but designed to handle large amounts of data efficiently across a cluster). These pipelines can handle both **batch** and **streaming** data, and the output can be saved to various destinations like **Google Cloud Storage** or **BigQuery**.

The key benefit of using Dataflow is its ability to elastically scale based on the data size, automatically adjusting the compute resources as needed. The pipelines are executed using a **runner**, which can be run on Dataflow or other platforms like Apache Spark. Dataflow also handles output **sharding** to avoid file contention when writing results (meaning **Dataflow** splits the output into multiple smaller files, called **shards**, when writing the results). This prevents multiple machines from trying to write to the same file at the same time, avoiding errors or delays.

Overall, Apache Beam and Dataflow provide an efficient and flexible way to process and transform data at scale with minimal manual configuration.















