# 🚨 Credit Card Fraud Detection System

**A Production-Ready Machine Learning Pipeline for Real-Time Fraud Detection**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview

This project demonstrates a **complete end-to-end machine learning solution** for detecting fraudulent credit card transactions in real-time. Built with production-grade practices, it showcases expertise in data science, software engineering, and DevOps through a comprehensive pipeline that spans from exploratory data analysis to containerized deployment with live transaction simulation.

### 🏆 Key Achievements

- **79% Precision, 75% Recall** on highly imbalanced dataset (0.172% fraud rate)
- **Production-ready API** with FastAPI and Docker containerization
- **Real-time simulation** using socket programming for live transaction feeds
- **Comprehensive evaluation** with threshold optimization and comparative analysis
- **Modular architecture** with clean separation of concerns

---

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.8+** - Primary development language
- **TensorFlow/Keras** - Deep learning model implementation
- **Scikit-learn** - Traditional ML algorithms and preprocessing
- **FastAPI** - High-performance web API framework
- **Docker** - Containerization and deployment

### Data Science Libraries
- **Pandas/NumPy** - Data manipulation and numerical computing
- **Matplotlib/Seaborn** - Data visualization and analysis
- **Joblib** - Model serialization and persistence

### Infrastructure & Deployment
- **Docker** - Containerized deployment
- **Socket Programming** - Real-time transaction simulation
- **Uvicorn** - ASGI server for production deployment

---

## 📊 Dataset & Problem Statement

### Dataset Characteristics
- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions from European cardholders
- **Features**: 30 features (28 PCA-transformed + Time + Amount)
- **Class Imbalance**: 0.172% fraudulent transactions (492 out of 284,807)

### Business Challenge
In fraud detection, **missing a fraudulent transaction (False Negative) is significantly more costly** than incorrectly flagging a legitimate one (False Positive). This project prioritizes **recall optimization** while maintaining acceptable precision to minimize financial losses.

---

## 🏗️ Architecture & Implementation

### Project Structure
```
Credit-card-Fraud-Detection-Project/
├── 📁 src/                    # Source code modules
│   ├── app.py                # FastAPI application
│   ├── data_preprocessing.py  # Data pipeline
│   ├── model_development.py   # Model training
│   ├── evaluate.py           # Model evaluation
│   ├── predict.py            # Prediction utilities
│   ├── server.py             # Socket server
│   ├── client.py             # Socket client
│   └── utils.py              # Helper functions
├── 📁 notebooks/             # Jupyter notebook analysis
├── 📁 data/                  # Processed datasets
├── 📁 models/                # Trained models & scalers
├── 📁 APIs/                  # Kaggle API configuration
├── dockerfile               # Docker configuration
└── requirements.txt         # Dependencies
```

### Development Workflow

#### 1. **Exploratory Data Analysis** (`notebooks/`)
- Comprehensive data exploration and visualization
- Statistical analysis of class imbalance
- Feature distribution analysis
- Data quality assessment

#### 2. **Data Preprocessing** (`src/data_preprocessing.py`)
- **RobustScaler** for Amount (handles outliers effectively)
- **StandardScaler** for Time (normalization to [0,1] range)
- **Stratified splitting**: 75% train, 12.5% validation, 12.5% test
- Persistent data storage with Joblib

#### 3. **Model Development** (`src/model_development.py`)
- **Neural Network Architecture**:
  - Input Layer: 30 features
  - Hidden Layers: 2x Dense(128) with ReLU activation
  - Batch Normalization for training stability
  - Output: Sigmoid activation for binary classification
- **Training Configuration**:
  - Class weights: {0: 1, 1: 15} to handle imbalance
  - Early stopping with patience=3
  - Model checkpointing for best weights

#### 4. **Model Evaluation** (`src/evaluate.py`)
- **Threshold Optimization**: F-beta score maximization
- **Comprehensive Metrics**: Precision, Recall, F1-score, AUC
- **Visualization**: ROC curves, Precision-Recall curves, Confusion matrices
- **Comparative Analysis**: Neural Network vs Logistic Regression

#### 5. **Production API** (`src/app.py`)
- **FastAPI Application** with automatic documentation
- **Model Loading**: Efficient startup with lifespan management
- **Input Validation**: Pydantic models for type safety
- **Real-time Prediction**: Sub-second response times

#### 6. **Deployment & Simulation**
- **Docker Containerization**: Production-ready deployment
- **Socket Programming**: Real-time transaction simulation
- **Client-Server Architecture**: Simulates live transaction feeds

---

## 🎯 Model Performance

### Final Neural Network Results
| Metric | Validation Set | Test Set |
|--------|---------------|----------|
| **Precision** | 81.03% | 79.00% |
| **Recall** | 81.03% | 75.00% |
| **F1-Score** | 0.8103 | 0.7680 |
| **Optimal Threshold** | 0.57 | 0.57 |

### Model Comparison
| Model | Precision | Recall | F1-Score | Use Case |
|-------|-----------|--------|----------|----------|
| **Neural Network** | 81% | 81% | 0.81 | **Balanced performance** |
| Logistic Regression (Default) | 88% | 78% | 0.83 | High precision needed |
| Logistic Regression (Balanced) | 58% | 84% | 0.69 | High recall needed |

**Decision**: Neural Network selected for **balanced precision-recall trade-off**, making it ideal for production fraud detection systems.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized deployment)
- Kaggle API credentials (for dataset download)

### Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection-Project.git
cd Credit-Card-Fraud-Detection-Project
```

2. **Create Virtual Environment**
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Dataset**
```bash
# Place kaggle.json in ~/.kaggle/ directory
# Or manually place creditcard.csv in data/ directory
```

### Running the Pipeline

#### Option 1: Complete Pipeline
```bash
# 1. Preprocess data
python src/data_preprocessing.py

# 2. Train model
python src/model_development.py

# 3. Evaluate model
python src/evaluate.py

# 4. Start API server
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

#### Option 2: Docker Deployment
```bash
# Build and run container
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

#### Option 3: Real-time Simulation
```bash
# Terminal 1: Start API server
uvicorn src.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Start transaction simulator
python src/server.py

# Terminal 3: Start client for real-time processing
python src/client.py
```

### API Usage

#### Health Check
```bash
curl http://localhost:8000/
```

#### Fraud Detection
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Time": 10000,
       "V1": -1.359807134,
       "V2": -0.072781173,
       "V3": 2.536346738,
       "V4": 1.378155224,
       "V5": -0.338261018,
       "V6": 0.462387778,
       "V7": 0.239598554,
       "V8": 0.098697901,
       "V9": 0.36378697,
       "V10": 0.090794172,
       "V11": -0.551599533,
       "V12": -0.617800856,
       "V13": -0.991389847,
       "V14": -0.311169354,
       "V15": 1.468176972,
       "V16": -0.470400525,
       "V17": 0.207971242,
       "V18": 0.02579058,
       "V19": 0.40399296,
       "V20": 0.251412098,
       "V21": -0.018306778,
       "V22": 0.277837576,
       "V23": -0.11047391,
       "V24": 0.066928075,
       "V25": 0.128539358,
       "V26": -0.189114844,
       "V27": 0.133558377,
       "V28": -0.021053053,
       "Amount": 149.62
     }'
```

**Response:**
```json
{
  "probability": 0.0234,
  "threshold": 0.57,
  "prediction": 0
}
```

---

## 🔬 Technical Highlights

### Advanced Data Science Techniques
- **Class Imbalance Handling**: Multiple strategies including class weighting and threshold optimization
- **Feature Engineering**: Robust scaling for outlier-prone features
- **Model Selection**: Comprehensive comparison between traditional and deep learning approaches
- **Threshold Optimization**: F-beta score maximization for business-specific requirements

### Software Engineering Best Practices
- **Modular Architecture**: Clean separation of concerns across modules
- **Error Handling**: Comprehensive exception handling and logging
- **Type Safety**: Pydantic models for API input validation
- **Documentation**: Extensive docstrings and API documentation

### Production Readiness
- **Containerization**: Docker for consistent deployment environments
- **API Design**: RESTful API with automatic OpenAPI documentation
- **Real-time Processing**: Socket-based transaction simulation
- **Model Persistence**: Efficient model and scaler serialization

### Performance Optimization
- **Early Stopping**: Prevents overfitting during training
- **Batch Normalization**: Improves training stability and convergence
- **Efficient Scaling**: Optimized preprocessing pipeline
- **Memory Management**: Proper resource cleanup and management

---

## 📈 Business Impact

### Financial Benefits
- **Reduced False Negatives**: 75% recall means catching 3 out of 4 fraudulent transactions
- **Balanced Precision**: 79% precision minimizes false alarms and operational overhead
- **Real-time Processing**: Sub-second response times enable immediate transaction blocking

### Operational Advantages
- **Scalable Architecture**: Docker containerization enables horizontal scaling
- **Monitoring Ready**: Comprehensive logging and metrics for production monitoring
- **Easy Integration**: RESTful API design facilitates system integration

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

### Data Science & Machine Learning
- **Imbalanced Dataset Handling**: Advanced techniques for rare event detection
- **Model Evaluation**: Comprehensive metrics and visualization techniques
- **Deep Learning**: Neural network architecture design and optimization
- **Feature Engineering**: Domain-specific preprocessing strategies

### Software Engineering
- **API Development**: FastAPI for high-performance web services
- **Containerization**: Docker for deployment and scaling
- **Network Programming**: Socket programming for real-time systems
- **Code Organization**: Modular, maintainable codebase structure

### DevOps & Deployment
- **Production Deployment**: Containerized application deployment
- **Monitoring**: Health checks and performance metrics
- **Documentation**: Comprehensive project documentation
- **Version Control**: Git workflow and project organization

---

## 🔮 Future Enhancements

### Technical Improvements
- **Model Ensemble**: Combine multiple models for improved performance
- **Feature Selection**: Automated feature importance analysis
- **Hyperparameter Tuning**: Grid search or Bayesian optimization
- **Online Learning**: Incremental model updates with new data

### Infrastructure Enhancements
- **Database Integration**: Persistent storage for transaction history
- **Message Queues**: Kafka or RabbitMQ for high-throughput processing
- **Monitoring**: Prometheus/Grafana for production monitoring
- **Load Balancing**: Multiple API instances for high availability

### Business Features
- **Risk Scoring**: Continuous risk assessment beyond binary classification
- **Alert Management**: Configurable alerting and notification systems
- **Audit Trails**: Comprehensive logging for compliance requirements
- **A/B Testing**: Framework for model performance comparison

---

## 📞 Contact & Support

**Developer**: Pranav Chandrabhatla  
**Email**: pranavq50@gmail.com 
**LinkedIn**: linkedin.com/in/pranav-chandrabhatla-4b7673280/ 

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Kaggle** for providing the credit card fraud detection dataset
- **TensorFlow Team** for the excellent deep learning framework
- **FastAPI Team** for the high-performance web framework
- **Open Source Community** for the amazing tools and libraries

---

*This project showcases a complete machine learning pipeline from research to production, demonstrating both technical depth and practical implementation skills. The combination of advanced data science techniques, software engineering best practices, and production-ready deployment makes it an excellent example of modern ML engineering.*
