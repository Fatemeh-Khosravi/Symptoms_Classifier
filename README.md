# 🩺 Fatemeh Khosravi - Machine Learning Portfolio

## 👋 About Me

I'm a passionate Machine Learning Engineer and Data Scientist with expertise in healthcare AI applications. I specialize in developing intelligent systems that solve real-world problems using cutting-edge machine learning techniques.

## 🎯 Featured Project: AI Medical Symptoms Classifier

### Project Overview
An intelligent machine learning system that predicts diseases based on patient symptoms with **95.24% accuracy** across 41 medical conditions. This project demonstrates practical application of ML in healthcare, showcasing end-to-end development from data preprocessing to web application deployment.

### 🏆 Key Achievements
- **95.24% Accuracy** on test dataset (40/42 correct predictions)
- **41 Diseases** covered with single model
- **132 Symptoms** analyzed using TF-IDF vectorization
- **4,922 Training Samples** for robust model development
- **Real-time Predictions** via interactive web interface

### 🛠️ Technical Stack
- **Machine Learning**: scikit-learn, pandas, numpy
- **Web Framework**: Streamlit
- **Text Processing**: TF-IDF Vectorization
- **Model Persistence**: joblib
- **Data Visualization**: matplotlib, seaborn, plotly
- **Version Control**: Git

### 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 95.24% |
| **Test Cases** | 42 samples |
| **Correct Predictions** | 40/42 |
| **Disease Coverage** | 41 conditions |
| **Symptom Features** | 132 symptoms |

### 🏥 Medical Conditions Covered
- **Infectious Diseases**: Malaria, Tuberculosis, Hepatitis A/B/C/D/E
- **Cardiovascular**: Heart Attack, Hypertension, Varicose Veins
- **Endocrine**: Diabetes, Hypothyroidism, Hyperthyroidism
- **Gastrointestinal**: GERD, Peptic Ulcer, Gastroenteritis
- **Respiratory**: Bronchial Asthma, Pneumonia, Common Cold
- **Dermatological**: Fungal Infection, Acne, Psoriasis
- **And 25+ more conditions**

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn joblib plotly
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/Fatemeh-Khosravi/symptoms-classifier.git
cd symptoms-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run symptoms-classifier-main/app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
symptoms-classifier/
├── symptoms-classifier-main/
│   ├── app.py                 # Streamlit web application
│   ├── model.pkl             # Trained ML model
│   ├── vectorizer.pkl        # TF-IDF vectorizer
│   └── requirements.txt      # Python dependencies
├── Training.csv              # Training dataset (4,922 samples)
├── Testing.csv               # Test dataset (42 samples)
├── model_training.ipynb      # Jupyter notebook for model development
├── performance_analysis.py   # Model evaluation scripts
└── README.md                 # This file
```

## 🔬 Technical Implementation

### Data Preprocessing
- **Feature Engineering**: Converted binary symptom data to text format
- **Text Vectorization**: Applied TF-IDF to capture symptom importance
- **Data Balancing**: Handled class imbalance in medical dataset

### Model Development
- **Algorithm Selection**: Evaluated multiple ML algorithms
- **Hyperparameter Tuning**: Optimized model parameters
- **Cross-validation**: Ensured robust performance evaluation

### Web Application
- **User Interface**: Intuitive symptom selection interface
- **Real-time Processing**: Instant prediction generation
- **Error Handling**: Graceful handling of edge cases

## 📈 Model Performance Analysis

### Accuracy Breakdown
- **Overall Accuracy**: 95.24%
- **Precision**: High across most disease classes
- **Recall**: Excellent detection of rare conditions
- **F1-Score**: Balanced performance metrics

### Confusion Matrix Analysis
- **True Positives**: 40 correct predictions
- **False Negatives**: 2 misclassifications
- **Model Robustness**: Consistent performance across disease categories

## 🎯 Business Impact

### Healthcare Applications
- **Preliminary Screening**: Quick initial disease assessment
- **Resource Optimization**: Reduce unnecessary medical visits
- **Patient Education**: Help patients understand symptom-disease relationships
- **Telemedicine Support**: Assist remote healthcare consultations

### Technical Achievements
- **Scalability**: Handles 41 diseases with single model
- **Accuracy**: Comparable to medical AI benchmarks
- **Usability**: Intuitive interface for non-technical users
- **Deployability**: Production-ready web application

## 🔮 Future Enhancements

### Planned Features
- [ ] **Confidence Scoring**: Add prediction confidence levels
- [ ] **Symptom Severity**: Include symptom intensity levels
- [ ] **Medical Information**: Add disease descriptions and treatments
- [ ] **Multi-language Support**: International symptom descriptions
- [ ] **Mobile Application**: iOS/Android app development

### Advanced Capabilities
- [ ] **Integration with Medical APIs**: Connect to medical databases
- [ ] **Real-time Learning**: Continuous model improvement
- [ ] **Patient History**: Track symptom patterns over time
- [ ] **Doctor Consultation**: Direct connection to healthcare providers

## 🛠️ Skills & Technologies

### Programming Languages
- **Python** (Primary)
- **SQL**
- **HTML/CSS/JavaScript**

### Machine Learning & Data Science
- **scikit-learn**
- **pandas & numpy**
- **matplotlib & seaborn**
- **Jupyter Notebooks**
- **Feature Engineering**
- **Model Evaluation**

### Web Development
- **Streamlit**
- **Flask/Django**
- **RESTful APIs**
- **Git & GitHub**

### Tools & Platforms
- **Google Colab**
- **VS Code**
- **Docker**
- **AWS/Cloud Platforms**

## 📚 Education & Certifications

- **Degree**: [Your Degree]
- **Institution**: [Your University]
- **Focus**: Machine Learning, Data Science, Computer Science

## 🤝 Contact Information

- **Email**: fatemeh.khosravi@example.com
- **LinkedIn**: [Fatemeh Khosravi](https://linkedin.com/in/fatemeh-khosravi)
- **GitHub**: [@Fatemeh-Khosravi](https://github.com/Fatemeh-Khosravi)
- **Portfolio**: [Your Portfolio Website]

## 🏆 Awards & Recognition

- [Any awards, hackathons, or recognitions]
- [Academic achievements]
- [Professional certifications]

## 📄 Resume

[Download my resume](link-to-resume.pdf)

---

⭐ **Star this repository if you find it helpful!**

*This portfolio showcases my passion for applying machine learning to solve real-world healthcare challenges. I'm always open to new opportunities and collaborations in the field of AI and data science.* 
