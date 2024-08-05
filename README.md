# Master-Thesis-Transparency_MedTranscription_BERT_XAI
## Overview

The proposed master's thesis aims to enhance the transparency and interpretability of BERT in medical transcription. It will investigate the performance of BERT when fine-tuned on medical-specific data and apply Explainable AI (XAI) techniques to elucidate its decision-making process. The goal is to improve the accuracy and reliability of medical transcription systems, thereby aiding clinical decision-making.

## Research Field

This thesis aims to assess the effectiveness of BERT-based models in medical transcription and investigate XAI methods to increase the model's interpretability. The research objectives aim to uncover the model's decision-making process, which is crucial for fostering trust in healthcare applications. Implementing BERT, along with XAI, aims to develop a better transcription system that could improve clinical decisions and, consequently, patients' outcomes.

## Research Problem

The research problem focuses on the opaqueness of deep learning models used in medical document transcription. Because of the nature of applications in the healthcare industry, it is necessary to know how these models arrive at decisions to improve dependability and safety. Current models’ lack of transparency is worrisome when it comes to applications in clinical settings. This work aims to identify the components that influence model predictions, thereby enhancing the model's reliability and transparency in the medical field.

## Research Objectives

This study seeks to compare the performance of fine-tuned BERT models for medical transcription data, aiming at providing the best results. It also aims to examine how BERT models make decisions using SHAP XAI for explanation and determine which pre-trained BERT model gives the best result. Furthermore, it uses word importance metrics to analyze models' systematic errors and identify risks that may threaten patient safety in clinical environments.

## Literature Review Summary

Medical texts are transcribed. It holds significant importance in the field of health as it aids in the communication and handling of health-related concerns. People have used AI-based methods like BERT and its variants, such as BioBERT and ClinicalBERT, but interpretability typically presents a challenge. The integration of such XAI approaches as SHAP is still under construction, and comparable studies across medical fields are lacking. More free-text clinical notes and less annotated datasets have some issues, which need domain-specific pre-processing and more clinical experts’ help for AI model enhancement and usability.

## Research Gap

Despite the evident progress in using BERT-based models in medical text transcription, there is still no comprehensive comparative analysis of their performance. The strategies used in modern studies are not very transparent, which makes it critical for clinicians to adopt the methods. Currently, the implementation of XAI methods such as SHAP is quite limited, and the comparison between traditional and transformer-based models on standardized datasets is not available. Moreover, due to the unstructured nature of clinical notes and the scarcity of annotated datasets, there is a need to improve the preprocessing and data augmentation steps.

## Research Methodology

This study focuses on the analysis of BERT-based language models to classify medical text transcription in all subject areas, solving essential problems from previous research. Because of this, the study aims to make clinical applications more accurate and easier to understand by systematically comparing different versions of BERT and using XAI techniques. The study's goal is to show that AI can reliably and ethically sort medical transcriptions into groups. This will be done through thorough data preparation, exploratory analysis, and rigorous model evaluation.

## Dataset 
This dataset has 4999 rows of records and six columns (‘Unnamed: 0,’ ‘description,’ ‘medical_specialty,’ ‘sample_name,’ ‘transcription,’ and ‘keywords’), as shown below.

![Dataset](https://github.com/Kaushik-Puttaswamy/Master-Thesis-Transparency_MedTranscription_BERT_XAI/blob/main/Dataset%20.png?raw=true)


## Research Design 

![Research Design](https://github.com/Kaushik-Puttaswamy/Master-Thesis-Transparency_MedTranscription_BERT_XAI/raw/main/Research%20Design.png)

This research design follows a reliable and comprehensive approach to assess BERT models in medical text transcription. First, it involves the collection and preprocessing of data, where it is made sure that the data collected is appropriate for medical use. Thus, through the combination of advanced model selection, hyperparameters, and XAI approaches, the study will strive to improve medical AI accuracy, transparency, and replicatability.

## Data Preprocessing

![Preprocessing Flow Chart](https://github.com/Kaushik-Puttaswamy/Master-Thesis-Transparency_MedTranscription_BERT_XAI/raw/main/Preprocessing%20Flow%20Chart.png)

Preprocessing text data is important in NLP, particularly in medical transcription. This process includes steps such as cleaning of raw text, normalizing, tokenizing and lemmatizing of data to help get data in the proper format for machine learning models. This helps in maintaining the quality of data and in minimizing noise and increasing relevance using the domain knowledge necessary for data analysis and modeling.

## Model Selection

In ML and AI, choosing models is important when making analyses and predictions. Our approach employs some of the conventional machine learning algorithms, such as Random Forest, SVM, XGBoost, and Logistic Regression, which are among the most effective techniques for handling high-order data relationships and data operating in high-dimensional space. Also, we utilize contemporary deep learning models such as BERT, BioBERT, Clinical BERT, and RoBERTa; these models are effective in identifying subtle patterns across large data sets, especially in fields like natural language processing and clinical research. This approach to selecting the model enables us to incorporate different aspects of data for proper classification and decision-making.

## Baseline Model Development

![Baseline Model Development](https://github.com/Kaushik-Puttaswamy/Master-Thesis-Transparency_MedTranscription_BERT_XAI/raw/main/Baseline%20Model%20Development.png)

In our baseline model development, starting with word representation is crucial, transforming text inputs into vector forms for algorithmic comprehension and generalization. For medical text categorization, we employed three prominent word representation models, detailed in Figure.

## Avance Model Development

![Advance Model Development](https://github.com/Kaushik-Puttaswamy/Master-Thesis-Transparency_MedTranscription_BERT_XAI/raw/main/Advance%20Model%20Development.png)

After developing the baseline model, we must now develop the advanced model, as illustrated in the above Figure

![Proposed Methodology for BERT as Advanced Model with XAI Technique for Decision Making](https://github.com/Kaushik-Puttaswamy/Master-Thesis-Transparency_MedTranscription_BERT_XAI/raw/main/Proposed%20Methodology%20for%20BERT%20as%20Advanced%20Model%20with%20XAI%20Technique%20for%20Decision%20Making.png)

The figure demonstrates the advanced model development approach, utilizing medical transcripts to train a model that classifies medical specialties. It uses color-coded outputs to indicate the important terms, thus improving the decision-making process and the confidence of using machine learning in medical text analysis.

## BERT Models for Sequence Classification

![Fine-tuning BERT Model for Medical Text Classification](https://github.com/Kaushik-Puttaswamy/Master-Thesis-Transparency_MedTranscription_BERT_XAI/blob/main/Fine-tuning%20BERT%20Model%20for%20Medical%20Text%20Classification.png?raw=true)


Figure shows the integration of a fine-tuned BERT model for sequence classification, especially for the medical text categorization. We learned special tokens like [CLS] and [SEP] while using BERT for tasks like 'next sentence prediction' and 'masked language modeling.' Because of the fine-tuning, it is possible in medical applications to better classify the given medical transcripts into the respective medical specialties, improving both context awareness and prediction.

## Result Comparision between Baseline and Advanced Model 

![Result Comparison between Baseline and Advanced Model](https://github.com/Kaushik-Puttaswamy/Master-Thesis-Transparency_MedTranscription_BERT_XAI/blob/main/Result%20Comparision%20between%20Baseline%20and%20Advanced%20Model%20.png?raw=true)


In comparing medical specialty models, BioBERT, ClinicalBERT, BERT, and RoBERTa achieved weighted average F1 scores of 0.93, 0.92, 0.90, and 0.92 respectively, surpassing previous non-transformer-based models (0.77 to 0.82). This underscores the superior accuracy of BERT models over traditional ML methods in medical classification tasks.

## Integration of XAI with BERT Model

![SHAP Values Visualization](https://github.com/Kaushik-Puttaswamy/Master-Thesis-Transparency_MedTranscription_BERT_XAI/raw/main/SHAP%20Values%20%20Visualization.png)

Integration of XAI with BioBERT helps to improve the interpretability of the medical specialty classification by using SHAP values to explain the model’s decisions. This approach visualizes token contributions, which can aid in the analysis of critical decision-making processes for healthcare applications. Because SHAP is consistent and easy to understand across a range of inputs, it is clear that SHAP explanations make the models more reliable.

## SHAP Integration and Robustness Testing with BioBERT

Combining SHAP values with BioBERT makes the interpretation of the medical specialty prediction more clear and supports explaining important terms such as "ventricular" or "carpal tunnel" in cardiovascular and other cases. Robustness testing explains the model's ability to return the correct results by removing essential terms and adding noisy data. This capability mimics the nature of the real-world input that BioBERT is bound to encounter, making it reliable in healthcare applications.

## Error Analysis with BioBERT and XAI in Medical Specialty Classification

SHAP-based error analysis identifies the precise terms that influence the incorrect classification of medical specialties in BioBERT and suggests ways to improve contextual training data and feature engineering for future healthcare-related tasks to enhance accuracy.

## Validity and Reliability of Research Methodology, Ethical Considerations, and Limitations
The research methodology ensures validity and reliability by selecting multiple sources and preprocessing data, giving careful consideration to data collection and exploratory data analysis. The proposed data handling measures, such as text normalization, tokenization, and model validation using methods including SHAP values and cross-validation, help to achieve accurate results. Concerns about ethical implications include protecting patients' identities, proper disclosure of data usage, and model explainability. We must consider limitations such as small sample size, data biases, anonymization issues, and issues with normalizing text and understanding contexts to enhance the accuracy and ethicality of the studies.

## Research Findings and Interpretation
Superior pre-processing, sophisticated machine learning algorithms, and the use of XAI have all significantly accelerated the classification process. We found that transformer-based models like BioBERT are more effective than traditional models, underscoring the importance of developing more specific architectures. specific architecture.   SHAP values offer excellent interpretability over the features, contributing to the model’s acceptance in clinical setups. Such developments enhance the effectiveness of AI and its applications in the healthcare field, which leads to improvements in the overall results and the field of medical informatics.

## Summary and Future Directions
This research assessed the efficiency and interpretability of BERT-based models for medical text transcription and applied XAI tools such as SHAP. Traditional methods were less accurate and ineffective in classification as compared to BERT models, particularly BioBERT. Feature-important insights provided by SHAP enhanced model interpretability and, thus, the doctors’ confidence in the models. Exploring the combination of multiple XAI techniques, expanding the number and variety of datasets, and conducting clinical studies are necessary to evaluate the generalization and adaptation of the model in healthcare organizations. We should focus on the following areas to bolster AI's role in medical text analysis and enhance the efficacy of clinical decisions.

### Note:

In the repository, you'll find two separate folders related to our project:

Python code - ML model with XAI (ipynb file):

Contains the Medical Text Transcription Classification ML model with XAI Integration.ipynb file. This Jupyter Notebook file houses the Python code for our machine learning model with XAI. GitHub cannot display this file directly due to its size (50 MB). You can download the notebook, and to view the code and outputs, access it in a Google Colab notebook.

Python code - ML model with XAI (PDF file):

Contains the Medical Text Transcription Classification ML model with XAI Integration.pdf file. This PDF version of the notebook provides another way to review the code and results conveniently.

### Contact:

Author: https://www.linkedin.com/in/kaushik-puttaswamy-317475148
