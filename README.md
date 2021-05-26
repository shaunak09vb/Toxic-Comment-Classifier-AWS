<h1 align="center">Toxic-Comment-Classifier-AWS</h1>

<p align="center">
  <img height="500" alt="logo" src="https://miro.medium.com/max/1575/0*v8WSU__4_zTAQg-t">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/release/baishalidutta/Comments-Toxicity-Detection?include_prereleases&sort=semver)](https://github.com/shaunak09vb/Toxic-Comment-Classifier-AWS/releases/)
![Python](https://img.shields.io/badge/python-v3.8.3+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/shaunak09vb/Toxic-Comment-Classifier-AWS/issues)


## Introduction
Online forums and social media platforms have provided individuals with the means to put forward their thoughts and freely express their opinion on various issues and incidents. In some cases, these online comments contain explicit language which may have an adverse effect on the readers. Comments containing explicit language can be classified into myriad categories such as Toxic, Severe Toxic, Obscene, Threat, Insult, and Identity Hate. The threat of abuse and harassment means that many people stop expressing themselves and give up on seeking different opinions.

To protect users from being exposed to offensive language on online forums or social media sites, companies have started flagging comments and blocking users who are found guilty of using unpleasant language. Several Machine Learning models have been developed and deployed to filter out the unruly language and protect internet users from becoming victims of online harassment and cyberbullying.

## Requirements

- Matplotlib>=3.3.3
- Keras>=2.4.3
- Gradio>=1.5.3
- Scipy==1.5.4
- Numpy>=1.19.5
- Pandas~=1.2.1
- Scikit-learn~=0.24.1
- Nltk~=3.5
- Spacy~=3.0.3
- Tensorflow~=2.4.1

## Installation

* Clone the repository 

`https://github.com/shaunak09vb/Toxic-Comment-Classifier-AWS.git`

* Install the required libraries

`pip3 install -r requirements.txt`

## Data Processing Steps

- Remove special characters present in between text
- Remove repeated characters
- Convert data to lower-case
- Remove numbers from the data
- Remove punctuation
- Remove whitespaces
- Remove spaces in between words
- Remove "\n"
- Remove emojis
- Remove non-english characters

## Usage

Locate the `source` directory and execute the following python files.

* To create the model, run:

`python3 model_training.py`

* You can also provide your own data:

`python3 model_training.py --data=csv_file_location`

You can also view the NLP_Deep_Learning.ipynb file present in the `notebooks` directory to understand the step-by-step approach undertaken for this project. 

## Website

To run the hosted website, access the `website` directory and execute:

`python3 website.py`

The website will start on your local server which can be viewed in your desired browser. You can type in any comment and find out what toxicity the model predicts.

## Blog Link

If you wish discover in detail, the steps taken by me for the implementation of the project. You can read my blog on <a href='https://towardsdatascience.com/toxic-comment-classification-using-lstm-and-deployment-using-aws-ec2-b84afe2b266b'>Medium</a>.

## License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under MIT License
