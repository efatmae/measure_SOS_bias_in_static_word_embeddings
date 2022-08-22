Project Description:
=====================
This is the code for the COLING 2022 paper "SOS:Systematic Offensive stereotyping Bias in Word Embeddings". 


Experiments replication
========================
There are 4 main experiments in this paper. They can be found in the following folders:


1- Measuring SOS bias
======================
These are the code used to measure the SOS scores in each word embeddings which are used discussed in Section 3.1 in the paper and can be found in folder "measure_SOS_bias_in_WE" and "measure_SOS_bias_in_WE_swear_words2". The results and analysis discussed in Section 3.2 can be found in "measure_SOS_bias/results".

To run these sets of experiments, you'd need the swear words list found in folder "Data". The file "swear_words_list_2.txt" is the one used in the main experiments in the paper. While the file "swear_words.txt" is the one refereed to in the footnote on Page No.3 "We repeated the same experiment with a different set of 427 swear words from (Agrawal and Awekar, 2018) and also observed a significantly higher SOS bias scores for marginalised groups for 11 word embedding".


2- Measuring Social bias
========================
These are the code used to measure the social bias (gender and racial) scores which are used discussed in Section 3.3 in the paper and can be found in folder "measure_social_bias_in_WE".

3- SOS Bias Validation
======================
The code and the data used to validate the SOS bias scores described in Section 3.4 in the paper can be found in folder "Bias_scores_validation". This folder also contain the published statistics on hate speech as "Hate_stats.csv". 


4- Hate Speech Detection
==========================
The code used to train and test the hate speech detection can be found in folder "Model_training". To train the used models, you have to install all the packages in requirements.txt. Then run bash files Train_model.sh after specifying the model, dataset, word emebddings to trian. The trianing process saves the trained models in trained_models.

4.1 Data:
===========
The following datasets has to be downloaded first:

1- Twitter-sexism and Twitter-racism datasets from "Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter".
2- HateEval dataset from "SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter".
3- Twitter-Hate from "Automated Hate Speech Detection and the Problem of Offensive Language".


4.2 Correlation between Model performance and Bias scores
===========================================================
The code for this experiment can be found in "Model_performance_analysis".



All the measured bias scores are saved as csv files in "Bias_scores" folder.



Disclaimer
==========
While every care has been taken to ensure the accuracy of the data and code provided in this repository, the authors, the University of the West of Scotland, Oakland University, and Durham University do not provide any guaranties and disclaim all responsibility and all liability (including without limitation, liability in negligence) for all expenses, losses, damages (including indirect or consequential damage) and costs which you might incur as a result of the provided data being inaccurate or incomplete in any way and for any reason. 2021, University of the West of Scotland, Scotland, United Kingdom.





