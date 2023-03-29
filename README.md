# Banzai-predicament
DS 4002 Projects

## Group information
### Rallying Cry: "Here it comes.... AHHHHHHHH"
### National Anthem: The Foggy Dew
### Patron animal: Walrus (googoogachoo)

## SRC
### Installing/Building our code
Our code can be installed by anyone with access to this repo and run on their computer with a few unique adjustments. Users will need to edit the file path's for the "bald" and "notbald" datasets in each of the python scripts for the classification models. The path's should lead to the "Bald" and "NotBald" folders within the "ProjectData" folder within the general "Data" folder. 

### Usage of our code
As mentioned, our code is comptaible with use on any device containing a python environment. Just clone the repo, open up your environment, open our code, change the file paths, and hit "Run". We allow anyone and everyone to use and copy our code.

### src Folder Files
knn_model: runs our k-Nearest-Neighbor model on the ProjectData

decision_tree_model: runs our decision tree model on the ProjectData

deep_image_model: runs our CNN model on the ProjectData; also plots Accuracy and Loss of our model

## Data
File name | Description 
--- | --- 
train |original Train split of data as downloaded directly from Kaggle; contains ~160,000 images split between a Bald and NotBald folder; 2% bald / 98% not bald
test | original Test split of data as downloaded directly from Kaggle; contains ~20,000 images split between a Bald and NotBald folder; 2% bald / 98% not bald
validation | original Validation split of data as downloaded directly from Kaggle; contains ~25,000 images split between a Bald and NotBald folder; 2% bald / 98% not bald
ProjectData | dataset we fabricated for our project use which is significantly smaller to account for our limited processing power; contains 2,000 images split between a Bald and NotBald folder; 50% bald / 50% not bald

Kaggle data from: https://www.kaggle.com/datasets/ashishjangra27/bald-classification-200k-images-celeba

## Figures
Figure name | Description and importance
--- | ---
CNN_Accuracy | line graph of CNN model accuracy by epoch
CNN_Loss | line graph of CNN model loss by epoch
CNN_model_results | confusion matrix and evaluation metrics for CNN model on test data
DT_model_results | confusion matrix and evaluation metrics for DT (decision tree) model on test data
knn_model_results | confusion matrix and evaluation metrics for knn model on test data
k_cross_val | line graph of knn model f1 score graphed against different k


## References
[1] “Convolutional Neural Network (CNN) | TensorFlow Core.” n.d. TensorFlow. Accessed March 29, 2023. https://www.tensorflow.org/tutorials/images/cnn.

[2] Hall, Christine. 2018. “Could Baldness Predict Other Health Risks?” TMC News (blog). June 6, 2018. https://www.tmc.edu/news/2018/06/could-baldness-predict-other-health-risks/.

[3] Gautam, Tanishq. 2020. “Create Your Own Image Classification Model Using Python and Keras.” Analytics Vidhya (blog). October 16, 2020. https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/.

[4] V, Nithyashree. 2022. “Image Classification Using Machine Learning.” Analytics Vidhya (blog). January 20, 2022. https://www.analyticsvidhya.com/blog/2022/01/image-classification-using-machine-learning/.

[5] “Psychology of Hair Loss Patients and Importance of Counseling - PMC.” n.d. Accessed March 29, 2023. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8719979/.

[6] Rosebrock, Adrian. 2021. “Your First Image Classifier: Using k-NN to Classify Images.” PyImageSearch (blog). April 17, 2021. https://pyimagesearch.com/2021/04/17/your-first-image-classifier-using-k-nn-to-classify-images/.
