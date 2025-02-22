# relation-extraction
HW1 Report

Yifei Gan

1 Introduction

The task in this study is to train deep learning models to identify core knowledge graph relations from user utterances in a conversational system. Specifically, this is a supervised learning problem where the goal is to classify key relations related to movies (e.g., actors, ratings, directors) invoked in user queries. The dataset consists of two parts: a training set and a test set. The training dataset includes both the input (user utterances) and out- put (corresponding core relations). This dataset comprises 2,312 entries, with each entry represent- ing an utterance-relation pair. There are no null or missing values in this dataset.(Showed in Table 1). The testing dataset include only the input utter- ances, and requiring program to write the output core relations. It has 980 entries with no missing values.(Showed in Table 2).

2 Models

In this work, I experimented with various word embeddings, including GloVe, FastText, and Word2Vec, to see their effect on model per- formance. The specific embeddings used are: glove-wiki-gigaword-50, word2vec-google-news- 300, and fasttext-wiki-news-subwords-300. I was trying all the large sized embedding and figure out whether they can affect my model accuracy. Un- fortunately, none of them increased the accuracy of my model when doing model evaluations. Take glove-wiki-gigaword-50 as an example. Using the same epochs without embedding, the accuracy cal- culated is around 95%. However, using this epochs

ID UTTERANCE![](Aspose.Words.b485898c-3b19-4923-bc41-8d299d998ed9.001.png)

0 star of thor

1  who is in the movie the campaign
1  list the cast of the movie the campaign ![](Aspose.Words.b485898c-3b19-4923-bc41-8d299d998ed9.002.png)Table 1: Example of testing dataset.

with embedding, it can only reach to 55%.

Model 1: Baseline Model (No Embedding) Archi- tecture: Layers: 256 Activation Function: ReLU Optimizer: AdamLossFunction: CrossentropyHy- perparameters: Learning Rate: 0.001 Epochs:1000 Batch Size: 32 Results: This model served as a control and achieved an accuracy of 95% on the training set and 92% on the validation set.

Model 2: GloVe Embedding Implementation: This model was similar to the baseline but with the addi- tion of the GloVe-Wiki-Gigaword-50 embeddings, where each word in the input utterance was con- verted to a 50-dimensional vector. To some reason, the epochs I need to train the model to increase its accuracy rate increased when I use embeddings. Architecture: Embedding Layer: GloVe (frozen weights) Layers: 256 Activation Function: ReLU Optimizer: Adam Loss Function: Crossentropy Hyperparameters: Learning Rate: 0.001 Epochs: 5000 Batch Size: 32 Results: Despite using GloVe embeddings, the model’s performance decreased to 55% on both training and validation sets, showing that pre-trained embeddings didn’t benefit the task. Model 3: Word2Vec Embedding Implementa- tion: Similar to the GloVe model, this model used Word2Vec-Google-News-300 embeddings (300 dimensions). Architecture: Embedding Layer: Word2Vec (frozen weights) Layers: 256 Activation Function: ReLU Optimizer: Adam Loss Function: Crossentropy Hyperparameters: Learning Rate: 0.001 Epochs: 5000 Batch Size: 32 Results: Like GloVe, this model also underperformed, reaching an accuracy of around 54%.

Model 4: FastText Embedding Implementation: This model used FastText-Wiki-News-Subwords- 300 embeddings (300 dimensions). Architecture: Embedding Layer: FastText (frozen weights) Lay- ers: 256 Activation Function: ReLU Optimizer: Adam Loss Function: Categorical Crossentropy Hyperparameters: Learning Rate: 0.001 Epochs: 5000 Batch Size: 32 Results: The FastText em-

|<p>ID UTTERANCE CORE RELATIONS</p><p>0 who plays luke on star wars new hope movie.starring.actor movie.starring.character</p>|||||||
| - | :- | :- | :- | :- | :- | :- |
|1 show credits for the godfather movie.starring.actor|||||||
|2 who was the main actor in the exorcist movie.starring.actor|||||||
||||Table 2: Example of training dataset.||||
||||||||
|Model 1|Test Size 0.6|<p>Layers</p><p>256</p>|<p>Learning rate</p><p>0\.001</p>|<p>Epochs Train a</p><p>500</p>|<p>ccuracy</p><p>98%</p>|Test accuracy 64%|
|2|0\.25|256|0\.001|1000|95%|71%|
|3|0\.2|512|0\.0005|6000|94%|65%|
|4|0\.2|128|0\.001|1000|95%|66%|
|5|0\.2|256|0\.005|1000|96%|72%|
|6\*|0\.2|128|0\.0027|1000|96%|76%|
|7|0\.2|128|0\.0027|100|99%|79%|

Table 3: Comparison of model performance under different configurations. Model 6 is the only one using Adagrad optimizer.

bedding also did not significantly improve model performance, achieving only around 60-65% accu- racy on both training and validation sets.

3 Experiments

For the final model, I experimented with different train-test splits. Initially, a test size of 0.6 was used, but it resulted in poor model performance, likely due to insufficient training data. I gradually adjusted the test size, with 0.2 proving to be the most effective, as it provided enough training data while ensuring reliable validation results. To han- dle data sparsity and ensure balanced training, the utterances were tokenized, padded to a uniform length, and mapped to word embeddings (when applicable). I observed no significant class imbal- ance in the dataset, so no specific techniques were needed for addressing this issue.

3\.1 Hyperparameter Tuning

Each model was tested with different configura- tions of learning rates, hidden units, and epochs to determine the best performing setup. Table 3 has showed all the combinations I’ve tried.

Model1: TestSize=0.6, 256layers, LearningRate

- 0.001, 500 epochs, batch size = 64.

Model 2: Test Size = 0.25, 256 layers, Learning Rate = 0.001, 1000 epochs, batch size = 64. Model3: TestSize=0.2, 512layers, LearningRate

- 0.0005, 6000 epochs, batch size = 128. Model4: TestSize=0.2, 128layers, LearningRate
- 0.001, 1000 epochs, batch size = 32

  Model5: TestSize=0.2, 256layers, LearningRate

- 0.005, 1000 epochs, batch size = 16.

Model6: TestSize=0.2, 128layers, LearningRate

- 0.0027, 1000 epochs, batch size = 16, optimizer: Adagrad.

Model7: TestSize=0.2, 128layers, LearningRate

- 0.0027, 100 epochs, batch size = 16, optimizer: Adam.

  All models were evaluated based on accuracy and lost during training and validation.

4 Results

The performance of the models, in terms of accu- racy on the training and test sets, is summarized in Table 3. Model 7, which use 0.2 test size, 128 layers, 0.0027 learning rate and 100 epochs performed exceptionally well. It achieved a training accuracy of 99% and a test accuracy of 79%, making it the best-performing model in this experiment.

Model 1 and Model 2 have higher test sizes (0.6 and 0.25, respectively). This means more data is allocated to testing, which could lead to better performance estimates but might sacrifice training efficiency since less data is available for training. Model 3 (512 layers) has the highest number of hidden units, suggesting it might capture more complex patterns but at the risk of overfitting if the model is too large for the available data. The test accuracy rate also shows that there’s overfitting as the accuracy rate is really low. Model 4, Models 6 and 7 (128 layers) are the smallest models. Fewer hidden units may reduce complexity and risk of overfitting, but they may struggle with more

complex patterns in the data.

Model 5 (learning rate of 0.005) uses the largest learning rate, meaning it will take larger steps in updating weights during training. This can lead to faster training but risks missing a local minimum or converging improperly. Models 3, 6, and 7 (learning rate 0.0027 and 0.0005) use smaller learning rates, which may result in more gradual and potentially stable training but may take longer to converge. Models 1, 2, and 4 (learning rate

0\.001) are moderate and might provide more stable

(GloVe, Word2Vec, FastText) performed worse. These models suffered from underfitting, where the low accuracy on both training and validation sets indicated that the models were not complex enough or that the embeddings failed to capture the relevant information needed to predict knowledge graph relations effectively. Increasing the number of epochs or adjusting the learning rate did not lead to significant improvements, reinforcing that the embeddings were not well-suited for this specific task.

convergence without being too slow.

![](Aspose.Words.b485898c-3b19-4923-bc41-8d299d998ed9.003.jpeg)

Figure 1: Accuracy and Loss for Model 6

![](Aspose.Words.b485898c-3b19-4923-bc41-8d299d998ed9.004.png)

Figure 2: Accuracy and Loss for Model 7

One thing worth mentioning is that a different optimizer has been used for 6 and 7. For 6, Ada- grad is used, and for 7, Adam is used. I believe the optimizer change has made some differnece. For Adagrad, its having relatively long process of train- ing, and accuracy rate is increasing slowly. Adam on the other hand, increase the accuracy rate dur- ing training rapidly, and took it only 100 epochs to reach to the same accuracy rate. Model 1, 2, and 5 (256 layers) are moderate in terms of complexity, offering a balance between capacity and efficiency. In addition, models with pre-trained embeddings
