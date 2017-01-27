## Tutorial: Machine Learning with Text in scikit-learn

Presented by [Rahul Yadav](https://in.linkedin.com/in/rahulmanuwas)
 

### Description

Although numeric data is easy to work with in Python, most knowledge created by humans is actually raw, unstructured text. By learning how to transform text into data that is usable by machine learning models, you drastically increase the amount of data that your models can learn from. In this tutorial, we'll build and evaluate predictive models from real-world text using scikit-learn.

### Objectives

By the end of this tutorial, attendees will be able to confidently build a predictive model from their own text-based data, including feature extraction, model building and model evaluation.



### Abstract

It can be difficult to figure out how to work with text in scikit-learn, even if you're already comfortable with the scikit-learn API. Many questions immediately come up: Which vectorizer should I use, and why? What's the difference between a "fit" and a "transform"? What's a document-term matrix, and why is it so sparse? Is it okay for my training data to have more features than observations? What's the appropriate machine learning model to use? And so on...

In this tutorial, we'll answer all of those questions, and more! We'll start by walking through the vectorization process in order to understand the input and output formats. Then we'll read a simple dataset into pandas, and immediately apply what we've learned about vectorization. We'll move on to the model building process, including a discussion of which model is most appropriate for the task. We'll evaluate our model a few different ways, and then examine the model for greater insight into how the text is influencing its predictions. Finally, we'll practice this entire workflow on a new dataset, and end with a discussion of which parts of the process are worth tuning for improved performance.

### Detailed Outline

1. Model building in scikit-learn (refresher)
2. Representing text as numerical data
3. Reading a text-based dataset into pandas
4. Vectorizing our dataset
5. Building and evaluating a model
6. Comparing models
7. Examining a model for further insight
8. Practicing this workflow on another dataset
9. Tuning the vectorizer (discussion)


### Recommended Resources

**Text classification:**
* Read Paul Graham's classic post, [A Plan for Spam](http://www.paulgraham.com/spam.html), for an overview of a basic text classification system using a Bayesian approach. (He also wrote a [follow-up post](http://www.paulgraham.com/better.html) about how he improved his spam filter.)
* Coursera's Natural Language Processing (NLP) course has [video lectures](https://class.coursera.org/nlp/lecture) on text classification, tokenization, Naive Bayes, and many other fundamental NLP topics. (Here are the [slides](http://web.stanford.edu/~jurafsky/NLPCourseraSlides.html) used in all of the videos.)
* [Automatically Categorizing Yelp Businesses](http://engineeringblog.yelp.com/2015/09/automatically-categorizing-yelp-businesses.html) discusses how Yelp uses NLP and scikit-learn to solve the problem of uncategorized businesses.
* [How to Read the Mind of a Supreme Court Justice](http://fivethirtyeight.com/features/how-to-read-the-mind-of-a-supreme-court-justice/) discusses CourtCast, a machine learning model that predicts the outcome of Supreme Court cases using text-based features only. (The CourtCast creator wrote a post explaining [how it works](https://sciencecowboy.wordpress.com/2015/03/05/predicting-the-supreme-court-from-oral-arguments/), and the [Python code](https://github.com/nasrallah/CourtCast) is available on GitHub.)
* [Identifying Humorous Cartoon Captions](http://www.cs.huji.ac.il/~dshahaf/pHumor.pdf) is a readable paper about identifying funny captions submitted to the New Yorker Caption Contest.
* In this [PyData video](https://www.youtube.com/watch?v=y3ZTKFZ-1QQ) (50 minutes), Facebook explains how they use scikit-learn for sentiment classification by training a Naive Bayes model on emoji-labeled data.

**Naive Bayes and logistic regression:**
* Read this brief Quora post on [airport security](http://www.quora.com/In-laymans-terms-how-does-Naive-Bayes-work/answer/Konstantin-Tt) for an intuitive explanation of how Naive Bayes classification works.
* For a longer introduction to Naive Bayes, read Sebastian Raschka's article on [Naive Bayes and Text Classification](http://sebastianraschka.com/Articles/2014_naive_bayes_1.html). As well, Wikipedia has two excellent articles ([Naive Bayes classifier](http://en.wikipedia.org/wiki/Naive_Bayes_classifier) and [Naive Bayes spam filtering](http://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering)), and Cross Validated has a good [Q&A](http://stats.stackexchange.com/questions/21822/understanding-naive-bayes).

**scikit-learn:**
* The scikit-learn user guide includes an excellent section on [text feature extraction](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) that includes many details not covered in today's tutorial.
* The user guide also describes the [performance trade-offs](http://scikit-learn.org/stable/modules/computational_performance.html#influence-of-the-input-data-representation) involved when choosing between sparse and dense input data representations.


