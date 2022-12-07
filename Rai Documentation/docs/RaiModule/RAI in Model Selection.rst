.. _RAI in Model Selection:

===================
**Model Selection**
===================

Model selection is the process of selecting one of the models as the final ML-model for a training dataset.

- To figure this out, RAI will usually come up with some kind of evaluation metric. 
- Then it will divide the training dataset into 3 parts: Training set, Validation set(sometimes called development), and a Test dataset. 
 - **The Training** - It is used to fit the models, 
 - **The Validation** - It is used to estimate prediction error for model selection, 
 - **The Test set** - It is used to do a final evaluation and assessment of the generalization error of the chosen model on the test dataset. 
- This way, we can determine the model with the lowest generalization error.It refers to the performance of the model on unseen data, i.e. data that the model hasn’t been trained on.

**When constructed:**

 - Models are optionally passed the name, the models functions for 
   inferences, its name, the model, its optimizer, its loss function, its class and a 
   description. Attributes of the model are used to determine which metrics are relevant.

 - We need to give inputs like output_features,predict_fun,predict_prob_fun,generate_text_fun,
   generate_image_fun,name,display_name,agent,loss_function,optimizer,model_class,description.

 - Here name is the mandatory parameter if not provided it will throw error.
 - output_features if not provided will throw error like "output_features must be a Feature or array of Features".


.. admonition:: Example
    :class: dropdown

    We may have a dataset for which we are interested in developing a classification or regression predictive model. We do not know beforehand as to which model will perform best on this problem, as it is unknowable. Therefore, we fit and evaluate a suite of different models on the problem.


For instance

- It will create an empty list and populate it with the pair (model_name, model)
- It will define the parameters for splitting the data through the Scikit-Learn KFold cross-validation
- It will create a for loop where it will cross-validate each model and save its performance
- It will view the performance of each model in order to choose the one that performed best
- It will define a list and insert the models we want to test.
- If it is a regression problem, it will use the mean squared error (MSE) metric.
- It will cross-validate, tests, and its performance saved in the results for each model
- The visualization is very simple and will be done through a boxplot.


