.. _RAI in Model Selection:

===================
**Model Selection**
===================

Model selection is the process of selecting one of the models as the final ML model for a training dataset.

- To figure this out, RAI will usually come up with some kind of evaluation metric.
- Then it will divide the training dataset into three parts: A training set, a Validation set(sometimes called development), and a Test dataset.

 - **The Training** - It is used to fit the models, 
 - **The Validation** - It is used to estimate prediction error for model selection, 
 - **The Test set** - It is used to do a final evaluation and assessment of the generalization error of the chosen model on the test dataset. 
 
- This way, we can determine the model with the lowest generalization error. It refers to the performance of the model on unseen data, i.e., data that the model hasn’t been trained on.



.. admonition:: Example
    :class: dropdown

    We may have a dataset for which we are interested in visualizing the performance of the individual case. We do not know beforehand as to which model will perform best on this problem, as it is unknowable. Therefore, we fit and evaluate a suite of different models for the problem.


- Rai can help us with the Model selection
- We can select a Project here

.. container:: toggle, toggle-hidden

    .. admonition:: Select project

        .. image::  /images/Select_project.png


- We can go to Metric Graphs
- Metric Graphs show here how individual parameters and metrics have changed during model development


.. container:: toggle, toggle-hidden

    .. admonition:: Metric graph

        .. image::  /images/metric_graph.png


- Here, for instance, we have performed some Grid searches to select the best model for the task
- We can show individual metrics of interest


.. container:: toggle, toggle-hidden

    .. admonition:: Metric performance

        .. image::  /images/metric_performance.png


- Monitor how the system is performing in each individual case
- This helps us to select the best model that fits our desired characteristics


.. container:: toggle, toggle-hidden

    .. admonition:: Individual case

        .. image::  /images/each_case.png


.. figure:: ../images/Model_selection.gif
   :align: center
   :scale: 40 %

   Model_selection


.. important::
    :class: dropdown

    Through RAI we can detect it before it becomes a problem or respond to it when it arises by putting the right systems in place early and staying on top of data collection, labeling, and implementation.

