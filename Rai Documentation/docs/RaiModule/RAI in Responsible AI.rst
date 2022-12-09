.. _RAI in Responsible AI:


=========================
**RAI Responsible in AI**
=========================

**Ethical AI**
==============

- Ethical AI Utilization

 - RAI practices and adheres to well-defined ethical guidelines regarding fundamental values, including such things as individual rights, privacy, non-discrimination, and non-manipulation.
 - It places fundamental importance on ethical considerations in determining legitimate and illegitimate uses of AI. 
 - It believes integrity and ethical behavior are fundamental to a succesful AI applications. 
 - It help them to dramatically improve their operations and their products for the betterment of humankind.


**Robustness and the Adversarial AI**
=====================================

- Robustness and the Adversarial AI Utilization

 - In the real world, AI models can encounter both incidental adversity, such as when data becomes corrupted, and intentional adversity, such as when hackers actively sabotage them.
 - Both can mislead a model into delivering incorrect predictions or results. 
 - Adversarial robustness refers to a model’s ability to resist being fooled.
 - RAI helps to improve the adversarial robustness of AI models, making them more impervious to irregularities and attacks.
 - RAI also focuses on the threats of Evasion (change the model behavior with input modifications), Trojaning (can access to the model and its parameters and retrain this model), Poisoning (control a model with training data modifications), Extraction (steal a model through queries) and Inference (attack the privacy of the training data). 
 - RAI aims to support tasks, and data types in continuous development by defending AI against adversarial attacks and making AI systems more secure.

.. admonition:: Example
    :class: dropdown

    If i give you a dataset of cat and dog photos, in which cats are always wearing bright red bow ties, your model may learn to associate bow ties with cats. If I then give it a picture of a dog with a bow tie, your model may label it as a cat. Adversarial machine learning also often includes the ability to identify specific pieces of noise that can be added to inputs to confound a model.
    Therefore, if a model is robust, it basically means that it is difficult to find adversarial examples for the model. Usually this is because the model has learned some desirable correlations (e.g. cats have a different muzzle shape than dogs), rather than undesirable ones (cats have bow ties; pictures containing cats are 0.025% more blue than those containing dogs; dog pictures have humans in them more often; etc.).
    So here RAI try to directly exploit this idea, by training the model on both true data and data designed by an adversary to resemble the true data. 

.. _Contribution to principle of AI:


**Explainable AI aspects of model development**
===============================================

- Dashboard tools Utilization

 - RAI has the ability to handle large varieties of models like text, images and tabular data.
 - It can show Analysis of the models in the dashboard. 
 - Visualization can help in understanding of the models so that we can have the idea of where it is failing and where it is succeeding. 
 - It can use Grad-cam to help in highlighting the important regions in the images by bound boxing. 
 - After fitting the model RAI will help the show the results to the dashboard and we can see from all the category Explanability, Robustness, Performance and Fairness how the model is working on these categories and we can analyse how to make it better. 
 - We can check how many tests are passed in the categories and perform the changes accordingly. 
 - We can also make use of metric graphs and get to know how individual parameters and metric changes during the model development.
