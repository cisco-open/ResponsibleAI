.. _RAI in Responsible AI:


=========================
**RAI Responsible in AI**
=========================

**Ethical AI**
==============

- Ethical AI Utilization

 - RAI practices and adheres to well-defined ethical guidelines regarding fundamental values, including individual rights, privacy, non-discrimination, and non-manipulation.
 - RAI places a fundamental importance on ethical considerations in determining legitimate and illegitimate uses of AI.
 - RAI encodes the belief that integrity and ethical behavior are fundamental to succesful AI applications.
 - RAI helps users to dramatically improve their operations and products for the betterment of humankind.


**Robustness and the Adversarial AI**
=====================================

- Robustness and the Adversarial AI Utilization

 - In the real world, AI models can encounter incidental adversity, such as when data becomes corrupted, and intentional adversity, such as when hackers actively sabotage them.
 - Both can mislead a model into delivering incorrect predictions or results. 
 - Adversarial robustness refers to a model’s ability to resist being fooled.
 - RAI helps to improve the adversarial robustness of AI models, making them more impervious to irregularities and attacks.
 - RAI also focuses on the threats of Evasion (change the model behavior with input modifications), Trojaning (can access the model and its parameters and retrain this model), Poisoning (control a model with training data modifications), Extraction (steal a model through queries) and Inference (attack the privacy of the training data). 
 - RAI aims to support tasks, and data types in continuous development by defending AI against adversarial attacks and making AI systems more secure.

.. admonition:: Example
    :class: dropdown

    If i give you a dataset of cat and dog photos, in which cats always wear bright red bow ties, your model may learn to associate bow ties with cats. If I give it a picture of a dog with a bow tie, your model may label it as a cat. 
    Adversarial machine learning also often includes identifying specific noise that can be added to inputs to confound a model. Therefore, if a model is robust, it means that it is difficult to find adversarial examples for the model. 
    Usually this is because the model has learned some desirable correlations (e.g. cats have a different muzzle shape than dogs), rather than undesirable ones (cats have bow ties; pictures containing cats are 0.025% more blue than those containing dogs; dog pictures have humans in them more often; etc.). 
    So here RAI try to directly exploit this idea, by training the model on both true data and data designed by an adversary to resemble the true data.

.. _Contribution to principle of AI:


**Explainable AI aspects of model development**
===============================================

- Dashboard tools Utilization

 - RAI can handle large varieties of models like text, images and tabular data.
 - It can show Analysis of the models in the dashboard.
 - Visualization can help in understanding the models so that we can have the idea of where it is failing and succeeding.
 - It can use Grad-cam to help in highlighting the important regions in the images by bound boxing.
 - RAI will show the results on the dashboard after fitting the model, allowing us to see how the model performs on Explanability, Robustness, Performance and Fairness and help us analyze how we can improve the model
 - In each category, we can check how many tests have passed and make the changes as necessary
 - Additionally, metric graphs can assist in understanding how parameter and metric changes during model development
