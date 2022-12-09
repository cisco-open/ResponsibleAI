.. _RAI Fairness:

==================
**RAI - Fairness**
==================


The RAI goal is not to make the data unbiased or the ML model fair but to make the overall system and outcomes fair.


- Fairness in machine learning refers to the various attempts at correcting algorithmic bias in automated decision processes based on machine learning models. Decisions made by computers after a machine-learning process may be considered unfair if they were based on variables considered sensitive.
- RAI will use fairness as the process of correcting and eliminating algorithmic bias (of race and ethnicity, gender, sexual orientation, disability, and class) from machine learning models.
- If a model is trained using an unbalanced dataset, such as one that contains far more people with lighter skin than people with darker skin, there is serious risk the model’s predictions will be unfair when it is deployed.
- Here RAI enables fairness directly into the model internal representation itself to produce fair outputs even if it is trained on unfair data.


**Fairness Metrics**
====================

Fairness measures allow us to assess and audit for possible biases in a trained model. There are several types of metrics that are used in RAI in order to assess a model’s fairness. They can be classified into groups as follows:


**Individual Fairness**

=================================================  =================================================================================
display_name                                       Description
=================================================  =================================================================================
generalized_entropy_index                          A measure of information theoretic redundancy in data. 
                                                   Describes how unequally the outcomes of an algorithm benefit 
                                                   different individuals or groups in a population
                                                   
theil_index                                        The generalized entropy of benefit for all individuals in the dataset, 
                                                   with alpha = 1.\nMeasures the inequality in benefit allocation for individuals.
                                                   \nA value of 0 implies perfect fairness
                                                                                        
coefficient_of_variation                           The square root of twice the generalized entropy index with alpha = 2.
                                                   \nThe ideal value is 0.           
=================================================  =================================================================================


.. container:: toggle, toggle-hidden

    .. admonition:: Individual Fairness

        .. image::  /images/Individual_fairness.png


**Group Fairness**

=================================================  ====================================================================================================================
display_name                                       Description
=================================================  ====================================================================================================================
disparate_impact_ratio                             The ratio of rate of favorable outcome for the unprivileged group to that of the privileged group.
                                                   \nThe ideal value of this metric is 1.0 A value < 1 implies higher benefit for the privileged group 
                                                   and a value > 1 implies a higher benefit for the unprivileged group.
                                                                                      
statistical_parity_difference                      The difference of the rate of favorable outcomes received by the unprivileged group to the privileged group.
                                                   \nThe idea value is 0.0  

between_group_generalized_entropy_error            The between group decomposition for generalized entropy error

equal_opportunity_difference                       The difference of true positive rates between the unprivileged and the privileged groups.
                                                   \nThe true positive rate is the ratio of true positives to the total number of actual positives for a given group.
                                                   \nThe ideal value is 0. A value of < 0 implies higher benefit for the privileged group and a value > 0 implies 
                                                   higher benefit for the unprivileged group
=================================================  ====================================================================================================================

.. container:: toggle, toggle-hidden

    .. admonition:: Group fairness

        .. image::  /images/Group_fairness.png


**General Fairness**

=================================================  =======================================================================================
display_name                                       Description
=================================================  =======================================================================================
average_odds_difference                            The average difference of false positive rate (false positives / negatives) and 
                                                   true positive rate (true positives / positives)
                                                   between unprivileged and privileged groups.
                                                   \nThe ideal value is 0.  A value of < 0 implies higher benefit for the privileged group 
                                                   and a value > 0 implies higher benefit for the unprivileged group
                                                                                      
between_all_groups_coefficient_of_variation        The square root of twice the pairwise entropy between every pair of privileged and 
                                                   underprivileged groups with alpha = 2.\nThe ideal value is 0  

between_all_groups_generalized_entropy_index       The pairwise entropy between every pair of privileged and underprivileged groups.
                                                   \nThe ideal value is 0.0

between_all_groups_theil_index                     The pairwise entropy between every pair of privileged and underprivileged groups with
                                                   alpha = 1.\nThe ideal value is 0.0
=================================================  =======================================================================================


.. container:: toggle, toggle-hidden

    .. admonition:: General Fairness

        .. image::  /images/general_fairness.png


**Dataset Fairness**

=================================================  =======================================================================================
display_name                                       Description
=================================================  =======================================================================================
base_rate                                          Base Rate is the rate at which a positive outcome occurs in Data. 
                                                   In formula it is, Pr(Y=pos_label) = P/(P+N)
                                                                                      
num_instances                                      Num Instances counts the number of examples in Data 

num_negatives                                      Num Negatives counts the number of negative labels in Data 

num_positives                                      Num Positives calculates the number of positive labels in Data
=================================================  =======================================================================================

.. container:: toggle, toggle-hidden

    .. admonition:: Dataset Fairness

        .. image::  /images/Dataset_fairness.png


For Instance:

- User can obtain to compute specialized metrics like Disparate Impact Ratio to show the fairness of the models classification across sensitive characteristics

.. figure:: ../images/fairness.gif
   :align: center
   :scale: 40 %

   fairness_of_the_model



You can see quick Demo here - How RAi can be used for detecting and :ref:`resolving bias and fairness <Robustness of AI>` in AI models.