.. _Robustness of AI:


====================
**Robustness of AI**
====================

In this Demo case we can see how RAi can be used for detecting and resolving bias and fairness in AI models.


- To demonstrate how RAI works let’s consider a simple data science project to predict income level of participants.
- In this dataset there is imbalance between white and black participants.
- Here rai will show how to identify and mitigate the problem.
- After fitting the model, we can ask rai to send the measurements back to the dashboard.


.. container:: toggle, toggle-hidden

    .. admonition:: fitting the model

        .. image::  /images/rai_demo_2_Moment.png



- We can now to back to the dashboard and for each individual category we can see how system have performed.
- For instance, we can see that 1 out of 3 test is passed for fairness. This show significant problem in fairness.

.. container:: toggle, toggle-hidden

    .. admonition:: significant problem in fairness

        .. image::  /images/rai_demo_2.2_Moment.png


- Now we can investigate this problem by taking a look at the individual metrics.
- We can select the category of interest and for each category we can take a look at individual metric that has been calculated. 
- For instance, we can go to frequency statistic and look at the race parameter which show more that 85% of participants are white.

.. container:: toggle, toggle-hidden

    .. admonition:: race parameter

        .. image::  /images/rai_demo_3_Moment.png


- To mitigate this imbalance problem, we can go back to data science project and perform some form of mitigation strategy. 
- Here we are using Reweighing algorithm and after fitting the model once again.
- we can ask RAI to compute the metrics once again and evaluate our model.

.. container:: toggle, toggle-hidden

    .. admonition:: Reweighing algorithm

        .. image::  /images/rai_demo_4_Moment.png

- Now we can go back to the dashboard. 
- At the homepage of dashboard we can take a look how system have performed after this mitigation which shows that all the fairness tests have passed this time.


.. container:: toggle, toggle-hidden

    .. admonition:: fairness tests have passed

        .. image::  /images/rai_demo_5_Moment.png