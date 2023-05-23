.. _Robustness of AI:


====================
**Robustness of AI**
====================

In this Demo case, we can see how RAI can detect and resolve bias and fairness in AI models.


- To demonstrate how RAI works, let's consider a simple data science project to predict the income level of participants.
- In this dataset, there is an imbalance between white and black participants.
- Here RAI will show how to identify and mitigate the problem.
- After fitting the model, we can ask RAI to send the measurements back to the dashboard.


.. container:: toggle, toggle-hidden

    .. admonition:: fitting the model

        .. image::  ../images/rai_demo_2_Moment.png



- We can now go back to the dashboard and see how the system has performed for each category.
- For instance, we can see that 1 out of 3 tests is passed for fairness. This shows a significant problem in fairness.

.. container:: toggle, toggle-hidden

    .. admonition:: significant problem in fairness

        .. image::  ../images/rai_demo_2.2_Moment.png


- Now we can investigate this problem by looking at the individual metrics.
- We can select the category of interest, and for each category, we can look at the individual metric that has been calculated.
- For instance, we can go to frequency statistics and look at the race parameter, which shows more than 85% of participants are white.


.. container:: toggle, toggle-hidden

    .. admonition:: race parameter

        .. image::  ../images/rai_demo_3_Moment.png


- To mitigate this imbalance problem, we can go back to the data science project and perform some mitigation strategies.
- Here we are using Reweighing algorithm after fitting the model once again.
- We can ask RAI to compute the metrics again and evaluate our model.


.. container:: toggle, toggle-hidden

    .. admonition:: Reweighing algorithm

        .. image::  ../images/rai_demo_4_Moment.png


- Now we can go back to the dashboard.
- At the dashboard's homepage, we can look at how the system has performed after this mitigation, which shows that all the fairness tests have passed this time.


.. container:: toggle, toggle-hidden

    .. admonition:: fairness tests have passed

        .. image::  ../images/rai_demo_5_Moment.png