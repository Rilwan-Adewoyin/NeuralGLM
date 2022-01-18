When defining the Loss Class in loss_utils:
1- Should I keep all the logits?
2- I do not understand each line of the code. Some parts I get what is going on, but not familiar with most of the code used. Some parts I just don't understand. - need to go through line by line and take notes.
3- I defined a function for CP loss and another to convert to GLM parameters. Are they in the right place? Is it problematic that the optimization is done on different parameters, given that the GLM and regular parameters are linked?
4-line 154, is it specific to loghurdle? What does it do?
5- Should I use different names inside the CPloss class compared to the other loss classes?

7- Not sure how to say what is being optimized in loss calculations
8- How/where does the link function come into play? Is it in target_rainfall?
9- Which expression makes computations more efficient for the model, in the losses for CP and gamma?
10- The best performing model from UK rainfall GLM modelling uses observations from all locations as predictors for every single location. Is that too computationally intense?

