Beginner ML Olympic Medals Predictor:
- Use a Linear Regression Model to make predictions of the number of olympic medals each country will make
- Line of best fit will determine predictions for unknown data
- Evaluate error by calculating mean absolute error

** This particular model is good at predicting larger countries future medals, but poor at smaller countries predictions

Ways to improve predictions:
- Add in more predictors (events, age, heights)
- Try different models (random forest, nueral networks)
- Use more detailed data sources (ex. individual athlete data - see which specific athletes will win medal and add to find country medals won)
- Reshape columns (those non-linear to medals) with mathmatical transformations to create a more linear correlation
- Measure error more predictibly (ex. backtesting system)
- Train different models on different types of metrics (low-achieving countries vs high-achieving countries)