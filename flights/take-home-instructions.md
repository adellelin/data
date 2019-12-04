# Warby Parker Data Science Interview Take-Home Analysis

## Overview

We're not a fan of gotchas or pop quizzes in interviews, so we've designed this take-home exercise to give you the time and space to think and work on a realistic problem in a more normal setting. The technical portion of the on-site interview will be centered on this problem and allow you and the interviewer to discuss a problem you've had time to process and explore.

For this take-home exercise, we're giving you a dataset, and asking you to create a brief reproducible report where you:

- import and manipulate some data,
- do exploratory analysis with graphs and summaries, and
- build a preliminary model to estimate and predict some quantities.

The goal is to see how you approach a preliminary analysis of a new dataset. We're not looking for an exhaustive analysis or a perfect model. We just want to get a sense of how think about data and models, and how you write code to express those thoughts.

You should aim to spend no more than 4-5 hours on this analysis, including and hour or so to get acquainted with the dataset. If it takes less time, great. If you find it taking more time, feel free to stop and explain anything extra you would do if you had more time (or more data).

We recognize that people have widely varying levels of commitments in their lives. If dedicating time to this would be an impediment to continuing to apply, please do be in touch and we'll try to work something out.

If you have questions about this analysis, scope, or expectations, don't hesitate to reach out to carl.vogel@warbyparker.com.

### What we're looking for

- *A reproducible report.* We recommend either an RMarkdown or Jupyter notebook. Regardless of the format, we should be able to produce your results from source code with few-to-no manual steps in between (for example, copy and pasting results, etc.)
   - If you're relying on any libraries or packages that **aren't** available via standard sources such as CRAN or PyPI, please indicate where you got them from.

- *Clear, well-organized code.* It doesn't need to be immaculate, but it should be clear enough that another data scientist can understand what you're doing without too much trouble.

- *Clear graphs and tables.* Graphs should be well-labeled and relatively self-contained. We should be able to understand what's being graphed just by reading the graph, and without referring to the code.

- *Clear and concise written explanations of your assumptions, modeling choices, and findings.* Imagine your audience is another data scientist who understands the data set, but is trying to understand how you've analyzed it.
  - For example, you don't need to explain *what* a Lasso regression is, but we should be able to understand *why* you used it.
  - Try to use full, grammatically-correct sentences.

- *Ideas for further analysis.* If you have ideas about what you'd try with more time or additional data, feel free to tell us.

### What we're NOT looking for

- *"Production-quality" code.* You don't need to write unit tests, create a package, or thoroughly document functions.

- *Exhaustive data cleaning.* The data set is a real one, so it won't be perfect. If you find potential anomolies, feel free point them out and handle them in a reasonable way. Just dropping weird observations, or ignoring them if you don't think they're material is fine.

- *An amazing model.* All models are wrong; some are useful. One you create in an hour for a take-home assignment probably won't even be useful. That's okay. This should be your "first pass" model. We're more interested in seeing how you choose and reason about a model, regardless of its performance.

- *Specific answers.* These questions are intentionally open-ended and you may interpret them questions differently than we intended. You might analyze the data in a way we didn't expect. That's fine! We're more interested in seeing your thought process than in getting particular answers.

## The data

The data is a small SQLite database, in a file called `flights.sqlite`. This data is extracted from the ["Reporting Carrier On-Time Performance" database][1], maintained by the Bureau of Transportation Statistics. The database has four tables:

- `airlines`
- `airports`
- `cities`
- `ny_flights`

The first three tables are lookup tables for identifying the airlines, airports, and cities in the `ny_flights` table.

The `ny_flights` table contains data on domestic flights departing from and arriving at airports in New York state between June 1, 2018 and June 30, 2019. The fields include information on the scheduled and actual departure and arrival times of the flights.

The file `flights database documentation.pdf` contains a relationship diagram for the tables, as well as a listing of the field definitions in the `ny_flights` table.


## Exercises

For the purpose of this analysis we're only interested in *flights departing from the major NYC airports*. These are LaGuardia, John F. Kennedy, and Newark-Liberty.

### Part 1: Exploring a route

Choose a route between two airports that you're interested in, for example JFK to Los Angeles Int'l, or LaGuardia to O'Hare. Do a brief exploratory analysis of this route using graphs to vizualize the data when you can. You don't need to examine every possible variable or relationship; a handful of interesting ones is fine.

Some example questions you might investigate:

- How busy is this route? Are there more flights on some days than others?
- Which airlines provide the most flights on this route?
- What do delays on this route look like?
- Are there strange-looking days on this route with fewer-than-normal flights or longer-than-normal delays?

... or anything you might be curious about.

### Part 2: Comparing airports

Create some summary data and visualizations comparing the three NYC-area airports. Of particular interest are departure delays from the airports. Are there differences in the probability of a late (say, more than 30 min.) departure? How does time spent on the runway compare amongst the airports?

### Part 3: Modeling delays

1. Build a simple model that estimates the probability that a given flight will have a late (greater than 30 min.) departure or arrival.

	- You can interpret the term "model" loosely: Any method that, given some inputs known prior to a flight, will produce an estimated probability that the flight will be delayed or arrive late. You should also be able to use the model to show the uncertainty around the estimate.

2. Discuss the results of your fitted model. Here are a few things to describe:

	- The performance of your model---either its in-sample fit, or out-of-sample prediction accuracy (or both).
	- For a few interesting example flights or routes, use the model to estimate probability of a late departure or arrival and its uncertainty. Describe the results in plain English.
	- If your model is suited to estimating factors that affect delays (that is, it has interpretable coefficients), describe the estimated effects.

3. Describe anything you feel might be missing from your model, or alternative approaches you might try with more time or data. If you think additional data would be helpful, describe what you would want.


[1]: https://www.transtats.bts.gov/DL_SelectFields.asp
