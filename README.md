# Bracketology.jl

Yet another computer scientist/statistician building a model for NCAA basketball 🏀🏀🏀

This repository contains code for:
1. Fitting a statistical model to game data.
2. Predicting the outcomes of future games (and assigning probabilities to those outcomes).
3. Filling a most-probable bracket based on the model's inferences.

Of course, this code could be used for other sports or games than basketball.

## Model

We currently use matrix factorization to model teams' game scores.

Imagine a giant matrix **A**:
* Each row represents a team on a particular date
* Each column represents a team on a particular date
* Entry (i,j) represents team i's score against team j on the corresponding date.

The matrix **A** is extremely sparse. In fact, each row and each column contain only one observed entry.
It's empty otherwise.

Our model supposes **A** is generated from four much smaller arrays, **X, b, Y,** and **c**.
* **X** and **b** model *offensive* ability
* **Y** and **c** model *defensive* ability.

Entry (i,j) is generated by *multiplying* **X** and **Y**, and *adding* **b** and **c**. 

See the graphic below for illustration.

<p align="center">
  <img src="bracketology_model.png" width="600" title="hover text">
</p>

We **regularize** the entries of **X, b, Y, c** such that
* Entries belonging to the same team, but on different dates, are encouraged to be similar to each other.
* This encouragement is stronger when dates are near in time, and weaker when dates are distant in time.
* These assumptions basically amount to Gaussian process priors on **X, b, Y,** and **c**.

I rely on my own [SparseMatFac.jl](https://github.com/dpmerrell/SparseMatFac.jl) Julia package for model implementation.

## Data

The only data we need are final scores of NCAA games from recent years.

The data in `scripts/games_19-22.tsv` were scraped from [Bart Torvik's website](https://barttorvik.com/gamestat.php).

It contains NCAA games from 2019 through January 2022.


## Results

A bracket constructed from the `games_19-22.tsv` data can be found in `scripts/filled_bracket.csv`. Alternatively, see it in [Google Sheets](https://docs.google.com/spreadsheets/d/1V59gH4-CDqSBj4xKBC4ClzewRVO1hap4pQ00M74tS8Y/edit?usp=sharing).

It predicted the correct champion (Kansas) of 2022 March Madness.

I'm working on a more rigorous evaluation&mdash;scoring against the true results and comparing against simpler baselines (e.g., based on seed).

