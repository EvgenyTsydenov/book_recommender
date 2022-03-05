# Book Recommender

This project was created for educational purposes to get understanding of how recommendation systems work. 
To practice and understand aspects of different approaches, several models were built to provide personalized book recommendations.

First, analysis of [Book-Crossing](book_crossing/preprocessing.ipynb) and [Goodreads](goodreads/preprocessing.ipynb) data was implemented, which showed that these datasets contained a lot of missing and wrong information. Thus, only rating information was saved, and the corresponding book and author data were downloaded [from Penguin Random House](penguin_random_house/downloading.ipynb). The downloaded raw data were parsed and [preprocessed](penguin_random_house/preprocessing.ipynb) for further use in building recommenders.

[The practice](recommender.ipynb) started from collaborative filtering approaches. The following types of collaborative algorithms were implemented and described in detail:
- [memory-based](models/memory_based.py) item-item and user-user algorithms
- matrix factorization approach based on [singular value decomposition](models/svd.py)
- matrix factorization approach using [gradient descent](models/gradient_descent.py) with or without user and item biases.

These approaches contain a lot of hyperparameters that should be tuned. So, [Optuna](https://optuna.org/) was used for the optimization of hyperparameters, [neptune.ai](https://neptune.ai/) — for monitoring. The results of tuning can be found [in this dashboard](https://app.neptune.ai/evgenytsydenov/book-recommender/experiments?split=tbl&dash=charts&viewId=95a8ad38-184d-4545-8042-0cc3a284a3f0). 

Next, a brief explanation of content-based algorithms and their comparison with collaborative filtering were provided, but their implementation is a plan for the future.
