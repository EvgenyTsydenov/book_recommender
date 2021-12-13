from typing import Optional

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dot, Flatten, Add
from tensorflow.keras.regularizers import l2


class RecommenderGD(tf.keras.Model):
    """Matrix factorization recommender trained with gradient descent."""

    def __init__(self, users_count: int, books_count: int, random_seed: int,
                 embed_size: int, l2_regularizer: float = 0, **kwargs):
        """Create recommender.

        :param users_count: number of users.
        :param books_count: number of books.
        :param random_seed: random seed.
        :param embed_size: size of embeddings.
        :param l2_regularizer: value for L2 regularization.
        :param kwargs: additional arguments to keras Model.
        """
        tf.random.set_seed(random_seed)
        super().__init__(**kwargs)
        self._l2_penalty = l2_regularizer
        self._embed_size = embed_size
        self._users_count = users_count
        self._books_count = books_count
        self._user_embedding = self._get_embed(
            self._users_count, 'UserEmbedding')
        self._book_embedding = self._get_embed(
            self._books_count, 'BookEmbedding')
        self._rating_flatten = Flatten(name='RatingValue')
        self._dot_product = Dot(axes=2, name='DotProduct')

    def _get_embed(self, items_count: int, name: Optional[str] = None):
        """Create embedding layer.

        :param items_count: number of items which embeddings to create.
        :return: embedding layer.
        """
        l2_reg = l2(self._l2_penalty) if self._l2_penalty > 0 else None
        return Embedding(items_count, self._embed_size,
                         embeddings_initializer='he_normal',
                         embeddings_regularizer=l2_reg, name=name)

    def call(self, inputs, **kwargs):
        """Call the model.

        :param inputs: model inputs as user_id and book_id.
        :return: model output.
        """
        user_ids, work_ids = inputs
        user_embed = self._user_embedding(user_ids)
        book_embed = self._book_embedding(work_ids)
        product = self._dot_product([user_embed, book_embed])
        return self._rating_flatten(product)

    def build_graph(self) -> tf.keras.Model:
        """Compute model graph for visualization.

        :return: model.
        """
        user_input = Input(shape=(1,), name='UserIndex')
        book_input = Input(shape=(1,), name='BookIndex')
        return Model(inputs=(user_input, book_input),
                     outputs=self.call((user_input, book_input)))


class RecommenderGDBiased(RecommenderGD):
    """Matrix factorization recommender trained with gradient descent.

    Uses embeddings with users' and items' biases.
    """

    def __init__(self, users_count: int, books_count: int, random_seed: int,
                 embed_size: int, l2_regularizer: float = 0, **kwargs):
        """Create recommender.

        :param users_count: number of users.
        :param books_count: number of books.
        :param random_seed: random seed.
        :param embed_size: size of embeddings.
        :param l2_regularizer: value for L2 regularization.
        :param kwargs: additional arguments to keras Model.
        """
        super().__init__(users_count, books_count, random_seed,
                         embed_size, l2_regularizer, **kwargs)
        self._book_bias = Embedding(self._books_count, 1, name='BookBias')
        self._user_bias = Embedding(self._users_count, 1, name='UserBias')
        self._rating_biased = Add(name='BiasedRating')

    def call(self, inputs, **kwargs):
        """Call the model.

        :param inputs: model inputs as user_id and book_id.
        :return: model output.
        """
        user_ids, work_ids = inputs
        user_embed = self._user_embedding(user_ids)
        book_embed = self._book_embedding(work_ids)
        user_bias = self._user_bias(user_ids)
        book_bias = self._book_bias(work_ids)
        product = self._dot_product([user_embed, book_embed])
        biased_rating = self._rating_biased([product, user_bias, book_bias])
        return self._rating_flatten(biased_rating)
