from typing import Optional

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dot, Flatten
from tensorflow.keras.regularizers import l2


class RecommenderGD(tf.keras.Model):
    """Matrix factorization recommender trained with gradient descent"""

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
        self._user_flatten = Flatten(name='UserFlatten')
        self._book_embedding = self._get_embed(
            self._books_count, 'BookEmbedding')
        self._book_flatten = Flatten(name='BookFlatten')
        self._dot_product = Dot(axes=1, name='RatingValue')

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
        user_embed = self._user_flatten(user_embed)
        book_embed = self._book_embedding(work_ids)
        book_embed = self._book_flatten(book_embed)
        return self._dot_product([user_embed, book_embed])

    def build_graph(self) -> tf.keras.Model:
        """Compute model graph for visualization.

        :return: model.
        """
        user_input = Input(shape=(1,), name='UserIndex')
        book_input = Input(shape=(1,), name='BookIndex')
        return Model(inputs=(user_input, book_input),
                     outputs=self.call((user_input, book_input)))
