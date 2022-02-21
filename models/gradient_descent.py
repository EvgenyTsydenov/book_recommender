from typing import Optional, Tuple, Sequence

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dot, Flatten, Add, StringLookup
from tensorflow.keras.regularizers import l2


class RecommenderGD(tf.keras.Model):
    """Collaborative filtering recommender using matrix factorization approach
    based on gradient descent.

    :param embed_size: size of embeddings.
    :param users: unique users.
    :param items: unique items.
    :param random_seed: random seed.
    :param l2_penalty: penalty value for L2 regularization.
    :param kwargs: additional arguments to tensorflow.keras.Model.
    """

    def __init__(self, embed_size: int, users: Sequence, items: Sequence,
                 random_seed: Optional[int] = None,
                 l2_penalty: Optional[float] = None, **kwargs) -> None:
        if random_seed is not None:
            tf.random.set_seed(random_seed)
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.embed_size = embed_size

        # Layers
        self._user_lookup = StringLookup(vocabulary=users, name='UserLookup')
        self._item_lookup = StringLookup(vocabulary=items, name='ItemLookup')
        self._flatten = Flatten(name='RatingValue')
        self._dot_product = Dot(axes=2, name='EmbeddingProduct')
        self._user_embedding = self._get_embed(
            len(users), self.embed_size, 'UserEmbedding')
        self._item_embedding = self._get_embed(
            len(items), self.embed_size, 'ItemEmbedding')

    def _get_embed(self, items_count: int, embed_size: int,
                   name: Optional[str] = None) -> tf.keras.layers.Embedding:
        """Create embedding layer.

        :param items_count: number of items which embeddings to create.
        :param embed_size: size of embeddings.
        :param name: name of layer.
        :return: embedding layer.
        """
        l2_reg = l2(self.l2_penalty) if self.l2_penalty else None
        return Embedding(items_count, embed_size, name=name,
                         embeddings_initializer='he_normal',
                         embeddings_regularizer=l2_reg)

    def call(self, inputs: Tuple[Sequence, Sequence], **kwargs) -> tf.Tensor:
        """Call the model.

        :param inputs: model inputs as user_id and item_id.
        :return: model output.
        """
        user_ids, item_ids = inputs
        user_indices = self._user_lookup(user_ids)
        item_indices = self._item_lookup(item_ids)
        user_embeds = self._user_embedding(user_indices)
        item_embeds = self._item_embedding(item_indices)
        product = self._dot_product([user_embeds, item_embeds])
        return self._flatten(product)

    def build_graph(self) -> tf.keras.Model:
        """Compute model graph for visualization.

        :return: model.
        """
        user_input = Input(shape=(1,), name='UserID')
        item_input = Input(shape=(1,), name='ItemID')
        return Model(inputs=(user_input, item_input),
                     outputs=self.call((user_input, item_input)),
                     name=self.name)


class RecommenderGDBiased(RecommenderGD):
    """Collaborative filtering recommender using matrix factorization approach
    based on the gradient descent.

    Uses embeddings with users' and items' biases.

    :param embed_size: size of embeddings.
    :param users: unique users.
    :param items: unique items.
    :param random_seed: random seed.
    :param l2_penalty: penalty value for L2 regularization.
    :param kwargs: additional arguments to tensorflow.keras.Model.
    """

    def __init__(self, embed_size: int, users: Sequence, items: Sequence,
                 random_seed: Optional[int] = None,
                 l2_penalty: Optional[float] = None, **kwargs) -> None:
        super().__init__(embed_size, users, items, random_seed,
                         l2_penalty, **kwargs)

        # Additional layers
        self._user_bias = self._get_embed(len(users), 1, 'UserBias')
        self._item_bias = self._get_embed(len(items), 1, 'ItemBias')
        self._component_sum = Add(name='ComponentSum')
        self._global_bias = BiasLayer(name='GlobalBias')

    def call(self, inputs: Tuple[Sequence, Sequence], **kwargs) -> tf.Tensor:
        """Call the model.

        :param inputs: model inputs as user_id and item_id.
        :return: model output.
        """
        user_ids, item_ids = inputs
        user_indices = self._user_lookup(user_ids)
        user_embed = self._user_embedding(user_indices)
        item_indices = self._item_lookup(item_ids)
        item_embed = self._item_embedding(item_indices)
        product = self._dot_product([user_embed, item_embed])
        user_bias = self._user_bias(user_indices)
        item_bias = self._item_bias(item_indices)
        component_sum = self._component_sum([product, user_bias, item_bias])
        biased_rating = self._global_bias(component_sum)
        return self._flatten(biased_rating)


class BiasLayer(tf.keras.layers.Layer):
    """Layer that adds a value to the input."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bias = self.add_weight('bias', shape=(1, 1),
                                    initializer='random_uniform',
                                    trainable=True)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Call the layer.

        :param inputs: layer inputs.
        :return: layer output.
        """
        return inputs + self.bias
