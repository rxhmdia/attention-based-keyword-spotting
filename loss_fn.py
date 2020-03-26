import tensorflow as tf


def _cross_entropy_loss(logits, labels):
    return tf.reduce_mean(
        -tf.reduce_sum(labels * tf.log(logits), reduction_indices=[1]))


def _focal_loss(logits, labels, alpha=2):
    return tf.reduce_mean(-tf.reduce_sum(tf.pow(1 - logits, 2) * labels *
                                         tf.log(logits),
                                         reduction_indices=[1]))


# kl_divergence_loss is not extract kl_divergence_loss
def _kl_div_loss(logits, labels):
    return tf.reduce_mean(
        -tf.reduce_sum(labels * tf.log(logits), reduction_indices=[1]))


loss_map = {
    "cross_entropy": _cross_entropy_loss,
    "focal_loss": _focal_loss,
    "kl_loss": _kl_div_loss,
}


def loss_func(loss_name, logits, labels):
    if loss_name not in loss_map:
        raise ValueError("{} is not supported.".format(loss_name))
    return loss_map[loss_name](logits, labels)
