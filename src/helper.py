import logging
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

from tfsom import SelfOrganizingMap
from umatrix import get_umatrix_optimized


def create_som(input_data, dims, num_inputs, batch_size=128, num_rows_neurons=20, num_cols_neurons=20, max_epochs=100,
               weights_init=None):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    graph = tf.Graph()
    with graph.as_default():
        # Make sure you allow_soft_placement, some ops have to be put on the CPU (e.g. summary operations)
        session = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False))

        # scale the data
        scaler = StandardScaler()

        # Old version used index 0 because the output is a (data, label) tuple
        #input_data = scaler.fit_transform(input_data[0])

        # New version assumes you'll only be inputting the data to this function
        input_data = scaler.fit_transform(input_data) # CRB data is in matrix form?

        # Build the TensorFlow dataset pipeline per the standard tutorial.
        dataset = tf.data.Dataset.from_tensor_slices(input_data.astype(np.float32))
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        next_element = iterator.get_next()

        # Build the SOM object and place all of its ops on the graph
        som = SelfOrganizingMap(
            num_rows_neurons=num_rows_neurons,
            num_cols_neurons=num_cols_neurons,
            dim=dims,
            max_epochs=max_epochs,
            gpus=1,
            session=session,
            graph=graph,
            input_tensor=next_element,
            batch_size=batch_size,
            initial_learning_rate=0.1,
            input_dataset=input_data,
            weights_init=weights_init
        )

        init_op = tf.compat.v1.global_variables_initializer()
        session.run([init_op])

        # Note that I don't pass a SummaryWriter because I don't really want to record summaries in this script
        # If you want Tensorboard support just make a new SummaryWriter and pass it to this method
        som.train(num_inputs=num_inputs)

        print("Final QE={}", som.quantization_error(tf.constant(input_data, dtype=tf.float32)))
        print("Final TE={}", som.topographic_error(tf.constant(input_data, dtype=tf.float32)))

        weights = som.output_weights

        umatrix, bmu_loc = get_umatrix_optimized(som, input_data, weights, num_rows_neurons, num_cols_neurons)

    return som, umatrix, bmu_loc
