from simplelearn.nodes import Linear
from datasets import Mnist

def main():
    training_set = Mnist()

    chain = [training_set.data_node]
    for output_size in (500, 500):
        chain.append(make_rectified_linear(output_size, input=chain[-1]))

    chain.append(Softmax(output_size=10, input=chain[-1]))

    loss = CrossEntropy(model=chain[-1],
                        label=training_set.label_node)

    input_symbols = [x.get_output_symbol() for x in (training_set.data_node,
                                                     training_set.label_node)]

    model_nodes = chain[1:]
    model_gradient_funcs = []
    for layer in chain[1:]:
        # Guessed this call signature for theano.gradient. Look it up.
        model_gradient_funcs.append(theano.gradient(
            input_symbols,  # func inputs
            loss.get_output_symbol(),  # func output
            layer.weights))  # gradient of func output wrt this

    for data, labels in training_set.iterator(batch_size=batch_size):
        for model_node, model_gradient_func in safe_izip(model_nodes,
                                                         model_gradient_funcs):



if __name__ == '__main__':
    main()
