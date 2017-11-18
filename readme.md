# LINE in TensorFlow

TensorFlow implementation of paper _LINE: Large-scale Information Network Embedding_ by Jian Tang, et al.

You can see [my slide](Network_Embedding_with_TensorFlow.pdf) on GDG DevFest 2017 for more detail about LINE and TensorFlow. Notice: code shown in the slide are pseudocode, minibatch and negative sampling are omitted in the slide. 

## Prerequisites

* Python 3.6
* TensorFlow 1.3.0
* Networkx
* NumPy

## Setup

* Prepare a network using networkx. Write the graph to a file by [nx.write_gpickle](https://networkx.github.io/documentation/stable/reference/readwrite/generated/networkx.readwrite.gpickle.write_gpickle.html).
* Put the network file in `data` folder.
* Run `line.py --graph_file graph.pkl` to start training. `graph.pkl` is the name of your network file.
* Embedding will be stored in `data/embedding_XXX-order.pkl`. You can load it by `pickle.load()` in python.

## References

- Tang, Jian, et al. "[Line: Large-scale information network embedding.](https://dl.acm.org/citation.cfm?id=2741093)" _Proceedings of the 24th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee_, 2015.
