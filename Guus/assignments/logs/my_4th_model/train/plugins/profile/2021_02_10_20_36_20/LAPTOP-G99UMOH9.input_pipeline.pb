	��z6��?��z6��?!��z6��?	���O��@���O��@!���O��@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��z6��?�	��?An4��@��?Y�J�4�?*	     �R@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԛ?!>�S�A@)�I+��?1�L�Ϻ=@:Preprocessing2F
Iterator::Model�D���J�?!S�n�@@)-C��6�?1���L1@:Preprocessing2U
Iterator::Model::ParallelMapV2tF��_�?!L�Ϻ�0@)tF��_�?1L�Ϻ�0@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�� �rh�?!�Ϻ��&@)�� �rh�?1�Ϻ��&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateL7�A`�?!���L6@)����Mb�?1�Y7�"�%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��b�=�?!|��ȧP@)��ZӼ�t?1��L�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4q?!?�S�@)�J�4q?1?�S�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap;�O��n�?!1E>�S8@)�~j�t�X?1v�)�Y7 @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t17.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9���O��@Iu&��W@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�	��?�	��?!�	��?      ��!       "      ��!       *      ��!       2	n4��@��?n4��@��?!n4��@��?:      ��!       B      ��!       J	�J�4�?�J�4�?!�J�4�?R      ��!       Z	�J�4�?�J�4�?!�J�4�?b      ��!       JCPU_ONLYY���O��@b qu&��W@