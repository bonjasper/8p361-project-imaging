"�;
uHostFlushSummaryWriter"FlushSummaryWriter(1�����q�@9�����q�@A�����q�@I�����q�@a�@Rd+�?i�@Rd+�?�Unknown�
BHostIDLE"IDLE1     ғ@A     ғ@ay�0��?i~�x��}�?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(133333ci@933333ci@A33333ci@I33333ci@a��R�?i�>��?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1�����\b@9�����\b@A�����\b@I�����\b@a���%�מ?i�6s���?�Unknown
^HostGatherV2"GatherV2(1      U@9      U@A      U@I      U@aV"Jį��?i�gX��W�?�Unknown
�HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1�����,S@9�����,S@A�����,S@I�����,S@a�8t�?i����?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(133333�L@933333�L@A33333�L@I33333�L@aD����O�?i����9�?�Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1����̌B@9����̌B@A����̌B@I����̌B@a�L�J+(?i��H�+x�?�Unknown
o	HostSoftmax"sequential/dense_1/Softmax(133333sB@933333sB@A33333sB@I33333sB@a���+�~?i��8&��?�Unknown
d
HostDataset"Iterator::Model(1�����D@9�����D@A������:@I������:@a�\6�v?iX�f�)��?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1333333>@9333333>@Affffff:@Iffffff:@a����+v?i��E��?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1�����L:@9�����L:@A�����L:@I�����L:@aBw�Cv?i�x�ͭ;�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      2@9      2@A      2@I      2@a�x�ͭ;n?i8�{�Y�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1������/@9������/@A������/@I������/@aHf�d��j?i}��2�t�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1������*@9������*@A������*@I������*@a�D�Vf?i�6���?�Unknown
eHost
LogicalAnd"
LogicalAnd(1������(@9������(@A������(@I������(@a�� �Ǩd?iQ#����?�Unknown�
iHostWriteSummary"WriteSummary(1������&@9������&@A������&@I������&@a!�=��%c?i�`��ò�?�Unknown�
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      .@9      .@A333333&@I333333&@a���Oͤb?i�RLWh��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1������%@9������%@A������%@I������%@a S�9�Nb?i��%���?�Unknown
ZHostArgMax"ArgMax(1ffffff%@9ffffff%@Affffff%@Iffffff%@aY׌#��a?iҞ�����?�Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1������$@9������$@A������$@I������$@a�'��La?i�Ơ����?�Unknown
�HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1      "@9      "@A      "@I      "@a�x�ͭ;^?in���
�?�Unknown
`HostGatherV2"
GatherV2_1(1������ @9������ @A������ @I������ @a^2��[?izՠv�?�Unknown
|HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1ffffff @9ffffff @Affffff @Iffffff @ah�+��[?iH�.Q�%�?�Unknown
gHostStridedSlice"strided_slice(1ffffff @9ffffff @Affffff @Iffffff @ah�+��[?i�+�3�?�Unknown
�HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1������@9������@A������@I������@au(�ٷ�Z?i*˩�@�?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a(1/���Y?i�b���M�?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(133333�J@933333�J@Affffff@Iffffff@a����+V?i��\��X�?�Unknown
bHostDivNoNan"div_no_nan_1(1333333@9333333@A333333@I333333@a	nlv�)U?i���uc�?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a	nlv�)U?i!Qӌ
n�?�Unknown
�HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@a�p�{S?i#	Wr�w�?�Unknown
l HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@a!�=��%S?i�OX[��?�Unknown
V!HostSum"Sum_2(1������@9������@A������@I������@a!�=��%S?i�FH>��?�Unknown
}"HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1333333@9333333@A333333@I333333@a��s���Q?ix�&Փ�?�Unknown
�#HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a�&�U��P?i�n?;��?�Unknown
\$HostArgMax"ArgMax_1(1������@9������@A������@I������@a목?�uP?i`C��u��?�Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_1(1      @9      @A      @I      @a�x�ͭ;N?i>�R���?�Unknown
�&HostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a�x�ͭ;N?i#�ϓ��?�Unknown
�'HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@aάaø3J?i���� ��?�Unknown
�(HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff@9ffffff�?Affffff@Iffffff�?a������I?i�������?�Unknown
�)HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1333333@9333333@A333333@I333333@aK�h���F?i��8��?�Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_3(1������@9������@A������@I������@a-���#B?io������?�Unknown
{+HostSum"*categorical_crossentropy/weighted_loss/Sum(1������@9������@A������@I������@a-���#B?i�gm�J��?�Unknown
V,HostCast"Cast(1������@9������@A������@I������@a�A��wA?i;��x���?�Unknown
t-HostAssignAddVariableOp"AssignAddVariableOp(1������@9������@A������@I������@aR�Z���=?i�#�nZ��?�Unknown
X.HostEqual"Equal(1������@9������@A������@I������@aR�Z���=?i�Nvd��?�Unknown
�/HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1�����1@9�����1@A������ @I������ @a���H�7<?i�`�Z���?�Unknown
X0HostCast"Cast_1(1       @9       @A       @I       @a����:?i�Y�Q���?�Unknown
X1HostCast"Cast_2(1       @9       @A       @I       @a����:?i�R�HK��?�Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_4(1333333�?9333333�?A333333�?I333333�?aK�h���6?i���@&��?�Unknown
T3HostMul"Mul(1�������?9�������?A�������?I�������?a�鞌�5?i���9���?�Unknown
�4HostDivNoNan",categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a�鞌�5?il';2���?�Unknown
s5HostReadVariableOp"SGD/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a�A��w1?i�oK,���?�Unknown
w6HostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a�A��w1?i��[&���?�Unknown
`7HostDivNoNan"
div_no_nan(1333333�?9333333�?A333333�?I333333�?aD/w)�0?i��� ���?�Unknown
a8HostIdentity"Identity(1�������?9�������?A�������?I�������?aR�Z���-?iB�����?�Unknown�
�9HostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1�������?9�������?A�������?I�������?aR�Z���-?i�����?�Unknown
y:HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?a����*?iTDH��?�Unknown
�;HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a��2>�/(?i��'���?�Unknown
u<HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1�������?9�������?A�������?I�������?a�鞌�%?io��
#��?�Unknown
u=HostReadVariableOp"div_no_nan/ReadVariableOp(1�������?9�������?A�������?I�������?a�鞌�%?i^��{��?�Unknown
�>HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�鞌�%?iMO2���?�Unknown
w?HostReadVariableOp"div_no_nan/ReadVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?ay���"?i�������?�Unknown*�;
uHostFlushSummaryWriter"FlushSummaryWriter(1�����q�@9�����q�@A�����q�@I�����q�@aP�ba4�?iP�ba4�?�Unknown�
yHostMatMul"%gradient_tape/sequential/dense/MatMul(133333ci@933333ci@A33333ci@I33333ci@a(/��Ь?iB�x3k�?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1�����\b@9�����\b@A�����\b@I�����\b@a��slפ?i/]���N�?�Unknown
^HostGatherV2"GatherV2(1      U@9      U@A      U@I      U@aqe��?iÅUò�?�Unknown
�HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1�����,S@9�����,S@A�����,S@I�����,S@a�K�Õ?i��/�μ�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(133333�L@933333�L@A33333�L@I33333�L@aZ�WJ�m�?i����=@�?�Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1����̌B@9����̌B@A����̌B@I����̌B@a�[�^��?i���[u��?�Unknown
oHostSoftmax"sequential/dense_1/Softmax(133333sB@933333sB@A33333sB@I33333sB@a2<���?ic�
�8��?�Unknown
d	HostDataset"Iterator::Model(1�����D@9�����D@A������:@I������:@a��e�k~?i���%�?�Unknown
�
HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1333333>@9333333>@Affffff:@Iffffff:@a��u��}?iź�l�`�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1�����L:@9�����L:@A�����L:@I�����L:@a%�l��}?i��쯜�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      2@9      2@A      2@I      2@a��gnt?i������?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1������/@9������/@A������/@I������/@a_s���r?i�ԃ���?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1������*@9������*@A������*@I������*@a\X5,�0n?i�	����?�Unknown
eHost
LogicalAnd"
LogicalAnd(1������(@9������(@A������(@I������(@a�!P��k?i Z���#�?�Unknown�
iHostWriteSummary"WriteSummary(1������&@9������&@A������&@I������&@aOp����i?i��Gt�=�?�Unknown�
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      .@9      .@A333333&@I333333&@a��	ns2i?ip����V�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1������%@9������%@A������%@I������%@a�Ԩ�9�h?iE��!�o�?�Unknown
ZHostArgMax"ArgMax(1ffffff%@9ffffff%@Affffff%@Iffffff%@a��G� Jh?i�F"܇�?�Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1������$@9������$@A������$@I������$@a����ag?i�u��=��?�Unknown
�HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1      "@9      "@A      "@I      "@a��gnd?i��bȫ��?�Unknown
`HostGatherV2"
GatherV2_1(1������ @9������ @A������ @I������ @a�F�dO�b?i�?����?�Unknown
|HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1ffffff @9ffffff @Affffff @Iffffff @a���2�b?i��tJ ��?�Unknown
gHostStridedSlice"strided_slice(1ffffff @9ffffff @Affffff @Iffffff @a���2�b?iyU"}���?�Unknown
�HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1������@9������@A������@I������@a�0����a?i�N�Y���?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a�%��za?i����&�?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(133333�J@933333�J@Affffff@Iffffff@a��u��]?i9i�c"�?�Unknown
bHostDivNoNan"div_no_nan_1(1333333@9333333@A333333@I333333@a��)"�\?iZuo,�?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a��)"�\?i�J(��:�?�Unknown
�HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@aB{�UZ?i)I��G�?�Unknown
lHostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@aOp����Y?i�sl�T�?�Unknown
V HostSum"Sum_2(1������@9������@A������@I������@aOp����Y?i��<��a�?�Unknown
}!HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1333333@9333333@A333333@I333333@aD��X?i;�)��m�?�Unknown
�"HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a�#�7�V?iMj�^)y�?�Unknown
\#HostArgMax"ArgMax_1(1������@9������@A������@I������@a�� �>V?iٳ�H��?�Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_1(1      @9      @A      @I      @a��gnT?iO;5���?�Unknown
�%HostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a��gnT?i��h����?�Unknown
�&HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a(��п�Q?i'QV���?�Unknown
�'HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff@9ffffff�?Affffff@Iffffff�?a4�gb�@Q?i�Z��1��?�Unknown
�(HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1333333@9333333@A333333@I333333@a���QA�N?i���i��?�Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_3(1������@9������@A������@I������@asOxH�H?i��(q
��?�Unknown
{*HostSum"*categorical_crossentropy/weighted_loss/Sum(1������@9������@A������@I������@asOxH�H?i�{x+��?�Unknown
V+HostCast"Cast(1������@9������@A������@I������@a�9�k��G?i[�c��?�Unknown
t,HostAssignAddVariableOp"AssignAddVariableOp(1������@9������@A������@I������@a�����C?i�!����?�Unknown
X-HostEqual"Equal(1������@9������@A������@I������@a�����C?iKM�R��?�Unknown
�.HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1�����1@9�����1@A������ @I������ @a��lC?i>H�����?�Unknown
X/HostCast"Cast_1(1       @9       @A       @I       @a�)?�(B?i���]��?�Unknown
X0HostCast"Cast_2(1       @9       @A       @I       @a�)?�(B?i�8*���?�Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_4(1333333�?9333333�?A333333�?I333333�?a���QA�>?i�c���?�Unknown
T2HostMul"Mul(1�������?9�������?A�������?I�������?a��B�[=?iO��e��?�Unknown
�3HostDivNoNan",categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a��B�[=?i�&I���?�Unknown
s4HostReadVariableOp"SGD/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a�9�k��7?in�����?�Unknown
w5HostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a�9�k��7?i5����?�Unknown
`6HostDivNoNan"
div_no_nan(1333333�?9333333�?A333333�?I333333�?a�2���5?iwZz���?�Unknown
a7HostIdentity"Identity(1�������?9�������?A�������?I�������?a�����3?i3pY('��?�Unknown�
�8HostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1�������?9�������?A�������?I�������?a�����3?i�8d���?�Unknown
y9HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?a�)?�(2?i&k`����?�Unknown
�:HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?aM���X0?i�х���?�Unknown
u;HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1�������?9�������?A�������?I�������?a��B�[-?i��k���?�Unknown
u<HostReadVariableOp"div_no_nan/ReadVariableOp(1�������?9�������?A�������?I�������?a��B�[-?i/(DQ���?�Unknown
�=HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a��B�[-?i[��6i��?�Unknown
w>HostReadVariableOp"div_no_nan/ReadVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?aZe:%�l)?i     �?�Unknown2CPU