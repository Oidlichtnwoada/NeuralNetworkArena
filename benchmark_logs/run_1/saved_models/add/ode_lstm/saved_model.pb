��%
��
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.4.12v2.4.0-49-g85c8b2a817f8��#
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
rnn/odelstm/input_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_namernn/odelstm/input_kernel
�
,rnn/odelstm/input_kernel/Read/ReadVariableOpReadVariableOprnn/odelstm/input_kernel*
_output_shapes
:	�*
dtype0
�
rnn/odelstm/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*-
shared_namernn/odelstm/recurrent_kernel
�
0rnn/odelstm/recurrent_kernel/Read/ReadVariableOpReadVariableOprnn/odelstm/recurrent_kernel*
_output_shapes
:	@�*
dtype0
y
rnn/odelstm/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_namernn/odelstm/bias
r
$rnn/odelstm/bias/Read/ReadVariableOpReadVariableOprnn/odelstm/bias*
_output_shapes	
:�*
dtype0
�
rnn/odelstm/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*#
shared_namernn/odelstm/kernel
y
&rnn/odelstm/kernel/Read/ReadVariableOpReadVariableOprnn/odelstm/kernel*
_output_shapes

:@@*
dtype0
�
rnn/odelstm/recurrent_kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*/
shared_name rnn/odelstm/recurrent_kernel_1
�
2rnn/odelstm/recurrent_kernel_1/Read/ReadVariableOpReadVariableOprnn/odelstm/recurrent_kernel_1*
_output_shapes

:@@*
dtype0
|
rnn/odelstm/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namernn/odelstm/bias_1
u
&rnn/odelstm/bias_1/Read/ReadVariableOpReadVariableOprnn/odelstm/bias_1*
_output_shapes
:@*
dtype0
z
rnn/odelstm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namernn/odelstm/scale
s
%rnn/odelstm/scale/Read/ReadVariableOpReadVariableOprnn/odelstm/scale*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
�
Adam/rnn/odelstm/input_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*0
shared_name!Adam/rnn/odelstm/input_kernel/m
�
3Adam/rnn/odelstm/input_kernel/m/Read/ReadVariableOpReadVariableOpAdam/rnn/odelstm/input_kernel/m*
_output_shapes
:	�*
dtype0
�
#Adam/rnn/odelstm/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*4
shared_name%#Adam/rnn/odelstm/recurrent_kernel/m
�
7Adam/rnn/odelstm/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp#Adam/rnn/odelstm/recurrent_kernel/m*
_output_shapes
:	@�*
dtype0
�
Adam/rnn/odelstm/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameAdam/rnn/odelstm/bias/m
�
+Adam/rnn/odelstm/bias/m/Read/ReadVariableOpReadVariableOpAdam/rnn/odelstm/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/rnn/odelstm/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@**
shared_nameAdam/rnn/odelstm/kernel/m
�
-Adam/rnn/odelstm/kernel/m/Read/ReadVariableOpReadVariableOpAdam/rnn/odelstm/kernel/m*
_output_shapes

:@@*
dtype0
�
%Adam/rnn/odelstm/recurrent_kernel/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*6
shared_name'%Adam/rnn/odelstm/recurrent_kernel/m_1
�
9Adam/rnn/odelstm/recurrent_kernel/m_1/Read/ReadVariableOpReadVariableOp%Adam/rnn/odelstm/recurrent_kernel/m_1*
_output_shapes

:@@*
dtype0
�
Adam/rnn/odelstm/bias/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/rnn/odelstm/bias/m_1
�
-Adam/rnn/odelstm/bias/m_1/Read/ReadVariableOpReadVariableOpAdam/rnn/odelstm/bias/m_1*
_output_shapes
:@*
dtype0
�
Adam/rnn/odelstm/scale/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/rnn/odelstm/scale/m
�
,Adam/rnn/odelstm/scale/m/Read/ReadVariableOpReadVariableOpAdam/rnn/odelstm/scale/m*
_output_shapes
:@*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
�
Adam/rnn/odelstm/input_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*0
shared_name!Adam/rnn/odelstm/input_kernel/v
�
3Adam/rnn/odelstm/input_kernel/v/Read/ReadVariableOpReadVariableOpAdam/rnn/odelstm/input_kernel/v*
_output_shapes
:	�*
dtype0
�
#Adam/rnn/odelstm/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*4
shared_name%#Adam/rnn/odelstm/recurrent_kernel/v
�
7Adam/rnn/odelstm/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp#Adam/rnn/odelstm/recurrent_kernel/v*
_output_shapes
:	@�*
dtype0
�
Adam/rnn/odelstm/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameAdam/rnn/odelstm/bias/v
�
+Adam/rnn/odelstm/bias/v/Read/ReadVariableOpReadVariableOpAdam/rnn/odelstm/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/rnn/odelstm/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@**
shared_nameAdam/rnn/odelstm/kernel/v
�
-Adam/rnn/odelstm/kernel/v/Read/ReadVariableOpReadVariableOpAdam/rnn/odelstm/kernel/v*
_output_shapes

:@@*
dtype0
�
%Adam/rnn/odelstm/recurrent_kernel/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*6
shared_name'%Adam/rnn/odelstm/recurrent_kernel/v_1
�
9Adam/rnn/odelstm/recurrent_kernel/v_1/Read/ReadVariableOpReadVariableOp%Adam/rnn/odelstm/recurrent_kernel/v_1*
_output_shapes

:@@*
dtype0
�
Adam/rnn/odelstm/bias/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/rnn/odelstm/bias/v_1
�
-Adam/rnn/odelstm/bias/v_1/Read/ReadVariableOpReadVariableOpAdam/rnn/odelstm/bias/v_1*
_output_shapes
:@*
dtype0
�
Adam/rnn/odelstm/scale/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/rnn/odelstm/scale/v
�
,Adam/rnn/odelstm/scale/v/Read/ReadVariableOpReadVariableOpAdam/rnn/odelstm/scale/v*
_output_shapes
:@*
dtype0

NoOpNoOp
�/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�.
value�.B�. B�.
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
 
l
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
�
iter

beta_1

beta_2
	decay
learning_ratemLmMmNmOmPmQ mR!mS"mTvUvVvWvXvYvZ v[!v\"v]
?
0
1
2
3
 4
!5
"6
7
8
 
?
0
1
2
3
 4
!5
"6
7
8
�
#metrics
	variables
$layer_regularization_losses
%layer_metrics

&layers
'non_trainable_variables
regularization_losses
trainable_variables
 
�
	(ctrnn
input_kernel
recurrent_kernel
bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
 
1
0
1
2
3
 4
!5
"6
 
1
0
1
2
3
 4
!5
"6
�
-metrics
.layer_regularization_losses
/layer_metrics

0layers
	variables
1non_trainable_variables
regularization_losses
trainable_variables

2states
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
3metrics
4layer_regularization_losses
5layer_metrics

6layers
	variables
7non_trainable_variables
regularization_losses
trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUErnn/odelstm/input_kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUErnn/odelstm/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUErnn/odelstm/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUErnn/odelstm/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUErnn/odelstm/recurrent_kernel_1&variables/4/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUErnn/odelstm/bias_1&variables/5/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUErnn/odelstm/scale&variables/6/.ATTRIBUTES/VARIABLE_VALUE

80
 
 

0
1
2
3
 
�
9fixed_step_methods

kernel
 recurrent_kernel
!bias
	"scale
:	variables
;regularization_losses
<trainable_variables
=	keras_api
1
0
1
2
3
 4
!5
"6
 
1
0
1
2
3
 4
!5
"6
�
>metrics
?layer_regularization_losses
@layer_metrics

Alayers
)	variables
Bnon_trainable_variables
*regularization_losses
+trainable_variables
 
 
 

0
 
 
 
 
 
 
 
4
	Ctotal
	Dcount
E	variables
F	keras_api
 

0
 1
!2
"3
 

0
 1
!2
"3
�
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics

Jlayers
:	variables
Knon_trainable_variables
;regularization_losses
<trainable_variables
 
 
 

(0
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

E	variables
 
 
 
 
 
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/rnn/odelstm/input_kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/rnn/odelstm/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/rnn/odelstm/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/rnn/odelstm/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%Adam/rnn/odelstm/recurrent_kernel/m_1Bvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/rnn/odelstm/bias/m_1Bvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/rnn/odelstm/scale/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/rnn/odelstm/input_kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/rnn/odelstm/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/rnn/odelstm/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/rnn/odelstm/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%Adam/rnn/odelstm/recurrent_kernel/v_1Bvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/rnn/odelstm/bias/v_1Bvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/rnn/odelstm/scale/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
r
serving_default_input_1Placeholder*#
_output_shapes
:�d*
dtype0*
shape:�d
r
serving_default_input_2Placeholder*#
_output_shapes
:�d*
dtype0*
shape:�d
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2rnn/odelstm/input_kernelrnn/odelstm/recurrent_kernelrnn/odelstm/biasrnn/odelstm/kernelrnn/odelstm/recurrent_kernel_1rnn/odelstm/bias_1rnn/odelstm/scaledense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_59645
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp,rnn/odelstm/input_kernel/Read/ReadVariableOp0rnn/odelstm/recurrent_kernel/Read/ReadVariableOp$rnn/odelstm/bias/Read/ReadVariableOp&rnn/odelstm/kernel/Read/ReadVariableOp2rnn/odelstm/recurrent_kernel_1/Read/ReadVariableOp&rnn/odelstm/bias_1/Read/ReadVariableOp%rnn/odelstm/scale/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp3Adam/rnn/odelstm/input_kernel/m/Read/ReadVariableOp7Adam/rnn/odelstm/recurrent_kernel/m/Read/ReadVariableOp+Adam/rnn/odelstm/bias/m/Read/ReadVariableOp-Adam/rnn/odelstm/kernel/m/Read/ReadVariableOp9Adam/rnn/odelstm/recurrent_kernel/m_1/Read/ReadVariableOp-Adam/rnn/odelstm/bias/m_1/Read/ReadVariableOp,Adam/rnn/odelstm/scale/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp3Adam/rnn/odelstm/input_kernel/v/Read/ReadVariableOp7Adam/rnn/odelstm/recurrent_kernel/v/Read/ReadVariableOp+Adam/rnn/odelstm/bias/v/Read/ReadVariableOp-Adam/rnn/odelstm/kernel/v/Read/ReadVariableOp9Adam/rnn/odelstm/recurrent_kernel/v_1/Read/ReadVariableOp-Adam/rnn/odelstm/bias/v_1/Read/ReadVariableOp,Adam/rnn/odelstm/scale/v/Read/ReadVariableOpConst*/
Tin(
&2$	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_61856
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_raternn/odelstm/input_kernelrnn/odelstm/recurrent_kernelrnn/odelstm/biasrnn/odelstm/kernelrnn/odelstm/recurrent_kernel_1rnn/odelstm/bias_1rnn/odelstm/scaletotalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/rnn/odelstm/input_kernel/m#Adam/rnn/odelstm/recurrent_kernel/mAdam/rnn/odelstm/bias/mAdam/rnn/odelstm/kernel/m%Adam/rnn/odelstm/recurrent_kernel/m_1Adam/rnn/odelstm/bias/m_1Adam/rnn/odelstm/scale/mAdam/dense/kernel/vAdam/dense/bias/vAdam/rnn/odelstm/input_kernel/v#Adam/rnn/odelstm/recurrent_kernel/vAdam/rnn/odelstm/bias/vAdam/rnn/odelstm/kernel/v%Adam/rnn/odelstm/recurrent_kernel/v_1Adam/rnn/odelstm/bias/v_1Adam/rnn/odelstm/scale/v*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_61968��"
��
�

C__inference_ode_lstm_layer_call_and_return_conditional_losses_60259
inputs_0
inputs_1.
*rnn_odelstm_matmul_readvariableop_resource0
,rnn_odelstm_matmul_1_readvariableop_resource-
)rnn_odelstm_add_1_readvariableop_resource0
,rnn_odelstm_matmul_2_readvariableop_resource0
,rnn_odelstm_matmul_3_readvariableop_resource-
)rnn_odelstm_add_5_readvariableop_resource'
#rnn_odelstm_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�!rnn/odelstm/MatMul/ReadVariableOp�#rnn/odelstm/MatMul_1/ReadVariableOp�#rnn/odelstm/MatMul_2/ReadVariableOp�#rnn/odelstm/MatMul_3/ReadVariableOp�#rnn/odelstm/MatMul_4/ReadVariableOp�#rnn/odelstm/MatMul_5/ReadVariableOp�#rnn/odelstm/MatMul_6/ReadVariableOp�#rnn/odelstm/MatMul_7/ReadVariableOp�#rnn/odelstm/MatMul_8/ReadVariableOp�#rnn/odelstm/MatMul_9/ReadVariableOp�rnn/odelstm/ReadVariableOp�rnn/odelstm/ReadVariableOp_1�rnn/odelstm/ReadVariableOp_2�rnn/odelstm/ReadVariableOp_3� rnn/odelstm/add_1/ReadVariableOp�!rnn/odelstm/add_11/ReadVariableOp�!rnn/odelstm/add_14/ReadVariableOp� rnn/odelstm/add_5/ReadVariableOp� rnn/odelstm/add_8/ReadVariableOp�	rnn/whilek
	rnn/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"�   d      2
	rnn/Shape|
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice/stack�
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_1�
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_2�
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_sliced
rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
rnn/zeros/mul/y|
rnn/zeros/mulMulrnn/strided_slice:output:0rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/mulg
rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
rnn/zeros/Less/yw
rnn/zeros/LessLessrnn/zeros/mul:z:0rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/Lessj
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
rnn/zeros/packed/1�
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros/packedg
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros/Const}
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*
_output_shapes
:	�@2
	rnn/zerosh
rnn/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
rnn/zeros_1/mul/y�
rnn/zeros_1/mulMulrnn/strided_slice:output:0rnn/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/mulk
rnn/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
rnn/zeros_1/Less/y
rnn/zeros_1/LessLessrnn/zeros_1/mul:z:0rnn/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/Lessn
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
rnn/zeros_1/packed/1�
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros_1/packedk
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros_1/Const�
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*
_output_shapes
:	�@2
rnn/zeros_1}
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose/perm�
rnn/transpose	Transposeinputs_0rnn/transpose/perm:output:0*
T0*#
_output_shapes
:d�2
rnn/transposeo
rnn/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   �      2
rnn/Shape_1�
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_1/stack�
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_1�
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_2�
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_slice_1�
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
rnn/TensorArrayV2/element_shape�
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2�
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�      2;
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shape�
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+rnn/TensorArrayUnstack/TensorListFromTensor�
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_2/stack�
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_1�
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_2�
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*
shrink_axis_mask2
rnn/strided_slice_2�
!rnn/odelstm/MatMul/ReadVariableOpReadVariableOp*rnn_odelstm_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!rnn/odelstm/MatMul/ReadVariableOp�
rnn/odelstm/MatMulMatMulrnn/strided_slice_2:output:0)rnn/odelstm/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
rnn/odelstm/MatMul�
#rnn/odelstm/MatMul_1/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02%
#rnn/odelstm/MatMul_1/ReadVariableOp�
rnn/odelstm/MatMul_1MatMulrnn/zeros_1:output:0+rnn/odelstm/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
rnn/odelstm/MatMul_1�
rnn/odelstm/addAddV2rnn/odelstm/MatMul:product:0rnn/odelstm/MatMul_1:product:0*
T0* 
_output_shapes
:
��2
rnn/odelstm/add�
 rnn/odelstm/add_1/ReadVariableOpReadVariableOp)rnn_odelstm_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 rnn/odelstm/add_1/ReadVariableOp�
rnn/odelstm/add_1AddV2rnn/odelstm/add:z:0(rnn/odelstm/add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
rnn/odelstm/add_1h
rnn/odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/odelstm/Const�
rnn/odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
rnn/odelstm/split/split_dim�
rnn/odelstm/splitSplit$rnn/odelstm/split/split_dim:output:0rnn/odelstm/add_1:z:0*
T0*@
_output_shapes.
,:	�@:	�@:	�@:	�@*
	num_split2
rnn/odelstm/splitr
rnn/odelstm/TanhTanhrnn/odelstm/split:output:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/Tanh{
rnn/odelstm/SigmoidSigmoidrnn/odelstm/split:output:1*
T0*
_output_shapes
:	�@2
rnn/odelstm/Sigmoido
rnn/odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
rnn/odelstm/add_2/y�
rnn/odelstm/add_2AddV2rnn/odelstm/split:output:2rnn/odelstm/add_2/y:output:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_2z
rnn/odelstm/Sigmoid_1Sigmoidrnn/odelstm/add_2:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/Sigmoid_1
rnn/odelstm/Sigmoid_2Sigmoidrnn/odelstm/split:output:3*
T0*
_output_shapes
:	�@2
rnn/odelstm/Sigmoid_2�
rnn/odelstm/mulMulrnn/zeros:output:0rnn/odelstm/Sigmoid_1:y:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul�
rnn/odelstm/mul_1Mulrnn/odelstm/Tanh:y:0rnn/odelstm/Sigmoid:y:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_1�
rnn/odelstm/add_3AddV2rnn/odelstm/mul:z:0rnn/odelstm/mul_1:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_3q
rnn/odelstm/Tanh_1Tanhrnn/odelstm/add_3:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/Tanh_1�
rnn/odelstm/mul_2Mulrnn/odelstm/Tanh_1:y:0rnn/odelstm/Sigmoid_2:y:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_2�
#rnn/odelstm/MatMul_2/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_2/ReadVariableOp�
rnn/odelstm/MatMul_2MatMulrnn/odelstm/mul_2:z:0+rnn/odelstm/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_2�
#rnn/odelstm/MatMul_3/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_3/ReadVariableOp�
rnn/odelstm/MatMul_3MatMulrnn/zeros_1:output:0+rnn/odelstm/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_3�
rnn/odelstm/add_4AddV2rnn/odelstm/MatMul_2:product:0rnn/odelstm/MatMul_3:product:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_4�
 rnn/odelstm/add_5/ReadVariableOpReadVariableOp)rnn_odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02"
 rnn/odelstm/add_5/ReadVariableOp�
rnn/odelstm/add_5AddV2rnn/odelstm/add_4:z:0(rnn/odelstm/add_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_5q
rnn/odelstm/Tanh_2Tanhrnn/odelstm/add_5:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/Tanh_2�
rnn/odelstm/ReadVariableOpReadVariableOp#rnn_odelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
rnn/odelstm/ReadVariableOp�
rnn/odelstm/mul_3Mul"rnn/odelstm/ReadVariableOp:value:0rnn/odelstm/Tanh_2:y:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_3o
rnn/odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/odelstm/mul_4/y�
rnn/odelstm/mul_4Mulrnn/zeros_1:output:0rnn/odelstm/mul_4/y:output:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_4�
rnn/odelstm/subSubrnn/odelstm/mul_3:z:0rnn/odelstm/mul_4:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/subo
rnn/odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/odelstm/mul_5/x�
rnn/odelstm/mul_5Mulrnn/odelstm/mul_5/x:output:0rnn/odelstm/sub:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_5�
rnn/odelstm/add_6AddV2rnn/zeros_1:output:0rnn/odelstm/mul_5:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_6�
#rnn/odelstm/MatMul_4/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_4/ReadVariableOp�
rnn/odelstm/MatMul_4MatMulrnn/odelstm/mul_2:z:0+rnn/odelstm/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_4�
#rnn/odelstm/MatMul_5/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_5/ReadVariableOp�
rnn/odelstm/MatMul_5MatMulrnn/odelstm/add_6:z:0+rnn/odelstm/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_5�
rnn/odelstm/add_7AddV2rnn/odelstm/MatMul_4:product:0rnn/odelstm/MatMul_5:product:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_7�
 rnn/odelstm/add_8/ReadVariableOpReadVariableOp)rnn_odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02"
 rnn/odelstm/add_8/ReadVariableOp�
rnn/odelstm/add_8AddV2rnn/odelstm/add_7:z:0(rnn/odelstm/add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_8q
rnn/odelstm/Tanh_3Tanhrnn/odelstm/add_8:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/Tanh_3�
rnn/odelstm/ReadVariableOp_1ReadVariableOp#rnn_odelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
rnn/odelstm/ReadVariableOp_1�
rnn/odelstm/mul_6Mul$rnn/odelstm/ReadVariableOp_1:value:0rnn/odelstm/Tanh_3:y:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_6o
rnn/odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/odelstm/mul_7/y�
rnn/odelstm/mul_7Mulrnn/odelstm/add_6:z:0rnn/odelstm/mul_7/y:output:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_7�
rnn/odelstm/sub_1Subrnn/odelstm/mul_6:z:0rnn/odelstm/mul_7:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/sub_1o
rnn/odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/odelstm/mul_8/x�
rnn/odelstm/mul_8Mulrnn/odelstm/mul_8/x:output:0rnn/odelstm/sub_1:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_8�
rnn/odelstm/add_9AddV2rnn/odelstm/add_6:z:0rnn/odelstm/mul_8:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_9�
#rnn/odelstm/MatMul_6/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_6/ReadVariableOp�
rnn/odelstm/MatMul_6MatMulrnn/odelstm/mul_2:z:0+rnn/odelstm/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_6�
#rnn/odelstm/MatMul_7/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_7/ReadVariableOp�
rnn/odelstm/MatMul_7MatMulrnn/odelstm/add_9:z:0+rnn/odelstm/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_7�
rnn/odelstm/add_10AddV2rnn/odelstm/MatMul_6:product:0rnn/odelstm/MatMul_7:product:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_10�
!rnn/odelstm/add_11/ReadVariableOpReadVariableOp)rnn_odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02#
!rnn/odelstm/add_11/ReadVariableOp�
rnn/odelstm/add_11AddV2rnn/odelstm/add_10:z:0)rnn/odelstm/add_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_11r
rnn/odelstm/Tanh_4Tanhrnn/odelstm/add_11:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/Tanh_4�
rnn/odelstm/ReadVariableOp_2ReadVariableOp#rnn_odelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
rnn/odelstm/ReadVariableOp_2�
rnn/odelstm/mul_9Mul$rnn/odelstm/ReadVariableOp_2:value:0rnn/odelstm/Tanh_4:y:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_9q
rnn/odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/odelstm/mul_10/y�
rnn/odelstm/mul_10Mulrnn/odelstm/add_9:z:0rnn/odelstm/mul_10/y:output:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_10�
rnn/odelstm/sub_2Subrnn/odelstm/mul_9:z:0rnn/odelstm/mul_10:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/sub_2q
rnn/odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/odelstm/mul_11/x�
rnn/odelstm/mul_11Mulrnn/odelstm/mul_11/x:output:0rnn/odelstm/sub_2:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_11�
rnn/odelstm/add_12AddV2rnn/odelstm/add_9:z:0rnn/odelstm/mul_11:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_12�
#rnn/odelstm/MatMul_8/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_8/ReadVariableOp�
rnn/odelstm/MatMul_8MatMulrnn/odelstm/mul_2:z:0+rnn/odelstm/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_8�
#rnn/odelstm/MatMul_9/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_9/ReadVariableOp�
rnn/odelstm/MatMul_9MatMulrnn/odelstm/add_12:z:0+rnn/odelstm/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_9�
rnn/odelstm/add_13AddV2rnn/odelstm/MatMul_8:product:0rnn/odelstm/MatMul_9:product:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_13�
!rnn/odelstm/add_14/ReadVariableOpReadVariableOp)rnn_odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02#
!rnn/odelstm/add_14/ReadVariableOp�
rnn/odelstm/add_14AddV2rnn/odelstm/add_13:z:0)rnn/odelstm/add_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_14r
rnn/odelstm/Tanh_5Tanhrnn/odelstm/add_14:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/Tanh_5�
rnn/odelstm/ReadVariableOp_3ReadVariableOp#rnn_odelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
rnn/odelstm/ReadVariableOp_3�
rnn/odelstm/mul_12Mul$rnn/odelstm/ReadVariableOp_3:value:0rnn/odelstm/Tanh_5:y:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_12q
rnn/odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/odelstm/mul_13/y�
rnn/odelstm/mul_13Mulrnn/odelstm/add_12:z:0rnn/odelstm/mul_13/y:output:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_13�
rnn/odelstm/sub_3Subrnn/odelstm/mul_12:z:0rnn/odelstm/mul_13:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/sub_3q
rnn/odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/odelstm/mul_14/x�
rnn/odelstm/mul_14Mulrnn/odelstm/mul_14/x:output:0rnn/odelstm/sub_3:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_14�
rnn/odelstm/add_15AddV2rnn/odelstm/add_12:z:0rnn/odelstm/mul_14:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_15�
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�   @   2#
!rnn/TensorArrayV2_1/element_shape�
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2_1V
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn/time�
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
rnn/while/maximum_iterationsr
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/while/loop_counter�
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0*rnn_odelstm_matmul_readvariableop_resource,rnn_odelstm_matmul_1_readvariableop_resource)rnn_odelstm_add_1_readvariableop_resource,rnn_odelstm_matmul_2_readvariableop_resource,rnn_odelstm_matmul_3_readvariableop_resource)rnn_odelstm_add_5_readvariableop_resource#rnn_odelstm_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*D
_output_shapes2
0: : : : :	�@:	�@: : : : : : : : : *)
_read_only_resource_inputs
		
* 
bodyR
rnn_while_body_60091* 
condR
rnn_while_cond_60090*C
output_shapes2
0: : : : :	�@:	�@: : : : : : : : : *
parallel_iterations 2
	rnn/while�
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�   @   26
4rnn/TensorArrayV2Stack/TensorListStack/element_shape�
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:d�@*
element_dtype02(
&rnn/TensorArrayV2Stack/TensorListStack�
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
rnn/strided_slice_3/stack�
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_3/stack_1�
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_3/stack_2�
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�@*
shrink_axis_mask2
rnn/strided_slice_3�
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose_1/perm�
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*#
_output_shapes
:�d@2
rnn/transpose_1�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulrnn/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense/BiasAdd�
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp"^rnn/odelstm/MatMul/ReadVariableOp$^rnn/odelstm/MatMul_1/ReadVariableOp$^rnn/odelstm/MatMul_2/ReadVariableOp$^rnn/odelstm/MatMul_3/ReadVariableOp$^rnn/odelstm/MatMul_4/ReadVariableOp$^rnn/odelstm/MatMul_5/ReadVariableOp$^rnn/odelstm/MatMul_6/ReadVariableOp$^rnn/odelstm/MatMul_7/ReadVariableOp$^rnn/odelstm/MatMul_8/ReadVariableOp$^rnn/odelstm/MatMul_9/ReadVariableOp^rnn/odelstm/ReadVariableOp^rnn/odelstm/ReadVariableOp_1^rnn/odelstm/ReadVariableOp_2^rnn/odelstm/ReadVariableOp_3!^rnn/odelstm/add_1/ReadVariableOp"^rnn/odelstm/add_11/ReadVariableOp"^rnn/odelstm/add_14/ReadVariableOp!^rnn/odelstm/add_5/ReadVariableOp!^rnn/odelstm/add_8/ReadVariableOp
^rnn/while*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:�d:�d:::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2F
!rnn/odelstm/MatMul/ReadVariableOp!rnn/odelstm/MatMul/ReadVariableOp2J
#rnn/odelstm/MatMul_1/ReadVariableOp#rnn/odelstm/MatMul_1/ReadVariableOp2J
#rnn/odelstm/MatMul_2/ReadVariableOp#rnn/odelstm/MatMul_2/ReadVariableOp2J
#rnn/odelstm/MatMul_3/ReadVariableOp#rnn/odelstm/MatMul_3/ReadVariableOp2J
#rnn/odelstm/MatMul_4/ReadVariableOp#rnn/odelstm/MatMul_4/ReadVariableOp2J
#rnn/odelstm/MatMul_5/ReadVariableOp#rnn/odelstm/MatMul_5/ReadVariableOp2J
#rnn/odelstm/MatMul_6/ReadVariableOp#rnn/odelstm/MatMul_6/ReadVariableOp2J
#rnn/odelstm/MatMul_7/ReadVariableOp#rnn/odelstm/MatMul_7/ReadVariableOp2J
#rnn/odelstm/MatMul_8/ReadVariableOp#rnn/odelstm/MatMul_8/ReadVariableOp2J
#rnn/odelstm/MatMul_9/ReadVariableOp#rnn/odelstm/MatMul_9/ReadVariableOp28
rnn/odelstm/ReadVariableOprnn/odelstm/ReadVariableOp2<
rnn/odelstm/ReadVariableOp_1rnn/odelstm/ReadVariableOp_12<
rnn/odelstm/ReadVariableOp_2rnn/odelstm/ReadVariableOp_22<
rnn/odelstm/ReadVariableOp_3rnn/odelstm/ReadVariableOp_32D
 rnn/odelstm/add_1/ReadVariableOp rnn/odelstm/add_1/ReadVariableOp2F
!rnn/odelstm/add_11/ReadVariableOp!rnn/odelstm/add_11/ReadVariableOp2F
!rnn/odelstm/add_14/ReadVariableOp!rnn/odelstm/add_14/ReadVariableOp2D
 rnn/odelstm/add_5/ReadVariableOp rnn/odelstm/add_5/ReadVariableOp2D
 rnn/odelstm/add_8/ReadVariableOp rnn/odelstm/add_8/ReadVariableOp2
	rnn/while	rnn/while:M I
#
_output_shapes
:�d
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:�d
"
_user_specified_name
inputs/1
��
�
>__inference_rnn_layer_call_and_return_conditional_losses_61245

inputs*
&odelstm_matmul_readvariableop_resource,
(odelstm_matmul_1_readvariableop_resource)
%odelstm_add_1_readvariableop_resource,
(odelstm_matmul_2_readvariableop_resource,
(odelstm_matmul_3_readvariableop_resource)
%odelstm_add_5_readvariableop_resource#
odelstm_readvariableop_resource
identity��odelstm/MatMul/ReadVariableOp�odelstm/MatMul_1/ReadVariableOp�odelstm/MatMul_2/ReadVariableOp�odelstm/MatMul_3/ReadVariableOp�odelstm/MatMul_4/ReadVariableOp�odelstm/MatMul_5/ReadVariableOp�odelstm/MatMul_6/ReadVariableOp�odelstm/MatMul_7/ReadVariableOp�odelstm/MatMul_8/ReadVariableOp�odelstm/MatMul_9/ReadVariableOp�odelstm/ReadVariableOp�odelstm/ReadVariableOp_1�odelstm/ReadVariableOp_2�odelstm/ReadVariableOp_3�odelstm/add_1/ReadVariableOp�odelstm/add_11/ReadVariableOp�odelstm/add_14/ReadVariableOp�odelstm/add_5/ReadVariableOp�odelstm/add_8/ReadVariableOp�whilec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"�   d      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constm
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*
_output_shapes
:	�@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constu
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*
_output_shapes
:	�@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permr
	transpose	Transposeinputstranspose/perm:output:0*
T0*#
_output_shapes
:d�2
	transposeg
Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   �      2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�      27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*
shrink_axis_mask2
strided_slice_2�
odelstm/MatMul/ReadVariableOpReadVariableOp&odelstm_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
odelstm/MatMul/ReadVariableOp�
odelstm/MatMulMatMulstrided_slice_2:output:0%odelstm/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
odelstm/MatMul�
odelstm/MatMul_1/ReadVariableOpReadVariableOp(odelstm_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02!
odelstm/MatMul_1/ReadVariableOp�
odelstm/MatMul_1MatMulzeros_1:output:0'odelstm/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
odelstm/MatMul_1�
odelstm/addAddV2odelstm/MatMul:product:0odelstm/MatMul_1:product:0*
T0* 
_output_shapes
:
��2
odelstm/add�
odelstm/add_1/ReadVariableOpReadVariableOp%odelstm_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
odelstm/add_1/ReadVariableOp�
odelstm/add_1AddV2odelstm/add:z:0$odelstm/add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
odelstm/add_1`
odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
odelstm/Const}
odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
odelstm/split/split_dim�
odelstm/splitSplit odelstm/split/split_dim:output:0odelstm/add_1:z:0*
T0*@
_output_shapes.
,:	�@:	�@:	�@:	�@*
	num_split2
odelstm/splitf
odelstm/TanhTanhodelstm/split:output:0*
T0*
_output_shapes
:	�@2
odelstm/Tanho
odelstm/SigmoidSigmoidodelstm/split:output:1*
T0*
_output_shapes
:	�@2
odelstm/Sigmoidg
odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
odelstm/add_2/y�
odelstm/add_2AddV2odelstm/split:output:2odelstm/add_2/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/add_2n
odelstm/Sigmoid_1Sigmoidodelstm/add_2:z:0*
T0*
_output_shapes
:	�@2
odelstm/Sigmoid_1s
odelstm/Sigmoid_2Sigmoidodelstm/split:output:3*
T0*
_output_shapes
:	�@2
odelstm/Sigmoid_2r
odelstm/mulMulzeros:output:0odelstm/Sigmoid_1:y:0*
T0*
_output_shapes
:	�@2
odelstm/mulv
odelstm/mul_1Mulodelstm/Tanh:y:0odelstm/Sigmoid:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_1u
odelstm/add_3AddV2odelstm/mul:z:0odelstm/mul_1:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_3e
odelstm/Tanh_1Tanhodelstm/add_3:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_1z
odelstm/mul_2Mulodelstm/Tanh_1:y:0odelstm/Sigmoid_2:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_2�
odelstm/MatMul_2/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_2/ReadVariableOp�
odelstm/MatMul_2MatMulodelstm/mul_2:z:0'odelstm/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_2�
odelstm/MatMul_3/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_3/ReadVariableOp�
odelstm/MatMul_3MatMulzeros_1:output:0'odelstm/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_3�
odelstm/add_4AddV2odelstm/MatMul_2:product:0odelstm/MatMul_3:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_4�
odelstm/add_5/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_5/ReadVariableOp�
odelstm/add_5AddV2odelstm/add_4:z:0$odelstm/add_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_5e
odelstm/Tanh_2Tanhodelstm/add_5:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_2�
odelstm/ReadVariableOpReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp�
odelstm/mul_3Mulodelstm/ReadVariableOp:value:0odelstm/Tanh_2:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_3g
odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_4/y{
odelstm/mul_4Mulzeros_1:output:0odelstm/mul_4/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_4q
odelstm/subSubodelstm/mul_3:z:0odelstm/mul_4:z:0*
T0*
_output_shapes
:	�@2
odelstm/subg
odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_5/xz
odelstm/mul_5Mulodelstm/mul_5/x:output:0odelstm/sub:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_5v
odelstm/add_6AddV2zeros_1:output:0odelstm/mul_5:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_6�
odelstm/MatMul_4/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_4/ReadVariableOp�
odelstm/MatMul_4MatMulodelstm/mul_2:z:0'odelstm/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_4�
odelstm/MatMul_5/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_5/ReadVariableOp�
odelstm/MatMul_5MatMulodelstm/add_6:z:0'odelstm/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_5�
odelstm/add_7AddV2odelstm/MatMul_4:product:0odelstm/MatMul_5:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_7�
odelstm/add_8/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_8/ReadVariableOp�
odelstm/add_8AddV2odelstm/add_7:z:0$odelstm/add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_8e
odelstm/Tanh_3Tanhodelstm/add_8:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_3�
odelstm/ReadVariableOp_1ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_1�
odelstm/mul_6Mul odelstm/ReadVariableOp_1:value:0odelstm/Tanh_3:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_6g
odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_7/y|
odelstm/mul_7Mulodelstm/add_6:z:0odelstm/mul_7/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_7u
odelstm/sub_1Subodelstm/mul_6:z:0odelstm/mul_7:z:0*
T0*
_output_shapes
:	�@2
odelstm/sub_1g
odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_8/x|
odelstm/mul_8Mulodelstm/mul_8/x:output:0odelstm/sub_1:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_8w
odelstm/add_9AddV2odelstm/add_6:z:0odelstm/mul_8:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_9�
odelstm/MatMul_6/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_6/ReadVariableOp�
odelstm/MatMul_6MatMulodelstm/mul_2:z:0'odelstm/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_6�
odelstm/MatMul_7/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_7/ReadVariableOp�
odelstm/MatMul_7MatMulodelstm/add_9:z:0'odelstm/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_7�
odelstm/add_10AddV2odelstm/MatMul_6:product:0odelstm/MatMul_7:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_10�
odelstm/add_11/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_11/ReadVariableOp�
odelstm/add_11AddV2odelstm/add_10:z:0%odelstm/add_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_11f
odelstm/Tanh_4Tanhodelstm/add_11:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_4�
odelstm/ReadVariableOp_2ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_2�
odelstm/mul_9Mul odelstm/ReadVariableOp_2:value:0odelstm/Tanh_4:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_9i
odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_10/y
odelstm/mul_10Mulodelstm/add_9:z:0odelstm/mul_10/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_10v
odelstm/sub_2Subodelstm/mul_9:z:0odelstm/mul_10:z:0*
T0*
_output_shapes
:	�@2
odelstm/sub_2i
odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_11/x
odelstm/mul_11Mulodelstm/mul_11/x:output:0odelstm/sub_2:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_11z
odelstm/add_12AddV2odelstm/add_9:z:0odelstm/mul_11:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_12�
odelstm/MatMul_8/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_8/ReadVariableOp�
odelstm/MatMul_8MatMulodelstm/mul_2:z:0'odelstm/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_8�
odelstm/MatMul_9/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_9/ReadVariableOp�
odelstm/MatMul_9MatMulodelstm/add_12:z:0'odelstm/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_9�
odelstm/add_13AddV2odelstm/MatMul_8:product:0odelstm/MatMul_9:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_13�
odelstm/add_14/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_14/ReadVariableOp�
odelstm/add_14AddV2odelstm/add_13:z:0%odelstm/add_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_14f
odelstm/Tanh_5Tanhodelstm/add_14:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_5�
odelstm/ReadVariableOp_3ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_3�
odelstm/mul_12Mul odelstm/ReadVariableOp_3:value:0odelstm/Tanh_5:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_12i
odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_13/y�
odelstm/mul_13Mulodelstm/add_12:z:0odelstm/mul_13/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_13w
odelstm/sub_3Subodelstm/mul_12:z:0odelstm/mul_13:z:0*
T0*
_output_shapes
:	�@2
odelstm/sub_3i
odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_14/x
odelstm/mul_14Mulodelstm/mul_14/x:output:0odelstm/sub_3:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_14{
odelstm/add_15AddV2odelstm/add_12:z:0odelstm/mul_14:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_15�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�   @   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0&odelstm_matmul_readvariableop_resource(odelstm_matmul_1_readvariableop_resource%odelstm_add_1_readvariableop_resource(odelstm_matmul_2_readvariableop_resource(odelstm_matmul_3_readvariableop_resource%odelstm_add_5_readvariableop_resourceodelstm_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*D
_output_shapes2
0: : : : :	�@:	�@: : : : : : : : : *)
_read_only_resource_inputs
		
*
bodyR
while_body_61083*
condR
while_cond_61082*C
output_shapes2
0: : : : :	�@:	�@: : : : : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�   @   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:d�@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*#
_output_shapes
:�d@2
transpose_1�
IdentityIdentitystrided_slice_3:output:0^odelstm/MatMul/ReadVariableOp ^odelstm/MatMul_1/ReadVariableOp ^odelstm/MatMul_2/ReadVariableOp ^odelstm/MatMul_3/ReadVariableOp ^odelstm/MatMul_4/ReadVariableOp ^odelstm/MatMul_5/ReadVariableOp ^odelstm/MatMul_6/ReadVariableOp ^odelstm/MatMul_7/ReadVariableOp ^odelstm/MatMul_8/ReadVariableOp ^odelstm/MatMul_9/ReadVariableOp^odelstm/ReadVariableOp^odelstm/ReadVariableOp_1^odelstm/ReadVariableOp_2^odelstm/ReadVariableOp_3^odelstm/add_1/ReadVariableOp^odelstm/add_11/ReadVariableOp^odelstm/add_14/ReadVariableOp^odelstm/add_5/ReadVariableOp^odelstm/add_8/ReadVariableOp^while*
T0*
_output_shapes
:	�@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:�d:::::::2>
odelstm/MatMul/ReadVariableOpodelstm/MatMul/ReadVariableOp2B
odelstm/MatMul_1/ReadVariableOpodelstm/MatMul_1/ReadVariableOp2B
odelstm/MatMul_2/ReadVariableOpodelstm/MatMul_2/ReadVariableOp2B
odelstm/MatMul_3/ReadVariableOpodelstm/MatMul_3/ReadVariableOp2B
odelstm/MatMul_4/ReadVariableOpodelstm/MatMul_4/ReadVariableOp2B
odelstm/MatMul_5/ReadVariableOpodelstm/MatMul_5/ReadVariableOp2B
odelstm/MatMul_6/ReadVariableOpodelstm/MatMul_6/ReadVariableOp2B
odelstm/MatMul_7/ReadVariableOpodelstm/MatMul_7/ReadVariableOp2B
odelstm/MatMul_8/ReadVariableOpodelstm/MatMul_8/ReadVariableOp2B
odelstm/MatMul_9/ReadVariableOpodelstm/MatMul_9/ReadVariableOp20
odelstm/ReadVariableOpodelstm/ReadVariableOp24
odelstm/ReadVariableOp_1odelstm/ReadVariableOp_124
odelstm/ReadVariableOp_2odelstm/ReadVariableOp_224
odelstm/ReadVariableOp_3odelstm/ReadVariableOp_32<
odelstm/add_1/ReadVariableOpodelstm/add_1/ReadVariableOp2>
odelstm/add_11/ReadVariableOpodelstm/add_11/ReadVariableOp2>
odelstm/add_14/ReadVariableOpodelstm/add_14/ReadVariableOp2<
odelstm/add_5/ReadVariableOpodelstm/add_5/ReadVariableOp2<
odelstm/add_8/ReadVariableOpodelstm/add_8/ReadVariableOp2
whilewhile:K G
#
_output_shapes
:�d
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_59645
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_580562
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:�d:�d:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:�d
!
_user_specified_name	input_1:LH
#
_output_shapes
:�d
!
_user_specified_name	input_2
ܲ
�
>__inference_rnn_layer_call_and_return_conditional_losses_60607
inputs_0*
&odelstm_matmul_readvariableop_resource,
(odelstm_matmul_1_readvariableop_resource)
%odelstm_add_1_readvariableop_resource,
(odelstm_matmul_2_readvariableop_resource,
(odelstm_matmul_3_readvariableop_resource)
%odelstm_add_5_readvariableop_resource#
odelstm_readvariableop_resource
identity��odelstm/MatMul/ReadVariableOp�odelstm/MatMul_1/ReadVariableOp�odelstm/MatMul_2/ReadVariableOp�odelstm/MatMul_3/ReadVariableOp�odelstm/MatMul_4/ReadVariableOp�odelstm/MatMul_5/ReadVariableOp�odelstm/MatMul_6/ReadVariableOp�odelstm/MatMul_7/ReadVariableOp�odelstm/MatMul_8/ReadVariableOp�odelstm/MatMul_9/ReadVariableOp�odelstm/ReadVariableOp�odelstm/ReadVariableOp_1�odelstm/ReadVariableOp_2�odelstm/ReadVariableOp_3�odelstm/add_1/ReadVariableOp�odelstm/add_11/ReadVariableOp�odelstm/add_14/ReadVariableOp�odelstm/add_5/ReadVariableOp�odelstm/add_8/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
odelstm/MatMul/ReadVariableOpReadVariableOp&odelstm_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
odelstm/MatMul/ReadVariableOp�
odelstm/MatMulMatMulstrided_slice_2:output:0%odelstm/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
odelstm/MatMul�
odelstm/MatMul_1/ReadVariableOpReadVariableOp(odelstm_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02!
odelstm/MatMul_1/ReadVariableOp�
odelstm/MatMul_1MatMulzeros_1:output:0'odelstm/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
odelstm/MatMul_1�
odelstm/addAddV2odelstm/MatMul:product:0odelstm/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
odelstm/add�
odelstm/add_1/ReadVariableOpReadVariableOp%odelstm_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
odelstm/add_1/ReadVariableOp�
odelstm/add_1AddV2odelstm/add:z:0$odelstm/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
odelstm/add_1`
odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
odelstm/Const}
odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
odelstm/split/split_dim�
odelstm/splitSplit odelstm/split/split_dim:output:0odelstm/add_1:z:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
odelstm/splitn
odelstm/TanhTanhodelstm/split:output:0*
T0*'
_output_shapes
:���������@2
odelstm/Tanhw
odelstm/SigmoidSigmoidodelstm/split:output:1*
T0*'
_output_shapes
:���������@2
odelstm/Sigmoidg
odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
odelstm/add_2/y�
odelstm/add_2AddV2odelstm/split:output:2odelstm/add_2/y:output:0*
T0*'
_output_shapes
:���������@2
odelstm/add_2v
odelstm/Sigmoid_1Sigmoidodelstm/add_2:z:0*
T0*'
_output_shapes
:���������@2
odelstm/Sigmoid_1{
odelstm/Sigmoid_2Sigmoidodelstm/split:output:3*
T0*'
_output_shapes
:���������@2
odelstm/Sigmoid_2z
odelstm/mulMulzeros:output:0odelstm/Sigmoid_1:y:0*
T0*'
_output_shapes
:���������@2
odelstm/mul~
odelstm/mul_1Mulodelstm/Tanh:y:0odelstm/Sigmoid:y:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_1}
odelstm/add_3AddV2odelstm/mul:z:0odelstm/mul_1:z:0*
T0*'
_output_shapes
:���������@2
odelstm/add_3m
odelstm/Tanh_1Tanhodelstm/add_3:z:0*
T0*'
_output_shapes
:���������@2
odelstm/Tanh_1�
odelstm/mul_2Mulodelstm/Tanh_1:y:0odelstm/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_2�
odelstm/MatMul_2/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_2/ReadVariableOp�
odelstm/MatMul_2MatMulodelstm/mul_2:z:0'odelstm/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_2�
odelstm/MatMul_3/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_3/ReadVariableOp�
odelstm/MatMul_3MatMulzeros_1:output:0'odelstm/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_3�
odelstm/add_4AddV2odelstm/MatMul_2:product:0odelstm/MatMul_3:product:0*
T0*'
_output_shapes
:���������@2
odelstm/add_4�
odelstm/add_5/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_5/ReadVariableOp�
odelstm/add_5AddV2odelstm/add_4:z:0$odelstm/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/add_5m
odelstm/Tanh_2Tanhodelstm/add_5:z:0*
T0*'
_output_shapes
:���������@2
odelstm/Tanh_2�
odelstm/ReadVariableOpReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp�
odelstm/mul_3Mulodelstm/ReadVariableOp:value:0odelstm/Tanh_2:y:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_3g
odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_4/y�
odelstm/mul_4Mulzeros_1:output:0odelstm/mul_4/y:output:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_4y
odelstm/subSubodelstm/mul_3:z:0odelstm/mul_4:z:0*
T0*'
_output_shapes
:���������@2
odelstm/subg
odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_5/x�
odelstm/mul_5Mulodelstm/mul_5/x:output:0odelstm/sub:z:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_5~
odelstm/add_6AddV2zeros_1:output:0odelstm/mul_5:z:0*
T0*'
_output_shapes
:���������@2
odelstm/add_6�
odelstm/MatMul_4/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_4/ReadVariableOp�
odelstm/MatMul_4MatMulodelstm/mul_2:z:0'odelstm/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_4�
odelstm/MatMul_5/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_5/ReadVariableOp�
odelstm/MatMul_5MatMulodelstm/add_6:z:0'odelstm/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_5�
odelstm/add_7AddV2odelstm/MatMul_4:product:0odelstm/MatMul_5:product:0*
T0*'
_output_shapes
:���������@2
odelstm/add_7�
odelstm/add_8/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_8/ReadVariableOp�
odelstm/add_8AddV2odelstm/add_7:z:0$odelstm/add_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/add_8m
odelstm/Tanh_3Tanhodelstm/add_8:z:0*
T0*'
_output_shapes
:���������@2
odelstm/Tanh_3�
odelstm/ReadVariableOp_1ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_1�
odelstm/mul_6Mul odelstm/ReadVariableOp_1:value:0odelstm/Tanh_3:y:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_6g
odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_7/y�
odelstm/mul_7Mulodelstm/add_6:z:0odelstm/mul_7/y:output:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_7}
odelstm/sub_1Subodelstm/mul_6:z:0odelstm/mul_7:z:0*
T0*'
_output_shapes
:���������@2
odelstm/sub_1g
odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_8/x�
odelstm/mul_8Mulodelstm/mul_8/x:output:0odelstm/sub_1:z:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_8
odelstm/add_9AddV2odelstm/add_6:z:0odelstm/mul_8:z:0*
T0*'
_output_shapes
:���������@2
odelstm/add_9�
odelstm/MatMul_6/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_6/ReadVariableOp�
odelstm/MatMul_6MatMulodelstm/mul_2:z:0'odelstm/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_6�
odelstm/MatMul_7/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_7/ReadVariableOp�
odelstm/MatMul_7MatMulodelstm/add_9:z:0'odelstm/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_7�
odelstm/add_10AddV2odelstm/MatMul_6:product:0odelstm/MatMul_7:product:0*
T0*'
_output_shapes
:���������@2
odelstm/add_10�
odelstm/add_11/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_11/ReadVariableOp�
odelstm/add_11AddV2odelstm/add_10:z:0%odelstm/add_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/add_11n
odelstm/Tanh_4Tanhodelstm/add_11:z:0*
T0*'
_output_shapes
:���������@2
odelstm/Tanh_4�
odelstm/ReadVariableOp_2ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_2�
odelstm/mul_9Mul odelstm/ReadVariableOp_2:value:0odelstm/Tanh_4:y:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_9i
odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_10/y�
odelstm/mul_10Mulodelstm/add_9:z:0odelstm/mul_10/y:output:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_10~
odelstm/sub_2Subodelstm/mul_9:z:0odelstm/mul_10:z:0*
T0*'
_output_shapes
:���������@2
odelstm/sub_2i
odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_11/x�
odelstm/mul_11Mulodelstm/mul_11/x:output:0odelstm/sub_2:z:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_11�
odelstm/add_12AddV2odelstm/add_9:z:0odelstm/mul_11:z:0*
T0*'
_output_shapes
:���������@2
odelstm/add_12�
odelstm/MatMul_8/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_8/ReadVariableOp�
odelstm/MatMul_8MatMulodelstm/mul_2:z:0'odelstm/MatMul_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_8�
odelstm/MatMul_9/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_9/ReadVariableOp�
odelstm/MatMul_9MatMulodelstm/add_12:z:0'odelstm/MatMul_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_9�
odelstm/add_13AddV2odelstm/MatMul_8:product:0odelstm/MatMul_9:product:0*
T0*'
_output_shapes
:���������@2
odelstm/add_13�
odelstm/add_14/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_14/ReadVariableOp�
odelstm/add_14AddV2odelstm/add_13:z:0%odelstm/add_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/add_14n
odelstm/Tanh_5Tanhodelstm/add_14:z:0*
T0*'
_output_shapes
:���������@2
odelstm/Tanh_5�
odelstm/ReadVariableOp_3ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_3�
odelstm/mul_12Mul odelstm/ReadVariableOp_3:value:0odelstm/Tanh_5:y:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_12i
odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_13/y�
odelstm/mul_13Mulodelstm/add_12:z:0odelstm/mul_13/y:output:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_13
odelstm/sub_3Subodelstm/mul_12:z:0odelstm/mul_13:z:0*
T0*'
_output_shapes
:���������@2
odelstm/sub_3i
odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_14/x�
odelstm/mul_14Mulodelstm/mul_14/x:output:0odelstm/sub_3:z:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_14�
odelstm/add_15AddV2odelstm/add_12:z:0odelstm/mul_14:z:0*
T0*'
_output_shapes
:���������@2
odelstm/add_15�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0&odelstm_matmul_readvariableop_resource(odelstm_matmul_1_readvariableop_resource%odelstm_add_1_readvariableop_resource(odelstm_matmul_2_readvariableop_resource(odelstm_matmul_3_readvariableop_resource%odelstm_add_5_readvariableop_resourceodelstm_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*T
_output_shapesB
@: : : : :���������@:���������@: : : : : : : : : *)
_read_only_resource_inputs
		
*
bodyR
while_body_60445*
condR
while_cond_60444*S
output_shapesB
@: : : : :���������@:���������@: : : : : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1�
IdentityIdentitystrided_slice_3:output:0^odelstm/MatMul/ReadVariableOp ^odelstm/MatMul_1/ReadVariableOp ^odelstm/MatMul_2/ReadVariableOp ^odelstm/MatMul_3/ReadVariableOp ^odelstm/MatMul_4/ReadVariableOp ^odelstm/MatMul_5/ReadVariableOp ^odelstm/MatMul_6/ReadVariableOp ^odelstm/MatMul_7/ReadVariableOp ^odelstm/MatMul_8/ReadVariableOp ^odelstm/MatMul_9/ReadVariableOp^odelstm/ReadVariableOp^odelstm/ReadVariableOp_1^odelstm/ReadVariableOp_2^odelstm/ReadVariableOp_3^odelstm/add_1/ReadVariableOp^odelstm/add_11/ReadVariableOp^odelstm/add_14/ReadVariableOp^odelstm/add_5/ReadVariableOp^odelstm/add_8/ReadVariableOp^while*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:������������������:::::::2>
odelstm/MatMul/ReadVariableOpodelstm/MatMul/ReadVariableOp2B
odelstm/MatMul_1/ReadVariableOpodelstm/MatMul_1/ReadVariableOp2B
odelstm/MatMul_2/ReadVariableOpodelstm/MatMul_2/ReadVariableOp2B
odelstm/MatMul_3/ReadVariableOpodelstm/MatMul_3/ReadVariableOp2B
odelstm/MatMul_4/ReadVariableOpodelstm/MatMul_4/ReadVariableOp2B
odelstm/MatMul_5/ReadVariableOpodelstm/MatMul_5/ReadVariableOp2B
odelstm/MatMul_6/ReadVariableOpodelstm/MatMul_6/ReadVariableOp2B
odelstm/MatMul_7/ReadVariableOpodelstm/MatMul_7/ReadVariableOp2B
odelstm/MatMul_8/ReadVariableOpodelstm/MatMul_8/ReadVariableOp2B
odelstm/MatMul_9/ReadVariableOpodelstm/MatMul_9/ReadVariableOp20
odelstm/ReadVariableOpodelstm/ReadVariableOp24
odelstm/ReadVariableOp_1odelstm/ReadVariableOp_124
odelstm/ReadVariableOp_2odelstm/ReadVariableOp_224
odelstm/ReadVariableOp_3odelstm/ReadVariableOp_32<
odelstm/add_1/ReadVariableOpodelstm/add_1/ReadVariableOp2>
odelstm/add_11/ReadVariableOpodelstm/add_11/ReadVariableOp2>
odelstm/add_14/ReadVariableOpodelstm/add_14/ReadVariableOp2<
odelstm/add_5/ReadVariableOpodelstm/add_5/ReadVariableOp2<
odelstm/add_8/ReadVariableOpodelstm/add_8/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
C__inference_ode_lstm_layer_call_and_return_conditional_losses_59590

inputs
inputs_1
	rnn_59569
	rnn_59571
	rnn_59573
	rnn_59575
	rnn_59577
	rnn_59579
	rnn_59581
dense_59584
dense_59586
identity��dense/StatefulPartitionedCall�rnn/StatefulPartitionedCall�
rnn/StatefulPartitionedCallStatefulPartitionedCallinputs	rnn_59569	rnn_59571	rnn_59573	rnn_59575	rnn_59577	rnn_59579	rnn_59581*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�@*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_594062
rnn/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_59584dense_59586*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_594702
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^rnn/StatefulPartitionedCall*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:�d:�d:::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:K G
#
_output_shapes
:�d
 
_user_specified_nameinputs:KG
#
_output_shapes
:�d
 
_user_specified_nameinputs
ܲ
�
>__inference_rnn_layer_call_and_return_conditional_losses_60907
inputs_0*
&odelstm_matmul_readvariableop_resource,
(odelstm_matmul_1_readvariableop_resource)
%odelstm_add_1_readvariableop_resource,
(odelstm_matmul_2_readvariableop_resource,
(odelstm_matmul_3_readvariableop_resource)
%odelstm_add_5_readvariableop_resource#
odelstm_readvariableop_resource
identity��odelstm/MatMul/ReadVariableOp�odelstm/MatMul_1/ReadVariableOp�odelstm/MatMul_2/ReadVariableOp�odelstm/MatMul_3/ReadVariableOp�odelstm/MatMul_4/ReadVariableOp�odelstm/MatMul_5/ReadVariableOp�odelstm/MatMul_6/ReadVariableOp�odelstm/MatMul_7/ReadVariableOp�odelstm/MatMul_8/ReadVariableOp�odelstm/MatMul_9/ReadVariableOp�odelstm/ReadVariableOp�odelstm/ReadVariableOp_1�odelstm/ReadVariableOp_2�odelstm/ReadVariableOp_3�odelstm/add_1/ReadVariableOp�odelstm/add_11/ReadVariableOp�odelstm/add_14/ReadVariableOp�odelstm/add_5/ReadVariableOp�odelstm/add_8/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
odelstm/MatMul/ReadVariableOpReadVariableOp&odelstm_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
odelstm/MatMul/ReadVariableOp�
odelstm/MatMulMatMulstrided_slice_2:output:0%odelstm/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
odelstm/MatMul�
odelstm/MatMul_1/ReadVariableOpReadVariableOp(odelstm_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02!
odelstm/MatMul_1/ReadVariableOp�
odelstm/MatMul_1MatMulzeros_1:output:0'odelstm/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
odelstm/MatMul_1�
odelstm/addAddV2odelstm/MatMul:product:0odelstm/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
odelstm/add�
odelstm/add_1/ReadVariableOpReadVariableOp%odelstm_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
odelstm/add_1/ReadVariableOp�
odelstm/add_1AddV2odelstm/add:z:0$odelstm/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
odelstm/add_1`
odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
odelstm/Const}
odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
odelstm/split/split_dim�
odelstm/splitSplit odelstm/split/split_dim:output:0odelstm/add_1:z:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
odelstm/splitn
odelstm/TanhTanhodelstm/split:output:0*
T0*'
_output_shapes
:���������@2
odelstm/Tanhw
odelstm/SigmoidSigmoidodelstm/split:output:1*
T0*'
_output_shapes
:���������@2
odelstm/Sigmoidg
odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
odelstm/add_2/y�
odelstm/add_2AddV2odelstm/split:output:2odelstm/add_2/y:output:0*
T0*'
_output_shapes
:���������@2
odelstm/add_2v
odelstm/Sigmoid_1Sigmoidodelstm/add_2:z:0*
T0*'
_output_shapes
:���������@2
odelstm/Sigmoid_1{
odelstm/Sigmoid_2Sigmoidodelstm/split:output:3*
T0*'
_output_shapes
:���������@2
odelstm/Sigmoid_2z
odelstm/mulMulzeros:output:0odelstm/Sigmoid_1:y:0*
T0*'
_output_shapes
:���������@2
odelstm/mul~
odelstm/mul_1Mulodelstm/Tanh:y:0odelstm/Sigmoid:y:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_1}
odelstm/add_3AddV2odelstm/mul:z:0odelstm/mul_1:z:0*
T0*'
_output_shapes
:���������@2
odelstm/add_3m
odelstm/Tanh_1Tanhodelstm/add_3:z:0*
T0*'
_output_shapes
:���������@2
odelstm/Tanh_1�
odelstm/mul_2Mulodelstm/Tanh_1:y:0odelstm/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_2�
odelstm/MatMul_2/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_2/ReadVariableOp�
odelstm/MatMul_2MatMulodelstm/mul_2:z:0'odelstm/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_2�
odelstm/MatMul_3/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_3/ReadVariableOp�
odelstm/MatMul_3MatMulzeros_1:output:0'odelstm/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_3�
odelstm/add_4AddV2odelstm/MatMul_2:product:0odelstm/MatMul_3:product:0*
T0*'
_output_shapes
:���������@2
odelstm/add_4�
odelstm/add_5/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_5/ReadVariableOp�
odelstm/add_5AddV2odelstm/add_4:z:0$odelstm/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/add_5m
odelstm/Tanh_2Tanhodelstm/add_5:z:0*
T0*'
_output_shapes
:���������@2
odelstm/Tanh_2�
odelstm/ReadVariableOpReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp�
odelstm/mul_3Mulodelstm/ReadVariableOp:value:0odelstm/Tanh_2:y:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_3g
odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_4/y�
odelstm/mul_4Mulzeros_1:output:0odelstm/mul_4/y:output:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_4y
odelstm/subSubodelstm/mul_3:z:0odelstm/mul_4:z:0*
T0*'
_output_shapes
:���������@2
odelstm/subg
odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_5/x�
odelstm/mul_5Mulodelstm/mul_5/x:output:0odelstm/sub:z:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_5~
odelstm/add_6AddV2zeros_1:output:0odelstm/mul_5:z:0*
T0*'
_output_shapes
:���������@2
odelstm/add_6�
odelstm/MatMul_4/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_4/ReadVariableOp�
odelstm/MatMul_4MatMulodelstm/mul_2:z:0'odelstm/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_4�
odelstm/MatMul_5/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_5/ReadVariableOp�
odelstm/MatMul_5MatMulodelstm/add_6:z:0'odelstm/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_5�
odelstm/add_7AddV2odelstm/MatMul_4:product:0odelstm/MatMul_5:product:0*
T0*'
_output_shapes
:���������@2
odelstm/add_7�
odelstm/add_8/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_8/ReadVariableOp�
odelstm/add_8AddV2odelstm/add_7:z:0$odelstm/add_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/add_8m
odelstm/Tanh_3Tanhodelstm/add_8:z:0*
T0*'
_output_shapes
:���������@2
odelstm/Tanh_3�
odelstm/ReadVariableOp_1ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_1�
odelstm/mul_6Mul odelstm/ReadVariableOp_1:value:0odelstm/Tanh_3:y:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_6g
odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_7/y�
odelstm/mul_7Mulodelstm/add_6:z:0odelstm/mul_7/y:output:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_7}
odelstm/sub_1Subodelstm/mul_6:z:0odelstm/mul_7:z:0*
T0*'
_output_shapes
:���������@2
odelstm/sub_1g
odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_8/x�
odelstm/mul_8Mulodelstm/mul_8/x:output:0odelstm/sub_1:z:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_8
odelstm/add_9AddV2odelstm/add_6:z:0odelstm/mul_8:z:0*
T0*'
_output_shapes
:���������@2
odelstm/add_9�
odelstm/MatMul_6/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_6/ReadVariableOp�
odelstm/MatMul_6MatMulodelstm/mul_2:z:0'odelstm/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_6�
odelstm/MatMul_7/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_7/ReadVariableOp�
odelstm/MatMul_7MatMulodelstm/add_9:z:0'odelstm/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_7�
odelstm/add_10AddV2odelstm/MatMul_6:product:0odelstm/MatMul_7:product:0*
T0*'
_output_shapes
:���������@2
odelstm/add_10�
odelstm/add_11/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_11/ReadVariableOp�
odelstm/add_11AddV2odelstm/add_10:z:0%odelstm/add_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/add_11n
odelstm/Tanh_4Tanhodelstm/add_11:z:0*
T0*'
_output_shapes
:���������@2
odelstm/Tanh_4�
odelstm/ReadVariableOp_2ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_2�
odelstm/mul_9Mul odelstm/ReadVariableOp_2:value:0odelstm/Tanh_4:y:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_9i
odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_10/y�
odelstm/mul_10Mulodelstm/add_9:z:0odelstm/mul_10/y:output:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_10~
odelstm/sub_2Subodelstm/mul_9:z:0odelstm/mul_10:z:0*
T0*'
_output_shapes
:���������@2
odelstm/sub_2i
odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_11/x�
odelstm/mul_11Mulodelstm/mul_11/x:output:0odelstm/sub_2:z:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_11�
odelstm/add_12AddV2odelstm/add_9:z:0odelstm/mul_11:z:0*
T0*'
_output_shapes
:���������@2
odelstm/add_12�
odelstm/MatMul_8/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_8/ReadVariableOp�
odelstm/MatMul_8MatMulodelstm/mul_2:z:0'odelstm/MatMul_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_8�
odelstm/MatMul_9/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_9/ReadVariableOp�
odelstm/MatMul_9MatMulodelstm/add_12:z:0'odelstm/MatMul_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/MatMul_9�
odelstm/add_13AddV2odelstm/MatMul_8:product:0odelstm/MatMul_9:product:0*
T0*'
_output_shapes
:���������@2
odelstm/add_13�
odelstm/add_14/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_14/ReadVariableOp�
odelstm/add_14AddV2odelstm/add_13:z:0%odelstm/add_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
odelstm/add_14n
odelstm/Tanh_5Tanhodelstm/add_14:z:0*
T0*'
_output_shapes
:���������@2
odelstm/Tanh_5�
odelstm/ReadVariableOp_3ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_3�
odelstm/mul_12Mul odelstm/ReadVariableOp_3:value:0odelstm/Tanh_5:y:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_12i
odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_13/y�
odelstm/mul_13Mulodelstm/add_12:z:0odelstm/mul_13/y:output:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_13
odelstm/sub_3Subodelstm/mul_12:z:0odelstm/mul_13:z:0*
T0*'
_output_shapes
:���������@2
odelstm/sub_3i
odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_14/x�
odelstm/mul_14Mulodelstm/mul_14/x:output:0odelstm/sub_3:z:0*
T0*'
_output_shapes
:���������@2
odelstm/mul_14�
odelstm/add_15AddV2odelstm/add_12:z:0odelstm/mul_14:z:0*
T0*'
_output_shapes
:���������@2
odelstm/add_15�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0&odelstm_matmul_readvariableop_resource(odelstm_matmul_1_readvariableop_resource%odelstm_add_1_readvariableop_resource(odelstm_matmul_2_readvariableop_resource(odelstm_matmul_3_readvariableop_resource%odelstm_add_5_readvariableop_resourceodelstm_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*T
_output_shapesB
@: : : : :���������@:���������@: : : : : : : : : *)
_read_only_resource_inputs
		
*
bodyR
while_body_60745*
condR
while_cond_60744*S
output_shapesB
@: : : : :���������@:���������@: : : : : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1�
IdentityIdentitystrided_slice_3:output:0^odelstm/MatMul/ReadVariableOp ^odelstm/MatMul_1/ReadVariableOp ^odelstm/MatMul_2/ReadVariableOp ^odelstm/MatMul_3/ReadVariableOp ^odelstm/MatMul_4/ReadVariableOp ^odelstm/MatMul_5/ReadVariableOp ^odelstm/MatMul_6/ReadVariableOp ^odelstm/MatMul_7/ReadVariableOp ^odelstm/MatMul_8/ReadVariableOp ^odelstm/MatMul_9/ReadVariableOp^odelstm/ReadVariableOp^odelstm/ReadVariableOp_1^odelstm/ReadVariableOp_2^odelstm/ReadVariableOp_3^odelstm/add_1/ReadVariableOp^odelstm/add_11/ReadVariableOp^odelstm/add_14/ReadVariableOp^odelstm/add_5/ReadVariableOp^odelstm/add_8/ReadVariableOp^while*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:������������������:::::::2>
odelstm/MatMul/ReadVariableOpodelstm/MatMul/ReadVariableOp2B
odelstm/MatMul_1/ReadVariableOpodelstm/MatMul_1/ReadVariableOp2B
odelstm/MatMul_2/ReadVariableOpodelstm/MatMul_2/ReadVariableOp2B
odelstm/MatMul_3/ReadVariableOpodelstm/MatMul_3/ReadVariableOp2B
odelstm/MatMul_4/ReadVariableOpodelstm/MatMul_4/ReadVariableOp2B
odelstm/MatMul_5/ReadVariableOpodelstm/MatMul_5/ReadVariableOp2B
odelstm/MatMul_6/ReadVariableOpodelstm/MatMul_6/ReadVariableOp2B
odelstm/MatMul_7/ReadVariableOpodelstm/MatMul_7/ReadVariableOp2B
odelstm/MatMul_8/ReadVariableOpodelstm/MatMul_8/ReadVariableOp2B
odelstm/MatMul_9/ReadVariableOpodelstm/MatMul_9/ReadVariableOp20
odelstm/ReadVariableOpodelstm/ReadVariableOp24
odelstm/ReadVariableOp_1odelstm/ReadVariableOp_124
odelstm/ReadVariableOp_2odelstm/ReadVariableOp_224
odelstm/ReadVariableOp_3odelstm/ReadVariableOp_32<
odelstm/add_1/ReadVariableOpodelstm/add_1/ReadVariableOp2>
odelstm/add_11/ReadVariableOpodelstm/add_11/ReadVariableOp2>
odelstm/add_14/ReadVariableOpodelstm/add_14/ReadVariableOp2<
odelstm/add_5/ReadVariableOpodelstm/add_5/ReadVariableOp2<
odelstm/add_8/ReadVariableOpodelstm/add_8/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
@__inference_dense_layer_call_and_return_conditional_losses_59470

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*&
_input_shapes
:	�@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	�@
 
_user_specified_nameinputs
��
�

C__inference_ode_lstm_layer_call_and_return_conditional_losses_59952
inputs_0
inputs_1.
*rnn_odelstm_matmul_readvariableop_resource0
,rnn_odelstm_matmul_1_readvariableop_resource-
)rnn_odelstm_add_1_readvariableop_resource0
,rnn_odelstm_matmul_2_readvariableop_resource0
,rnn_odelstm_matmul_3_readvariableop_resource-
)rnn_odelstm_add_5_readvariableop_resource'
#rnn_odelstm_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�!rnn/odelstm/MatMul/ReadVariableOp�#rnn/odelstm/MatMul_1/ReadVariableOp�#rnn/odelstm/MatMul_2/ReadVariableOp�#rnn/odelstm/MatMul_3/ReadVariableOp�#rnn/odelstm/MatMul_4/ReadVariableOp�#rnn/odelstm/MatMul_5/ReadVariableOp�#rnn/odelstm/MatMul_6/ReadVariableOp�#rnn/odelstm/MatMul_7/ReadVariableOp�#rnn/odelstm/MatMul_8/ReadVariableOp�#rnn/odelstm/MatMul_9/ReadVariableOp�rnn/odelstm/ReadVariableOp�rnn/odelstm/ReadVariableOp_1�rnn/odelstm/ReadVariableOp_2�rnn/odelstm/ReadVariableOp_3� rnn/odelstm/add_1/ReadVariableOp�!rnn/odelstm/add_11/ReadVariableOp�!rnn/odelstm/add_14/ReadVariableOp� rnn/odelstm/add_5/ReadVariableOp� rnn/odelstm/add_8/ReadVariableOp�	rnn/whilek
	rnn/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"�   d      2
	rnn/Shape|
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice/stack�
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_1�
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_2�
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_sliced
rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
rnn/zeros/mul/y|
rnn/zeros/mulMulrnn/strided_slice:output:0rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/mulg
rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
rnn/zeros/Less/yw
rnn/zeros/LessLessrnn/zeros/mul:z:0rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/Lessj
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
rnn/zeros/packed/1�
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros/packedg
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros/Const}
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*
_output_shapes
:	�@2
	rnn/zerosh
rnn/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
rnn/zeros_1/mul/y�
rnn/zeros_1/mulMulrnn/strided_slice:output:0rnn/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/mulk
rnn/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
rnn/zeros_1/Less/y
rnn/zeros_1/LessLessrnn/zeros_1/mul:z:0rnn/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/Lessn
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
rnn/zeros_1/packed/1�
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros_1/packedk
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros_1/Const�
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*
_output_shapes
:	�@2
rnn/zeros_1}
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose/perm�
rnn/transpose	Transposeinputs_0rnn/transpose/perm:output:0*
T0*#
_output_shapes
:d�2
rnn/transposeo
rnn/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   �      2
rnn/Shape_1�
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_1/stack�
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_1�
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_2�
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_slice_1�
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
rnn/TensorArrayV2/element_shape�
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2�
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�      2;
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shape�
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+rnn/TensorArrayUnstack/TensorListFromTensor�
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_2/stack�
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_1�
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_2�
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*
shrink_axis_mask2
rnn/strided_slice_2�
!rnn/odelstm/MatMul/ReadVariableOpReadVariableOp*rnn_odelstm_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!rnn/odelstm/MatMul/ReadVariableOp�
rnn/odelstm/MatMulMatMulrnn/strided_slice_2:output:0)rnn/odelstm/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
rnn/odelstm/MatMul�
#rnn/odelstm/MatMul_1/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02%
#rnn/odelstm/MatMul_1/ReadVariableOp�
rnn/odelstm/MatMul_1MatMulrnn/zeros_1:output:0+rnn/odelstm/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
rnn/odelstm/MatMul_1�
rnn/odelstm/addAddV2rnn/odelstm/MatMul:product:0rnn/odelstm/MatMul_1:product:0*
T0* 
_output_shapes
:
��2
rnn/odelstm/add�
 rnn/odelstm/add_1/ReadVariableOpReadVariableOp)rnn_odelstm_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 rnn/odelstm/add_1/ReadVariableOp�
rnn/odelstm/add_1AddV2rnn/odelstm/add:z:0(rnn/odelstm/add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
rnn/odelstm/add_1h
rnn/odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/odelstm/Const�
rnn/odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
rnn/odelstm/split/split_dim�
rnn/odelstm/splitSplit$rnn/odelstm/split/split_dim:output:0rnn/odelstm/add_1:z:0*
T0*@
_output_shapes.
,:	�@:	�@:	�@:	�@*
	num_split2
rnn/odelstm/splitr
rnn/odelstm/TanhTanhrnn/odelstm/split:output:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/Tanh{
rnn/odelstm/SigmoidSigmoidrnn/odelstm/split:output:1*
T0*
_output_shapes
:	�@2
rnn/odelstm/Sigmoido
rnn/odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
rnn/odelstm/add_2/y�
rnn/odelstm/add_2AddV2rnn/odelstm/split:output:2rnn/odelstm/add_2/y:output:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_2z
rnn/odelstm/Sigmoid_1Sigmoidrnn/odelstm/add_2:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/Sigmoid_1
rnn/odelstm/Sigmoid_2Sigmoidrnn/odelstm/split:output:3*
T0*
_output_shapes
:	�@2
rnn/odelstm/Sigmoid_2�
rnn/odelstm/mulMulrnn/zeros:output:0rnn/odelstm/Sigmoid_1:y:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul�
rnn/odelstm/mul_1Mulrnn/odelstm/Tanh:y:0rnn/odelstm/Sigmoid:y:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_1�
rnn/odelstm/add_3AddV2rnn/odelstm/mul:z:0rnn/odelstm/mul_1:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_3q
rnn/odelstm/Tanh_1Tanhrnn/odelstm/add_3:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/Tanh_1�
rnn/odelstm/mul_2Mulrnn/odelstm/Tanh_1:y:0rnn/odelstm/Sigmoid_2:y:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_2�
#rnn/odelstm/MatMul_2/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_2/ReadVariableOp�
rnn/odelstm/MatMul_2MatMulrnn/odelstm/mul_2:z:0+rnn/odelstm/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_2�
#rnn/odelstm/MatMul_3/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_3/ReadVariableOp�
rnn/odelstm/MatMul_3MatMulrnn/zeros_1:output:0+rnn/odelstm/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_3�
rnn/odelstm/add_4AddV2rnn/odelstm/MatMul_2:product:0rnn/odelstm/MatMul_3:product:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_4�
 rnn/odelstm/add_5/ReadVariableOpReadVariableOp)rnn_odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02"
 rnn/odelstm/add_5/ReadVariableOp�
rnn/odelstm/add_5AddV2rnn/odelstm/add_4:z:0(rnn/odelstm/add_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_5q
rnn/odelstm/Tanh_2Tanhrnn/odelstm/add_5:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/Tanh_2�
rnn/odelstm/ReadVariableOpReadVariableOp#rnn_odelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
rnn/odelstm/ReadVariableOp�
rnn/odelstm/mul_3Mul"rnn/odelstm/ReadVariableOp:value:0rnn/odelstm/Tanh_2:y:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_3o
rnn/odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/odelstm/mul_4/y�
rnn/odelstm/mul_4Mulrnn/zeros_1:output:0rnn/odelstm/mul_4/y:output:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_4�
rnn/odelstm/subSubrnn/odelstm/mul_3:z:0rnn/odelstm/mul_4:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/subo
rnn/odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/odelstm/mul_5/x�
rnn/odelstm/mul_5Mulrnn/odelstm/mul_5/x:output:0rnn/odelstm/sub:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_5�
rnn/odelstm/add_6AddV2rnn/zeros_1:output:0rnn/odelstm/mul_5:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_6�
#rnn/odelstm/MatMul_4/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_4/ReadVariableOp�
rnn/odelstm/MatMul_4MatMulrnn/odelstm/mul_2:z:0+rnn/odelstm/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_4�
#rnn/odelstm/MatMul_5/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_5/ReadVariableOp�
rnn/odelstm/MatMul_5MatMulrnn/odelstm/add_6:z:0+rnn/odelstm/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_5�
rnn/odelstm/add_7AddV2rnn/odelstm/MatMul_4:product:0rnn/odelstm/MatMul_5:product:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_7�
 rnn/odelstm/add_8/ReadVariableOpReadVariableOp)rnn_odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02"
 rnn/odelstm/add_8/ReadVariableOp�
rnn/odelstm/add_8AddV2rnn/odelstm/add_7:z:0(rnn/odelstm/add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_8q
rnn/odelstm/Tanh_3Tanhrnn/odelstm/add_8:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/Tanh_3�
rnn/odelstm/ReadVariableOp_1ReadVariableOp#rnn_odelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
rnn/odelstm/ReadVariableOp_1�
rnn/odelstm/mul_6Mul$rnn/odelstm/ReadVariableOp_1:value:0rnn/odelstm/Tanh_3:y:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_6o
rnn/odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/odelstm/mul_7/y�
rnn/odelstm/mul_7Mulrnn/odelstm/add_6:z:0rnn/odelstm/mul_7/y:output:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_7�
rnn/odelstm/sub_1Subrnn/odelstm/mul_6:z:0rnn/odelstm/mul_7:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/sub_1o
rnn/odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/odelstm/mul_8/x�
rnn/odelstm/mul_8Mulrnn/odelstm/mul_8/x:output:0rnn/odelstm/sub_1:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_8�
rnn/odelstm/add_9AddV2rnn/odelstm/add_6:z:0rnn/odelstm/mul_8:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_9�
#rnn/odelstm/MatMul_6/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_6/ReadVariableOp�
rnn/odelstm/MatMul_6MatMulrnn/odelstm/mul_2:z:0+rnn/odelstm/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_6�
#rnn/odelstm/MatMul_7/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_7/ReadVariableOp�
rnn/odelstm/MatMul_7MatMulrnn/odelstm/add_9:z:0+rnn/odelstm/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_7�
rnn/odelstm/add_10AddV2rnn/odelstm/MatMul_6:product:0rnn/odelstm/MatMul_7:product:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_10�
!rnn/odelstm/add_11/ReadVariableOpReadVariableOp)rnn_odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02#
!rnn/odelstm/add_11/ReadVariableOp�
rnn/odelstm/add_11AddV2rnn/odelstm/add_10:z:0)rnn/odelstm/add_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_11r
rnn/odelstm/Tanh_4Tanhrnn/odelstm/add_11:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/Tanh_4�
rnn/odelstm/ReadVariableOp_2ReadVariableOp#rnn_odelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
rnn/odelstm/ReadVariableOp_2�
rnn/odelstm/mul_9Mul$rnn/odelstm/ReadVariableOp_2:value:0rnn/odelstm/Tanh_4:y:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_9q
rnn/odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/odelstm/mul_10/y�
rnn/odelstm/mul_10Mulrnn/odelstm/add_9:z:0rnn/odelstm/mul_10/y:output:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_10�
rnn/odelstm/sub_2Subrnn/odelstm/mul_9:z:0rnn/odelstm/mul_10:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/sub_2q
rnn/odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/odelstm/mul_11/x�
rnn/odelstm/mul_11Mulrnn/odelstm/mul_11/x:output:0rnn/odelstm/sub_2:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_11�
rnn/odelstm/add_12AddV2rnn/odelstm/add_9:z:0rnn/odelstm/mul_11:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_12�
#rnn/odelstm/MatMul_8/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_8/ReadVariableOp�
rnn/odelstm/MatMul_8MatMulrnn/odelstm/mul_2:z:0+rnn/odelstm/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_8�
#rnn/odelstm/MatMul_9/ReadVariableOpReadVariableOp,rnn_odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#rnn/odelstm/MatMul_9/ReadVariableOp�
rnn/odelstm/MatMul_9MatMulrnn/odelstm/add_12:z:0+rnn/odelstm/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/MatMul_9�
rnn/odelstm/add_13AddV2rnn/odelstm/MatMul_8:product:0rnn/odelstm/MatMul_9:product:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_13�
!rnn/odelstm/add_14/ReadVariableOpReadVariableOp)rnn_odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02#
!rnn/odelstm/add_14/ReadVariableOp�
rnn/odelstm/add_14AddV2rnn/odelstm/add_13:z:0)rnn/odelstm/add_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_14r
rnn/odelstm/Tanh_5Tanhrnn/odelstm/add_14:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/Tanh_5�
rnn/odelstm/ReadVariableOp_3ReadVariableOp#rnn_odelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
rnn/odelstm/ReadVariableOp_3�
rnn/odelstm/mul_12Mul$rnn/odelstm/ReadVariableOp_3:value:0rnn/odelstm/Tanh_5:y:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_12q
rnn/odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/odelstm/mul_13/y�
rnn/odelstm/mul_13Mulrnn/odelstm/add_12:z:0rnn/odelstm/mul_13/y:output:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_13�
rnn/odelstm/sub_3Subrnn/odelstm/mul_12:z:0rnn/odelstm/mul_13:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/sub_3q
rnn/odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/odelstm/mul_14/x�
rnn/odelstm/mul_14Mulrnn/odelstm/mul_14/x:output:0rnn/odelstm/sub_3:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/mul_14�
rnn/odelstm/add_15AddV2rnn/odelstm/add_12:z:0rnn/odelstm/mul_14:z:0*
T0*
_output_shapes
:	�@2
rnn/odelstm/add_15�
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�   @   2#
!rnn/TensorArrayV2_1/element_shape�
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2_1V
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn/time�
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
rnn/while/maximum_iterationsr
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/while/loop_counter�
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0*rnn_odelstm_matmul_readvariableop_resource,rnn_odelstm_matmul_1_readvariableop_resource)rnn_odelstm_add_1_readvariableop_resource,rnn_odelstm_matmul_2_readvariableop_resource,rnn_odelstm_matmul_3_readvariableop_resource)rnn_odelstm_add_5_readvariableop_resource#rnn_odelstm_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*D
_output_shapes2
0: : : : :	�@:	�@: : : : : : : : : *)
_read_only_resource_inputs
		
* 
bodyR
rnn_while_body_59784* 
condR
rnn_while_cond_59783*C
output_shapes2
0: : : : :	�@:	�@: : : : : : : : : *
parallel_iterations 2
	rnn/while�
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�   @   26
4rnn/TensorArrayV2Stack/TensorListStack/element_shape�
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:d�@*
element_dtype02(
&rnn/TensorArrayV2Stack/TensorListStack�
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
rnn/strided_slice_3/stack�
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_3/stack_1�
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_3/stack_2�
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�@*
shrink_axis_mask2
rnn/strided_slice_3�
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose_1/perm�
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*#
_output_shapes
:�d@2
rnn/transpose_1�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulrnn/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
dense/BiasAdd�
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp"^rnn/odelstm/MatMul/ReadVariableOp$^rnn/odelstm/MatMul_1/ReadVariableOp$^rnn/odelstm/MatMul_2/ReadVariableOp$^rnn/odelstm/MatMul_3/ReadVariableOp$^rnn/odelstm/MatMul_4/ReadVariableOp$^rnn/odelstm/MatMul_5/ReadVariableOp$^rnn/odelstm/MatMul_6/ReadVariableOp$^rnn/odelstm/MatMul_7/ReadVariableOp$^rnn/odelstm/MatMul_8/ReadVariableOp$^rnn/odelstm/MatMul_9/ReadVariableOp^rnn/odelstm/ReadVariableOp^rnn/odelstm/ReadVariableOp_1^rnn/odelstm/ReadVariableOp_2^rnn/odelstm/ReadVariableOp_3!^rnn/odelstm/add_1/ReadVariableOp"^rnn/odelstm/add_11/ReadVariableOp"^rnn/odelstm/add_14/ReadVariableOp!^rnn/odelstm/add_5/ReadVariableOp!^rnn/odelstm/add_8/ReadVariableOp
^rnn/while*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:�d:�d:::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2F
!rnn/odelstm/MatMul/ReadVariableOp!rnn/odelstm/MatMul/ReadVariableOp2J
#rnn/odelstm/MatMul_1/ReadVariableOp#rnn/odelstm/MatMul_1/ReadVariableOp2J
#rnn/odelstm/MatMul_2/ReadVariableOp#rnn/odelstm/MatMul_2/ReadVariableOp2J
#rnn/odelstm/MatMul_3/ReadVariableOp#rnn/odelstm/MatMul_3/ReadVariableOp2J
#rnn/odelstm/MatMul_4/ReadVariableOp#rnn/odelstm/MatMul_4/ReadVariableOp2J
#rnn/odelstm/MatMul_5/ReadVariableOp#rnn/odelstm/MatMul_5/ReadVariableOp2J
#rnn/odelstm/MatMul_6/ReadVariableOp#rnn/odelstm/MatMul_6/ReadVariableOp2J
#rnn/odelstm/MatMul_7/ReadVariableOp#rnn/odelstm/MatMul_7/ReadVariableOp2J
#rnn/odelstm/MatMul_8/ReadVariableOp#rnn/odelstm/MatMul_8/ReadVariableOp2J
#rnn/odelstm/MatMul_9/ReadVariableOp#rnn/odelstm/MatMul_9/ReadVariableOp28
rnn/odelstm/ReadVariableOprnn/odelstm/ReadVariableOp2<
rnn/odelstm/ReadVariableOp_1rnn/odelstm/ReadVariableOp_12<
rnn/odelstm/ReadVariableOp_2rnn/odelstm/ReadVariableOp_22<
rnn/odelstm/ReadVariableOp_3rnn/odelstm/ReadVariableOp_32D
 rnn/odelstm/add_1/ReadVariableOp rnn/odelstm/add_1/ReadVariableOp2F
!rnn/odelstm/add_11/ReadVariableOp!rnn/odelstm/add_11/ReadVariableOp2F
!rnn/odelstm/add_14/ReadVariableOp!rnn/odelstm/add_14/ReadVariableOp2D
 rnn/odelstm/add_5/ReadVariableOp rnn/odelstm/add_5/ReadVariableOp2D
 rnn/odelstm/add_8/ReadVariableOp rnn/odelstm/add_8/ReadVariableOp2
	rnn/while	rnn/while:M I
#
_output_shapes
:�d
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:�d
"
_user_specified_name
inputs/1
�i
�
B__inference_odelstm_layer_call_and_return_conditional_losses_58199

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_1_readvariableop_resource$
 matmul_2_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_5_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�MatMul_2/ReadVariableOp�MatMul_3/ReadVariableOp�MatMul_4/ReadVariableOp�MatMul_5/ReadVariableOp�MatMul_6/ReadVariableOp�MatMul_7/ReadVariableOp�MatMul_8/ReadVariableOp�MatMul_9/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�add_1/ReadVariableOp�add_11/ReadVariableOp�add_14/ReadVariableOp�add_5/ReadVariableOp�add_8/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_1MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
add_1/ReadVariableOpq
add_1AddV2add:z:0add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
add_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0	add_1:z:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
splitV
TanhTanhsplit:output:0*
T0*'
_output_shapes
:���������@2
Tanh_
SigmoidSigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2	
SigmoidW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2	
add_2/yk
add_2AddV2split:output:2add_2/y:output:0*
T0*'
_output_shapes
:���������@2
add_2^
	Sigmoid_1Sigmoid	add_2:z:0*
T0*'
_output_shapes
:���������@2
	Sigmoid_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2Z
mulMulstatesSigmoid_1:y:0*
T0*'
_output_shapes
:���������@2
mul^
mul_1MulTanh:y:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_3AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_3U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:���������@2
Tanh_1b
mul_2Mul
Tanh_1:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������@2
mul_2�
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_2/ReadVariableOp|
MatMul_2MatMul	mul_2:z:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_2�
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_3/ReadVariableOp{
MatMul_3MatMulstates_1MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_3q
add_4AddV2MatMul_2:product:0MatMul_3:product:0*
T0*'
_output_shapes
:���������@2
add_4�
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:@*
dtype02
add_5/ReadVariableOpr
add_5AddV2	add_4:z:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
add_5U
Tanh_2Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@2
Tanh_2t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpk
mul_3MulReadVariableOp:value:0
Tanh_2:y:0*
T0*'
_output_shapes
:���������@2
mul_3W
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
mul_4/yc
mul_4Mulstates_1mul_4/y:output:0*
T0*'
_output_shapes
:���������@2
mul_4Y
subSub	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:���������@2
subW
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2	
mul_5/xb
mul_5Mulmul_5/x:output:0sub:z:0*
T0*'
_output_shapes
:���������@2
mul_5^
add_6AddV2states_1	mul_5:z:0*
T0*'
_output_shapes
:���������@2
add_6�
MatMul_4/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_4/ReadVariableOp|
MatMul_4MatMul	mul_2:z:0MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_4�
MatMul_5/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_5/ReadVariableOp|
MatMul_5MatMul	add_6:z:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_5q
add_7AddV2MatMul_4:product:0MatMul_5:product:0*
T0*'
_output_shapes
:���������@2
add_7�
add_8/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:@*
dtype02
add_8/ReadVariableOpr
add_8AddV2	add_7:z:0add_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
add_8U
Tanh_3Tanh	add_8:z:0*
T0*'
_output_shapes
:���������@2
Tanh_3x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1m
mul_6MulReadVariableOp_1:value:0
Tanh_3:y:0*
T0*'
_output_shapes
:���������@2
mul_6W
mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
mul_7/yd
mul_7Mul	add_6:z:0mul_7/y:output:0*
T0*'
_output_shapes
:���������@2
mul_7]
sub_1Sub	mul_6:z:0	mul_7:z:0*
T0*'
_output_shapes
:���������@2
sub_1W
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2	
mul_8/xd
mul_8Mulmul_8/x:output:0	sub_1:z:0*
T0*'
_output_shapes
:���������@2
mul_8_
add_9AddV2	add_6:z:0	mul_8:z:0*
T0*'
_output_shapes
:���������@2
add_9�
MatMul_6/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_6/ReadVariableOp|
MatMul_6MatMul	mul_2:z:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_6�
MatMul_7/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_7/ReadVariableOp|
MatMul_7MatMul	add_9:z:0MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_7s
add_10AddV2MatMul_6:product:0MatMul_7:product:0*
T0*'
_output_shapes
:���������@2
add_10�
add_11/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:@*
dtype02
add_11/ReadVariableOpv
add_11AddV2
add_10:z:0add_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
add_11V
Tanh_4Tanh
add_11:z:0*
T0*'
_output_shapes
:���������@2
Tanh_4x
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_2m
mul_9MulReadVariableOp_2:value:0
Tanh_4:y:0*
T0*'
_output_shapes
:���������@2
mul_9Y
mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

mul_10/yg
mul_10Mul	add_9:z:0mul_10/y:output:0*
T0*'
_output_shapes
:���������@2
mul_10^
sub_2Sub	mul_9:z:0
mul_10:z:0*
T0*'
_output_shapes
:���������@2
sub_2Y
mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2

mul_11/xg
mul_11Mulmul_11/x:output:0	sub_2:z:0*
T0*'
_output_shapes
:���������@2
mul_11b
add_12AddV2	add_9:z:0
mul_11:z:0*
T0*'
_output_shapes
:���������@2
add_12�
MatMul_8/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_8/ReadVariableOp|
MatMul_8MatMul	mul_2:z:0MatMul_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_8�
MatMul_9/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_9/ReadVariableOp}
MatMul_9MatMul
add_12:z:0MatMul_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_9s
add_13AddV2MatMul_8:product:0MatMul_9:product:0*
T0*'
_output_shapes
:���������@2
add_13�
add_14/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:@*
dtype02
add_14/ReadVariableOpv
add_14AddV2
add_13:z:0add_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
add_14V
Tanh_5Tanh
add_14:z:0*
T0*'
_output_shapes
:���������@2
Tanh_5x
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_3o
mul_12MulReadVariableOp_3:value:0
Tanh_5:y:0*
T0*'
_output_shapes
:���������@2
mul_12Y
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

mul_13/yh
mul_13Mul
add_12:z:0mul_13/y:output:0*
T0*'
_output_shapes
:���������@2
mul_13_
sub_3Sub
mul_12:z:0
mul_13:z:0*
T0*'
_output_shapes
:���������@2
sub_3Y
mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2

mul_14/xg
mul_14Mulmul_14/x:output:0	sub_3:z:0*
T0*'
_output_shapes
:���������@2
mul_14c
add_15AddV2
add_12:z:0
mul_14:z:0*
T0*'
_output_shapes
:���������@2
add_15�
IdentityIdentity
add_15:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^MatMul_8/ReadVariableOp^MatMul_9/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^add_1/ReadVariableOp^add_11/ReadVariableOp^add_14/ReadVariableOp^add_5/ReadVariableOp^add_8/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^MatMul_8/ReadVariableOp^MatMul_9/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^add_1/ReadVariableOp^add_11/ReadVariableOp^add_14/ReadVariableOp^add_5/ReadVariableOp^add_8/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity
add_15:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^MatMul_8/ReadVariableOp^MatMul_9/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^add_1/ReadVariableOp^add_11/ReadVariableOp^add_14/ReadVariableOp^add_5/ReadVariableOp^add_8/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*h
_input_shapesW
U:���������:���������@:���������@:::::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp22
MatMul_7/ReadVariableOpMatMul_7/ReadVariableOp22
MatMul_8/ReadVariableOpMatMul_8/ReadVariableOp22
MatMul_9/ReadVariableOpMatMul_9/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
add_1/ReadVariableOpadd_1/ReadVariableOp2.
add_11/ReadVariableOpadd_11/ReadVariableOp2.
add_14/ReadVariableOpadd_14/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_8/ReadVariableOpadd_8/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
��
�
>__inference_rnn_layer_call_and_return_conditional_losses_59106

inputs*
&odelstm_matmul_readvariableop_resource,
(odelstm_matmul_1_readvariableop_resource)
%odelstm_add_1_readvariableop_resource,
(odelstm_matmul_2_readvariableop_resource,
(odelstm_matmul_3_readvariableop_resource)
%odelstm_add_5_readvariableop_resource#
odelstm_readvariableop_resource
identity��odelstm/MatMul/ReadVariableOp�odelstm/MatMul_1/ReadVariableOp�odelstm/MatMul_2/ReadVariableOp�odelstm/MatMul_3/ReadVariableOp�odelstm/MatMul_4/ReadVariableOp�odelstm/MatMul_5/ReadVariableOp�odelstm/MatMul_6/ReadVariableOp�odelstm/MatMul_7/ReadVariableOp�odelstm/MatMul_8/ReadVariableOp�odelstm/MatMul_9/ReadVariableOp�odelstm/ReadVariableOp�odelstm/ReadVariableOp_1�odelstm/ReadVariableOp_2�odelstm/ReadVariableOp_3�odelstm/add_1/ReadVariableOp�odelstm/add_11/ReadVariableOp�odelstm/add_14/ReadVariableOp�odelstm/add_5/ReadVariableOp�odelstm/add_8/ReadVariableOp�whilec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"�   d      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constm
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*
_output_shapes
:	�@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constu
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*
_output_shapes
:	�@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permr
	transpose	Transposeinputstranspose/perm:output:0*
T0*#
_output_shapes
:d�2
	transposeg
Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   �      2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�      27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*
shrink_axis_mask2
strided_slice_2�
odelstm/MatMul/ReadVariableOpReadVariableOp&odelstm_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
odelstm/MatMul/ReadVariableOp�
odelstm/MatMulMatMulstrided_slice_2:output:0%odelstm/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
odelstm/MatMul�
odelstm/MatMul_1/ReadVariableOpReadVariableOp(odelstm_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02!
odelstm/MatMul_1/ReadVariableOp�
odelstm/MatMul_1MatMulzeros_1:output:0'odelstm/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
odelstm/MatMul_1�
odelstm/addAddV2odelstm/MatMul:product:0odelstm/MatMul_1:product:0*
T0* 
_output_shapes
:
��2
odelstm/add�
odelstm/add_1/ReadVariableOpReadVariableOp%odelstm_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
odelstm/add_1/ReadVariableOp�
odelstm/add_1AddV2odelstm/add:z:0$odelstm/add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
odelstm/add_1`
odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
odelstm/Const}
odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
odelstm/split/split_dim�
odelstm/splitSplit odelstm/split/split_dim:output:0odelstm/add_1:z:0*
T0*@
_output_shapes.
,:	�@:	�@:	�@:	�@*
	num_split2
odelstm/splitf
odelstm/TanhTanhodelstm/split:output:0*
T0*
_output_shapes
:	�@2
odelstm/Tanho
odelstm/SigmoidSigmoidodelstm/split:output:1*
T0*
_output_shapes
:	�@2
odelstm/Sigmoidg
odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
odelstm/add_2/y�
odelstm/add_2AddV2odelstm/split:output:2odelstm/add_2/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/add_2n
odelstm/Sigmoid_1Sigmoidodelstm/add_2:z:0*
T0*
_output_shapes
:	�@2
odelstm/Sigmoid_1s
odelstm/Sigmoid_2Sigmoidodelstm/split:output:3*
T0*
_output_shapes
:	�@2
odelstm/Sigmoid_2r
odelstm/mulMulzeros:output:0odelstm/Sigmoid_1:y:0*
T0*
_output_shapes
:	�@2
odelstm/mulv
odelstm/mul_1Mulodelstm/Tanh:y:0odelstm/Sigmoid:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_1u
odelstm/add_3AddV2odelstm/mul:z:0odelstm/mul_1:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_3e
odelstm/Tanh_1Tanhodelstm/add_3:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_1z
odelstm/mul_2Mulodelstm/Tanh_1:y:0odelstm/Sigmoid_2:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_2�
odelstm/MatMul_2/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_2/ReadVariableOp�
odelstm/MatMul_2MatMulodelstm/mul_2:z:0'odelstm/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_2�
odelstm/MatMul_3/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_3/ReadVariableOp�
odelstm/MatMul_3MatMulzeros_1:output:0'odelstm/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_3�
odelstm/add_4AddV2odelstm/MatMul_2:product:0odelstm/MatMul_3:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_4�
odelstm/add_5/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_5/ReadVariableOp�
odelstm/add_5AddV2odelstm/add_4:z:0$odelstm/add_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_5e
odelstm/Tanh_2Tanhodelstm/add_5:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_2�
odelstm/ReadVariableOpReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp�
odelstm/mul_3Mulodelstm/ReadVariableOp:value:0odelstm/Tanh_2:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_3g
odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_4/y{
odelstm/mul_4Mulzeros_1:output:0odelstm/mul_4/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_4q
odelstm/subSubodelstm/mul_3:z:0odelstm/mul_4:z:0*
T0*
_output_shapes
:	�@2
odelstm/subg
odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_5/xz
odelstm/mul_5Mulodelstm/mul_5/x:output:0odelstm/sub:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_5v
odelstm/add_6AddV2zeros_1:output:0odelstm/mul_5:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_6�
odelstm/MatMul_4/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_4/ReadVariableOp�
odelstm/MatMul_4MatMulodelstm/mul_2:z:0'odelstm/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_4�
odelstm/MatMul_5/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_5/ReadVariableOp�
odelstm/MatMul_5MatMulodelstm/add_6:z:0'odelstm/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_5�
odelstm/add_7AddV2odelstm/MatMul_4:product:0odelstm/MatMul_5:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_7�
odelstm/add_8/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_8/ReadVariableOp�
odelstm/add_8AddV2odelstm/add_7:z:0$odelstm/add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_8e
odelstm/Tanh_3Tanhodelstm/add_8:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_3�
odelstm/ReadVariableOp_1ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_1�
odelstm/mul_6Mul odelstm/ReadVariableOp_1:value:0odelstm/Tanh_3:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_6g
odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_7/y|
odelstm/mul_7Mulodelstm/add_6:z:0odelstm/mul_7/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_7u
odelstm/sub_1Subodelstm/mul_6:z:0odelstm/mul_7:z:0*
T0*
_output_shapes
:	�@2
odelstm/sub_1g
odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_8/x|
odelstm/mul_8Mulodelstm/mul_8/x:output:0odelstm/sub_1:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_8w
odelstm/add_9AddV2odelstm/add_6:z:0odelstm/mul_8:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_9�
odelstm/MatMul_6/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_6/ReadVariableOp�
odelstm/MatMul_6MatMulodelstm/mul_2:z:0'odelstm/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_6�
odelstm/MatMul_7/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_7/ReadVariableOp�
odelstm/MatMul_7MatMulodelstm/add_9:z:0'odelstm/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_7�
odelstm/add_10AddV2odelstm/MatMul_6:product:0odelstm/MatMul_7:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_10�
odelstm/add_11/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_11/ReadVariableOp�
odelstm/add_11AddV2odelstm/add_10:z:0%odelstm/add_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_11f
odelstm/Tanh_4Tanhodelstm/add_11:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_4�
odelstm/ReadVariableOp_2ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_2�
odelstm/mul_9Mul odelstm/ReadVariableOp_2:value:0odelstm/Tanh_4:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_9i
odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_10/y
odelstm/mul_10Mulodelstm/add_9:z:0odelstm/mul_10/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_10v
odelstm/sub_2Subodelstm/mul_9:z:0odelstm/mul_10:z:0*
T0*
_output_shapes
:	�@2
odelstm/sub_2i
odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_11/x
odelstm/mul_11Mulodelstm/mul_11/x:output:0odelstm/sub_2:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_11z
odelstm/add_12AddV2odelstm/add_9:z:0odelstm/mul_11:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_12�
odelstm/MatMul_8/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_8/ReadVariableOp�
odelstm/MatMul_8MatMulodelstm/mul_2:z:0'odelstm/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_8�
odelstm/MatMul_9/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_9/ReadVariableOp�
odelstm/MatMul_9MatMulodelstm/add_12:z:0'odelstm/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_9�
odelstm/add_13AddV2odelstm/MatMul_8:product:0odelstm/MatMul_9:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_13�
odelstm/add_14/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_14/ReadVariableOp�
odelstm/add_14AddV2odelstm/add_13:z:0%odelstm/add_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_14f
odelstm/Tanh_5Tanhodelstm/add_14:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_5�
odelstm/ReadVariableOp_3ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_3�
odelstm/mul_12Mul odelstm/ReadVariableOp_3:value:0odelstm/Tanh_5:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_12i
odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_13/y�
odelstm/mul_13Mulodelstm/add_12:z:0odelstm/mul_13/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_13w
odelstm/sub_3Subodelstm/mul_12:z:0odelstm/mul_13:z:0*
T0*
_output_shapes
:	�@2
odelstm/sub_3i
odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_14/x
odelstm/mul_14Mulodelstm/mul_14/x:output:0odelstm/sub_3:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_14{
odelstm/add_15AddV2odelstm/add_12:z:0odelstm/mul_14:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_15�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�   @   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0&odelstm_matmul_readvariableop_resource(odelstm_matmul_1_readvariableop_resource%odelstm_add_1_readvariableop_resource(odelstm_matmul_2_readvariableop_resource(odelstm_matmul_3_readvariableop_resource%odelstm_add_5_readvariableop_resourceodelstm_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*D
_output_shapes2
0: : : : :	�@:	�@: : : : : : : : : *)
_read_only_resource_inputs
		
*
bodyR
while_body_58944*
condR
while_cond_58943*C
output_shapes2
0: : : : :	�@:	�@: : : : : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�   @   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:d�@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*#
_output_shapes
:�d@2
transpose_1�
IdentityIdentitystrided_slice_3:output:0^odelstm/MatMul/ReadVariableOp ^odelstm/MatMul_1/ReadVariableOp ^odelstm/MatMul_2/ReadVariableOp ^odelstm/MatMul_3/ReadVariableOp ^odelstm/MatMul_4/ReadVariableOp ^odelstm/MatMul_5/ReadVariableOp ^odelstm/MatMul_6/ReadVariableOp ^odelstm/MatMul_7/ReadVariableOp ^odelstm/MatMul_8/ReadVariableOp ^odelstm/MatMul_9/ReadVariableOp^odelstm/ReadVariableOp^odelstm/ReadVariableOp_1^odelstm/ReadVariableOp_2^odelstm/ReadVariableOp_3^odelstm/add_1/ReadVariableOp^odelstm/add_11/ReadVariableOp^odelstm/add_14/ReadVariableOp^odelstm/add_5/ReadVariableOp^odelstm/add_8/ReadVariableOp^while*
T0*
_output_shapes
:	�@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:�d:::::::2>
odelstm/MatMul/ReadVariableOpodelstm/MatMul/ReadVariableOp2B
odelstm/MatMul_1/ReadVariableOpodelstm/MatMul_1/ReadVariableOp2B
odelstm/MatMul_2/ReadVariableOpodelstm/MatMul_2/ReadVariableOp2B
odelstm/MatMul_3/ReadVariableOpodelstm/MatMul_3/ReadVariableOp2B
odelstm/MatMul_4/ReadVariableOpodelstm/MatMul_4/ReadVariableOp2B
odelstm/MatMul_5/ReadVariableOpodelstm/MatMul_5/ReadVariableOp2B
odelstm/MatMul_6/ReadVariableOpodelstm/MatMul_6/ReadVariableOp2B
odelstm/MatMul_7/ReadVariableOpodelstm/MatMul_7/ReadVariableOp2B
odelstm/MatMul_8/ReadVariableOpodelstm/MatMul_8/ReadVariableOp2B
odelstm/MatMul_9/ReadVariableOpodelstm/MatMul_9/ReadVariableOp20
odelstm/ReadVariableOpodelstm/ReadVariableOp24
odelstm/ReadVariableOp_1odelstm/ReadVariableOp_124
odelstm/ReadVariableOp_2odelstm/ReadVariableOp_224
odelstm/ReadVariableOp_3odelstm/ReadVariableOp_32<
odelstm/add_1/ReadVariableOpodelstm/add_1/ReadVariableOp2>
odelstm/add_11/ReadVariableOpodelstm/add_11/ReadVariableOp2>
odelstm/add_14/ReadVariableOpodelstm/add_14/ReadVariableOp2<
odelstm/add_5/ReadVariableOpodelstm/add_5/ReadVariableOp2<
odelstm/add_8/ReadVariableOpodelstm/add_8/ReadVariableOp2
whilewhile:K G
#
_output_shapes
:�d
 
_user_specified_nameinputs
ʾ
�
while_body_59244
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
.while_odelstm_matmul_readvariableop_resource_04
0while_odelstm_matmul_1_readvariableop_resource_01
-while_odelstm_add_1_readvariableop_resource_04
0while_odelstm_matmul_2_readvariableop_resource_04
0while_odelstm_matmul_3_readvariableop_resource_01
-while_odelstm_add_5_readvariableop_resource_0+
'while_odelstm_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
,while_odelstm_matmul_readvariableop_resource2
.while_odelstm_matmul_1_readvariableop_resource/
+while_odelstm_add_1_readvariableop_resource2
.while_odelstm_matmul_2_readvariableop_resource2
.while_odelstm_matmul_3_readvariableop_resource/
+while_odelstm_add_5_readvariableop_resource)
%while_odelstm_readvariableop_resource��#while/odelstm/MatMul/ReadVariableOp�%while/odelstm/MatMul_1/ReadVariableOp�%while/odelstm/MatMul_2/ReadVariableOp�%while/odelstm/MatMul_3/ReadVariableOp�%while/odelstm/MatMul_4/ReadVariableOp�%while/odelstm/MatMul_5/ReadVariableOp�%while/odelstm/MatMul_6/ReadVariableOp�%while/odelstm/MatMul_7/ReadVariableOp�%while/odelstm/MatMul_8/ReadVariableOp�%while/odelstm/MatMul_9/ReadVariableOp�while/odelstm/ReadVariableOp�while/odelstm/ReadVariableOp_1�while/odelstm/ReadVariableOp_2�while/odelstm/ReadVariableOp_3�"while/odelstm/add_1/ReadVariableOp�#while/odelstm/add_11/ReadVariableOp�#while/odelstm/add_14/ReadVariableOp�"while/odelstm/add_5/ReadVariableOp�"while/odelstm/add_8/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	�*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
#while/odelstm/MatMul/ReadVariableOpReadVariableOp.while_odelstm_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02%
#while/odelstm/MatMul/ReadVariableOp�
while/odelstm/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/odelstm/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
while/odelstm/MatMul�
%while/odelstm/MatMul_1/ReadVariableOpReadVariableOp0while_odelstm_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02'
%while/odelstm/MatMul_1/ReadVariableOp�
while/odelstm/MatMul_1MatMulwhile_placeholder_3-while/odelstm/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
while/odelstm/MatMul_1�
while/odelstm/addAddV2while/odelstm/MatMul:product:0 while/odelstm/MatMul_1:product:0*
T0* 
_output_shapes
:
��2
while/odelstm/add�
"while/odelstm/add_1/ReadVariableOpReadVariableOp-while_odelstm_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype02$
"while/odelstm/add_1/ReadVariableOp�
while/odelstm/add_1AddV2while/odelstm/add:z:0*while/odelstm/add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
while/odelstm/add_1l
while/odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/odelstm/Const�
while/odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/odelstm/split/split_dim�
while/odelstm/splitSplit&while/odelstm/split/split_dim:output:0while/odelstm/add_1:z:0*
T0*@
_output_shapes.
,:	�@:	�@:	�@:	�@*
	num_split2
while/odelstm/splitx
while/odelstm/TanhTanhwhile/odelstm/split:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh�
while/odelstm/SigmoidSigmoidwhile/odelstm/split:output:1*
T0*
_output_shapes
:	�@2
while/odelstm/Sigmoids
while/odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
while/odelstm/add_2/y�
while/odelstm/add_2AddV2while/odelstm/split:output:2while/odelstm/add_2/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_2�
while/odelstm/Sigmoid_1Sigmoidwhile/odelstm/add_2:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Sigmoid_1�
while/odelstm/Sigmoid_2Sigmoidwhile/odelstm/split:output:3*
T0*
_output_shapes
:	�@2
while/odelstm/Sigmoid_2�
while/odelstm/mulMulwhile_placeholder_2while/odelstm/Sigmoid_1:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul�
while/odelstm/mul_1Mulwhile/odelstm/Tanh:y:0while/odelstm/Sigmoid:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_1�
while/odelstm/add_3AddV2while/odelstm/mul:z:0while/odelstm/mul_1:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_3w
while/odelstm/Tanh_1Tanhwhile/odelstm/add_3:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_1�
while/odelstm/mul_2Mulwhile/odelstm/Tanh_1:y:0while/odelstm/Sigmoid_2:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_2�
%while/odelstm/MatMul_2/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_2/ReadVariableOp�
while/odelstm/MatMul_2MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_2�
%while/odelstm/MatMul_3/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_3/ReadVariableOp�
while/odelstm/MatMul_3MatMulwhile_placeholder_3-while/odelstm/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_3�
while/odelstm/add_4AddV2 while/odelstm/MatMul_2:product:0 while/odelstm/MatMul_3:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_4�
"while/odelstm/add_5/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"while/odelstm/add_5/ReadVariableOp�
while/odelstm/add_5AddV2while/odelstm/add_4:z:0*while/odelstm/add_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_5w
while/odelstm/Tanh_2Tanhwhile/odelstm/add_5:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_2�
while/odelstm/ReadVariableOpReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02
while/odelstm/ReadVariableOp�
while/odelstm/mul_3Mul$while/odelstm/ReadVariableOp:value:0while/odelstm/Tanh_2:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_3s
while/odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_4/y�
while/odelstm/mul_4Mulwhile_placeholder_3while/odelstm/mul_4/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_4�
while/odelstm/subSubwhile/odelstm/mul_3:z:0while/odelstm/mul_4:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/subs
while/odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_5/x�
while/odelstm/mul_5Mulwhile/odelstm/mul_5/x:output:0while/odelstm/sub:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_5�
while/odelstm/add_6AddV2while_placeholder_3while/odelstm/mul_5:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_6�
%while/odelstm/MatMul_4/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_4/ReadVariableOp�
while/odelstm/MatMul_4MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_4�
%while/odelstm/MatMul_5/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_5/ReadVariableOp�
while/odelstm/MatMul_5MatMulwhile/odelstm/add_6:z:0-while/odelstm/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_5�
while/odelstm/add_7AddV2 while/odelstm/MatMul_4:product:0 while/odelstm/MatMul_5:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_7�
"while/odelstm/add_8/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"while/odelstm/add_8/ReadVariableOp�
while/odelstm/add_8AddV2while/odelstm/add_7:z:0*while/odelstm/add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_8w
while/odelstm/Tanh_3Tanhwhile/odelstm/add_8:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_3�
while/odelstm/ReadVariableOp_1ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_1�
while/odelstm/mul_6Mul&while/odelstm/ReadVariableOp_1:value:0while/odelstm/Tanh_3:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_6s
while/odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_7/y�
while/odelstm/mul_7Mulwhile/odelstm/add_6:z:0while/odelstm/mul_7/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_7�
while/odelstm/sub_1Subwhile/odelstm/mul_6:z:0while/odelstm/mul_7:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/sub_1s
while/odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_8/x�
while/odelstm/mul_8Mulwhile/odelstm/mul_8/x:output:0while/odelstm/sub_1:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_8�
while/odelstm/add_9AddV2while/odelstm/add_6:z:0while/odelstm/mul_8:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_9�
%while/odelstm/MatMul_6/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_6/ReadVariableOp�
while/odelstm/MatMul_6MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_6�
%while/odelstm/MatMul_7/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_7/ReadVariableOp�
while/odelstm/MatMul_7MatMulwhile/odelstm/add_9:z:0-while/odelstm/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_7�
while/odelstm/add_10AddV2 while/odelstm/MatMul_6:product:0 while/odelstm/MatMul_7:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_10�
#while/odelstm/add_11/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02%
#while/odelstm/add_11/ReadVariableOp�
while/odelstm/add_11AddV2while/odelstm/add_10:z:0+while/odelstm/add_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_11x
while/odelstm/Tanh_4Tanhwhile/odelstm/add_11:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_4�
while/odelstm/ReadVariableOp_2ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_2�
while/odelstm/mul_9Mul&while/odelstm/ReadVariableOp_2:value:0while/odelstm/Tanh_4:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_9u
while/odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_10/y�
while/odelstm/mul_10Mulwhile/odelstm/add_9:z:0while/odelstm/mul_10/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_10�
while/odelstm/sub_2Subwhile/odelstm/mul_9:z:0while/odelstm/mul_10:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/sub_2u
while/odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_11/x�
while/odelstm/mul_11Mulwhile/odelstm/mul_11/x:output:0while/odelstm/sub_2:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_11�
while/odelstm/add_12AddV2while/odelstm/add_9:z:0while/odelstm/mul_11:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_12�
%while/odelstm/MatMul_8/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_8/ReadVariableOp�
while/odelstm/MatMul_8MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_8�
%while/odelstm/MatMul_9/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_9/ReadVariableOp�
while/odelstm/MatMul_9MatMulwhile/odelstm/add_12:z:0-while/odelstm/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_9�
while/odelstm/add_13AddV2 while/odelstm/MatMul_8:product:0 while/odelstm/MatMul_9:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_13�
#while/odelstm/add_14/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02%
#while/odelstm/add_14/ReadVariableOp�
while/odelstm/add_14AddV2while/odelstm/add_13:z:0+while/odelstm/add_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_14x
while/odelstm/Tanh_5Tanhwhile/odelstm/add_14:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_5�
while/odelstm/ReadVariableOp_3ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_3�
while/odelstm/mul_12Mul&while/odelstm/ReadVariableOp_3:value:0while/odelstm/Tanh_5:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_12u
while/odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_13/y�
while/odelstm/mul_13Mulwhile/odelstm/add_12:z:0while/odelstm/mul_13/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_13�
while/odelstm/sub_3Subwhile/odelstm/mul_12:z:0while/odelstm/mul_13:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/sub_3u
while/odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_14/x�
while/odelstm/mul_14Mulwhile/odelstm/mul_14/x:output:0while/odelstm/sub_3:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_14�
while/odelstm/add_15AddV2while/odelstm/add_12:z:0while/odelstm/mul_14:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_15�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/odelstm/add_15:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/odelstm/add_3:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
:	�@2
while/Identity_4�
while/Identity_5Identitywhile/odelstm/add_15:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
:	�@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"\
+while_odelstm_add_1_readvariableop_resource-while_odelstm_add_1_readvariableop_resource_0"\
+while_odelstm_add_5_readvariableop_resource-while_odelstm_add_5_readvariableop_resource_0"b
.while_odelstm_matmul_1_readvariableop_resource0while_odelstm_matmul_1_readvariableop_resource_0"b
.while_odelstm_matmul_2_readvariableop_resource0while_odelstm_matmul_2_readvariableop_resource_0"b
.while_odelstm_matmul_3_readvariableop_resource0while_odelstm_matmul_3_readvariableop_resource_0"^
,while_odelstm_matmul_readvariableop_resource.while_odelstm_matmul_readvariableop_resource_0"P
%while_odelstm_readvariableop_resource'while_odelstm_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :	�@:	�@: : :::::::2J
#while/odelstm/MatMul/ReadVariableOp#while/odelstm/MatMul/ReadVariableOp2N
%while/odelstm/MatMul_1/ReadVariableOp%while/odelstm/MatMul_1/ReadVariableOp2N
%while/odelstm/MatMul_2/ReadVariableOp%while/odelstm/MatMul_2/ReadVariableOp2N
%while/odelstm/MatMul_3/ReadVariableOp%while/odelstm/MatMul_3/ReadVariableOp2N
%while/odelstm/MatMul_4/ReadVariableOp%while/odelstm/MatMul_4/ReadVariableOp2N
%while/odelstm/MatMul_5/ReadVariableOp%while/odelstm/MatMul_5/ReadVariableOp2N
%while/odelstm/MatMul_6/ReadVariableOp%while/odelstm/MatMul_6/ReadVariableOp2N
%while/odelstm/MatMul_7/ReadVariableOp%while/odelstm/MatMul_7/ReadVariableOp2N
%while/odelstm/MatMul_8/ReadVariableOp%while/odelstm/MatMul_8/ReadVariableOp2N
%while/odelstm/MatMul_9/ReadVariableOp%while/odelstm/MatMul_9/ReadVariableOp2<
while/odelstm/ReadVariableOpwhile/odelstm/ReadVariableOp2@
while/odelstm/ReadVariableOp_1while/odelstm/ReadVariableOp_12@
while/odelstm/ReadVariableOp_2while/odelstm/ReadVariableOp_22@
while/odelstm/ReadVariableOp_3while/odelstm/ReadVariableOp_32H
"while/odelstm/add_1/ReadVariableOp"while/odelstm/add_1/ReadVariableOp2J
#while/odelstm/add_11/ReadVariableOp#while/odelstm/add_11/ReadVariableOp2J
#while/odelstm/add_14/ReadVariableOp#while/odelstm/add_14/ReadVariableOp2H
"while/odelstm/add_5/ReadVariableOp"while/odelstm/add_5/ReadVariableOp2H
"while/odelstm/add_8/ReadVariableOp"while/odelstm/add_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :

_output_shapes
: 
��
�
>__inference_rnn_layer_call_and_return_conditional_losses_61545

inputs*
&odelstm_matmul_readvariableop_resource,
(odelstm_matmul_1_readvariableop_resource)
%odelstm_add_1_readvariableop_resource,
(odelstm_matmul_2_readvariableop_resource,
(odelstm_matmul_3_readvariableop_resource)
%odelstm_add_5_readvariableop_resource#
odelstm_readvariableop_resource
identity��odelstm/MatMul/ReadVariableOp�odelstm/MatMul_1/ReadVariableOp�odelstm/MatMul_2/ReadVariableOp�odelstm/MatMul_3/ReadVariableOp�odelstm/MatMul_4/ReadVariableOp�odelstm/MatMul_5/ReadVariableOp�odelstm/MatMul_6/ReadVariableOp�odelstm/MatMul_7/ReadVariableOp�odelstm/MatMul_8/ReadVariableOp�odelstm/MatMul_9/ReadVariableOp�odelstm/ReadVariableOp�odelstm/ReadVariableOp_1�odelstm/ReadVariableOp_2�odelstm/ReadVariableOp_3�odelstm/add_1/ReadVariableOp�odelstm/add_11/ReadVariableOp�odelstm/add_14/ReadVariableOp�odelstm/add_5/ReadVariableOp�odelstm/add_8/ReadVariableOp�whilec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"�   d      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constm
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*
_output_shapes
:	�@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constu
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*
_output_shapes
:	�@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permr
	transpose	Transposeinputstranspose/perm:output:0*
T0*#
_output_shapes
:d�2
	transposeg
Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   �      2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�      27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*
shrink_axis_mask2
strided_slice_2�
odelstm/MatMul/ReadVariableOpReadVariableOp&odelstm_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
odelstm/MatMul/ReadVariableOp�
odelstm/MatMulMatMulstrided_slice_2:output:0%odelstm/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
odelstm/MatMul�
odelstm/MatMul_1/ReadVariableOpReadVariableOp(odelstm_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02!
odelstm/MatMul_1/ReadVariableOp�
odelstm/MatMul_1MatMulzeros_1:output:0'odelstm/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
odelstm/MatMul_1�
odelstm/addAddV2odelstm/MatMul:product:0odelstm/MatMul_1:product:0*
T0* 
_output_shapes
:
��2
odelstm/add�
odelstm/add_1/ReadVariableOpReadVariableOp%odelstm_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
odelstm/add_1/ReadVariableOp�
odelstm/add_1AddV2odelstm/add:z:0$odelstm/add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
odelstm/add_1`
odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
odelstm/Const}
odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
odelstm/split/split_dim�
odelstm/splitSplit odelstm/split/split_dim:output:0odelstm/add_1:z:0*
T0*@
_output_shapes.
,:	�@:	�@:	�@:	�@*
	num_split2
odelstm/splitf
odelstm/TanhTanhodelstm/split:output:0*
T0*
_output_shapes
:	�@2
odelstm/Tanho
odelstm/SigmoidSigmoidodelstm/split:output:1*
T0*
_output_shapes
:	�@2
odelstm/Sigmoidg
odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
odelstm/add_2/y�
odelstm/add_2AddV2odelstm/split:output:2odelstm/add_2/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/add_2n
odelstm/Sigmoid_1Sigmoidodelstm/add_2:z:0*
T0*
_output_shapes
:	�@2
odelstm/Sigmoid_1s
odelstm/Sigmoid_2Sigmoidodelstm/split:output:3*
T0*
_output_shapes
:	�@2
odelstm/Sigmoid_2r
odelstm/mulMulzeros:output:0odelstm/Sigmoid_1:y:0*
T0*
_output_shapes
:	�@2
odelstm/mulv
odelstm/mul_1Mulodelstm/Tanh:y:0odelstm/Sigmoid:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_1u
odelstm/add_3AddV2odelstm/mul:z:0odelstm/mul_1:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_3e
odelstm/Tanh_1Tanhodelstm/add_3:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_1z
odelstm/mul_2Mulodelstm/Tanh_1:y:0odelstm/Sigmoid_2:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_2�
odelstm/MatMul_2/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_2/ReadVariableOp�
odelstm/MatMul_2MatMulodelstm/mul_2:z:0'odelstm/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_2�
odelstm/MatMul_3/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_3/ReadVariableOp�
odelstm/MatMul_3MatMulzeros_1:output:0'odelstm/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_3�
odelstm/add_4AddV2odelstm/MatMul_2:product:0odelstm/MatMul_3:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_4�
odelstm/add_5/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_5/ReadVariableOp�
odelstm/add_5AddV2odelstm/add_4:z:0$odelstm/add_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_5e
odelstm/Tanh_2Tanhodelstm/add_5:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_2�
odelstm/ReadVariableOpReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp�
odelstm/mul_3Mulodelstm/ReadVariableOp:value:0odelstm/Tanh_2:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_3g
odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_4/y{
odelstm/mul_4Mulzeros_1:output:0odelstm/mul_4/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_4q
odelstm/subSubodelstm/mul_3:z:0odelstm/mul_4:z:0*
T0*
_output_shapes
:	�@2
odelstm/subg
odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_5/xz
odelstm/mul_5Mulodelstm/mul_5/x:output:0odelstm/sub:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_5v
odelstm/add_6AddV2zeros_1:output:0odelstm/mul_5:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_6�
odelstm/MatMul_4/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_4/ReadVariableOp�
odelstm/MatMul_4MatMulodelstm/mul_2:z:0'odelstm/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_4�
odelstm/MatMul_5/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_5/ReadVariableOp�
odelstm/MatMul_5MatMulodelstm/add_6:z:0'odelstm/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_5�
odelstm/add_7AddV2odelstm/MatMul_4:product:0odelstm/MatMul_5:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_7�
odelstm/add_8/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_8/ReadVariableOp�
odelstm/add_8AddV2odelstm/add_7:z:0$odelstm/add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_8e
odelstm/Tanh_3Tanhodelstm/add_8:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_3�
odelstm/ReadVariableOp_1ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_1�
odelstm/mul_6Mul odelstm/ReadVariableOp_1:value:0odelstm/Tanh_3:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_6g
odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_7/y|
odelstm/mul_7Mulodelstm/add_6:z:0odelstm/mul_7/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_7u
odelstm/sub_1Subodelstm/mul_6:z:0odelstm/mul_7:z:0*
T0*
_output_shapes
:	�@2
odelstm/sub_1g
odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_8/x|
odelstm/mul_8Mulodelstm/mul_8/x:output:0odelstm/sub_1:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_8w
odelstm/add_9AddV2odelstm/add_6:z:0odelstm/mul_8:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_9�
odelstm/MatMul_6/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_6/ReadVariableOp�
odelstm/MatMul_6MatMulodelstm/mul_2:z:0'odelstm/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_6�
odelstm/MatMul_7/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_7/ReadVariableOp�
odelstm/MatMul_7MatMulodelstm/add_9:z:0'odelstm/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_7�
odelstm/add_10AddV2odelstm/MatMul_6:product:0odelstm/MatMul_7:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_10�
odelstm/add_11/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_11/ReadVariableOp�
odelstm/add_11AddV2odelstm/add_10:z:0%odelstm/add_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_11f
odelstm/Tanh_4Tanhodelstm/add_11:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_4�
odelstm/ReadVariableOp_2ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_2�
odelstm/mul_9Mul odelstm/ReadVariableOp_2:value:0odelstm/Tanh_4:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_9i
odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_10/y
odelstm/mul_10Mulodelstm/add_9:z:0odelstm/mul_10/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_10v
odelstm/sub_2Subodelstm/mul_9:z:0odelstm/mul_10:z:0*
T0*
_output_shapes
:	�@2
odelstm/sub_2i
odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_11/x
odelstm/mul_11Mulodelstm/mul_11/x:output:0odelstm/sub_2:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_11z
odelstm/add_12AddV2odelstm/add_9:z:0odelstm/mul_11:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_12�
odelstm/MatMul_8/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_8/ReadVariableOp�
odelstm/MatMul_8MatMulodelstm/mul_2:z:0'odelstm/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_8�
odelstm/MatMul_9/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_9/ReadVariableOp�
odelstm/MatMul_9MatMulodelstm/add_12:z:0'odelstm/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_9�
odelstm/add_13AddV2odelstm/MatMul_8:product:0odelstm/MatMul_9:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_13�
odelstm/add_14/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_14/ReadVariableOp�
odelstm/add_14AddV2odelstm/add_13:z:0%odelstm/add_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_14f
odelstm/Tanh_5Tanhodelstm/add_14:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_5�
odelstm/ReadVariableOp_3ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_3�
odelstm/mul_12Mul odelstm/ReadVariableOp_3:value:0odelstm/Tanh_5:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_12i
odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_13/y�
odelstm/mul_13Mulodelstm/add_12:z:0odelstm/mul_13/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_13w
odelstm/sub_3Subodelstm/mul_12:z:0odelstm/mul_13:z:0*
T0*
_output_shapes
:	�@2
odelstm/sub_3i
odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_14/x
odelstm/mul_14Mulodelstm/mul_14/x:output:0odelstm/sub_3:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_14{
odelstm/add_15AddV2odelstm/add_12:z:0odelstm/mul_14:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_15�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�   @   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0&odelstm_matmul_readvariableop_resource(odelstm_matmul_1_readvariableop_resource%odelstm_add_1_readvariableop_resource(odelstm_matmul_2_readvariableop_resource(odelstm_matmul_3_readvariableop_resource%odelstm_add_5_readvariableop_resourceodelstm_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*D
_output_shapes2
0: : : : :	�@:	�@: : : : : : : : : *)
_read_only_resource_inputs
		
*
bodyR
while_body_61383*
condR
while_cond_61382*C
output_shapes2
0: : : : :	�@:	�@: : : : : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�   @   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:d�@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*#
_output_shapes
:�d@2
transpose_1�
IdentityIdentitystrided_slice_3:output:0^odelstm/MatMul/ReadVariableOp ^odelstm/MatMul_1/ReadVariableOp ^odelstm/MatMul_2/ReadVariableOp ^odelstm/MatMul_3/ReadVariableOp ^odelstm/MatMul_4/ReadVariableOp ^odelstm/MatMul_5/ReadVariableOp ^odelstm/MatMul_6/ReadVariableOp ^odelstm/MatMul_7/ReadVariableOp ^odelstm/MatMul_8/ReadVariableOp ^odelstm/MatMul_9/ReadVariableOp^odelstm/ReadVariableOp^odelstm/ReadVariableOp_1^odelstm/ReadVariableOp_2^odelstm/ReadVariableOp_3^odelstm/add_1/ReadVariableOp^odelstm/add_11/ReadVariableOp^odelstm/add_14/ReadVariableOp^odelstm/add_5/ReadVariableOp^odelstm/add_8/ReadVariableOp^while*
T0*
_output_shapes
:	�@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:�d:::::::2>
odelstm/MatMul/ReadVariableOpodelstm/MatMul/ReadVariableOp2B
odelstm/MatMul_1/ReadVariableOpodelstm/MatMul_1/ReadVariableOp2B
odelstm/MatMul_2/ReadVariableOpodelstm/MatMul_2/ReadVariableOp2B
odelstm/MatMul_3/ReadVariableOpodelstm/MatMul_3/ReadVariableOp2B
odelstm/MatMul_4/ReadVariableOpodelstm/MatMul_4/ReadVariableOp2B
odelstm/MatMul_5/ReadVariableOpodelstm/MatMul_5/ReadVariableOp2B
odelstm/MatMul_6/ReadVariableOpodelstm/MatMul_6/ReadVariableOp2B
odelstm/MatMul_7/ReadVariableOpodelstm/MatMul_7/ReadVariableOp2B
odelstm/MatMul_8/ReadVariableOpodelstm/MatMul_8/ReadVariableOp2B
odelstm/MatMul_9/ReadVariableOpodelstm/MatMul_9/ReadVariableOp20
odelstm/ReadVariableOpodelstm/ReadVariableOp24
odelstm/ReadVariableOp_1odelstm/ReadVariableOp_124
odelstm/ReadVariableOp_2odelstm/ReadVariableOp_224
odelstm/ReadVariableOp_3odelstm/ReadVariableOp_32<
odelstm/add_1/ReadVariableOpodelstm/add_1/ReadVariableOp2>
odelstm/add_11/ReadVariableOpodelstm/add_11/ReadVariableOp2>
odelstm/add_14/ReadVariableOpodelstm/add_14/ReadVariableOp2<
odelstm/add_5/ReadVariableOpodelstm/add_5/ReadVariableOp2<
odelstm/add_8/ReadVariableOpodelstm/add_8/ReadVariableOp2
whilewhile:K G
#
_output_shapes
:�d
 
_user_specified_nameinputs
ʾ
�
while_body_58944
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
.while_odelstm_matmul_readvariableop_resource_04
0while_odelstm_matmul_1_readvariableop_resource_01
-while_odelstm_add_1_readvariableop_resource_04
0while_odelstm_matmul_2_readvariableop_resource_04
0while_odelstm_matmul_3_readvariableop_resource_01
-while_odelstm_add_5_readvariableop_resource_0+
'while_odelstm_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
,while_odelstm_matmul_readvariableop_resource2
.while_odelstm_matmul_1_readvariableop_resource/
+while_odelstm_add_1_readvariableop_resource2
.while_odelstm_matmul_2_readvariableop_resource2
.while_odelstm_matmul_3_readvariableop_resource/
+while_odelstm_add_5_readvariableop_resource)
%while_odelstm_readvariableop_resource��#while/odelstm/MatMul/ReadVariableOp�%while/odelstm/MatMul_1/ReadVariableOp�%while/odelstm/MatMul_2/ReadVariableOp�%while/odelstm/MatMul_3/ReadVariableOp�%while/odelstm/MatMul_4/ReadVariableOp�%while/odelstm/MatMul_5/ReadVariableOp�%while/odelstm/MatMul_6/ReadVariableOp�%while/odelstm/MatMul_7/ReadVariableOp�%while/odelstm/MatMul_8/ReadVariableOp�%while/odelstm/MatMul_9/ReadVariableOp�while/odelstm/ReadVariableOp�while/odelstm/ReadVariableOp_1�while/odelstm/ReadVariableOp_2�while/odelstm/ReadVariableOp_3�"while/odelstm/add_1/ReadVariableOp�#while/odelstm/add_11/ReadVariableOp�#while/odelstm/add_14/ReadVariableOp�"while/odelstm/add_5/ReadVariableOp�"while/odelstm/add_8/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	�*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
#while/odelstm/MatMul/ReadVariableOpReadVariableOp.while_odelstm_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02%
#while/odelstm/MatMul/ReadVariableOp�
while/odelstm/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/odelstm/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
while/odelstm/MatMul�
%while/odelstm/MatMul_1/ReadVariableOpReadVariableOp0while_odelstm_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02'
%while/odelstm/MatMul_1/ReadVariableOp�
while/odelstm/MatMul_1MatMulwhile_placeholder_3-while/odelstm/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
while/odelstm/MatMul_1�
while/odelstm/addAddV2while/odelstm/MatMul:product:0 while/odelstm/MatMul_1:product:0*
T0* 
_output_shapes
:
��2
while/odelstm/add�
"while/odelstm/add_1/ReadVariableOpReadVariableOp-while_odelstm_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype02$
"while/odelstm/add_1/ReadVariableOp�
while/odelstm/add_1AddV2while/odelstm/add:z:0*while/odelstm/add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
while/odelstm/add_1l
while/odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/odelstm/Const�
while/odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/odelstm/split/split_dim�
while/odelstm/splitSplit&while/odelstm/split/split_dim:output:0while/odelstm/add_1:z:0*
T0*@
_output_shapes.
,:	�@:	�@:	�@:	�@*
	num_split2
while/odelstm/splitx
while/odelstm/TanhTanhwhile/odelstm/split:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh�
while/odelstm/SigmoidSigmoidwhile/odelstm/split:output:1*
T0*
_output_shapes
:	�@2
while/odelstm/Sigmoids
while/odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
while/odelstm/add_2/y�
while/odelstm/add_2AddV2while/odelstm/split:output:2while/odelstm/add_2/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_2�
while/odelstm/Sigmoid_1Sigmoidwhile/odelstm/add_2:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Sigmoid_1�
while/odelstm/Sigmoid_2Sigmoidwhile/odelstm/split:output:3*
T0*
_output_shapes
:	�@2
while/odelstm/Sigmoid_2�
while/odelstm/mulMulwhile_placeholder_2while/odelstm/Sigmoid_1:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul�
while/odelstm/mul_1Mulwhile/odelstm/Tanh:y:0while/odelstm/Sigmoid:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_1�
while/odelstm/add_3AddV2while/odelstm/mul:z:0while/odelstm/mul_1:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_3w
while/odelstm/Tanh_1Tanhwhile/odelstm/add_3:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_1�
while/odelstm/mul_2Mulwhile/odelstm/Tanh_1:y:0while/odelstm/Sigmoid_2:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_2�
%while/odelstm/MatMul_2/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_2/ReadVariableOp�
while/odelstm/MatMul_2MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_2�
%while/odelstm/MatMul_3/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_3/ReadVariableOp�
while/odelstm/MatMul_3MatMulwhile_placeholder_3-while/odelstm/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_3�
while/odelstm/add_4AddV2 while/odelstm/MatMul_2:product:0 while/odelstm/MatMul_3:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_4�
"while/odelstm/add_5/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"while/odelstm/add_5/ReadVariableOp�
while/odelstm/add_5AddV2while/odelstm/add_4:z:0*while/odelstm/add_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_5w
while/odelstm/Tanh_2Tanhwhile/odelstm/add_5:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_2�
while/odelstm/ReadVariableOpReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02
while/odelstm/ReadVariableOp�
while/odelstm/mul_3Mul$while/odelstm/ReadVariableOp:value:0while/odelstm/Tanh_2:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_3s
while/odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_4/y�
while/odelstm/mul_4Mulwhile_placeholder_3while/odelstm/mul_4/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_4�
while/odelstm/subSubwhile/odelstm/mul_3:z:0while/odelstm/mul_4:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/subs
while/odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_5/x�
while/odelstm/mul_5Mulwhile/odelstm/mul_5/x:output:0while/odelstm/sub:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_5�
while/odelstm/add_6AddV2while_placeholder_3while/odelstm/mul_5:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_6�
%while/odelstm/MatMul_4/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_4/ReadVariableOp�
while/odelstm/MatMul_4MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_4�
%while/odelstm/MatMul_5/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_5/ReadVariableOp�
while/odelstm/MatMul_5MatMulwhile/odelstm/add_6:z:0-while/odelstm/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_5�
while/odelstm/add_7AddV2 while/odelstm/MatMul_4:product:0 while/odelstm/MatMul_5:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_7�
"while/odelstm/add_8/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"while/odelstm/add_8/ReadVariableOp�
while/odelstm/add_8AddV2while/odelstm/add_7:z:0*while/odelstm/add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_8w
while/odelstm/Tanh_3Tanhwhile/odelstm/add_8:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_3�
while/odelstm/ReadVariableOp_1ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_1�
while/odelstm/mul_6Mul&while/odelstm/ReadVariableOp_1:value:0while/odelstm/Tanh_3:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_6s
while/odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_7/y�
while/odelstm/mul_7Mulwhile/odelstm/add_6:z:0while/odelstm/mul_7/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_7�
while/odelstm/sub_1Subwhile/odelstm/mul_6:z:0while/odelstm/mul_7:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/sub_1s
while/odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_8/x�
while/odelstm/mul_8Mulwhile/odelstm/mul_8/x:output:0while/odelstm/sub_1:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_8�
while/odelstm/add_9AddV2while/odelstm/add_6:z:0while/odelstm/mul_8:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_9�
%while/odelstm/MatMul_6/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_6/ReadVariableOp�
while/odelstm/MatMul_6MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_6�
%while/odelstm/MatMul_7/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_7/ReadVariableOp�
while/odelstm/MatMul_7MatMulwhile/odelstm/add_9:z:0-while/odelstm/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_7�
while/odelstm/add_10AddV2 while/odelstm/MatMul_6:product:0 while/odelstm/MatMul_7:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_10�
#while/odelstm/add_11/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02%
#while/odelstm/add_11/ReadVariableOp�
while/odelstm/add_11AddV2while/odelstm/add_10:z:0+while/odelstm/add_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_11x
while/odelstm/Tanh_4Tanhwhile/odelstm/add_11:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_4�
while/odelstm/ReadVariableOp_2ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_2�
while/odelstm/mul_9Mul&while/odelstm/ReadVariableOp_2:value:0while/odelstm/Tanh_4:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_9u
while/odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_10/y�
while/odelstm/mul_10Mulwhile/odelstm/add_9:z:0while/odelstm/mul_10/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_10�
while/odelstm/sub_2Subwhile/odelstm/mul_9:z:0while/odelstm/mul_10:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/sub_2u
while/odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_11/x�
while/odelstm/mul_11Mulwhile/odelstm/mul_11/x:output:0while/odelstm/sub_2:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_11�
while/odelstm/add_12AddV2while/odelstm/add_9:z:0while/odelstm/mul_11:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_12�
%while/odelstm/MatMul_8/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_8/ReadVariableOp�
while/odelstm/MatMul_8MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_8�
%while/odelstm/MatMul_9/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_9/ReadVariableOp�
while/odelstm/MatMul_9MatMulwhile/odelstm/add_12:z:0-while/odelstm/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_9�
while/odelstm/add_13AddV2 while/odelstm/MatMul_8:product:0 while/odelstm/MatMul_9:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_13�
#while/odelstm/add_14/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02%
#while/odelstm/add_14/ReadVariableOp�
while/odelstm/add_14AddV2while/odelstm/add_13:z:0+while/odelstm/add_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_14x
while/odelstm/Tanh_5Tanhwhile/odelstm/add_14:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_5�
while/odelstm/ReadVariableOp_3ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_3�
while/odelstm/mul_12Mul&while/odelstm/ReadVariableOp_3:value:0while/odelstm/Tanh_5:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_12u
while/odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_13/y�
while/odelstm/mul_13Mulwhile/odelstm/add_12:z:0while/odelstm/mul_13/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_13�
while/odelstm/sub_3Subwhile/odelstm/mul_12:z:0while/odelstm/mul_13:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/sub_3u
while/odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_14/x�
while/odelstm/mul_14Mulwhile/odelstm/mul_14/x:output:0while/odelstm/sub_3:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_14�
while/odelstm/add_15AddV2while/odelstm/add_12:z:0while/odelstm/mul_14:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_15�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/odelstm/add_15:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/odelstm/add_3:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
:	�@2
while/Identity_4�
while/Identity_5Identitywhile/odelstm/add_15:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
:	�@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"\
+while_odelstm_add_1_readvariableop_resource-while_odelstm_add_1_readvariableop_resource_0"\
+while_odelstm_add_5_readvariableop_resource-while_odelstm_add_5_readvariableop_resource_0"b
.while_odelstm_matmul_1_readvariableop_resource0while_odelstm_matmul_1_readvariableop_resource_0"b
.while_odelstm_matmul_2_readvariableop_resource0while_odelstm_matmul_2_readvariableop_resource_0"b
.while_odelstm_matmul_3_readvariableop_resource0while_odelstm_matmul_3_readvariableop_resource_0"^
,while_odelstm_matmul_readvariableop_resource.while_odelstm_matmul_readvariableop_resource_0"P
%while_odelstm_readvariableop_resource'while_odelstm_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :	�@:	�@: : :::::::2J
#while/odelstm/MatMul/ReadVariableOp#while/odelstm/MatMul/ReadVariableOp2N
%while/odelstm/MatMul_1/ReadVariableOp%while/odelstm/MatMul_1/ReadVariableOp2N
%while/odelstm/MatMul_2/ReadVariableOp%while/odelstm/MatMul_2/ReadVariableOp2N
%while/odelstm/MatMul_3/ReadVariableOp%while/odelstm/MatMul_3/ReadVariableOp2N
%while/odelstm/MatMul_4/ReadVariableOp%while/odelstm/MatMul_4/ReadVariableOp2N
%while/odelstm/MatMul_5/ReadVariableOp%while/odelstm/MatMul_5/ReadVariableOp2N
%while/odelstm/MatMul_6/ReadVariableOp%while/odelstm/MatMul_6/ReadVariableOp2N
%while/odelstm/MatMul_7/ReadVariableOp%while/odelstm/MatMul_7/ReadVariableOp2N
%while/odelstm/MatMul_8/ReadVariableOp%while/odelstm/MatMul_8/ReadVariableOp2N
%while/odelstm/MatMul_9/ReadVariableOp%while/odelstm/MatMul_9/ReadVariableOp2<
while/odelstm/ReadVariableOpwhile/odelstm/ReadVariableOp2@
while/odelstm/ReadVariableOp_1while/odelstm/ReadVariableOp_12@
while/odelstm/ReadVariableOp_2while/odelstm/ReadVariableOp_22@
while/odelstm/ReadVariableOp_3while/odelstm/ReadVariableOp_32H
"while/odelstm/add_1/ReadVariableOp"while/odelstm/add_1/ReadVariableOp2J
#while/odelstm/add_11/ReadVariableOp#while/odelstm/add_11/ReadVariableOp2J
#while/odelstm/add_14/ReadVariableOp#while/odelstm/add_14/ReadVariableOp2H
"while/odelstm/add_5/ReadVariableOp"while/odelstm/add_5/ReadVariableOp2H
"while/odelstm/add_8/ReadVariableOp"while/odelstm/add_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :

_output_shapes
: 
��
�
>__inference_rnn_layer_call_and_return_conditional_losses_59406

inputs*
&odelstm_matmul_readvariableop_resource,
(odelstm_matmul_1_readvariableop_resource)
%odelstm_add_1_readvariableop_resource,
(odelstm_matmul_2_readvariableop_resource,
(odelstm_matmul_3_readvariableop_resource)
%odelstm_add_5_readvariableop_resource#
odelstm_readvariableop_resource
identity��odelstm/MatMul/ReadVariableOp�odelstm/MatMul_1/ReadVariableOp�odelstm/MatMul_2/ReadVariableOp�odelstm/MatMul_3/ReadVariableOp�odelstm/MatMul_4/ReadVariableOp�odelstm/MatMul_5/ReadVariableOp�odelstm/MatMul_6/ReadVariableOp�odelstm/MatMul_7/ReadVariableOp�odelstm/MatMul_8/ReadVariableOp�odelstm/MatMul_9/ReadVariableOp�odelstm/ReadVariableOp�odelstm/ReadVariableOp_1�odelstm/ReadVariableOp_2�odelstm/ReadVariableOp_3�odelstm/add_1/ReadVariableOp�odelstm/add_11/ReadVariableOp�odelstm/add_14/ReadVariableOp�odelstm/add_5/ReadVariableOp�odelstm/add_8/ReadVariableOp�whilec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"�   d      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constm
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*
_output_shapes
:	�@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constu
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*
_output_shapes
:	�@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permr
	transpose	Transposeinputstranspose/perm:output:0*
T0*#
_output_shapes
:d�2
	transposeg
Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   �      2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�      27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*
shrink_axis_mask2
strided_slice_2�
odelstm/MatMul/ReadVariableOpReadVariableOp&odelstm_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
odelstm/MatMul/ReadVariableOp�
odelstm/MatMulMatMulstrided_slice_2:output:0%odelstm/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
odelstm/MatMul�
odelstm/MatMul_1/ReadVariableOpReadVariableOp(odelstm_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02!
odelstm/MatMul_1/ReadVariableOp�
odelstm/MatMul_1MatMulzeros_1:output:0'odelstm/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
odelstm/MatMul_1�
odelstm/addAddV2odelstm/MatMul:product:0odelstm/MatMul_1:product:0*
T0* 
_output_shapes
:
��2
odelstm/add�
odelstm/add_1/ReadVariableOpReadVariableOp%odelstm_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
odelstm/add_1/ReadVariableOp�
odelstm/add_1AddV2odelstm/add:z:0$odelstm/add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
odelstm/add_1`
odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
odelstm/Const}
odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
odelstm/split/split_dim�
odelstm/splitSplit odelstm/split/split_dim:output:0odelstm/add_1:z:0*
T0*@
_output_shapes.
,:	�@:	�@:	�@:	�@*
	num_split2
odelstm/splitf
odelstm/TanhTanhodelstm/split:output:0*
T0*
_output_shapes
:	�@2
odelstm/Tanho
odelstm/SigmoidSigmoidodelstm/split:output:1*
T0*
_output_shapes
:	�@2
odelstm/Sigmoidg
odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
odelstm/add_2/y�
odelstm/add_2AddV2odelstm/split:output:2odelstm/add_2/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/add_2n
odelstm/Sigmoid_1Sigmoidodelstm/add_2:z:0*
T0*
_output_shapes
:	�@2
odelstm/Sigmoid_1s
odelstm/Sigmoid_2Sigmoidodelstm/split:output:3*
T0*
_output_shapes
:	�@2
odelstm/Sigmoid_2r
odelstm/mulMulzeros:output:0odelstm/Sigmoid_1:y:0*
T0*
_output_shapes
:	�@2
odelstm/mulv
odelstm/mul_1Mulodelstm/Tanh:y:0odelstm/Sigmoid:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_1u
odelstm/add_3AddV2odelstm/mul:z:0odelstm/mul_1:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_3e
odelstm/Tanh_1Tanhodelstm/add_3:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_1z
odelstm/mul_2Mulodelstm/Tanh_1:y:0odelstm/Sigmoid_2:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_2�
odelstm/MatMul_2/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_2/ReadVariableOp�
odelstm/MatMul_2MatMulodelstm/mul_2:z:0'odelstm/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_2�
odelstm/MatMul_3/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_3/ReadVariableOp�
odelstm/MatMul_3MatMulzeros_1:output:0'odelstm/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_3�
odelstm/add_4AddV2odelstm/MatMul_2:product:0odelstm/MatMul_3:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_4�
odelstm/add_5/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_5/ReadVariableOp�
odelstm/add_5AddV2odelstm/add_4:z:0$odelstm/add_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_5e
odelstm/Tanh_2Tanhodelstm/add_5:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_2�
odelstm/ReadVariableOpReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp�
odelstm/mul_3Mulodelstm/ReadVariableOp:value:0odelstm/Tanh_2:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_3g
odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_4/y{
odelstm/mul_4Mulzeros_1:output:0odelstm/mul_4/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_4q
odelstm/subSubodelstm/mul_3:z:0odelstm/mul_4:z:0*
T0*
_output_shapes
:	�@2
odelstm/subg
odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_5/xz
odelstm/mul_5Mulodelstm/mul_5/x:output:0odelstm/sub:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_5v
odelstm/add_6AddV2zeros_1:output:0odelstm/mul_5:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_6�
odelstm/MatMul_4/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_4/ReadVariableOp�
odelstm/MatMul_4MatMulodelstm/mul_2:z:0'odelstm/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_4�
odelstm/MatMul_5/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_5/ReadVariableOp�
odelstm/MatMul_5MatMulodelstm/add_6:z:0'odelstm/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_5�
odelstm/add_7AddV2odelstm/MatMul_4:product:0odelstm/MatMul_5:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_7�
odelstm/add_8/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_8/ReadVariableOp�
odelstm/add_8AddV2odelstm/add_7:z:0$odelstm/add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_8e
odelstm/Tanh_3Tanhodelstm/add_8:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_3�
odelstm/ReadVariableOp_1ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_1�
odelstm/mul_6Mul odelstm/ReadVariableOp_1:value:0odelstm/Tanh_3:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_6g
odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_7/y|
odelstm/mul_7Mulodelstm/add_6:z:0odelstm/mul_7/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_7u
odelstm/sub_1Subodelstm/mul_6:z:0odelstm/mul_7:z:0*
T0*
_output_shapes
:	�@2
odelstm/sub_1g
odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_8/x|
odelstm/mul_8Mulodelstm/mul_8/x:output:0odelstm/sub_1:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_8w
odelstm/add_9AddV2odelstm/add_6:z:0odelstm/mul_8:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_9�
odelstm/MatMul_6/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_6/ReadVariableOp�
odelstm/MatMul_6MatMulodelstm/mul_2:z:0'odelstm/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_6�
odelstm/MatMul_7/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_7/ReadVariableOp�
odelstm/MatMul_7MatMulodelstm/add_9:z:0'odelstm/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_7�
odelstm/add_10AddV2odelstm/MatMul_6:product:0odelstm/MatMul_7:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_10�
odelstm/add_11/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_11/ReadVariableOp�
odelstm/add_11AddV2odelstm/add_10:z:0%odelstm/add_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_11f
odelstm/Tanh_4Tanhodelstm/add_11:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_4�
odelstm/ReadVariableOp_2ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_2�
odelstm/mul_9Mul odelstm/ReadVariableOp_2:value:0odelstm/Tanh_4:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_9i
odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_10/y
odelstm/mul_10Mulodelstm/add_9:z:0odelstm/mul_10/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_10v
odelstm/sub_2Subodelstm/mul_9:z:0odelstm/mul_10:z:0*
T0*
_output_shapes
:	�@2
odelstm/sub_2i
odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_11/x
odelstm/mul_11Mulodelstm/mul_11/x:output:0odelstm/sub_2:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_11z
odelstm/add_12AddV2odelstm/add_9:z:0odelstm/mul_11:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_12�
odelstm/MatMul_8/ReadVariableOpReadVariableOp(odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_8/ReadVariableOp�
odelstm/MatMul_8MatMulodelstm/mul_2:z:0'odelstm/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_8�
odelstm/MatMul_9/ReadVariableOpReadVariableOp(odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02!
odelstm/MatMul_9/ReadVariableOp�
odelstm/MatMul_9MatMulodelstm/add_12:z:0'odelstm/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/MatMul_9�
odelstm/add_13AddV2odelstm/MatMul_8:product:0odelstm/MatMul_9:product:0*
T0*
_output_shapes
:	�@2
odelstm/add_13�
odelstm/add_14/ReadVariableOpReadVariableOp%odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/add_14/ReadVariableOp�
odelstm/add_14AddV2odelstm/add_13:z:0%odelstm/add_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
odelstm/add_14f
odelstm/Tanh_5Tanhodelstm/add_14:z:0*
T0*
_output_shapes
:	�@2
odelstm/Tanh_5�
odelstm/ReadVariableOp_3ReadVariableOpodelstm_readvariableop_resource*
_output_shapes
:@*
dtype02
odelstm/ReadVariableOp_3�
odelstm/mul_12Mul odelstm/ReadVariableOp_3:value:0odelstm/Tanh_5:y:0*
T0*
_output_shapes
:	�@2
odelstm/mul_12i
odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
odelstm/mul_13/y�
odelstm/mul_13Mulodelstm/add_12:z:0odelstm/mul_13/y:output:0*
T0*
_output_shapes
:	�@2
odelstm/mul_13w
odelstm/sub_3Subodelstm/mul_12:z:0odelstm/mul_13:z:0*
T0*
_output_shapes
:	�@2
odelstm/sub_3i
odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
odelstm/mul_14/x
odelstm/mul_14Mulodelstm/mul_14/x:output:0odelstm/sub_3:z:0*
T0*
_output_shapes
:	�@2
odelstm/mul_14{
odelstm/add_15AddV2odelstm/add_12:z:0odelstm/mul_14:z:0*
T0*
_output_shapes
:	�@2
odelstm/add_15�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�   @   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0&odelstm_matmul_readvariableop_resource(odelstm_matmul_1_readvariableop_resource%odelstm_add_1_readvariableop_resource(odelstm_matmul_2_readvariableop_resource(odelstm_matmul_3_readvariableop_resource%odelstm_add_5_readvariableop_resourceodelstm_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*D
_output_shapes2
0: : : : :	�@:	�@: : : : : : : : : *)
_read_only_resource_inputs
		
*
bodyR
while_body_59244*
condR
while_cond_59243*C
output_shapes2
0: : : : :	�@:	�@: : : : : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�   @   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:d�@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*#
_output_shapes
:�d@2
transpose_1�
IdentityIdentitystrided_slice_3:output:0^odelstm/MatMul/ReadVariableOp ^odelstm/MatMul_1/ReadVariableOp ^odelstm/MatMul_2/ReadVariableOp ^odelstm/MatMul_3/ReadVariableOp ^odelstm/MatMul_4/ReadVariableOp ^odelstm/MatMul_5/ReadVariableOp ^odelstm/MatMul_6/ReadVariableOp ^odelstm/MatMul_7/ReadVariableOp ^odelstm/MatMul_8/ReadVariableOp ^odelstm/MatMul_9/ReadVariableOp^odelstm/ReadVariableOp^odelstm/ReadVariableOp_1^odelstm/ReadVariableOp_2^odelstm/ReadVariableOp_3^odelstm/add_1/ReadVariableOp^odelstm/add_11/ReadVariableOp^odelstm/add_14/ReadVariableOp^odelstm/add_5/ReadVariableOp^odelstm/add_8/ReadVariableOp^while*
T0*
_output_shapes
:	�@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:�d:::::::2>
odelstm/MatMul/ReadVariableOpodelstm/MatMul/ReadVariableOp2B
odelstm/MatMul_1/ReadVariableOpodelstm/MatMul_1/ReadVariableOp2B
odelstm/MatMul_2/ReadVariableOpodelstm/MatMul_2/ReadVariableOp2B
odelstm/MatMul_3/ReadVariableOpodelstm/MatMul_3/ReadVariableOp2B
odelstm/MatMul_4/ReadVariableOpodelstm/MatMul_4/ReadVariableOp2B
odelstm/MatMul_5/ReadVariableOpodelstm/MatMul_5/ReadVariableOp2B
odelstm/MatMul_6/ReadVariableOpodelstm/MatMul_6/ReadVariableOp2B
odelstm/MatMul_7/ReadVariableOpodelstm/MatMul_7/ReadVariableOp2B
odelstm/MatMul_8/ReadVariableOpodelstm/MatMul_8/ReadVariableOp2B
odelstm/MatMul_9/ReadVariableOpodelstm/MatMul_9/ReadVariableOp20
odelstm/ReadVariableOpodelstm/ReadVariableOp24
odelstm/ReadVariableOp_1odelstm/ReadVariableOp_124
odelstm/ReadVariableOp_2odelstm/ReadVariableOp_224
odelstm/ReadVariableOp_3odelstm/ReadVariableOp_32<
odelstm/add_1/ReadVariableOpodelstm/add_1/ReadVariableOp2>
odelstm/add_11/ReadVariableOpodelstm/add_11/ReadVariableOp2>
odelstm/add_14/ReadVariableOpodelstm/add_14/ReadVariableOp2<
odelstm/add_5/ReadVariableOpodelstm/add_5/ReadVariableOp2<
odelstm/add_8/ReadVariableOpodelstm/add_8/ReadVariableOp2
whilewhile:K G
#
_output_shapes
:�d
 
_user_specified_nameinputs
�

�
while_cond_60744
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_60744___redundant_placeholder03
/while_while_cond_60744___redundant_placeholder13
/while_while_cond_60744___redundant_placeholder23
/while_while_cond_60744___redundant_placeholder33
/while_while_cond_60744___redundant_placeholder43
/while_while_cond_60744___redundant_placeholder53
/while_while_cond_60744___redundant_placeholder63
/while_while_cond_60744___redundant_placeholder7
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*c
_input_shapesR
P: : : : :���������@:���������@: ::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
'__inference_odelstm_layer_call_fn_61730

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2
*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*)
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_odelstm_layer_call_and_return_conditional_losses_581992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*h
_input_shapesW
U:���������:���������@:���������@:::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
��
�
 __inference__wrapped_model_58056
input_1
input_27
3ode_lstm_rnn_odelstm_matmul_readvariableop_resource9
5ode_lstm_rnn_odelstm_matmul_1_readvariableop_resource6
2ode_lstm_rnn_odelstm_add_1_readvariableop_resource9
5ode_lstm_rnn_odelstm_matmul_2_readvariableop_resource9
5ode_lstm_rnn_odelstm_matmul_3_readvariableop_resource6
2ode_lstm_rnn_odelstm_add_5_readvariableop_resource0
,ode_lstm_rnn_odelstm_readvariableop_resource1
-ode_lstm_dense_matmul_readvariableop_resource2
.ode_lstm_dense_biasadd_readvariableop_resource
identity��%ode_lstm/dense/BiasAdd/ReadVariableOp�$ode_lstm/dense/MatMul/ReadVariableOp�*ode_lstm/rnn/odelstm/MatMul/ReadVariableOp�,ode_lstm/rnn/odelstm/MatMul_1/ReadVariableOp�,ode_lstm/rnn/odelstm/MatMul_2/ReadVariableOp�,ode_lstm/rnn/odelstm/MatMul_3/ReadVariableOp�,ode_lstm/rnn/odelstm/MatMul_4/ReadVariableOp�,ode_lstm/rnn/odelstm/MatMul_5/ReadVariableOp�,ode_lstm/rnn/odelstm/MatMul_6/ReadVariableOp�,ode_lstm/rnn/odelstm/MatMul_7/ReadVariableOp�,ode_lstm/rnn/odelstm/MatMul_8/ReadVariableOp�,ode_lstm/rnn/odelstm/MatMul_9/ReadVariableOp�#ode_lstm/rnn/odelstm/ReadVariableOp�%ode_lstm/rnn/odelstm/ReadVariableOp_1�%ode_lstm/rnn/odelstm/ReadVariableOp_2�%ode_lstm/rnn/odelstm/ReadVariableOp_3�)ode_lstm/rnn/odelstm/add_1/ReadVariableOp�*ode_lstm/rnn/odelstm/add_11/ReadVariableOp�*ode_lstm/rnn/odelstm/add_14/ReadVariableOp�)ode_lstm/rnn/odelstm/add_5/ReadVariableOp�)ode_lstm/rnn/odelstm/add_8/ReadVariableOp�ode_lstm/rnn/while}
ode_lstm/rnn/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"�   d      2
ode_lstm/rnn/Shape�
 ode_lstm/rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 ode_lstm/rnn/strided_slice/stack�
"ode_lstm/rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"ode_lstm/rnn/strided_slice/stack_1�
"ode_lstm/rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ode_lstm/rnn/strided_slice/stack_2�
ode_lstm/rnn/strided_sliceStridedSliceode_lstm/rnn/Shape:output:0)ode_lstm/rnn/strided_slice/stack:output:0+ode_lstm/rnn/strided_slice/stack_1:output:0+ode_lstm/rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ode_lstm/rnn/strided_slicev
ode_lstm/rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
ode_lstm/rnn/zeros/mul/y�
ode_lstm/rnn/zeros/mulMul#ode_lstm/rnn/strided_slice:output:0!ode_lstm/rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
ode_lstm/rnn/zeros/muly
ode_lstm/rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
ode_lstm/rnn/zeros/Less/y�
ode_lstm/rnn/zeros/LessLessode_lstm/rnn/zeros/mul:z:0"ode_lstm/rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
ode_lstm/rnn/zeros/Less|
ode_lstm/rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
ode_lstm/rnn/zeros/packed/1�
ode_lstm/rnn/zeros/packedPack#ode_lstm/rnn/strided_slice:output:0$ode_lstm/rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
ode_lstm/rnn/zeros/packedy
ode_lstm/rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ode_lstm/rnn/zeros/Const�
ode_lstm/rnn/zerosFill"ode_lstm/rnn/zeros/packed:output:0!ode_lstm/rnn/zeros/Const:output:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/zerosz
ode_lstm/rnn/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
ode_lstm/rnn/zeros_1/mul/y�
ode_lstm/rnn/zeros_1/mulMul#ode_lstm/rnn/strided_slice:output:0#ode_lstm/rnn/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
ode_lstm/rnn/zeros_1/mul}
ode_lstm/rnn/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
ode_lstm/rnn/zeros_1/Less/y�
ode_lstm/rnn/zeros_1/LessLessode_lstm/rnn/zeros_1/mul:z:0$ode_lstm/rnn/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
ode_lstm/rnn/zeros_1/Less�
ode_lstm/rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
ode_lstm/rnn/zeros_1/packed/1�
ode_lstm/rnn/zeros_1/packedPack#ode_lstm/rnn/strided_slice:output:0&ode_lstm/rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
ode_lstm/rnn/zeros_1/packed}
ode_lstm/rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ode_lstm/rnn/zeros_1/Const�
ode_lstm/rnn/zeros_1Fill$ode_lstm/rnn/zeros_1/packed:output:0#ode_lstm/rnn/zeros_1/Const:output:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/zeros_1�
ode_lstm/rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
ode_lstm/rnn/transpose/perm�
ode_lstm/rnn/transpose	Transposeinput_1$ode_lstm/rnn/transpose/perm:output:0*
T0*#
_output_shapes
:d�2
ode_lstm/rnn/transpose�
ode_lstm/rnn/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"d   �      2
ode_lstm/rnn/Shape_1�
"ode_lstm/rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"ode_lstm/rnn/strided_slice_1/stack�
$ode_lstm/rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$ode_lstm/rnn/strided_slice_1/stack_1�
$ode_lstm/rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$ode_lstm/rnn/strided_slice_1/stack_2�
ode_lstm/rnn/strided_slice_1StridedSliceode_lstm/rnn/Shape_1:output:0+ode_lstm/rnn/strided_slice_1/stack:output:0-ode_lstm/rnn/strided_slice_1/stack_1:output:0-ode_lstm/rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ode_lstm/rnn/strided_slice_1�
(ode_lstm/rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(ode_lstm/rnn/TensorArrayV2/element_shape�
ode_lstm/rnn/TensorArrayV2TensorListReserve1ode_lstm/rnn/TensorArrayV2/element_shape:output:0%ode_lstm/rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
ode_lstm/rnn/TensorArrayV2�
Bode_lstm/rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�      2D
Bode_lstm/rnn/TensorArrayUnstack/TensorListFromTensor/element_shape�
4ode_lstm/rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorode_lstm/rnn/transpose:y:0Kode_lstm/rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4ode_lstm/rnn/TensorArrayUnstack/TensorListFromTensor�
"ode_lstm/rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"ode_lstm/rnn/strided_slice_2/stack�
$ode_lstm/rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$ode_lstm/rnn/strided_slice_2/stack_1�
$ode_lstm/rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$ode_lstm/rnn/strided_slice_2/stack_2�
ode_lstm/rnn/strided_slice_2StridedSliceode_lstm/rnn/transpose:y:0+ode_lstm/rnn/strided_slice_2/stack:output:0-ode_lstm/rnn/strided_slice_2/stack_1:output:0-ode_lstm/rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*
shrink_axis_mask2
ode_lstm/rnn/strided_slice_2�
*ode_lstm/rnn/odelstm/MatMul/ReadVariableOpReadVariableOp3ode_lstm_rnn_odelstm_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*ode_lstm/rnn/odelstm/MatMul/ReadVariableOp�
ode_lstm/rnn/odelstm/MatMulMatMul%ode_lstm/rnn/strided_slice_2:output:02ode_lstm/rnn/odelstm/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
ode_lstm/rnn/odelstm/MatMul�
,ode_lstm/rnn/odelstm/MatMul_1/ReadVariableOpReadVariableOp5ode_lstm_rnn_odelstm_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02.
,ode_lstm/rnn/odelstm/MatMul_1/ReadVariableOp�
ode_lstm/rnn/odelstm/MatMul_1MatMulode_lstm/rnn/zeros_1:output:04ode_lstm/rnn/odelstm/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
ode_lstm/rnn/odelstm/MatMul_1�
ode_lstm/rnn/odelstm/addAddV2%ode_lstm/rnn/odelstm/MatMul:product:0'ode_lstm/rnn/odelstm/MatMul_1:product:0*
T0* 
_output_shapes
:
��2
ode_lstm/rnn/odelstm/add�
)ode_lstm/rnn/odelstm/add_1/ReadVariableOpReadVariableOp2ode_lstm_rnn_odelstm_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)ode_lstm/rnn/odelstm/add_1/ReadVariableOp�
ode_lstm/rnn/odelstm/add_1AddV2ode_lstm/rnn/odelstm/add:z:01ode_lstm/rnn/odelstm/add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
ode_lstm/rnn/odelstm/add_1z
ode_lstm/rnn/odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ode_lstm/rnn/odelstm/Const�
$ode_lstm/rnn/odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2&
$ode_lstm/rnn/odelstm/split/split_dim�
ode_lstm/rnn/odelstm/splitSplit-ode_lstm/rnn/odelstm/split/split_dim:output:0ode_lstm/rnn/odelstm/add_1:z:0*
T0*@
_output_shapes.
,:	�@:	�@:	�@:	�@*
	num_split2
ode_lstm/rnn/odelstm/split�
ode_lstm/rnn/odelstm/TanhTanh#ode_lstm/rnn/odelstm/split:output:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/Tanh�
ode_lstm/rnn/odelstm/SigmoidSigmoid#ode_lstm/rnn/odelstm/split:output:1*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/Sigmoid�
ode_lstm/rnn/odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
ode_lstm/rnn/odelstm/add_2/y�
ode_lstm/rnn/odelstm/add_2AddV2#ode_lstm/rnn/odelstm/split:output:2%ode_lstm/rnn/odelstm/add_2/y:output:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/add_2�
ode_lstm/rnn/odelstm/Sigmoid_1Sigmoidode_lstm/rnn/odelstm/add_2:z:0*
T0*
_output_shapes
:	�@2 
ode_lstm/rnn/odelstm/Sigmoid_1�
ode_lstm/rnn/odelstm/Sigmoid_2Sigmoid#ode_lstm/rnn/odelstm/split:output:3*
T0*
_output_shapes
:	�@2 
ode_lstm/rnn/odelstm/Sigmoid_2�
ode_lstm/rnn/odelstm/mulMulode_lstm/rnn/zeros:output:0"ode_lstm/rnn/odelstm/Sigmoid_1:y:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul�
ode_lstm/rnn/odelstm/mul_1Mulode_lstm/rnn/odelstm/Tanh:y:0 ode_lstm/rnn/odelstm/Sigmoid:y:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul_1�
ode_lstm/rnn/odelstm/add_3AddV2ode_lstm/rnn/odelstm/mul:z:0ode_lstm/rnn/odelstm/mul_1:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/add_3�
ode_lstm/rnn/odelstm/Tanh_1Tanhode_lstm/rnn/odelstm/add_3:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/Tanh_1�
ode_lstm/rnn/odelstm/mul_2Mulode_lstm/rnn/odelstm/Tanh_1:y:0"ode_lstm/rnn/odelstm/Sigmoid_2:y:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul_2�
,ode_lstm/rnn/odelstm/MatMul_2/ReadVariableOpReadVariableOp5ode_lstm_rnn_odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02.
,ode_lstm/rnn/odelstm/MatMul_2/ReadVariableOp�
ode_lstm/rnn/odelstm/MatMul_2MatMulode_lstm/rnn/odelstm/mul_2:z:04ode_lstm/rnn/odelstm/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/MatMul_2�
,ode_lstm/rnn/odelstm/MatMul_3/ReadVariableOpReadVariableOp5ode_lstm_rnn_odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02.
,ode_lstm/rnn/odelstm/MatMul_3/ReadVariableOp�
ode_lstm/rnn/odelstm/MatMul_3MatMulode_lstm/rnn/zeros_1:output:04ode_lstm/rnn/odelstm/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/MatMul_3�
ode_lstm/rnn/odelstm/add_4AddV2'ode_lstm/rnn/odelstm/MatMul_2:product:0'ode_lstm/rnn/odelstm/MatMul_3:product:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/add_4�
)ode_lstm/rnn/odelstm/add_5/ReadVariableOpReadVariableOp2ode_lstm_rnn_odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02+
)ode_lstm/rnn/odelstm/add_5/ReadVariableOp�
ode_lstm/rnn/odelstm/add_5AddV2ode_lstm/rnn/odelstm/add_4:z:01ode_lstm/rnn/odelstm/add_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/add_5�
ode_lstm/rnn/odelstm/Tanh_2Tanhode_lstm/rnn/odelstm/add_5:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/Tanh_2�
#ode_lstm/rnn/odelstm/ReadVariableOpReadVariableOp,ode_lstm_rnn_odelstm_readvariableop_resource*
_output_shapes
:@*
dtype02%
#ode_lstm/rnn/odelstm/ReadVariableOp�
ode_lstm/rnn/odelstm/mul_3Mul+ode_lstm/rnn/odelstm/ReadVariableOp:value:0ode_lstm/rnn/odelstm/Tanh_2:y:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul_3�
ode_lstm/rnn/odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
ode_lstm/rnn/odelstm/mul_4/y�
ode_lstm/rnn/odelstm/mul_4Mulode_lstm/rnn/zeros_1:output:0%ode_lstm/rnn/odelstm/mul_4/y:output:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul_4�
ode_lstm/rnn/odelstm/subSubode_lstm/rnn/odelstm/mul_3:z:0ode_lstm/rnn/odelstm/mul_4:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/sub�
ode_lstm/rnn/odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
ode_lstm/rnn/odelstm/mul_5/x�
ode_lstm/rnn/odelstm/mul_5Mul%ode_lstm/rnn/odelstm/mul_5/x:output:0ode_lstm/rnn/odelstm/sub:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul_5�
ode_lstm/rnn/odelstm/add_6AddV2ode_lstm/rnn/zeros_1:output:0ode_lstm/rnn/odelstm/mul_5:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/add_6�
,ode_lstm/rnn/odelstm/MatMul_4/ReadVariableOpReadVariableOp5ode_lstm_rnn_odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02.
,ode_lstm/rnn/odelstm/MatMul_4/ReadVariableOp�
ode_lstm/rnn/odelstm/MatMul_4MatMulode_lstm/rnn/odelstm/mul_2:z:04ode_lstm/rnn/odelstm/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/MatMul_4�
,ode_lstm/rnn/odelstm/MatMul_5/ReadVariableOpReadVariableOp5ode_lstm_rnn_odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02.
,ode_lstm/rnn/odelstm/MatMul_5/ReadVariableOp�
ode_lstm/rnn/odelstm/MatMul_5MatMulode_lstm/rnn/odelstm/add_6:z:04ode_lstm/rnn/odelstm/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/MatMul_5�
ode_lstm/rnn/odelstm/add_7AddV2'ode_lstm/rnn/odelstm/MatMul_4:product:0'ode_lstm/rnn/odelstm/MatMul_5:product:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/add_7�
)ode_lstm/rnn/odelstm/add_8/ReadVariableOpReadVariableOp2ode_lstm_rnn_odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02+
)ode_lstm/rnn/odelstm/add_8/ReadVariableOp�
ode_lstm/rnn/odelstm/add_8AddV2ode_lstm/rnn/odelstm/add_7:z:01ode_lstm/rnn/odelstm/add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/add_8�
ode_lstm/rnn/odelstm/Tanh_3Tanhode_lstm/rnn/odelstm/add_8:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/Tanh_3�
%ode_lstm/rnn/odelstm/ReadVariableOp_1ReadVariableOp,ode_lstm_rnn_odelstm_readvariableop_resource*
_output_shapes
:@*
dtype02'
%ode_lstm/rnn/odelstm/ReadVariableOp_1�
ode_lstm/rnn/odelstm/mul_6Mul-ode_lstm/rnn/odelstm/ReadVariableOp_1:value:0ode_lstm/rnn/odelstm/Tanh_3:y:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul_6�
ode_lstm/rnn/odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
ode_lstm/rnn/odelstm/mul_7/y�
ode_lstm/rnn/odelstm/mul_7Mulode_lstm/rnn/odelstm/add_6:z:0%ode_lstm/rnn/odelstm/mul_7/y:output:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul_7�
ode_lstm/rnn/odelstm/sub_1Subode_lstm/rnn/odelstm/mul_6:z:0ode_lstm/rnn/odelstm/mul_7:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/sub_1�
ode_lstm/rnn/odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
ode_lstm/rnn/odelstm/mul_8/x�
ode_lstm/rnn/odelstm/mul_8Mul%ode_lstm/rnn/odelstm/mul_8/x:output:0ode_lstm/rnn/odelstm/sub_1:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul_8�
ode_lstm/rnn/odelstm/add_9AddV2ode_lstm/rnn/odelstm/add_6:z:0ode_lstm/rnn/odelstm/mul_8:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/add_9�
,ode_lstm/rnn/odelstm/MatMul_6/ReadVariableOpReadVariableOp5ode_lstm_rnn_odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02.
,ode_lstm/rnn/odelstm/MatMul_6/ReadVariableOp�
ode_lstm/rnn/odelstm/MatMul_6MatMulode_lstm/rnn/odelstm/mul_2:z:04ode_lstm/rnn/odelstm/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/MatMul_6�
,ode_lstm/rnn/odelstm/MatMul_7/ReadVariableOpReadVariableOp5ode_lstm_rnn_odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02.
,ode_lstm/rnn/odelstm/MatMul_7/ReadVariableOp�
ode_lstm/rnn/odelstm/MatMul_7MatMulode_lstm/rnn/odelstm/add_9:z:04ode_lstm/rnn/odelstm/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/MatMul_7�
ode_lstm/rnn/odelstm/add_10AddV2'ode_lstm/rnn/odelstm/MatMul_6:product:0'ode_lstm/rnn/odelstm/MatMul_7:product:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/add_10�
*ode_lstm/rnn/odelstm/add_11/ReadVariableOpReadVariableOp2ode_lstm_rnn_odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02,
*ode_lstm/rnn/odelstm/add_11/ReadVariableOp�
ode_lstm/rnn/odelstm/add_11AddV2ode_lstm/rnn/odelstm/add_10:z:02ode_lstm/rnn/odelstm/add_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/add_11�
ode_lstm/rnn/odelstm/Tanh_4Tanhode_lstm/rnn/odelstm/add_11:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/Tanh_4�
%ode_lstm/rnn/odelstm/ReadVariableOp_2ReadVariableOp,ode_lstm_rnn_odelstm_readvariableop_resource*
_output_shapes
:@*
dtype02'
%ode_lstm/rnn/odelstm/ReadVariableOp_2�
ode_lstm/rnn/odelstm/mul_9Mul-ode_lstm/rnn/odelstm/ReadVariableOp_2:value:0ode_lstm/rnn/odelstm/Tanh_4:y:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul_9�
ode_lstm/rnn/odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
ode_lstm/rnn/odelstm/mul_10/y�
ode_lstm/rnn/odelstm/mul_10Mulode_lstm/rnn/odelstm/add_9:z:0&ode_lstm/rnn/odelstm/mul_10/y:output:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul_10�
ode_lstm/rnn/odelstm/sub_2Subode_lstm/rnn/odelstm/mul_9:z:0ode_lstm/rnn/odelstm/mul_10:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/sub_2�
ode_lstm/rnn/odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
ode_lstm/rnn/odelstm/mul_11/x�
ode_lstm/rnn/odelstm/mul_11Mul&ode_lstm/rnn/odelstm/mul_11/x:output:0ode_lstm/rnn/odelstm/sub_2:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul_11�
ode_lstm/rnn/odelstm/add_12AddV2ode_lstm/rnn/odelstm/add_9:z:0ode_lstm/rnn/odelstm/mul_11:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/add_12�
,ode_lstm/rnn/odelstm/MatMul_8/ReadVariableOpReadVariableOp5ode_lstm_rnn_odelstm_matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02.
,ode_lstm/rnn/odelstm/MatMul_8/ReadVariableOp�
ode_lstm/rnn/odelstm/MatMul_8MatMulode_lstm/rnn/odelstm/mul_2:z:04ode_lstm/rnn/odelstm/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/MatMul_8�
,ode_lstm/rnn/odelstm/MatMul_9/ReadVariableOpReadVariableOp5ode_lstm_rnn_odelstm_matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02.
,ode_lstm/rnn/odelstm/MatMul_9/ReadVariableOp�
ode_lstm/rnn/odelstm/MatMul_9MatMulode_lstm/rnn/odelstm/add_12:z:04ode_lstm/rnn/odelstm/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/MatMul_9�
ode_lstm/rnn/odelstm/add_13AddV2'ode_lstm/rnn/odelstm/MatMul_8:product:0'ode_lstm/rnn/odelstm/MatMul_9:product:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/add_13�
*ode_lstm/rnn/odelstm/add_14/ReadVariableOpReadVariableOp2ode_lstm_rnn_odelstm_add_5_readvariableop_resource*
_output_shapes
:@*
dtype02,
*ode_lstm/rnn/odelstm/add_14/ReadVariableOp�
ode_lstm/rnn/odelstm/add_14AddV2ode_lstm/rnn/odelstm/add_13:z:02ode_lstm/rnn/odelstm/add_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/add_14�
ode_lstm/rnn/odelstm/Tanh_5Tanhode_lstm/rnn/odelstm/add_14:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/Tanh_5�
%ode_lstm/rnn/odelstm/ReadVariableOp_3ReadVariableOp,ode_lstm_rnn_odelstm_readvariableop_resource*
_output_shapes
:@*
dtype02'
%ode_lstm/rnn/odelstm/ReadVariableOp_3�
ode_lstm/rnn/odelstm/mul_12Mul-ode_lstm/rnn/odelstm/ReadVariableOp_3:value:0ode_lstm/rnn/odelstm/Tanh_5:y:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul_12�
ode_lstm/rnn/odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
ode_lstm/rnn/odelstm/mul_13/y�
ode_lstm/rnn/odelstm/mul_13Mulode_lstm/rnn/odelstm/add_12:z:0&ode_lstm/rnn/odelstm/mul_13/y:output:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul_13�
ode_lstm/rnn/odelstm/sub_3Subode_lstm/rnn/odelstm/mul_12:z:0ode_lstm/rnn/odelstm/mul_13:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/sub_3�
ode_lstm/rnn/odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
ode_lstm/rnn/odelstm/mul_14/x�
ode_lstm/rnn/odelstm/mul_14Mul&ode_lstm/rnn/odelstm/mul_14/x:output:0ode_lstm/rnn/odelstm/sub_3:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/mul_14�
ode_lstm/rnn/odelstm/add_15AddV2ode_lstm/rnn/odelstm/add_12:z:0ode_lstm/rnn/odelstm/mul_14:z:0*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/odelstm/add_15�
*ode_lstm/rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�   @   2,
*ode_lstm/rnn/TensorArrayV2_1/element_shape�
ode_lstm/rnn/TensorArrayV2_1TensorListReserve3ode_lstm/rnn/TensorArrayV2_1/element_shape:output:0%ode_lstm/rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
ode_lstm/rnn/TensorArrayV2_1h
ode_lstm/rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
ode_lstm/rnn/time�
%ode_lstm/rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%ode_lstm/rnn/while/maximum_iterations�
ode_lstm/rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
ode_lstm/rnn/while/loop_counter�
ode_lstm/rnn/whileWhile(ode_lstm/rnn/while/loop_counter:output:0.ode_lstm/rnn/while/maximum_iterations:output:0ode_lstm/rnn/time:output:0%ode_lstm/rnn/TensorArrayV2_1:handle:0ode_lstm/rnn/zeros:output:0ode_lstm/rnn/zeros_1:output:0%ode_lstm/rnn/strided_slice_1:output:0Dode_lstm/rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:03ode_lstm_rnn_odelstm_matmul_readvariableop_resource5ode_lstm_rnn_odelstm_matmul_1_readvariableop_resource2ode_lstm_rnn_odelstm_add_1_readvariableop_resource5ode_lstm_rnn_odelstm_matmul_2_readvariableop_resource5ode_lstm_rnn_odelstm_matmul_3_readvariableop_resource2ode_lstm_rnn_odelstm_add_5_readvariableop_resource,ode_lstm_rnn_odelstm_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*D
_output_shapes2
0: : : : :	�@:	�@: : : : : : : : : *)
_read_only_resource_inputs
		
*)
body!R
ode_lstm_rnn_while_body_57888*)
cond!R
ode_lstm_rnn_while_cond_57887*C
output_shapes2
0: : : : :	�@:	�@: : : : : : : : : *
parallel_iterations 2
ode_lstm/rnn/while�
=ode_lstm/rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�   @   2?
=ode_lstm/rnn/TensorArrayV2Stack/TensorListStack/element_shape�
/ode_lstm/rnn/TensorArrayV2Stack/TensorListStackTensorListStackode_lstm/rnn/while:output:3Fode_lstm/rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:d�@*
element_dtype021
/ode_lstm/rnn/TensorArrayV2Stack/TensorListStack�
"ode_lstm/rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2$
"ode_lstm/rnn/strided_slice_3/stack�
$ode_lstm/rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$ode_lstm/rnn/strided_slice_3/stack_1�
$ode_lstm/rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$ode_lstm/rnn/strided_slice_3/stack_2�
ode_lstm/rnn/strided_slice_3StridedSlice8ode_lstm/rnn/TensorArrayV2Stack/TensorListStack:tensor:0+ode_lstm/rnn/strided_slice_3/stack:output:0-ode_lstm/rnn/strided_slice_3/stack_1:output:0-ode_lstm/rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�@*
shrink_axis_mask2
ode_lstm/rnn/strided_slice_3�
ode_lstm/rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
ode_lstm/rnn/transpose_1/perm�
ode_lstm/rnn/transpose_1	Transpose8ode_lstm/rnn/TensorArrayV2Stack/TensorListStack:tensor:0&ode_lstm/rnn/transpose_1/perm:output:0*
T0*#
_output_shapes
:�d@2
ode_lstm/rnn/transpose_1�
$ode_lstm/dense/MatMul/ReadVariableOpReadVariableOp-ode_lstm_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02&
$ode_lstm/dense/MatMul/ReadVariableOp�
ode_lstm/dense/MatMulMatMul%ode_lstm/rnn/strided_slice_3:output:0,ode_lstm/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
ode_lstm/dense/MatMul�
%ode_lstm/dense/BiasAdd/ReadVariableOpReadVariableOp.ode_lstm_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%ode_lstm/dense/BiasAdd/ReadVariableOp�
ode_lstm/dense/BiasAddBiasAddode_lstm/dense/MatMul:product:0-ode_lstm/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
ode_lstm/dense/BiasAdd�
IdentityIdentityode_lstm/dense/BiasAdd:output:0&^ode_lstm/dense/BiasAdd/ReadVariableOp%^ode_lstm/dense/MatMul/ReadVariableOp+^ode_lstm/rnn/odelstm/MatMul/ReadVariableOp-^ode_lstm/rnn/odelstm/MatMul_1/ReadVariableOp-^ode_lstm/rnn/odelstm/MatMul_2/ReadVariableOp-^ode_lstm/rnn/odelstm/MatMul_3/ReadVariableOp-^ode_lstm/rnn/odelstm/MatMul_4/ReadVariableOp-^ode_lstm/rnn/odelstm/MatMul_5/ReadVariableOp-^ode_lstm/rnn/odelstm/MatMul_6/ReadVariableOp-^ode_lstm/rnn/odelstm/MatMul_7/ReadVariableOp-^ode_lstm/rnn/odelstm/MatMul_8/ReadVariableOp-^ode_lstm/rnn/odelstm/MatMul_9/ReadVariableOp$^ode_lstm/rnn/odelstm/ReadVariableOp&^ode_lstm/rnn/odelstm/ReadVariableOp_1&^ode_lstm/rnn/odelstm/ReadVariableOp_2&^ode_lstm/rnn/odelstm/ReadVariableOp_3*^ode_lstm/rnn/odelstm/add_1/ReadVariableOp+^ode_lstm/rnn/odelstm/add_11/ReadVariableOp+^ode_lstm/rnn/odelstm/add_14/ReadVariableOp*^ode_lstm/rnn/odelstm/add_5/ReadVariableOp*^ode_lstm/rnn/odelstm/add_8/ReadVariableOp^ode_lstm/rnn/while*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:�d:�d:::::::::2N
%ode_lstm/dense/BiasAdd/ReadVariableOp%ode_lstm/dense/BiasAdd/ReadVariableOp2L
$ode_lstm/dense/MatMul/ReadVariableOp$ode_lstm/dense/MatMul/ReadVariableOp2X
*ode_lstm/rnn/odelstm/MatMul/ReadVariableOp*ode_lstm/rnn/odelstm/MatMul/ReadVariableOp2\
,ode_lstm/rnn/odelstm/MatMul_1/ReadVariableOp,ode_lstm/rnn/odelstm/MatMul_1/ReadVariableOp2\
,ode_lstm/rnn/odelstm/MatMul_2/ReadVariableOp,ode_lstm/rnn/odelstm/MatMul_2/ReadVariableOp2\
,ode_lstm/rnn/odelstm/MatMul_3/ReadVariableOp,ode_lstm/rnn/odelstm/MatMul_3/ReadVariableOp2\
,ode_lstm/rnn/odelstm/MatMul_4/ReadVariableOp,ode_lstm/rnn/odelstm/MatMul_4/ReadVariableOp2\
,ode_lstm/rnn/odelstm/MatMul_5/ReadVariableOp,ode_lstm/rnn/odelstm/MatMul_5/ReadVariableOp2\
,ode_lstm/rnn/odelstm/MatMul_6/ReadVariableOp,ode_lstm/rnn/odelstm/MatMul_6/ReadVariableOp2\
,ode_lstm/rnn/odelstm/MatMul_7/ReadVariableOp,ode_lstm/rnn/odelstm/MatMul_7/ReadVariableOp2\
,ode_lstm/rnn/odelstm/MatMul_8/ReadVariableOp,ode_lstm/rnn/odelstm/MatMul_8/ReadVariableOp2\
,ode_lstm/rnn/odelstm/MatMul_9/ReadVariableOp,ode_lstm/rnn/odelstm/MatMul_9/ReadVariableOp2J
#ode_lstm/rnn/odelstm/ReadVariableOp#ode_lstm/rnn/odelstm/ReadVariableOp2N
%ode_lstm/rnn/odelstm/ReadVariableOp_1%ode_lstm/rnn/odelstm/ReadVariableOp_12N
%ode_lstm/rnn/odelstm/ReadVariableOp_2%ode_lstm/rnn/odelstm/ReadVariableOp_22N
%ode_lstm/rnn/odelstm/ReadVariableOp_3%ode_lstm/rnn/odelstm/ReadVariableOp_32V
)ode_lstm/rnn/odelstm/add_1/ReadVariableOp)ode_lstm/rnn/odelstm/add_1/ReadVariableOp2X
*ode_lstm/rnn/odelstm/add_11/ReadVariableOp*ode_lstm/rnn/odelstm/add_11/ReadVariableOp2X
*ode_lstm/rnn/odelstm/add_14/ReadVariableOp*ode_lstm/rnn/odelstm/add_14/ReadVariableOp2V
)ode_lstm/rnn/odelstm/add_5/ReadVariableOp)ode_lstm/rnn/odelstm/add_5/ReadVariableOp2V
)ode_lstm/rnn/odelstm/add_8/ReadVariableOp)ode_lstm/rnn/odelstm/add_8/ReadVariableOp2(
ode_lstm/rnn/whileode_lstm/rnn/while:L H
#
_output_shapes
:�d
!
_user_specified_name	input_1:LH
#
_output_shapes
:�d
!
_user_specified_name	input_2
�

�
while_cond_58700
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_58700___redundant_placeholder03
/while_while_cond_58700___redundant_placeholder13
/while_while_cond_58700___redundant_placeholder23
/while_while_cond_58700___redundant_placeholder33
/while_while_cond_58700___redundant_placeholder43
/while_while_cond_58700___redundant_placeholder53
/while_while_cond_58700___redundant_placeholder63
/while_while_cond_58700___redundant_placeholder7
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*c
_input_shapesR
P: : : : :���������@:���������@: ::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�	
�
while_cond_61382
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_61382___redundant_placeholder03
/while_while_cond_61382___redundant_placeholder13
/while_while_cond_61382___redundant_placeholder23
/while_while_cond_61382___redundant_placeholder33
/while_while_cond_61382___redundant_placeholder43
/while_while_cond_61382___redundant_placeholder53
/while_while_cond_61382___redundant_placeholder63
/while_while_cond_61382___redundant_placeholder7
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :	�@:	�@: ::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :

_output_shapes
:
ʾ
�
while_body_61383
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
.while_odelstm_matmul_readvariableop_resource_04
0while_odelstm_matmul_1_readvariableop_resource_01
-while_odelstm_add_1_readvariableop_resource_04
0while_odelstm_matmul_2_readvariableop_resource_04
0while_odelstm_matmul_3_readvariableop_resource_01
-while_odelstm_add_5_readvariableop_resource_0+
'while_odelstm_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
,while_odelstm_matmul_readvariableop_resource2
.while_odelstm_matmul_1_readvariableop_resource/
+while_odelstm_add_1_readvariableop_resource2
.while_odelstm_matmul_2_readvariableop_resource2
.while_odelstm_matmul_3_readvariableop_resource/
+while_odelstm_add_5_readvariableop_resource)
%while_odelstm_readvariableop_resource��#while/odelstm/MatMul/ReadVariableOp�%while/odelstm/MatMul_1/ReadVariableOp�%while/odelstm/MatMul_2/ReadVariableOp�%while/odelstm/MatMul_3/ReadVariableOp�%while/odelstm/MatMul_4/ReadVariableOp�%while/odelstm/MatMul_5/ReadVariableOp�%while/odelstm/MatMul_6/ReadVariableOp�%while/odelstm/MatMul_7/ReadVariableOp�%while/odelstm/MatMul_8/ReadVariableOp�%while/odelstm/MatMul_9/ReadVariableOp�while/odelstm/ReadVariableOp�while/odelstm/ReadVariableOp_1�while/odelstm/ReadVariableOp_2�while/odelstm/ReadVariableOp_3�"while/odelstm/add_1/ReadVariableOp�#while/odelstm/add_11/ReadVariableOp�#while/odelstm/add_14/ReadVariableOp�"while/odelstm/add_5/ReadVariableOp�"while/odelstm/add_8/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	�*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
#while/odelstm/MatMul/ReadVariableOpReadVariableOp.while_odelstm_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02%
#while/odelstm/MatMul/ReadVariableOp�
while/odelstm/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/odelstm/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
while/odelstm/MatMul�
%while/odelstm/MatMul_1/ReadVariableOpReadVariableOp0while_odelstm_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02'
%while/odelstm/MatMul_1/ReadVariableOp�
while/odelstm/MatMul_1MatMulwhile_placeholder_3-while/odelstm/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
while/odelstm/MatMul_1�
while/odelstm/addAddV2while/odelstm/MatMul:product:0 while/odelstm/MatMul_1:product:0*
T0* 
_output_shapes
:
��2
while/odelstm/add�
"while/odelstm/add_1/ReadVariableOpReadVariableOp-while_odelstm_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype02$
"while/odelstm/add_1/ReadVariableOp�
while/odelstm/add_1AddV2while/odelstm/add:z:0*while/odelstm/add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
while/odelstm/add_1l
while/odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/odelstm/Const�
while/odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/odelstm/split/split_dim�
while/odelstm/splitSplit&while/odelstm/split/split_dim:output:0while/odelstm/add_1:z:0*
T0*@
_output_shapes.
,:	�@:	�@:	�@:	�@*
	num_split2
while/odelstm/splitx
while/odelstm/TanhTanhwhile/odelstm/split:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh�
while/odelstm/SigmoidSigmoidwhile/odelstm/split:output:1*
T0*
_output_shapes
:	�@2
while/odelstm/Sigmoids
while/odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
while/odelstm/add_2/y�
while/odelstm/add_2AddV2while/odelstm/split:output:2while/odelstm/add_2/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_2�
while/odelstm/Sigmoid_1Sigmoidwhile/odelstm/add_2:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Sigmoid_1�
while/odelstm/Sigmoid_2Sigmoidwhile/odelstm/split:output:3*
T0*
_output_shapes
:	�@2
while/odelstm/Sigmoid_2�
while/odelstm/mulMulwhile_placeholder_2while/odelstm/Sigmoid_1:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul�
while/odelstm/mul_1Mulwhile/odelstm/Tanh:y:0while/odelstm/Sigmoid:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_1�
while/odelstm/add_3AddV2while/odelstm/mul:z:0while/odelstm/mul_1:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_3w
while/odelstm/Tanh_1Tanhwhile/odelstm/add_3:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_1�
while/odelstm/mul_2Mulwhile/odelstm/Tanh_1:y:0while/odelstm/Sigmoid_2:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_2�
%while/odelstm/MatMul_2/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_2/ReadVariableOp�
while/odelstm/MatMul_2MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_2�
%while/odelstm/MatMul_3/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_3/ReadVariableOp�
while/odelstm/MatMul_3MatMulwhile_placeholder_3-while/odelstm/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_3�
while/odelstm/add_4AddV2 while/odelstm/MatMul_2:product:0 while/odelstm/MatMul_3:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_4�
"while/odelstm/add_5/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"while/odelstm/add_5/ReadVariableOp�
while/odelstm/add_5AddV2while/odelstm/add_4:z:0*while/odelstm/add_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_5w
while/odelstm/Tanh_2Tanhwhile/odelstm/add_5:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_2�
while/odelstm/ReadVariableOpReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02
while/odelstm/ReadVariableOp�
while/odelstm/mul_3Mul$while/odelstm/ReadVariableOp:value:0while/odelstm/Tanh_2:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_3s
while/odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_4/y�
while/odelstm/mul_4Mulwhile_placeholder_3while/odelstm/mul_4/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_4�
while/odelstm/subSubwhile/odelstm/mul_3:z:0while/odelstm/mul_4:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/subs
while/odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_5/x�
while/odelstm/mul_5Mulwhile/odelstm/mul_5/x:output:0while/odelstm/sub:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_5�
while/odelstm/add_6AddV2while_placeholder_3while/odelstm/mul_5:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_6�
%while/odelstm/MatMul_4/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_4/ReadVariableOp�
while/odelstm/MatMul_4MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_4�
%while/odelstm/MatMul_5/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_5/ReadVariableOp�
while/odelstm/MatMul_5MatMulwhile/odelstm/add_6:z:0-while/odelstm/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_5�
while/odelstm/add_7AddV2 while/odelstm/MatMul_4:product:0 while/odelstm/MatMul_5:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_7�
"while/odelstm/add_8/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"while/odelstm/add_8/ReadVariableOp�
while/odelstm/add_8AddV2while/odelstm/add_7:z:0*while/odelstm/add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_8w
while/odelstm/Tanh_3Tanhwhile/odelstm/add_8:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_3�
while/odelstm/ReadVariableOp_1ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_1�
while/odelstm/mul_6Mul&while/odelstm/ReadVariableOp_1:value:0while/odelstm/Tanh_3:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_6s
while/odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_7/y�
while/odelstm/mul_7Mulwhile/odelstm/add_6:z:0while/odelstm/mul_7/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_7�
while/odelstm/sub_1Subwhile/odelstm/mul_6:z:0while/odelstm/mul_7:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/sub_1s
while/odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_8/x�
while/odelstm/mul_8Mulwhile/odelstm/mul_8/x:output:0while/odelstm/sub_1:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_8�
while/odelstm/add_9AddV2while/odelstm/add_6:z:0while/odelstm/mul_8:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_9�
%while/odelstm/MatMul_6/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_6/ReadVariableOp�
while/odelstm/MatMul_6MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_6�
%while/odelstm/MatMul_7/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_7/ReadVariableOp�
while/odelstm/MatMul_7MatMulwhile/odelstm/add_9:z:0-while/odelstm/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_7�
while/odelstm/add_10AddV2 while/odelstm/MatMul_6:product:0 while/odelstm/MatMul_7:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_10�
#while/odelstm/add_11/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02%
#while/odelstm/add_11/ReadVariableOp�
while/odelstm/add_11AddV2while/odelstm/add_10:z:0+while/odelstm/add_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_11x
while/odelstm/Tanh_4Tanhwhile/odelstm/add_11:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_4�
while/odelstm/ReadVariableOp_2ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_2�
while/odelstm/mul_9Mul&while/odelstm/ReadVariableOp_2:value:0while/odelstm/Tanh_4:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_9u
while/odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_10/y�
while/odelstm/mul_10Mulwhile/odelstm/add_9:z:0while/odelstm/mul_10/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_10�
while/odelstm/sub_2Subwhile/odelstm/mul_9:z:0while/odelstm/mul_10:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/sub_2u
while/odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_11/x�
while/odelstm/mul_11Mulwhile/odelstm/mul_11/x:output:0while/odelstm/sub_2:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_11�
while/odelstm/add_12AddV2while/odelstm/add_9:z:0while/odelstm/mul_11:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_12�
%while/odelstm/MatMul_8/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_8/ReadVariableOp�
while/odelstm/MatMul_8MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_8�
%while/odelstm/MatMul_9/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_9/ReadVariableOp�
while/odelstm/MatMul_9MatMulwhile/odelstm/add_12:z:0-while/odelstm/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_9�
while/odelstm/add_13AddV2 while/odelstm/MatMul_8:product:0 while/odelstm/MatMul_9:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_13�
#while/odelstm/add_14/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02%
#while/odelstm/add_14/ReadVariableOp�
while/odelstm/add_14AddV2while/odelstm/add_13:z:0+while/odelstm/add_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_14x
while/odelstm/Tanh_5Tanhwhile/odelstm/add_14:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_5�
while/odelstm/ReadVariableOp_3ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_3�
while/odelstm/mul_12Mul&while/odelstm/ReadVariableOp_3:value:0while/odelstm/Tanh_5:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_12u
while/odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_13/y�
while/odelstm/mul_13Mulwhile/odelstm/add_12:z:0while/odelstm/mul_13/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_13�
while/odelstm/sub_3Subwhile/odelstm/mul_12:z:0while/odelstm/mul_13:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/sub_3u
while/odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_14/x�
while/odelstm/mul_14Mulwhile/odelstm/mul_14/x:output:0while/odelstm/sub_3:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_14�
while/odelstm/add_15AddV2while/odelstm/add_12:z:0while/odelstm/mul_14:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_15�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/odelstm/add_15:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/odelstm/add_3:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
:	�@2
while/Identity_4�
while/Identity_5Identitywhile/odelstm/add_15:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
:	�@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"\
+while_odelstm_add_1_readvariableop_resource-while_odelstm_add_1_readvariableop_resource_0"\
+while_odelstm_add_5_readvariableop_resource-while_odelstm_add_5_readvariableop_resource_0"b
.while_odelstm_matmul_1_readvariableop_resource0while_odelstm_matmul_1_readvariableop_resource_0"b
.while_odelstm_matmul_2_readvariableop_resource0while_odelstm_matmul_2_readvariableop_resource_0"b
.while_odelstm_matmul_3_readvariableop_resource0while_odelstm_matmul_3_readvariableop_resource_0"^
,while_odelstm_matmul_readvariableop_resource.while_odelstm_matmul_readvariableop_resource_0"P
%while_odelstm_readvariableop_resource'while_odelstm_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :	�@:	�@: : :::::::2J
#while/odelstm/MatMul/ReadVariableOp#while/odelstm/MatMul/ReadVariableOp2N
%while/odelstm/MatMul_1/ReadVariableOp%while/odelstm/MatMul_1/ReadVariableOp2N
%while/odelstm/MatMul_2/ReadVariableOp%while/odelstm/MatMul_2/ReadVariableOp2N
%while/odelstm/MatMul_3/ReadVariableOp%while/odelstm/MatMul_3/ReadVariableOp2N
%while/odelstm/MatMul_4/ReadVariableOp%while/odelstm/MatMul_4/ReadVariableOp2N
%while/odelstm/MatMul_5/ReadVariableOp%while/odelstm/MatMul_5/ReadVariableOp2N
%while/odelstm/MatMul_6/ReadVariableOp%while/odelstm/MatMul_6/ReadVariableOp2N
%while/odelstm/MatMul_7/ReadVariableOp%while/odelstm/MatMul_7/ReadVariableOp2N
%while/odelstm/MatMul_8/ReadVariableOp%while/odelstm/MatMul_8/ReadVariableOp2N
%while/odelstm/MatMul_9/ReadVariableOp%while/odelstm/MatMul_9/ReadVariableOp2<
while/odelstm/ReadVariableOpwhile/odelstm/ReadVariableOp2@
while/odelstm/ReadVariableOp_1while/odelstm/ReadVariableOp_12@
while/odelstm/ReadVariableOp_2while/odelstm/ReadVariableOp_22@
while/odelstm/ReadVariableOp_3while/odelstm/ReadVariableOp_32H
"while/odelstm/add_1/ReadVariableOp"while/odelstm/add_1/ReadVariableOp2J
#while/odelstm/add_11/ReadVariableOp#while/odelstm/add_11/ReadVariableOp2J
#while/odelstm/add_14/ReadVariableOp#while/odelstm/add_14/ReadVariableOp2H
"while/odelstm/add_5/ReadVariableOp"while/odelstm/add_5/ReadVariableOp2H
"while/odelstm/add_8/ReadVariableOp"while/odelstm/add_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :

_output_shapes
: 
�	
�
(__inference_ode_lstm_layer_call_fn_60283
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_ode_lstm_layer_call_and_return_conditional_losses_595412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:�d:�d:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:�d
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:�d
"
_user_specified_name
inputs/1
��
�
rnn_while_body_59784$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_06
2rnn_while_odelstm_matmul_readvariableop_resource_08
4rnn_while_odelstm_matmul_1_readvariableop_resource_05
1rnn_while_odelstm_add_1_readvariableop_resource_08
4rnn_while_odelstm_matmul_2_readvariableop_resource_08
4rnn_while_odelstm_matmul_3_readvariableop_resource_05
1rnn_while_odelstm_add_5_readvariableop_resource_0/
+rnn_while_odelstm_readvariableop_resource_0
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor4
0rnn_while_odelstm_matmul_readvariableop_resource6
2rnn_while_odelstm_matmul_1_readvariableop_resource3
/rnn_while_odelstm_add_1_readvariableop_resource6
2rnn_while_odelstm_matmul_2_readvariableop_resource6
2rnn_while_odelstm_matmul_3_readvariableop_resource3
/rnn_while_odelstm_add_5_readvariableop_resource-
)rnn_while_odelstm_readvariableop_resource��'rnn/while/odelstm/MatMul/ReadVariableOp�)rnn/while/odelstm/MatMul_1/ReadVariableOp�)rnn/while/odelstm/MatMul_2/ReadVariableOp�)rnn/while/odelstm/MatMul_3/ReadVariableOp�)rnn/while/odelstm/MatMul_4/ReadVariableOp�)rnn/while/odelstm/MatMul_5/ReadVariableOp�)rnn/while/odelstm/MatMul_6/ReadVariableOp�)rnn/while/odelstm/MatMul_7/ReadVariableOp�)rnn/while/odelstm/MatMul_8/ReadVariableOp�)rnn/while/odelstm/MatMul_9/ReadVariableOp� rnn/while/odelstm/ReadVariableOp�"rnn/while/odelstm/ReadVariableOp_1�"rnn/while/odelstm/ReadVariableOp_2�"rnn/while/odelstm/ReadVariableOp_3�&rnn/while/odelstm/add_1/ReadVariableOp�'rnn/while/odelstm/add_11/ReadVariableOp�'rnn/while/odelstm/add_14/ReadVariableOp�&rnn/while/odelstm/add_5/ReadVariableOp�&rnn/while/odelstm/add_8/ReadVariableOp�
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�      2=
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape�
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	�*
element_dtype02/
-rnn/while/TensorArrayV2Read/TensorListGetItem�
'rnn/while/odelstm/MatMul/ReadVariableOpReadVariableOp2rnn_while_odelstm_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02)
'rnn/while/odelstm/MatMul/ReadVariableOp�
rnn/while/odelstm/MatMulMatMul4rnn/while/TensorArrayV2Read/TensorListGetItem:item:0/rnn/while/odelstm/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
rnn/while/odelstm/MatMul�
)rnn/while/odelstm/MatMul_1/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02+
)rnn/while/odelstm/MatMul_1/ReadVariableOp�
rnn/while/odelstm/MatMul_1MatMulrnn_while_placeholder_31rnn/while/odelstm/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
rnn/while/odelstm/MatMul_1�
rnn/while/odelstm/addAddV2"rnn/while/odelstm/MatMul:product:0$rnn/while/odelstm/MatMul_1:product:0*
T0* 
_output_shapes
:
��2
rnn/while/odelstm/add�
&rnn/while/odelstm/add_1/ReadVariableOpReadVariableOp1rnn_while_odelstm_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype02(
&rnn/while/odelstm/add_1/ReadVariableOp�
rnn/while/odelstm/add_1AddV2rnn/while/odelstm/add:z:0.rnn/while/odelstm/add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
rnn/while/odelstm/add_1t
rnn/while/odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/odelstm/Const�
!rnn/while/odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!rnn/while/odelstm/split/split_dim�
rnn/while/odelstm/splitSplit*rnn/while/odelstm/split/split_dim:output:0rnn/while/odelstm/add_1:z:0*
T0*@
_output_shapes.
,:	�@:	�@:	�@:	�@*
	num_split2
rnn/while/odelstm/split�
rnn/while/odelstm/TanhTanh rnn/while/odelstm/split:output:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Tanh�
rnn/while/odelstm/SigmoidSigmoid rnn/while/odelstm/split:output:1*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Sigmoid{
rnn/while/odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
rnn/while/odelstm/add_2/y�
rnn/while/odelstm/add_2AddV2 rnn/while/odelstm/split:output:2"rnn/while/odelstm/add_2/y:output:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_2�
rnn/while/odelstm/Sigmoid_1Sigmoidrnn/while/odelstm/add_2:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Sigmoid_1�
rnn/while/odelstm/Sigmoid_2Sigmoid rnn/while/odelstm/split:output:3*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Sigmoid_2�
rnn/while/odelstm/mulMulrnn_while_placeholder_2rnn/while/odelstm/Sigmoid_1:y:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul�
rnn/while/odelstm/mul_1Mulrnn/while/odelstm/Tanh:y:0rnn/while/odelstm/Sigmoid:y:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_1�
rnn/while/odelstm/add_3AddV2rnn/while/odelstm/mul:z:0rnn/while/odelstm/mul_1:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_3�
rnn/while/odelstm/Tanh_1Tanhrnn/while/odelstm/add_3:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Tanh_1�
rnn/while/odelstm/mul_2Mulrnn/while/odelstm/Tanh_1:y:0rnn/while/odelstm/Sigmoid_2:y:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_2�
)rnn/while/odelstm/MatMul_2/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_2/ReadVariableOp�
rnn/while/odelstm/MatMul_2MatMulrnn/while/odelstm/mul_2:z:01rnn/while/odelstm/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_2�
)rnn/while/odelstm/MatMul_3/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_3/ReadVariableOp�
rnn/while/odelstm/MatMul_3MatMulrnn_while_placeholder_31rnn/while/odelstm/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_3�
rnn/while/odelstm/add_4AddV2$rnn/while/odelstm/MatMul_2:product:0$rnn/while/odelstm/MatMul_3:product:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_4�
&rnn/while/odelstm/add_5/ReadVariableOpReadVariableOp1rnn_while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&rnn/while/odelstm/add_5/ReadVariableOp�
rnn/while/odelstm/add_5AddV2rnn/while/odelstm/add_4:z:0.rnn/while/odelstm/add_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_5�
rnn/while/odelstm/Tanh_2Tanhrnn/while/odelstm/add_5:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Tanh_2�
 rnn/while/odelstm/ReadVariableOpReadVariableOp+rnn_while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 rnn/while/odelstm/ReadVariableOp�
rnn/while/odelstm/mul_3Mul(rnn/while/odelstm/ReadVariableOp:value:0rnn/while/odelstm/Tanh_2:y:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_3{
rnn/while/odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/while/odelstm/mul_4/y�
rnn/while/odelstm/mul_4Mulrnn_while_placeholder_3"rnn/while/odelstm/mul_4/y:output:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_4�
rnn/while/odelstm/subSubrnn/while/odelstm/mul_3:z:0rnn/while/odelstm/mul_4:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/sub{
rnn/while/odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/while/odelstm/mul_5/x�
rnn/while/odelstm/mul_5Mul"rnn/while/odelstm/mul_5/x:output:0rnn/while/odelstm/sub:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_5�
rnn/while/odelstm/add_6AddV2rnn_while_placeholder_3rnn/while/odelstm/mul_5:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_6�
)rnn/while/odelstm/MatMul_4/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_4/ReadVariableOp�
rnn/while/odelstm/MatMul_4MatMulrnn/while/odelstm/mul_2:z:01rnn/while/odelstm/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_4�
)rnn/while/odelstm/MatMul_5/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_5/ReadVariableOp�
rnn/while/odelstm/MatMul_5MatMulrnn/while/odelstm/add_6:z:01rnn/while/odelstm/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_5�
rnn/while/odelstm/add_7AddV2$rnn/while/odelstm/MatMul_4:product:0$rnn/while/odelstm/MatMul_5:product:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_7�
&rnn/while/odelstm/add_8/ReadVariableOpReadVariableOp1rnn_while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&rnn/while/odelstm/add_8/ReadVariableOp�
rnn/while/odelstm/add_8AddV2rnn/while/odelstm/add_7:z:0.rnn/while/odelstm/add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_8�
rnn/while/odelstm/Tanh_3Tanhrnn/while/odelstm/add_8:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Tanh_3�
"rnn/while/odelstm/ReadVariableOp_1ReadVariableOp+rnn_while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"rnn/while/odelstm/ReadVariableOp_1�
rnn/while/odelstm/mul_6Mul*rnn/while/odelstm/ReadVariableOp_1:value:0rnn/while/odelstm/Tanh_3:y:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_6{
rnn/while/odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/while/odelstm/mul_7/y�
rnn/while/odelstm/mul_7Mulrnn/while/odelstm/add_6:z:0"rnn/while/odelstm/mul_7/y:output:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_7�
rnn/while/odelstm/sub_1Subrnn/while/odelstm/mul_6:z:0rnn/while/odelstm/mul_7:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/sub_1{
rnn/while/odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/while/odelstm/mul_8/x�
rnn/while/odelstm/mul_8Mul"rnn/while/odelstm/mul_8/x:output:0rnn/while/odelstm/sub_1:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_8�
rnn/while/odelstm/add_9AddV2rnn/while/odelstm/add_6:z:0rnn/while/odelstm/mul_8:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_9�
)rnn/while/odelstm/MatMul_6/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_6/ReadVariableOp�
rnn/while/odelstm/MatMul_6MatMulrnn/while/odelstm/mul_2:z:01rnn/while/odelstm/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_6�
)rnn/while/odelstm/MatMul_7/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_7/ReadVariableOp�
rnn/while/odelstm/MatMul_7MatMulrnn/while/odelstm/add_9:z:01rnn/while/odelstm/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_7�
rnn/while/odelstm/add_10AddV2$rnn/while/odelstm/MatMul_6:product:0$rnn/while/odelstm/MatMul_7:product:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_10�
'rnn/while/odelstm/add_11/ReadVariableOpReadVariableOp1rnn_while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02)
'rnn/while/odelstm/add_11/ReadVariableOp�
rnn/while/odelstm/add_11AddV2rnn/while/odelstm/add_10:z:0/rnn/while/odelstm/add_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_11�
rnn/while/odelstm/Tanh_4Tanhrnn/while/odelstm/add_11:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Tanh_4�
"rnn/while/odelstm/ReadVariableOp_2ReadVariableOp+rnn_while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"rnn/while/odelstm/ReadVariableOp_2�
rnn/while/odelstm/mul_9Mul*rnn/while/odelstm/ReadVariableOp_2:value:0rnn/while/odelstm/Tanh_4:y:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_9}
rnn/while/odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/while/odelstm/mul_10/y�
rnn/while/odelstm/mul_10Mulrnn/while/odelstm/add_9:z:0#rnn/while/odelstm/mul_10/y:output:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_10�
rnn/while/odelstm/sub_2Subrnn/while/odelstm/mul_9:z:0rnn/while/odelstm/mul_10:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/sub_2}
rnn/while/odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/while/odelstm/mul_11/x�
rnn/while/odelstm/mul_11Mul#rnn/while/odelstm/mul_11/x:output:0rnn/while/odelstm/sub_2:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_11�
rnn/while/odelstm/add_12AddV2rnn/while/odelstm/add_9:z:0rnn/while/odelstm/mul_11:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_12�
)rnn/while/odelstm/MatMul_8/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_8/ReadVariableOp�
rnn/while/odelstm/MatMul_8MatMulrnn/while/odelstm/mul_2:z:01rnn/while/odelstm/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_8�
)rnn/while/odelstm/MatMul_9/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_9/ReadVariableOp�
rnn/while/odelstm/MatMul_9MatMulrnn/while/odelstm/add_12:z:01rnn/while/odelstm/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_9�
rnn/while/odelstm/add_13AddV2$rnn/while/odelstm/MatMul_8:product:0$rnn/while/odelstm/MatMul_9:product:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_13�
'rnn/while/odelstm/add_14/ReadVariableOpReadVariableOp1rnn_while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02)
'rnn/while/odelstm/add_14/ReadVariableOp�
rnn/while/odelstm/add_14AddV2rnn/while/odelstm/add_13:z:0/rnn/while/odelstm/add_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_14�
rnn/while/odelstm/Tanh_5Tanhrnn/while/odelstm/add_14:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Tanh_5�
"rnn/while/odelstm/ReadVariableOp_3ReadVariableOp+rnn_while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"rnn/while/odelstm/ReadVariableOp_3�
rnn/while/odelstm/mul_12Mul*rnn/while/odelstm/ReadVariableOp_3:value:0rnn/while/odelstm/Tanh_5:y:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_12}
rnn/while/odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/while/odelstm/mul_13/y�
rnn/while/odelstm/mul_13Mulrnn/while/odelstm/add_12:z:0#rnn/while/odelstm/mul_13/y:output:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_13�
rnn/while/odelstm/sub_3Subrnn/while/odelstm/mul_12:z:0rnn/while/odelstm/mul_13:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/sub_3}
rnn/while/odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/while/odelstm/mul_14/x�
rnn/while/odelstm/mul_14Mul#rnn/while/odelstm/mul_14/x:output:0rnn/while/odelstm/sub_3:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_14�
rnn/while/odelstm/add_15AddV2rnn/while/odelstm/add_12:z:0rnn/while/odelstm/mul_14:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_15�
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholderrnn/while/odelstm/add_15:z:0*
_output_shapes
: *
element_dtype020
.rnn/while/TensorArrayV2Write/TensorListSetItemd
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add/yy
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: 2
rnn/while/addh
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add_1/y�
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rnn/while/add_1�
rnn/while/IdentityIdentityrnn/while/add_1:z:0(^rnn/while/odelstm/MatMul/ReadVariableOp*^rnn/while/odelstm/MatMul_1/ReadVariableOp*^rnn/while/odelstm/MatMul_2/ReadVariableOp*^rnn/while/odelstm/MatMul_3/ReadVariableOp*^rnn/while/odelstm/MatMul_4/ReadVariableOp*^rnn/while/odelstm/MatMul_5/ReadVariableOp*^rnn/while/odelstm/MatMul_6/ReadVariableOp*^rnn/while/odelstm/MatMul_7/ReadVariableOp*^rnn/while/odelstm/MatMul_8/ReadVariableOp*^rnn/while/odelstm/MatMul_9/ReadVariableOp!^rnn/while/odelstm/ReadVariableOp#^rnn/while/odelstm/ReadVariableOp_1#^rnn/while/odelstm/ReadVariableOp_2#^rnn/while/odelstm/ReadVariableOp_3'^rnn/while/odelstm/add_1/ReadVariableOp(^rnn/while/odelstm/add_11/ReadVariableOp(^rnn/while/odelstm/add_14/ReadVariableOp'^rnn/while/odelstm/add_5/ReadVariableOp'^rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity�
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations(^rnn/while/odelstm/MatMul/ReadVariableOp*^rnn/while/odelstm/MatMul_1/ReadVariableOp*^rnn/while/odelstm/MatMul_2/ReadVariableOp*^rnn/while/odelstm/MatMul_3/ReadVariableOp*^rnn/while/odelstm/MatMul_4/ReadVariableOp*^rnn/while/odelstm/MatMul_5/ReadVariableOp*^rnn/while/odelstm/MatMul_6/ReadVariableOp*^rnn/while/odelstm/MatMul_7/ReadVariableOp*^rnn/while/odelstm/MatMul_8/ReadVariableOp*^rnn/while/odelstm/MatMul_9/ReadVariableOp!^rnn/while/odelstm/ReadVariableOp#^rnn/while/odelstm/ReadVariableOp_1#^rnn/while/odelstm/ReadVariableOp_2#^rnn/while/odelstm/ReadVariableOp_3'^rnn/while/odelstm/add_1/ReadVariableOp(^rnn/while/odelstm/add_11/ReadVariableOp(^rnn/while/odelstm/add_14/ReadVariableOp'^rnn/while/odelstm/add_5/ReadVariableOp'^rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity_1�
rnn/while/Identity_2Identityrnn/while/add:z:0(^rnn/while/odelstm/MatMul/ReadVariableOp*^rnn/while/odelstm/MatMul_1/ReadVariableOp*^rnn/while/odelstm/MatMul_2/ReadVariableOp*^rnn/while/odelstm/MatMul_3/ReadVariableOp*^rnn/while/odelstm/MatMul_4/ReadVariableOp*^rnn/while/odelstm/MatMul_5/ReadVariableOp*^rnn/while/odelstm/MatMul_6/ReadVariableOp*^rnn/while/odelstm/MatMul_7/ReadVariableOp*^rnn/while/odelstm/MatMul_8/ReadVariableOp*^rnn/while/odelstm/MatMul_9/ReadVariableOp!^rnn/while/odelstm/ReadVariableOp#^rnn/while/odelstm/ReadVariableOp_1#^rnn/while/odelstm/ReadVariableOp_2#^rnn/while/odelstm/ReadVariableOp_3'^rnn/while/odelstm/add_1/ReadVariableOp(^rnn/while/odelstm/add_11/ReadVariableOp(^rnn/while/odelstm/add_14/ReadVariableOp'^rnn/while/odelstm/add_5/ReadVariableOp'^rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity_2�
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^rnn/while/odelstm/MatMul/ReadVariableOp*^rnn/while/odelstm/MatMul_1/ReadVariableOp*^rnn/while/odelstm/MatMul_2/ReadVariableOp*^rnn/while/odelstm/MatMul_3/ReadVariableOp*^rnn/while/odelstm/MatMul_4/ReadVariableOp*^rnn/while/odelstm/MatMul_5/ReadVariableOp*^rnn/while/odelstm/MatMul_6/ReadVariableOp*^rnn/while/odelstm/MatMul_7/ReadVariableOp*^rnn/while/odelstm/MatMul_8/ReadVariableOp*^rnn/while/odelstm/MatMul_9/ReadVariableOp!^rnn/while/odelstm/ReadVariableOp#^rnn/while/odelstm/ReadVariableOp_1#^rnn/while/odelstm/ReadVariableOp_2#^rnn/while/odelstm/ReadVariableOp_3'^rnn/while/odelstm/add_1/ReadVariableOp(^rnn/while/odelstm/add_11/ReadVariableOp(^rnn/while/odelstm/add_14/ReadVariableOp'^rnn/while/odelstm/add_5/ReadVariableOp'^rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity_3�
rnn/while/Identity_4Identityrnn/while/odelstm/add_3:z:0(^rnn/while/odelstm/MatMul/ReadVariableOp*^rnn/while/odelstm/MatMul_1/ReadVariableOp*^rnn/while/odelstm/MatMul_2/ReadVariableOp*^rnn/while/odelstm/MatMul_3/ReadVariableOp*^rnn/while/odelstm/MatMul_4/ReadVariableOp*^rnn/while/odelstm/MatMul_5/ReadVariableOp*^rnn/while/odelstm/MatMul_6/ReadVariableOp*^rnn/while/odelstm/MatMul_7/ReadVariableOp*^rnn/while/odelstm/MatMul_8/ReadVariableOp*^rnn/while/odelstm/MatMul_9/ReadVariableOp!^rnn/while/odelstm/ReadVariableOp#^rnn/while/odelstm/ReadVariableOp_1#^rnn/while/odelstm/ReadVariableOp_2#^rnn/while/odelstm/ReadVariableOp_3'^rnn/while/odelstm/add_1/ReadVariableOp(^rnn/while/odelstm/add_11/ReadVariableOp(^rnn/while/odelstm/add_14/ReadVariableOp'^rnn/while/odelstm/add_5/ReadVariableOp'^rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
:	�@2
rnn/while/Identity_4�
rnn/while/Identity_5Identityrnn/while/odelstm/add_15:z:0(^rnn/while/odelstm/MatMul/ReadVariableOp*^rnn/while/odelstm/MatMul_1/ReadVariableOp*^rnn/while/odelstm/MatMul_2/ReadVariableOp*^rnn/while/odelstm/MatMul_3/ReadVariableOp*^rnn/while/odelstm/MatMul_4/ReadVariableOp*^rnn/while/odelstm/MatMul_5/ReadVariableOp*^rnn/while/odelstm/MatMul_6/ReadVariableOp*^rnn/while/odelstm/MatMul_7/ReadVariableOp*^rnn/while/odelstm/MatMul_8/ReadVariableOp*^rnn/while/odelstm/MatMul_9/ReadVariableOp!^rnn/while/odelstm/ReadVariableOp#^rnn/while/odelstm/ReadVariableOp_1#^rnn/while/odelstm/ReadVariableOp_2#^rnn/while/odelstm/ReadVariableOp_3'^rnn/while/odelstm/add_1/ReadVariableOp(^rnn/while/odelstm/add_11/ReadVariableOp(^rnn/while/odelstm/add_14/ReadVariableOp'^rnn/while/odelstm/add_5/ReadVariableOp'^rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
:	�@2
rnn/while/Identity_5"1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"d
/rnn_while_odelstm_add_1_readvariableop_resource1rnn_while_odelstm_add_1_readvariableop_resource_0"d
/rnn_while_odelstm_add_5_readvariableop_resource1rnn_while_odelstm_add_5_readvariableop_resource_0"j
2rnn_while_odelstm_matmul_1_readvariableop_resource4rnn_while_odelstm_matmul_1_readvariableop_resource_0"j
2rnn_while_odelstm_matmul_2_readvariableop_resource4rnn_while_odelstm_matmul_2_readvariableop_resource_0"j
2rnn_while_odelstm_matmul_3_readvariableop_resource4rnn_while_odelstm_matmul_3_readvariableop_resource_0"f
0rnn_while_odelstm_matmul_readvariableop_resource2rnn_while_odelstm_matmul_readvariableop_resource_0"X
)rnn_while_odelstm_readvariableop_resource+rnn_while_odelstm_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"�
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :	�@:	�@: : :::::::2R
'rnn/while/odelstm/MatMul/ReadVariableOp'rnn/while/odelstm/MatMul/ReadVariableOp2V
)rnn/while/odelstm/MatMul_1/ReadVariableOp)rnn/while/odelstm/MatMul_1/ReadVariableOp2V
)rnn/while/odelstm/MatMul_2/ReadVariableOp)rnn/while/odelstm/MatMul_2/ReadVariableOp2V
)rnn/while/odelstm/MatMul_3/ReadVariableOp)rnn/while/odelstm/MatMul_3/ReadVariableOp2V
)rnn/while/odelstm/MatMul_4/ReadVariableOp)rnn/while/odelstm/MatMul_4/ReadVariableOp2V
)rnn/while/odelstm/MatMul_5/ReadVariableOp)rnn/while/odelstm/MatMul_5/ReadVariableOp2V
)rnn/while/odelstm/MatMul_6/ReadVariableOp)rnn/while/odelstm/MatMul_6/ReadVariableOp2V
)rnn/while/odelstm/MatMul_7/ReadVariableOp)rnn/while/odelstm/MatMul_7/ReadVariableOp2V
)rnn/while/odelstm/MatMul_8/ReadVariableOp)rnn/while/odelstm/MatMul_8/ReadVariableOp2V
)rnn/while/odelstm/MatMul_9/ReadVariableOp)rnn/while/odelstm/MatMul_9/ReadVariableOp2D
 rnn/while/odelstm/ReadVariableOp rnn/while/odelstm/ReadVariableOp2H
"rnn/while/odelstm/ReadVariableOp_1"rnn/while/odelstm/ReadVariableOp_12H
"rnn/while/odelstm/ReadVariableOp_2"rnn/while/odelstm/ReadVariableOp_22H
"rnn/while/odelstm/ReadVariableOp_3"rnn/while/odelstm/ReadVariableOp_32P
&rnn/while/odelstm/add_1/ReadVariableOp&rnn/while/odelstm/add_1/ReadVariableOp2R
'rnn/while/odelstm/add_11/ReadVariableOp'rnn/while/odelstm/add_11/ReadVariableOp2R
'rnn/while/odelstm/add_14/ReadVariableOp'rnn/while/odelstm/add_14/ReadVariableOp2P
&rnn/while/odelstm/add_5/ReadVariableOp&rnn/while/odelstm/add_5/ReadVariableOp2P
&rnn/while/odelstm/add_8/ReadVariableOp&rnn/while/odelstm/add_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :

_output_shapes
: 
�	
�
while_cond_58943
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_58943___redundant_placeholder03
/while_while_cond_58943___redundant_placeholder13
/while_while_cond_58943___redundant_placeholder23
/while_while_cond_58943___redundant_placeholder33
/while_while_cond_58943___redundant_placeholder43
/while_while_cond_58943___redundant_placeholder53
/while_while_cond_58943___redundant_placeholder63
/while_while_cond_58943___redundant_placeholder7
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :	�@:	�@: ::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :

_output_shapes
:
ʾ
�
while_body_61083
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
.while_odelstm_matmul_readvariableop_resource_04
0while_odelstm_matmul_1_readvariableop_resource_01
-while_odelstm_add_1_readvariableop_resource_04
0while_odelstm_matmul_2_readvariableop_resource_04
0while_odelstm_matmul_3_readvariableop_resource_01
-while_odelstm_add_5_readvariableop_resource_0+
'while_odelstm_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
,while_odelstm_matmul_readvariableop_resource2
.while_odelstm_matmul_1_readvariableop_resource/
+while_odelstm_add_1_readvariableop_resource2
.while_odelstm_matmul_2_readvariableop_resource2
.while_odelstm_matmul_3_readvariableop_resource/
+while_odelstm_add_5_readvariableop_resource)
%while_odelstm_readvariableop_resource��#while/odelstm/MatMul/ReadVariableOp�%while/odelstm/MatMul_1/ReadVariableOp�%while/odelstm/MatMul_2/ReadVariableOp�%while/odelstm/MatMul_3/ReadVariableOp�%while/odelstm/MatMul_4/ReadVariableOp�%while/odelstm/MatMul_5/ReadVariableOp�%while/odelstm/MatMul_6/ReadVariableOp�%while/odelstm/MatMul_7/ReadVariableOp�%while/odelstm/MatMul_8/ReadVariableOp�%while/odelstm/MatMul_9/ReadVariableOp�while/odelstm/ReadVariableOp�while/odelstm/ReadVariableOp_1�while/odelstm/ReadVariableOp_2�while/odelstm/ReadVariableOp_3�"while/odelstm/add_1/ReadVariableOp�#while/odelstm/add_11/ReadVariableOp�#while/odelstm/add_14/ReadVariableOp�"while/odelstm/add_5/ReadVariableOp�"while/odelstm/add_8/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	�*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
#while/odelstm/MatMul/ReadVariableOpReadVariableOp.while_odelstm_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02%
#while/odelstm/MatMul/ReadVariableOp�
while/odelstm/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/odelstm/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
while/odelstm/MatMul�
%while/odelstm/MatMul_1/ReadVariableOpReadVariableOp0while_odelstm_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02'
%while/odelstm/MatMul_1/ReadVariableOp�
while/odelstm/MatMul_1MatMulwhile_placeholder_3-while/odelstm/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
while/odelstm/MatMul_1�
while/odelstm/addAddV2while/odelstm/MatMul:product:0 while/odelstm/MatMul_1:product:0*
T0* 
_output_shapes
:
��2
while/odelstm/add�
"while/odelstm/add_1/ReadVariableOpReadVariableOp-while_odelstm_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype02$
"while/odelstm/add_1/ReadVariableOp�
while/odelstm/add_1AddV2while/odelstm/add:z:0*while/odelstm/add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
while/odelstm/add_1l
while/odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/odelstm/Const�
while/odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/odelstm/split/split_dim�
while/odelstm/splitSplit&while/odelstm/split/split_dim:output:0while/odelstm/add_1:z:0*
T0*@
_output_shapes.
,:	�@:	�@:	�@:	�@*
	num_split2
while/odelstm/splitx
while/odelstm/TanhTanhwhile/odelstm/split:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh�
while/odelstm/SigmoidSigmoidwhile/odelstm/split:output:1*
T0*
_output_shapes
:	�@2
while/odelstm/Sigmoids
while/odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
while/odelstm/add_2/y�
while/odelstm/add_2AddV2while/odelstm/split:output:2while/odelstm/add_2/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_2�
while/odelstm/Sigmoid_1Sigmoidwhile/odelstm/add_2:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Sigmoid_1�
while/odelstm/Sigmoid_2Sigmoidwhile/odelstm/split:output:3*
T0*
_output_shapes
:	�@2
while/odelstm/Sigmoid_2�
while/odelstm/mulMulwhile_placeholder_2while/odelstm/Sigmoid_1:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul�
while/odelstm/mul_1Mulwhile/odelstm/Tanh:y:0while/odelstm/Sigmoid:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_1�
while/odelstm/add_3AddV2while/odelstm/mul:z:0while/odelstm/mul_1:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_3w
while/odelstm/Tanh_1Tanhwhile/odelstm/add_3:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_1�
while/odelstm/mul_2Mulwhile/odelstm/Tanh_1:y:0while/odelstm/Sigmoid_2:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_2�
%while/odelstm/MatMul_2/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_2/ReadVariableOp�
while/odelstm/MatMul_2MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_2�
%while/odelstm/MatMul_3/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_3/ReadVariableOp�
while/odelstm/MatMul_3MatMulwhile_placeholder_3-while/odelstm/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_3�
while/odelstm/add_4AddV2 while/odelstm/MatMul_2:product:0 while/odelstm/MatMul_3:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_4�
"while/odelstm/add_5/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"while/odelstm/add_5/ReadVariableOp�
while/odelstm/add_5AddV2while/odelstm/add_4:z:0*while/odelstm/add_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_5w
while/odelstm/Tanh_2Tanhwhile/odelstm/add_5:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_2�
while/odelstm/ReadVariableOpReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02
while/odelstm/ReadVariableOp�
while/odelstm/mul_3Mul$while/odelstm/ReadVariableOp:value:0while/odelstm/Tanh_2:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_3s
while/odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_4/y�
while/odelstm/mul_4Mulwhile_placeholder_3while/odelstm/mul_4/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_4�
while/odelstm/subSubwhile/odelstm/mul_3:z:0while/odelstm/mul_4:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/subs
while/odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_5/x�
while/odelstm/mul_5Mulwhile/odelstm/mul_5/x:output:0while/odelstm/sub:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_5�
while/odelstm/add_6AddV2while_placeholder_3while/odelstm/mul_5:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_6�
%while/odelstm/MatMul_4/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_4/ReadVariableOp�
while/odelstm/MatMul_4MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_4�
%while/odelstm/MatMul_5/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_5/ReadVariableOp�
while/odelstm/MatMul_5MatMulwhile/odelstm/add_6:z:0-while/odelstm/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_5�
while/odelstm/add_7AddV2 while/odelstm/MatMul_4:product:0 while/odelstm/MatMul_5:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_7�
"while/odelstm/add_8/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"while/odelstm/add_8/ReadVariableOp�
while/odelstm/add_8AddV2while/odelstm/add_7:z:0*while/odelstm/add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_8w
while/odelstm/Tanh_3Tanhwhile/odelstm/add_8:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_3�
while/odelstm/ReadVariableOp_1ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_1�
while/odelstm/mul_6Mul&while/odelstm/ReadVariableOp_1:value:0while/odelstm/Tanh_3:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_6s
while/odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_7/y�
while/odelstm/mul_7Mulwhile/odelstm/add_6:z:0while/odelstm/mul_7/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_7�
while/odelstm/sub_1Subwhile/odelstm/mul_6:z:0while/odelstm/mul_7:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/sub_1s
while/odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_8/x�
while/odelstm/mul_8Mulwhile/odelstm/mul_8/x:output:0while/odelstm/sub_1:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_8�
while/odelstm/add_9AddV2while/odelstm/add_6:z:0while/odelstm/mul_8:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_9�
%while/odelstm/MatMul_6/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_6/ReadVariableOp�
while/odelstm/MatMul_6MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_6�
%while/odelstm/MatMul_7/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_7/ReadVariableOp�
while/odelstm/MatMul_7MatMulwhile/odelstm/add_9:z:0-while/odelstm/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_7�
while/odelstm/add_10AddV2 while/odelstm/MatMul_6:product:0 while/odelstm/MatMul_7:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_10�
#while/odelstm/add_11/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02%
#while/odelstm/add_11/ReadVariableOp�
while/odelstm/add_11AddV2while/odelstm/add_10:z:0+while/odelstm/add_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_11x
while/odelstm/Tanh_4Tanhwhile/odelstm/add_11:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_4�
while/odelstm/ReadVariableOp_2ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_2�
while/odelstm/mul_9Mul&while/odelstm/ReadVariableOp_2:value:0while/odelstm/Tanh_4:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_9u
while/odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_10/y�
while/odelstm/mul_10Mulwhile/odelstm/add_9:z:0while/odelstm/mul_10/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_10�
while/odelstm/sub_2Subwhile/odelstm/mul_9:z:0while/odelstm/mul_10:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/sub_2u
while/odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_11/x�
while/odelstm/mul_11Mulwhile/odelstm/mul_11/x:output:0while/odelstm/sub_2:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_11�
while/odelstm/add_12AddV2while/odelstm/add_9:z:0while/odelstm/mul_11:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_12�
%while/odelstm/MatMul_8/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_8/ReadVariableOp�
while/odelstm/MatMul_8MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_8�
%while/odelstm/MatMul_9/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_9/ReadVariableOp�
while/odelstm/MatMul_9MatMulwhile/odelstm/add_12:z:0-while/odelstm/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/MatMul_9�
while/odelstm/add_13AddV2 while/odelstm/MatMul_8:product:0 while/odelstm/MatMul_9:product:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_13�
#while/odelstm/add_14/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02%
#while/odelstm/add_14/ReadVariableOp�
while/odelstm/add_14AddV2while/odelstm/add_13:z:0+while/odelstm/add_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_14x
while/odelstm/Tanh_5Tanhwhile/odelstm/add_14:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/Tanh_5�
while/odelstm/ReadVariableOp_3ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_3�
while/odelstm/mul_12Mul&while/odelstm/ReadVariableOp_3:value:0while/odelstm/Tanh_5:y:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_12u
while/odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_13/y�
while/odelstm/mul_13Mulwhile/odelstm/add_12:z:0while/odelstm/mul_13/y:output:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_13�
while/odelstm/sub_3Subwhile/odelstm/mul_12:z:0while/odelstm/mul_13:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/sub_3u
while/odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_14/x�
while/odelstm/mul_14Mulwhile/odelstm/mul_14/x:output:0while/odelstm/sub_3:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/mul_14�
while/odelstm/add_15AddV2while/odelstm/add_12:z:0while/odelstm/mul_14:z:0*
T0*
_output_shapes
:	�@2
while/odelstm/add_15�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/odelstm/add_15:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/odelstm/add_3:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
:	�@2
while/Identity_4�
while/Identity_5Identitywhile/odelstm/add_15:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
:	�@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"\
+while_odelstm_add_1_readvariableop_resource-while_odelstm_add_1_readvariableop_resource_0"\
+while_odelstm_add_5_readvariableop_resource-while_odelstm_add_5_readvariableop_resource_0"b
.while_odelstm_matmul_1_readvariableop_resource0while_odelstm_matmul_1_readvariableop_resource_0"b
.while_odelstm_matmul_2_readvariableop_resource0while_odelstm_matmul_2_readvariableop_resource_0"b
.while_odelstm_matmul_3_readvariableop_resource0while_odelstm_matmul_3_readvariableop_resource_0"^
,while_odelstm_matmul_readvariableop_resource.while_odelstm_matmul_readvariableop_resource_0"P
%while_odelstm_readvariableop_resource'while_odelstm_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :	�@:	�@: : :::::::2J
#while/odelstm/MatMul/ReadVariableOp#while/odelstm/MatMul/ReadVariableOp2N
%while/odelstm/MatMul_1/ReadVariableOp%while/odelstm/MatMul_1/ReadVariableOp2N
%while/odelstm/MatMul_2/ReadVariableOp%while/odelstm/MatMul_2/ReadVariableOp2N
%while/odelstm/MatMul_3/ReadVariableOp%while/odelstm/MatMul_3/ReadVariableOp2N
%while/odelstm/MatMul_4/ReadVariableOp%while/odelstm/MatMul_4/ReadVariableOp2N
%while/odelstm/MatMul_5/ReadVariableOp%while/odelstm/MatMul_5/ReadVariableOp2N
%while/odelstm/MatMul_6/ReadVariableOp%while/odelstm/MatMul_6/ReadVariableOp2N
%while/odelstm/MatMul_7/ReadVariableOp%while/odelstm/MatMul_7/ReadVariableOp2N
%while/odelstm/MatMul_8/ReadVariableOp%while/odelstm/MatMul_8/ReadVariableOp2N
%while/odelstm/MatMul_9/ReadVariableOp%while/odelstm/MatMul_9/ReadVariableOp2<
while/odelstm/ReadVariableOpwhile/odelstm/ReadVariableOp2@
while/odelstm/ReadVariableOp_1while/odelstm/ReadVariableOp_12@
while/odelstm/ReadVariableOp_2while/odelstm/ReadVariableOp_22@
while/odelstm/ReadVariableOp_3while/odelstm/ReadVariableOp_32H
"while/odelstm/add_1/ReadVariableOp"while/odelstm/add_1/ReadVariableOp2J
#while/odelstm/add_11/ReadVariableOp#while/odelstm/add_11/ReadVariableOp2J
#while/odelstm/add_14/ReadVariableOp#while/odelstm/add_14/ReadVariableOp2H
"while/odelstm/add_5/ReadVariableOp"while/odelstm/add_5/ReadVariableOp2H
"while/odelstm/add_8/ReadVariableOp"while/odelstm/add_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :

_output_shapes
: 
�'
�
while_body_58538
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_odelstm_58562_0
while_odelstm_58564_0
while_odelstm_58566_0
while_odelstm_58568_0
while_odelstm_58570_0
while_odelstm_58572_0
while_odelstm_58574_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_odelstm_58562
while_odelstm_58564
while_odelstm_58566
while_odelstm_58568
while_odelstm_58570
while_odelstm_58572
while_odelstm_58574��%while/odelstm/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
%while/odelstm/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_odelstm_58562_0while_odelstm_58564_0while_odelstm_58566_0while_odelstm_58568_0while_odelstm_58570_0while_odelstm_58572_0while_odelstm_58574_0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*)
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_odelstm_layer_call_and_return_conditional_losses_581992'
%while/odelstm/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder.while/odelstm/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0&^while/odelstm/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations&^while/odelstm/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0&^while/odelstm/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^while/odelstm/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity.while/odelstm/StatefulPartitionedCall:output:1&^while/odelstm/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity.while/odelstm/StatefulPartitionedCall:output:2&^while/odelstm/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_odelstm_58562while_odelstm_58562_0",
while_odelstm_58564while_odelstm_58564_0",
while_odelstm_58566while_odelstm_58566_0",
while_odelstm_58568while_odelstm_58568_0",
while_odelstm_58570while_odelstm_58570_0",
while_odelstm_58572while_odelstm_58572_0",
while_odelstm_58574while_odelstm_58574_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*a
_input_shapesP
N: : : : :���������@:���������@: : :::::::2N
%while/odelstm/StatefulPartitionedCall%while/odelstm/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�J
�
__inference__traced_save_61856
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop7
3savev2_rnn_odelstm_input_kernel_read_readvariableop;
7savev2_rnn_odelstm_recurrent_kernel_read_readvariableop/
+savev2_rnn_odelstm_bias_read_readvariableop1
-savev2_rnn_odelstm_kernel_read_readvariableop=
9savev2_rnn_odelstm_recurrent_kernel_1_read_readvariableop1
-savev2_rnn_odelstm_bias_1_read_readvariableop0
,savev2_rnn_odelstm_scale_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop>
:savev2_adam_rnn_odelstm_input_kernel_m_read_readvariableopB
>savev2_adam_rnn_odelstm_recurrent_kernel_m_read_readvariableop6
2savev2_adam_rnn_odelstm_bias_m_read_readvariableop8
4savev2_adam_rnn_odelstm_kernel_m_read_readvariableopD
@savev2_adam_rnn_odelstm_recurrent_kernel_m_1_read_readvariableop8
4savev2_adam_rnn_odelstm_bias_m_1_read_readvariableop7
3savev2_adam_rnn_odelstm_scale_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop>
:savev2_adam_rnn_odelstm_input_kernel_v_read_readvariableopB
>savev2_adam_rnn_odelstm_recurrent_kernel_v_read_readvariableop6
2savev2_adam_rnn_odelstm_bias_v_read_readvariableop8
4savev2_adam_rnn_odelstm_kernel_v_read_readvariableopD
@savev2_adam_rnn_odelstm_recurrent_kernel_v_1_read_readvariableop8
4savev2_adam_rnn_odelstm_bias_v_1_read_readvariableop7
3savev2_adam_rnn_odelstm_scale_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop3savev2_rnn_odelstm_input_kernel_read_readvariableop7savev2_rnn_odelstm_recurrent_kernel_read_readvariableop+savev2_rnn_odelstm_bias_read_readvariableop-savev2_rnn_odelstm_kernel_read_readvariableop9savev2_rnn_odelstm_recurrent_kernel_1_read_readvariableop-savev2_rnn_odelstm_bias_1_read_readvariableop,savev2_rnn_odelstm_scale_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop:savev2_adam_rnn_odelstm_input_kernel_m_read_readvariableop>savev2_adam_rnn_odelstm_recurrent_kernel_m_read_readvariableop2savev2_adam_rnn_odelstm_bias_m_read_readvariableop4savev2_adam_rnn_odelstm_kernel_m_read_readvariableop@savev2_adam_rnn_odelstm_recurrent_kernel_m_1_read_readvariableop4savev2_adam_rnn_odelstm_bias_m_1_read_readvariableop3savev2_adam_rnn_odelstm_scale_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop:savev2_adam_rnn_odelstm_input_kernel_v_read_readvariableop>savev2_adam_rnn_odelstm_recurrent_kernel_v_read_readvariableop2savev2_adam_rnn_odelstm_bias_v_read_readvariableop4savev2_adam_rnn_odelstm_kernel_v_read_readvariableop@savev2_adam_rnn_odelstm_recurrent_kernel_v_1_read_readvariableop4savev2_adam_rnn_odelstm_bias_v_1_read_readvariableop3savev2_adam_rnn_odelstm_scale_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:: : : : : :	�:	@�:�:@@:@@:@:@: : :@::	�:	@�:�:@@:@@:@:@:@::	�:	@�:�:@@:@@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:%	!

_output_shapes
:	@�:!


_output_shapes	
:�:$ 

_output_shapes

:@@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	@�:!

_output_shapes	
:�:$ 

_output_shapes

:@@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	@�:!

_output_shapes	
:�:$ 

_output_shapes

:@@:$  

_output_shapes

:@@: !

_output_shapes
:@: "

_output_shapes
:@:#

_output_shapes
: 
�
�
C__inference_ode_lstm_layer_call_and_return_conditional_losses_59541

inputs
inputs_1
	rnn_59520
	rnn_59522
	rnn_59524
	rnn_59526
	rnn_59528
	rnn_59530
	rnn_59532
dense_59535
dense_59537
identity��dense/StatefulPartitionedCall�rnn/StatefulPartitionedCall�
rnn/StatefulPartitionedCallStatefulPartitionedCallinputs	rnn_59520	rnn_59522	rnn_59524	rnn_59526	rnn_59528	rnn_59530	rnn_59532*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�@*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_591062
rnn/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_59535dense_59537*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_594702
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^rnn/StatefulPartitionedCall*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:�d:�d:::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:K G
#
_output_shapes
:�d
 
_user_specified_nameinputs:KG
#
_output_shapes
:�d
 
_user_specified_nameinputs
��
�
!__inference__traced_restore_61968
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate/
+assignvariableop_7_rnn_odelstm_input_kernel3
/assignvariableop_8_rnn_odelstm_recurrent_kernel'
#assignvariableop_9_rnn_odelstm_bias*
&assignvariableop_10_rnn_odelstm_kernel6
2assignvariableop_11_rnn_odelstm_recurrent_kernel_1*
&assignvariableop_12_rnn_odelstm_bias_1)
%assignvariableop_13_rnn_odelstm_scale
assignvariableop_14_total
assignvariableop_15_count+
'assignvariableop_16_adam_dense_kernel_m)
%assignvariableop_17_adam_dense_bias_m7
3assignvariableop_18_adam_rnn_odelstm_input_kernel_m;
7assignvariableop_19_adam_rnn_odelstm_recurrent_kernel_m/
+assignvariableop_20_adam_rnn_odelstm_bias_m1
-assignvariableop_21_adam_rnn_odelstm_kernel_m=
9assignvariableop_22_adam_rnn_odelstm_recurrent_kernel_m_11
-assignvariableop_23_adam_rnn_odelstm_bias_m_10
,assignvariableop_24_adam_rnn_odelstm_scale_m+
'assignvariableop_25_adam_dense_kernel_v)
%assignvariableop_26_adam_dense_bias_v7
3assignvariableop_27_adam_rnn_odelstm_input_kernel_v;
7assignvariableop_28_adam_rnn_odelstm_recurrent_kernel_v/
+assignvariableop_29_adam_rnn_odelstm_bias_v1
-assignvariableop_30_adam_rnn_odelstm_kernel_v=
9assignvariableop_31_adam_rnn_odelstm_recurrent_kernel_v_11
-assignvariableop_32_adam_rnn_odelstm_bias_v_10
,assignvariableop_33_adam_rnn_odelstm_scale_v
identity_35��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_rnn_odelstm_input_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp/assignvariableop_8_rnn_odelstm_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_rnn_odelstm_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_rnn_odelstm_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp2assignvariableop_11_rnn_odelstm_recurrent_kernel_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_rnn_odelstm_bias_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp%assignvariableop_13_rnn_odelstm_scaleIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp%assignvariableop_17_adam_dense_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp3assignvariableop_18_adam_rnn_odelstm_input_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp7assignvariableop_19_adam_rnn_odelstm_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_rnn_odelstm_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp-assignvariableop_21_adam_rnn_odelstm_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp9assignvariableop_22_adam_rnn_odelstm_recurrent_kernel_m_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp-assignvariableop_23_adam_rnn_odelstm_bias_m_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_rnn_odelstm_scale_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp3assignvariableop_27_adam_rnn_odelstm_input_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp7assignvariableop_28_adam_rnn_odelstm_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_rnn_odelstm_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp-assignvariableop_30_adam_rnn_odelstm_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp9assignvariableop_31_adam_rnn_odelstm_recurrent_kernel_v_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp-assignvariableop_32_adam_rnn_odelstm_bias_v_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_rnn_odelstm_scale_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_339
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_34�
Identity_35IdentityIdentity_34:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_35"#
identity_35Identity_35:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�i
�
B__inference_odelstm_layer_call_and_return_conditional_losses_61705

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_1_readvariableop_resource$
 matmul_2_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_5_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�MatMul_2/ReadVariableOp�MatMul_3/ReadVariableOp�MatMul_4/ReadVariableOp�MatMul_5/ReadVariableOp�MatMul_6/ReadVariableOp�MatMul_7/ReadVariableOp�MatMul_8/ReadVariableOp�MatMul_9/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�add_1/ReadVariableOp�add_11/ReadVariableOp�add_14/ReadVariableOp�add_5/ReadVariableOp�add_8/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_1MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
add_1/ReadVariableOpq
add_1AddV2add:z:0add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
add_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0	add_1:z:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
splitV
TanhTanhsplit:output:0*
T0*'
_output_shapes
:���������@2
Tanh_
SigmoidSigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2	
SigmoidW
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2	
add_2/yk
add_2AddV2split:output:2add_2/y:output:0*
T0*'
_output_shapes
:���������@2
add_2^
	Sigmoid_1Sigmoid	add_2:z:0*
T0*'
_output_shapes
:���������@2
	Sigmoid_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2\
mulMulstates_0Sigmoid_1:y:0*
T0*'
_output_shapes
:���������@2
mul^
mul_1MulTanh:y:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_3AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_3U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:���������@2
Tanh_1b
mul_2Mul
Tanh_1:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������@2
mul_2�
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_2/ReadVariableOp|
MatMul_2MatMul	mul_2:z:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_2�
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_3/ReadVariableOp{
MatMul_3MatMulstates_1MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_3q
add_4AddV2MatMul_2:product:0MatMul_3:product:0*
T0*'
_output_shapes
:���������@2
add_4�
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:@*
dtype02
add_5/ReadVariableOpr
add_5AddV2	add_4:z:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
add_5U
Tanh_2Tanh	add_5:z:0*
T0*'
_output_shapes
:���������@2
Tanh_2t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpk
mul_3MulReadVariableOp:value:0
Tanh_2:y:0*
T0*'
_output_shapes
:���������@2
mul_3W
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
mul_4/yc
mul_4Mulstates_1mul_4/y:output:0*
T0*'
_output_shapes
:���������@2
mul_4Y
subSub	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:���������@2
subW
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2	
mul_5/xb
mul_5Mulmul_5/x:output:0sub:z:0*
T0*'
_output_shapes
:���������@2
mul_5^
add_6AddV2states_1	mul_5:z:0*
T0*'
_output_shapes
:���������@2
add_6�
MatMul_4/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_4/ReadVariableOp|
MatMul_4MatMul	mul_2:z:0MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_4�
MatMul_5/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_5/ReadVariableOp|
MatMul_5MatMul	add_6:z:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_5q
add_7AddV2MatMul_4:product:0MatMul_5:product:0*
T0*'
_output_shapes
:���������@2
add_7�
add_8/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:@*
dtype02
add_8/ReadVariableOpr
add_8AddV2	add_7:z:0add_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
add_8U
Tanh_3Tanh	add_8:z:0*
T0*'
_output_shapes
:���������@2
Tanh_3x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1m
mul_6MulReadVariableOp_1:value:0
Tanh_3:y:0*
T0*'
_output_shapes
:���������@2
mul_6W
mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
mul_7/yd
mul_7Mul	add_6:z:0mul_7/y:output:0*
T0*'
_output_shapes
:���������@2
mul_7]
sub_1Sub	mul_6:z:0	mul_7:z:0*
T0*'
_output_shapes
:���������@2
sub_1W
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2	
mul_8/xd
mul_8Mulmul_8/x:output:0	sub_1:z:0*
T0*'
_output_shapes
:���������@2
mul_8_
add_9AddV2	add_6:z:0	mul_8:z:0*
T0*'
_output_shapes
:���������@2
add_9�
MatMul_6/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_6/ReadVariableOp|
MatMul_6MatMul	mul_2:z:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_6�
MatMul_7/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_7/ReadVariableOp|
MatMul_7MatMul	add_9:z:0MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_7s
add_10AddV2MatMul_6:product:0MatMul_7:product:0*
T0*'
_output_shapes
:���������@2
add_10�
add_11/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:@*
dtype02
add_11/ReadVariableOpv
add_11AddV2
add_10:z:0add_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
add_11V
Tanh_4Tanh
add_11:z:0*
T0*'
_output_shapes
:���������@2
Tanh_4x
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_2m
mul_9MulReadVariableOp_2:value:0
Tanh_4:y:0*
T0*'
_output_shapes
:���������@2
mul_9Y
mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

mul_10/yg
mul_10Mul	add_9:z:0mul_10/y:output:0*
T0*'
_output_shapes
:���������@2
mul_10^
sub_2Sub	mul_9:z:0
mul_10:z:0*
T0*'
_output_shapes
:���������@2
sub_2Y
mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2

mul_11/xg
mul_11Mulmul_11/x:output:0	sub_2:z:0*
T0*'
_output_shapes
:���������@2
mul_11b
add_12AddV2	add_9:z:0
mul_11:z:0*
T0*'
_output_shapes
:���������@2
add_12�
MatMul_8/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_8/ReadVariableOp|
MatMul_8MatMul	mul_2:z:0MatMul_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_8�
MatMul_9/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul_9/ReadVariableOp}
MatMul_9MatMul
add_12:z:0MatMul_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_9s
add_13AddV2MatMul_8:product:0MatMul_9:product:0*
T0*'
_output_shapes
:���������@2
add_13�
add_14/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:@*
dtype02
add_14/ReadVariableOpv
add_14AddV2
add_13:z:0add_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
add_14V
Tanh_5Tanh
add_14:z:0*
T0*'
_output_shapes
:���������@2
Tanh_5x
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_3o
mul_12MulReadVariableOp_3:value:0
Tanh_5:y:0*
T0*'
_output_shapes
:���������@2
mul_12Y
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

mul_13/yh
mul_13Mul
add_12:z:0mul_13/y:output:0*
T0*'
_output_shapes
:���������@2
mul_13_
sub_3Sub
mul_12:z:0
mul_13:z:0*
T0*'
_output_shapes
:���������@2
sub_3Y
mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2

mul_14/xg
mul_14Mulmul_14/x:output:0	sub_3:z:0*
T0*'
_output_shapes
:���������@2
mul_14c
add_15AddV2
add_12:z:0
mul_14:z:0*
T0*'
_output_shapes
:���������@2
add_15�
IdentityIdentity
add_15:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^MatMul_8/ReadVariableOp^MatMul_9/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^add_1/ReadVariableOp^add_11/ReadVariableOp^add_14/ReadVariableOp^add_5/ReadVariableOp^add_8/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^MatMul_8/ReadVariableOp^MatMul_9/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^add_1/ReadVariableOp^add_11/ReadVariableOp^add_14/ReadVariableOp^add_5/ReadVariableOp^add_8/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_1�

Identity_2Identity
add_15:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^MatMul_8/ReadVariableOp^MatMul_9/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^add_1/ReadVariableOp^add_11/ReadVariableOp^add_14/ReadVariableOp^add_5/ReadVariableOp^add_8/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*h
_input_shapesW
U:���������:���������@:���������@:::::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp22
MatMul_7/ReadVariableOpMatMul_7/ReadVariableOp22
MatMul_8/ReadVariableOpMatMul_8/ReadVariableOp22
MatMul_9/ReadVariableOpMatMul_9/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
add_1/ReadVariableOpadd_1/ReadVariableOp2.
add_11/ReadVariableOpadd_11/ReadVariableOp2.
add_14/ReadVariableOpadd_14/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_8/ReadVariableOpadd_8/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�

�
while_cond_58537
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_58537___redundant_placeholder03
/while_while_cond_58537___redundant_placeholder13
/while_while_cond_58537___redundant_placeholder23
/while_while_cond_58537___redundant_placeholder33
/while_while_cond_58537___redundant_placeholder43
/while_while_cond_58537___redundant_placeholder53
/while_while_cond_58537___redundant_placeholder63
/while_while_cond_58537___redundant_placeholder7
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*c
_input_shapesR
P: : : : :���������@:���������@: ::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�	
�
while_cond_61082
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_61082___redundant_placeholder03
/while_while_cond_61082___redundant_placeholder13
/while_while_cond_61082___redundant_placeholder23
/while_while_cond_61082___redundant_placeholder33
/while_while_cond_61082___redundant_placeholder43
/while_while_cond_61082___redundant_placeholder53
/while_while_cond_61082___redundant_placeholder63
/while_while_cond_61082___redundant_placeholder7
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :	�@:	�@: ::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :

_output_shapes
:
�
�
#__inference_rnn_layer_call_fn_61564

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�@*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_591062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	�@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:�d:::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:�d
 
_user_specified_nameinputs
�	
�
(__inference_ode_lstm_layer_call_fn_59611
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_ode_lstm_layer_call_and_return_conditional_losses_595902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:�d:�d:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:�d
!
_user_specified_name	input_1:LH
#
_output_shapes
:�d
!
_user_specified_name	input_2
�
�
@__inference_dense_layer_call_and_return_conditional_losses_61593

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*&
_input_shapes
:	�@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	�@
 
_user_specified_nameinputs
�
�
C__inference_ode_lstm_layer_call_and_return_conditional_losses_59487
input_1
input_2
	rnn_59445
	rnn_59447
	rnn_59449
	rnn_59451
	rnn_59453
	rnn_59455
	rnn_59457
dense_59481
dense_59483
identity��dense/StatefulPartitionedCall�rnn/StatefulPartitionedCall�
rnn/StatefulPartitionedCallStatefulPartitionedCallinput_1	rnn_59445	rnn_59447	rnn_59449	rnn_59451	rnn_59453	rnn_59455	rnn_59457*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�@*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_591062
rnn/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_59481dense_59483*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_594702
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^rnn/StatefulPartitionedCall*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:�d:�d:::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:L H
#
_output_shapes
:�d
!
_user_specified_name	input_1:LH
#
_output_shapes
:�d
!
_user_specified_name	input_2
�
�
#__inference_rnn_layer_call_fn_60926
inputs_0
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_586222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:������������������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�E
�
>__inference_rnn_layer_call_and_return_conditional_losses_58622

inputs
odelstm_58517
odelstm_58519
odelstm_58521
odelstm_58523
odelstm_58525
odelstm_58527
odelstm_58529
identity��odelstm/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
odelstm/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0odelstm_58517odelstm_58519odelstm_58521odelstm_58523odelstm_58525odelstm_58527odelstm_58529*
Tin
2
*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*)
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_odelstm_layer_call_and_return_conditional_losses_581992!
odelstm/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0odelstm_58517odelstm_58519odelstm_58521odelstm_58523odelstm_58525odelstm_58527odelstm_58529*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*T
_output_shapesB
@: : : : :���������@:���������@: : : : : : : : : *)
_read_only_resource_inputs
		
*
bodyR
while_body_58538*
condR
while_cond_58537*S
output_shapesB
@: : : : :���������@:���������@: : : : : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1�
IdentityIdentitystrided_slice_3:output:0 ^odelstm/StatefulPartitionedCall^while*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:������������������:::::::2B
odelstm/StatefulPartitionedCallodelstm/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
ode_lstm_rnn_while_cond_578876
2ode_lstm_rnn_while_ode_lstm_rnn_while_loop_counter<
8ode_lstm_rnn_while_ode_lstm_rnn_while_maximum_iterations"
ode_lstm_rnn_while_placeholder$
 ode_lstm_rnn_while_placeholder_1$
 ode_lstm_rnn_while_placeholder_2$
 ode_lstm_rnn_while_placeholder_38
4ode_lstm_rnn_while_less_ode_lstm_rnn_strided_slice_1M
Iode_lstm_rnn_while_ode_lstm_rnn_while_cond_57887___redundant_placeholder0M
Iode_lstm_rnn_while_ode_lstm_rnn_while_cond_57887___redundant_placeholder1M
Iode_lstm_rnn_while_ode_lstm_rnn_while_cond_57887___redundant_placeholder2M
Iode_lstm_rnn_while_ode_lstm_rnn_while_cond_57887___redundant_placeholder3M
Iode_lstm_rnn_while_ode_lstm_rnn_while_cond_57887___redundant_placeholder4M
Iode_lstm_rnn_while_ode_lstm_rnn_while_cond_57887___redundant_placeholder5M
Iode_lstm_rnn_while_ode_lstm_rnn_while_cond_57887___redundant_placeholder6M
Iode_lstm_rnn_while_ode_lstm_rnn_while_cond_57887___redundant_placeholder7
ode_lstm_rnn_while_identity
�
ode_lstm/rnn/while/LessLessode_lstm_rnn_while_placeholder4ode_lstm_rnn_while_less_ode_lstm_rnn_strided_slice_1*
T0*
_output_shapes
: 2
ode_lstm/rnn/while/Less�
ode_lstm/rnn/while/IdentityIdentityode_lstm/rnn/while/Less:z:0*
T0
*
_output_shapes
: 2
ode_lstm/rnn/while/Identity"C
ode_lstm_rnn_while_identity$ode_lstm/rnn/while/Identity:output:0*S
_input_shapesB
@: : : : :	�@:	�@: ::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :

_output_shapes
:
��
�
while_body_60445
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
.while_odelstm_matmul_readvariableop_resource_04
0while_odelstm_matmul_1_readvariableop_resource_01
-while_odelstm_add_1_readvariableop_resource_04
0while_odelstm_matmul_2_readvariableop_resource_04
0while_odelstm_matmul_3_readvariableop_resource_01
-while_odelstm_add_5_readvariableop_resource_0+
'while_odelstm_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
,while_odelstm_matmul_readvariableop_resource2
.while_odelstm_matmul_1_readvariableop_resource/
+while_odelstm_add_1_readvariableop_resource2
.while_odelstm_matmul_2_readvariableop_resource2
.while_odelstm_matmul_3_readvariableop_resource/
+while_odelstm_add_5_readvariableop_resource)
%while_odelstm_readvariableop_resource��#while/odelstm/MatMul/ReadVariableOp�%while/odelstm/MatMul_1/ReadVariableOp�%while/odelstm/MatMul_2/ReadVariableOp�%while/odelstm/MatMul_3/ReadVariableOp�%while/odelstm/MatMul_4/ReadVariableOp�%while/odelstm/MatMul_5/ReadVariableOp�%while/odelstm/MatMul_6/ReadVariableOp�%while/odelstm/MatMul_7/ReadVariableOp�%while/odelstm/MatMul_8/ReadVariableOp�%while/odelstm/MatMul_9/ReadVariableOp�while/odelstm/ReadVariableOp�while/odelstm/ReadVariableOp_1�while/odelstm/ReadVariableOp_2�while/odelstm/ReadVariableOp_3�"while/odelstm/add_1/ReadVariableOp�#while/odelstm/add_11/ReadVariableOp�#while/odelstm/add_14/ReadVariableOp�"while/odelstm/add_5/ReadVariableOp�"while/odelstm/add_8/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
#while/odelstm/MatMul/ReadVariableOpReadVariableOp.while_odelstm_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02%
#while/odelstm/MatMul/ReadVariableOp�
while/odelstm/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/odelstm/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/odelstm/MatMul�
%while/odelstm/MatMul_1/ReadVariableOpReadVariableOp0while_odelstm_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02'
%while/odelstm/MatMul_1/ReadVariableOp�
while/odelstm/MatMul_1MatMulwhile_placeholder_3-while/odelstm/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/odelstm/MatMul_1�
while/odelstm/addAddV2while/odelstm/MatMul:product:0 while/odelstm/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/odelstm/add�
"while/odelstm/add_1/ReadVariableOpReadVariableOp-while_odelstm_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype02$
"while/odelstm/add_1/ReadVariableOp�
while/odelstm/add_1AddV2while/odelstm/add:z:0*while/odelstm/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/odelstm/add_1l
while/odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/odelstm/Const�
while/odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/odelstm/split/split_dim�
while/odelstm/splitSplit&while/odelstm/split/split_dim:output:0while/odelstm/add_1:z:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/odelstm/split�
while/odelstm/TanhTanhwhile/odelstm/split:output:0*
T0*'
_output_shapes
:���������@2
while/odelstm/Tanh�
while/odelstm/SigmoidSigmoidwhile/odelstm/split:output:1*
T0*'
_output_shapes
:���������@2
while/odelstm/Sigmoids
while/odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
while/odelstm/add_2/y�
while/odelstm/add_2AddV2while/odelstm/split:output:2while/odelstm/add_2/y:output:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_2�
while/odelstm/Sigmoid_1Sigmoidwhile/odelstm/add_2:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/Sigmoid_1�
while/odelstm/Sigmoid_2Sigmoidwhile/odelstm/split:output:3*
T0*'
_output_shapes
:���������@2
while/odelstm/Sigmoid_2�
while/odelstm/mulMulwhile_placeholder_2while/odelstm/Sigmoid_1:y:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul�
while/odelstm/mul_1Mulwhile/odelstm/Tanh:y:0while/odelstm/Sigmoid:y:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_1�
while/odelstm/add_3AddV2while/odelstm/mul:z:0while/odelstm/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_3
while/odelstm/Tanh_1Tanhwhile/odelstm/add_3:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/Tanh_1�
while/odelstm/mul_2Mulwhile/odelstm/Tanh_1:y:0while/odelstm/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_2�
%while/odelstm/MatMul_2/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_2/ReadVariableOp�
while/odelstm/MatMul_2MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_2�
%while/odelstm/MatMul_3/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_3/ReadVariableOp�
while/odelstm/MatMul_3MatMulwhile_placeholder_3-while/odelstm/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_3�
while/odelstm/add_4AddV2 while/odelstm/MatMul_2:product:0 while/odelstm/MatMul_3:product:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_4�
"while/odelstm/add_5/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"while/odelstm/add_5/ReadVariableOp�
while/odelstm/add_5AddV2while/odelstm/add_4:z:0*while/odelstm/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_5
while/odelstm/Tanh_2Tanhwhile/odelstm/add_5:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/Tanh_2�
while/odelstm/ReadVariableOpReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02
while/odelstm/ReadVariableOp�
while/odelstm/mul_3Mul$while/odelstm/ReadVariableOp:value:0while/odelstm/Tanh_2:y:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_3s
while/odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_4/y�
while/odelstm/mul_4Mulwhile_placeholder_3while/odelstm/mul_4/y:output:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_4�
while/odelstm/subSubwhile/odelstm/mul_3:z:0while/odelstm/mul_4:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/subs
while/odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_5/x�
while/odelstm/mul_5Mulwhile/odelstm/mul_5/x:output:0while/odelstm/sub:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_5�
while/odelstm/add_6AddV2while_placeholder_3while/odelstm/mul_5:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_6�
%while/odelstm/MatMul_4/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_4/ReadVariableOp�
while/odelstm/MatMul_4MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_4�
%while/odelstm/MatMul_5/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_5/ReadVariableOp�
while/odelstm/MatMul_5MatMulwhile/odelstm/add_6:z:0-while/odelstm/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_5�
while/odelstm/add_7AddV2 while/odelstm/MatMul_4:product:0 while/odelstm/MatMul_5:product:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_7�
"while/odelstm/add_8/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"while/odelstm/add_8/ReadVariableOp�
while/odelstm/add_8AddV2while/odelstm/add_7:z:0*while/odelstm/add_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_8
while/odelstm/Tanh_3Tanhwhile/odelstm/add_8:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/Tanh_3�
while/odelstm/ReadVariableOp_1ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_1�
while/odelstm/mul_6Mul&while/odelstm/ReadVariableOp_1:value:0while/odelstm/Tanh_3:y:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_6s
while/odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_7/y�
while/odelstm/mul_7Mulwhile/odelstm/add_6:z:0while/odelstm/mul_7/y:output:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_7�
while/odelstm/sub_1Subwhile/odelstm/mul_6:z:0while/odelstm/mul_7:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/sub_1s
while/odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_8/x�
while/odelstm/mul_8Mulwhile/odelstm/mul_8/x:output:0while/odelstm/sub_1:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_8�
while/odelstm/add_9AddV2while/odelstm/add_6:z:0while/odelstm/mul_8:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_9�
%while/odelstm/MatMul_6/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_6/ReadVariableOp�
while/odelstm/MatMul_6MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_6�
%while/odelstm/MatMul_7/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_7/ReadVariableOp�
while/odelstm/MatMul_7MatMulwhile/odelstm/add_9:z:0-while/odelstm/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_7�
while/odelstm/add_10AddV2 while/odelstm/MatMul_6:product:0 while/odelstm/MatMul_7:product:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_10�
#while/odelstm/add_11/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02%
#while/odelstm/add_11/ReadVariableOp�
while/odelstm/add_11AddV2while/odelstm/add_10:z:0+while/odelstm/add_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_11�
while/odelstm/Tanh_4Tanhwhile/odelstm/add_11:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/Tanh_4�
while/odelstm/ReadVariableOp_2ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_2�
while/odelstm/mul_9Mul&while/odelstm/ReadVariableOp_2:value:0while/odelstm/Tanh_4:y:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_9u
while/odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_10/y�
while/odelstm/mul_10Mulwhile/odelstm/add_9:z:0while/odelstm/mul_10/y:output:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_10�
while/odelstm/sub_2Subwhile/odelstm/mul_9:z:0while/odelstm/mul_10:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/sub_2u
while/odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_11/x�
while/odelstm/mul_11Mulwhile/odelstm/mul_11/x:output:0while/odelstm/sub_2:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_11�
while/odelstm/add_12AddV2while/odelstm/add_9:z:0while/odelstm/mul_11:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_12�
%while/odelstm/MatMul_8/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_8/ReadVariableOp�
while/odelstm/MatMul_8MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_8�
%while/odelstm/MatMul_9/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_9/ReadVariableOp�
while/odelstm/MatMul_9MatMulwhile/odelstm/add_12:z:0-while/odelstm/MatMul_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_9�
while/odelstm/add_13AddV2 while/odelstm/MatMul_8:product:0 while/odelstm/MatMul_9:product:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_13�
#while/odelstm/add_14/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02%
#while/odelstm/add_14/ReadVariableOp�
while/odelstm/add_14AddV2while/odelstm/add_13:z:0+while/odelstm/add_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_14�
while/odelstm/Tanh_5Tanhwhile/odelstm/add_14:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/Tanh_5�
while/odelstm/ReadVariableOp_3ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_3�
while/odelstm/mul_12Mul&while/odelstm/ReadVariableOp_3:value:0while/odelstm/Tanh_5:y:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_12u
while/odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_13/y�
while/odelstm/mul_13Mulwhile/odelstm/add_12:z:0while/odelstm/mul_13/y:output:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_13�
while/odelstm/sub_3Subwhile/odelstm/mul_12:z:0while/odelstm/mul_13:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/sub_3u
while/odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_14/x�
while/odelstm/mul_14Mulwhile/odelstm/mul_14/x:output:0while/odelstm/sub_3:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_14�
while/odelstm/add_15AddV2while/odelstm/add_12:z:0while/odelstm/mul_14:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_15�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/odelstm/add_15:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/odelstm/add_3:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/odelstm/add_15:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"\
+while_odelstm_add_1_readvariableop_resource-while_odelstm_add_1_readvariableop_resource_0"\
+while_odelstm_add_5_readvariableop_resource-while_odelstm_add_5_readvariableop_resource_0"b
.while_odelstm_matmul_1_readvariableop_resource0while_odelstm_matmul_1_readvariableop_resource_0"b
.while_odelstm_matmul_2_readvariableop_resource0while_odelstm_matmul_2_readvariableop_resource_0"b
.while_odelstm_matmul_3_readvariableop_resource0while_odelstm_matmul_3_readvariableop_resource_0"^
,while_odelstm_matmul_readvariableop_resource.while_odelstm_matmul_readvariableop_resource_0"P
%while_odelstm_readvariableop_resource'while_odelstm_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*a
_input_shapesP
N: : : : :���������@:���������@: : :::::::2J
#while/odelstm/MatMul/ReadVariableOp#while/odelstm/MatMul/ReadVariableOp2N
%while/odelstm/MatMul_1/ReadVariableOp%while/odelstm/MatMul_1/ReadVariableOp2N
%while/odelstm/MatMul_2/ReadVariableOp%while/odelstm/MatMul_2/ReadVariableOp2N
%while/odelstm/MatMul_3/ReadVariableOp%while/odelstm/MatMul_3/ReadVariableOp2N
%while/odelstm/MatMul_4/ReadVariableOp%while/odelstm/MatMul_4/ReadVariableOp2N
%while/odelstm/MatMul_5/ReadVariableOp%while/odelstm/MatMul_5/ReadVariableOp2N
%while/odelstm/MatMul_6/ReadVariableOp%while/odelstm/MatMul_6/ReadVariableOp2N
%while/odelstm/MatMul_7/ReadVariableOp%while/odelstm/MatMul_7/ReadVariableOp2N
%while/odelstm/MatMul_8/ReadVariableOp%while/odelstm/MatMul_8/ReadVariableOp2N
%while/odelstm/MatMul_9/ReadVariableOp%while/odelstm/MatMul_9/ReadVariableOp2<
while/odelstm/ReadVariableOpwhile/odelstm/ReadVariableOp2@
while/odelstm/ReadVariableOp_1while/odelstm/ReadVariableOp_12@
while/odelstm/ReadVariableOp_2while/odelstm/ReadVariableOp_22@
while/odelstm/ReadVariableOp_3while/odelstm/ReadVariableOp_32H
"while/odelstm/add_1/ReadVariableOp"while/odelstm/add_1/ReadVariableOp2J
#while/odelstm/add_11/ReadVariableOp#while/odelstm/add_11/ReadVariableOp2J
#while/odelstm/add_14/ReadVariableOp#while/odelstm/add_14/ReadVariableOp2H
"while/odelstm/add_5/ReadVariableOp"while/odelstm/add_5/ReadVariableOp2H
"while/odelstm/add_8/ReadVariableOp"while/odelstm/add_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�	
�
(__inference_ode_lstm_layer_call_fn_59562
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_ode_lstm_layer_call_and_return_conditional_losses_595412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:�d:�d:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:�d
!
_user_specified_name	input_1:LH
#
_output_shapes
:�d
!
_user_specified_name	input_2
�
�
#__inference_rnn_layer_call_fn_60945
inputs_0
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_587852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:������������������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
rnn_while_cond_59783$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_59783___redundant_placeholder0;
7rnn_while_rnn_while_cond_59783___redundant_placeholder1;
7rnn_while_rnn_while_cond_59783___redundant_placeholder2;
7rnn_while_rnn_while_cond_59783___redundant_placeholder3;
7rnn_while_rnn_while_cond_59783___redundant_placeholder4;
7rnn_while_rnn_while_cond_59783___redundant_placeholder5;
7rnn_while_rnn_while_cond_59783___redundant_placeholder6;
7rnn_while_rnn_while_cond_59783___redundant_placeholder7
rnn_while_identity
�
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: 2
rnn/while/Lessi
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: 2
rnn/while/Identity"1
rnn_while_identityrnn/while/Identity:output:0*S
_input_shapesB
@: : : : :	�@:	�@: ::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :

_output_shapes
:
��
�
rnn_while_body_60091$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_06
2rnn_while_odelstm_matmul_readvariableop_resource_08
4rnn_while_odelstm_matmul_1_readvariableop_resource_05
1rnn_while_odelstm_add_1_readvariableop_resource_08
4rnn_while_odelstm_matmul_2_readvariableop_resource_08
4rnn_while_odelstm_matmul_3_readvariableop_resource_05
1rnn_while_odelstm_add_5_readvariableop_resource_0/
+rnn_while_odelstm_readvariableop_resource_0
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor4
0rnn_while_odelstm_matmul_readvariableop_resource6
2rnn_while_odelstm_matmul_1_readvariableop_resource3
/rnn_while_odelstm_add_1_readvariableop_resource6
2rnn_while_odelstm_matmul_2_readvariableop_resource6
2rnn_while_odelstm_matmul_3_readvariableop_resource3
/rnn_while_odelstm_add_5_readvariableop_resource-
)rnn_while_odelstm_readvariableop_resource��'rnn/while/odelstm/MatMul/ReadVariableOp�)rnn/while/odelstm/MatMul_1/ReadVariableOp�)rnn/while/odelstm/MatMul_2/ReadVariableOp�)rnn/while/odelstm/MatMul_3/ReadVariableOp�)rnn/while/odelstm/MatMul_4/ReadVariableOp�)rnn/while/odelstm/MatMul_5/ReadVariableOp�)rnn/while/odelstm/MatMul_6/ReadVariableOp�)rnn/while/odelstm/MatMul_7/ReadVariableOp�)rnn/while/odelstm/MatMul_8/ReadVariableOp�)rnn/while/odelstm/MatMul_9/ReadVariableOp� rnn/while/odelstm/ReadVariableOp�"rnn/while/odelstm/ReadVariableOp_1�"rnn/while/odelstm/ReadVariableOp_2�"rnn/while/odelstm/ReadVariableOp_3�&rnn/while/odelstm/add_1/ReadVariableOp�'rnn/while/odelstm/add_11/ReadVariableOp�'rnn/while/odelstm/add_14/ReadVariableOp�&rnn/while/odelstm/add_5/ReadVariableOp�&rnn/while/odelstm/add_8/ReadVariableOp�
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�      2=
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape�
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	�*
element_dtype02/
-rnn/while/TensorArrayV2Read/TensorListGetItem�
'rnn/while/odelstm/MatMul/ReadVariableOpReadVariableOp2rnn_while_odelstm_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02)
'rnn/while/odelstm/MatMul/ReadVariableOp�
rnn/while/odelstm/MatMulMatMul4rnn/while/TensorArrayV2Read/TensorListGetItem:item:0/rnn/while/odelstm/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
rnn/while/odelstm/MatMul�
)rnn/while/odelstm/MatMul_1/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02+
)rnn/while/odelstm/MatMul_1/ReadVariableOp�
rnn/while/odelstm/MatMul_1MatMulrnn_while_placeholder_31rnn/while/odelstm/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
rnn/while/odelstm/MatMul_1�
rnn/while/odelstm/addAddV2"rnn/while/odelstm/MatMul:product:0$rnn/while/odelstm/MatMul_1:product:0*
T0* 
_output_shapes
:
��2
rnn/while/odelstm/add�
&rnn/while/odelstm/add_1/ReadVariableOpReadVariableOp1rnn_while_odelstm_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype02(
&rnn/while/odelstm/add_1/ReadVariableOp�
rnn/while/odelstm/add_1AddV2rnn/while/odelstm/add:z:0.rnn/while/odelstm/add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2
rnn/while/odelstm/add_1t
rnn/while/odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/odelstm/Const�
!rnn/while/odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!rnn/while/odelstm/split/split_dim�
rnn/while/odelstm/splitSplit*rnn/while/odelstm/split/split_dim:output:0rnn/while/odelstm/add_1:z:0*
T0*@
_output_shapes.
,:	�@:	�@:	�@:	�@*
	num_split2
rnn/while/odelstm/split�
rnn/while/odelstm/TanhTanh rnn/while/odelstm/split:output:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Tanh�
rnn/while/odelstm/SigmoidSigmoid rnn/while/odelstm/split:output:1*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Sigmoid{
rnn/while/odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
rnn/while/odelstm/add_2/y�
rnn/while/odelstm/add_2AddV2 rnn/while/odelstm/split:output:2"rnn/while/odelstm/add_2/y:output:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_2�
rnn/while/odelstm/Sigmoid_1Sigmoidrnn/while/odelstm/add_2:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Sigmoid_1�
rnn/while/odelstm/Sigmoid_2Sigmoid rnn/while/odelstm/split:output:3*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Sigmoid_2�
rnn/while/odelstm/mulMulrnn_while_placeholder_2rnn/while/odelstm/Sigmoid_1:y:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul�
rnn/while/odelstm/mul_1Mulrnn/while/odelstm/Tanh:y:0rnn/while/odelstm/Sigmoid:y:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_1�
rnn/while/odelstm/add_3AddV2rnn/while/odelstm/mul:z:0rnn/while/odelstm/mul_1:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_3�
rnn/while/odelstm/Tanh_1Tanhrnn/while/odelstm/add_3:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Tanh_1�
rnn/while/odelstm/mul_2Mulrnn/while/odelstm/Tanh_1:y:0rnn/while/odelstm/Sigmoid_2:y:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_2�
)rnn/while/odelstm/MatMul_2/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_2/ReadVariableOp�
rnn/while/odelstm/MatMul_2MatMulrnn/while/odelstm/mul_2:z:01rnn/while/odelstm/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_2�
)rnn/while/odelstm/MatMul_3/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_3/ReadVariableOp�
rnn/while/odelstm/MatMul_3MatMulrnn_while_placeholder_31rnn/while/odelstm/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_3�
rnn/while/odelstm/add_4AddV2$rnn/while/odelstm/MatMul_2:product:0$rnn/while/odelstm/MatMul_3:product:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_4�
&rnn/while/odelstm/add_5/ReadVariableOpReadVariableOp1rnn_while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&rnn/while/odelstm/add_5/ReadVariableOp�
rnn/while/odelstm/add_5AddV2rnn/while/odelstm/add_4:z:0.rnn/while/odelstm/add_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_5�
rnn/while/odelstm/Tanh_2Tanhrnn/while/odelstm/add_5:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Tanh_2�
 rnn/while/odelstm/ReadVariableOpReadVariableOp+rnn_while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 rnn/while/odelstm/ReadVariableOp�
rnn/while/odelstm/mul_3Mul(rnn/while/odelstm/ReadVariableOp:value:0rnn/while/odelstm/Tanh_2:y:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_3{
rnn/while/odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/while/odelstm/mul_4/y�
rnn/while/odelstm/mul_4Mulrnn_while_placeholder_3"rnn/while/odelstm/mul_4/y:output:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_4�
rnn/while/odelstm/subSubrnn/while/odelstm/mul_3:z:0rnn/while/odelstm/mul_4:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/sub{
rnn/while/odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/while/odelstm/mul_5/x�
rnn/while/odelstm/mul_5Mul"rnn/while/odelstm/mul_5/x:output:0rnn/while/odelstm/sub:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_5�
rnn/while/odelstm/add_6AddV2rnn_while_placeholder_3rnn/while/odelstm/mul_5:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_6�
)rnn/while/odelstm/MatMul_4/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_4/ReadVariableOp�
rnn/while/odelstm/MatMul_4MatMulrnn/while/odelstm/mul_2:z:01rnn/while/odelstm/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_4�
)rnn/while/odelstm/MatMul_5/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_5/ReadVariableOp�
rnn/while/odelstm/MatMul_5MatMulrnn/while/odelstm/add_6:z:01rnn/while/odelstm/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_5�
rnn/while/odelstm/add_7AddV2$rnn/while/odelstm/MatMul_4:product:0$rnn/while/odelstm/MatMul_5:product:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_7�
&rnn/while/odelstm/add_8/ReadVariableOpReadVariableOp1rnn_while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02(
&rnn/while/odelstm/add_8/ReadVariableOp�
rnn/while/odelstm/add_8AddV2rnn/while/odelstm/add_7:z:0.rnn/while/odelstm/add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_8�
rnn/while/odelstm/Tanh_3Tanhrnn/while/odelstm/add_8:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Tanh_3�
"rnn/while/odelstm/ReadVariableOp_1ReadVariableOp+rnn_while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"rnn/while/odelstm/ReadVariableOp_1�
rnn/while/odelstm/mul_6Mul*rnn/while/odelstm/ReadVariableOp_1:value:0rnn/while/odelstm/Tanh_3:y:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_6{
rnn/while/odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/while/odelstm/mul_7/y�
rnn/while/odelstm/mul_7Mulrnn/while/odelstm/add_6:z:0"rnn/while/odelstm/mul_7/y:output:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_7�
rnn/while/odelstm/sub_1Subrnn/while/odelstm/mul_6:z:0rnn/while/odelstm/mul_7:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/sub_1{
rnn/while/odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/while/odelstm/mul_8/x�
rnn/while/odelstm/mul_8Mul"rnn/while/odelstm/mul_8/x:output:0rnn/while/odelstm/sub_1:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_8�
rnn/while/odelstm/add_9AddV2rnn/while/odelstm/add_6:z:0rnn/while/odelstm/mul_8:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_9�
)rnn/while/odelstm/MatMul_6/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_6/ReadVariableOp�
rnn/while/odelstm/MatMul_6MatMulrnn/while/odelstm/mul_2:z:01rnn/while/odelstm/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_6�
)rnn/while/odelstm/MatMul_7/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_7/ReadVariableOp�
rnn/while/odelstm/MatMul_7MatMulrnn/while/odelstm/add_9:z:01rnn/while/odelstm/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_7�
rnn/while/odelstm/add_10AddV2$rnn/while/odelstm/MatMul_6:product:0$rnn/while/odelstm/MatMul_7:product:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_10�
'rnn/while/odelstm/add_11/ReadVariableOpReadVariableOp1rnn_while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02)
'rnn/while/odelstm/add_11/ReadVariableOp�
rnn/while/odelstm/add_11AddV2rnn/while/odelstm/add_10:z:0/rnn/while/odelstm/add_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_11�
rnn/while/odelstm/Tanh_4Tanhrnn/while/odelstm/add_11:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Tanh_4�
"rnn/while/odelstm/ReadVariableOp_2ReadVariableOp+rnn_while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"rnn/while/odelstm/ReadVariableOp_2�
rnn/while/odelstm/mul_9Mul*rnn/while/odelstm/ReadVariableOp_2:value:0rnn/while/odelstm/Tanh_4:y:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_9}
rnn/while/odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/while/odelstm/mul_10/y�
rnn/while/odelstm/mul_10Mulrnn/while/odelstm/add_9:z:0#rnn/while/odelstm/mul_10/y:output:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_10�
rnn/while/odelstm/sub_2Subrnn/while/odelstm/mul_9:z:0rnn/while/odelstm/mul_10:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/sub_2}
rnn/while/odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/while/odelstm/mul_11/x�
rnn/while/odelstm/mul_11Mul#rnn/while/odelstm/mul_11/x:output:0rnn/while/odelstm/sub_2:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_11�
rnn/while/odelstm/add_12AddV2rnn/while/odelstm/add_9:z:0rnn/while/odelstm/mul_11:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_12�
)rnn/while/odelstm/MatMul_8/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_8/ReadVariableOp�
rnn/while/odelstm/MatMul_8MatMulrnn/while/odelstm/mul_2:z:01rnn/while/odelstm/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_8�
)rnn/while/odelstm/MatMul_9/ReadVariableOpReadVariableOp4rnn_while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02+
)rnn/while/odelstm/MatMul_9/ReadVariableOp�
rnn/while/odelstm/MatMul_9MatMulrnn/while/odelstm/add_12:z:01rnn/while/odelstm/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/MatMul_9�
rnn/while/odelstm/add_13AddV2$rnn/while/odelstm/MatMul_8:product:0$rnn/while/odelstm/MatMul_9:product:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_13�
'rnn/while/odelstm/add_14/ReadVariableOpReadVariableOp1rnn_while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02)
'rnn/while/odelstm/add_14/ReadVariableOp�
rnn/while/odelstm/add_14AddV2rnn/while/odelstm/add_13:z:0/rnn/while/odelstm/add_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_14�
rnn/while/odelstm/Tanh_5Tanhrnn/while/odelstm/add_14:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/Tanh_5�
"rnn/while/odelstm/ReadVariableOp_3ReadVariableOp+rnn_while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"rnn/while/odelstm/ReadVariableOp_3�
rnn/while/odelstm/mul_12Mul*rnn/while/odelstm/ReadVariableOp_3:value:0rnn/while/odelstm/Tanh_5:y:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_12}
rnn/while/odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rnn/while/odelstm/mul_13/y�
rnn/while/odelstm/mul_13Mulrnn/while/odelstm/add_12:z:0#rnn/while/odelstm/mul_13/y:output:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_13�
rnn/while/odelstm/sub_3Subrnn/while/odelstm/mul_12:z:0rnn/while/odelstm/mul_13:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/sub_3}
rnn/while/odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
rnn/while/odelstm/mul_14/x�
rnn/while/odelstm/mul_14Mul#rnn/while/odelstm/mul_14/x:output:0rnn/while/odelstm/sub_3:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/mul_14�
rnn/while/odelstm/add_15AddV2rnn/while/odelstm/add_12:z:0rnn/while/odelstm/mul_14:z:0*
T0*
_output_shapes
:	�@2
rnn/while/odelstm/add_15�
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholderrnn/while/odelstm/add_15:z:0*
_output_shapes
: *
element_dtype020
.rnn/while/TensorArrayV2Write/TensorListSetItemd
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add/yy
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: 2
rnn/while/addh
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add_1/y�
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rnn/while/add_1�
rnn/while/IdentityIdentityrnn/while/add_1:z:0(^rnn/while/odelstm/MatMul/ReadVariableOp*^rnn/while/odelstm/MatMul_1/ReadVariableOp*^rnn/while/odelstm/MatMul_2/ReadVariableOp*^rnn/while/odelstm/MatMul_3/ReadVariableOp*^rnn/while/odelstm/MatMul_4/ReadVariableOp*^rnn/while/odelstm/MatMul_5/ReadVariableOp*^rnn/while/odelstm/MatMul_6/ReadVariableOp*^rnn/while/odelstm/MatMul_7/ReadVariableOp*^rnn/while/odelstm/MatMul_8/ReadVariableOp*^rnn/while/odelstm/MatMul_9/ReadVariableOp!^rnn/while/odelstm/ReadVariableOp#^rnn/while/odelstm/ReadVariableOp_1#^rnn/while/odelstm/ReadVariableOp_2#^rnn/while/odelstm/ReadVariableOp_3'^rnn/while/odelstm/add_1/ReadVariableOp(^rnn/while/odelstm/add_11/ReadVariableOp(^rnn/while/odelstm/add_14/ReadVariableOp'^rnn/while/odelstm/add_5/ReadVariableOp'^rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity�
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations(^rnn/while/odelstm/MatMul/ReadVariableOp*^rnn/while/odelstm/MatMul_1/ReadVariableOp*^rnn/while/odelstm/MatMul_2/ReadVariableOp*^rnn/while/odelstm/MatMul_3/ReadVariableOp*^rnn/while/odelstm/MatMul_4/ReadVariableOp*^rnn/while/odelstm/MatMul_5/ReadVariableOp*^rnn/while/odelstm/MatMul_6/ReadVariableOp*^rnn/while/odelstm/MatMul_7/ReadVariableOp*^rnn/while/odelstm/MatMul_8/ReadVariableOp*^rnn/while/odelstm/MatMul_9/ReadVariableOp!^rnn/while/odelstm/ReadVariableOp#^rnn/while/odelstm/ReadVariableOp_1#^rnn/while/odelstm/ReadVariableOp_2#^rnn/while/odelstm/ReadVariableOp_3'^rnn/while/odelstm/add_1/ReadVariableOp(^rnn/while/odelstm/add_11/ReadVariableOp(^rnn/while/odelstm/add_14/ReadVariableOp'^rnn/while/odelstm/add_5/ReadVariableOp'^rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity_1�
rnn/while/Identity_2Identityrnn/while/add:z:0(^rnn/while/odelstm/MatMul/ReadVariableOp*^rnn/while/odelstm/MatMul_1/ReadVariableOp*^rnn/while/odelstm/MatMul_2/ReadVariableOp*^rnn/while/odelstm/MatMul_3/ReadVariableOp*^rnn/while/odelstm/MatMul_4/ReadVariableOp*^rnn/while/odelstm/MatMul_5/ReadVariableOp*^rnn/while/odelstm/MatMul_6/ReadVariableOp*^rnn/while/odelstm/MatMul_7/ReadVariableOp*^rnn/while/odelstm/MatMul_8/ReadVariableOp*^rnn/while/odelstm/MatMul_9/ReadVariableOp!^rnn/while/odelstm/ReadVariableOp#^rnn/while/odelstm/ReadVariableOp_1#^rnn/while/odelstm/ReadVariableOp_2#^rnn/while/odelstm/ReadVariableOp_3'^rnn/while/odelstm/add_1/ReadVariableOp(^rnn/while/odelstm/add_11/ReadVariableOp(^rnn/while/odelstm/add_14/ReadVariableOp'^rnn/while/odelstm/add_5/ReadVariableOp'^rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity_2�
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^rnn/while/odelstm/MatMul/ReadVariableOp*^rnn/while/odelstm/MatMul_1/ReadVariableOp*^rnn/while/odelstm/MatMul_2/ReadVariableOp*^rnn/while/odelstm/MatMul_3/ReadVariableOp*^rnn/while/odelstm/MatMul_4/ReadVariableOp*^rnn/while/odelstm/MatMul_5/ReadVariableOp*^rnn/while/odelstm/MatMul_6/ReadVariableOp*^rnn/while/odelstm/MatMul_7/ReadVariableOp*^rnn/while/odelstm/MatMul_8/ReadVariableOp*^rnn/while/odelstm/MatMul_9/ReadVariableOp!^rnn/while/odelstm/ReadVariableOp#^rnn/while/odelstm/ReadVariableOp_1#^rnn/while/odelstm/ReadVariableOp_2#^rnn/while/odelstm/ReadVariableOp_3'^rnn/while/odelstm/add_1/ReadVariableOp(^rnn/while/odelstm/add_11/ReadVariableOp(^rnn/while/odelstm/add_14/ReadVariableOp'^rnn/while/odelstm/add_5/ReadVariableOp'^rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity_3�
rnn/while/Identity_4Identityrnn/while/odelstm/add_3:z:0(^rnn/while/odelstm/MatMul/ReadVariableOp*^rnn/while/odelstm/MatMul_1/ReadVariableOp*^rnn/while/odelstm/MatMul_2/ReadVariableOp*^rnn/while/odelstm/MatMul_3/ReadVariableOp*^rnn/while/odelstm/MatMul_4/ReadVariableOp*^rnn/while/odelstm/MatMul_5/ReadVariableOp*^rnn/while/odelstm/MatMul_6/ReadVariableOp*^rnn/while/odelstm/MatMul_7/ReadVariableOp*^rnn/while/odelstm/MatMul_8/ReadVariableOp*^rnn/while/odelstm/MatMul_9/ReadVariableOp!^rnn/while/odelstm/ReadVariableOp#^rnn/while/odelstm/ReadVariableOp_1#^rnn/while/odelstm/ReadVariableOp_2#^rnn/while/odelstm/ReadVariableOp_3'^rnn/while/odelstm/add_1/ReadVariableOp(^rnn/while/odelstm/add_11/ReadVariableOp(^rnn/while/odelstm/add_14/ReadVariableOp'^rnn/while/odelstm/add_5/ReadVariableOp'^rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
:	�@2
rnn/while/Identity_4�
rnn/while/Identity_5Identityrnn/while/odelstm/add_15:z:0(^rnn/while/odelstm/MatMul/ReadVariableOp*^rnn/while/odelstm/MatMul_1/ReadVariableOp*^rnn/while/odelstm/MatMul_2/ReadVariableOp*^rnn/while/odelstm/MatMul_3/ReadVariableOp*^rnn/while/odelstm/MatMul_4/ReadVariableOp*^rnn/while/odelstm/MatMul_5/ReadVariableOp*^rnn/while/odelstm/MatMul_6/ReadVariableOp*^rnn/while/odelstm/MatMul_7/ReadVariableOp*^rnn/while/odelstm/MatMul_8/ReadVariableOp*^rnn/while/odelstm/MatMul_9/ReadVariableOp!^rnn/while/odelstm/ReadVariableOp#^rnn/while/odelstm/ReadVariableOp_1#^rnn/while/odelstm/ReadVariableOp_2#^rnn/while/odelstm/ReadVariableOp_3'^rnn/while/odelstm/add_1/ReadVariableOp(^rnn/while/odelstm/add_11/ReadVariableOp(^rnn/while/odelstm/add_14/ReadVariableOp'^rnn/while/odelstm/add_5/ReadVariableOp'^rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
:	�@2
rnn/while/Identity_5"1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"d
/rnn_while_odelstm_add_1_readvariableop_resource1rnn_while_odelstm_add_1_readvariableop_resource_0"d
/rnn_while_odelstm_add_5_readvariableop_resource1rnn_while_odelstm_add_5_readvariableop_resource_0"j
2rnn_while_odelstm_matmul_1_readvariableop_resource4rnn_while_odelstm_matmul_1_readvariableop_resource_0"j
2rnn_while_odelstm_matmul_2_readvariableop_resource4rnn_while_odelstm_matmul_2_readvariableop_resource_0"j
2rnn_while_odelstm_matmul_3_readvariableop_resource4rnn_while_odelstm_matmul_3_readvariableop_resource_0"f
0rnn_while_odelstm_matmul_readvariableop_resource2rnn_while_odelstm_matmul_readvariableop_resource_0"X
)rnn_while_odelstm_readvariableop_resource+rnn_while_odelstm_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"�
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :	�@:	�@: : :::::::2R
'rnn/while/odelstm/MatMul/ReadVariableOp'rnn/while/odelstm/MatMul/ReadVariableOp2V
)rnn/while/odelstm/MatMul_1/ReadVariableOp)rnn/while/odelstm/MatMul_1/ReadVariableOp2V
)rnn/while/odelstm/MatMul_2/ReadVariableOp)rnn/while/odelstm/MatMul_2/ReadVariableOp2V
)rnn/while/odelstm/MatMul_3/ReadVariableOp)rnn/while/odelstm/MatMul_3/ReadVariableOp2V
)rnn/while/odelstm/MatMul_4/ReadVariableOp)rnn/while/odelstm/MatMul_4/ReadVariableOp2V
)rnn/while/odelstm/MatMul_5/ReadVariableOp)rnn/while/odelstm/MatMul_5/ReadVariableOp2V
)rnn/while/odelstm/MatMul_6/ReadVariableOp)rnn/while/odelstm/MatMul_6/ReadVariableOp2V
)rnn/while/odelstm/MatMul_7/ReadVariableOp)rnn/while/odelstm/MatMul_7/ReadVariableOp2V
)rnn/while/odelstm/MatMul_8/ReadVariableOp)rnn/while/odelstm/MatMul_8/ReadVariableOp2V
)rnn/while/odelstm/MatMul_9/ReadVariableOp)rnn/while/odelstm/MatMul_9/ReadVariableOp2D
 rnn/while/odelstm/ReadVariableOp rnn/while/odelstm/ReadVariableOp2H
"rnn/while/odelstm/ReadVariableOp_1"rnn/while/odelstm/ReadVariableOp_12H
"rnn/while/odelstm/ReadVariableOp_2"rnn/while/odelstm/ReadVariableOp_22H
"rnn/while/odelstm/ReadVariableOp_3"rnn/while/odelstm/ReadVariableOp_32P
&rnn/while/odelstm/add_1/ReadVariableOp&rnn/while/odelstm/add_1/ReadVariableOp2R
'rnn/while/odelstm/add_11/ReadVariableOp'rnn/while/odelstm/add_11/ReadVariableOp2R
'rnn/while/odelstm/add_14/ReadVariableOp'rnn/while/odelstm/add_14/ReadVariableOp2P
&rnn/while/odelstm/add_5/ReadVariableOp&rnn/while/odelstm/add_5/ReadVariableOp2P
&rnn/while/odelstm/add_8/ReadVariableOp&rnn/while/odelstm/add_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :

_output_shapes
: 
�
�
rnn_while_cond_60090$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_60090___redundant_placeholder0;
7rnn_while_rnn_while_cond_60090___redundant_placeholder1;
7rnn_while_rnn_while_cond_60090___redundant_placeholder2;
7rnn_while_rnn_while_cond_60090___redundant_placeholder3;
7rnn_while_rnn_while_cond_60090___redundant_placeholder4;
7rnn_while_rnn_while_cond_60090___redundant_placeholder5;
7rnn_while_rnn_while_cond_60090___redundant_placeholder6;
7rnn_while_rnn_while_cond_60090___redundant_placeholder7
rnn_while_identity
�
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: 2
rnn/while/Lessi
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: 2
rnn/while/Identity"1
rnn_while_identityrnn/while/Identity:output:0*S
_input_shapesB
@: : : : :	�@:	�@: ::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :

_output_shapes
:
�	
�
while_cond_59243
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_59243___redundant_placeholder03
/while_while_cond_59243___redundant_placeholder13
/while_while_cond_59243___redundant_placeholder23
/while_while_cond_59243___redundant_placeholder33
/while_while_cond_59243___redundant_placeholder43
/while_while_cond_59243___redundant_placeholder53
/while_while_cond_59243___redundant_placeholder63
/while_while_cond_59243___redundant_placeholder7
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :	�@:	�@: ::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :

_output_shapes
:
�

�
while_cond_60444
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_60444___redundant_placeholder03
/while_while_cond_60444___redundant_placeholder13
/while_while_cond_60444___redundant_placeholder23
/while_while_cond_60444___redundant_placeholder33
/while_while_cond_60444___redundant_placeholder43
/while_while_cond_60444___redundant_placeholder53
/while_while_cond_60444___redundant_placeholder63
/while_while_cond_60444___redundant_placeholder7
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*c
_input_shapesR
P: : : : :���������@:���������@: ::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
��
�
while_body_60745
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
.while_odelstm_matmul_readvariableop_resource_04
0while_odelstm_matmul_1_readvariableop_resource_01
-while_odelstm_add_1_readvariableop_resource_04
0while_odelstm_matmul_2_readvariableop_resource_04
0while_odelstm_matmul_3_readvariableop_resource_01
-while_odelstm_add_5_readvariableop_resource_0+
'while_odelstm_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
,while_odelstm_matmul_readvariableop_resource2
.while_odelstm_matmul_1_readvariableop_resource/
+while_odelstm_add_1_readvariableop_resource2
.while_odelstm_matmul_2_readvariableop_resource2
.while_odelstm_matmul_3_readvariableop_resource/
+while_odelstm_add_5_readvariableop_resource)
%while_odelstm_readvariableop_resource��#while/odelstm/MatMul/ReadVariableOp�%while/odelstm/MatMul_1/ReadVariableOp�%while/odelstm/MatMul_2/ReadVariableOp�%while/odelstm/MatMul_3/ReadVariableOp�%while/odelstm/MatMul_4/ReadVariableOp�%while/odelstm/MatMul_5/ReadVariableOp�%while/odelstm/MatMul_6/ReadVariableOp�%while/odelstm/MatMul_7/ReadVariableOp�%while/odelstm/MatMul_8/ReadVariableOp�%while/odelstm/MatMul_9/ReadVariableOp�while/odelstm/ReadVariableOp�while/odelstm/ReadVariableOp_1�while/odelstm/ReadVariableOp_2�while/odelstm/ReadVariableOp_3�"while/odelstm/add_1/ReadVariableOp�#while/odelstm/add_11/ReadVariableOp�#while/odelstm/add_14/ReadVariableOp�"while/odelstm/add_5/ReadVariableOp�"while/odelstm/add_8/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
#while/odelstm/MatMul/ReadVariableOpReadVariableOp.while_odelstm_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02%
#while/odelstm/MatMul/ReadVariableOp�
while/odelstm/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/odelstm/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/odelstm/MatMul�
%while/odelstm/MatMul_1/ReadVariableOpReadVariableOp0while_odelstm_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02'
%while/odelstm/MatMul_1/ReadVariableOp�
while/odelstm/MatMul_1MatMulwhile_placeholder_3-while/odelstm/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/odelstm/MatMul_1�
while/odelstm/addAddV2while/odelstm/MatMul:product:0 while/odelstm/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/odelstm/add�
"while/odelstm/add_1/ReadVariableOpReadVariableOp-while_odelstm_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype02$
"while/odelstm/add_1/ReadVariableOp�
while/odelstm/add_1AddV2while/odelstm/add:z:0*while/odelstm/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/odelstm/add_1l
while/odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/odelstm/Const�
while/odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/odelstm/split/split_dim�
while/odelstm/splitSplit&while/odelstm/split/split_dim:output:0while/odelstm/add_1:z:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/odelstm/split�
while/odelstm/TanhTanhwhile/odelstm/split:output:0*
T0*'
_output_shapes
:���������@2
while/odelstm/Tanh�
while/odelstm/SigmoidSigmoidwhile/odelstm/split:output:1*
T0*'
_output_shapes
:���������@2
while/odelstm/Sigmoids
while/odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
while/odelstm/add_2/y�
while/odelstm/add_2AddV2while/odelstm/split:output:2while/odelstm/add_2/y:output:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_2�
while/odelstm/Sigmoid_1Sigmoidwhile/odelstm/add_2:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/Sigmoid_1�
while/odelstm/Sigmoid_2Sigmoidwhile/odelstm/split:output:3*
T0*'
_output_shapes
:���������@2
while/odelstm/Sigmoid_2�
while/odelstm/mulMulwhile_placeholder_2while/odelstm/Sigmoid_1:y:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul�
while/odelstm/mul_1Mulwhile/odelstm/Tanh:y:0while/odelstm/Sigmoid:y:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_1�
while/odelstm/add_3AddV2while/odelstm/mul:z:0while/odelstm/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_3
while/odelstm/Tanh_1Tanhwhile/odelstm/add_3:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/Tanh_1�
while/odelstm/mul_2Mulwhile/odelstm/Tanh_1:y:0while/odelstm/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_2�
%while/odelstm/MatMul_2/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_2/ReadVariableOp�
while/odelstm/MatMul_2MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_2�
%while/odelstm/MatMul_3/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_3/ReadVariableOp�
while/odelstm/MatMul_3MatMulwhile_placeholder_3-while/odelstm/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_3�
while/odelstm/add_4AddV2 while/odelstm/MatMul_2:product:0 while/odelstm/MatMul_3:product:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_4�
"while/odelstm/add_5/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"while/odelstm/add_5/ReadVariableOp�
while/odelstm/add_5AddV2while/odelstm/add_4:z:0*while/odelstm/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_5
while/odelstm/Tanh_2Tanhwhile/odelstm/add_5:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/Tanh_2�
while/odelstm/ReadVariableOpReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02
while/odelstm/ReadVariableOp�
while/odelstm/mul_3Mul$while/odelstm/ReadVariableOp:value:0while/odelstm/Tanh_2:y:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_3s
while/odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_4/y�
while/odelstm/mul_4Mulwhile_placeholder_3while/odelstm/mul_4/y:output:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_4�
while/odelstm/subSubwhile/odelstm/mul_3:z:0while/odelstm/mul_4:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/subs
while/odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_5/x�
while/odelstm/mul_5Mulwhile/odelstm/mul_5/x:output:0while/odelstm/sub:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_5�
while/odelstm/add_6AddV2while_placeholder_3while/odelstm/mul_5:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_6�
%while/odelstm/MatMul_4/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_4/ReadVariableOp�
while/odelstm/MatMul_4MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_4�
%while/odelstm/MatMul_5/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_5/ReadVariableOp�
while/odelstm/MatMul_5MatMulwhile/odelstm/add_6:z:0-while/odelstm/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_5�
while/odelstm/add_7AddV2 while/odelstm/MatMul_4:product:0 while/odelstm/MatMul_5:product:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_7�
"while/odelstm/add_8/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02$
"while/odelstm/add_8/ReadVariableOp�
while/odelstm/add_8AddV2while/odelstm/add_7:z:0*while/odelstm/add_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_8
while/odelstm/Tanh_3Tanhwhile/odelstm/add_8:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/Tanh_3�
while/odelstm/ReadVariableOp_1ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_1�
while/odelstm/mul_6Mul&while/odelstm/ReadVariableOp_1:value:0while/odelstm/Tanh_3:y:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_6s
while/odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_7/y�
while/odelstm/mul_7Mulwhile/odelstm/add_6:z:0while/odelstm/mul_7/y:output:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_7�
while/odelstm/sub_1Subwhile/odelstm/mul_6:z:0while/odelstm/mul_7:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/sub_1s
while/odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_8/x�
while/odelstm/mul_8Mulwhile/odelstm/mul_8/x:output:0while/odelstm/sub_1:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_8�
while/odelstm/add_9AddV2while/odelstm/add_6:z:0while/odelstm/mul_8:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_9�
%while/odelstm/MatMul_6/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_6/ReadVariableOp�
while/odelstm/MatMul_6MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_6�
%while/odelstm/MatMul_7/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_7/ReadVariableOp�
while/odelstm/MatMul_7MatMulwhile/odelstm/add_9:z:0-while/odelstm/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_7�
while/odelstm/add_10AddV2 while/odelstm/MatMul_6:product:0 while/odelstm/MatMul_7:product:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_10�
#while/odelstm/add_11/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02%
#while/odelstm/add_11/ReadVariableOp�
while/odelstm/add_11AddV2while/odelstm/add_10:z:0+while/odelstm/add_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_11�
while/odelstm/Tanh_4Tanhwhile/odelstm/add_11:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/Tanh_4�
while/odelstm/ReadVariableOp_2ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_2�
while/odelstm/mul_9Mul&while/odelstm/ReadVariableOp_2:value:0while/odelstm/Tanh_4:y:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_9u
while/odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_10/y�
while/odelstm/mul_10Mulwhile/odelstm/add_9:z:0while/odelstm/mul_10/y:output:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_10�
while/odelstm/sub_2Subwhile/odelstm/mul_9:z:0while/odelstm/mul_10:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/sub_2u
while/odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_11/x�
while/odelstm/mul_11Mulwhile/odelstm/mul_11/x:output:0while/odelstm/sub_2:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_11�
while/odelstm/add_12AddV2while/odelstm/add_9:z:0while/odelstm/mul_11:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_12�
%while/odelstm/MatMul_8/ReadVariableOpReadVariableOp0while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_8/ReadVariableOp�
while/odelstm/MatMul_8MatMulwhile/odelstm/mul_2:z:0-while/odelstm/MatMul_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_8�
%while/odelstm/MatMul_9/ReadVariableOpReadVariableOp0while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype02'
%while/odelstm/MatMul_9/ReadVariableOp�
while/odelstm/MatMul_9MatMulwhile/odelstm/add_12:z:0-while/odelstm/MatMul_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/MatMul_9�
while/odelstm/add_13AddV2 while/odelstm/MatMul_8:product:0 while/odelstm/MatMul_9:product:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_13�
#while/odelstm/add_14/ReadVariableOpReadVariableOp-while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype02%
#while/odelstm/add_14/ReadVariableOp�
while/odelstm/add_14AddV2while/odelstm/add_13:z:0+while/odelstm/add_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_14�
while/odelstm/Tanh_5Tanhwhile/odelstm/add_14:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/Tanh_5�
while/odelstm/ReadVariableOp_3ReadVariableOp'while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02 
while/odelstm/ReadVariableOp_3�
while/odelstm/mul_12Mul&while/odelstm/ReadVariableOp_3:value:0while/odelstm/Tanh_5:y:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_12u
while/odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/odelstm/mul_13/y�
while/odelstm/mul_13Mulwhile/odelstm/add_12:z:0while/odelstm/mul_13/y:output:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_13�
while/odelstm/sub_3Subwhile/odelstm/mul_12:z:0while/odelstm/mul_13:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/sub_3u
while/odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2
while/odelstm/mul_14/x�
while/odelstm/mul_14Mulwhile/odelstm/mul_14/x:output:0while/odelstm/sub_3:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/mul_14�
while/odelstm/add_15AddV2while/odelstm/add_12:z:0while/odelstm/mul_14:z:0*
T0*'
_output_shapes
:���������@2
while/odelstm/add_15�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/odelstm/add_15:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/odelstm/add_3:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/odelstm/add_15:z:0$^while/odelstm/MatMul/ReadVariableOp&^while/odelstm/MatMul_1/ReadVariableOp&^while/odelstm/MatMul_2/ReadVariableOp&^while/odelstm/MatMul_3/ReadVariableOp&^while/odelstm/MatMul_4/ReadVariableOp&^while/odelstm/MatMul_5/ReadVariableOp&^while/odelstm/MatMul_6/ReadVariableOp&^while/odelstm/MatMul_7/ReadVariableOp&^while/odelstm/MatMul_8/ReadVariableOp&^while/odelstm/MatMul_9/ReadVariableOp^while/odelstm/ReadVariableOp^while/odelstm/ReadVariableOp_1^while/odelstm/ReadVariableOp_2^while/odelstm/ReadVariableOp_3#^while/odelstm/add_1/ReadVariableOp$^while/odelstm/add_11/ReadVariableOp$^while/odelstm/add_14/ReadVariableOp#^while/odelstm/add_5/ReadVariableOp#^while/odelstm/add_8/ReadVariableOp*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"\
+while_odelstm_add_1_readvariableop_resource-while_odelstm_add_1_readvariableop_resource_0"\
+while_odelstm_add_5_readvariableop_resource-while_odelstm_add_5_readvariableop_resource_0"b
.while_odelstm_matmul_1_readvariableop_resource0while_odelstm_matmul_1_readvariableop_resource_0"b
.while_odelstm_matmul_2_readvariableop_resource0while_odelstm_matmul_2_readvariableop_resource_0"b
.while_odelstm_matmul_3_readvariableop_resource0while_odelstm_matmul_3_readvariableop_resource_0"^
,while_odelstm_matmul_readvariableop_resource.while_odelstm_matmul_readvariableop_resource_0"P
%while_odelstm_readvariableop_resource'while_odelstm_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*a
_input_shapesP
N: : : : :���������@:���������@: : :::::::2J
#while/odelstm/MatMul/ReadVariableOp#while/odelstm/MatMul/ReadVariableOp2N
%while/odelstm/MatMul_1/ReadVariableOp%while/odelstm/MatMul_1/ReadVariableOp2N
%while/odelstm/MatMul_2/ReadVariableOp%while/odelstm/MatMul_2/ReadVariableOp2N
%while/odelstm/MatMul_3/ReadVariableOp%while/odelstm/MatMul_3/ReadVariableOp2N
%while/odelstm/MatMul_4/ReadVariableOp%while/odelstm/MatMul_4/ReadVariableOp2N
%while/odelstm/MatMul_5/ReadVariableOp%while/odelstm/MatMul_5/ReadVariableOp2N
%while/odelstm/MatMul_6/ReadVariableOp%while/odelstm/MatMul_6/ReadVariableOp2N
%while/odelstm/MatMul_7/ReadVariableOp%while/odelstm/MatMul_7/ReadVariableOp2N
%while/odelstm/MatMul_8/ReadVariableOp%while/odelstm/MatMul_8/ReadVariableOp2N
%while/odelstm/MatMul_9/ReadVariableOp%while/odelstm/MatMul_9/ReadVariableOp2<
while/odelstm/ReadVariableOpwhile/odelstm/ReadVariableOp2@
while/odelstm/ReadVariableOp_1while/odelstm/ReadVariableOp_12@
while/odelstm/ReadVariableOp_2while/odelstm/ReadVariableOp_22@
while/odelstm/ReadVariableOp_3while/odelstm/ReadVariableOp_32H
"while/odelstm/add_1/ReadVariableOp"while/odelstm/add_1/ReadVariableOp2J
#while/odelstm/add_11/ReadVariableOp#while/odelstm/add_11/ReadVariableOp2J
#while/odelstm/add_14/ReadVariableOp#while/odelstm/add_14/ReadVariableOp2H
"while/odelstm/add_5/ReadVariableOp"while/odelstm/add_5/ReadVariableOp2H
"while/odelstm/add_8/ReadVariableOp"while/odelstm/add_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_ode_lstm_layer_call_and_return_conditional_losses_59512
input_1
input_2
	rnn_59491
	rnn_59493
	rnn_59495
	rnn_59497
	rnn_59499
	rnn_59501
	rnn_59503
dense_59506
dense_59508
identity��dense/StatefulPartitionedCall�rnn/StatefulPartitionedCall�
rnn/StatefulPartitionedCallStatefulPartitionedCallinput_1	rnn_59491	rnn_59493	rnn_59495	rnn_59497	rnn_59499	rnn_59501	rnn_59503*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�@*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_594062
rnn/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_59506dense_59508*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_594702
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^rnn/StatefulPartitionedCall*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:�d:�d:::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:L H
#
_output_shapes
:�d
!
_user_specified_name	input_1:LH
#
_output_shapes
:�d
!
_user_specified_name	input_2
��
�
ode_lstm_rnn_while_body_578886
2ode_lstm_rnn_while_ode_lstm_rnn_while_loop_counter<
8ode_lstm_rnn_while_ode_lstm_rnn_while_maximum_iterations"
ode_lstm_rnn_while_placeholder$
 ode_lstm_rnn_while_placeholder_1$
 ode_lstm_rnn_while_placeholder_2$
 ode_lstm_rnn_while_placeholder_35
1ode_lstm_rnn_while_ode_lstm_rnn_strided_slice_1_0q
mode_lstm_rnn_while_tensorarrayv2read_tensorlistgetitem_ode_lstm_rnn_tensorarrayunstack_tensorlistfromtensor_0?
;ode_lstm_rnn_while_odelstm_matmul_readvariableop_resource_0A
=ode_lstm_rnn_while_odelstm_matmul_1_readvariableop_resource_0>
:ode_lstm_rnn_while_odelstm_add_1_readvariableop_resource_0A
=ode_lstm_rnn_while_odelstm_matmul_2_readvariableop_resource_0A
=ode_lstm_rnn_while_odelstm_matmul_3_readvariableop_resource_0>
:ode_lstm_rnn_while_odelstm_add_5_readvariableop_resource_08
4ode_lstm_rnn_while_odelstm_readvariableop_resource_0
ode_lstm_rnn_while_identity!
ode_lstm_rnn_while_identity_1!
ode_lstm_rnn_while_identity_2!
ode_lstm_rnn_while_identity_3!
ode_lstm_rnn_while_identity_4!
ode_lstm_rnn_while_identity_53
/ode_lstm_rnn_while_ode_lstm_rnn_strided_slice_1o
kode_lstm_rnn_while_tensorarrayv2read_tensorlistgetitem_ode_lstm_rnn_tensorarrayunstack_tensorlistfromtensor=
9ode_lstm_rnn_while_odelstm_matmul_readvariableop_resource?
;ode_lstm_rnn_while_odelstm_matmul_1_readvariableop_resource<
8ode_lstm_rnn_while_odelstm_add_1_readvariableop_resource?
;ode_lstm_rnn_while_odelstm_matmul_2_readvariableop_resource?
;ode_lstm_rnn_while_odelstm_matmul_3_readvariableop_resource<
8ode_lstm_rnn_while_odelstm_add_5_readvariableop_resource6
2ode_lstm_rnn_while_odelstm_readvariableop_resource��0ode_lstm/rnn/while/odelstm/MatMul/ReadVariableOp�2ode_lstm/rnn/while/odelstm/MatMul_1/ReadVariableOp�2ode_lstm/rnn/while/odelstm/MatMul_2/ReadVariableOp�2ode_lstm/rnn/while/odelstm/MatMul_3/ReadVariableOp�2ode_lstm/rnn/while/odelstm/MatMul_4/ReadVariableOp�2ode_lstm/rnn/while/odelstm/MatMul_5/ReadVariableOp�2ode_lstm/rnn/while/odelstm/MatMul_6/ReadVariableOp�2ode_lstm/rnn/while/odelstm/MatMul_7/ReadVariableOp�2ode_lstm/rnn/while/odelstm/MatMul_8/ReadVariableOp�2ode_lstm/rnn/while/odelstm/MatMul_9/ReadVariableOp�)ode_lstm/rnn/while/odelstm/ReadVariableOp�+ode_lstm/rnn/while/odelstm/ReadVariableOp_1�+ode_lstm/rnn/while/odelstm/ReadVariableOp_2�+ode_lstm/rnn/while/odelstm/ReadVariableOp_3�/ode_lstm/rnn/while/odelstm/add_1/ReadVariableOp�0ode_lstm/rnn/while/odelstm/add_11/ReadVariableOp�0ode_lstm/rnn/while/odelstm/add_14/ReadVariableOp�/ode_lstm/rnn/while/odelstm/add_5/ReadVariableOp�/ode_lstm/rnn/while/odelstm/add_8/ReadVariableOp�
Dode_lstm/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�      2F
Dode_lstm/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape�
6ode_lstm/rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmode_lstm_rnn_while_tensorarrayv2read_tensorlistgetitem_ode_lstm_rnn_tensorarrayunstack_tensorlistfromtensor_0ode_lstm_rnn_while_placeholderMode_lstm/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	�*
element_dtype028
6ode_lstm/rnn/while/TensorArrayV2Read/TensorListGetItem�
0ode_lstm/rnn/while/odelstm/MatMul/ReadVariableOpReadVariableOp;ode_lstm_rnn_while_odelstm_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype022
0ode_lstm/rnn/while/odelstm/MatMul/ReadVariableOp�
!ode_lstm/rnn/while/odelstm/MatMulMatMul=ode_lstm/rnn/while/TensorArrayV2Read/TensorListGetItem:item:08ode_lstm/rnn/while/odelstm/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2#
!ode_lstm/rnn/while/odelstm/MatMul�
2ode_lstm/rnn/while/odelstm/MatMul_1/ReadVariableOpReadVariableOp=ode_lstm_rnn_while_odelstm_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype024
2ode_lstm/rnn/while/odelstm/MatMul_1/ReadVariableOp�
#ode_lstm/rnn/while/odelstm/MatMul_1MatMul ode_lstm_rnn_while_placeholder_3:ode_lstm/rnn/while/odelstm/MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2%
#ode_lstm/rnn/while/odelstm/MatMul_1�
ode_lstm/rnn/while/odelstm/addAddV2+ode_lstm/rnn/while/odelstm/MatMul:product:0-ode_lstm/rnn/while/odelstm/MatMul_1:product:0*
T0* 
_output_shapes
:
��2 
ode_lstm/rnn/while/odelstm/add�
/ode_lstm/rnn/while/odelstm/add_1/ReadVariableOpReadVariableOp:ode_lstm_rnn_while_odelstm_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype021
/ode_lstm/rnn/while/odelstm/add_1/ReadVariableOp�
 ode_lstm/rnn/while/odelstm/add_1AddV2"ode_lstm/rnn/while/odelstm/add:z:07ode_lstm/rnn/while/odelstm/add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2"
 ode_lstm/rnn/while/odelstm/add_1�
 ode_lstm/rnn/while/odelstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 ode_lstm/rnn/while/odelstm/Const�
*ode_lstm/rnn/while/odelstm/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*ode_lstm/rnn/while/odelstm/split/split_dim�
 ode_lstm/rnn/while/odelstm/splitSplit3ode_lstm/rnn/while/odelstm/split/split_dim:output:0$ode_lstm/rnn/while/odelstm/add_1:z:0*
T0*@
_output_shapes.
,:	�@:	�@:	�@:	�@*
	num_split2"
 ode_lstm/rnn/while/odelstm/split�
ode_lstm/rnn/while/odelstm/TanhTanh)ode_lstm/rnn/while/odelstm/split:output:0*
T0*
_output_shapes
:	�@2!
ode_lstm/rnn/while/odelstm/Tanh�
"ode_lstm/rnn/while/odelstm/SigmoidSigmoid)ode_lstm/rnn/while/odelstm/split:output:1*
T0*
_output_shapes
:	�@2$
"ode_lstm/rnn/while/odelstm/Sigmoid�
"ode_lstm/rnn/while/odelstm/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2$
"ode_lstm/rnn/while/odelstm/add_2/y�
 ode_lstm/rnn/while/odelstm/add_2AddV2)ode_lstm/rnn/while/odelstm/split:output:2+ode_lstm/rnn/while/odelstm/add_2/y:output:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/add_2�
$ode_lstm/rnn/while/odelstm/Sigmoid_1Sigmoid$ode_lstm/rnn/while/odelstm/add_2:z:0*
T0*
_output_shapes
:	�@2&
$ode_lstm/rnn/while/odelstm/Sigmoid_1�
$ode_lstm/rnn/while/odelstm/Sigmoid_2Sigmoid)ode_lstm/rnn/while/odelstm/split:output:3*
T0*
_output_shapes
:	�@2&
$ode_lstm/rnn/while/odelstm/Sigmoid_2�
ode_lstm/rnn/while/odelstm/mulMul ode_lstm_rnn_while_placeholder_2(ode_lstm/rnn/while/odelstm/Sigmoid_1:y:0*
T0*
_output_shapes
:	�@2 
ode_lstm/rnn/while/odelstm/mul�
 ode_lstm/rnn/while/odelstm/mul_1Mul#ode_lstm/rnn/while/odelstm/Tanh:y:0&ode_lstm/rnn/while/odelstm/Sigmoid:y:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/mul_1�
 ode_lstm/rnn/while/odelstm/add_3AddV2"ode_lstm/rnn/while/odelstm/mul:z:0$ode_lstm/rnn/while/odelstm/mul_1:z:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/add_3�
!ode_lstm/rnn/while/odelstm/Tanh_1Tanh$ode_lstm/rnn/while/odelstm/add_3:z:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/Tanh_1�
 ode_lstm/rnn/while/odelstm/mul_2Mul%ode_lstm/rnn/while/odelstm/Tanh_1:y:0(ode_lstm/rnn/while/odelstm/Sigmoid_2:y:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/mul_2�
2ode_lstm/rnn/while/odelstm/MatMul_2/ReadVariableOpReadVariableOp=ode_lstm_rnn_while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype024
2ode_lstm/rnn/while/odelstm/MatMul_2/ReadVariableOp�
#ode_lstm/rnn/while/odelstm/MatMul_2MatMul$ode_lstm/rnn/while/odelstm/mul_2:z:0:ode_lstm/rnn/while/odelstm/MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2%
#ode_lstm/rnn/while/odelstm/MatMul_2�
2ode_lstm/rnn/while/odelstm/MatMul_3/ReadVariableOpReadVariableOp=ode_lstm_rnn_while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype024
2ode_lstm/rnn/while/odelstm/MatMul_3/ReadVariableOp�
#ode_lstm/rnn/while/odelstm/MatMul_3MatMul ode_lstm_rnn_while_placeholder_3:ode_lstm/rnn/while/odelstm/MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2%
#ode_lstm/rnn/while/odelstm/MatMul_3�
 ode_lstm/rnn/while/odelstm/add_4AddV2-ode_lstm/rnn/while/odelstm/MatMul_2:product:0-ode_lstm/rnn/while/odelstm/MatMul_3:product:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/add_4�
/ode_lstm/rnn/while/odelstm/add_5/ReadVariableOpReadVariableOp:ode_lstm_rnn_while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype021
/ode_lstm/rnn/while/odelstm/add_5/ReadVariableOp�
 ode_lstm/rnn/while/odelstm/add_5AddV2$ode_lstm/rnn/while/odelstm/add_4:z:07ode_lstm/rnn/while/odelstm/add_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/add_5�
!ode_lstm/rnn/while/odelstm/Tanh_2Tanh$ode_lstm/rnn/while/odelstm/add_5:z:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/Tanh_2�
)ode_lstm/rnn/while/odelstm/ReadVariableOpReadVariableOp4ode_lstm_rnn_while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02+
)ode_lstm/rnn/while/odelstm/ReadVariableOp�
 ode_lstm/rnn/while/odelstm/mul_3Mul1ode_lstm/rnn/while/odelstm/ReadVariableOp:value:0%ode_lstm/rnn/while/odelstm/Tanh_2:y:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/mul_3�
"ode_lstm/rnn/while/odelstm/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2$
"ode_lstm/rnn/while/odelstm/mul_4/y�
 ode_lstm/rnn/while/odelstm/mul_4Mul ode_lstm_rnn_while_placeholder_3+ode_lstm/rnn/while/odelstm/mul_4/y:output:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/mul_4�
ode_lstm/rnn/while/odelstm/subSub$ode_lstm/rnn/while/odelstm/mul_3:z:0$ode_lstm/rnn/while/odelstm/mul_4:z:0*
T0*
_output_shapes
:	�@2 
ode_lstm/rnn/while/odelstm/sub�
"ode_lstm/rnn/while/odelstm/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2$
"ode_lstm/rnn/while/odelstm/mul_5/x�
 ode_lstm/rnn/while/odelstm/mul_5Mul+ode_lstm/rnn/while/odelstm/mul_5/x:output:0"ode_lstm/rnn/while/odelstm/sub:z:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/mul_5�
 ode_lstm/rnn/while/odelstm/add_6AddV2 ode_lstm_rnn_while_placeholder_3$ode_lstm/rnn/while/odelstm/mul_5:z:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/add_6�
2ode_lstm/rnn/while/odelstm/MatMul_4/ReadVariableOpReadVariableOp=ode_lstm_rnn_while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype024
2ode_lstm/rnn/while/odelstm/MatMul_4/ReadVariableOp�
#ode_lstm/rnn/while/odelstm/MatMul_4MatMul$ode_lstm/rnn/while/odelstm/mul_2:z:0:ode_lstm/rnn/while/odelstm/MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2%
#ode_lstm/rnn/while/odelstm/MatMul_4�
2ode_lstm/rnn/while/odelstm/MatMul_5/ReadVariableOpReadVariableOp=ode_lstm_rnn_while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype024
2ode_lstm/rnn/while/odelstm/MatMul_5/ReadVariableOp�
#ode_lstm/rnn/while/odelstm/MatMul_5MatMul$ode_lstm/rnn/while/odelstm/add_6:z:0:ode_lstm/rnn/while/odelstm/MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2%
#ode_lstm/rnn/while/odelstm/MatMul_5�
 ode_lstm/rnn/while/odelstm/add_7AddV2-ode_lstm/rnn/while/odelstm/MatMul_4:product:0-ode_lstm/rnn/while/odelstm/MatMul_5:product:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/add_7�
/ode_lstm/rnn/while/odelstm/add_8/ReadVariableOpReadVariableOp:ode_lstm_rnn_while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype021
/ode_lstm/rnn/while/odelstm/add_8/ReadVariableOp�
 ode_lstm/rnn/while/odelstm/add_8AddV2$ode_lstm/rnn/while/odelstm/add_7:z:07ode_lstm/rnn/while/odelstm/add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/add_8�
!ode_lstm/rnn/while/odelstm/Tanh_3Tanh$ode_lstm/rnn/while/odelstm/add_8:z:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/Tanh_3�
+ode_lstm/rnn/while/odelstm/ReadVariableOp_1ReadVariableOp4ode_lstm_rnn_while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02-
+ode_lstm/rnn/while/odelstm/ReadVariableOp_1�
 ode_lstm/rnn/while/odelstm/mul_6Mul3ode_lstm/rnn/while/odelstm/ReadVariableOp_1:value:0%ode_lstm/rnn/while/odelstm/Tanh_3:y:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/mul_6�
"ode_lstm/rnn/while/odelstm/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2$
"ode_lstm/rnn/while/odelstm/mul_7/y�
 ode_lstm/rnn/while/odelstm/mul_7Mul$ode_lstm/rnn/while/odelstm/add_6:z:0+ode_lstm/rnn/while/odelstm/mul_7/y:output:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/mul_7�
 ode_lstm/rnn/while/odelstm/sub_1Sub$ode_lstm/rnn/while/odelstm/mul_6:z:0$ode_lstm/rnn/while/odelstm/mul_7:z:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/sub_1�
"ode_lstm/rnn/while/odelstm/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2$
"ode_lstm/rnn/while/odelstm/mul_8/x�
 ode_lstm/rnn/while/odelstm/mul_8Mul+ode_lstm/rnn/while/odelstm/mul_8/x:output:0$ode_lstm/rnn/while/odelstm/sub_1:z:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/mul_8�
 ode_lstm/rnn/while/odelstm/add_9AddV2$ode_lstm/rnn/while/odelstm/add_6:z:0$ode_lstm/rnn/while/odelstm/mul_8:z:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/add_9�
2ode_lstm/rnn/while/odelstm/MatMul_6/ReadVariableOpReadVariableOp=ode_lstm_rnn_while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype024
2ode_lstm/rnn/while/odelstm/MatMul_6/ReadVariableOp�
#ode_lstm/rnn/while/odelstm/MatMul_6MatMul$ode_lstm/rnn/while/odelstm/mul_2:z:0:ode_lstm/rnn/while/odelstm/MatMul_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2%
#ode_lstm/rnn/while/odelstm/MatMul_6�
2ode_lstm/rnn/while/odelstm/MatMul_7/ReadVariableOpReadVariableOp=ode_lstm_rnn_while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype024
2ode_lstm/rnn/while/odelstm/MatMul_7/ReadVariableOp�
#ode_lstm/rnn/while/odelstm/MatMul_7MatMul$ode_lstm/rnn/while/odelstm/add_9:z:0:ode_lstm/rnn/while/odelstm/MatMul_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2%
#ode_lstm/rnn/while/odelstm/MatMul_7�
!ode_lstm/rnn/while/odelstm/add_10AddV2-ode_lstm/rnn/while/odelstm/MatMul_6:product:0-ode_lstm/rnn/while/odelstm/MatMul_7:product:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/add_10�
0ode_lstm/rnn/while/odelstm/add_11/ReadVariableOpReadVariableOp:ode_lstm_rnn_while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype022
0ode_lstm/rnn/while/odelstm/add_11/ReadVariableOp�
!ode_lstm/rnn/while/odelstm/add_11AddV2%ode_lstm/rnn/while/odelstm/add_10:z:08ode_lstm/rnn/while/odelstm/add_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/add_11�
!ode_lstm/rnn/while/odelstm/Tanh_4Tanh%ode_lstm/rnn/while/odelstm/add_11:z:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/Tanh_4�
+ode_lstm/rnn/while/odelstm/ReadVariableOp_2ReadVariableOp4ode_lstm_rnn_while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02-
+ode_lstm/rnn/while/odelstm/ReadVariableOp_2�
 ode_lstm/rnn/while/odelstm/mul_9Mul3ode_lstm/rnn/while/odelstm/ReadVariableOp_2:value:0%ode_lstm/rnn/while/odelstm/Tanh_4:y:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/mul_9�
#ode_lstm/rnn/while/odelstm/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2%
#ode_lstm/rnn/while/odelstm/mul_10/y�
!ode_lstm/rnn/while/odelstm/mul_10Mul$ode_lstm/rnn/while/odelstm/add_9:z:0,ode_lstm/rnn/while/odelstm/mul_10/y:output:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/mul_10�
 ode_lstm/rnn/while/odelstm/sub_2Sub$ode_lstm/rnn/while/odelstm/mul_9:z:0%ode_lstm/rnn/while/odelstm/mul_10:z:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/sub_2�
#ode_lstm/rnn/while/odelstm/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2%
#ode_lstm/rnn/while/odelstm/mul_11/x�
!ode_lstm/rnn/while/odelstm/mul_11Mul,ode_lstm/rnn/while/odelstm/mul_11/x:output:0$ode_lstm/rnn/while/odelstm/sub_2:z:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/mul_11�
!ode_lstm/rnn/while/odelstm/add_12AddV2$ode_lstm/rnn/while/odelstm/add_9:z:0%ode_lstm/rnn/while/odelstm/mul_11:z:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/add_12�
2ode_lstm/rnn/while/odelstm/MatMul_8/ReadVariableOpReadVariableOp=ode_lstm_rnn_while_odelstm_matmul_2_readvariableop_resource_0*
_output_shapes

:@@*
dtype024
2ode_lstm/rnn/while/odelstm/MatMul_8/ReadVariableOp�
#ode_lstm/rnn/while/odelstm/MatMul_8MatMul$ode_lstm/rnn/while/odelstm/mul_2:z:0:ode_lstm/rnn/while/odelstm/MatMul_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2%
#ode_lstm/rnn/while/odelstm/MatMul_8�
2ode_lstm/rnn/while/odelstm/MatMul_9/ReadVariableOpReadVariableOp=ode_lstm_rnn_while_odelstm_matmul_3_readvariableop_resource_0*
_output_shapes

:@@*
dtype024
2ode_lstm/rnn/while/odelstm/MatMul_9/ReadVariableOp�
#ode_lstm/rnn/while/odelstm/MatMul_9MatMul%ode_lstm/rnn/while/odelstm/add_12:z:0:ode_lstm/rnn/while/odelstm/MatMul_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2%
#ode_lstm/rnn/while/odelstm/MatMul_9�
!ode_lstm/rnn/while/odelstm/add_13AddV2-ode_lstm/rnn/while/odelstm/MatMul_8:product:0-ode_lstm/rnn/while/odelstm/MatMul_9:product:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/add_13�
0ode_lstm/rnn/while/odelstm/add_14/ReadVariableOpReadVariableOp:ode_lstm_rnn_while_odelstm_add_5_readvariableop_resource_0*
_output_shapes
:@*
dtype022
0ode_lstm/rnn/while/odelstm/add_14/ReadVariableOp�
!ode_lstm/rnn/while/odelstm/add_14AddV2%ode_lstm/rnn/while/odelstm/add_13:z:08ode_lstm/rnn/while/odelstm/add_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/add_14�
!ode_lstm/rnn/while/odelstm/Tanh_5Tanh%ode_lstm/rnn/while/odelstm/add_14:z:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/Tanh_5�
+ode_lstm/rnn/while/odelstm/ReadVariableOp_3ReadVariableOp4ode_lstm_rnn_while_odelstm_readvariableop_resource_0*
_output_shapes
:@*
dtype02-
+ode_lstm/rnn/while/odelstm/ReadVariableOp_3�
!ode_lstm/rnn/while/odelstm/mul_12Mul3ode_lstm/rnn/while/odelstm/ReadVariableOp_3:value:0%ode_lstm/rnn/while/odelstm/Tanh_5:y:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/mul_12�
#ode_lstm/rnn/while/odelstm/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2%
#ode_lstm/rnn/while/odelstm/mul_13/y�
!ode_lstm/rnn/while/odelstm/mul_13Mul%ode_lstm/rnn/while/odelstm/add_12:z:0,ode_lstm/rnn/while/odelstm/mul_13/y:output:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/mul_13�
 ode_lstm/rnn/while/odelstm/sub_3Sub%ode_lstm/rnn/while/odelstm/mul_12:z:0%ode_lstm/rnn/while/odelstm/mul_13:z:0*
T0*
_output_shapes
:	�@2"
 ode_lstm/rnn/while/odelstm/sub_3�
#ode_lstm/rnn/while/odelstm/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �>2%
#ode_lstm/rnn/while/odelstm/mul_14/x�
!ode_lstm/rnn/while/odelstm/mul_14Mul,ode_lstm/rnn/while/odelstm/mul_14/x:output:0$ode_lstm/rnn/while/odelstm/sub_3:z:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/mul_14�
!ode_lstm/rnn/while/odelstm/add_15AddV2%ode_lstm/rnn/while/odelstm/add_12:z:0%ode_lstm/rnn/while/odelstm/mul_14:z:0*
T0*
_output_shapes
:	�@2#
!ode_lstm/rnn/while/odelstm/add_15�
7ode_lstm/rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem ode_lstm_rnn_while_placeholder_1ode_lstm_rnn_while_placeholder%ode_lstm/rnn/while/odelstm/add_15:z:0*
_output_shapes
: *
element_dtype029
7ode_lstm/rnn/while/TensorArrayV2Write/TensorListSetItemv
ode_lstm/rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
ode_lstm/rnn/while/add/y�
ode_lstm/rnn/while/addAddV2ode_lstm_rnn_while_placeholder!ode_lstm/rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
ode_lstm/rnn/while/addz
ode_lstm/rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
ode_lstm/rnn/while/add_1/y�
ode_lstm/rnn/while/add_1AddV22ode_lstm_rnn_while_ode_lstm_rnn_while_loop_counter#ode_lstm/rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
ode_lstm/rnn/while/add_1�
ode_lstm/rnn/while/IdentityIdentityode_lstm/rnn/while/add_1:z:01^ode_lstm/rnn/while/odelstm/MatMul/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_1/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_2/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_3/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_4/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_5/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_6/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_7/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_8/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_9/ReadVariableOp*^ode_lstm/rnn/while/odelstm/ReadVariableOp,^ode_lstm/rnn/while/odelstm/ReadVariableOp_1,^ode_lstm/rnn/while/odelstm/ReadVariableOp_2,^ode_lstm/rnn/while/odelstm/ReadVariableOp_30^ode_lstm/rnn/while/odelstm/add_1/ReadVariableOp1^ode_lstm/rnn/while/odelstm/add_11/ReadVariableOp1^ode_lstm/rnn/while/odelstm/add_14/ReadVariableOp0^ode_lstm/rnn/while/odelstm/add_5/ReadVariableOp0^ode_lstm/rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
ode_lstm/rnn/while/Identity�
ode_lstm/rnn/while/Identity_1Identity8ode_lstm_rnn_while_ode_lstm_rnn_while_maximum_iterations1^ode_lstm/rnn/while/odelstm/MatMul/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_1/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_2/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_3/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_4/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_5/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_6/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_7/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_8/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_9/ReadVariableOp*^ode_lstm/rnn/while/odelstm/ReadVariableOp,^ode_lstm/rnn/while/odelstm/ReadVariableOp_1,^ode_lstm/rnn/while/odelstm/ReadVariableOp_2,^ode_lstm/rnn/while/odelstm/ReadVariableOp_30^ode_lstm/rnn/while/odelstm/add_1/ReadVariableOp1^ode_lstm/rnn/while/odelstm/add_11/ReadVariableOp1^ode_lstm/rnn/while/odelstm/add_14/ReadVariableOp0^ode_lstm/rnn/while/odelstm/add_5/ReadVariableOp0^ode_lstm/rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
ode_lstm/rnn/while/Identity_1�
ode_lstm/rnn/while/Identity_2Identityode_lstm/rnn/while/add:z:01^ode_lstm/rnn/while/odelstm/MatMul/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_1/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_2/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_3/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_4/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_5/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_6/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_7/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_8/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_9/ReadVariableOp*^ode_lstm/rnn/while/odelstm/ReadVariableOp,^ode_lstm/rnn/while/odelstm/ReadVariableOp_1,^ode_lstm/rnn/while/odelstm/ReadVariableOp_2,^ode_lstm/rnn/while/odelstm/ReadVariableOp_30^ode_lstm/rnn/while/odelstm/add_1/ReadVariableOp1^ode_lstm/rnn/while/odelstm/add_11/ReadVariableOp1^ode_lstm/rnn/while/odelstm/add_14/ReadVariableOp0^ode_lstm/rnn/while/odelstm/add_5/ReadVariableOp0^ode_lstm/rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
ode_lstm/rnn/while/Identity_2�
ode_lstm/rnn/while/Identity_3IdentityGode_lstm/rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^ode_lstm/rnn/while/odelstm/MatMul/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_1/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_2/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_3/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_4/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_5/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_6/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_7/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_8/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_9/ReadVariableOp*^ode_lstm/rnn/while/odelstm/ReadVariableOp,^ode_lstm/rnn/while/odelstm/ReadVariableOp_1,^ode_lstm/rnn/while/odelstm/ReadVariableOp_2,^ode_lstm/rnn/while/odelstm/ReadVariableOp_30^ode_lstm/rnn/while/odelstm/add_1/ReadVariableOp1^ode_lstm/rnn/while/odelstm/add_11/ReadVariableOp1^ode_lstm/rnn/while/odelstm/add_14/ReadVariableOp0^ode_lstm/rnn/while/odelstm/add_5/ReadVariableOp0^ode_lstm/rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
: 2
ode_lstm/rnn/while/Identity_3�
ode_lstm/rnn/while/Identity_4Identity$ode_lstm/rnn/while/odelstm/add_3:z:01^ode_lstm/rnn/while/odelstm/MatMul/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_1/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_2/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_3/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_4/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_5/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_6/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_7/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_8/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_9/ReadVariableOp*^ode_lstm/rnn/while/odelstm/ReadVariableOp,^ode_lstm/rnn/while/odelstm/ReadVariableOp_1,^ode_lstm/rnn/while/odelstm/ReadVariableOp_2,^ode_lstm/rnn/while/odelstm/ReadVariableOp_30^ode_lstm/rnn/while/odelstm/add_1/ReadVariableOp1^ode_lstm/rnn/while/odelstm/add_11/ReadVariableOp1^ode_lstm/rnn/while/odelstm/add_14/ReadVariableOp0^ode_lstm/rnn/while/odelstm/add_5/ReadVariableOp0^ode_lstm/rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/while/Identity_4�
ode_lstm/rnn/while/Identity_5Identity%ode_lstm/rnn/while/odelstm/add_15:z:01^ode_lstm/rnn/while/odelstm/MatMul/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_1/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_2/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_3/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_4/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_5/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_6/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_7/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_8/ReadVariableOp3^ode_lstm/rnn/while/odelstm/MatMul_9/ReadVariableOp*^ode_lstm/rnn/while/odelstm/ReadVariableOp,^ode_lstm/rnn/while/odelstm/ReadVariableOp_1,^ode_lstm/rnn/while/odelstm/ReadVariableOp_2,^ode_lstm/rnn/while/odelstm/ReadVariableOp_30^ode_lstm/rnn/while/odelstm/add_1/ReadVariableOp1^ode_lstm/rnn/while/odelstm/add_11/ReadVariableOp1^ode_lstm/rnn/while/odelstm/add_14/ReadVariableOp0^ode_lstm/rnn/while/odelstm/add_5/ReadVariableOp0^ode_lstm/rnn/while/odelstm/add_8/ReadVariableOp*
T0*
_output_shapes
:	�@2
ode_lstm/rnn/while/Identity_5"C
ode_lstm_rnn_while_identity$ode_lstm/rnn/while/Identity:output:0"G
ode_lstm_rnn_while_identity_1&ode_lstm/rnn/while/Identity_1:output:0"G
ode_lstm_rnn_while_identity_2&ode_lstm/rnn/while/Identity_2:output:0"G
ode_lstm_rnn_while_identity_3&ode_lstm/rnn/while/Identity_3:output:0"G
ode_lstm_rnn_while_identity_4&ode_lstm/rnn/while/Identity_4:output:0"G
ode_lstm_rnn_while_identity_5&ode_lstm/rnn/while/Identity_5:output:0"d
/ode_lstm_rnn_while_ode_lstm_rnn_strided_slice_11ode_lstm_rnn_while_ode_lstm_rnn_strided_slice_1_0"v
8ode_lstm_rnn_while_odelstm_add_1_readvariableop_resource:ode_lstm_rnn_while_odelstm_add_1_readvariableop_resource_0"v
8ode_lstm_rnn_while_odelstm_add_5_readvariableop_resource:ode_lstm_rnn_while_odelstm_add_5_readvariableop_resource_0"|
;ode_lstm_rnn_while_odelstm_matmul_1_readvariableop_resource=ode_lstm_rnn_while_odelstm_matmul_1_readvariableop_resource_0"|
;ode_lstm_rnn_while_odelstm_matmul_2_readvariableop_resource=ode_lstm_rnn_while_odelstm_matmul_2_readvariableop_resource_0"|
;ode_lstm_rnn_while_odelstm_matmul_3_readvariableop_resource=ode_lstm_rnn_while_odelstm_matmul_3_readvariableop_resource_0"x
9ode_lstm_rnn_while_odelstm_matmul_readvariableop_resource;ode_lstm_rnn_while_odelstm_matmul_readvariableop_resource_0"j
2ode_lstm_rnn_while_odelstm_readvariableop_resource4ode_lstm_rnn_while_odelstm_readvariableop_resource_0"�
kode_lstm_rnn_while_tensorarrayv2read_tensorlistgetitem_ode_lstm_rnn_tensorarrayunstack_tensorlistfromtensormode_lstm_rnn_while_tensorarrayv2read_tensorlistgetitem_ode_lstm_rnn_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :	�@:	�@: : :::::::2d
0ode_lstm/rnn/while/odelstm/MatMul/ReadVariableOp0ode_lstm/rnn/while/odelstm/MatMul/ReadVariableOp2h
2ode_lstm/rnn/while/odelstm/MatMul_1/ReadVariableOp2ode_lstm/rnn/while/odelstm/MatMul_1/ReadVariableOp2h
2ode_lstm/rnn/while/odelstm/MatMul_2/ReadVariableOp2ode_lstm/rnn/while/odelstm/MatMul_2/ReadVariableOp2h
2ode_lstm/rnn/while/odelstm/MatMul_3/ReadVariableOp2ode_lstm/rnn/while/odelstm/MatMul_3/ReadVariableOp2h
2ode_lstm/rnn/while/odelstm/MatMul_4/ReadVariableOp2ode_lstm/rnn/while/odelstm/MatMul_4/ReadVariableOp2h
2ode_lstm/rnn/while/odelstm/MatMul_5/ReadVariableOp2ode_lstm/rnn/while/odelstm/MatMul_5/ReadVariableOp2h
2ode_lstm/rnn/while/odelstm/MatMul_6/ReadVariableOp2ode_lstm/rnn/while/odelstm/MatMul_6/ReadVariableOp2h
2ode_lstm/rnn/while/odelstm/MatMul_7/ReadVariableOp2ode_lstm/rnn/while/odelstm/MatMul_7/ReadVariableOp2h
2ode_lstm/rnn/while/odelstm/MatMul_8/ReadVariableOp2ode_lstm/rnn/while/odelstm/MatMul_8/ReadVariableOp2h
2ode_lstm/rnn/while/odelstm/MatMul_9/ReadVariableOp2ode_lstm/rnn/while/odelstm/MatMul_9/ReadVariableOp2V
)ode_lstm/rnn/while/odelstm/ReadVariableOp)ode_lstm/rnn/while/odelstm/ReadVariableOp2Z
+ode_lstm/rnn/while/odelstm/ReadVariableOp_1+ode_lstm/rnn/while/odelstm/ReadVariableOp_12Z
+ode_lstm/rnn/while/odelstm/ReadVariableOp_2+ode_lstm/rnn/while/odelstm/ReadVariableOp_22Z
+ode_lstm/rnn/while/odelstm/ReadVariableOp_3+ode_lstm/rnn/while/odelstm/ReadVariableOp_32b
/ode_lstm/rnn/while/odelstm/add_1/ReadVariableOp/ode_lstm/rnn/while/odelstm/add_1/ReadVariableOp2d
0ode_lstm/rnn/while/odelstm/add_11/ReadVariableOp0ode_lstm/rnn/while/odelstm/add_11/ReadVariableOp2d
0ode_lstm/rnn/while/odelstm/add_14/ReadVariableOp0ode_lstm/rnn/while/odelstm/add_14/ReadVariableOp2b
/ode_lstm/rnn/while/odelstm/add_5/ReadVariableOp/ode_lstm/rnn/while/odelstm/add_5/ReadVariableOp2b
/ode_lstm/rnn/while/odelstm/add_8/ReadVariableOp/ode_lstm/rnn/while/odelstm/add_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@:%!

_output_shapes
:	�@:

_output_shapes
: :

_output_shapes
: 
�
�
#__inference_rnn_layer_call_fn_61583

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�@*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_594062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	�@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:�d:::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:�d
 
_user_specified_nameinputs
�'
�
while_body_58701
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_odelstm_58725_0
while_odelstm_58727_0
while_odelstm_58729_0
while_odelstm_58731_0
while_odelstm_58733_0
while_odelstm_58735_0
while_odelstm_58737_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_odelstm_58725
while_odelstm_58727
while_odelstm_58729
while_odelstm_58731
while_odelstm_58733
while_odelstm_58735
while_odelstm_58737��%while/odelstm/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
%while/odelstm/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_odelstm_58725_0while_odelstm_58727_0while_odelstm_58729_0while_odelstm_58731_0while_odelstm_58733_0while_odelstm_58735_0while_odelstm_58737_0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*)
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_odelstm_layer_call_and_return_conditional_losses_581992'
%while/odelstm/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder.while/odelstm/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0&^while/odelstm/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations&^while/odelstm/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0&^while/odelstm/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^while/odelstm/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity.while/odelstm/StatefulPartitionedCall:output:1&^while/odelstm/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity.while/odelstm/StatefulPartitionedCall:output:2&^while/odelstm/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_odelstm_58725while_odelstm_58725_0",
while_odelstm_58727while_odelstm_58727_0",
while_odelstm_58729while_odelstm_58729_0",
while_odelstm_58731while_odelstm_58731_0",
while_odelstm_58733while_odelstm_58733_0",
while_odelstm_58735while_odelstm_58735_0",
while_odelstm_58737while_odelstm_58737_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*a
_input_shapesP
N: : : : :���������@:���������@: : :::::::2N
%while/odelstm/StatefulPartitionedCall%while/odelstm/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�	
�
(__inference_ode_lstm_layer_call_fn_60307
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_ode_lstm_layer_call_and_return_conditional_losses_595902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:�d:�d:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:�d
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:�d
"
_user_specified_name
inputs/1
�
z
%__inference_dense_layer_call_fn_61602

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_594702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	�2

Identity"
identityIdentity:output:0*&
_input_shapes
:	�@::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	�@
 
_user_specified_nameinputs
�E
�
>__inference_rnn_layer_call_and_return_conditional_losses_58785

inputs
odelstm_58680
odelstm_58682
odelstm_58684
odelstm_58686
odelstm_58688
odelstm_58690
odelstm_58692
identity��odelstm/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
odelstm/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0odelstm_58680odelstm_58682odelstm_58684odelstm_58686odelstm_58688odelstm_58690odelstm_58692*
Tin
2
*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*)
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_odelstm_layer_call_and_return_conditional_losses_581992!
odelstm/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0odelstm_58680odelstm_58682odelstm_58684odelstm_58686odelstm_58688odelstm_58690odelstm_58692*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*T
_output_shapesB
@: : : : :���������@:���������@: : : : : : : : : *)
_read_only_resource_inputs
		
*
bodyR
while_body_58701*
condR
while_cond_58700*S
output_shapesB
@: : : : :���������@:���������@: : : : : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1�
IdentityIdentitystrided_slice_3:output:0 ^odelstm/StatefulPartitionedCall^while*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:������������������:::::::2B
odelstm/StatefulPartitionedCallodelstm/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
7
input_1,
serving_default_input_1:0�d
7
input_2,
serving_default_input_2:0�d1
dense(
StatefulPartitionedCall:0	�tensorflow/serving/predict:��
�$
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
*^&call_and_return_all_conditional_losses
___call__
`_default_save_signature"�!
_tf_keras_network�!{"class_name": "Functional", "name": "ode_lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "ode_lstm", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [128, 100, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "RNN", "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Custom>ODELSTM", "config": {"name": "odelstm", "trainable": true, "dtype": "float32", "units": 64}}}, "name": "rnn", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [128, 100, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["rnn", 0, 0, {}]]]}], "input_layers": {"class_name": "__tuple__", "items": [["input_1", 0, 0], ["input_2", 0, 0]]}, "output_layers": [["dense", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 100, 2]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 100, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "__tuple__", "items": [{"class_name": "TensorShape", "items": [128, 100, 2]}, {"class_name": "TensorShape", "items": [128, 100, 1]}]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "ode_lstm", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [128, 100, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "RNN", "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Custom>ODELSTM", "config": {"name": "odelstm", "trainable": true, "dtype": "float32", "units": 64}}}, "name": "rnn", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [128, 100, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["rnn", 0, 0, {}]]]}], "input_layers": {"class_name": "__tuple__", "items": [["input_1", 0, 0], ["input_2", 0, 0]]}, "output_layers": [["dense", 0, 0]]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 1.000000082740371e-08, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [128, 100, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [128, 100, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"�
_tf_keras_rnn_layer�{"class_name": "RNN", "name": "rnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Custom>ODELSTM", "config": {"name": "odelstm", "trainable": true, "dtype": "float32", "units": 64}}}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 2]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [128, 100, 2]}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [128, 100, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [128, 100, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 64]}}
�
iter

beta_1

beta_2
	decay
learning_ratemLmMmNmOmPmQ mR!mS"mTvUvVvWvXvYvZ v[!v\"v]"
	optimizer
_
0
1
2
3
 4
!5
"6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
 4
!5
"6
7
8"
trackable_list_wrapper
�
#metrics
	variables
$layer_regularization_losses
%layer_metrics

&layers
'non_trainable_variables
regularization_losses
trainable_variables
___call__
`_default_save_signature
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
,
eserving_default"
signature_map
�
	(ctrnn
input_kernel
recurrent_kernel
bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
*f&call_and_return_all_conditional_losses
g__call__"�
_tf_keras_layer�{"class_name": "Custom>ODELSTM", "name": "odelstm", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "odelstm", "trainable": true, "dtype": "float32", "units": 64}}
 "
trackable_list_wrapper
Q
0
1
2
3
 4
!5
"6"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
 4
!5
"6"
trackable_list_wrapper
�
-metrics
.layer_regularization_losses
/layer_metrics

0layers
	variables
1non_trainable_variables
regularization_losses
trainable_variables

2states
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
:@2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
3metrics
4layer_regularization_losses
5layer_metrics

6layers
	variables
7non_trainable_variables
regularization_losses
trainable_variables
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
+:)	�2rnn/odelstm/input_kernel
/:-	@�2rnn/odelstm/recurrent_kernel
:�2rnn/odelstm/bias
$:"@@2rnn/odelstm/kernel
.:,@@2rnn/odelstm/recurrent_kernel
:@2rnn/odelstm/bias
:@2rnn/odelstm/scale
'
80"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
9fixed_step_methods

kernel
 recurrent_kernel
!bias
	"scale
:	variables
;regularization_losses
<trainable_variables
=	keras_api
*h&call_and_return_all_conditional_losses
i__call__"�
_tf_keras_layer�{"class_name": "Custom>CTRNNCell", "name": "ctrnn_cell", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ctrnn_cell", "trainable": true, "dtype": "float32", "units": 64, "method": "euler", "num_unfolds": 4, "tau": 1}}
Q
0
1
2
3
 4
!5
"6"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
 4
!5
"6"
trackable_list_wrapper
�
>metrics
?layer_regularization_losses
@layer_metrics

Alayers
)	variables
Bnon_trainable_variables
*regularization_losses
+trainable_variables
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
!"
trackable_tuple_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	Ctotal
	Dcount
E	variables
F	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_dict_wrapper
<
0
 1
!2
"3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
 1
!2
"3"
trackable_list_wrapper
�
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics

Jlayers
:	variables
Knon_trainable_variables
;regularization_losses
<trainable_variables
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
C0
D1"
trackable_list_wrapper
-
E	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
#:!@2Adam/dense/kernel/m
:2Adam/dense/bias/m
0:.	�2Adam/rnn/odelstm/input_kernel/m
4:2	@�2#Adam/rnn/odelstm/recurrent_kernel/m
$:"�2Adam/rnn/odelstm/bias/m
):'@@2Adam/rnn/odelstm/kernel/m
3:1@@2#Adam/rnn/odelstm/recurrent_kernel/m
#:!@2Adam/rnn/odelstm/bias/m
$:"@2Adam/rnn/odelstm/scale/m
#:!@2Adam/dense/kernel/v
:2Adam/dense/bias/v
0:.	�2Adam/rnn/odelstm/input_kernel/v
4:2	@�2#Adam/rnn/odelstm/recurrent_kernel/v
$:"�2Adam/rnn/odelstm/bias/v
):'@@2Adam/rnn/odelstm/kernel/v
3:1@@2#Adam/rnn/odelstm/recurrent_kernel/v
#:!@2Adam/rnn/odelstm/bias/v
$:"@2Adam/rnn/odelstm/scale/v
�2�
C__inference_ode_lstm_layer_call_and_return_conditional_losses_60259
C__inference_ode_lstm_layer_call_and_return_conditional_losses_59487
C__inference_ode_lstm_layer_call_and_return_conditional_losses_59512
C__inference_ode_lstm_layer_call_and_return_conditional_losses_59952�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_ode_lstm_layer_call_fn_59562
(__inference_ode_lstm_layer_call_fn_59611
(__inference_ode_lstm_layer_call_fn_60283
(__inference_ode_lstm_layer_call_fn_60307�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
 __inference__wrapped_model_58056�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *F�C
A�>
�
input_1�d
�
input_2�d
�2�
>__inference_rnn_layer_call_and_return_conditional_losses_60607
>__inference_rnn_layer_call_and_return_conditional_losses_60907
>__inference_rnn_layer_call_and_return_conditional_losses_61245
>__inference_rnn_layer_call_and_return_conditional_losses_61545�
���
FullArgSpecO
argsG�D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults�

 
p 

 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference_rnn_layer_call_fn_60945
#__inference_rnn_layer_call_fn_60926
#__inference_rnn_layer_call_fn_61583
#__inference_rnn_layer_call_fn_61564�
���
FullArgSpecO
argsG�D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults�

 
p 

 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_dense_layer_call_and_return_conditional_losses_61593�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_dense_layer_call_fn_61602�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_59645input_1input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_odelstm_layer_call_and_return_conditional_losses_61705�
���
FullArgSpec'
args�
jself
jinputs
jstates
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_odelstm_layer_call_fn_61730�
���
FullArgSpec'
args�
jself
jinputs
jstates
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jstates
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jstates
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_58056�	 !"P�M
F�C
A�>
�
input_1�d
�
input_2�d
� "%�"
 
dense�
dense	��
@__inference_dense_layer_call_and_return_conditional_losses_61593L'�$
�
�
inputs	�@
� "�
�
0	�
� h
%__inference_dense_layer_call_fn_61602?'�$
�
�
inputs	�@
� "�	��
C__inference_ode_lstm_layer_call_and_return_conditional_losses_59487�	 !"X�U
N�K
A�>
�
input_1�d
�
input_2�d
p

 
� "�
�
0	�
� �
C__inference_ode_lstm_layer_call_and_return_conditional_losses_59512�	 !"X�U
N�K
A�>
�
input_1�d
�
input_2�d
p 

 
� "�
�
0	�
� �
C__inference_ode_lstm_layer_call_and_return_conditional_losses_59952�	 !"Z�W
P�M
C�@
�
inputs/0�d
�
inputs/1�d
p

 
� "�
�
0	�
� �
C__inference_ode_lstm_layer_call_and_return_conditional_losses_60259�	 !"Z�W
P�M
C�@
�
inputs/0�d
�
inputs/1�d
p 

 
� "�
�
0	�
� �
(__inference_ode_lstm_layer_call_fn_59562w	 !"X�U
N�K
A�>
�
input_1�d
�
input_2�d
p

 
� "�	��
(__inference_ode_lstm_layer_call_fn_59611w	 !"X�U
N�K
A�>
�
input_1�d
�
input_2�d
p 

 
� "�	��
(__inference_ode_lstm_layer_call_fn_60283y	 !"Z�W
P�M
C�@
�
inputs/0�d
�
inputs/1�d
p

 
� "�	��
(__inference_ode_lstm_layer_call_fn_60307y	 !"Z�W
P�M
C�@
�
inputs/0�d
�
inputs/1�d
p 

 
� "�	��
B__inference_odelstm_layer_call_and_return_conditional_losses_61705� !"|�y
r�o
 �
inputs���������
K�H
"�
states/0���������@
"�
states/1���������@
� "s�p
i�f
�
0/0���������@
E�B
�
0/1/0���������@
�
0/1/1���������@
� �
'__inference_odelstm_layer_call_fn_61730� !"|�y
r�o
 �
inputs���������
K�H
"�
states/0���������@
"�
states/1���������@
� "c�`
�
0���������@
A�>
�
1/0���������@
�
1/1���������@�
>__inference_rnn_layer_call_and_return_conditional_losses_60607� !"S�P
I�F
4�1
/�,
inputs/0������������������

 
p

 

 
� "%�"
�
0���������@
� �
>__inference_rnn_layer_call_and_return_conditional_losses_60907� !"S�P
I�F
4�1
/�,
inputs/0������������������

 
p 

 

 
� "%�"
�
0���������@
� �
>__inference_rnn_layer_call_and_return_conditional_losses_61245e !";�8
1�.
�
inputs�d

 
p

 

 
� "�
�
0	�@
� �
>__inference_rnn_layer_call_and_return_conditional_losses_61545e !";�8
1�.
�
inputs�d

 
p 

 

 
� "�
�
0	�@
� �
#__inference_rnn_layer_call_fn_60926x !"S�P
I�F
4�1
/�,
inputs/0������������������

 
p

 

 
� "����������@�
#__inference_rnn_layer_call_fn_60945x !"S�P
I�F
4�1
/�,
inputs/0������������������

 
p 

 

 
� "����������@
#__inference_rnn_layer_call_fn_61564X !";�8
1�.
�
inputs�d

 
p

 

 
� "�	�@
#__inference_rnn_layer_call_fn_61583X !";�8
1�.
�
inputs�d

 
p 

 

 
� "�	�@�
#__inference_signature_wrapper_59645�	 !"a�^
� 
W�T
(
input_1�
input_1�d
(
input_2�
input_2�d"%�"
 
dense�
dense	�