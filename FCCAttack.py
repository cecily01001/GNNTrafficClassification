import tensorflow as tf
from keras import backend as K
from keras.layers import Input
import time
from keras.models import load_model
import numpy as np
class Crafted_Dummy_Packet():
    def __init__(self,json_path=None,keras_model_path=None,class_number = None, batch_size = None,flow_seq_size = None):
        self.keras_model_path = keras_model_path
        self.model_inf_json = json_path
        self.class_number = class_number
        self.graph = tf.Graph()
        #self.sess = tf.Session(graph=self.graph)
        self.sess = tf.compat.v1.Session(graph=self.graph)

        #self.sess = tf.Session(graph=self.graph)
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.batch_size = batch_size
        self.flow_seq_size = flow_seq_size
        self.load_advpay_operations()


    def make_advpay(self,data_x=None,data_y=None, lr = 0.001, train_steps = 1000,advpay_size=None,start_index_vector=None):
        start_time = time.time()
        self.reset_advpay()

        for i in range(train_steps):
            if i % 100 == 0:
                print('Step:',i)
            data_index = np.random.randint(len(data_x), size=self.batch_size)
            batch_x = data_x[data_index]
            batch_y = data_y[data_index]
            start_index = start_index_vector[data_index]
            self.train_step(batch_x,batch_y,start_index=start_index, lr=lr,advpay_size=advpay_size)
        elapsed_time = time.time() - start_time
        print("Finished training Advpay, took {:.0f}s".format(elapsed_time))
        return self.advpay()

    def train_step(self,train_data,src_label,start_index, lr=0.001,advpay_size=None):
        feed_dict = {self.flow_seq_input: np.asarray(train_data).reshape([self.batch_size,self.flow_seq_size,1]),
                     self._src_ys: src_label,
                     self._learning_rate: lr,
                     self.start_loc_of_advpay_placeholder: start_index,
                     self.learning_phase: False,
                     self.dropout: 1.0,
                     self.advpay_length: advpay_size}
        assert np.sum(np.array([i[j[0]:j[0]+int(advpay_size)]for i,j in zip(train_data,start_index)])) == 0, "Bad packet to add advpay"
        _,fmask,advmask,grad,loss,new_advpay,prob,pi = self.sess.run([self._train_op,self.flow_seq_mask,self.advpay_mask,self._grad_opt,self._loss,self._clipped_advpay,self._probabilities,self._advpayed_input], feed_dict)
        assert np.sum( new_advpay < 0 ) == 0, "Bad Advpay"


    def advpay(self, new_advpay=None):
        if new_advpay is None:
            return self.sess.run(self._clipped_advpay, {self.learning_phase: False})

        self.sess.run(self._assign_advpay, {self._advpay_placeholder: new_advpay, self.learning_phase: False})
        return self


    def reset_advpay(self):
        self.advpay(np.zeros((self.flow_seq_size,1)))

    def load_advpay_operations(self):
        start = time.time()
        #K.set_session(self.sess)

        tf.compat.v1.keras.backend.set_session(self.sess)

        with self.sess.graph.as_default():
            keras_model = flow_classification_keras_model(self.model_inf_json,self.keras_model_path,input_length=self.flow_seq_size,num_classes=self.class_number)
            self.learning_phase = K.learning_phase() # if 1 -> train. if 0 -> test. (must be set in test or train time )
            Trace_shape = (self.flow_seq_size,1)
            self.flow_seq_input = Input(shape=Trace_shape) # input (set in test or train time )
            Trace_input = self.flow_seq_input
            self.Trace_advpay = tf.Variable(initial_value=tf.zeros((self.flow_seq_size,1)), dtype=tf.float32,name="advpay")
            '''self._advpay_placeholder = tf.placeholder(dtype=tf.float32, shape=(self.flow_seq_size,1),name='advpay_placeholder')
            self.start_loc_of_advpay_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.batch_size,1],name='start_location_placeholder')
            self.advpay_length = tf.placeholder(dtype=tf.int32, shape=(),name='advpay_length')'''
            self._advpay_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=(self.flow_seq_size,1),name='advpay_placeholder')
            self.start_loc_of_advpay_placeholder = tf.compat.v1.placeholder(dtype=tf.int32, shape=[self.batch_size,1],name='start_location_placeholder')
            self.advpay_length = tf.compat.v1.placeholder(dtype=tf.int32, shape=(),name='advpay_length')

            #self._assign_advpay = tf.assign(self.Trace_advpay, self._advpay_placeholder) # assign _advpay_placeholder to advpay
            self._assign_advpay = tf.compat.v1.assign(self.Trace_advpay, self._advpay_placeholder) # assign _advpay_placeholder to advpay
            self.modified_advpay = self.Trace_advpay


            def clip_to_valid_advpay(x):
                return tf.clip_by_value(x, clip_value_min=0., clip_value_max=1.) # client to server packet is positive


            self._clipped_advpay = clip_to_valid_advpay(
                self.modified_advpay)  # after making advpay, use this variable to get path

            #self.dropout = tf.placeholder_with_default(1.0, [])
            self.dropout = tf.compat.v1.placeholder_with_default(1.0, [])
            advpay_with_dropout = tf.compat.v1.nn.dropout(self._clipped_advpay, keep_prob=self.dropout)
            advpayed_input = self.add_advpay_to_flow_seq(Trace_input, advpay_with_dropout,self.start_loc_of_advpay_placeholder,self.advpay_length)

            self._advpayed_input = advpayed_input

            #self._src_ys = tf.placeholder(tf.float32, shape=(None, self.class_number[0]), name="target_ys")
            self._src_ys = tf.compat.v1.placeholder(tf.float32, shape=(None, self.class_number[0]), name="target_ys")
            self.logits = keras_model.predict([advpayed_input])

            self._loss_per_example = tf.nn.softmax_cross_entropy_with_logits(
                labels=self._src_ys,
                logits=self.logits
            )
            self._loss = tf.reduce_mean(self._loss_per_example) * -1 # it increases the loss of src class

            # Train our attack by only training on the advpay variable
            #self._learning_rate = tf.placeholder(tf.float32,name='learning_rate')
            self._learning_rate = tf.compat.v1.placeholder(tf.float32,name='learning_rate')
            self._optimizer = tf.compat.v1.train.RMSPropOptimizer(self._learning_rate)
            self._grad_opt = self._optimizer.compute_gradients(self._loss,var_list=[self.Trace_advpay])
            self._train_op = self._optimizer.apply_gradients(self._grad_opt)

            self._probabilities = tf.compat.v1.nn.softmax(logits=self.logits)
            elapsed = time.time() - start
            print("Finished loading , took {:.0f}s".format( elapsed))
            self.sess.run(tf.compat.v1.variables_initializer(self._optimizer.variables()))

    def add_advpay_to_flow_seq(self, data, advpay,start_advpay_index,advpay_length):

        stacked_advpay = tf.stack([advpay] * self.batch_size)
        self.clean_data_seq = data

        flow_seq_mask = []
        def _mask_vecs(start,advpay_size):
            flow_mask = np.zeros([self.flow_seq_size, 1])
            flow_mask[:start[0]] = 1
            flow_mask[start[0]+advpay_size:self.flow_seq_size] = 1
            return flow_mask.astype(np.float32)

        for i in range(self.batch_size):
            flow_seq_mask_vec = tf.compat.v1.py_func(_mask_vecs, [start_advpay_index[i],advpay_length], tf.float32)
            flow_seq_mask_vec.set_shape([self.flow_seq_size, 1])
            flow_seq_mask.append(flow_seq_mask_vec)
        self.flow_seq_mask = tf.stack(flow_seq_mask)

        shifted_stacked_advpay = []
        for i in range(self.batch_size):
            shifted_stacked_advpay_vec = tf.roll(stacked_advpay[i], start_advpay_index[i][0], axis=0)
            shifted_stacked_advpay.append(shifted_stacked_advpay_vec)
        self.shifted_stacked_advpay_array = tf.stack(shifted_stacked_advpay)

        self.advpay_mask = 1 - self.flow_seq_mask
        self.advpayed_flow_seq = self.clean_data_seq * self.flow_seq_mask + self.shifted_stacked_advpay_array * self.advpay_mask
        return self.advpayed_flow_seq

    def inference_batch(self, test_x, test_y,start_index_vec=0, advpay_size=None):
        test_x = np.expand_dims(np.array(test_x), axis=2)
        Evade = 0
        total = 0
        pred = []
        for i in range(int(np.ceil(len(test_x) / self.batch_size))):
            batch_x = test_x[i * self.batch_size:min(len(test_x), (i + 1) * self.batch_size)]
            batch_true = test_y[i * self.batch_size:min(len(test_x), (i + 1) * self.batch_size)]
            start_index = start_index_vec[i * self.batch_size:min(len(test_x), (i + 1) * self.batch_size)]
            orginal_len_batch = len(batch_x)
            while len(batch_x) != self.batch_size:
                dummy_data_num = self.batch_size - len(batch_x)
                batch_x = np.append(batch_x, batch_x[:dummy_data_num], axis=0)
                batch_true = np.append(batch_true, batch_true[:dummy_data_num], axis=0)
                start_index = np.append(start_index, start_index[:dummy_data_num], axis=0)

            feed_dict = {self.flow_seq_input: batch_x, self._src_ys: batch_true,
                         self.start_loc_of_advpay_placeholder: start_index,
                         self.learning_phase: False,
                         self.advpay_length: advpay_size}
            assert np.sum(np.array([i[j[0]:j[0] + int(advpay_size)] for i, j in
                                    zip(batch_x, start_index)])) == 0, "Bad packet to add advpay"

            ps, advpayed_input = self.sess.run([self._probabilities, self._advpayed_input], feed_dict)
            ps = np.array(ps)[:orginal_len_batch]
            if i == 0:
                pred = ps
            else:
                pred = np.append(pred, ps, axis=0)

            for j in range(len(ps)):
                total += 1
                if np.argmax(ps[j]) != np.argmax(batch_true[j]):
                    Evade += 1
        if total == 0:
            total += 1
        return np.round((Evade / total) * 100, 2),total, pred


import json
from keras.models import Sequential, load_model
from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.advanced_activations import ELU
from keras.initializers import glorot_uniform, glorot_normal, he_normal

class flow_classification_keras_model:
    def __init__(self,model_inf_json_path,load_path=None,input_length=None,num_classes=None):
        with open(model_inf_json_path) as json_file:
            models = json.load(json_file)
        model_info = models[0]
        if model_info['Num of blocks'] != 0:
            input_shape = (input_length, 1)
        else:
            input_shape = (input_length,)
        print(input_shape)
        model = Sequential()

        for i in range(model_info['Num of blocks']):
            for j in range(model_info['Num of conv layer in block']):
                conv_index = i * model_info['Num of conv layer in block'] + j
                if conv_index == 0:
                    model.add(Conv1D(filters=model_info['Conv_Filter_num'][conv_index],
                                     kernel_size=model_info['Conv_Kernel_size'][conv_index],
                                     input_shape=input_shape,
                                     strides=model_info['Conv_stride_size'][conv_index],
                                     padding=model_info['Conv padding'],
                                     name='block' + str(i + 1) + '_conv' + str(j + 1)))
                else:
                    model.add(Conv1D(filters=model_info['Conv_Filter_num'][conv_index],
                                     kernel_size=model_info['Conv_Kernel_size'][conv_index],
                                     strides=model_info['Conv_stride_size'][conv_index],
                                     padding=model_info['Conv padding'],
                                     name='block' + str(i + 1) + '_conv' + str(j + 1)))
                if model_info['Conv_Batch normalization'] == 'yes' and model_info[
                    'Conv_Batch normalization place'] == 'before activation':
                    model.add(BatchNormalization(axis=-1, name='block' + str(i + 1) + '_BN' + str(j + 1)))
                if model_info['Conv_Activations'] == 'relu':
                    model.add(Activation('relu', name='block' + str(i + 1) + '_act' + str(j + 1)))
                if model_info['Conv_Activations'] == 'elu':
                    model.add(ELU(alpha=1.0, name='block' + str(i + 1) + '_act' + str(j + 1)))
                if model_info['Conv_Batch normalization'] == 'yes' and model_info[
                    'Conv_Batch normalization place'] == 'after activation':
                    model.add(BatchNormalization(axis=-1, name='block' + str(i + 1) + '_BN' + str(j + 1)))
                if model_info['Conv_Dropout'] == 'yes':
                    if model_info['Conv_Dropout rate'][conv_index] >= 1.:
                        print("Bad Dropout rate", model_info['Conv_Dropout rate'])
                        quit()
                    model.add(Dropout(model_info['Conv_Dropout rate'][conv_index],
                                      name='block' + str(i + 1) + '_dropout' + str(j + 1)))
            if model_info['Conv_MaxPooling1D'] == 'yes':
                model.add(
                    MaxPooling1D(pool_size=model_info['Pool_size'][i], strides=model_info['Pool_stride_size'][i],
                                 padding=model_info['Conv_MaxPooling1D padding'],
                                 name='block' + str(i + 1) + '_pool'))
        if model_info['Num of blocks'] != 0:
            model.add(Flatten(name='flatten'))
        for i in range(model_info['Num of fully connected layers']):
            if model_info['Num of blocks'] == 0:
                model.add(
                    Dense(model_info['FC_Num of neurons in hidden layers'][i], kernel_initializer=he_normal(),
                          input_shape=input_shape, name='fc' + str(i + 1)))
            else:
                model.add(
                    Dense(model_info['FC_Num of neurons in hidden layers'][i], kernel_initializer=he_normal(),
                          name='fc' + str(i + 1)))
            if model_info['FC_Batch normalization'] == 'yes' and model_info[
                'FC_Batch normalization place'] == 'before activation':
                model.add(BatchNormalization(name='fc' + str(i + 1) + '_BN'))
            if model_info['FC_Activations'] == 'relu':
                model.add(Activation('relu', name='fc' + str(i + 1) + '_act'))
            if model_info['FC_Activations'] == 'elu':
                model.add(ELU(alpha=1.0, name='fc' + str(i + 1) + '_act'))
            if model_info['FC_Batch normalization'] == 'yes' and model_info[
                'FC_Batch normalization place'] == 'after activation':
                model.add(BatchNormalization(name='fc' + str(i + 1) + '_BN'))
            if model_info['FC_Dropout'] == 'yes':
                if model_info['FC_Dropout rate'][i] >= 1.:
                    print("Bad Dropout rate", model_info['FC_Dropout rate'])
                    quit()
                model.add(Dropout(model_info['FC_Dropout rate'][i], name='fc' + str(i + 1) + '_dropout'))
        model.add(Dense(num_classes[0], kernel_initializer=he_normal(), name='fc' + str(i + 2)))

        model.load_weights(load_path)

        self.model = model


    def predict(self, data):
        return self.model(data)


def tune_sequence_number(data,information,advpay_pkt_index_vector=None,advpay_size=None,refrence_pkt_size_vector=None):
    valid_data = []
    for (seq, inf,advpay_pkt_index,ref_pkt_size) in zip(data, information, advpay_pkt_index_vector,refrence_pkt_size_vector):
        if inf[4] == '6':
            for i in range(MAX_NUM_PKT_IN_FLOW_SEQ):
                if i < advpay_pkt_index:
                    continue
                pkt_seq = seq[i*pkt_size:(i+1)*pkt_size]
                hex_seq_no = pkt_seq[4:8]
                if hex_seq_no[0] + hex_seq_no[1] + hex_seq_no[2] + hex_seq_no[3] == 0:
                    continue

                #print(hex_seq_no)
                sign = np.sign(hex_seq_no[0] + hex_seq_no[1] + hex_seq_no[2] + hex_seq_no[3])
                assert len(''.join('{:02X}'.format(np.abs(int(i))) for i in hex_seq_no)) == 8,"very bad"

                seq_no = int(''.join('{:02X}'.format(np.abs(int(i))) for i in hex_seq_no),16)
                #print(seq_no)
                if i == advpay_pkt_index:
                    new_dec_seq_no = np.mod(seq_no + ref_pkt_size,pow(2,32)-1)
                    assert np.sum(pkt_seq[TL_LENGTH:]) == 0, "bad change in header of real packet"
                else:
                    new_dec_seq_no = np.mod(seq_no + advpay_size, pow(2, 32) - 1)
                new_hex = hex(int(new_dec_seq_no)).lstrip("0x")
                if len(new_hex) < 8:
                    new_hex = '0'*(8 - len(new_hex)) + new_hex
                #print(new_hex)
                new_hex_seq_no = [int(j,16) * sign for j in [new_hex[i*2:(i+1)*2] for i in range(4)]]
                #print(new_hex_seq_no)
                new_seq_no = int(''.join('{:02X}'.format(np.abs(int(i))) for i in new_hex_seq_no),16) * sign
                #print(''.join('{:02X}'.format(np.abs(int(i))) for i in new_hex_seq_no))
                #print("new_seq_no",new_seq_no)
                pkt_seq[4:8] = new_hex_seq_no
        valid_data.append(seq)
    return np.array(valid_data)

def make_dummy_pkt_header(data,information,advpay_pkt_index_vector=None,refrence_pkt_header_index=None,advpay_size=None):
    valid_data = []
    for (seq, inf, advpay_pkt_index,ref_hdr_index) in zip(data, information, advpay_pkt_index_vector,refrence_pkt_header_index):
        advpay_pkt_index = advpay_pkt_index[0]
        ref_hdr_index = ref_hdr_index[0]
        assert advpay_pkt_index != ref_hdr_index,"advpay_pkt_index == ref_hdr_index"
        first_part = seq[0:advpay_pkt_index*pkt_size]
        second_part = seq[advpay_pkt_index*pkt_size:]
        seq = np.zeros(seq.shape)
        seq[:advpay_pkt_index*pkt_size] = first_part
        seq[(advpay_pkt_index+1)*pkt_size:] = second_part[:NUM_OF_BYTE_IN_FLOW_SEQUENCE-(advpay_pkt_index+1)*pkt_size]
        seq[advpay_pkt_index*pkt_size:(advpay_pkt_index*pkt_size)+TL_LENGTH] = seq[ref_hdr_index*pkt_size:(ref_hdr_index*pkt_size)+TL_LENGTH]
        if inf[4] == '6':
            seq[(advpay_pkt_index * pkt_size) + 8 :(advpay_pkt_index * pkt_size) + 14] = 0  # All flags set zero
            seq[(advpay_pkt_index * pkt_size) + 18:(advpay_pkt_index * pkt_size) + TL_LENGTH] = 0  # All flags set zero
        elif inf[4] == '17':
            hex_advpay_size = np.array(['{:04X}'.format(np.abs(int(advpay_size)))[:2], '{:04X}'.format(np.abs(int(advpay_size)))[2:]])
            seq[(advpay_pkt_index * pkt_size) + 4:(advpay_pkt_index * pkt_size) + 6] = np.array(
                [int(hex_advpay_size[0], 16), int(hex_advpay_size[1], 16)])
            seq[(advpay_pkt_index * pkt_size) + 8:(advpay_pkt_index * pkt_size) + TL_LENGTH] = 0
        valid_data.append(seq)
    return np.array(valid_data)

def make_room_for_advpay(data,advpay_pkt_index_vector=None):
    valid_data = []
    for (seq,advpay_pkt_index) in zip(data,advpay_pkt_index_vector):
        advpay_pkt_index = advpay_pkt_index[0]
        first_part = seq[:advpay_pkt_index*pkt_size]
        second_part = seq[advpay_pkt_index*pkt_size:]
        seq = np.zeros(seq.shape)
        seq[:advpay_pkt_index*pkt_size] = first_part
        seq[(advpay_pkt_index+1)*pkt_size:] = second_part[:NUM_OF_BYTE_IN_FLOW_SEQUENCE-(advpay_pkt_index+1)*pkt_size]
        valid_data.append(seq)
    return np.array(valid_data)

def prepare_advpay_attack_data(raw_data=None,advpay_pkt_index_vector=None, information=None,datasize=None,advpay_len=None):
    refrence_pkt_header_index = advpay_pkt_index_vector - 1
    assert np.sum(advpay_pkt_index_vector[advpay_pkt_index_vector == 11]) == 0, "Bad advpay index"
    refrence_pkt_size_vector = np.array(
        [datasize[i][int(refrence_pkt_header_index[i])] for i in range(len(raw_data))])

    manipulated_data = make_dummy_pkt_header(data=raw_data, information=information,
                                                     advpay_pkt_index_vector=advpay_pkt_index_vector,
                                                     refrence_pkt_header_index=refrence_pkt_header_index,
                                                     advpay_size=advpay_len)
    manipulated_data= tune_sequence_number(data=manipulated_data,
                                                    information=information,
                                                    advpay_pkt_index_vector=advpay_pkt_index_vector,
                                                    advpay_size=advpay_len,
                                                    refrence_pkt_size_vector=refrence_pkt_size_vector)
    return manipulated_data

def health_check(attack_data=None,clean_data=None,start_index_vec=None, information=None):
    assert np.sum([np.sum(k) != 0 for k in np.array([i[:j[0] - (TL_LENGTH * ('TL' in INPUT_ITEMS))] for i, j in
                                                     zip((attack_data - clean_data),
                                                         start_index_vec)])]) == 0, "Packets before advpay are wrong!!! (packet before advpas must have no change)"
    if 'TL' in INPUT_ITEMS:
        for i,j,k,l in zip(attack_data, clean_data, start_index_vec,information):
            if l[4] == '6':
                 assert (np.unique(np.where(
                    i[k[0] - TL_LENGTH + pkt_size:] - j[k[0] - TL_LENGTH:NUM_OF_BYTE_IN_FLOW_SEQUENCE - pkt_size] != 0)[
                              0] % pkt_size) == np.intersect1d(np.where(
                    i[k[0] - TL_LENGTH + pkt_size:] - j[k[0] - TL_LENGTH:NUM_OF_BYTE_IN_FLOW_SEQUENCE - pkt_size] != 0)[
                                                                   0] % pkt_size, [4, 5, 6, 7]))[0], "Tcp packet only change in seq number (4,5,6,7 index in TCP header)"
            elif l[4] == '17':
                assert np.sum(i[k[0] - TL_LENGTH + pkt_size:] - j[k[0] - TL_LENGTH:NUM_OF_BYTE_IN_FLOW_SEQUENCE - pkt_size]) == 0, "UDP packet after advpay must have no change"
    else:
        assert np.sum([np.sum(z) for z in
                       [i[k[0] + pkt_size:] != j[k[0]:NUM_OF_BYTE_IN_FLOW_SEQUENCE - pkt_size] for i, j, k in
                        zip(attack_data, clean_data,
                            start_index_vec)]]) == 0, "Packets after advpay are wrong!!!"
    if 'TL' in INPUT_ITEMS:
        assert np.sum([np.in1d( ii,[0,1,2,3]).any()for ii in [np.where( i[k[0]-TL_LENGTH:k[0]] - j[k[0]-TL_LENGTH - pkt_size:k[0]-pkt_size ] != 0)[0] for i,j,k in zip(attack_data,clean_data,start_index_vector)]]) == 0,"Advpey port differernt from last packet before advpay"
        #assert np.sum([jj == 0 for jj in [ np.sum(ii) for ii in [i[k[0]-TL_LENGTH:k[0]][4:(l[4] == '6') * 8 + (l[4] == '17') * 6] != j[k[0]-TL_LENGTH - pkt_size:k[0]-pkt_size][4:(l[4] == '6') * 8 + (l[4] == '17') * 6] for i,j,k,l in zip(attack_data,clean_data,start_index_vector,information)]]]) == 0, "Advpay sequnce number in TCP or length in UDP is not different from last packet before advpay"

