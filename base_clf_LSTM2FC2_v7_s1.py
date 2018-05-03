#base_clf_LSTM2FC2_v7_s1.py
from tensorflow.contrib.slim.python.slim.data import prefetch_queue
from datetime import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import shutil
import time
import os
import warnings
warnings.filterwarnings("ignore")

######################################################
#main.py
#학습에 필요한 요소들을 설정하는 부분
VERSION = 7

ATK_DIR_PATH = "D:/attack"
LABEL_FILE_PATH = "./data/label_list.csv"
SAMPLING_FILE_PATH = "./data/sampling1.csv"
SAVE_DIR = "./output/v{}_s1_output/".format(VERSION)
BOARD_PATH = './board/v{}_s1_board'.format(VERSION)

NCLASS = 7
INPUT_DIM = 3
NUM_RNN_UNITS = 200
NLAYERS =2
MODEL_NAME = 'LSTM_v{}_s1'.format(VERSION)
NUM_GPU_DEVICE = 0
NUM_CPU_DEVICE = 0

TOTAL_EPOCH = 201
BATCH_SIZE = 150
INIT_LEARNING_RATE = 0.001

#Freq, PW, PRI순 
col_min_list = [1381000, 100, 3905]
col_max_list = [18317000, 400100, 9758045]

######################################################
#load_data_dir.py
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
def search(dirname):
    filelist = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        filelist.append(full_filename.replace('\\', '/'))
    return filelist

print("=== 파일 분류를 시작합니다 ===")
with tf.device('/cpu:{}'.format(NUM_CPU_DEVICE)):
    data_classify_start_time = time.perf_counter()
    atk_dirs = search(ATK_DIR_PATH)
    atk_dirs = sorted(atk_dirs)

    trainX = []
    validationX = []
    test1X = []
    test2X = []
    atk_dir_num = len(atk_dirs)
    print("총 attack의 수 : ", atk_dir_num)

    try:
        np.loadtxt(SAMPLING_FILE_PATH,delimiter=',')
    except FileNotFoundError as e:
        print("Test2X에 대한 파일이 없습니다")
        print("=== 본 실험은 샘플링 없이 진행됩니다 ===")
        test_sample = []
    else:
        test_sample = np.loadtxt(SAMPLING_FILE_PATH, delimiter=',')

    count = 0  
    for atk_dir in atk_dirs:
        temp = int((atk_dir.split('/')[-1]).split('E')[-1])
        if temp in test_sample:
            beam_dirs = search(atk_dir)
            for beam_dir in beam_dirs:
                pdw_list = search(beam_dir)
                num_pdw = len(pdw_list)
                for j in range(num_pdw):
                    test2X.append(pdw_list[j])
        else:            
            beam_dirs = search(atk_dir)  
            for beam_dir in beam_dirs:
                random.seed(0)
                pdw_list = search(beam_dir)
                num_pdw = len(pdw_list)
                random.shuffle(pdw_list)
                ntrain = int(num_pdw * 0.7) # 트레이닝 파일 개수
                nvalidation = int(num_pdw * 0.1) # validation 파일 개수
                ntest1 = int(num_pdw - ntrain - nvalidation) # 테스트 파일 개수
                trainX.extend(pds_list[:ntrain])
                validationX.extend(pdw_list[ntrain:ntrain+nvalidation])
                test1X.extend(pdw_list[-ntest1:])
        print("[{:5d}/{:5d}] ".format(count+1, atk_dir_num))
        count+=1
        
    # 각각의 데이터 셋들의 pdw경로명들을 저장해둔다. 이 작업은 복원을 위한 작업임.
    np.savetxt(SAVE_DIR+"trainX.csv",trainX, fmt='%s',delimiter=',')
    np.savetxt(SAVE_DIR+"validationX.csv",validationX, fmt='%s',delimiter=',')
    np.savetxt(SAVE_DIR+"test1X.csv",test1X, fmt='%s',delimiter=',')
    np.savetxt(SAVE_DIR+"test2X.csv",test2X, fmt='%s',delimiter=',')

    data_classify_end_time= time.perf_counter()
    data_classify_duration = data_classify_end_time - data_classify_start_time
    print("파일 분류 걸린 시간 : {:.6f}(s)".format(data_classify_duration))
    print("=== 파일 분류가 끝났습니다 ===\n")
    print("=== 분류 결과 ===")

    total_train_num = len(trainX)
    total_validation_num = len(validationX)
    total_test1X_num = len(test1X)
    total_test2X_num = len(test2X)

    print("The number of train samples : ",total_train_num)
    print("The number of validation samples : ",total_validation_num)
    print("The number of test1X samples : ",total_test1X_num)
    print("The number of test2X samples : ",total_test2X_num)

######################################################
#clf_utils.py
def MinMaxScaler(cur_col, col_min, col_max):
    numerator = tf.subtract(cur_col, col_min)
    denominator = tf.subtract(col_max, col_min)
    addEpsilon = tf.add(denominator, 1e-7)
    result = tf.divide(numerator, addEpsilon)
    return result

def read_whole_file_as_one_sample(filename_queue, table1, table2, col_min, col_max):
    header_line = 5 
    max_col = 15
    reader = tf.WholeFileReader()
    filename, value = reader.read(filename_queue)
    features = tf.string_split([value], delimiter='\n').values[header_line:]  
    features = tf.string_split(features, delimiter='\t').values   
    features = tf.string_split(features, delimiter='\r').values 
    features = tf.string_to_number(features)
    features = tf.reshape(features,[-1,max_col ])
    t_feature = tf.gather(features, [10, 12, 14], axis=1)
    t_feature = MinMaxScaler(t_feature, col_min, col_max)

    name = tf.string_split([filename],'/')
    emitter_string = name.values[-3]
    temp = tf.string_split([emitter_string],delimiter="E").values[0]
    emitter= tf.string_to_number(temp, out_type=tf.int32)

    train_label1 = table1.lookup(emitter_string)
    train_label1 = tf.subtract(train_label1,1)
    train_label1= tf.reshape(train_label1,[1])
    
    train_label2 = table2.lookup(emitter_string)
    train_label2 = tf.subtract(train_label2,1)
    train_label2= tf.reshape(train_label2,[1])
    return t_feature, train_label1, train_label2, emitter 

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def _last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    output_size = tf.shape(output)[2]
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant
    
######################################################
#nets.py
def lstm_cell(rnn_hidden_units):
    cell = tf.contrib.rnn.LSTMCell(num_units=rnn_hidden_units, initializer = tf.glorot_uniform_initializer(seed=0), state_is_tuple=True)
    return cell

def linear_layer(x, input_dim, output_dim, stddev, name):
    with tf.variable_scope(name):
        W = tf.get_variable('W', [input_dim, output_dim], initializer = tf.truncated_normal_initializer(seed=0, stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer = tf.constant_initializer(0.1))
        h = tf.add(tf.matmul(x, W), b, name = 'h')
    return h

def tanh_layer(x, input_dim, output_dim, stddev, name):
    with tf.variable_scope(name):
        W =tf.get_variable('W', [input_dim, output_dim], initializer = tf.truncated_normal_initializer(seed=0, stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer = tf.constant_initializer(0.1))
        h = tf.nn.tanh(tf.add(tf.matmul(x, W), b), name = 'h')
    return h
######################################################
class Model:
    def __init__(self,sess):
        self.sess = sess
        self.num_rnn_units = NUM_RNN_UNITS
        self.model_name = MODEL_NAME
        self.input_dim = INPUT_DIM
        self.nlayers = NLAYERS
        self.nclass = NCLASS
        self.board_path = BOARD_PATH
        self._build_model()
        
    def _build_model(self):
        with tf.device("/device:GPU:{}".format(NUM_GPU_DEVICE)):
            tf.set_random_seed(0)
            with tf.variable_scope("Inputs"):
                self.X = tf.placeholder(tf.float32, [None, None, self.input_dim], name='X')
                self.Y = tf.placeholder(tf.int32, [None, 1], name='Y')
                self.learning_rate = tf.placeholder(tf.float32,name = 'learning_rate')
                self.Y_one_hot = tf.reshape(tf.one_hot(self.Y, self.nclass), [-1, self.nclass], name = 'Y_one_hot')
                len_X= length(self.X)

            with tf.variable_scope(self.model_name):
                with tf.variable_scope("LSTMLayer"):
                    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.num_rnn_units) for _ in range(self.nlayers)])
                    outputs, _states = tf.nn.dynamic_rnn(stacked_lstm, self.X, dtype=tf.float32, sequence_length=len_X)
                    last = _last_relevant(outputs, len_X)

                h1 = tanh_layer(last, self.num_rnn_units, self.num_rnn_units*2, stddev=0.01, name = "FCLayer1")
                self.logits = linear_layer(h1, self.num_rnn_units*2, self.nclass, stddev=0.01, name = "FCLayer2")
                
                with tf.variable_scope("Softmax"):
                    self.hypothesis = tf.nn.softmax(self.logits, name='hypothesis')
                    
                with tf.variable_scope("Optimization"):
                    tmp = self.Y_one_hot * tf.log(self.hypothesis)
                    v = tf.reshape(tmp,[1,-1])
                    denominator =tf.cast(tf.multiply(tf.shape(tmp)[0], tf.shape(tmp)[1]), tf.float32)
                    self.cost = -tf.divide(tf.reshape(tf.matmul(v, tf.ones_like(v), transpose_b=True),[]), denominator, name='cost')
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

            with tf.variable_scope("Predictions"):
                topTwo = tf.nn.top_k(self.hypothesis, k=2)
                top1 = topTwo[1][:,0]
                top2 = topTwo[1][:,1]
                self.prediction1 = tf.cast(top1, tf.int64, name = 'prediction1')
                self.prediction2 = tf.cast(top2, tf.int64, name = 'prediction2')
                
            with tf.variable_scope("Accuracy"):
                self.correct_prediction = tf.equal(self.prediction1, tf.argmax(self.Y_one_hot, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')

            with tf.variable_scope("Scalars"):            
                self.avg_loss = tf.placeholder(tf.float32)
                self.avg_acc = tf.placeholder(tf.float32)
                self.loss_scalar = tf.summary.scalar('loss', self.avg_loss)
                self.acc_scalar = tf.summary.scalar('acc', self.avg_acc)
                self.merged = tf.summary.merge_all()
                
            if os.path.exists(self.board_path):
                shutil.rmtree(self.board_path)
            self.writer = tf.summary.FileWriter(self.board_path)
            self.writer.add_graph(self.sess.graph)
                                     
    def prediction(self, x_test):
        return self.sess.run([self.prediction1,self.prediction2], feed_dict = {self.X:x_test})

    def get_proba(self, x_test):
        return self.sess.run(self.hypothesis, feed_dict = {self.X:x_test})

    def get_logits(self, x_test):
        return self.sess.run(self.logits, feed_dict = {self.X:x_test})

    def train(self, x_train, y_train, u):
        return self.sess.run([self.accuracy, self.cost, self.optimizer], feed_dict={self.X: x_train, self.Y: y_train, self.learning_rate: u})

    def summary_log(self, avg_loss, avg_acc, epoch):
        s = self.sess.run(self.merged, feed_dict={self.avg_loss : avg_loss, self.avg_acc : avg_acc})
        self.writer.add_summary(s, global_step = epoch)

    def save(self, dirname):
        saver = tf.train.Saver()
        saver.save(self.sess,dirname)
######################################################
with tf.device('/cpu:{}'.format(NUM_CPU_DEVICE)):
    col_min = tf.constant(col_min_list, dtype=tf.float32 ,name = 'col_min')
    col_max = tf.constant(col_max_list, dtype=tf.float32, name = 'col_max')

    keys=[]
    labels1=[]
    labels2=[]
    for line in open(LABEL_FILE_PATH):
        fields = line.rstrip().split(',')
        keys.append(fields[0])
        label1 = int(fields[1])
        label2 = int(fields[2])
        labels1.append(label1)
        labels2.append(label2)

    with tf.variable_scope("HashTable"):
        keys = tf.constant(keys)
        labels1 = tf.constant(labels1)
        labels2 = tf.constant(labels2)
        table1 = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, labels1),-1,name = 'table1')
        table2 = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, labels2),-1,name = 'table2')

    with tf.variable_scope("TrainQueue"):
        tfilename_queue = tf.train.string_input_producer(trainX, shuffle=True,seed=0, num_epochs=None)
        t_feature, t_label1,t_label2, t_emitter = read_whole_file_as_one_sample(tfilename_queue,table1, table2, col_min,col_max)
        t_batchX, t_batchY1, t_batchY2, t_emitter =tf.train.batch([t_feature, t_label1, t_label2, t_emitter], batch_size=BATCH_SIZE,dynamic_pad=True)
        batch_queue = prefetch_queue.prefetch_queue([t_batchX, t_batchY1], capacity = 30, num_threads = 3, dynamic_pad=True)
        train_batch_x, train_batch_y = batch_queue.dequeue()
        
    with tf.variable_scope("ValidQueue"):
        v_batchsize = tf.placeholder(tf.int32)
        vfilename_queue = tf.train.string_input_producer(validationX, shuffle=False, seed=0, num_epochs=None)
        v_feature, v_label1, v_label2, v_emitter = read_whole_file_as_one_sample(vfilename_queue,table1, table2, col_min,col_max)
        v_batchX, v_batchY1, v_batchY2, v_emitter = tf.train.batch([v_feature, v_label1, v_label2, v_emitter], capacity = 30, num_threads = 3, batch_size=v_batchsize, allow_smaller_final_batch=True, dynamic_pad=True)

config=tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

m = Model(sess)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
sess.run(table1.init)
sess.run(table2.init)
        
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

total_step = int(len(trainX)/BATCH_SIZE)
print("=== Train 시작 ===")
print('total step : {}\n'.format(total_step))

u = INIT_LEARNING_RATE
for epoch in range(TOTAL_EPOCH): 
    loss_per_epoch = 0
    acc_per_epoch = 0
    epoch_start_time = time.perf_counter()

    for step in range(total_step):
        data_load_start_time = time.perf_counter()
        minibatchX, minibatchY = sess.run([train_batch_x, train_batch_y])
        data_load_end_time = time.perf_counter()
        data_load_duration = data_load_end_time-data_load_start_time
        
        step_start_time = time.perf_counter()
        a, c, _ = m.train(minibatchX, minibatchY, u)
        step_end_time = time.perf_counter()
        step_duration = step_end_time-step_start_time
        '''
        num_examples_per_step = BATCH_SIZE
        examples_per_sec = num_examples_per_step / step_duration
        sec_per_batch = step_duration 
        format_str = ('%s: step %d, Loss = %.6f Accuracy = %.2f%%, (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (datetime.now(), step, c, a, examples_per_sec, sec_per_batch))
        '''
        if step%10 == 0:
            print("Epoch : {:5d}  [{:5d}/{:5d}]  Loss = {:.6f}  Accuracy = {:.2%}, DataLoadingDuration : {:.6f}(s) TrainDuration : {:.6f}(s)".format(epoch, step, total_step, c, a, data_load_duration, step_duration)) 
        loss_per_epoch+=c
        acc_per_epoch +=a

    u = u*0.95
    loss_per_epoch = loss_per_epoch/total_step
    acc_per_epoch = acc_per_epoch/total_step
        
    m.summary_log(loss_per_epoch, acc_per_epoch , epoch)
    
    epoch_end_time = time.perf_counter()
    epoch_duration = epoch_end_time - epoch_start_time
    
    print("\nEpoch : {:5d} Loss : {:.6f}  Accuracy : {:.2%} EpochDuration : {:.6f}(s)".format(epoch,loss_per_epoch,acc_per_epoch,epoch_duration))
    print("Learning Rate : {:.6f}".format(u))
    
    with open(SAVE_DIR + "train_result.csv",'a') as f:
        f.write('Epoch,{},{:.9f},'.format(epoch,loss_per_epoch))
        f.write('{:.2%}\n'.format(acc_per_epoch))

    valid_start_time = time.perf_counter()
    if acc_per_epoch>0.95:
        valid_acc1 =0
        valid_acc2 = 0
        valid_loss =0
        emit = []
        trueth1 = []
        trueth2 = []

        residual = total_validation_num%BATCH_SIZE
        if residual ==0:
            valid_step = int(total_validation_num/BATCH_SIZE)
        else:
            valid_step = int(total_validation_num/BATCH_SIZE)+1
        for i in range(valid_step):
            if residual != 0 and i == (valid_step-1):
                minibatchX, minibatchY, minibatchY2, valid_emitter = sess.run([v_batchX, v_batchY1, v_batchY2, v_emitter],feed_dict = {v_batchsize:residual})
            else:
                minibatchX, minibatchY, minibatchY2, valid_emitter = sess.run([v_batchX, v_batchY1, v_batchY2, v_emitter],feed_dict = {v_batchsize:BATCH_SIZE})
                
            p1,p2 = m.prediction(minibatchX)
            temp = minibatchY
            cham1 = np.sum([p1==minibatchY[:,0]]*1)

            logic0 = (p1 != minibatchY[:,0])
            logic1 = (p1 == minibatchY2[:,0])
            logic2 = (p2 == minibatchY[:,0])
            logic3 = (p2 == minibatchY2[:,0])
            temp_logic = np.logical_or(logic1, logic2)
            logic = np.logical_or(temp_logic, logic3)
            logic = np.logical_and(logic0, logic)
            
            cham2 = np.sum(logic*1)
            valid_acc1 += cham1
            valid_acc2 += cham2

            a = valid_emitter
            b = (p1==temp[:,0])*1
            c = logic*1
            emit.extend(a)
            trueth1.extend(b)
            trueth2.extend(c)

        valid_acc1 = valid_acc1 / total_validation_num
        valid_acc2 = valid_acc2 / total_validation_num
        print('Epoch:', '%04d' % (epoch), 'validation_accuracy1 =', '{:.2%}'.format(valid_acc1))
        print('Epoch:', '%04d' % (epoch), 'validation_accuracy2 =', '{:.2%}'.format(valid_acc2))
        with open(SAVE_DIR+'validation_result.csv','a') as f:
            f.write('{:.2%}, {:.2%}\n'.format(valid_acc1,valid_acc2))

        # Step8 matrix 제조 시작
        basket = []
        basket.append(emit)
        basket.append(trueth1)
        basket.append(trueth2)
        result = np.transpose(basket)
        
        item = np.sort(np.unique(basket[0]))
        emitter_list = np.expand_dims(item,axis=0)
        emitter_list = np.transpose(emitter_list)
        
        cham11 = np.zeros_like(emitter_list,dtype=np.float32)
        cham12 = np.zeros_like(emitter_list,dtype=np.float32)
        totalE = np.zeros_like(emitter_list,dtype=np.float32)
        acc11 = np.zeros_like(emitter_list,dtype=np.float32)
        acc12 = np.zeros_like(emitter_list,dtype=np.float32)
        
        _matrix = np.concatenate((emitter_list, cham11, cham12,totalE,acc11,acc12),axis=1)
        for i in range(len(item)):
            s = item[i]
            emitter_idx = np.where(result[:,0]==s)[0]
            ch1 = 0
            ch2 = 0
            count = 0
            for j in emitter_idx:
                ch1 += result[j,1]
                ch2 += result[j,2]
                count +=1
            _matrix[i,1] = ch1
            _matrix[i,2] = ch2
            _matrix[i,3] = count
            _matrix[i,4] = ch1/count
            _matrix[i,5] = ch2/count
        np.savetxt(SAVE_DIR+"matrix{}.csv".format(epoch),_matrix,delimiter=',')
    valid_end_time = time.perf_counter()
    valid_duration = valid_end_time - valid_start_time
    print("<<< Validation Finished>>> , Duration : {:.6f}".format(valid_duration)) 

    m.save(SAVE_DIR+MODEL_NAME+"_{}/Model_Adam".format(epoch))
        
coord.request_stop()
coord.join(threads)



