from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import numpy as np
import time
from physics_engine import gen, make_video
FLAGS = None

def m(O,Rr,Rs,Ra):
  return tf.concat([tf.matmul(O,Rr),tf.matmul(O,Rs),Ra],1);

  
def phi_R(B):
  h_size=150;
  B_trans=tf.transpose(B,[0,2,1]);
  B_trans=tf.reshape(B_trans,[-1,(2*FLAGS.Ds+FLAGS.Dr)]);
  w1 = tf.Variable(tf.truncated_normal([(2*FLAGS.Ds+FLAGS.Dr), h_size], stddev=0.1), name="r_w1", dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([h_size]), name="r_b1", dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1);
  w2 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w2", dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([h_size]), name="r_b2", dtype=tf.float32);
  h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
  w3 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w3", dtype=tf.float32);
  b3 = tf.Variable(tf.zeros([h_size]), name="r_b3", dtype=tf.float32);
  h3 = tf.nn.relu(tf.matmul(h2, w3) + b3);
  w4 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w4", dtype=tf.float32);
  b4 = tf.Variable(tf.zeros([h_size]), name="r_b4", dtype=tf.float32);
  h4 = tf.nn.relu(tf.matmul(h3, w4) + b4);
  w5 = tf.Variable(tf.truncated_normal([h_size, FLAGS.De], stddev=0.1), name="r_w5", dtype=tf.float32);
  b5 = tf.Variable(tf.zeros([FLAGS.De]), name="r_b5", dtype=tf.float32);
  h4 = tf.matmul(h4, w5) + b5;
  h4_trans=tf.reshape(h4,[-1,FLAGS.Nr,FLAGS.De]);
  h4_trans=tf.transpose(h4_trans,[0,2,1]);
  return(h4_trans);

def a(O,Rr,X,E):
  E_bar=tf.matmul(E,tf.transpose(Rr,[0,2,1]));
  return (tf.concat([O,X,E_bar],1));

def phi_O(C):
  h_size=100;
  C_trans=tf.transpose(C,[0,2,1]);
  C_trans=tf.reshape(C_trans,[-1,(FLAGS.Ds+FLAGS.Dx+FLAGS.De)]);
  w1 = tf.Variable(tf.truncated_normal([(FLAGS.Ds+FLAGS.Dx+FLAGS.De), h_size], stddev=0.1), name="o_w1", dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([h_size]), name="o_b1", dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1);
  w2 = tf.Variable(tf.truncated_normal([h_size, FLAGS.Dp], stddev=0.1), name="o_w1", dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([FLAGS.Dp]), name="o_b1", dtype=tf.float32);
  h2 = tf.matmul(h1, w2) + b2;
  h2_trans=tf.reshape(h2,[-1,FLAGS.No,FLAGS.Dp]);
  h2_trans=tf.transpose(h2_trans,[0,2,1]);
  return(h2_trans);

def phi_A(P):
  h_size=25;
  p_bar=tf.reduce_sum(P,2);
  w1 = tf.Variable(tf.truncated_normal([FLAGS.Dp, h_size], stddev=0.1), name="a_w1", dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([h_size]), name="a_b1", dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul(p_bar, w1) + b1);
  w2 = tf.Variable(tf.truncated_normal([h_size, FLAGS.Da], stddev=0.1), name="a_w2", dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([FLAGS.Da]), name="a_b2", dtype=tf.float32);
  h2 = tf.matmul(h1, w2) + b2;
  return(h2);
  
def train():

  # Object Matrix
  O = tf.placeholder(tf.float32, [None,FLAGS.Ds,FLAGS.No], name="O");
  # Relation Matrics R=<Rr,Rs,Ra>
  Rr = tf.placeholder(tf.float32, [None,FLAGS.No,FLAGS.Nr], name="Rr");
  Rs = tf.placeholder(tf.float32, [None,FLAGS.No,FLAGS.Nr], name="Rs");
  Ra = tf.placeholder(tf.float32, [None,FLAGS.Dr,FLAGS.Nr], name="Ra");
  # next velocities
  P_label = tf.placeholder(tf.float32, [None,FLAGS.Dp,FLAGS.No], name="P_label");
  # External Effects
  X = tf.placeholder(tf.float32, [None,FLAGS.Dx,FLAGS.No], name="X");
  
  # marshalling function, m(G)=B, G=<O,R>  
  B=m(O,Rr,Rs,Ra);
  
  # relational modeling phi_R(B)=E
  E=phi_R(B);
  
  # aggregator
  C=a(O,Rr,X,E);
  
  # object modeling phi_O(C)=P
  P=phi_O(C);
  
  # abstract modeling phi_A(P)=q
  #q=phi_A(P);

  # loss and optimizer
  params_list=tf.global_variables();
  loss = tf.nn.l2_loss(P-P_label)+0.001*tf.nn.l2_loss(E);
  for i in params_list:
    loss+=0.001*tf.nn.l2_loss(i);
  optimizer = tf.train.AdamOptimizer(0.001);
  trainer=optimizer.minimize(loss);

  sess=tf.InteractiveSession();
  tf.global_variables_initializer().run();

  # Data Generation
  #set_num=2000;
  set_num=1;
  total_data=np.zeros((999*set_num,FLAGS.Ds,FLAGS.No),dtype=object);
  total_label=np.zeros((999*set_num,FLAGS.Dp,FLAGS.No),dtype=object);
  for i in range(set_num):
    raw_data=gen(FLAGS.No,True);
    data=np.zeros((999,FLAGS.Ds,FLAGS.No),dtype=object);
    label=np.zeros((999,FLAGS.Dp,FLAGS.No),dtype=object);
    for j in range(1000-1):
      data[j]=np.transpose(raw_data[j]);label[j]=np.transpose(raw_data[j,:,3:5]);
    total_data[i*999:(i+1)*999,:]=data;
    total_label[i*999:(i+1)*999,:]=label;

  # Shuffle
  #tr_data_num=1000000;
  #val_data_num=200000;
  tr_data_num=400;
  val_data_num=300;
  total_idx=range(len(total_data));np.random.shuffle(total_idx);
  mixed_data=total_data[total_idx];
  mixed_label=total_label[total_idx];
  # Training/Validation/Test
  train_data=mixed_data[:tr_data_num];
  train_label=mixed_label[:tr_data_num];
  val_data=mixed_data[tr_data_num:tr_data_num+val_data_num];
  val_label=mixed_label[tr_data_num:tr_data_num+val_data_num];
  test_data=mixed_data[tr_data_num+val_data_num:];
  test_label=mixed_label[tr_data_num+val_data_num:];
 
  mini_batch_num=100;
  # Set Rr_data, Rs_data, Ra_data and X_data
  Rr_data=np.zeros((mini_batch_num,FLAGS.No,FLAGS.Nr),dtype=float);
  Rs_data=np.zeros((mini_batch_num,FLAGS.No,FLAGS.Nr),dtype=float);
  Ra_data=np.zeros((mini_batch_num,FLAGS.Dr,FLAGS.Nr),dtype=float); 
  X_data=np.zeros((mini_batch_num,FLAGS.Dx,FLAGS.No),dtype=float); 
  cnt=0;
  for i in range(FLAGS.No):
    for j in range(FLAGS.No):
      if(i!=j):
        Rr_data[:,i,cnt]=1.0;
        Rs_data[:,j,cnt]=1.0;
        cnt+=1;

  # Training
  max_epoches=2000;
  for i in range(max_epoches):
    for j in range(int(len(train_data)/mini_batch_num)):
      batch_data=train_data[j*mini_batch_num:(j+1)*mini_batch_num];
      batch_label=train_label[j*mini_batch_num:(j+1)*mini_batch_num];
      sess.run(trainer,feed_dict={O:batch_data,Rr:Rr_data,Rs:Rs_data,Ra:Ra_data,P_label:batch_label,X:X_data});
    val_loss=0;
    for j in range(int(len(val_data)/mini_batch_num)):
      batch_data=val_data[j*mini_batch_num:(j+1)*mini_batch_num];
      batch_label=val_label[j*mini_batch_num:(j+1)*mini_batch_num];
      val_loss+=sess.run(loss,feed_dict={O:batch_data,Rr:Rr_data,Rs:Rs_data,Ra:Ra_data,P_label:batch_label,X:X_data});
    print("Epoch "+str(i+1)+" Validation MSE: "+str(val_loss/(j+1)));

  # Make Video
  frame_len=250;
  raw_data=gen(FLAGS.No,True);
  xy_origin=raw_data[:frame_len,:,1:3];
  estimated_data=np.zeros((frame_len,FLAGS.No,FLAGS.Ds),dtype=float);
  estimated_data[0]=raw_data[0];
  for i in range(1,frame_len):
    velocities=sess.run(P,feed_dict={O:[np.transpose(estimated_data[i-1])],Rr:[Rr_data[0]],Rs:[Rs_data[0]],Ra:[Ra_data[0]],X:[X_data[0]]})[0];
    estimated_data[i,:,0]=estimated_data[i-1][:,0];
    estimated_data[i,:,3:5]=np.transpose(velocities);
    estimated_data[i,:,1:3]=estimated_data[i-1,:,1:3]+estimated_data[i,:,3:5]*0.001;
  xy_estimated=estimated_data[:,:,1:3];
  print("Video Recording");
  make_video(xy_origin,"true.mp4");
  make_video(xy_estimated,"modeling.mp4");
  
      
def main(_):
  FLAGS.log_dir+=str(int(time.time()));
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir', type=str, default='/tmp/interaction-network/',
                      help='Summaries log directry')
  parser.add_argument('--Ds', type=int, default=5,
                      help='The State Dimention')
  parser.add_argument('--No', type=int, default=6,
                      help='The Number of Objects')
  parser.add_argument('--Nr', type=int, default=30,
                      help='The Number of Relations')
  parser.add_argument('--Dr', type=int, default=1,
                      help='The Relationship Dimension')
  parser.add_argument('--Dx', type=int, default=1,
                      help='The External Effect Dimension')
  parser.add_argument('--De', type=int, default=50,
                      help='The Effect Dimension')
  parser.add_argument('--Dp', type=int, default=2,
                      help='The Object Modeling Output Dimension')
  parser.add_argument('--Da', type=int, default=1,
                      help='The Abstract Modeling Output Dimension')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
