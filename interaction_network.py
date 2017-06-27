from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import numpy as np
import time

FLAGS = None

def m(O,Rr,Rs,Ra):
  return tf.concat([tf.matmul(O,Rr),tf.matmul(O,Rs),Ra],0);

  
def phi_R(B):
  h_size=150;
  B_trans=tf.transpose(B);
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
  h4_trans=tf.transpose(h4);
  return(h4_trans);

def a(O,Rr,X,E):
  E_bar=tf.matmul(E,tf.transpose(Rr));
  return (tf.concat([O,X,E_bar],0));

def phi_O(C):
  h_size=100;
  C_trans=tf.transpose(C);
  w1 = tf.Variable(tf.truncated_normal([(FLAGS.Ds+FLAGS.Dx+FLAGS.De), h_size], stddev=0.1), name="o_w1", dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([h_size]), name="o_b1", dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1);
  w2 = tf.Variable(tf.truncated_normal([h_size, FLAGS.Dp], stddev=0.1), name="o_w1", dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([FLAGS.Dp]), name="o_b1", dtype=tf.float32);
  h2 = tf.matmul(h1, w2) + b2;
  h2_trans=tf.transpose(h2);
  return(h2_trans);

def phi_A(P):
  h_size=25;
  p_bar=tf.reduce_sum(P,1);
  w1 = tf.Variable(tf.truncated_normal([FLAGS.Dp, h_size], stddev=0.1), name="a_w1", dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([h_size]), name="a_b1", dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul([p_bar], w1) + b1);
  w2 = tf.Variable(tf.truncated_normal([h_size, FLAGS.Da], stddev=0.1), name="a_w2", dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([FLAGS.Da]), name="a_b2", dtype=tf.float32);
  h2 = tf.matmul(h1, w2) + b2;
  return(h1);
  
def train():
  """
  # Object Matrix
  O=np.zeros((FLAGS.Ds,FLAGS.No),dtype=float);
  # Relation Matrics R=<Rr,Rs,Ra>
  R=np.zeros(3,dtype=object);
  R[0]=np.zeros((FLAGS.No,FLAGS.Nr),dtype=float);
  R[1]=np.zeros((FLAGS.No,FLAGS.Nr),dtype=float);
  R[2]=np.zeros((FLAGS.Dr,FLAGS.Nr),dtype=float);
  # External Effects
  X=np.zeros((FLAGS.Dx,FLAGS.No),dtype=float);
  
  # marshalling function, m(G)=B, G=<O,R>  
  B=m(O,R);
  """

  # Object Matrix
  O = tf.placeholder(tf.float32, [FLAGS.Ds,FLAGS.No], name="O");
  # Relation Matrics R=<Rr,Rs,Ra>
  Rr = tf.placeholder(tf.float32, [FLAGS.No,FLAGS.Nr], name="Rr");
  Rs = tf.placeholder(tf.float32, [FLAGS.No,FLAGS.Nr], name="Rs");
  Ra = tf.placeholder(tf.float32, [FLAGS.Dr,FLAGS.Nr], name="Ra");
  # External Effects
  X = tf.placeholder(tf.float32, [FLAGS.Dx,FLAGS.No], name="X");
   
  # marshalling function, m(G)=B, G=<O,R>  
  B=m(O,Rr,Rs,Ra);
  
  # relational modeling phi_R(B)=E
  E=phi_R(B);
  
  # aggregator
  C=a(O,Rr,X,E);
  
  # object modeling phi_O(C)=P
  P=phi_O(C);
  
  # abstract modeling phi_A(P)=q
  q=phi_A(P);
  print(q);exit(1);

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
                      help='The Number of State')
  parser.add_argument('--No', type=int, default=5,
                      help='The Number of Objects')
  parser.add_argument('--Nr', type=int, default=5,
                      help='The Number of Relations')
  parser.add_argument('--Dr', type=int, default=5,
                      help='The Relationship Dimension')
  parser.add_argument('--Dx', type=int, default=3,
                      help='The External Effect Dimension')
  parser.add_argument('--De', type=int, default=50,
                      help='The Effect Dimension')
  parser.add_argument('--Dp', type=int, default=2,
                      help='The Object Modeling Output Dimension')
  parser.add_argument('--Da', type=int, default=1,
                      help='The Abstract Modeling Output Dimension')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
