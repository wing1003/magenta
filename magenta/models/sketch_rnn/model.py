# Copyright 2024 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sketch-RNN Model."""

import random

from magenta.contrib import training as contrib_training
from magenta.models.sketch_rnn import rnn
import numpy as np
import tensorflow.compat.v1 as tf


def copy_hparams(hparams):
  """Return a copy of an HParams instance."""
  return contrib_training.HParams(**hparams.values())


def get_default_hparams():
  """Return default HParams for sketch-rnn."""
  hparams = contrib_training.HParams(
      // data_set=['aaron_sheep.npz'],  # Our dataset.
      data_set=['swing set.npz,spoon.npz,fan.npz,cruise ship.npz,hourglass.full.npz,helicopter.full.npz,duck.npz,parrot.npz,dog.npz,owl.full.npz,school bus.full.npz,map.full.npz,flying saucer.npz,oven.full.npz,crocodile.full.npz,elephant.full.npz,picture frame.npz,mushroom.npz,headphones.npz,horse.npz,clarinet.npz,elbow.full.npz,boomerang.npz,hospital.full.npz,hospital.npz,tent.npz,picture frame.full.npz,rabbit.npz,pillow.npz,saw.npz,fence.npz,ceiling fan.npz,horse.full.npz,piano.npz,church.full.npz,rhinoceros.npz,pizza.full.npz,eye.full.npz,roller coaster.npz,snowflake.npz,star.full.npz,chandelier.full.npz,skateboard.full.npz,megaphone.full.npz,aircraft carrier.full.npz,square.full.npz,knee.npz,bulldozer.full.npz,campfire.full.npz,rabbit.full.npz,shoe.full.npz,spider.npz,rain.full.npz,cake.npz,lantern.full.npz,popsicle.full.npz,toaster.npz,paint can.npz,fireplace.full.npz,hot dog.full.npz,fish.full.npz,skull.npz,rifle.npz,dumbbell.full.npz,drums.npz,smiley face.full.npz,book.npz,saxophone.full.npz,radio.npz,arm.full.npz,nose.full.npz,hat.npz,mouth.npz,paint can.full.npz,onion.full.npz,castle.npz,river.npz,ambulance.full.npz,purse.full.npz,bandage.npz,guitar.full.npz,brain.full.npz,floor lamp.full.npz,jacket.full.npz,watermelon.full.npz,golf club.full.npz,remote control.npz,sea turtle.npz,lighter.full.npz,popsicle.npz,pickup truck.npz,envelope.npz,bread.npz,sea turtle.full.npz,cat.full.npz,skyscraper.full.npz,ambulance.npz,pliers.npz,tooth.full.npz,stereo.full.npz,postcard.full.npz,sink.full.npz,hexagon.full.npz,campfire.npz,hurricane.npz,police car.npz,sailboat.npz,cooler.npz,bathtub.npz,umbrella.full.npz,dolphin.npz,key.npz,elephant.npz,soccer ball.npz,potato.npz,fork.full.npz,diving board.full.npz,palm tree.full.npz,submarine.full.npz,bench.npz,syringe.full.npz,lion.npz,birthday cake.full.npz,paper clip.full.npz,postcard.npz,garden.full.npz,soccer ball.full.npz,pear.full.npz,ladder.full.npz,peas.full.npz,arm.npz,keyboard.npz,parachute.npz,streetlight.npz,passport.full.npz,canoe.full.npz,spreadsheet.npz,drums.full.npz,monkey.full.npz,strawberry.full.npz,cow.full.npz,hockey puck.npz,blackberry.full.npz,goatee.full.npz,camel.npz,bridge.npz,snail.full.npz,moustache.full.npz,lighthouse.npz,flamingo.npz,broom.full.npz,beard.full.npz,sheep.full.npz,airplane.full.npz,parrot.full.npz,book.full.npz,helicopter.npz,swan.npz,nose.npz,motorbike.npz,octopus.npz,cactus.full.npz,bracelet.npz,brain.npz,The Mona Lisa.npz,toothbrush.npz,eyeglasses.full.npz,tent.full.npz,knee.full.npz,knife.full.npz,jacket.npz,lighthouse.full.npz,microphone.npz,map.npz,binoculars.full.npz,bracelet.full.npz,screwdriver.full.npz,pants.full.npz,lightning.full.npz,carrot.npz,stop sign.full.npz,barn.npz,car.full.npz,cannon.full.npz,scorpion.full.npz,dragon.npz,pond.full.npz,cell phone.npz,paintbrush.full.npz,pineapple.npz,hand.full.npz,cookie.npz,kangaroo.npz,blueberry.npz,dresser.full.npz,stove.full.npz,church.npz,couch.npz,rhinoceros.full.npz,bread.full.npz,The Great Wall of China.npz,candle.npz,sheep.npz,cookie.full.npz,cactus.npz,angel.npz,squiggle.full.npz,door.full.npz,mosquito.npz,bird.npz,animal migration.npz,finger.npz,finger.full.npz,The Mona Lisa.full.npz,steak.npz,backpack.npz,lobster.npz,golf club.npz,flashlight.full.npz,power outlet.full.npz,hexagon.npz,garden hose.npz,jail.full.npz,cow.npz,crayon.npz,palm tree.npz,mushroom.full.npz,mailbox.npz,shark.npz,cello.full.npz,mountain.full.npz,television.npz,frog.full.npz,mermaid.npz,circle.npz,banana.full.npz,cake.full.npz,lighter.npz,onion.npz,matches.full.npz,spider.full.npz,panda.npz,pool.npz,fireplace.npz,leaf.full.npz,raccoon.npz,crab.full.npz,compass.full.npz,bowtie.npz,umbrella.npz,butterfly.npz,light bulb.npz,flip flops.npz,envelope.full.npz,alarm clock.npz,aircraft carrier.npz,hamburger.npz,saw.full.npz,school bus.npz,lipstick.npz,bottlecap.full.npz,boomerang.full.npz,pineapple.full.npz,snake.full.npz,microwave.npz,stitches.full.npz,hourglass.npz,hockey puck.full.npz,ceiling fan.full.npz,face.npz,ant.npz,asparagus.full.npz,clarinet.full.npz,snowman.full.npz,remote control.full.npz,peanut.full.npz,string bean.npz,baseball bat.npz,owl.npz,house.full.npz,speedboat.full.npz,rollerskates.full.npz,scissors.npz,power outlet.npz,stop sign.npz,circle.full.npz,bed.npz,camel.full.npz,lantern.npz,house.npz,elbow.npz,swan.full.npz,steak.full.npz,hockey stick.full.npz,flashlight.npz,drill.full.npz,megaphone.npz,stethoscope.full.npz,rainbow.full.npz,eraser.npz,bee.npz,fence.full.npz,submarine.npz,scissors.full.npz,ladder.npz,asparagus.npz,shoe.npz,t-shirt.npz,passport.npz,river.full.npz,hand.npz,triangle.npz,sun.full.npz,lightning.npz,mug.npz,floor lamp.npz,bridge.full.npz,ice cream.full.npz,train.full.npz,nail.npz,leg.npz,t-shirt.full.npz,calculator.full.npz,toe.full.npz,moon.npz,see saw.full.npz,washing machine.npz,canoe.npz,lipstick.full.npz,suitcase.full.npz,bat.full.npz,carrot.full.npz,lollipop.npz,sword.full.npz,binoculars.npz,garden.npz,basket.npz,rake.full.npz,penguin.npz,hurricane.full.npz,backpack.full.npz,parachute.full.npz,streetlight.full.npz,eraser.full.npz,apple.full.npz,mouth.full.npz,grass.npz,pig.full.npz,blueberry.full.npz,eyeglasses.npz,screwdriver.npz,beach.npz,mouse.npz,apple.npz,bicycle.full.npz,grapes.npz,spoon.full.npz,calculator.npz,airplane.npz,pickup truck.full.npz,stairs.full.npz,squiggle.npz,squirrel.npz,matches.npz,sword.npz,cat.npz,toe.npz,snorkel.full.npz,snorkel.npz,barn.full.npz,pond.npz,blackberry.npz,ear.npz,swing set.full.npz,hedgehog.full.npz,frying pan.npz,chandelier.npz,mouse.full.npz,calendar.full.npz,marker.full.npz,ear.full.npz,goatee.npz,beach.full.npz,rollerskates.npz,potato.full.npz,house plant.full.npz,lobster.full.npz,leg.full.npz,peanut.npz,cup.npz,anvil.npz,suitcase.npz,chair.npz,drill.npz,The Great Wall of China.full.npz,angel.full.npz,stereo.npz,shorts.npz,cloud.npz,broccoli.full.npz,face.full.npz,piano.full.npz,alarm clock.full.npz,mountain.npz,tooth.npz,firetruck.npz,cannon.npz,hammer.full.npz,dishwasher.npz,frog.npz,laptop.npz,vase.npz,rifle.full.npz,bus.full.npz,diving board.npz,paintbrush.npz,diamond.full.npz,pear.npz,bird.full.npz,octagon.full.npz,anvil.full.npz,hot tub.npz,peas.npz,door.npz,calendar.npz,baseball.full.npz,bear.full.npz,kangaroo.full.npz,light bulb.full.npz,pliers.full.npz,hockey stick.npz,toothpaste.npz,moustache.npz,sandwich.full.npz,lion.full.npz,stove.npz,hot tub.full.npz,cup.full.npz,sleeping bag.npz,baseball.npz,car.npz,pool.full.npz,axe.npz,washing machine.full.npz,helmet.full.npz,flower.npz,hot air balloon.npz,sailboat.full.npz,table.full.npz,radio.full.npz,eye.npz,dumbbell.npz,frying pan.full.npz,chair.full.npz,sweater.npz,stitches.npz,tractor.npz,shorts.full.npz,octagon.npz,smiley face.npz,couch.full.npz,dishwasher.full.npz,shovel.full.npz,basketball.npz,helmet.npz,crab.npz,grapes.full.npz,clock.npz,shark.full.npz,diamond.npz,foot.npz,dog.full.npz,computer.npz,pencil.npz,dolphin.full.npz,flip flops.full.npz,snowman.npz,monkey.npz,duck.full.npz,harp.full.npz,knife.npz,necklace.npz,bat.npz,compass.npz,bicycle.npz,microphone.full.npz,crown.full.npz,animal migration.full.npz,crayon.full.npz,coffee cup.full.npz,keyboard.full.npz,shovel.npz,axe.full.npz,bench.full.npz,bucket.full.npz,dresser.npz,house plant.npz,firetruck.full.npz,skyscraper.npz,skateboard.npz,mailbox.full.npz,bottlecap.npz,coffee cup.npz,microwave.full.npz,banana.npz,hammer.npz,moon.full.npz,teapot.npz,giraffe.npz,pizza.npz,donut.full.npz,raccoon.full.npz,snowflake.full.npz,nail.full.npz,sandwich.npz,ocean.full.npz,basket.full.npz,sun.npz,camouflage.npz,cell phone.full.npz,bush.full.npz,string bean.full.npz,harp.npz,telephone.npz,stairs.npz,star.npz,guitar.npz,flying saucer.full.npz,strawberry.npz,octopus.full.npz,police car.full.npz,necklace.full.npz,mug.full.npz,feather.npz,pillow.full.npz,leaf.npz,bee.full.npz,bed.full.npz,castle.full.npz,bus.npz,cello.npz,key.full.npz,laptop.full.npz,belt.full.npz,sleeping bag.full.npz,line.npz,bucket.npz,grass.full.npz,camera.full.npz,motorbike.full.npz,pencil.full.npz,ocean.npz,camera.npz,sweater.full.npz,cooler.full.npz,candle.full.npz,computer.full.npz,The Eiffel Tower.npz,hat.full.npz,headphones.full.npz,hamburger.full.npz,trumpet.npz,table.npz,The Eiffel Tower.full.npz,bush.npz,purse.npz,feather.full.npz,lollipop.full.npz,ice cream.npz,squirrel.full.npz,pig.npz,broom.npz,stethoscope.npz,crown.npz,square.npz,basketball.full.npz,fire hydrant.npz,bowtie.full.npz,cloud.full.npz,donut.npz,spreadsheet.full.npz,jail.npz,butterfly.full.npz,clock.full.npz,birthday cake.npz,saxophone.npz,rake.npz,fire hydrant.full.npz,panda.full.npz,telephone.full.npz,flamingo.full.npz,beard.npz,bandage.full.npz,syringe.npz,oven.npz,penguin.full.npz,mermaid.full.npz,baseball bat.full.npz,sock.npz,flower.full.npz,fan.full.npz,dragon.full.npz,fork.npz,bulldozer.npz,foot.full.npz,skull.full.npz,marker.npz,snake.npz,paper clip.npz,bear.npz,snail.npz,cruise ship.full.npz,sink.npz,sock.full.npz,hot air balloon.full.npz,belt.npz,speedboat.npz,garden hose.full.npz,scorpion.npz,hot dog.npz,fish.npz,mosquito.full.npz,roller coaster.full.npz,giraffe.full.npz,see saw.npz,rain.npz,camouflage.full.npz,broccoli.npz,line.full.npz,hedgehog.npz,rainbow.npz,ant.full.npz,pants.npz,crocodile.npz,bathtub.full.npz'],  # Our dataset.
      num_steps=10000000,  # Total number of steps of training. Keep large.
      save_every=500,  # Number of batches per checkpoint creation.
      max_seq_len=250,  # Not used. Will be changed by model. [Eliminate?]
      dec_rnn_size=512,  # Size of decoder.
      dec_model='lstm',  # Decoder: lstm, layer_norm or hyper.
      enc_rnn_size=256,  # Size of encoder.
      enc_model='lstm',  # Encoder: lstm, layer_norm or hyper.
      z_size=128,  # Size of latent vector z. Recommend 32, 64 or 128.
      kl_weight=0.5,  # KL weight of loss equation. Recommend 0.5 or 1.0.
      kl_weight_start=0.01,  # KL start weight when annealing.
      kl_tolerance=0.2,  # Level of KL loss at which to stop optimizing for KL.
      batch_size=100,  # Minibatch size. Recommend leaving at 100.
      grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
      num_mixture=20,  # Number of mixtures in Gaussian mixture model.
      learning_rate=0.001,  # Learning rate.
      decay_rate=0.9999,  # Learning rate decay per minibatch.
      kl_decay_rate=0.99995,  # KL annealing decay rate per minibatch.
      min_learning_rate=0.00001,  # Minimum learning rate.
      use_recurrent_dropout=True,  # Dropout with memory loss. Recommended
      recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep.
      use_input_dropout=False,  # Input dropout. Recommend leaving False.
      input_dropout_prob=0.90,  # Probability of input dropout keep.
      use_output_dropout=False,  # Output dropout. Recommend leaving False.
      output_dropout_prob=0.90,  # Probability of output dropout keep.
      random_scale_factor=0.15,  # Random scaling data augmentation proportion.
      augment_stroke_prob=0.10,  # Point dropping augmentation proportion.
      conditional=True,  # When False, use unconditional decoder-only model.
      is_training=True  # Is model training? Recommend keeping true.
  )
  return hparams


class Model(object):
  """Define a SketchRNN model."""

  def __init__(self, hps, gpu_mode=True, reuse=False):
    """Initializer for the SketchRNN model.

    Args:
       hps: a HParams object containing model hyperparameters
       gpu_mode: a boolean that when True, uses GPU mode.
       reuse: a boolean that when true, attemps to reuse variables.
    """
    self.hps = hps
    with tf.variable_scope('vector_rnn', reuse=reuse):
      if not gpu_mode:
        with tf.device('/cpu:0'):
          tf.logging.info('Model using cpu.')
          self.build_model(hps)
      else:
        tf.logging.info('Model using gpu.')
        self.build_model(hps)

  def encoder(self, batch, sequence_lengths):
    """Define the bi-directional encoder module of sketch-rnn."""
    unused_outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
        self.enc_cell_fw,
        self.enc_cell_bw,
        batch,
        sequence_length=sequence_lengths,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='ENC_RNN')

    last_state_fw, last_state_bw = last_states
    last_h_fw = self.enc_cell_fw.get_output(last_state_fw)
    last_h_bw = self.enc_cell_bw.get_output(last_state_bw)
    last_h = tf.concat([last_h_fw, last_h_bw], 1)
    mu = rnn.super_linear(
        last_h,
        self.hps.z_size,
        input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
        scope='ENC_RNN_mu',
        init_w='gaussian',
        weight_start=0.001)
    presig = rnn.super_linear(
        last_h,
        self.hps.z_size,
        input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
        scope='ENC_RNN_sigma',
        init_w='gaussian',
        weight_start=0.001)
    return mu, presig

  def build_model(self, hps):
    """Define model architecture."""
    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    if hps.dec_model == 'lstm':
      cell_fn = rnn.LSTMCell
    elif hps.dec_model == 'layer_norm':
      cell_fn = rnn.LayerNormLSTMCell
    elif hps.dec_model == 'hyper':
      cell_fn = rnn.HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'

    if hps.enc_model == 'lstm':
      enc_cell_fn = rnn.LSTMCell
    elif hps.enc_model == 'layer_norm':
      enc_cell_fn = rnn.LayerNormLSTMCell
    elif hps.enc_model == 'hyper':
      enc_cell_fn = rnn.HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'

    use_recurrent_dropout = self.hps.use_recurrent_dropout
    use_input_dropout = self.hps.use_input_dropout
    use_output_dropout = self.hps.use_output_dropout

    cell = cell_fn(
        hps.dec_rnn_size,
        use_recurrent_dropout=use_recurrent_dropout,
        dropout_keep_prob=self.hps.recurrent_dropout_prob)

    if hps.conditional:  # vae mode:
      if hps.enc_model == 'hyper':
        self.enc_cell_fw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
      else:
        self.enc_cell_fw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)

    # dropout:
    tf.logging.info('Input dropout mode = %s.', use_input_dropout)
    tf.logging.info('Output dropout mode = %s.', use_output_dropout)
    tf.logging.info('Recurrent dropout mode = %s.', use_recurrent_dropout)
    if use_input_dropout:
      tf.logging.info('Dropout to input w/ keep_prob = %4.4f.',
                      self.hps.input_dropout_prob)
      cell = tf.nn.rnn_cell.DropoutWrapper(
          cell, input_keep_prob=self.hps.input_dropout_prob)
    if use_output_dropout:
      tf.logging.info('Dropout to output w/ keep_prob = %4.4f.',
                      self.hps.output_dropout_prob)
      cell = tf.nn.rnn_cell.DropoutWrapper(
          cell, output_keep_prob=self.hps.output_dropout_prob)
    self.cell = cell

    self.sequence_lengths = tf.placeholder(
        dtype=tf.int32, shape=[self.hps.batch_size])
    self.input_data = tf.placeholder(
        dtype=tf.float32,
        shape=[self.hps.batch_size, self.hps.max_seq_len + 1, 5])

    # The target/expected vectors of strokes
    self.output_x = self.input_data[:, 1:self.hps.max_seq_len + 1, :]
    # vectors of strokes to be fed to decoder (same as above, but lagged behind
    # one step to include initial dummy value of (0, 0, 1, 0, 0))
    self.input_x = self.input_data[:, :self.hps.max_seq_len, :]

    # either do vae-bit and get z, or do unconditional, decoder-only
    if hps.conditional:  # vae mode:
      self.mean, self.presig = self.encoder(self.output_x,
                                            self.sequence_lengths)
      self.sigma = tf.exp(self.presig / 2.0)  # sigma > 0. div 2.0 -> sqrt.
      eps = tf.random_normal(
          (self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)
      self.batch_z = self.mean + tf.multiply(self.sigma, eps)
      # KL cost
      self.kl_cost = -0.5 * tf.reduce_mean(
          (1 + self.presig - tf.square(self.mean) - tf.exp(self.presig)))
      self.kl_cost = tf.maximum(self.kl_cost, self.hps.kl_tolerance)
      pre_tile_y = tf.reshape(self.batch_z,
                              [self.hps.batch_size, 1, self.hps.z_size])
      overlay_x = tf.tile(pre_tile_y, [1, self.hps.max_seq_len, 1])
      actual_input_x = tf.concat([self.input_x, overlay_x], 2)
      self.initial_state = tf.nn.tanh(
          rnn.super_linear(
              self.batch_z,
              cell.state_size,
              init_w='gaussian',
              weight_start=0.001,
              input_size=self.hps.z_size))
    else:  # unconditional, decoder-only generation
      self.batch_z = tf.zeros(
          (self.hps.batch_size, self.hps.z_size), dtype=tf.float32)
      self.kl_cost = tf.zeros([], dtype=tf.float32)
      actual_input_x = self.input_x
      self.initial_state = cell.zero_state(
          batch_size=hps.batch_size, dtype=tf.float32)

    self.num_mixture = hps.num_mixture

    # TODO(deck): Better understand this comment.
    # Number of outputs is 3 (one logit per pen state) plus 6 per mixture
    # component: mean_x, stdev_x, mean_y, stdev_y, correlation_xy, and the
    # mixture weight/probability (Pi_k)
    n_out = (3 + self.num_mixture * 6)

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable('output_w', [self.hps.dec_rnn_size, n_out])
      output_b = tf.get_variable('output_b', [n_out])

    # decoder module of sketch-rnn is below
    output, last_state = tf.nn.dynamic_rnn(
        cell,
        actual_input_x,
        initial_state=self.initial_state,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='RNN')

    output = tf.reshape(output, [-1, hps.dec_rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    self.final_state = last_state

    # NB: the below are inner functions, not methods of Model
    def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
      """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
      norm1 = tf.subtract(x1, mu1)
      norm2 = tf.subtract(x2, mu2)
      s1s2 = tf.multiply(s1, s2)
      # eq 25
      z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
           2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
      neg_rho = 1 - tf.square(rho)
      result = tf.exp(tf.div(-z, 2 * neg_rho))
      denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
      result = tf.div(result, denom)
      return result

    def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,
                     z_pen_logits, x1_data, x2_data, pen_data):
      """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
      # This represents the L_R only (i.e. does not include the KL loss term).

      result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,
                             z_corr)
      epsilon = 1e-6
      # result1 is the loss wrt pen offset (L_s in equation 9 of
      # https://arxiv.org/pdf/1704.03477.pdf)
      result1 = tf.multiply(result0, z_pi)
      result1 = tf.reduce_sum(result1, 1, keep_dims=True)
      result1 = -tf.log(result1 + epsilon)  # avoid log(0)

      fs = 1.0 - pen_data[:, 2]  # use training data for this
      fs = tf.reshape(fs, [-1, 1])
      # Zero out loss terms beyond N_s, the last actual stroke
      result1 = tf.multiply(result1, fs)

      # result2: loss wrt pen state, (L_p in equation 9)
      result2 = tf.nn.softmax_cross_entropy_with_logits(
          labels=pen_data, logits=z_pen_logits)
      result2 = tf.reshape(result2, [-1, 1])
      if not self.hps.is_training:  # eval mode, mask eos columns
        result2 = tf.multiply(result2, fs)

      result = result1 + result2
      return result

    # below is where we need to do MDN (Mixture Density Network) splitting of
    # distribution params
    def get_mixture_coef(output):
      """Returns the tf slices containing mdn dist params."""
      # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
      z = output
      z_pen_logits = z[:, 0:3]  # pen states
      z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

      # process output z's into MDN parameters

      # softmax all the pi's and pen states:
      z_pi = tf.nn.softmax(z_pi)
      z_pen = tf.nn.softmax(z_pen_logits)

      # exponentiate the sigmas and also make corr between -1 and 1.
      z_sigma1 = tf.exp(z_sigma1)
      z_sigma2 = tf.exp(z_sigma2)
      z_corr = tf.tanh(z_corr)

      r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
      return r

    out = get_mixture_coef(output)
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out

    self.pi = o_pi
    self.mu1 = o_mu1
    self.mu2 = o_mu2
    self.sigma1 = o_sigma1
    self.sigma2 = o_sigma2
    self.corr = o_corr
    self.pen_logits = o_pen_logits
    # pen state probabilities (result of applying softmax to self.pen_logits)
    self.pen = o_pen

    # reshape target data so that it is compatible with prediction shape
    target = tf.reshape(self.output_x, [-1, 5])
    [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(target, 5, 1)
    pen_data = tf.concat([eos_data, eoc_data, cont_data], 1)

    lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr,
                            o_pen_logits, x1_data, x2_data, pen_data)

    self.r_cost = tf.reduce_mean(lossfunc)

    if self.hps.is_training:
      self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
      optimizer = tf.train.AdamOptimizer(self.lr)

      self.kl_weight = tf.Variable(self.hps.kl_weight_start, trainable=False)
      self.cost = self.r_cost + self.kl_cost * self.kl_weight

      gvs = optimizer.compute_gradients(self.cost)
      g = self.hps.grad_clip
      capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
      self.train_op = optimizer.apply_gradients(
          capped_gvs, global_step=self.global_step, name='train_step')


def sample(sess, model, seq_len=250, temperature=1.0, greedy_mode=False,
           z=None):
  """Samples a sequence from a pre-trained model."""

  def adjust_temp(pi_pdf, temp):
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf

  def get_pi_idx(x, pdf, temp=1.0, greedy=False):
    """Samples from a pdf, optionally greedily."""
    if greedy:
      return np.argmax(pdf)
    pdf = adjust_temp(np.copy(pdf), temp)
    accumulate = 0
    for i in range(0, pdf.size):
      accumulate += pdf[i]
      if accumulate >= x:
        return i
    tf.logging.info('Error with sampling ensemble.')
    return -1

  def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
    if greedy:
      return mu1, mu2
    mean = [mu1, mu2]
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

  prev_x = np.zeros((1, 1, 5), dtype=np.float32)
  prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
  if z is None:
    z = np.random.randn(1, model.hps.z_size)  # not used if unconditional

  if not model.hps.conditional:
    prev_state = sess.run(model.initial_state)
  else:
    prev_state = sess.run(model.initial_state, feed_dict={model.batch_z: z})

  strokes = np.zeros((seq_len, 5), dtype=np.float32)
  mixture_params = []

  greedy = greedy_mode
  temp = temperature

  for i in range(seq_len):
    if not model.hps.conditional:
      feed = {
          model.input_x: prev_x,
          model.sequence_lengths: [1],
          model.initial_state: prev_state
      }
    else:
      feed = {
          model.input_x: prev_x,
          model.sequence_lengths: [1],
          model.initial_state: prev_state,
          model.batch_z: z
      }

    params = sess.run([
        model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr,
        model.pen, model.final_state
    ], feed)

    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = params

    idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)

    idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)
    eos = [0, 0, 0]
    eos[idx_eos] = 1

    next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                          o_sigma1[0][idx], o_sigma2[0][idx],
                                          o_corr[0][idx], np.sqrt(temp), greedy)

    strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

    params = [
        o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0],
        o_pen[0]
    ]

    mixture_params.append(params)

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0][0] = np.array(
        [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
    prev_state = next_state

  return strokes, mixture_params
