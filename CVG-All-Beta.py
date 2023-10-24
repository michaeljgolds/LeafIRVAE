import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

from sklearn.utils import class_weight

import scipy.io


class Sampling(tf.keras.layers.Layer):
    def call(self,inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    
img_shape = (400,1)
latent_dim = 10
batch_size = 512
num_classes = 4
disc_f_dim = 10

#Encoder network, maps 400 dimensional IR with conditioning information to the normal distributed latent space.
def createEncoder():
    inp = Input(img_shape)
    
    inp_t = Input(4,)
    
    
    le = Dense(400)(inp_t)
    le = Reshape((400,1))(le)
    
    inp_other = Input(3,)
    le_other = Dense(400)(inp_other)
    le_other = Reshape((400,1))(le_other)
    
    layer = Flatten()(inp)
    layer = Reshape((400,1))(layer)
    
    
    merge = Concatenate(axis=1)([layer,le])
    
    merge = Concatenate(axis=1)([merge,le_other])
    
    layer = Flatten()(merge)
    
    layer = Dense(300,activation='relu')(layer)
    layer = Dense(200,activation='relu')(layer)
    layer = Dense(100,activation='relu')(layer)
    mu = Dense(latent_dim)(layer)
    log_var = Dense(latent_dim)(layer)
    z = Sampling()([mu, log_var])
    
    model = Model([inp,inp_t,inp_other],[mu, log_var, z])
    model.summary()
    return model

#Decoder network, maps from latent space with conditioning information back to 400 dimensional IR
def createGenerator():
    inp = Input(latent_dim)
    
    
    inp_t = Input(4,)
    
    le = Dense(latent_dim,name='dec_emb')(inp_t)
    le = Reshape((latent_dim,1))(le)
    
    layer = Flatten()(inp)
    layer = Reshape((latent_dim,1))(layer)
    
    inp_other = Input(3,)
    le_other = Dense(400)(inp_other)
    le_other = Reshape((400,1))(le_other)
    
    merge = Concatenate(axis=1)([layer,le])
    
    merge = Concatenate(axis=1)([merge,le_other])
    
    layer = Flatten()(merge)
    layer = Dense(100,activation='relu')(layer)
    layer = Dense(200,activation='relu')(layer)
    layer = Dense(300,activation='relu')(layer)
    layer = Dense(400,activation='relu')(layer)
    layer = Dense(400,activation='sigmoid')(layer)
    out = Reshape((400,1))(layer)
    
    model = Model([inp,inp_t,inp_other],out)
    model.summary()
    return model

#Discriminator network, guesses whether given IR is real or generated
def createDiscriminator():
    inp = Input(img_shape)
    layer = Flatten()(inp)
    layer = Dense(200,activation='relu')(layer)
    layer = Dense(disc_f_dim,activation='relu')(layer)
    out = Dense(1,activation='sigmoid')(layer)
    
    model = Model(inp,[layer, out])
    model.summary()
    return model

#Regressor network, estimates the size and viewing angles of a given latent space value (used for disentagling)
def createRegressor():
    inp = Input(latent_dim)
    layer = Flatten()(inp)
    layer = Dense(10,activation='relu')(layer)
    layer = Dense(5,activation='relu')(layer)
    out = Dense(3,activation='sigmoid')(layer)
    
    model = Model(inp,out)
    model.summary()
    return model
    
#Adversarial classifier network, similar to the regressor, but for leaf species.
def createAdversary():
    inp = Input(latent_dim)
      
    layer = Dense(10,activation='relu')(inp)
    layer = Dense(5,activation='relu')(layer)
    out = Dense(num_classes,activation='softmax')(layer)
    
    model = Model(inp,out)
    model.summary()
    return model

encoder = createEncoder()
generator = createGenerator()
adversary = createAdversary()
discriminator = createDiscriminator()
regressor = createRegressor()

#Model that combines the networks, the losses, and trains the networks.
class CVG(tf.keras.Model):
    def __init__(self,encoder,generator,adversary,discriminator,regressor,set_size,**kwargs):
        super(CVG, self).__init__(**kwargs)
        self.img_shape = (400,1)
        self.latent_dim = latent_dim
        
        self.enc = encoder
        self.gen = generator
        self.disc = discriminator
        self.adv = adversary
        self.reg = regressor
        
        self.set_size = set_size
        self.beta = 1
        self.epoch=1
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.adv_loss_tracker_pro = tf.keras.metrics.Mean(name="adv_loss_pro")
        self.adv_loss_tracker_con = tf.keras.metrics.Mean(name="adv_loss_con")
        self.disc_loss_tracker = tf.keras.metrics.Mean('disc_loss')
        self.gd_loss_tracker = tf.keras.metrics.Mean('gd_loss')
        self.recon_loss_tracker = tf.keras.metrics.Mean('recon_loss')
        self.feature_loss_tracker = tf.keras.metrics.Mean('feature_loss')
        self.reg_loss_tracker = tf.keras.metrics.Mean('reg_loss')
        self.reg_loss_con_tracker = tf.keras.metrics.Mean('reg_loss_con')
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.g_loss_tracker,
            self.kl_loss_tracker,
            self.adv_loss_tracker_pro,
            self.adv_loss_tracker_con,
            self.disc_loss_tracker,
            self.gd_loss_tracker,
            self.recon_loss_tracker,
            self.feature_loss_tracker,
            self.reg_loss_tracker,
            self.reg_loss_con_tracker
        ]
    
    def train_step(self,data_):
        data, d_t, class_weight = data_
        d_lab = d_t[:,4:7]
        d_lab_con = np.tile(0.5,d_lab.shape)
        d_t = d_t[:,0:4]
        noise = np.random.normal(0,1,(data.shape[0],latent_dim))
        noise_t = tf.keras.utils.to_categorical(np.random.randint(0,num_classes,(d_t.shape[0])))
        while (noise_t.shape[1] != num_classes):
            noise_t = tf.keras.utils.to_categorical(np.random.randint(0,num_classes,(d_t.shape[0])))
            
        noise_lab = np.random.uniform(size=d_lab.shape)
        
	#First the 'auxilliary' networks are trained
        self.adv.trainable=True
        self.gen.trainable=False
        self.enc.trainable=False
        self.disc.trainable = True
        self.reg.trainable = True
        
        with tf.GradientTape() as gt:
            z_mean,z_log_var,z = self.enc([data,d_t,d_lab])
            x_f = self.gen([z,d_t,d_lab])
            class_guess = self.adv(z)
            lab_guess = self.reg(z)
            
            x_p = self.gen([noise,noise_t,noise_lab])
            d_features,d_r = self.disc(data)
            x_f_features,d_f = self.disc(x_f)
            x_p_features,d_p = self.disc(x_p)

            ld = (tf.keras.losses.BinaryCrossentropy()(np.ones((batch_size)),d_r) 
                + tf.keras.losses.BinaryCrossentropy()(np.zeros((batch_size)),d_f) 
                + tf.keras.losses.BinaryCrossentropy()(np.zeros((batch_size)),d_p))
            
            adv_loss_pro = tf.reduce_mean(class_weight*tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(d_t,class_guess))
            reg_loss = tf.keras.losses.MeanSquaredError()(d_lab,lab_guess)
            
            total_loss_1 = ld + adv_loss_pro + reg_loss
            
            
        
        grads = gt.gradient(total_loss_1, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        with tf.GradientTape() as gt:
            z_mean,z_log_var,z = self.enc([data,d_t,d_lab])
            x_f = self.gen([z,d_t,d_lab])
            class_guess = self.adv(z)
            lab_guess = self.reg(z)
            x_p = self.gen([noise,noise_t,noise_lab])
            d_features,d_r = self.disc(data)
            x_f_features,d_f = self.disc(x_f)
            x_p_features,d_p = self.disc(x_p)
            
            
            reconstruction_loss = tf.reduce_mean(class_weight *
                tf.reduce_sum(
                    tf.math.square(data - x_f), axis=1
                )
            )
            
            feature_loss = tf.reduce_mean(class_weight *
                tf.reduce_sum(
                    tf.math.square(d_features - x_f_features), axis=1
                )
            )
            
            g_loss = np.ceil(100/self.epoch) * self.beta * reconstruction_loss + feature_loss
            
            gd_loss =  tf.reduce_sum(tf.math.square(tf.reduce_mean(d_features,axis=0) - tf.reduce_mean(x_p_features,axis=0)))
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(class_weight * tf.reduce_sum(kl_loss, axis=1))
            
            adv_loss_con = tf.reduce_sum(tf.math.log(class_guess),axis=1)
            reg_loss_con = tf.keras.losses.MeanSquaredError()(d_lab_con,lab_guess)
            
            total_loss_2 =  g_loss + gd_loss + kl_loss + adv_loss_con + reg_loss_con
        
	#Next the 'main' networks are trained
        self.adv.trainable=False
        self.gen.trainable=True
        self.enc.trainable=True
        self.disc.trainable = False
        self.reg.trainable = False
        
        self.epoch = self.epoch + batch_size/self.set_size
        self.beta = np.max([self.beta,(1/(2 * reconstruction_loss.numpy() * reconstruction_loss.numpy()))])
        
        grads = gt.gradient(total_loss_2, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss_2)
        self.recon_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.adv_loss_tracker_con.update_state(adv_loss_con)
        self.adv_loss_tracker_pro.update_state(adv_loss_pro)
        self.disc_loss_tracker.update_state(ld)
        self.gd_loss_tracker.update_state(gd_loss)
        self.feature_loss_tracker.update_state(feature_loss)
        self.reg_loss_tracker.update_state(reg_loss)
        self.reg_loss_con_tracker.update_state(reg_loss_con)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "feat_loss":self.feature_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "adv_loss_pro": self.adv_loss_tracker_pro.result(),
            "adv_loss_con": self.adv_loss_tracker_con.result(),
            "disc_loss":self.disc_loss_tracker.result(),
            "gd_loss":self.gd_loss_tracker.result(),
            "reg_loss":self.reg_loss_tracker.result(),
            "reg_loss_con":self.reg_loss_con_tracker.result()
            
        }
    
    def load_weights(self):
        self.enc.load_weights('CVG-All-beta-weights/enc')
        self.gen.load_weights('CVG-All-beta-weights/gen')
        
        self.disc.load_weights('CVG-All-beta-weights/disc')
        
        self.reg.load_weights('CVG-All-beta-weights/reg')
        
        self.adv.load_weights('CVG-All-beta-weights/adv')
    
def sampleForRegressor(vae):
    
    noise = np.random.normal(0, 1, (10000, latent_dim))
    noise_lab = np.random.uniform(size=(10000,3))

    noise_lab[:,2] = np.arange(0,10000)/10000
    
    noise_t = tf.keras.utils.to_categorical(np.random.randint(0,num_classes,(10000)))
    while (noise_t.shape[1] != num_classes):
        noise_t = tf.keras.utils.to_categorical(np.random.randint(0,num_classes,(10000)))
    
    gen_imgs = vae.gen.predict([noise,noise_t,noise_lab])
    
    np.save('fakeDataCVGRegressor.npy',gen_imgs)
    np.save('fakeLabelsCVG.npy',noise_lab)
    

#Load data
X_train = np.load("./cleanData.npy")
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))

X_train = X_train/2 + 0.5


#Load labels
X_labs = np.load("./cleanLabs.npy")
labelset = X_labs.reshape((X_labs.shape[0],X_labs.shape[1],1))
real_types = labelset[:,0,0]
real_l = tf.keras.utils.to_categorical(real_types)


real_other = X_labs[:,1:4]
d_t = np.concatenate((real_l,real_other),axis=1)

cw = {0: real_types.shape[0]/(np.where(real_types==0)[0].shape[0]),
                1: real_types.shape[0]/(np.where(real_types==1)[0].shape[0]),
                2: real_types.shape[0]/(np.where(real_types==2)[0].shape[0]),
                3: real_types.shape[0]/(np.where(real_types==3)[0].shape[0])}

#Make model
cvg = CVG(encoder,generator,adversary,discriminator,regressor,real_types.shape[0])

#If model is pretrained, load weights
#cvg.load_weights()

#Train networks
cvg.compile(optimizer=tf.keras.optimizers.Adam(),run_eagerly=True)
cvg.fit(X_train,d_t, epochs=1, batch_size=batch_size, class_weight=cw)

#Sample IRs for the regressor
#sampleForRegressor(cvg)