import tensorflow.keras as k
import numpy as np
from sklearn.model_selection import train_test_split

class CNN:
    def __init__(self,inputDim,classNum):
        self.inputDim = inputDim
        self.classNum = classNum
        optimizer = k.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
        self.model = self.makeNetwork()
        
        self.model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,metrics=['accuracy'])
        
        data = np.load('cleanData.npy')
        data = data.reshape((data.shape[0],data.shape[1],1))
        labels = np.load('cleanLabs.npy')
        labels = labels[:,0]
        labels = labels.astype('uint')
        self.train_data, self.test_data, self.train_labs, self.test_labs = train_test_split(data,labels,test_size=0.25,random_state=42)
        
    def makeNetwork(self):
        inputLayer = k.layers.Input(shape=(self.inputDim,1,))
        x = k.layers.Conv1D(64,20,strides = 2,padding='same')(inputLayer)
        x = k.layers.Activation('relu')(x)
        x = k.layers.BatchNormalization(momentum=0.8)(x)
        x = k.layers.Conv1D(128,20,strides = 2,padding='same')(x)
        x = k.layers.Activation('relu')(x)
        x = k.layers.BatchNormalization(momentum=0.8)(x)
        x = k.layers.Conv1D(256,20,strides = 2,padding='same')(x)
        x = k.layers.Activation('relu')(x)
        x = k.layers.BatchNormalization(momentum=0.8)(x)
        x = k.layers.Conv1D(256,20,strides = 1,padding='same')(x)
        x = k.layers.Activation('relu')(x)
        x = k.layers.BatchNormalization(momentum=0.8)(x)
        x = k.layers.Conv1D(256,20,strides = 2,padding='same')(x)
        x = k.layers.Activation('relu')(x)
        x = k.layers.BatchNormalization(momentum=0.8)(x)
        x = k.layers.Conv1D(256,20,strides = 1,padding='same')(x)
        x = k.layers.Activation('relu')(x)
        x = k.layers.BatchNormalization(momentum=0.8)(x)
        x = k.layers.Dropout(0.4)(x)
        x = k.layers.Flatten()(x)
        x = k.layers.Dense(1000)(x)
        x = k.layers.Dropout(0.4)(x)
        x = k.layers.Dense(1000)(x)
        x = k.layers.Dropout(0.4)(x)
        x = k.layers.Dense(self.classNum)(x)
        x = k.layers.Activation('softmax')(x)
        model = k.models.Model(inputs = inputLayer, outputs = x)
        model.summary()
        return model
    
    
    def train(self,epochs,batch_size):
        train_data = self.train_data
        train_labels = self.train_labs
        
        for epoch in range(epochs):
            idx = np.random.randint(0, train_data.shape[0], batch_size)
            imgs = train_data[idx]
            labs = train_labels[idx]
            
            d_loss = self.model.train_on_batch(imgs,labs)
            print ("%d [train loss: %f] " % (epoch, d_loss[0]))
        self.model.save_weights('./weights/my_classifier')
        
    def predict(self):
        test_data = self.test_data
        test_labels = self.test_labs
        
        test_loss = self.model.evaluate(test_data,test_labels)
        print ("[test loss: %f] " % (test_loss[0]))
        
        from sklearn.metrics import roc_curve
        y_pred_keras = self.model.predict(test_data)
        
        from sklearn.metrics import auc
        
        oak_probs = y_pred_keras[:,0]
        oak_r = test_labels==0
        oak_r.astype('int')
        fpr_oak, tpr_oak, thresholds_oak = roc_curve(oak_r,oak_probs)
        auc_oak = auc(fpr_oak, tpr_oak)
        
        dog_probs = y_pred_keras[:,1]
        dog_r = test_labels==1
        dog_r.astype('int')
        fpr_dog, tpr_dog, thresholds_dog = roc_curve(dog_r,dog_probs)
        auc_dog = auc(fpr_dog, tpr_dog)
        
        tulip_probs = y_pred_keras[:,2]
        tulip_r = test_labels==2
        tulip_r.astype('int')
        fpr_tulip, tpr_tulip, thresholds_tulip = roc_curve(tulip_r,tulip_probs)
        auc_tulip = auc(fpr_tulip, tpr_tulip)
        
        yew_probs = y_pred_keras[:,3]
        yew_r = test_labels==3
        yew_r.astype('int')
        fpr_yew, tpr_yew, thresholds_yew = roc_curve(yew_r,yew_probs)
        auc_yew = auc(fpr_yew, tpr_yew)
        
        import matplotlib.pyplot as plt
        plt.figure(1,figsize=(4,4))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_oak, tpr_oak, label='AUC = {:.3f}'.format(auc_oak),color='0.2')
        plt.plot(fpr_dog, tpr_dog, label='AUC = {:.3f}'.format(auc_dog),color='0.4')
        plt.plot(fpr_tulip, tpr_tulip, label='AUC = {:.3f}'.format(auc_tulip),color='0.1')
        plt.plot(fpr_yew, tpr_yew, label='AUC = {:.3f}'.format(auc_yew),color='0.3')
        plt.gca().xaxis.set_ticklabels([])
        plt.gca().yaxis.set_ticklabels([])
        plt.savefig("ROC_curve_box.eps",transparent=True)
        plt.close()  
        

    def loadWeights(self):
        self.model.load_weights('./weights/my_classifier')
    
    
    
if __name__ == '__main__':
    cnn = CNN(400,4)
    cnn.train(epochs=10000, batch_size=32)
    #cnn.loadWeights()
    cnn.predict()
    


