import sys

question = sys.argv[1]

def VefaOvunc_Ozer_21902836_hw1(question):
    if question == '1':
        """
        @author: vovuncozer
        """

        #Import necessary libraries 
        import h5py
        import numpy as np
        import matplotlib.pyplot as plt

        # Import the datasets as numpy arrays
        f = h5py.File("data1.h5", "r")
        data = np.array(f["data"])

        #Obtaining R,G,B values seperately
        R = data[:,0,:,:]
        G = data[:,1,:,:]
        B = data[:,2,:,:]

        #Converting the images to graysale
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

        #Mean-centering the data
        Y = Y - np.mean(Y,axis=0)

        #Clip the data
        std = 3 * np.std(Y)
        clip_data = np.clip(Y,-std,std)

        #Map data range to [0.1 0.9]
        normalized_data = (clip_data - np.amin(Y)) / (np.amax(Y)-np.amin(Y))
        proc_data = normalized_data * 0.8 + 0.1

        #Create random integers for choosing random images 
        random_numbers = np.random.randint(data.shape[0], size=200)

        #Plotting 200 random images in RGB and gray-scale
        for i,j in zip(range(1,201),random_numbers):
            plt.subplot(10, 20, i)
            plt.imshow(np.transpose(data[j]))
            plt.axis(False)

        plt.figure()

        for i,j in zip(range(1,201),random_numbers):
            plt.subplot(10, 20, i)
            plt.imshow(np.transpose(proc_data[j]),cmap="gray")
            plt.axis(False)

        #Design the neural network architecture 
        def wo (L_pre,L_post):
            return np.sqrt(6/(L_pre+L_post)) 

        def init_weights(L_input,L_hidden,L_output): 
            W1 = np.random.uniform(-wo(L_input,L_hidden) ,wo(L_input,L_hidden) ,size=(L_input,L_hidden))
            b1 = np.random.uniform(-wo(L_input,L_hidden) ,wo(L_input,L_hidden) ,size=(1,L_hidden))
            W2 = np.random.uniform(-wo(L_hidden,L_output),wo(L_hidden,L_output),size=(L_hidden,L_output))
            b2 = np.random.uniform(-wo(L_hidden,L_output),wo(L_hidden,L_output),size=(1,L_output))
            We = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}
            
            return We

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        def derivative_sigmoid(z):
            return sigmoid(z)*(1- sigmoid(z))

        def forward_propagation(X,We):
            W1 = We["W1"]
            b1 = We["b1"]
            W2 = We["W2"]
            b2 = We["b2"]
            
            Z1 = np.dot(X,W1) + b1
            A1 = sigmoid(Z1)
            Z2 = np.dot(A1,W2) + b2
            A2 = sigmoid(Z2)
            return A1, A2

        def aeCost(We, data, params):
            lmbd  = params["lmbda"]
            beta  = params["beta"]
            rho   = params["rho"]
            
            N = data.shape[0]
            
            W_1 = We["W1"]
            W_2 = We["W2"]
            
            A1, A2 = forward_propagation(data, We)
            rho_b = np.mean(A1, axis=0)
            
            ASE     = np.sum((data-A2)**2)/(2*N)
            T_term  = lmbd * (np.sum(W_1**2) + np.sum(W_2**2))/2
            KL_term = beta * np.sum(np.where(rho != 0, rho * np.log(rho / rho_b), 0)) 

            rho_transposed = np.transpose(rho_b)
            dh = (np.dot(W_2,(-(data-A2)*derivative_sigmoid(A2)).T)+(np.tile(beta*(-(rho/rho_transposed)+((1-rho)/(1-rho_transposed))), (10240,1)).T))
            dh = dh * np.transpose(derivative_sigmoid(A1))  
           
            gradW1 = (np.dot( np.transpose(data), np.transpose(dh)) + lmbd * W_1)/N
            gradW2 = (np.dot((-(data-A2)*derivative_sigmoid(A2)).T,A1).T + lmbd * W_2)/N
            gradb1 = np.mean(dh, axis=1)
            gradb2 = np.mean((-(data-A2)*derivative_sigmoid(A2)), axis=0)
            
            J     = ASE + T_term + KL_term
            Jgrad = {"dW1":gradW1, "dW2":gradW2, "db1":gradb1, "db2":gradb2}    
            return J, Jgrad

        def update_parameters(X, We,learning_rate, params):
            J,Jgrad = aeCost(We,X,params)
            
            W1 = We["W1"]
            b1 = We["b1"]
            W2 = We["W2"]
            b2 = We["b2"] 

            dW1 = Jgrad["dW1"]
            db1 = Jgrad["db1"]
            dW2 = Jgrad["dW2"]
            db2 = Jgrad["db2"] 
            
            W1 = W1 - learning_rate * dW1
            b1 = b1 - learning_rate * db1
            W2 = W2 - learning_rate * dW2
            b2 = b2 - learning_rate * db2
            We = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}
            return J,We

        def nn_model(X, We, params, learning_rate, num_iterations): 
            flatten_data = np.reshape(X, (X.shape[0],X.shape[1]**2))
            epoch = []
            loss = []

            for k in range(num_iterations):        
                J,We = update_parameters(flatten_data,We,learning_rate,params)
                loss.append(J)
                epoch.append(k)
                #print(k,"    ",J)
            return We,loss,epoch


        We = init_weights(256,64,256)
        params = {"Lin":256, "Lhid":64, "lmbda":0.0005, "beta":0.01, "rho":0.03}
        We,loss,epoch = nn_model(proc_data, We, params, 10**-2, 1000)
        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(epoch,loss)
        plt.title("Lhid:64, lmbda:0.0005")
        plt.show()
        W1 = We["W1"]
        plt.figure(figsize=(8,8))
        for i in range(W1.shape[1]):
            plt.subplot(8,8,i+1)
            plt.suptitle("Lhid:64, lmbda:0.0005")
            plt.imshow(np.reshape(W1[:,i],(16,16)), cmap='gray')
            plt.axis('off')


        We = init_weights(256,8,256)
        params = {"Lin":256, "Lhid":8, "lmbda":0.0005, "beta":0.01, "rho":0.03}
        We,loss,epoch = nn_model(proc_data, We, params, 10**-2, 1000)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(epoch,loss)
        plt.title("Lhid:8, lmbda:0.0005")
        plt.show()
        W1 = We["W1"]
        plt.figure(figsize=(1,8))
        for i in range(W1.shape[1]):
            plt.subplot(8,1,i+1)
            plt.suptitle("Lhid:8, lmbda:0.0005")
            plt.imshow(np.reshape(W1[:,i],(16,16)), cmap='gray')
            plt.axis('off')
            

        We = init_weights(256,32,256)
        params = {"Lin":256, "Lhid":32, "lmbda":0.0005, "beta":0.01, "rho":0.03}
        We,loss,epoch = nn_model(proc_data, We, params, 10**-2, 1000)
        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(epoch,loss)
        plt.title("Lhid:32, lmbda:0.0005")
        plt.show()
        W1 = We["W1"]
        plt.figure(figsize=(4,8))
        for i in range(W1.shape[1]):
            plt.subplot(8,4,i+1)
            plt.suptitle("Lhid:32, lmbda:0.0005")
            plt.imshow(np.reshape(W1[:,i],(16,16)), cmap='gray')
            plt.axis('off')

           
        We = init_weights(256,96,256)
        params = {"Lin":256, "Lhid":8, "lmbda":0.0005, "beta":0.01, "rho":0.03}
        We,loss,epoch = nn_model(proc_data, We, params, 10**-2, 1000)
        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(epoch,loss)
        plt.title("Lhid:96, lmbda:0.0005")
        plt.show()
        W1 = We["W1"]
        plt.figure(figsize=(12,8))
        for i in range(W1.shape[1]):
            plt.subplot(8,12,i+1)
            plt.suptitle("Lhid:96, lmbda:0.0005")
            plt.imshow(np.reshape(W1[:,i],(16,16)), cmap='gray')
            plt.axis('off')


        We = init_weights(256,64,256)
        params = {"Lin":256, "Lhid":64, "lmbda":0.0, "beta":0.01, "rho":0.03}
        We,loss,epoch = nn_model(proc_data, We, params, 10**-2, 1000)
        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(epoch,loss)
        plt.title("Lhid:64, lmbda:0.0")
        plt.show()
        W1 = We["W1"]
        plt.figure(figsize=(8,8))
        for i in range(W1.shape[1]):
            plt.subplot(8,8,i+1)
            plt.suptitle("Lhid:64, lmbda:0.0")
            plt.imshow(np.reshape(W1[:,i],(16,16)), cmap='gray')
            plt.axis('off')


        We = init_weights(256,64,256)
        params = {"Lin":256, "Lhid":64, "lmbda":0.001, "beta":0.01, "rho":0.03}
        We,loss,epoch = nn_model(proc_data, We, params, 10**-2, 1000)
        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(epoch,loss)
        plt.title("Lhid:64, lmbda:0.001")
        plt.show()
        W1 = We["W1"]
        plt.figure(figsize=(8,8))
        for i in range(W1.shape[1]):
            plt.subplot(8,8,i+1)
            plt.suptitle("Lhid:64, lmbda:0.001")
            plt.imshow(np.reshape(W1[:,i],(16,16)), cmap='gray')
            plt.axis('off')


        We = init_weights(256,64,256)
        params = {"Lin":256, "Lhid":64, "lmbda":0.1, "beta":0.01, "rho":0.03}
        We,loss,epoch = nn_model(proc_data, We, params, 10**-2, 1000)
        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(epoch,loss)
        plt.title("Lhid:64, lmbda:0.1")
        plt.show()
        W1 = We["W1"]
        plt.figure(figsize=(8,8))
        for i in range(W1.shape[1]):
            plt.subplot(8,8,i+1)
            plt.imshow(np.reshape(W1[:,i],(16,16)), cmap='gray')
            plt.suptitle("Lhid:64, lmbda:0.1")
            plt.axis('off')
    
    y = myfunction(3,5)
    
    if question == '2' :
        print (question)
        """
        @author: vovuncozer
        """

        #Import necessary libraries 
        import h5py
        import numpy as np
        from sklearn.metrics import accuracy_score
        import matplotlib.pyplot as plt

        np.random.seed(87)

        f = h5py.File('data2.h5', 'r')

        trainx = np.array(f["trainx"])
        traind = np.array(f["traind"])
        traind = traind.reshape(traind.shape[0],1)

        valx = np.array(f["valx"])
        vald = np.array(f["vald"])
        vald = vald.reshape(vald.shape[0],1)

        testx = np.array(f["testx"])
        testd = np.array(f["testd"])
        testd = testd.reshape(testd.shape[0],1)

        words = np.array(f["words"])

        def init_weights(N,D,P):
            W0 = np.random.normal(0,  0.01, (N,D))
            W1 = np.random.normal(0,  0.01, (D,P))
            b1 = np.random.normal(0,  0.01, (1,P))
            W2 = np.random.normal(0,  0.01, (P,N))    
            b2 = np.random.normal(0,  0.01, (1,N))
            
            We = {"W0": W0,"W1": W1, "W2": W2, "b1": b1, "b2": b2}
            return We
           
        def x_vectorize(x):
            x_vec = np.zeros((x.shape[0],3,250))  
            for i in range(x.shape[0]):
                for j in range(3):
                    k = np.zeros(250)
                    k[x[i,j]-1] = 1
                    x_vec[i,j,:] = k
            return x_vec

        def y_vectorize(y): 
            y_vec = np.zeros((y.shape[0],250))        
            for i in range(y.shape[0]):
                y_vec[i,int(y[i])-1]=1             
            return y_vec

        enc_traind = y_vectorize(traind)
        enc_testd  = y_vectorize(testd)
        enc_vald   = y_vectorize(vald)

        enc_trainx = x_vectorize(trainx)
        enc_testx  = x_vectorize(testx)
        enc_valx   = x_vectorize(valx)

        train_data  = np.sum((enc_trainx[:,0,:],enc_trainx[:,1,:],enc_trainx[:,2,:]),axis=0)
        test_data   = np.sum((enc_testx[:,0,:],enc_testx[:,1,:],enc_testx[:,2,:]),axis=0)
        val_data    = np.sum((enc_valx[:,0,:],enc_valx[:,1,:],enc_valx[:,2,:]),axis=0)
        train_label = enc_traind
        test_label  = enc_testd
        val_label   = enc_vald

        def sigmoid(X):
            return 1/(1 + np.exp(-X)) 

        def sigmoid_gradient(z):
            return sigmoid(z) * (1-sigmoid(z))
          
        def softmax(z):
            ez = np.exp(z - np.max(z, axis=-1, keepdims=True))
            return ez / np.sum(ez, axis=-1, keepdims=True)

        def softmax_gradient( X):      
            return softmax(X) * (1 - softmax(X)) 
            
        def cross_entropy(pred,label):        
            m = pred.shape[0]
            m = np.clip(m, 1e-9, pred.shape[0])
            preds = np.clip(pred, 1e-9, 1 - 1e-9)
            loss = np.sum(-label * np.log(preds) - (1 - label) * np.log(1 - preds))      
            return loss/m 

        def cross_entropy_gradient( preds, label):        
            preds = np.clip(preds, 1e-15, 1 - 1e-15)
            grad_ce = - (label/preds) + (1 - label) / (1 - preds)
            return grad_ce

        def cross_validation(train_data,val_data,label_train,label_val,th):
            train_preds = predict(train_data)
            val_preds   = predict(val_data)
            train_loss  = cross_entropy(train_preds,label_train)
            val_loss    = cross_entropy(val_preds,label_val)

            if train_loss - val_loss < th:
                return True
            return False

        def lin_grads(inp,change): 
            return { 'dW': np.dot(inp,change)/batch_size, 'dB': np.sum(change, axis=0, keepdims=True)/batch_size}

        def forward_propagation(X):          
            Z0 = np.dot(X,W0)          
            Z1 = np.dot(Z0,W1) - b1
            A1 = sigmoid(Z1)      
            Z2 = np.dot(A1,W2) - b2        
            A2 = softmax(Z2)
            pro_dict = {"Z1": Z1,"A1": A1,"Z2": Z2,"A2": A2,"Z0" : Z0} 
            return pro_dict

        def backward_propagation(ch, X, Y):         
            Z0 = ch['Z0']
            Z1 = ch['Z1']
            A1 = ch['A1']
            Z2 = ch['Z2']
            A2 = ch['A2'] 
                          
            dZ2 = cross_entropy_gradient(A2,Y) * softmax_gradient(Z2) 
            dW2 = lin_grads(A1.T,dZ2)['dW']
            db2 = lin_grads(A1.T,dZ2)['dB']
            
            dZ1 = np.dot(dZ2,W2.T) * sigmoid_gradient(Z1)
            dW1 = lin_grads(Z0.T ,dZ1)['dW']
            db1 = lin_grads(Z0.T ,dZ1)['dB']
            
            dZ0 = np.dot(dZ1,W1.T)     
            dW0 = lin_grads(X.T,dZ0)['dW']
            grads = {"dW1": dW1, "dW2": dW2,"db1": db1, "db2": db2,"dW0": dW0}
            return grads


        def update_weights(W0,W1,W2,b1,b2,grads):
            dl_W0 = -learning_rate * grads['dW0'] +  momentum_rate * update_history['W0']
            dl_W1 = -learning_rate * grads['dW1'] +  momentum_rate * update_history['W1']
            dl_W2 = -learning_rate * grads['dW2'] +  momentum_rate * update_history['W2']
            dl_b1 = -learning_rate * grads['db1'] +  momentum_rate * update_history['b1']    
            dl_b2 = -learning_rate * grads['db2'] +  momentum_rate * update_history['b2']
            
            W0 += dl_W0     
            W1 += dl_W1
            W2 += dl_W2
            b1 += dl_b1
            b2 += dl_b2        
           
            update_history['W0'] = dl_W0
            update_history['W1'] = dl_W1
            update_history['W2'] = dl_W2
            update_history['b1'] = dl_b1
            update_history['b2'] = dl_b2

            return               

        def nn_model(X,Y,X_val,Y_val,epochs,crossVal = False):   
            
            for e in range(epochs):
                         
                rnge = np.random.permutation(372500)
                
                for i in range(trainx.shape[0]//batch_size): 

                    index  = rnge[batch_size * i : batch_size * i + batch_size]            
                    subX = X[index]
                    subY = Y[index]
                    
                    train_output = forward_propagation(subX)                                                          
                    grads = backward_propagation(train_output,subX,subY)                
                    update_weights(W0,W1,W2,b1,b2,grads)
                    
                    # if cross_validation(X,X_val,Y,y_val,th = 5):
                    #     break

                cross_loss_train = cross_entropy(train_output['A2'],subY)
                predictions_train = predict(X)
                acc_train = accuracy_score(predictions_train,np.argmax(Y,1))

                cross_loss_val = cross_entropy(X_val,Y_val)
                predictions_val = predict(X_val)
                acc_val = accuracy_score(predictions_val,np.argmax(Y_val,1))
                # print("Epoch: ", e+1)       
                # print("Training Accuracy   : ", acc_train)                        
                # print("Validation Accuracy : ", acc_val)

                train_loss.append(cross_loss_train)              
                test_loss.append(cross_loss_val) 
                train_accuracy.append(acc_train)              
                test_accuracy.append(acc_val)
                           
        def predict(X):
            cches = forward_propagation(X)
            return np.argmax(cches['A2'],axis=1) 

        #(D,P) = (32,256)
        train_loss = []
        test_loss  = []
        train_accuracy = []
        test_accuracy  = []

        We = init_weights(250,32,256)  
        W0 = We["W0"]      
        W1 = We["W1"] 
        b1 = We["b1"]
        W2 = We["W2"] 
        b2 = We["b2"] 

        update_history = {'W0':0,'W1':0,'W2':0,'b1':0,'b2':0}  

        batch_size = 200
        learning_rate = 0.15
        momentum_rate = 0.85


        nn_model(train_data,train_label,val_data,val_label,50)

        plt.plot(train_loss)
        plt.xlabel('Epoch')
        plt.title('Cross-Entropy (Training), (D,P) = (32,256)')
        plt.show()


        plt.plot(train_accuracy)
        plt.xlabel('Epoch')
        plt.title(' Accuracy (Training), (D,P) = (32,256)')
        plt.show()

        plt.plot(test_accuracy)
        plt.xlabel('Epoch')
        plt.title('Accuracy (Validation), (D,P) = (32,256)')
        plt.show()

        test_preds = predict(test_data)
        test_acc32_256   = accuracy_score(test_preds,np.argmax(test_label,1))
        print('Test Accuracy (D,P) = (32,256): ',test_acc32_256)

        caches = forward_propagation(test_data)
        prediction = caches['A2']

        random_int = np.random.randint(test_data.shape[0],size = 5)

        five_random_num_probs = prediction[random_int]
        random_five_words = words[testx[random_int]].astype('U13')

        def predict_10(probs,n_predict):
            word = np.zeros((n_predict,10))
            for i in range(n_predict):
                 word[i] = probs.argsort()[i][-10:][::-1]
            return word

        prediction_10      = predict_10(five_random_num_probs,5) 
        best_10_prediction = words[prediction_10.astype('int')].astype('U13')

        input_words = [list(random_five_words[i])  for i in range(5)] 
        preds_words = [list(best_10_prediction[i]) for i in range(5)]


        print('Top 10 predictions for (D,P) = (32,256)')
        for i in range(5):    
            print('Input words: ',i+1,'=',  ', '.join(input_words[i]),'\nTop 10 predicted candicates:', ', '.join(preds_words[i]))

        #(D,P) = (16,128)
        train_loss = []
        test_loss  = []
        train_accuracy = []
        test_accuracy  = []

        We = init_weights(250,16,128)  
        W0 = We["W0"]      
        W1 = We["W1"] 
        b1 = We["b1"]
        W2 = We["W2"] 
        b2 = We["b2"] 

        update_history = {'W0':0,'W1':0,'W2':0,'b1':0,'b2':0}  

        batch_size = 200
        learning_rate = 0.15
        momentum_rate = 0.85

        nn_model(train_data,train_label,val_data,val_label,50)

        plt.plot(train_loss)
        plt.xlabel('Epoch')
        plt.title('Cross-Entropy (Training), (D,P) = (16,128)')
        plt.show()

        plt.plot(train_accuracy)
        plt.xlabel('Epoch')
        plt.title(' Accuracy (Training), (D,P) = (16,128)')
        plt.show()

        plt.plot(test_accuracy)
        plt.xlabel('Epoch')
        plt.title('Accuracy (Validation), (D,P) = (16,128)')
        plt.show()

        test_preds = predict(test_data)
        test_acc16_128   = accuracy_score(test_preds,np.argmax(test_label,1))
        print('Test Accuracy (D,P) = (16,128): ',test_acc16_128)

        caches = forward_propagation(test_data)
        prediction = caches['A2']

        random_int = np.random.randint(test_data.shape[0],size = 5)

        five_random_num_probs = prediction[random_int]
        random_five_words = words[testx[random_int]].astype('U13')

        def predict_10(probs,n_predict):
            word = np.zeros((n_predict,10))
            for i in range(n_predict):
                 word[i] = probs.argsort()[i][-10:][::-1]
            return word

        prediction_10      = predict_10(five_random_num_probs,5) 
        best_10_prediction = words[prediction_10.astype('int')].astype('U13')

        input_words = [list(random_five_words[i])  for i in range(5)] 
        preds_words = [list(best_10_prediction[i]) for i in range(5)]

        print('Top 10 predictions for (D,P) = (16,128)')
        for i in range(5):    
            print('Input words: ',i+1,'=',  ', '.join(input_words[i]),'\nTop 10 predicted candicates:', ', '.join(preds_words[i]))

        #(D,P) = (8,64)
        train_loss = []
        test_loss  = []
        train_accuracy = []
        test_accuracy  = []

        We = init_weights(250,8,64)  
        W0 = We["W0"]      
        W1 = We["W1"] 
        b1 = We["b1"]
        W2 = We["W2"] 
        b2 = We["b2"] 

        update_history = {'W0':0,'W1':0,'W2':0,'b1':0,'b2':0}  

        batch_size = 200
        learning_rate = 0.15
        momentum_rate = 0.85


        nn_model(train_data,train_label,val_data,val_label,50)

        plt.plot(train_loss)
        plt.xlabel('Epoch')
        plt.title('Cross-Entropy (Training), (D,P) = (8,64)')
        plt.show()

        plt.plot(train_accuracy)
        plt.xlabel('Epoch')
        plt.title(' Accuracy (Training), (D,P) = (8,64)')
        plt.show()

        plt.plot(test_accuracy)
        plt.xlabel('Epoch')
        plt.title('Accuracy (Validation), (D,P) = (8,64)')
        plt.show()

        test_preds = predict(test_data)
        test_acc8_64   = accuracy_score(test_preds,np.argmax(test_label,1))
        print('Test Accuracy (D,P) = (8,64): ',test_acc8_64)

        caches = forward_propagation(test_data)
        prediction = caches['A2']

        random_int = np.random.randint(test_data.shape[0],size = 5)

        five_random_num_probs = prediction[random_int]
        random_five_words = words[testx[random_int]].astype('U13')

        def predict_10(probs,n_predict):
            word = np.zeros((n_predict,10))
            for i in range(n_predict):
                 word[i] = probs.argsort()[i][-10:][::-1]
            return word

        prediction_10      = predict_10(five_random_num_probs,5) 
        best_10_prediction = words[prediction_10.astype('int')].astype('U13')

        input_words = [list(random_five_words[i])  for i in range(5)] 
        preds_words = [list(best_10_prediction[i]) for i in range(5)]

        print('Top 10 predictions for (D,P) = (8,64)')
        for i in range(5):    
            print('Input words: ',i+1,'=',  ', '.join(input_words[i]),'\nTop 10 predicted candicates:', ', '.join(preds_words[i]))

        
    elif question == '3' :
        print (question)
        """
        @vovuncozer
        """

        #Question 3 PART A
        import h5py
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix
        from sklearn import metrics
        import matplotlib.pyplot as plt

        f    = h5py.File("data3.h5", "r")
        trX  = np.array(f["trX"])
        tstX = np.array(f["tstX"])
        trY  = np.array(f["trY"])
        tstY = np.array(f["tstY"])
        trX, valX,trY , valY = train_test_split(trX, trY, test_size=0.1, random_state=42, shuffle=False)



        def softmax(z):
            ez = np.exp(z - np.max(z, axis=-1, keepdims=True))
            return ez / np.sum(ez, axis=-1, keepdims=True)

        def htan(x):
          return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

        def forward_propagation(X):
            global W0,W1,W2,b1,b2
            
            state   = np.zeros((np.shape(X)[1],np.shape(X)[0],np.shape(X)[2]))
            h_state = np.zeros((np.shape(X)[1],np.shape(X)[0],L_hidden))  
            probs   = np.zeros((np.shape(X)[1],np.shape(X)[0],L_output))  
            
            for x in range(150):
                state[x,:,:]   = X[:,x]
                h_state[x,:,:] = htan(np.dot(state[x,:,:],W1) + np.dot(h_state[x-1,:,:],W0) + b1)
                probs[x,:,:]   = softmax(np.dot(h_state[x,:,:],W2)+b2)
            
            return [state,h_state,probs]

        def backward_propogation(current,trY):
            global dW0,dW1,dW2,db1,db2,W0,W1,W2,b1,b2
            dW0, dW1, dW2, db1, db2 = np.zeros_like(W0), np.zeros_like(W1), np.zeros_like(W2), np.zeros_like(b1), np.zeros_like(b2)
            
            state   = current[0]
            h_state = current[1]
            probs   = current[2]
            
            
            dh_n = np.zeros_like(h_state[0])
            ouput_d = np.copy(probs[-1])      
            ouput_d[np.arange(len(trY)),np.argmax(trY,1)] -= 1
            db2 = db2 + np.sum(ouput_d,axis = 0, keepdims = True)
            dW2 = dW2 + np.dot(np.transpose(h_state[-1]),ouput_d)
            
            for a in range(149,-1,-1):
                dh     = np.dot(ouput_d,np.transpose(W2)) + dh_n
                dh_rec = (1 - (h_state[a,:,:] **2)) * dh
                dW1    = dW1 + np.dot(state[a,:,:].T,dh_rec)         
                dW0    = dW0 + np.dot(h_state[a-1,:,:].T,dh_rec)
                db1    = db1 + np.sum(dh_rec,axis = 0, keepdims = True)
                dh_n   = np.dot(dh_rec,W0.T)
            
            grads = [dW1,db1,dW0,dW2,db2]
            
            for i in grads:
                np.clip(i, -10, 10, out = i)
                
            return grads

            
        def cross_entropy(trY,predicted_Y):
                predictions = np.clip(predicted_Y, 1e-9, 1. - 1e-9)
                total = predictions.shape[0]         
                return -np.sum(trY * np.log(predictions + 1e-9))/total
            
        def update_parameters(grads):
            global W1,b1,W0,W2,b2,update_history,alpha,dW1,db1,dW0,dW2,db2,params
            alpha    = params["alpha"]
            momentum = params["momentum"]
            alpha *= 0.9999
            
            dW1 = grads[0]
            db1 = grads[1]
            dW0 = grads[2]
            dW2 = grads[3]
            db2 = grads[4]   
            
            c_W1 =  alpha * dW1 +  momentum * update_history[0]
            c_b1 =  alpha * db1 +  momentum * update_history[1]
            c_W0 =  alpha * dW0 +  momentum * update_history[2]
            c_W2 =  alpha * dW2 +  momentum * update_history[3]
            c_b2 =  alpha * db2 +  momentum * update_history[4]     

            W1 = W1 - c_W1
            b1 = b1 - c_b1
            W0 = W0 - c_W0
            W2 = W2 - c_W2
            b2 = b2 - c_b2

            update_history[0] = c_W1
            update_history[1] = c_b1
            update_history[2] = c_W0
            update_history[3] = c_W2
            update_history[4] = c_b2

            return 

        def predict(data):
            x,y,probs = forward_propagation(data)
            return np.argmax(probs[-1],axis=1)


        def RNN(trX,trY,tstX,tstY,epochs):
            global dW1,dW0,dW2,db1,db2,W1,b1,W2,b2,W0,train_loss,test_loss,train_acc,test_acc,params
            batch = params["batch"]
            for epoch in range(epochs):
                rnge = np.random.permutation(2700)
                for i in range(84):
                    index  = rnge[batch * i : batch * i + batch]            
                    subX = trX[index]
                    subY = trY[index]
                    
                    caches = forward_propagation(subX)  
                    grads  = backward_propogation(caches,subY)
                    update_parameters(grads)
                
                prob_valid = forward_propagation(valX)[2]
                loss_valid = cross_entropy(valY,prob_valid[-1])
               
                predict_val = predict(trX)
                accuracy_val = accuracy_score(np.argmax(trY,1),predict_val)
                
                val_history.append(loss_valid) 
                val_accuracy.append(accuracy_val)   
                # print(accuracy_val)           
                
                if 0.5 > loss_valid:
                    break
                else:
                    continue
            
            probability  = forward_propagation(tstX)[2]
            ce_loss_test = cross_entropy(tstY,probability[-1])    
            prediction   = np.argmax(probability[-1],1)
            accuracy_tst = accuracy_score(np.argmax(tstY,1),prediction)

            return val_history,val_accuracy,accuracy_tst,ce_loss_test,W1,b1,W2,b2,W0

        #Case 1 
        def Xavier_init(F_in,F_out):
            limit = np.sqrt(6/(F_in + F_out))
            return limit
        np.random.seed(123)
        L_input = 3
        L_hidden = 128
        L_output = 6

        lim   = Xavier_init(3,128)
        limh  = Xavier_init(128,128)
        lim_y = Xavier_init(128,6)

        W0 = np.random.uniform(-limh,limh,(L_hidden,L_hidden))
        W1 = np.random.uniform(-lim,lim,(L_input,L_hidden))
        W2 = np.random.uniform(-lim_y,lim_y,(L_hidden,L_output))
        b1 = np.random.uniform(-lim,lim,(1,L_hidden))
        b2 = np.random.uniform(-lim,lim,(1,L_output))

        update_history = [0,0,0,0,0]
        val_history  = []        
        val_accuracy = []    

        params = {"alpha": 10**-4,"momentum": 0.85,"batch": 32}

        val_history,val_accuracy,accuracy_tst,ce_loss_test,W1,b1,W2,b2,W0 = RNN(trX,trY,tstX,tstY,50)
           
        print("Test loss is      : " ,ce_loss_test)
        print("Test accuracy is  : " ,accuracy_tst)
        print("alpha: 10**-4, momentum: 0.85, batch: 32")
        try:
            plt.figure()
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.plot(val_history)
            plt.title("alpha: 10**-4, momentum: 0.0, batch: 8")
            plt.figure()
            probability      = forward_propagation(tstX)[2]
            prediction       = np.argmax(probability[-1],1)
            accuracy_tst     = accuracy_score(np.argmax(tstY,1),prediction)
            prediction       = prediction.tolist()
            label            = np.argmax(tstY,1).tolist()
            conf_matrix      = confusion_matrix(label, prediction)
            cm_display       = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix,display_labels = False)
            cm_display.plot()
            plt.show()
            plt.figure()
            plt.plot(val_history)
            plt.xlabel("Epoch")
            plt.ylabel("Validation Error")
            plt.title("alpha: 10**-4, momentum: 0.85, batch: 32")
            plt.show()
        except:
            print("")


        #Case 2
        def Xavier_init(F_in,F_out):
            limit = np.sqrt(6/(F_in + F_out))
            return limit
        np.random.seed(123)
        L_input = 3
        L_hidden = 128
        L_output = 6

        lim   = Xavier_init(3,128)
        limh  = Xavier_init(128,128)
        lim_y = Xavier_init(128,6)

        W0 = np.random.uniform(-limh,limh,(L_hidden,L_hidden))
        W1 = np.random.uniform(-lim,lim,(L_input,L_hidden))
        W2 = np.random.uniform(-lim_y,lim_y,(L_hidden,L_output))
        b1 = np.random.uniform(-lim,lim,(1,L_hidden))
        b2 = np.random.uniform(-lim,lim,(1,L_output))

        update_history = [0,0,0,0,0]
        val_history  = []        
        val_accuracy = []    

        params = {"alpha": 10**-4,"momentum": 0.0,"batch": 32}

        val_history,val_accuracy,accuracy_tst,ce_loss_test,W1,b1,W2,b2,W0 = RNN(trX,trY,tstX,tstY,50)
        print("alpha: 10**-4, momentum: 0.0, batch: 32")
        print("Test loss is      : " ,ce_loss_test)
        print("Test accuracy is  : " ,accuracy_tst)

        try:
            plt.figure()
            probability      = forward_propagation(tstX)[2]
            prediction       = np.argmax(probability[-1],1)
            accuracy_tst     = accuracy_score(np.argmax(tstY,1),prediction)
            prediction       = prediction.tolist()
            label            = np.argmax(tstY,1).tolist()
            conf_matrix      = confusion_matrix(label, prediction)
            cm_display       = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix,display_labels = False)
            cm_display.plot()
            plt.show()
            plt.figure()
            plt.plot(val_history)
            plt.xlabel("Epoch")
            plt.ylabel("Validation Error")
            plt.title("alpha: 10**-4, momentum: 0.0, batch: 32")
            plt.show()
        except:
            print("")

        #Case 3
        def Xavier_init(F_in,F_out):
            limit = np.sqrt(6/(F_in + F_out))
            return limit
        np.random.seed(123)
        L_input = 3
        L_hidden = 128
        L_output = 6

        lim   = Xavier_init(3,128)
        limh  = Xavier_init(128,128)
        lim_y = Xavier_init(128,6)

        W0 = np.random.uniform(-limh,limh,(L_hidden,L_hidden))
        W1 = np.random.uniform(-lim,lim,(L_input,L_hidden))
        W2 = np.random.uniform(-lim_y,lim_y,(L_hidden,L_output))
        b1 = np.random.uniform(-lim,lim,(1,L_hidden))
        b2 = np.random.uniform(-lim,lim,(1,L_output))

        update_history = [0,0,0,0,0]
        val_history  = []        
        val_accuracy = []           
        params = {"alpha": 10**-4,"momentum": 0.0,"batch": 8}

        val_history,val_accuracy,accuracy_tst,ce_loss_test,W1,b1,W2,b2,W0 = RNN(trX,trY,tstX,tstY,50)
        print("alpha: 10**-4, momentum: 0.0, batch: 8")  
        print("Test loss is      : " ,ce_loss_test)
        print("Test accuracy is  : " ,accuracy_tst)
        try:
            plt.figure()
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.plot(val_history)
            plt.title("alpha: 10**-4, momentum: 0.0, batch: 8")
            plt.show()
            probability      = forward_propagation(tstX)[2]
            prediction       = np.argmax(probability[-1],1)
            accuracy_tst     = accuracy_score(np.argmax(tstY,1),prediction)
            prediction       = prediction.tolist()
            label            = np.argmax(tstY,1).tolist()
            conf_matrix      = confusion_matrix(label, prediction)
            cm_display       = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix,display_labels = False)
            cm_display.plot()
            plt.show()
            plt.figure()
            plt.plot(val_history)
            plt.xlabel("Epoch")
            plt.ylabel("Validation Error")
            plt.title("alpha: 10**-4, momentum: 0.0, batch: 8")
            plt.show()
        except:
            print("")

def myfunction(a,b):
    return a+b

VefaOvunc_Ozer_21902836_hw1(question)



