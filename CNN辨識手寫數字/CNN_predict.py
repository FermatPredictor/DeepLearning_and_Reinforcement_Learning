import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import glob,cv2
import os

def show_images_labels_predictions(images, labels, predictions,num=10):
    plt.gcf().set_size_inches(12, 14)
    num = min(num, 25) 
    for i in range(num):
        ax=plt.subplot(5,5, 1+i)
        #顯示黑白圖片
        ax.imshow(images[i], cmap='binary')
        
        # 有 AI 預測結果資料, 才在標題顯示預測結果
        if( len(predictions) > 0 ) :
            title = 'ai = ' + str(predictions[i])
            # 預測正確顯示(o), 錯誤顯示(x)
            title += (' (o)' if predictions[i]==labels[i] else ' (x)') 
            title += '\nlabel = ' + str(labels[i])
        # 沒有 AI 預測結果資料, 只在標題顯示真實數值
        else :
            title = 'label = ' + str(labels[i])
            
        # X, Y 軸不顯示刻度    
        ax.set_title(title,fontsize=12) 
        ax.set_xticks([])
        ax.set_yticks([])        
    plt.show()
    
def load_test_data(path):
    #建立測試特徵集、測試標籤	    
    files = glob.glob(f"{path}\*.png" ) + glob.glob(f"{path}\*.jpg" )
    test_feature=[]
    test_label=[]
    for file in files:
        img=cv2.imread(file)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #灰階    
        _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) #轉為反相黑白 
        test_feature.append(img)
        label = os.path.basename(file)[0]  # "imagedata\1.jpg"
        test_label.append(int(label))
    
    return np.array(test_feature), np.array(test_label)
       

if __name__ == '__main__':
    #從 HDF5 檔案中載入模型
    print("載入模型 CNN_handwrite_model.h5")
    model = load_model('CNN_handwrite_model.h5')
    
    # 讀測資
    path = './imagedata'
    test_feature, test_label = load_test_data(path)
    
    # reshape成CNN吃的格式，並標準化至0~1區間
    test_feature_normalize = test_feature.reshape(test_feature.shape[0],28,28,1).astype('float32')/255
        
    #預測
    predict = model.predict(test_feature_normalize)
    prediction = np.argmax(predict, axis=-1)
    
    #顯示圖像、預測值、真實值 
    show_images_labels_predictions(test_feature,test_label,prediction,len(test_feature))