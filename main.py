from flask import Flask, request
import pickle, json, cv2, math, threading,os
#from test_model import 
import testModel


# Endpoint to receive image data then localizes and classifies images
 @app.route('/', methods=['POST'])
def receiveImage():
    global target,label
    #clear the image in the save directory
    imagedir = os.path.join(os.getcwd(),'model_test/up')
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)
    filelist = glob.glob(os.path.join(imagedir, "*.jpg"))
        for f in filelist:
            os.remove(f)

    
    #clear the image in the result directory
    resultdir = os.path.join(os.getcwd(),'model_test/up/result')
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    filelist = glob.glob(os.path.join(resultdir, "*.jpg"))
        for f in filelist:
            os.remove(f)
    
    app = Flask(__name__)

    

    content = request.data
    frame = pickle.loads(content)
    #save received photo
    cv2.imwrite("./model_test/up/1.jpg", frame)  

    target,label = main()
    return target, label

# Endpoint to send classification results to algo team
@app.route('/end', methods=['GET'])
def finished():
    #threading.Thread(target = )
    print(json.dumps(target)+":"+json.dumps(label))
    return json.dumps(target)+":"+json.dumps(label)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8123)
