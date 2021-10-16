const classifier = knnClassifier.create();
//const webcamElement = document.getElementById('webcam');
let net;

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Successfully loaded model');

  // Create an object from Tensorflow.js data API which could capture image 
  // from the web camera as Tensor.
  //const webcam = await tf.data.webcam(webcamElement);

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample1 = async classId => {
    // Capture an image from the web camera.
    for (let i = 1; i <= 10; i++) {
      const im1 = new Image();
      im1.src = "../try/fındıklar/sağlam/sağlam " + i + ".png";

      // Get the intermediate activation of MobileNet 'conv_preds' and pass that
      // to the KNN classifier.
      const activation = net.infer(im1, true);

      // Pass the intermediate activation to the classifier.
      classifier.addExample(activation, classId);

      // Dispose the tensor to release the memory.
      //im1.dispose();
    }

  };
  const addExample2 = async classId => {
    // Capture an image from the web camera.
    for (let i = 1; i <= 10; i++) {
      const im2 = new Image();
      im2.src = "../try/fındıklar/tam_kabuklu/tam " + i + ".png";
      // Get the intermediate activation of MobileNet 'conv_preds' and pass that
      // to the KNN classifier.
      const activation = net.infer(im2, true);

      // Pass the intermediate activation to the classifier.
      classifier.addExample(activation, classId);

      // Dispose the tensor to release the memory.
      //im2.dispose();
    }


  };
  const addExample3 = async classId => {
    // Capture an image from the web camera.
    for (let i = 1; i <= 10; i++) {
      const im3 = new Image();
      im3.src = "../try/fındıklar/yarı_kabuklu/yarı " + i + ".png";

      // Get the intermediate activation of MobileNet 'conv_preds' and pass that
      // to the KNN classifier.
      const activation = net.infer(im3, true);

      // Pass the intermediate activation to the classifier.
      classifier.addExample(activation, classId);

      // Dispose the tensor to release the memory.
      //im3.dispose();
    }

  };

  // When clicking a button, add an example for that class.
  document.getElementById('class-1').addEventListener('click', () => addExample1(0));
  document.getElementById('class-2').addEventListener('click', () => addExample2(1));
  document.getElementById('class-3').addEventListener('click', () => addExample3(2));
 
  while (true) {
    if (classifier.getNumClasses() > 0) {
      //const img = await webcam.capture();
      const img= new Image();
      img.src = "../try/fındıklar/sağlam_sample.png";

      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(img, 'conv_preds');
      // Get the most likely class and confidence from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['sağlam', 'tam_kabuklu', 'yarım_kabuklu'];
      console.log(result);
      document.getElementById('console').innerText = `
          prediction: ${classes[result.label]}\n
          probability: ${result.confidences[result.label]}
        `;

      // Dispose the tensor to release the memory.
      //img.dispose();
    }

    await tf.nextFrame();
  }
}

app();