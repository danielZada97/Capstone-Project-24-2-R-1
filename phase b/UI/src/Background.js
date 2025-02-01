import React from "react";
import "./background.css"; // Import the CSS file for the card & body styles
import "bootstrap/dist/css/bootstrap.min.css";
import densenetImage from "./densenet1.png";
import cnn_model from "./CNN.png";
import tl_image from "./tl.png";
function Background() {
  return (
    <div>
      {/* 
        The outer <div> ensures the background spans full height
        and there's space for the card to sit on top.
      */}
      <div
        style={{
          minHeight: "100vh",
          padding: "2rem",
        }}
      >
        <div className="container">
          <div className="card semi-transparent-card" style={{}}>
            <div className="card-body">
              <h1 className="card-title">Background</h1>

              <p className="card-text">
                Welcome to our Background page! Below, we briefly describe some
                fundamental concepts relevant to our project, namely
                Convolutional Neural Networks, DenseNet architecture, and
                Transfer Learning.
              </p>
              <div className="densenet-container">
                <h2>Convolutional Neural Networks (CNNs)</h2>
                <p>
                  <b> Convolutional Neural Networks (CNNs)</b> are a type of
                  deep neural network widely used in computer vision tasks. They
                  work by applying convolutional filters to input images, which
                  helps the network automatically detect and learn different
                  features such as edges, textures, and patterns. This ability
                  to recognize spatial hierarchies of features makes CNNs highly
                  effective for tasks like image classification, where the goal
                  is to identify what an image represents; object detection,
                  which involves locating and classifying objects within an
                  image; and image segmentation, where the image is divided into
                  meaningful segments for further analysis. CNNs have
                  revolutionized the field of computer vision by providing
                  accurate and efficient solutions for analyzing and
                  interpreting visual data.
                </p>
                <img src={cnn_model} alt="" className="cnn-image" />
                <br /> <br />
              </div>
              <h2>DenseNet</h2>
              <p>
                <b>DenseNet</b>, short for Densely Connected Convolutional
                Network, is a deep learning architecture that improves how
                layers connect by linking each layer directly to all previous
                layers within the same dense block. This design allows better
                flow of information and gradients through the network, helping
                to prevent the vanishing-gradient problem and encouraging the
                reuse of features. As a result, DenseNet uses parameters more
                efficiently than traditional Convolutional Neural Networks
                (CNNs), enabling the creation of deeper models without
                significantly increasing computational complexity. Additionally,
                this architecture helps the network learn a wider variety of
                features, enhancing performance in tasks like image
                classification, object detection, and medical image analysis.
                Overall, DenseNet's unique design offers important benefits in
                both training and performance, making it a strong choice for
                advanced neural network applications.
              </p>
              <img src={densenetImage} alt="" className="densenet-image" />
              <br />
              <br />
              <h2>Transfer Learning</h2>
              <p>
                <b> Transfer Learning</b> utilizes pre-trained models that have
                been trained on large datasets like ImageNet to improve
                performance on new tasks or domains. Instead of building a model
                from the ground up, which can be time-consuming and require vast
                amounts of data, you can fine-tune an existing model using a
                smaller dataset. This approach saves both computational
                resources and development time while still achieving high
                accuracy. By leveraging the knowledge the pre-trained model has
                already acquired, Transfer Learning allows for quicker and more
                efficient training, making it an effective strategy for various
                applications such as image classification, object detection, and
                medical image analysis.
                <br />
                <br />
                <img src={tl_image} alt="" className="tl-image" />
                <br />
                <br />
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Background;
