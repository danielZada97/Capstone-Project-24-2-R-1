import React from "react";
import "./Results.css"; // Import the CSS file for the card & body styles
import "bootstrap/dist/css/bootstrap.min.css";
import Acc_res from "./resized.png";
import Loss_res from "./lr=1e-5_loss.png";
function Results() {
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
          <div className="card semi-transparent-card">
            <div className="card-body">
              <h1 className="card-title">
                <b>Results</b>
              </h1>
              <p className="card-text">
                Welcome to our Results page! Below, we present the key findings
                and performance metrics from our project, including model
                accuracy, loss curves, and other relevant evaluations.
              </p>

              <h2>
                <b>Model Accuracy</b>
              </h2>
              <p>
                Our Convolutional Neural Network achieved an impressive training
                accuracy of 95% and a validation accuracy of 83%, demonstrating
                its effectiveness in classifying lumbar spine degenerative
                conditions. this was achieved with the hyperparameters:
                <br />
                <b>Batch size:</b> 64&nbsp;&nbsp;
                <b>Learning Rate:</b> 1e-5&nbsp;&nbsp;
                <b>Epochs:</b> 50&nbsp;&nbsp;
                <b>Dropout:</b> 0.2&nbsp;&nbsp;
                <br />
              </p>
              <img
                src={Acc_res}
                className="acc-results"
                alt="accuracy result"
              />
              <br />
              <br />
              <h2>
                <b>Loss Curves</b>
              </h2>
              <p>
                The training and validation loss curves indicate a well-fitted
                model with minimal overfitting. The training loss steadily
                decreased over epochs, while the validation loss remained
                stable, suggesting good generalization performance.
              </p>
              <img src={Loss_res} className="acc-results" alt="loss results" />
              {/* You can add more sections as needed */}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Results;
