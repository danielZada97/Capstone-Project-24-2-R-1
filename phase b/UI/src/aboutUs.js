import React from "react";
import "./aboutus.css";
import "bootstrap/dist/css/bootstrap.min.css";

function AboutUs() {
  return (
    <div>
      <div className="container mt-4">
        {/* Bootstrap Card */}
        <div className="card">
          <div className="card-body">
            <h1 className="card-title">About Us</h1>
            <p className="card-text">
              We are Almog Kadosh, an Information Systems Engineering student,
              and Daniel Zada, a Software Engineering student. This project was
              created as part of our final capstone project. We were deeply
              interested in combining medicine and artificial intelligence.
              Together with our supervisor, Professor Miri Weiss-Cohen, we
              explored ways to integrate these fields. We concluded that AI
              could significantly speed up the analysis and interpretation of
              medical data. Specifically, we focused on interpreting lower back
              MRI images using deep learning and machine learning techniques.
              Over the past year, we extensively researched deep learning and
              its potential applications in medicine, leading to the development
              of this system. Our system processes MRI images of the lower back
              and, with 83% accuracy, identifies the specific condition based on
              the classifications it has learned. This solution drastically
              improves the speed of diagnosing lower back problems. Instead of
              waiting several days for results, the system delivers accurate
              diagnoses almost instantly. Additionally, by standardizing the
              analysis process, we eliminate inconsistencies between different
              groups of doctors, reducing the risk of errors in image
              interpretation.
            </p>

            <h2>Contact Us</h2>
            <p>
              Have questions or want to learn more about our project? <br />
              Feel free to get in touch with us at:
              <br />
              <strong>Almog Kadosh:</strong>
              <a href="mailto:almog2941@gmail.com">almog2941@gmail.com</a>
              <br />
              <strong>Daniel Zada:</strong>
              <a href="mailto:danieloszada90@gmail.com">
                Danieloszada90@gmail.com
              </a>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AboutUs;
