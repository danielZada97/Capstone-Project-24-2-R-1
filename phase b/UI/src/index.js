// index.js
import React from "react";
import ReactDOM from "react-dom/client";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import "./index.css";
import App from "./App"; // Suppose App is the Home page component
import Background from "./Background";
import AboutUs from "./aboutUs"; // Ensure correct casing
import Navbar from "./Navbar";
import reportWebVitals from "./reportWebVitals";
import "bootstrap/dist/css/bootstrap.min.css";
import Results from "./Results";
import Footer from "./footer"; // Corrected import path with uppercase 'F'

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <div className="d-flex flex-column min-vh-100">
      <Router>
        <Navbar />
        <main className="flex-fill">
          <Routes>
            <Route path="/" element={<Navigate to="/home" replace />} />
            <Route path="/home" element={<App />} />
            <Route path="/background" element={<Background />} />
            <Route path="/results" element={<Results />} />{" "}
            {/* Ensure path casing */}
            <Route path="/aboutus" element={<AboutUs />} />{" "}
            {/* Ensure path casing */}
          </Routes>
        </main>
        <Footer />
      </Router>
    </div>
  </React.StrictMode>
);

reportWebVitals();
