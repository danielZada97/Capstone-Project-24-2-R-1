/* App.js */
import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import uploadIcon from "./upload_icon.png";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState([]);

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    setFile(uploadedFile);

    if (uploadedFile) {
      setPreview(URL.createObjectURL(uploadedFile));
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please upload a file first.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/home",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      const classificationDict = response.data.results.results;
      if (!classificationDict || Object.keys(classificationDict).length === 0) {
        setResult([["All Normal/Mild", ""]]);
        return;
      }
      const lines = Object.entries(classificationDict);
      setResult(lines);
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("An error occurred while uploading the file.");
    }
  };

  return (
    <div className="page-container">
      <div className="main-container">
        <h1>Image Processing</h1>

        {/* Upload box */}
        <label htmlFor="file-upload" className="upload-box">
          {preview ? (
            <img src={preview} alt="Preview" className="preview-image" />
          ) : (
            <div className="upload-placeholder">
              <img id="uploadImg" src={uploadIcon} alt="upload_icon" />
              <p id="uploadText">Please upload an image</p>
            </div>
          )}
        </label>

        <input
          id="file-upload"
          type="file"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />

        {/* Buttons */}
        <div className="button-row">
          <label htmlFor="file-upload" className="custom-file-upload">
            Choose File
          </label>
          <button onClick={handleUpload} className="upload_button">
            Upload and Process
          </button>
        </div>

        {result && result.length > 0 && (
          <div className="result-container">
            <div style={{ whiteSpace: "pre-wrap" }}>
              {result.map(([condLevel, severity], index) => {
                let backgroundColor = "#7f8c11";
                let textColor = "white";

                if (severity.toLowerCase() === "severe") {
                  backgroundColor = "red";
                } else if (severity.toLowerCase() === "moderate") {
                  backgroundColor = "#ffa500";
                }

                return (
                  <p
                    key={index}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr auto",
                      alignItems: "center",
                      gap: "10px",
                      margin: "0 0 3px 0",
                    }}
                  >
                    <span>{condLevel ? condLevel + ":" : ""}</span>
                    <span
                      style={{
                        backgroundColor,
                        color: textColor,
                        padding: "4px 8px",
                        borderRadius: "4px",
                        minWidth: "80px",
                        textAlign: "center",
                      }}
                    >
                      {severity}
                    </span>
                  </p>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
