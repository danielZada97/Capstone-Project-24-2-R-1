import React from "react";
import { Link } from "react-router-dom";
import "./Navbar.css";
import logo from "./MediSight.png";

function Navbar() {
  return (
    <nav
      className="navbar navbar-expand-lg navbar-dark sticky-navbar"
      style={{ background: "#6da2bc" }}
    >
      <button
        className="navbar-toggler"
        type="button"
        data-toggle="collapse"
        data-target="#navbarSupportedContent"
        aria-controls="navbarSupportedContent"
        aria-expanded="false"
        aria-label="Toggle navigation"
      >
        <span className="navbar-toggler-icon"></span>
      </button>

      <Link className="navbar-brand mr-auto" to="/home">
        <img
          src={logo}
          alt="Medisight Logo"
          style={{ height: "80px", marginRight: "10px" }}
        />
      </Link>

      <div className="collapse navbar-collapse" id="navbarSupportedContent">
        <ul className="navbar-nav mr-auto">
          <li className="nav-item">
            {/* Adjust the 'to' prop to match your Routes */}
            <Link className="nav-link" to="/home">
              Home
            </Link>
          </li>
          <li className="nav-item">
            <Link className="nav-link" to="/background">
              Background
            </Link>
          </li>
          <li className="nav-item">
            <Link className="nav-link" to="/Results">
              Results
            </Link>
          </li>
          <li className="nav-item">
            <Link className="nav-link" to="/aboutUs">
              About Us
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar;
