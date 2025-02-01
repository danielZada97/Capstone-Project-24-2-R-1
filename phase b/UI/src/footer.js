// Footer.js
import React from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { FaGithub, FaLinkedin, FaGoogle } from "react-icons/fa";
import "./footer.css";

function Footer() {
  return (
    <footer className="footer  text-white py-2 mt-auto">
      <div className="container text-center">
        <p className="mb-1 small">
          &copy; {new Date().getFullYear()} Lumbar Spine Degenerative
          Classification using an Optimized CNN (24-2-R-1). All rights reserved.
        </p>
        <div className="d-flex justify-content-center">
          <a
            href="https://github.com/danielZada97/Capstone-Project-24-2-R-1"
            className="text-white mr-3 footer-link"
            aria-label="GitHub"
          >
            Project's GitHub <FaGithub size={20} />
            &nbsp; &nbsp;
          </a>
          <a
            href="https://www.linkedin.com/in/daniel-zada/"
            className="text-white mr-3 footer-link"
            aria-label="Daniel Zada's LinkedIn"
          >
            Daniel Zada's LinkedIn <FaLinkedin size={20} /> &nbsp; &nbsp;
          </a>
          <a
            href="https://www.linkedin.com/in/almog-kadosh-0038b2273/"
            className="text-white mr-3 footer-link"
            aria-label="Almog Kadosh's LinkedIn"
          >
            Almog Kadosh's LinkedIn <FaLinkedin size={20} /> &nbsp; &nbsp;
          </a>
          <a
            href="mailto:almog2941@gmail.com"
            className="text-white mr-3 footer-link"
            aria-label="Contact Almog"
          >
            Contact Almog <FaGoogle size={20} /> &nbsp; &nbsp;
          </a>
          <a
            href="mailto:Danieloszada90@gmail.com"
            className="text-white footer-link"
            aria-label="Contact Daniel"
          >
            Contact Daniel <FaGoogle size={20} /> &nbsp; &nbsp;
          </a>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
