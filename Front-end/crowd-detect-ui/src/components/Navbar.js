import { useState } from "react";
import "./Navbar.css";

function Navbar() {
  const [selected, setSelected] = useState("Filter");
  const [isOpen, setIsOpen] = useState(false);
  const [showProfile, setShowProfile] = useState(false);

  return (
    <nav className="navbar">
      {/* Dropdown Filter */}
      <div className="dropdown-container">
        <button className="dropdown-button" onClick={() => setIsOpen(!isOpen)}>
          {selected}
        </button>
        {isOpen && (
          <ul className="dropdown-menu">
            <li onClick={() => { setSelected("Risk"); setIsOpen(false); }}>Risk</li>
            <li onClick={() => { setSelected("Safe"); setIsOpen(false); }}>Safe</li>
          </ul>
        )}
      </div>

      {/* Profile Section */}
      <div className="profile-container">
        <button className="profile-button" onClick={() => setShowProfile(!showProfile)}>P</button>

        {/* Profile Popup */}
        {showProfile && (
          <div className="profile-popup">
            <h3>User Profile</h3>
            <p><strong>Name:</strong> John Doe</p>
            <p><strong>Email:</strong> johndoe@example.com</p>
            <button className="close-btn" onClick={() => setShowProfile(false)}>Close</button>
          </div>
        )}
      </div>
    </nav>
  );
}

export default Navbar;
