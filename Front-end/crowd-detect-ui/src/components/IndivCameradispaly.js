import React from 'react';
import "./IndivCameradisplay.css";

const IndivCameradisplay = ({ video, onBack }) => {
  return (
    <div className="indiv-camera-display">
      {/* Header Section */}
      <div className="header">
        <button className='backBtn' onClick={onBack}>⬅ Back</button>
        <h2 className='preview'>{video} - Live View</h2>
        <div className="spacer"></div> {/* Balancing empty div */}
      </div>

      {/* Grid Layout */}
      <div className="grid-contain">
        {/* Video Section (1/2 width) */}
        <div className="grid-item video-container">
          <video controls autoPlay>
            <source src="Video.mp4" type="video/mp4" />
          </video>
        </div>

        {/* Two Additional Sections (1/4 width each) */}
        <div className="grid-item info-section">
          <h3>Section 1</h3>
          <p>Content here...</p>
        </div>

        <div className="grid-item info-section">
          <h3>Section 2</h3>
          <p>More content...</p>
        </div>
      </div>
    </div>
  );
};

export default IndivCameradisplay;
