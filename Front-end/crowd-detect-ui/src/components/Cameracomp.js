import React from "react";
import crowdimg from "./crowd_patches.jpg";
import "./Cameracomp.css";
// import crowd from "./Video.mp4";

const Cameracomp = ({onClick}) => {
  return (
    <div className="camera-comp" onClick={onClick}>
      <img src={crowdimg} className='grid-image' /> 
    </div>
  );
};

export default Cameracomp;
