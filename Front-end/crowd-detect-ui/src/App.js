import { useState } from "react";
import "./App.css";
import Cameracomp from "./components/Cameracomp";
import Cameradisplay from "./components/Cameradisplay";
import Navbar from "./components/Navbar";
import IndivCameradispaly from "./components/IndivCameradispaly";

function App() {
  const [selectedVideo, setSelectedVideo] = useState(0);

  return (
    <div className="App">
      <Navbar />
      {selectedVideo ? (
        <IndivCameradispaly
          video={selectedVideo}
          onBack={() => setSelectedVideo(0)}
        />
      ) : (
        <Cameradisplay>
          {[...Array(6)].map((_, index) => (
            <Cameracomp
              key={index}
              onClick={() => setSelectedVideo(index + 1)}
            />
          ))}
        </Cameradisplay>
      )}
    </div>
  );
}

export default App;
