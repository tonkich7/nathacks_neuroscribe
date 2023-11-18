import React, { useState } from 'react';
import './App.css';
import MoodPicker from './components/MoodPicker';
import TextSelection from './components/TextSelection';

function App() {
  const [mood, setMood] = useState("Detecting...");

  const handleMoodUpdate = (newMood) => {
    setMood(newMood);
  };

  return (
    <div className="App">
      <div className='neuroscribe-logo'>
        <img src="nathacks2023logo.png" alt="Our Logo" width="120" height="100"></img>
      </div>
      <div className='people'>
        NatHacks 2023
      </div>
      <MoodPicker onMoodDetermined={handleMoodUpdate} />
      {mood !== "Detecting..." && <TextSelection mood={mood} />}
    </div>
  );
}

export default App;
