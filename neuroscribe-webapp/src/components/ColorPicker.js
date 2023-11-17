import React, { useState } from 'react';
import '../styles/ColorPicker.css';

const ColorPicker = () => {
  const [inputValue, setInputValue] = useState(5); // Initialize state

  // Function to handle input changes
  const handleInputChange = (event) => {
    setInputValue('event.target.value');
  };


  
  const mood = inputValue === 5 ? "Detecting..." : inputValue > 0.5 ? "Positive" : "Negative";
  
  

  return (
    <div>
      <div className='measure-text'>
        Mood Detected: {mood}
      </div>
    </div>
  );
};

export default ColorPicker;
