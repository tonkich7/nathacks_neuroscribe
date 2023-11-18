import React, { useState, useEffect } from 'react';
import '../styles/ColorPicker.css'; // Make sure this path is correct

const MoodPicker = () => {
  const [inputValue, setInputValue] = useState(5); // Initialize state with 5 for "Detecting..."
  const [showLine, setShowLine] = useState(false); // New state to manage the line display

  // Simulated async function to mimic a ML model response
  const wait = async () => {
    return new Promise(resolve => {
      setTimeout(() => {
        resolve(0.4); // Simulating a response after a delay
      }, 3000); // Delay in milliseconds
    });
  };

  useEffect(() => {
    const getModelPrediction = async () => {
      const result = await wait();
      setInputValue(result);
      setShowLine(true); // Show the line once the mood is determined
    };
    getModelPrediction();
  }, []);

  // Determine the mood based on the input value
  const mood = inputValue === 5 ? "Detecting..." : inputValue > 0.5 ? "Positive" : inputValue === 0.5 ? "Neutral" : "Negative";

  // Style object for the mood
  const moodStyle = {
    color: mood === "Positive" ? "#ffdb5d" : mood === "Negative" ? "#72f3f3" : mood === "Neutral" ? "#f5f5dc" : "black",
  };

// Inside your MoodPicker component

  return (
    <div className='mood-container'>
      <div className='measure-text'>
        <b>Mood detected : <span style={mood !== "Detecting..." ? moodStyle : {}}>{mood}</span></b>
      </div>
      {/* Add the visible class to mood-line when the mood is determined */}
      <div className={`mood-line ${showLine && mood !== "Detecting..." ? 'visible' : ''}`}></div>
    </div>
);

  
};

export default MoodPicker;
