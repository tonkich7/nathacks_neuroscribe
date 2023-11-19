import React, { useState, useEffect } from 'react';
import '../styles/ColorPicker.css';

const MoodPicker = ({ onMoodDetermined }) => {
  const [inputValue, setInputValue] = useState(5); 
  const [showLine, setShowLine] = useState(false);

  // Function to fetch mood data from the Flask backend
  const fetchMoodData = async () => {
    return new Promise((resolve, reject) => {
      setTimeout(async () => {
        try {
          const response = await fetch('http://127.0.0.1:5000/get-json');
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();
          console.log("HERE: ",data.mood)
          resolve(data.mood); // Assuming the response format is {"mood":1}
        } catch (e) {
          console.error('Fetch failed:', e.message);
          reject(null); // Reject with null in case of error
        }
      }, 4000); // Wait for 4 seconds before fetching
    });
  };

  useEffect(() => {
    const getModelPrediction = async () => {
      const moodValue = await fetchMoodData();
      if (moodValue !== null) {
        setInputValue(moodValue);
        setShowLine(true); 
        const determinedMood = moodValue > 0.5 ? "Positive" : moodValue === 0.5 ? "Neutral" : "Negative";
        onMoodDetermined(determinedMood); 
      }
    };
    getModelPrediction();
  }, [onMoodDetermined]);

  const mood = inputValue === 5 ? "Detecting..." : inputValue > 0.5 ? "Positive" : inputValue === 0.5 ? "Neutral" : "Negative";

  const moodStyle = {
    color: mood === "Positive" ? "#ffdb5d" : mood === "Negative" ? "#72f3f3" : mood === "Neutral" ? "#ffffff" : "black",
  };

  return (
    <div className='mood-container'>
      <div className='measure-text'>
        <b>Mood detected : <span style={mood !== "Detecting..." ? moodStyle : {}}>{mood}</span></b>
      </div>
      <div className={`mood-line ${showLine && mood !== "Detecting..." ? 'visible' : ''}`}></div>
    </div>
  );
};

export default MoodPicker;
