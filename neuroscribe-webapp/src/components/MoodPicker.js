import React, { useState, useEffect } from 'react';
import '../styles/ColorPicker.css';

const MoodPicker = ({ onMoodDetermined }) => {
  const [inputValue, setInputValue] = useState(5); 
  const [showLine, setShowLine] = useState(false);

  const wait = async () => {
    return new Promise(resolve => {
      setTimeout(() => {
        resolve(0.4); 
      }, 3000); 
    });
  };

  useEffect(() => {
    const getModelPrediction = async () => {
      const result = await wait();
      setInputValue(result);
      setShowLine(true); 
      const determinedMood = result > 0.5 ? "Positive" : result === 0.5 ? "Neutral" : "Negative";
      onMoodDetermined(determinedMood); 
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
