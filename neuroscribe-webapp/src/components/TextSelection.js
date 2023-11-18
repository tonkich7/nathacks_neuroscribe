import React, { useState, useEffect } from 'react';
import '../styles/TextSelection.css';

function TextSelection({ mood }) {
  const PositiveSelections1 = {
    38: "I",    // Up Arrow
    40: "They",  // Down Arrow
  };

  const positiveSelections2 = {
    38: "Next",
    40: "Work",
  };

  const NegativeSelections1 = {
    38: "Bad",    // Up Arrow
    40: "-",  // Down Arrow
  };

  const [currentSelection1, setCurrentSelection1] = useState("ㅤ ㅤ ㅤ ㅤ ");
  const [currentSelection2, setCurrentSelection2] = useState("ㅤ ㅤ ㅤ ㅤ ");
  const [firstBoxLocked, setFirstBoxLocked] = useState(false);
  const [secondBoxLocked, setSecondBoxLocked] = useState(false);

  let firstStarterSentence, secondStarterSentence;

  if (mood === 'Positive') {
    firstStarterSentence = PositiveSelections1;
    secondStarterSentence = positiveSelections2;
  } else if (mood === 'Negative') {
    firstStarterSentence = NegativeSelections1;
    secondStarterSentence = positiveSelections2;
  } else if (mood === "Neutral") {
    firstStarterSentence = PositiveSelections1;
    secondStarterSentence = PositiveSelections1;
  }

  // Determine the color based on the mood
  let boxColor;
  if (mood === 'Positive') {
    boxColor = '#ffdb5d'; // Color for Positive mood
  } else if (mood === 'Negative') {
    boxColor = '#72f3f3'; // Color for Negative mood
  } else if (mood === 'Neutral') {
    boxColor = '#FFFFFF'; // Color for Neutral mood
  }

  // Style object for the selection boxes
  const selectionBoxStyle = {
    backgroundColor: boxColor,
    color: 'rgb(19, 20, 20)',
    border: '2px solid #000',
    borderRadius: '15px',
    padding: '5px 10px',    
    display: 'inline-block',
    margin: '5px',
    marginLeft: mood === 'Positive' ? '2.4rem' : '4.5rem', // Adjust margin based on mood
    marginBottom: mood === 'Positive' ? '1rem' : undefined,
    marginTop: mood === 'Neutral' ? '1rem' : undefined,
  };

  const handleKeyDown = (e) => {
    if (!firstBoxLocked) {
      if (firstStarterSentence[e.keyCode]) {
        setCurrentSelection1(firstStarterSentence[e.keyCode]);
      }
      if (e.keyCode === 37) { // Left Arrow
        setFirstBoxLocked(true);
      }
    } else if (!secondBoxLocked) {
      if (secondStarterSentence[e.keyCode]) {
        setCurrentSelection2(secondStarterSentence[e.keyCode]);
      }
      if (e.keyCode === 37) { // Left Arrow
        setSecondBoxLocked(true);
      }
    }
  };

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [firstBoxLocked, secondBoxLocked]);

  return (
    <div className='text-select-container'>
        <div className='mood-text-container'>
            <div className='mood-text'>
                {/* {mood} */}
            </div>
        </div>
        <div className='selections-container'>
            <div style={selectionBoxStyle} className='selection-box-above'>{firstStarterSentence[38]}</div>
            <div className='mood-text-container'>
                <div className='sentences'>
                    <div className="selection-box"><u className='underline'>{currentSelection1}</u></div> want(s) 
                    {firstBoxLocked && <div className="selection-box"><u className='underline'>{currentSelection2}</u></div>}
                </div>
            </div>
            <div style={selectionBoxStyle} className='selection-box-below'>{firstStarterSentence[40]}</div>
        </div>
    </div>
  );
}

export default TextSelection;
