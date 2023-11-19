import React, { useState, useEffect } from 'react';
import '../styles/TextSelection.css';

function TextSelection({ mood }) {
  let HappyList = ['I', "Want", "They", "She", "He", "Want", "To", "Throw", "Say", "Need", "Desire", "Run", "Away", "Because"];

  // Define the selections for each mood
  const selections = {
    Positive: [
      { 38: HappyList[0], 40: HappyList[1] }, // Box 1
      { 38: HappyList[0], 40: HappyList[1] }, // Box 2
      { 38: HappyList[0], 40: HappyList[1] }, // Box 3
      { 38: HappyList[0], 40: HappyList[1] }, // Box 4
      { 38: HappyList[0], 40: HappyList[1] }, // Box 5
      { 38: HappyList[0], 40: HappyList[1] }, // Box 6
      { 38: HappyList[0], 40: HappyList[1] }, // Box 7

    ],
    Negative: [
      // Define Negative selections here
      { 38: HappyList[0], 40: HappyList[1] }, // Box 1
      { 38: HappyList[0], 40: HappyList[1] }, // Box 2
      { 38: HappyList[0], 40: HappyList[1] }, // Box 3
      { 38: HappyList[0], 40: HappyList[1] }, // Box 4
      { 38: HappyList[0], 40: HappyList[1] }, // Box 5
      { 38: HappyList[0], 40: HappyList[1] }, // Box 6
      { 38: HappyList[0], 40: HappyList[1] }, // Box 7
    ],
    Neutral: [
      // Define Neutral selections here
      { 38: HappyList[0], 40: HappyList[1] }, // Box 1
      { 38: HappyList[0], 40: HappyList[1] }, // Box 2
      { 38: HappyList[0], 40: HappyList[1] }, // Box 3
      { 38: HappyList[0], 40: HappyList[1] }, // Box 4
      { 38: HappyList[0], 40: HappyList[1] }, // Box 5
      { 38: HappyList[0], 40: HappyList[1] }, // Box 6
      { 38: HappyList[0], 40: HappyList[1] }, // Box 7
    ]
  };

  const [currentSelections, setCurrentSelections] = useState(Array(7).fill("ㅤ"));
  const [boxesLocked, setBoxesLocked] = useState(Array(7).fill(false));
  const [customSelections, setCustomSelections] = useState(selections[mood]);

  const boxColor = mood === 'Positive' ? '#ffdb5d' : mood === 'Negative' ? '#72f3f3' : '#FFFFFF';

  const selectionBoxStyle = {
    backgroundColor: boxColor,
    color: 'rgb(19, 20, 20)',
    border: '2px solid #000',
    borderRadius: '15px',
    padding: '5px 10px',
    display: 'inline-block',
    margin: '5px',
    marginLeft: '0.4rem',
    marginBottom: '2.4rem',
    marginTop: '2.2rem',
  };

  const handleKeyDown = (e) => {
    const boxToChange = boxesLocked.findIndex(locked => !locked);
    if (boxToChange === -1) return; // All boxes are locked
  
    if (e.keyCode === 38 || e.keyCode === 40) {
      setCurrentSelections(current => 
        current.map((sel, index) => 
          index === boxToChange ? customSelections[boxToChange][e.keyCode] : sel
        )
      );
    } else if (e.keyCode === 39) { // Right arrow key to lock the box
      setBoxesLocked(current => 
        current.map((locked, index) => 
          index === boxToChange ? true : locked
        )
      );
    } else if (e.keyCode === 37) { // Left arrow key to change the option
      const randomIndex1 = Math.floor(Math.random() * HappyList.length);
      const randomIndex2 = Math.floor(Math.random() * HappyList.length);
  
      const newSelections = { 38: HappyList[randomIndex1], 40: HappyList[randomIndex2] };
  
      setCustomSelections(current => 
        current.map((sel, index) => 
          index === boxToChange ? newSelections : sel
        )
      );
      setCurrentSelections(current => 
        current.map((sel, index) => 
          index === boxToChange ? newSelections[38] : sel // Set the current selection to match the new "above" selection
        )
      );
    }
  };
  
  

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [boxesLocked, customSelections, mood, HappyList]);

  return (
    <div className='text-select-container'>
      <div className='selections-container' style={{ display: 'flex' }}>
        {customSelections.map((boxSelections, index) => (
          <div key={index} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            {index <= boxesLocked.lastIndexOf(true) + 1 && (
              <>
                <div style={selectionBoxStyle} className='selection-box-above'>{boxSelections[38]}</div>
                <div className="selection-box">
                  <u className='underline'>{currentSelections[index]}</u>
                </div>
                <div style={selectionBoxStyle} className='selection-box-below'>{boxSelections[40]}</div>
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}


export default TextSelection;
