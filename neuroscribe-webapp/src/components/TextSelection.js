import React, { useState, useEffect } from 'react';
import '../styles/TextSelection.css';

function TextSelection({ mood }) {
  // Define the selections for each mood
  const selections = {
    Positive: [
      { 38: "I", 40: "They" }, // Box 1
      { 38: "Next", 40: "Work" }, // Box 2
      // Add additional selection pairs for Boxes 3, 4, and 5
      { 38: "Here", 40: "Now" }, // Box 3
      { 38: "Always", 40: "Never" }, // Box 4
      { 38: "Yes", 40: "No" }, // Box 5
    ],
    Negative: [
        { 38: "NEG", 40: "They" }, // Box 1
        { 38: "Next", 40: "Work" }, // Box 2
        // Add additional selection pairs for Boxes 3, 4, and 5
        { 38: "Here", 40: "Now" }, // Box 3
        { 38: "Always", 40: "Never" }, // Box 4
        { 38: "Yes", 40: "No" }, // Box 5
    ],
    Neutral: [
        { 38: "NEU", 40: "They" }, // Box 1
        { 38: "Next", 40: "Work" }, // Box 2
        // Add additional selection pairs for Boxes 3, 4, and 5
        { 38: "Here", 40: "Now" }, // Box 3
        { 38: "Always", 40: "Never" }, // Box 4
        { 38: "Yes", 40: "No" }, // Box 5
    ]
  };

  // States for the current selections and locks for each box
  const [currentSelections, setCurrentSelections] = useState(Array(5).fill("ㅤ"));
  const [boxesLocked, setBoxesLocked] = useState(Array(5).fill(false));

  // Determine the color based on the mood
  const boxColor = mood === 'Positive' ? '#ffdb5d' : mood === 'Negative' ? '#72f3f3' : '#FFFFFF';

  // Style object for the selection boxes
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
    // Find the first box that is not locked
    const boxToChange = boxesLocked.findIndex(locked => !locked);
    if (boxToChange === -1) return; // All boxes are locked

    // Up Arrow (38) and Down Arrow (40) to change the selection
    if (e.keyCode === 38 || e.keyCode === 40) {
      setCurrentSelections(current => 
        current.map((sel, index) => 
          index === boxToChange ? selections[mood][boxToChange][e.keyCode] : sel
        )
      );
    }

    // Left Arrow (37) to lock the current box and move to the next one
    if (e.keyCode === 37) {
      setBoxesLocked(current => 
        current.map((locked, index) => 
          index === boxToChange ? true : locked
        )
      );
    }
  };

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [boxesLocked, selections, mood]);

  return (
    <div className='text-select-container'>
      {/* ... mood display ... */}
      <div className='selections-container' style={{ display: 'flex' }}>
        {selections[mood].map((boxSelections, index) => (
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
