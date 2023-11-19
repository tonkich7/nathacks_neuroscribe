import React, { useState, useEffect } from 'react';
import '../styles/TextSelection.css';

function TextSelection({ mood }) {
  const [HappyList, setHappyList] = useState([]);
  const [currentSelections, setCurrentSelections] = useState(Array(7).fill(null));
  const [boxesLocked, setBoxesLocked] = useState(Array(7).fill(false));
  const [currentIndex, setCurrentIndex] = useState(0);

  const fetchWords = async () => {
    let url;
    switch (mood) {
      case 'Positive':
        url = 'http://127.0.0.1:5000/get-positive-words';
        break;
      case 'Negative':
        url = 'http://127.0.0.1:5000/get-negative-words';
        break;
      case 'Neutral':
        url = 'http://127.0.0.1:5000/get-neutral-words';
        break;
      default:
        console.error("Invalid mood");
        return;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setHappyList([data.word_1, data.word_2]);
    } catch (error) {
      console.error("Error fetching words:", error);
    }
  };

  const handleKeyDown = (e) => {
    if (currentIndex >= currentSelections.length) return; // Exit if all boxes are displayed

    console.log('Handling key down:', e.keyCode);

    if (e.keyCode === 38) { // Up arrow key
      setCurrentSelections((current) =>
        current.map((sel, index) =>
          index === currentIndex ? HappyList[0] : sel
        )
      );
    } else if (e.keyCode === 40) { // Down arrow key
      setCurrentSelections((current) =>
        current.map((sel, index) =>
          index === currentIndex ? HappyList[1] : sel
        )
      );
    } else if (e.keyCode === 39) { // Right arrow key to lock the box and move to the next
      setBoxesLocked((current) =>
        current.map((locked, index) =>
          index === currentIndex ? true : locked
        )
      );
      setCurrentIndex((index) => index + 1);
    } else if (e.keyCode === 37) { // Left arrow key to change the option
      fetchWords();
    }
  };

  useEffect(() => {
    fetchWords();
  }, [mood]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentIndex, currentSelections, mood]);

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

  return (
    <div className='text-select-container'>
      <div className='selections-container' style={{ display: 'flex' }}>
        {currentSelections.map((word, index) => (
          <div key={index} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            {index <= currentIndex && (
              <>
                <div style={selectionBoxStyle} className='selection-box-above'>{HappyList[0]}</div>
                <div className="selection-box">
                  <u className='underline'>{word}</u>
                </div>
                <div style={selectionBoxStyle} className='selection-box-below'>{HappyList[1]}</div>
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default TextSelection;
