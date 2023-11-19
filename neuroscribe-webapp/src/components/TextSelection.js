import React, { useState, useEffect } from 'react';
import '../styles/TextSelection.css';

function TextSelection({ mood }) {
  const [HappyList, setHappyList] = useState([]);
  
  console.log(mood)

  // Function to fetch new words and update HappyList
  const fetchWords = async () => {
    setTimeout(async () => {
      let url;
      switch (mood) {
        case 'Positive':
          url = 'http://127.0.0.1:5000/get-positive-words';
          console.log(url)
          break;
        case 'Negative':
          url = 'http://127.0.0.1:5000/get-negative-words';
          console.log(url)
          break;
        case 'Neutral':
          url = 'http://127.0.0.1:5000/get-neutral-words';
          console.log(url)
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
    }, 1000); // 1-second delay before fetch
  };
  
  

  // Function to simulate key press based on the model's output
  const simulateKeyPress = (direction) => {
    let keyCode;
    switch (direction) {
      case 'up':
        keyCode = 38;
        break;
      case 'down':
        keyCode = 40;
        break;
      case 'left':
        keyCode = 37;
        break;
      case 'right':
        keyCode = 39;
        break;
      default:
        return; // If the direction is not recognized, do nothing
    }
    handleKeyDown({ keyCode });
  };


  const handleKeyDown = (e) => {
    const boxToChange = boxesLocked.findIndex(locked => !locked);
    if (boxToChange === -1) return; // All boxes are locked
  
    if (e.keyCode === 38) { // Up arrow key
      setCurrentSelections(current => 
        current.map((sel, index) => 
          index === boxToChange ? HappyList[0] : sel
        )
      );
    } else if (e.keyCode === 40) { // Down arrow key
      setCurrentSelections(current => 
        current.map((sel, index) => 
          index === boxToChange ? HappyList[1] : sel
        )
      );
    } else if (e.keyCode === 39) { // Right arrow key to lock the box
      setBoxesLocked(current => 
        current.map((locked, index) => 
          index === boxToChange ? true : locked
        )
      );
      fetchWords(); // Fetch new words after locking a box
    } else if (e.keyCode === 37) { // Left arrow key to change the option
      fetchWords(); // Fetch new words when left arrow key is pressed
    }
  };
  
  
  

  const [currentSelections, setCurrentSelections] = useState(Array(7).fill("ㅤ"));
  const [boxesLocked, setBoxesLocked] = useState(Array(7).fill(false));

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

  useEffect(() => {
    fetchWords();
  }, [mood]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [boxesLocked, HappyList, mood]);

  return (
    <div className='text-select-container'>
      <div className='selections-container' style={{ display: 'flex' }}>
        {currentSelections.map((word, index) => (
          <div key={index} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            {index <= boxesLocked.lastIndexOf(true) + 1 && (
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
