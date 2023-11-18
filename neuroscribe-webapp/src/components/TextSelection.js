import React from 'react';
import '../styles/TextSelection.css';

function TextSelection({ mood }) {
  return (
    <div className='text-select-container'>
      <div className='mood-text'>
        Component Result: {mood}
        </div>
     
    </div>
  );
}

export default TextSelection;
