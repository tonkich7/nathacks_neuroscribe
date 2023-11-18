// import logo from './logo.svg';
import './App.css';
import MoodPicker from './components/MoodPicker'; // Import the component

function App() {
  return (
    
    <div className="App">
      <div className='neuroscribe-logo'>
        {/* NeuroScribe */}
        <img src="nathacks2023logo.png" alt="Our Logo" width="120" height="100"></img>
      </div>
      <div className='people'>
        NatHacks 2023
      </div>
      {/* <div className='people'>
        Harrison, Hilary, Rose, Kevin, Alvin
      </div> */}

      <MoodPicker />
    </div>
  
  );
}

export default App;
