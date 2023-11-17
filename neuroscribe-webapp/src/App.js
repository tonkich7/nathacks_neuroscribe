// import logo from './logo.svg';
import './App.css';
import ColorPicker from './components/ColorPicker'; // Import the component

function App() {
  return (
    
    <div className="App">
      <div className='neuroscribe-logo'>
        NeuroScribe
      </div>
      <div className='people'>
        Harrison, Hilary, Rose, Kevin, Alvin
      </div>

      <ColorPicker />
    </div>
  
  );
}

export default App;
