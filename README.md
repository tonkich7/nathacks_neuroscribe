# NeuroScribe - natHACKS 2023
_a hands-free tool that harnesses EEG waves for mood and speech communication, giving back voices to individuals who have lost it._


https://github.com/tonkich7/nathacks_neuroscribe/assets/100548563/a79a0677-4ed3-45b0-9dab-f54a65f3a0e9






“NeuroScribe” is a tool that harnesses EEG waves for mood and speech communication, giving voices to individuals who have lost it. There are many disorders that impact motor neurons across the entire peripheral body, impairing fine and gross motor control from walking to speaking.

Take “Davie” for example. He was diagnosed with Amyotrophic Lateral Sclerosis (ALS) and gradually lost all motor function, including the ability to walk and speak. He didn't just lose his ability to communicate; he also lost his ability to make facial expressions. A lot of patients report that they feel their identity and self expression is hindered with diagnosis that impair their independence.

We aren’t creating static text, but emulating the different dimensions of interpersonal communications such as facial expression, utilizing mood detection. Firstly, mood will be displayed to the listener, allowing individuals that lack the ability to convey tone and expression to do so. Mood detection also serves as a precursor to speech creation, allowing us to design a swifter and more efficient interface by predicting words tailored to the detected emotional state. Specifically, three models will be created, representing negative, neutral, and positive moods.

Word selection will be completely hands free, as a third model will recognize imagined directional cues. The user will only need to imagine pushing the letter block on the screen "up," "down," "right," and " left," towards a selection, making the interface accessible for all levels of motor impairment.

## Presentation Slideshow
<img src="https://github.com/tonkich7/nathacks_neuroscribe/assets/100548563/41e94559-5355-4bca-9090-40323baf9733" width="800" height="400">

<img src="https://github.com/tonkich7/nathacks_neuroscribe/assets/100548563/469cfa00-3e5e-4ab1-af73-9b1a66c8fc76" width="800" height="400">

<img src="https://github.com/tonkich7/nathacks_neuroscribe/assets/100548563/b6ec7bcf-4cd5-466f-afe1-c992d83dbafd" width="800" height="400">

<img src="https://github.com/tonkich7/nathacks_neuroscribe/assets/100548563/ac11b05d-33a3-4317-a768-cebf4728e1ff" width="800" height="400">

<img src="https://github.com/tonkich7/nathacks_neuroscribe/assets/100548563/a27e9c8e-53a8-46bf-acae-22e079be4f06" width="800" height="400">

<img src="https://github.com/tonkich7/nathacks_neuroscribe/assets/100548563/27c00439-e91d-4dd4-89e4-2e6cb95fb477" width="800" height="400">

<img src="https://github.com/tonkich7/nathacks_neuroscribe/assets/100548563/b2a04a25-bf9d-4a50-9a53-0aa104e13eed" width="800" height="400">

<img src="https://github.com/tonkich7/nathacks_neuroscribe/assets/100548563/87fac3c5-945f-4445-af39-4ed7fbb3c5f4" width="800" height="400">

<img src="https://github.com/tonkich7/nathacks_neuroscribe/assets/100548563/a976ff58-411e-4981-bef6-9bd4891a815b" width="800" height="400">

## How to Use NeuroScribe
To use NeuroScribe you must have the following installed:
1. Python 3.x and the following libraries:
  * Brainflow
  * PyTorch
  * NumPy
  * MNE
  * Flask
  * Pandas
  * TensorFlow
2. Node.js
  * npm

After all of these are installed:
1. From the main directory, enter the neuroscribe-webapp folder
2. Launch your terminal/command prompt from within this folder
3. Type "npm install" and wait for everything to install
4. Type "npm start", this will launch the reactjs front end
5. Go back one directory in your terminal
6. If you are using an OpenBCI Ganglion, open up app.py and change the SERIAL_PORT variable to the port your device is connected to
7. Type "py app.py", this will launch the python back end
8. In a separate tab in your browser go to http://localhost:5000/setup and wait for the page to load
9. Refresh http://localhost:3000/
10. Ta-da! You can now write with your brain.




