import React from "react";
import PredictForm from "./components/PredictForm";
import GraphView from "./components/GraphView";

function App(){
  return (
    <div>
      <header style={{padding:20, background:'#0b74de', color:'#fff'}}>
        <h1>Academic Collaboration Predictor</h1>
      </header>
      <main>
        <PredictForm/>
        <GraphView/>
      </main>
    </div>
  );
}

export default App;
