import {Route, Routes } from 'react-router-dom'

import Home from './pages/Home'
import History from './pages/History';
import Visulize from './pages/Visualize';
import Suggest from './pages/Suggest';

function App() {
  return(
    <Routes>
      <Route path="/home" element={<Home/>}/>
      <Route path="/history" element={<History/>}/>
      <Route path='/visualize' element={<Visulize/>}/>
      <Route path='/suggest' element={<Suggest/>}/>
    </Routes>
  );
}

export default App;