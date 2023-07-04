import React, { useLayoutEffect } from "react";
import axios from 'axios';
import { useNavigate } from 'react-router-dom'

import configData from "../config";
import '../assets/NavBar.css';

const NavBar = () => {
    const navigate = useNavigate();
    const path = window.location.pathname;

	return (
        <div className='nav-bar'>
            <div>
                <p>iNNvestiagte GUI</p>
                <button onClick={()=>{localStorage.state="home";  navigate('/home')}}
                className={path === "/home" ? 'nav-select' : 'nav-not-select'}>Home</button>  
                <button onClick={()=>{localStorage.state="history";  navigate('/history')}}
                className={path === "/history" ? 'nav-select' : 'nav-not-select'}>History</button>    
            </div>
        </div>
    );
}

export default NavBar;