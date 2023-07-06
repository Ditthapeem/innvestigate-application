import React, { useState, useEffect } from "react";
import axios from "axios";
import configData from "../config";

import NavBar from '../components/NavBar';

import '../assets/Visualize.css';


const Visualize = () => {

    let [visualize, SetVisualize] = useState(null)

    let [data, setData] = useState(JSON.parse(localStorage.getItem("data")))
    let [shareConstant, setShareConstant] = useState(JSON.parse(localStorage.getItem("share")))

    let [selectFilterImage, setSelectFilterImage] = useState([])

    let [showSelect, setShowSelect] = useState(false)

    const visualizeApi = async(e) => {
        const url = `${configData.API.PATH}visualize`
        await axios.post(url, JSON.stringify(data), {
                    headers: {
                        'Content-Type': 'application/json'
                }}).then((res) => {
                    console.log(res.data)
                    SetVisualize(res.data.content)
                    console.log(res.data.content.output_images)

                })
    }

    useEffect(() => {
        applyPostProcess();
        updateNullValues();
        visualizeApi();
    },[])

    function updateNullValues() {
        Object.keys(data).forEach(keys => {
            if(data[keys] === null){
                data[keys] = ""
            }
        })
    };

    const handleClick = (item) => {
        if(selectFilterImage.length === 0){
            console.log("empty");
            setSelectFilterImage([...selectFilterImage, item])
        }else{
            if(selectFilterImage.some(e => e.input_image == item.input_image)){
                setSelectFilterImage(selectFilterImage.filter(e =>
                    e.input_image != item.input_image
                ));
            }else{
                setSelectFilterImage([...selectFilterImage, item])
            }
        }
    };

    function ShowImage() {
        if(showSelect){
            return(
                <div>
                    {selectFilterImage.map((data,index) => (
                    <div>
                        <div 
                        className="input-image"
                        style={{background: configData.COLOR.GREEN, color: "white"}}
                        >
                            <img
                            style={{width: "100%"}}
                            src={`images/input/${data.input_image}`}/>
                            <p>{data.input_image}</p>
                        </div>
                        <div className="output-image" >
                            <img
                            src={`images/output/${data.output_image}`}/>
                            <p>{data.input_image}</p>
                            <h1>Model Prediction</h1>
                            <ul>
                                {/* <li>{`${visualize.visualizations[0].max_activations_neuron_indexes_per_image[index]}`}</li> */}
                                <li>{`${data.prediction[0].className} (${data.prediction[0].classId}) with probability  ${data.prediction[0].score}`}</li>
                                <li>{`${data.prediction[1].className} (${data.prediction[1].classId}) with probability  ${data.prediction[1].score}`}</li>
                                <li>{`${data.prediction[2].className} (${data.prediction[2].classId}) with probability  ${data.prediction[2].score}`}</li>
                                <li>{`${data.prediction[3].className} (${data.prediction[3].classId}) with probability  ${data.prediction[3].score}`}</li>
                                <li>{`${data.prediction[4].className} (${data.prediction[4].classId}) with probability  ${data.prediction[4].score}`}</li>
                            </ul>
                        </div>
                    </div>
                    ))}
                </div>
            );
        }else{
            return(
                <div> 
                    {visualize != null && visualize.input_images.map((image, index) => (
                        <div>
                            <div 
                            className="input-image"
                            style={ selectFilterImage.some(e => e.input_image == image) ? { background: configData.COLOR.GREEN, color: "white"} : {}}
                            onClick={() => handleClick({input_image: image,
                                                        output_image: visualize.visualizations[0].output_images[index],
                                                        prediction: visualize.visualizations[0].predictions[index]
                            })}
                            >
                                <img
                                style={{width: "100%"}}
                                src={`images/input/${image}`}/>
                                <p>{image}</p>
                            </div>
                            <div className="output-image" >
                                <img
                                src={`images/output/${visualize.visualizations[0].output_images[index]}`}/>
                                <p>{image}</p>
                                <h1>Model Prediction</h1>
                                <ul>
                                    {/* <li>{`${visualize.visualizations[0].max_activations_neuron_indexes_per_image[index]}`}</li> */}
                                    <li>{`${visualize.visualizations[0].predictions[index][0].className} (${visualize.visualizations[0].predictions[index][0].classId}) with probability  ${visualize.visualizations[0].predictions[index][0].score}`}</li>
                                    <li>{`${visualize.visualizations[0].predictions[index][1].className} (${visualize.visualizations[0].predictions[index][1].classId}) with probability  ${visualize.visualizations[0].predictions[index][1].score}`}</li>
                                    <li>{`${visualize.visualizations[0].predictions[index][2].className} (${visualize.visualizations[0].predictions[index][2].classId}) with probability  ${visualize.visualizations[0].predictions[index][2].score}`}</li>
                                    <li>{`${visualize.visualizations[0].predictions[index][3].className} (${visualize.visualizations[0].predictions[index][3].classId}) with probability  ${visualize.visualizations[0].predictions[index][3].score}`}</li>
                                    <li>{`${visualize.visualizations[0].predictions[index][4].className} (${visualize.visualizations[0].predictions[index][4].classId}) with probability  ${visualize.visualizations[0].predictions[index][4].score}`}</li>
                                </ul>
                            </div>
                        </div>
                    ))}
                </div>
            );
        }
    }

    function applyPostProcess() {
        console.log(data.method)
        console.log(shareConstant.postProcess)
        if (data.method === "gradient"){
            data.gradient_postprocess = shareConstant.postProcess
        } else if (data.method === "smoothgrad") {
            data.smoothgrad_postprocess = shareConstant.postProcess
        } else if (data.method === "integrated_gradients"){
            data.integrated_gradients_postprocess = shareConstant.postProcess
        }
    }

    function Debug() {
        console.log(data)
        console.log(visualize)
        console.log(selectFilterImage)
    }

    return(
        <div>
            <NavBar></NavBar>
                <div className="information">
                        <h1>Visualization</h1>
                        <ul>
                            <li>Model: {data.predefined_models[0]}</li>
                            <li>Method: {data.method}</li>
                            <li>Layer name: {data.layer_name}</li>
                            <li>Number of Images: {data.images.length}</li>
                        </ul>
                        <button 
                        style={showSelect ? { background: configData.COLOR.GREEN, color: "white"} : {}}
                        onClick={e => setShowSelect(!showSelect)}>
                            Show Collection
                        </button>
                </div>
                <div className="image-colum">
                    <ShowImage/>
                </div>
            <Debug/>
        </div>
    );
} 

export default Visualize