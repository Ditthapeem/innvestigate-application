import React, { useState, useEffect } from "react";
import axios from "axios";
import configData from "../config";

import NavBar from '../components/NavBar';

import '../assets/Visualize.css';


const Visualize = () => {

    let [index, SetIndex] = useState(null)

    let [finishIndex, setFinishIndex] = useState(0)

    let [visualize, SetVisualize] = useState(null)
    let [listVisualize, SetListVisualize] = useState([])

    let [data, setData] = useState(JSON.parse(localStorage.getItem("data")))
    let [shareConstant, setShareConstant] = useState(JSON.parse(localStorage.getItem("share")))

    let [selectFilterImage, setSelectFilterImage] = useState([])

    let [showSelect, setShowSelect] = useState(false)

    async function visualizeApi(index) {
        const url = `${configData.API.PATH}visualize`
        await axios.post(url, JSON.stringify(data[index]), {
                    headers: {
                        'Content-Type': 'application/json'
                }}).then((res) => {
                    finishIndex += 1
                    setFinishIndex(finishIndex)
                    console.log(res.data)
                    listVisualize.push(res.data.content)
                    console.log(res.data.content.output_images)
                }).catch((error) => {
                    alert(`Error occurred on visualize(${index-1})  image(s).`);
                })
    }

    function applyPostProcess(index) {
        console.log(data[index])
        console.log(shareConstant[index])
        if (data[index].method === "gradient"){
            data[index].gradient_postprocess = String(shareConstant[index]) 
        } else if (data[index].method === "smoothgrad") {
            data[index].smoothgrad_postprocess = String(shareConstant[index]) 
        } else if (data[index].method === "integrated_gradients"){
            data[index].integrated_gradients_postprocess = String(shareConstant[index]) 
        }
        console.log("data", data[index])
    }

    function updateNullValues(index) {
        Object.keys(data[index]).forEach(keys => {
            if(data[index][keys] === null){
                data[index][keys] = ""
            }
        })
    };

    useEffect(() => {
        console.log(data)
        console.log(shareConstant)
        for(let i = 0; i < data.length;i++){
            applyPostProcess(i);
            updateNullValues(i);
            // visualizeApi(i);
        }
    },[])

    const handleClick = (item) => {
        console.log(selectFilterImage)
        if(selectFilterImage.length === 0){
            console.log("empty");
            setSelectFilterImage([...selectFilterImage, item])
        }else{
            if(selectFilterImage.some(e => e.output_image == item.output_image)){
                setSelectFilterImage(selectFilterImage.filter(e =>
                    e.output_image != item.output_image
                ));
            }else{
                setSelectFilterImage([...selectFilterImage, item])
            }
        }
    };

    function handleVisulaize(index) {
        SetVisualize(listVisualize[index])
    }

    function VisualizeButton() {
        console.log(index)
        return(
            <div>
                <center><select value={index} onChange={(e) => {SetIndex(e.target.value); handleVisulaize(e.target.value)}}>
                    <option value={null}>Select an Visualize</option>
                    {data.map((option, index) => (
                    <option key={index} value={index}>
                        {index+1}
                    </option>
                    ))}
                </select></center><br/>
            </div>
        );
    }

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
                            src={`/images/input/${data.input_image}`}/>
                            <p>{data.input_image}</p>
                        </div>
                        <div className="output-image" >
                            <img
                            src={`/images/output/${data.output_image}`}/>
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
                            style={ selectFilterImage.some(e => e.output_image == visualize.visualizations[0].output_images[index]) ? { background: configData.COLOR.GREEN, color: "white"} : {}}
                            onClick={() => handleClick({input_image: image,
                                                        output_image: visualize.visualizations[0].output_images[index],
                                                        prediction: visualize.visualizations[0].predictions[index]
                            })}
                            >
                                <img
                                style={{width: "100%"}}
                                src={`/images/input/${image}`}/>
                                <p>{image}</p>
                            </div>
                            <div className="output-image" >
                                <img
                                src={`/images/output/${visualize.visualizations[0].output_images[index]}`}/>
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

    function Debug() {
        console.log(data)
        console.log(visualize)
        console.log(selectFilterImage)
    }

    return(
        <div>
            <NavBar></NavBar>
                <div className="information">
                    {showSelect == false ? 
                    <div>
                        <h1>Visualization</h1>
                        <VisualizeButton/>
                        { index != null &&
                        <div>
                            <ul>
                                <li>Model: {data[index].predefined_models[0]}</li>
                                <li>Method: {data[index].method}</li>
                                <li>Layer name: {data[index].layer_name}</li>
                                <li>Number of Images: {data[index].images.length}</li>
                            </ul>
                            <button 
                            style={showSelect ? { background: configData.COLOR.GREEN, color: "white"} : {}}
                            onClick={e => setShowSelect(!showSelect)}>
                                Show Collection
                            </button>
                        </div>
                        }
                    </div> : 
                    <div>
                        <h1>Visualization</h1>
                        <ul>
                            <li>Model:  
                                {" "}
                                {data.map((item, index) => (
                                    <div>
                                        <ul>
                                            <li>{item.predefined_models[0]}</li>
                                            {index !== data.length - 1 && ' '}
                                        </ul>
                                    </div>
                                ))}
                            </li>
                            <li>Method: 
                                {" "}
                                {data.map((item) => (
                                    <div>
                                        <ul>
                                            <li>{item.method}</li>
                                            {index !== data.length - 1 && ' '}
                                        </ul>
                                    </div>
                                ))}
                            </li>
                            <li>Layer name:
                                {" "} 
                                {data.map((item) => (
                                    <div>
                                        <ul>
                                            <li>{item.layer_name}</li>
                                            {index !== data.length - 1 && ' '}
                                        </ul>
                                    </div>
                                ))}
                            </li>
                            <li>Number of Images: {data[0].images.length}</li>
                        </ul>
                        <button 
                            style={showSelect ? { background: configData.COLOR.GREEN, color: "white"} : {}}
                            onClick={e => setShowSelect(!showSelect)}>
                                Show Collection
                        </button>
                    </div>}
                </div>
                { finishIndex === data.length ?
                    <div className="image-colum">
                        <ShowImage/>
                    </div>
                    : 
                    <div className="status">
                        <h1>Visualizing Please Wait {finishIndex}/{data.length}</h1>
                    </div>
                }
            <Debug/>
        </div>
    );
} 

export default Visualize
