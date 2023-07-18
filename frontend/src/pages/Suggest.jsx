import React, { useState, useEffect } from "react";
import axios from "axios";
import configData from "../config";
import { BarChart, Bar, CartesianGrid, XAxis, YAxis } from 'recharts';

import NavBar from '../components/NavBar';

import '../assets/Suggest.css';

const Suggest = () => {

    let [suggest, SetSuggest] = useState(null)

    let [data, setData] = useState(JSON.parse(localStorage.getItem("data")))
    let [shareConstant, setShareConstant] = useState(JSON.parse(localStorage.getItem("share")))

    let [image, setImage] = useState(null)

    let [score, setScore] = useState(null)

    let [prediction, setPrediction] = useState(null)

    let [selectBin, setSelectBin] = useState({group:null, number: null, indexList: [0]})

    let [group, setGroup] = useState([])

    let [selectFilterImage, setSelectFilterImage] = useState([])

    let [showSelect, setShowSelect] = useState(false)

    let chart = []

    useEffect(() => {
        console.log('data',data)
        data.input_images_belong_to_same_class = 'False'
        data.input_images_class_id = null
        applyPostProcess();
        updateNullValues();
        suggestApi();
    },[])

    function updateNullValues() {
        Object.keys(data).forEach(keys => {
            if(data[keys] === null){
                data[keys] = ""
            }
        })
    };

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

    const suggestApi = async(e) => {
        const url = `${configData.API.PATH}suggest`
        await axios.post(url, JSON.stringify(data), {
                    headers: {
                        'Content-Type': 'application/json'
                }}).then((res) => {
                    console.log(res.data.content[0])
                    SetSuggest(res.data.content[0])
                    setImage(res.data.content[0].input_images.reverse())
                    setPrediction(res.data.content[0].model_predictions.reverse())
                }).catch((error) => {
                    alert(error)
                })
    }

    function GenerateChart(){
        let data = suggest.layer_mean_activations
        let lowerBound = findLowerBound(Math.floor(data[data.length - 1])) 
        let range = findRange(data[0], data[data.length - 1])
        let highestNumber = data[0]
        let bar = 1
        let index = 0

        console.log("boung", (lowerBound*100)/50)
        console.log("range", findRange(0));

        for (let i = lowerBound; i < highestNumber+range; i += range) {
            let count = 0
            data.forEach(element => {
                if(element > i && element < i+range){
                    count +=1
                }
            })
            if(count > 0) {
                let tempDict = {name: `Group${bar}`, numberImg: count}
                chart.push(tempDict)
                index+=count
                if(data.length != index ){
                    if(!selectBin.indexList.includes(index)){
                        console.log("index", index)
                        selectBin.indexList.push(index)
                    }
                }
                if(!group.includes(bar)){
                    group.push(bar)
                }
                bar +=1
                // console.log(count)
            }  
        }
        console.log(selectBin.indexList)

        return(
            <BarChart width={400} height={300} data={chart}>
                    <Bar dataKey="numberImg" fill="green" />
                    <CartesianGrid stroke={configData.COLOR.GREEN} />
                    <XAxis dataKey="name" 
                    label={{ value: 'mean activation at selected layer', position: 'outsideBottom', dy:12 }}/>
                    <YAxis label={{ value: 'Number of image', angle: -90, position: 'outsideLeft' }}/>
                </BarChart>
        );
    }

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

    function findLowerBound(number) {
        const lowerBound = Math.floor(number / 10) * 10;
        return lowerBound;
    }

    function roundToNearestPowerOf10(number) {
        if (number < 0) {
            number = Math.abs(number);
        }
        if (number === 0) {
            return 0.5;
        }

        const power = Math.floor(Math.log10(number));
            return 10 ** power;
        }

    function findRange(max, min) {
        const diff = max - min;
        if (roundToNearestPowerOf10(diff) !== 1) {
            return Math.floor(roundToNearestPowerOf10(diff) / 2);
        } else {
            return 0.5;
        }
    }

    function ShowImages() {
        let showImg = []
        let showData = []
        let showScore = []
        let temp_score = [...suggest.layer_mean_activations].reverse()
        console.log(selectBin.indexList[selectBin.group], selectBin.indexList[selectBin.group]+selectBin.number)
        for(let i = selectBin.indexList[selectBin.group]; i < selectBin.indexList[selectBin.group]+selectBin.number; i++){
            console.log(i)
            showImg.push(image[i])
            showScore.push(temp_score[i])
            showData.push(prediction[i])
        }
        console.log(showImg)
        console.log(showScore)
        if(showSelect){
            console.log(selectFilterImage.group);
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
                            src={`/images/input/${data.input_image}`}/>
                            <p>{data.input_image}</p>
                            <h1>Model Prediction</h1>
                            <ul>
                                <li>Group: {data.group}</li>
                                <li>mean activation at layer: {data.score}</li>
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
                        {showImg.map((data,index) => (
                        <div>
                            <div 
                            className="input-image"
                            style={ selectFilterImage.some(e => e.input_image == data) ? { background: configData.COLOR.GREEN, color: "white"} : {}}
                            onClick={() => handleClick({input_image: data,
                                score: showScore[index],
                                prediction: showData[index],
                                group: selectBin.group+1})}
                            // style={{background: configData.COLOR.GREEN, color: "white"}}
                            >
                                <img
                                style={{width: "100%"}}
                                src={`/images/input/${data}`}/>
                                <p>{data}</p>
                            </div>
                            <div className="output-image" >
                                <img
                                style={{ width: '400px', height: '400px' }}
                                src={`/images/input/${data}`}/>
                                <p>{data}</p>
                                <h1>Model Prediction</h1>
                                <ul>
                                    <li>Group: {selectBin.group+1}</li>
                                    <li>mean activation at layer: {showScore[index]}</li>
                                    <li>{`${showData[index][0].className} (${showData[index][0].classId}) with probability: ${showData[index][0].score}`}</li>
                                    <li>{`${showData[index][1].className} (${showData[index][1].classId}) with probability: ${showData[index][1].score}`}</li>
                                    <li>{`${showData[index][2].className} (${showData[index][2].classId}) with probability: ${showData[index][2].score}`}</li>
                                    <li>{`${showData[index][3].className} (${showData[index][3].classId}) with probability: ${showData[index][3].score}`}</li>
                                    <li>{`${showData[index][4].className} (${showData[index][4].classId}) with probability: ${showData[index][4].score}`}</li>
                                </ul>
                            </div>
                        </div>
                        ))}
                </div>
            );
        }
    }

    function GenerateBinButtonImages() {
        const [binSelected, setBinSelected] = useState(null);
        return(
            <div className="model-button-group">
                <center><select
                    value={selectBin.group + 1}
                    onChange={(event) => {
                    const selectedGroup = parseInt(event.target.value) - 1;
                    setSelectBin((values) => ({
                        ...values,
                        group: selectedGroup,
                        number: chart[selectedGroup].numberImg,
                    }));
                    }}
                >
                <option value={1}>
                        Select group
                </option>
                {group.map((name) => (
                    <option key={name} value={name}>
                    Group {name}
                    </option>
                ))}
                </select></center>
            </div>
        );
    }

    function Debug() {
        console.log(chart)
        console.log(group)
        console.log(selectBin)
    }

    return(
        <div>
            <NavBar></NavBar>
            <div className="information">
                <h1>Suggestion</h1>
                {suggest !== null &&
                    <GenerateChart/>
                }
                <p>This graph shows distribution of mean activation values at layer '{data.layer_name}', among the input images dataset.</p>
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
                <Debug/>
                <GenerateBinButtonImages/>
            </div>
            <div className="image-colum">
                    { selectBin.group !== null &&
                        <div>
                            <ShowImages/>
                        </div>
                    }
            </div>
        </div>
    );
}
export default Suggest