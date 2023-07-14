import React, { useState, useEffect } from "react";
import axios from "axios";
import configData from "../config";

import NavBar from '../components/NavBar';

import '../assets/Home.css';

const Home = () => {

    const [select, setSelect] = useState({
        images: [],
        n_models: 1,
        custom_model_selected: "False",
        custom_class_index: null,
        custom_n_classes: null,
        predefined_models: [null],
        mobilenet_alpha: null,
        mobilenet_depth_multiplier: null,
        mobilenet_dropout: null,
        mobilenetv2_alpha: null,
        method: null,
        layer_name: null,
        neuron_selection_mode: null,
        n_max_activations: null,
        neuron_index: null,
        class_id: null,
        stddev: null,
        gradient_postprocess: null,
        augment_by_n: null,
        smoothgrad_postprocess: null,
        steps: null,
        integrated_gradients_postprocess: null,
        rule: null,
        epsilon: null,
        lrp_epsilon_bias: null,
        lrp_epsilon_IB_sequential_preset_AB_epsilon: null,
        alpha: null,
        beta: null,
        lrp_alpha_beta_bias: null,
        low: null,
        high: null,
        deep_taylor_low: null,
        deep_taylor_high: null,
        training_images: null,
    });

    const [shareConstant, setShareConstant] = useState({})

    let [models, setModels] = useState([
        "VGG16", "VGG19", "ResNet50", "InceptionV3", "InceptionResNetV2", "DenseNet121", "DenseNet169", "DenseNet201",
        "NASNetLarge", "NASNetMobile", "MobileNet", "MobileNetV2"
    ])

    let [layers, setLayers] = useState(null)

    let [imagenetClass, setImagenetClass] = useState({})

    let [methods, setMethods] = useState({
        "Gradient / Saliency":"gradient", "SmoothGrad (no lambda layers)":"smoothgrad", "DeconvNet":"deconvnet", 
        "Guided Backpropagation":"guided_backprop", "GradCAM":"gradcam", "Guided GradCAM":"guided_gradcam", 
        "Input * Gradient":"input_t_gradient", "Integrated Gradients (no lambda layers)":"integrated_gradients", 
        "LRP-z":"lrp.z", "LRP-ε":"lrp.epsilon", "LRP-wSquare":"lrp.w_square", "LRP-Flat":"lrp.flat", "LRP-αβ":"lrp.alpha_beta",
        "LRP-2α1β":"lrp.alpha_2_beta_1", "LRP-2α1β - IgnoreBias":"lrp.alpha_2_beta_1_IB", "LRP-1α0β":"lrp.alpha_1_beta_0", 
        "LRP-1α0β - IgnoreBias":"lrp.alpha_1_beta_0_IB", "LRP-z+":"lrp.z_plus", "LRP-z+ Fast":"lrp.z_plus_fast", 
        "LRP Sequential Preset A":"lrp.sequential_preset_a", "LRP Sequential Preset B":"lrp.sequential_preset_b",
        "LRP Sequential Preset A Flat":"lrp.sequential_preset_a_flat", "LRP Sequential Preset B Flat":"lrp.sequential_preset_b_flat",
        "DeepTaylor":"deep_taylor", "DeepTaylor-Bounded":"deep_taylor.bounded"
    })

    let [visualize, setVisualize] = useState([])
    let [visualizeShareConstant, setVisualizeShareConstant] = useState([])

    useEffect(() => {
        const fetchData = async (e) => {
        try {
            const temp_dict = {}
            const url = `${configData.API.PATH}download_classes`
            const jsonForm = {
                'custom_model_selected': "False",
                'custom_class_index': '',
                'custom_n_classes': ''
            };

            await axios.post(url, jsonForm, {
                headers: {
                    'Content-Type': 'application/json'
                }
            }).then((res) => {
                console.log(res.data.content)
                for (let index = 0; index < res.data.content.length; index++) {
                    const value = res.data.content[index];
                    temp_dict[index] = value;
                }
                setImagenetClass(values => temp_dict)
            })
        } catch (error) {
            console.error(error);
        }
        };
    
        fetchData();
    }, []);

    async function handleDownLoadLayesApi(models) {
        const url = `${configData.API.PATH}download_layers`
        const jsonForm = {
            'custom_model_selected': false,
            'model_name': models,
            'method': select.method
        };

        await axios.post(url, jsonForm, {
            headers: {
                'Content-Type': 'application/json'
            }}).then((res) => {
            setLayers(values => (res.data.content))
            console.log(res.data.content)
        })


    }

    function ModelsButtonGroup() {
        const [modelSelected, setModelSelected] = useState(null);
        return (
            <div className="model-button-group">
            {models.map(models => (
                <button 
                style={ models === select.predefined_models ? { background: configData.COLOR.RED, color: "white"} : { background: configData.COLOR.GRAY}}
                key={models}
                active={modelSelected === models}
                onClick={() => {setSelect(values => ({...values, predefined_models: models})); handleDownLoadLayesApi(models);}}
                >
                    {models}
                </button>
            ))}
            </div>
        );
    }

    function MethodButtonGroup() {
        const [methodsSelected, setMethodsSelected] = useState(null);
        return (
            <div className="method-button-group">
            {Object.entries(methods).map(([key, value]) => (
            <button
                style={value === select.method ? { background: configData.COLOR.RED, color: "white" } : {}}
                key={key}
                active={methodsSelected === value}
                onClick={() => {
                    setSelect(values => ({ ...values, method: value }));
                }}
            >
                {key}
            </button>
        ))}
            </div>
        );
    }

    function GenerateLayer() {
        return(
            <div>
                <label>Layers: </label><br/>
                <select value={select.layer_name} onChange={(e) => setSelect(values => ({...values, layer_name: e.target.value}))}>
                    <option value={null}>
                        Select layer
                    </option>
                    {layers.map(layers => (
                    <option
                    value={layers}
                    >
                        {layers}
                    </option>  
                    ))}
                </select>
            </div>
        );
    }

    function GenerateActivation() {
        return(
            <div>
                {select.neuron_selection_mode === "index" &&
                    <div className="input-number">
                        <label>Neuron index:</label><br/>
                        <input
                            className="neuron-index"
                            type="number"
                            value={select.neuron_index}
                            onChange={(e) => 
                                setSelect(values => 
                                    ({...values, neuron_index: e.target.value, n_max_activations:null}))}/>
                    </div>}
                {select.neuron_selection_mode === "max_activations" &&
                    <div className="generate">
                        <label>Number of max activations: </label><br/>
                        <select value={select.n_max_activations} onChange={(e) => setSelect(values => ({...values, n_max_activations: e.target.value, neuron_index:null}))}>
                            <option value={null}>
                                Select the number of index
                            </option>
                            <option value={"1"}>
                                1
                            </option>  
                            <option value={"4"}>
                                4
                            </option> 
                            <option value={"9"}>
                                9
                            </option>
                            <option value={"16"}>
                                16
                            </option>    
                        </select>
                    </div>
                }
            </div>
        );
    }

    function GenerateNeuronSelectionMode() {
        return(
        <div className="generate">
            <label>Neuron selection mode: </label><br/>
                <select value={select.neuron_selection_mode} onChange={(e) => setSelect(values => ({...values, neuron_selection_mode: e.target.value}))}>
                    <option
                    value={null}
                    >
                        Select the Neuron Mode
                    </option>
                    <option
                    value="max_activations"
                    >
                        Max Activations
                    </option>  
                    <option
                    value="index"
                    >
                        By Index
                    </option>  
                </select>
        </div>);
    }

    function GenerateClass() {
        let classKey = Object.keys(imagenetClass)
        return(
            <div className="generate">
            <label>Class: </label><br/>
                <select value={select.class_id} onChange={(e) => setSelect(values => ({...values, class_id: parseInt(e.target.value)}))}>
                    <option
                    value={null}
                    >
                        Select Class
                    </option>
                    {classKey.map(classKey => (
                        <option
                        value={classKey}
                        >
                            {imagenetClass[classKey]}
                        </option> 
                    ))}
                </select>
        </div>
        );
    }

    function GeneratePostProcess() {
        return(
            <div className="generate">
                <label>Select postprocess: </label><br/>
                <select value={shareConstant.postProcess} onChange={(e) => setShareConstant(values => ({...values, postProcess: e.target.value}))}>
                    <option
                    value={null}
                    >
                        Select postprocess
                    </option>
                    <option
                    value="abs"
                    >
                        Abs
                    </option>  
                    <option
                    value="None"
                    >
                        None
                    </option>
                    <option
                    value="square"
                    >
                        Square
                    </option>
                </select>
            </div>
        );
    }

    function GenerateEpsilon() {
        return(
            <div className="generate">
                <label>ε: </label><br/>
                <div className="input-number">
                    <input
                        className="neuron-index"
                        type="number"
                        value={select.epsilon}
                        onChange={(e) => 
                        setSelect(values => 
                        ({...values, epsilon: parseInt(e.target.value)}))}/>
                </div>
            </div>
        );
    }

    function GenerateBias() {
        return(
            <div className="generate">
                <label>Bias: </label><br/>
                <select value={shareConstant.bias} onChange={(e) => setShareConstant(values => ({...values, bias: e.target.value}))}>
                    <option
                    value={null}
                    >
                        Select Bias
                    </option>
                    <option
                    value="True"
                    >
                        True
                    </option>  
                    <option
                    value="False"
                    >
                        False
                    </option>
                </select>
            </div>
        );
    }

    function GroupGenerateAttributesNeuronMode() {
        return(
            <div className="generate">
                { layers != null &&
                    <div>
                        <GenerateLayer/>
                        <GenerateNeuronSelectionMode/>
                        <GenerateActivation/>
                    </div>
                }
            </div>
        );
    }

    function GroupGenerateAttributesClass() {
        return(
            <div className="generate">
            { layers != null &&
                <div>
                    <GenerateLayer/>
                    { imagenetClass != null &&
                    <div>
                            <GenerateClass/>
                    </div>
                    }
                </div>
            }
            </div>
        );
    }

    function GradientSaliencyAttributes() {
        return(
            <div className="generate">
                <GeneratePostProcess/>
            </div>
        );
    }

    function SmoothGradAttributes() {
        return(
            <div className="generate">
                <div className="input-number">
                    <label>Augment by n: </label><br/>
                        <input
                            className="neuron-index"
                            type="number"
                            value={select.augment_by_n}
                            onChange={(e) => 
                                setSelect(values => 
                                    ({...values, augment_by_n: parseInt(e.target.value)}))}/>
                </div>
                <GeneratePostProcess/>
            </div>
        );
    }

    function IntegratedGradientsAttributes() {
        return (
            <div className="generate">
                <div className="input-number">
                    <label>Steps: </label><br/>
                        <input
                            className="neuron-index"
                            type="number"
                            value={select.step}
                            onChange={(e) => 
                                setSelect(values => 
                                    ({...values, step: parseInt(e.target.value)}))}/>
                </div>
                <GeneratePostProcess/>
            </div>
        );
    }

    function LRPEpsilonAttributes() {
        return (
            <div className="generate">
                <GenerateBias/>
                <GenerateEpsilon/>
            </div>
        );
    }

    function LRPAlphaBetaAttributes() {
        return(
            <div className="generate">
                    <div className="input-number">
                        <label>α: </label><br/>
                            <input
                                className="neuron-index"
                                type="number"
                                value={select.alpha}
                                onChange={(e) => 
                                    setSelect(values => 
                                        ({...values, alpha: parseInt(e.target.value)}))}/>
                        <label>β: </label><br/>
                            <input
                                className="neuron-index"
                                type="number"
                                value={select.beta}
                                onChange={(e) => 
                                    setSelect(values => 
                                        ({...values, beta: parseInt(e.target.value)}))}/>
                    </div>
                    <GenerateBias/>
            </div>       
        );
    }

    function DeepTaylorBoundedAttributes() {
        return(
            <div className="generate">
                    <div className="input-number">
                        <label>Low bound: </label><br/>
                            <input
                                className="neuron-index"
                                type="number"
                                value={select.low}
                                onChange={(e) => 
                                    setSelect(values => 
                                        ({...values, low: parseInt(e.target.value)}))}/>
                        <label>High bound: </label><br/>
                            <input
                                className="neuron-index"
                                type="number"
                                value={select.high}
                                onChange={(e) => 
                                    setSelect(values => 
                                        ({...values, high: parseInt(e.target.value)}))}/>
                    </div>
            </div> 
        );
    }

    function MobileNetAttributes() {
        return(
            <div className="generate">
                <div className="input-number">
                <label>
                    MobileNet α: 
                </label><br/>
                <input
                    className="neuron-index"
                    type="number"
                    value={select.mobilenet_alpha}
                    onChange={(e) => 
                    setSelect(values => 
                    ({...values, mobilenet_alpha: parseInt(e.target.value)}))}/>
                <label>
                    MobileNet depth multiplier: 
                </label><br/>
                <input
                    className="neuron-index"
                    type="number"
                    value={select.mobilenet_depth_multiplier}
                    onChange={(e) => 
                    setSelect(values => 
                    ({...values, mobilenet_depth_multiplier: parseInt(e.target.value)}))}/>
                <label>
                    MobileNet dropout: 
                </label><br/>
                <input
                    className="neuron-index"
                    type="number"
                    value={select.mobilenet_dropout}
                    onChange={(e) => 
                    setSelect(values => 
                    ({...values, mobilenet_dropout: parseInt(e.target.value)}))}/>
                </div>
            </div>
        );
    }

    function MobileNetV2Attributes() {
        return(
            <div className="generate">
                <div className="input-number">
                    <label>
                        MobileNetV2 α: 
                    </label><br/>
                    <input
                        className="neuron-index"
                        type="number"
                        value={select.mobilenetv2_alpha}
                        onChange={(e) => 
                        setSelect(values => 
                        ({...values, mobilenetv2_alpha: parseInt(e.target.value)}))}/>
                </div>
            </div>);
    }

    function GenerateMethodAttributes() {
        const selectedMethod = select["method"]
        if(selectedMethod === "gradient" || selectedMethod === "smoothgrad" 
        || selectedMethod === "deconvnet" || selectedMethod === "guided_backprop"){
            return (
                <div className="attribute">
                    <table>
                        <tbody>
                            <tr>
                                {select.predefined_models == "MobileNet" &&
                                    <td><MobileNetAttributes/></td>
                                }
                                {select.predefined_models == "MobileNetV2" &&
                                    <td><MobileNetV2Attributes/></td>
                                }
                                <td>
                                    <GroupGenerateAttributesNeuronMode/>
                                </td>
                                <td>
                                {select.method === "gradient" &&
                                    <GradientSaliencyAttributes/>
                                }
                                {select.method === "smoothgrad" &&
                                    <SmoothGradAttributes/>
                                }
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            );
        } else {
            return(
                <div className="attribute">
                    <table>
                        <tbody>
                            <tr>
                                {select.predefined_models == "MobileNet" &&
                                    <td><MobileNetAttributes/></td>
                                }
                                {select.predefined_models == "MobileNetV2" &&
                                    <td><MobileNetV2Attributes/></td>
                                }
                                <td><GroupGenerateAttributesClass/></td>
                                <td>
                                    {select.method === "integrated_gradients" &&
                                        <IntegratedGradientsAttributes/>
                                    }
                                    {select.method === "lrp.epsilon" &&
                                        <LRPEpsilonAttributes/>
                                    }
                                    {select.method === "lrp.alpha_beta" &&
                                        <LRPAlphaBetaAttributes/>
                                    }
                                    {select.method === "lrp.sequential_preset_a" &&
                                        <GenerateEpsilon/>
                                    }
                                    {select.method === "lrp.sequential_preset_b" &&
                                        <GenerateEpsilon/>
                                    }
                                    {select.method === "deep_taylor.bounded" &&
                                        <DeepTaylorBoundedAttributes/>
                                    }
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            );

        }
    }

    function UploadImage() {
        const [image, setImage] = useState([])

        const handleImagesChange = (e) => {
            for (let i = 0; i < e.target.files.length; i++) {
                image.push(e.target.files[i])
            }
            console.log(image)
        }

        const handleUploadApi = async(e) => {
            e.preventDefault()
            const url = `${configData.API.PATH}upload_images`
            const formData = new FormData()
            image.forEach((file, index) => {
                formData.append(`images`, file);
            })
            console.log(formData)
            await axios.post(url, formData, {
                headers: {
                "Content-Type": "multipart/form-data",
            }})
            .then((res) => {
                setSelect(values => ({...values, images: res.data.content}))
                console.log(select.images)
            })
        }

        return(
            <div>
                <label
                    htmlFor="file"
                    className="inputImageLabel"
                    style={ select.images.length != 0 ? { background: configData.COLOR.GREEN, color: "white"} : {}}
                    >
                        Adding Images
                    <input 
                        style={{display: "none"}} 
                        type="file" 
                        name="file" 
                        id="file" 
                        onChange={handleImagesChange}
                        multiple>
                    </input>
                </label>
                <center>
                    { select.images.length === 0 &&
                    <button onClick={handleUploadApi}>
                        Upload
                    </button>}
                </center>
            </div>
        );
    }

    function handleVisualizePage() {
        // Register the constant value to localStorage
        localStorage.setItem('data', JSON.stringify(visualize));
        localStorage.setItem('share', JSON.stringify(visualizeShareConstant));
        
        // Redirect to a new page
        window.location.href = '/visualize'; 
    }

    function handleSuggestPage() {
        // Register the constant value to localStorage
        localStorage.setItem('data', JSON.stringify(select));
        localStorage.setItem('share', JSON.stringify(shareConstant));
        
        // Redirect to a new page
        window.location.href = '/suggest'; 
    } 

    function handleAddVisualize() {
        select.predefined_models = [select.predefined_models]
        let tempSelect = {...select} 
        let tempShareConstant
        if (shareConstant.postProcess === undefined){
            tempShareConstant = null
        }else{
            console.log(shareConstant.postProcess);
            tempShareConstant = shareConstant.postProcess
        }
        const updatedShareConstant = [...visualizeShareConstant, tempShareConstant]
        setVisualize(values => [...values, tempSelect])
        setVisualizeShareConstant(updatedShareConstant)
        shareConstant.postProcess = null
        setLayers(null)
        handleChange()
    }

    function handleChange() {
        const temp_model = select.predefined_models
        const temp_images = select.images
        const temp_n_models = select.n_models
        const temp_custom_model_selected = select.custom_model_selected
        Object.keys(select).forEach(key => select[key]=null);
        select.predefined_models = temp_model
        select.images = temp_images
        select.n_models = temp_n_models
        select.custom_model_selected = temp_custom_model_selected
    }

    function Debug() {
        console.log(select)
        console.log(visualize);
        console.log(visualizeShareConstant)
        // console.log(layers);
        // console.log(imagenetClass);
    }

    return (
        <div>
            <NavBar></NavBar>
            <div className="home">
                <table>
                    <tbody>
                        <tr>Input Images:</tr>
                        <tr>
                            {   Object.keys(imagenetClass).length !== 0 ?
                                <div>
                                    <UploadImage/>
                                </div>
                            :
                                <center><h2>Loading Please Wait</h2></center>
                            }  
                        </tr>
                        <tr>Predefined Models:</tr>
                        <tr><ModelsButtonGroup/></tr>
                        <tr>Method Of Visualization:</tr>
                        <tr><MethodButtonGroup/></tr>
                        {select.method !== null &&
                            <tr><GenerateMethodAttributes/></tr>
                        }
                        <tr>
                            <div className="button-group">
                                <center>
                                    <p>Number of visualize: {visualize.length}</p>
                                    <button onClick={handleAddVisualize}>Add Visualize</button>
                                </center>
                            </div>
                        </tr>
                        <tr>
                            <div className="button-group">
                                <center>
                                    <button onClick={handleSuggestPage}>Suggest</button>
                                    <button onClick={handleVisualizePage}>Visualize</button>
                                </center>
                            </div>
                        </tr>
                        <tr><Debug/></tr>
                    </tbody>
                </table>
            </div>
        </div>
    );   
}

export default Home