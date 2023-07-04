// file upload with ajax with progress bar and abort button
function ajaxAbortableFileUploadWithProgressListener(defaultLabel, inputId, progressBarId, abortButtonId, formData, url) {
    let input = $(inputId);
    let progressBar = $(progressBarId);
    let abortButton = $(abortButtonId);
    
    // prepare label for input
    let labels = input.siblings("label");

    let filename = "", input_parts;
    // if number of files loaded is greater than 1, show this information as label
    let input_length = input[0]["files"].length;
    if (input_length > 1) {
        filename = input_length + " files"
    }
    // else if number of files loaded is equal to 1, show the name of the file as label
    else if (input_length === 1) {
        input_parts = input.val().split("\\");
        filename = input_parts[input_parts.length - 1];
    }

    // let formData = new FormData();
    // formData.append(input[0].name, input[0].files[0]);

    return $.ajax({
        beforeSend: function () {
            input.prop("disabled", true);
            labels.css("font-weight", "normal");
            labels.text("Uploading " + filename);
        },
        xhr: function () {
            let xhr = new window.XMLHttpRequest();

            // add progress bar while uploading
            xhr.upload.addEventListener("progress", function (evt) {
                abortButton.show();
                if (evt.lengthComputable) {
                    let percentComplete = evt.loaded / evt.total;

                    progressBar.show();
                    progressBar.css({
                        width: percentComplete * 100 + '%'
                    });
                }
            }, false);

            // if abort button is clicked, abort upload
            abortButton.click(function () {
                xhr.abort();
            });

            return xhr;
        },
        type: 'POST',
        url: url,
        data: formData,
        contentType: false,
        cache: false,
        processData: false,
        success: function(response) {
            if (response["status"] === "error") {
                // when a controlled error occurs
                alert(response["message"]);

                input.val("");
                labels.css("font-weight", "normal");
                labels.text(defaultLabel);
            }
            else if (response["status"] === "aborted") {
                // when user click on cancel while selecting file to upload
                input.val("");
                labels.css("font-weight", "normal");
                labels.text(defaultLabel);
            }
            else {
                // when success
                input.closest("div").removeClass("danger");
                labels.css("font-weight", "bold");
                labels.text(filename);
            }
        },
        error: function(response) {
            // when uncontrolled error occurs
            input.val("");
            labels.css("font-weight", "normal");
            labels.text(defaultLabel);

            if (response.getAllResponseHeaders()) {
                alert(response["message"]);
            }
        },
        complete: function() {
            progressBar.hide();
            abortButton.hide();

            input.prop("disabled", false);
        },
    });
}


function fillUploadCustomModel(id, info_src) {
    return  "<div id='upload_custom_model_" + id + "' class='upload_custom_model secondary_block'>" +
            "    <div class='file_upload'>" +
            "        <form id='upload_custom_model_form_" + id + "' method='post' enctype='multipart/form-data'>" +
            "            <label for='custom_model_" + id + "'>Click to upload your model (HDF5 file)</label>" +
            "            <input class='custom_model' name='custom_model' id='custom_model_" + id + "' type='file'>" +
            "        </form>" +
            "        <div class='upload_status'>" +
            "            <div class='progress' id='upload_custom_model_progressbar_" + id + "'></div>" +
            "            <button type='reset' class='ajax_abort' id='upload_custom_model_abort_" + id + "'>Cancel upload</button>" +
            "        </div>" +
            "    </div>" +
            "    <div class='preprocessing'>" +
            "        <p><b>Insert</b> mean (and standard deviation, if needed) for custom pre-processing, <b>or leave blank</b> for standard ImageNet pre-processing</p>" +
            "        <div class='input_block'>" +
            "            <div class='tooltip'>" +
            "                <img class='icon' alt='info' src='" + info_src + "'/>" +
            "                <span class='tooltiptext'>Specify an overall single value in the first cell or three values, one per channel (RGB)</span>" +
            "            </div>" +
            "            <label>Mean: </label>" +
            "            <input type='number' step='.01' name='custom_mean1_" + id + "' id='custom_mean1_" + id + "'>" +
            "            <input type='number' step='.01' name='custom_mean2_" + id + "' id='custom_mean2_" + id + "'>" +
            "            <input type='number' step='.01' name='custom_mean3_" + id + "' id='custom_mean3_" + id + "'>" +
            "        </div>" +
            "        <div class='input_block'>" +
            "            <div class='tooltip'>" +
            "                <img class='icon' alt='info' src='" + info_src + "'/>" +
            "                <span class='tooltiptext'>Specify an overall single value in the first cell or three values, one per channel (RGB)</span>" +
            "            </div>" +
            "            <label>Standard deviation: </label>" +
            "            <input type='number' step='.01' name='custom_std_dev1_" + id + "' id='custom_std_dev1_" + id + "'>" +
            "            <input type='number' step='.01' name='custom_std_dev2_" + id + "' id='custom_std_dev2_" + id + "'>" +
            "            <input type='number' step='.01' name='custom_std_dev3_" + id + "' id='custom_std_dev3_" + id + "'>" +
            "        </div>" +
            "    </div>" +
            "    <button type='button' class='remove' id='remove_" + id + "'>Remove</button>" +
            "</div>";
}


let slideIndex = [];

function getInputs(reason, main_div, images, images_input, custom_model_selected, custom_models_div,
                   custom_n_classes_input, custom_class_index, custom_class_index_input, custom_model_divs_counter,
                   custom_models, predefined_models_div, predefined_model_button, custom_model_button,
                   mobilenet_alpha_input, mobilenet_depth_multiplier_input, mobilenet_dropout_input,
                   mobilenetv2_alpha_input, method_input, layer_selection_enabled, layer_name_input,
                   neuron_selection_mode_input, n_max_activations_input, n_max_activations_div, neuron_index_input,
                   neuron_index_div, class_id_input, stddev_input, gradient_postprocess_input, augment_by_n_input,
                   smoothgrad_postprocess_input, steps_input, integrated_gradients_postprocess_input, lrp_rule_input,
                   epsilon_input, lrp_epsilon_bias_input, lrp_epsilon_IB_sequential_preset_AB_epsilon_input,
                   alpha_input, beta_input, lrp_alpha_beta_bias_input, low_input, high_input, deep_taylor_low_input,
                   deep_taylor_high_input, training_images, method_selection_div) {

    let n_models, missing_custom_model, custom_model_inputs, i, neuron_selection_mode, n_max_activations, neuron_index,
        layer_name, class_id, data, valid_n_max_activations, final_custom_models, method_additional_parameters_loaded,
        error, error_message, custom_n_classes, file_input_object, predefined_models, method, classes_loaded,
        custom_mean1, custom_std_dev1, custom_mean1_input, custom_std_dev1_input, custom_mean2, custom_std_dev2,
        custom_mean2_input, custom_std_dev2_input, custom_mean3, custom_std_dev3, custom_mean3_input,
        custom_std_dev3_input, rgb_mean, custom_preprocessing_error, stddev, gradient_postprocess, augment_by_n,
        smoothgrad_postprocess, steps, integrated_gradients_postprocess, rule, epsilon, lrp_epsilon_bias,
        lrp_epsilon_IB_sequential_preset_AB_epsilon, alpha, beta, lrp_alpha_beta_bias, low, high, deep_taylor_low,
        deep_taylor_high, mobilenet_alpha, mobilenet_depth_multiplier, mobilenet_dropout, mobilenetv2_alpha;

    valid_n_max_activations = ["1", "4", "9", "16"];

    // remove all css for signaling errors and reset errors related variables
    main_div.find(".danger").removeClass("danger");
    error = false;
    error_message = "";


    // check input image/s
    if (images === "") {
        images_input.closest("div").addClass("danger");
        error_message += "No input image/s selected.\n";
        error = true;
    }

    if (reason === "suggestion") {
        if (images.length === 1) {
            images_input.closest("div").addClass("danger");
            error_message += "More than a single image is needed for receiving any suggestion.\n";
            error = true;
        }
    }


    // save models
    custom_n_classes = "";
    mobilenet_alpha = "";
    mobilenet_depth_multiplier = "";
    mobilenet_dropout = "";
    mobilenetv2_alpha = "";
    if (custom_model_selected) {
        missing_custom_model = false;
        custom_model_inputs = custom_models_div.find(".file_upload").find("input");

        for (let custom_model_input of custom_model_inputs) {
            file_input_object = $("#" + custom_model_input.id);

            if (file_input_object.val() === "") {
                file_input_object.closest(".file_upload").addClass("danger");
                missing_custom_model = true;
            }
        }

        if (missing_custom_model) {
            error_message += "Custom model/s not loaded.\n";
            error = true;
        }
        else {
            final_custom_models = [];
            n_models = custom_model_inputs.length;

            // if user choose to use custom models and then load custom class index file,
            // also custom_n_classes must be inserted and must be greater or equal than 2
            custom_n_classes = custom_n_classes_input.val();
            if (custom_class_index !== "" && (
                custom_n_classes === "" || parseInt(custom_n_classes) < 2)) {
                custom_n_classes_input.closest("div").addClass("danger");
                error_message += "Custom class index was loaded, number of classes (>=2) is required too.\n";
                error = true;
            }

            if (custom_n_classes !== "" && parseInt(custom_n_classes) >= 2 &&
                custom_class_index === "") {
                custom_class_index_input.closest("div").addClass("danger");
                error_message += "Number of classes inserted, class index file is required too.\n";
                error = true;
            }

            if (custom_n_classes !== "") custom_n_classes = parseInt(custom_n_classes);

            // save custom mean and standard deviation for custom pre-processing for all inserted custom models
            for (let i = 0, j = 0; j < custom_model_divs_counter; j++) {
                if (custom_models[j]) {
                    rgb_mean = null;
                    custom_preprocessing_error = false;

                    custom_mean1_input = $("#custom_mean1_" + j);
                    custom_mean2_input = $("#custom_mean2_" + j);
                    custom_mean3_input = $("#custom_mean3_" + j);

                    custom_mean1 = custom_mean1_input.val();
                    custom_mean2 = custom_mean2_input.val();
                    custom_mean3 = custom_mean3_input.val();

                    if ((custom_mean1 !== "" &&
                        (custom_mean2 !== "" && custom_mean3 === "" ||
                            custom_mean2 === "" && custom_mean3 !== "")) ||
                        (custom_mean1 === "" && (custom_mean2 !== "" || custom_mean3 !== ""))) {

                        custom_mean1_input.closest("div").addClass("danger");
                        error_message += "Either the first input value as overall mean or 3 per channel (RGB) mean input values are allowed.\n";
                        custom_preprocessing_error = true;
                    }
                    else if (custom_mean1 !== "" && parseFloat(custom_mean1) < 0 ||
                        custom_mean2 !== "" && parseFloat(custom_mean2) < 0 ||
                        custom_mean3 !== "" && parseFloat(custom_mean3) < 0) {

                        custom_mean1_input.closest("div").addClass("danger");
                        error_message += "Mean can't be negative.\n";
                        custom_preprocessing_error = true;
                    }

                    if (custom_mean1 !== "" && custom_mean2 === "" && custom_mean3 === "") {
                        rgb_mean = false;
                    }
                    else if (custom_mean1 !== "" && custom_mean2 !== "" && custom_mean3 !== "") {
                        rgb_mean = true;
                    }


                    custom_std_dev1_input = $("#custom_std_dev1_" + j);
                    custom_std_dev2_input = $("#custom_std_dev2_" + j);
                    custom_std_dev3_input = $("#custom_std_dev3_" + j);

                    custom_std_dev1 = custom_std_dev1_input.val();
                    custom_std_dev2 = custom_std_dev2_input.val();
                    custom_std_dev3 = custom_std_dev3_input.val();

                    if (rgb_mean !== null) {
                        if ((custom_std_dev1 !== "" &&
                            (custom_std_dev2 !== "" && custom_std_dev3 === "" ||
                                custom_std_dev2 === "" && custom_std_dev3 !== "")) ||
                            (custom_std_dev1 === "" && (custom_std_dev1 !== "" || custom_std_dev3 !== ""))) {

                            custom_std_dev1_input.closest("div").addClass("danger");
                            error_message += "Either the first input value as overall standard deviation or 3 per channel (RGB) standard deviation input values are allowed.\n";
                            custom_preprocessing_error = true;
                        } else if (rgb_mean === true &&
                            !(custom_std_dev1 === "" && custom_std_dev2 === "" && custom_std_dev3 === "") &&
                            (custom_std_dev1 === "" || custom_std_dev2 === "" || custom_std_dev3 === "") ||
                            rgb_mean === false && (custom_std_dev2 !== "" || custom_std_dev3 !== "")) {

                            custom_std_dev1_input.closest("div").addClass("danger");
                            error_message += "Mean and standard deviation must be defined in the same way, both either with a single value or with 3 per channel (RGB) values.\n";
                            custom_preprocessing_error = true;
                        } else if (custom_std_dev1 !== "" && parseFloat(custom_std_dev1) < 0 ||
                            custom_std_dev2 !== "" && parseFloat(custom_std_dev2) < 0 ||
                            custom_std_dev3 !== "" && parseFloat(custom_std_dev3) < 0) {

                            custom_std_dev1_input.closest("div").addClass("danger");
                            error_message += "Standard deviation can't be negative.\n";
                            custom_preprocessing_error = true;
                        }
                    }
                    else if (custom_preprocessing_error === false &&
                        (custom_std_dev1 !== "" || custom_std_dev2 !== "" || custom_std_dev3 !== "")) {
                        custom_mean1_input.closest("div").addClass("danger");
                        custom_std_dev1_input.closest("div").addClass("danger");
                        error_message += "For custom pre-processing, training dataset's mean is required too.\n";
                        custom_preprocessing_error = true;
                    }


                    if (custom_preprocessing_error) error = true;


                    if (rgb_mean === false) {
                        custom_models[j].means = [parseFloat(custom_mean1)];

                        if (custom_std_dev1 !== "") {
                            custom_models[j].std_devs = [parseFloat(custom_std_dev1)]
                        }
                    }
                    else if (rgb_mean === true) {
                        custom_models[j].means = [
                            parseFloat(custom_mean1),
                            parseFloat(custom_mean2),
                            parseFloat(custom_mean3)
                        ];

                        if (custom_std_dev1 !== "" && custom_std_dev2 !== "" && custom_std_dev3 !== "") {
                            custom_models[j].std_devs = [
                                parseFloat(custom_std_dev1),
                                parseFloat(custom_std_dev2),
                                parseFloat(custom_std_dev3)
                            ];
                        }
                    }

                    final_custom_models[i++] = custom_models[j];
                }
            }
        }
    }
    else {
        // save predefined model
        i = 0;
        predefined_models = [];

        predefined_models_div.find("input:checked").each(function () {
            predefined_models[i] = $(this)[0].id;
            i++;
        });

        n_models = i;

        if (predefined_models.length === 0) {
            if (custom_model_selected === null) {
                // it means no options was chosen
                predefined_model_button.addClass("danger");
                custom_model_button.addClass("danger");
            } else {
                predefined_models_div.closest("div").addClass("danger");
            }

            error_message += "Model not selected.\n";
            error = true;
        }

        // save additional parameters for MobileNet model
        mobilenet_alpha = mobilenet_alpha_input.val();
        if (mobilenet_alpha !== "") {
            mobilenet_alpha = parseFloat(mobilenet_alpha);
        }

        mobilenet_depth_multiplier = mobilenet_depth_multiplier_input.val();
        if (mobilenet_depth_multiplier !== "") {
            mobilenet_depth_multiplier = parseInt(mobilenet_depth_multiplier);
        }

        mobilenet_dropout = mobilenet_dropout_input.val();
        if (mobilenet_dropout !== "") {
            mobilenet_dropout = parseFloat(mobilenet_dropout);
        }

        // save additional parameters for MobileNetV2 model
        mobilenetv2_alpha = mobilenetv2_alpha_input.val();
        if (mobilenetv2_alpha !== "") {
            mobilenetv2_alpha = parseFloat(mobilenetv2_alpha);
        }
    }


    // save visualization method
    method = method_input.val();
    if (reason === "visualization" || reason === "suggestion" && n_models === 1) {
        if (method === "") {
            method_input.closest("div").addClass("danger");
            error_message += "Visualization method not selected.\n";
            error = true;
        }
    }

    // check layer/neuron/class selection
    neuron_selection_mode = "";
    n_max_activations = "";
    neuron_index = "";
    layer_name = "";
    class_id = "";

    if (method !== "input" && method !== "random" && method !== "") {
        // save selected layer
        if (layer_selection_enabled) {
            layer_name = layer_name_input.val();
        } else {
            layer_name = ""; // use default one (depending on the selected visualization method)
        }

        // save neuron selection
        if (method.startsWith("gradient") || method === "smoothgrad" ||
            method === "deconvnet" || method === "guided_backprop" || method === "pattern.net") {

            if (layer_selection_enabled) {
                // save neuron selection mode and neuron index if needed
                neuron_selection_mode = neuron_selection_mode_input.val();
                if (neuron_selection_mode === "max_activations") {
                    n_max_activations = n_max_activations_input.val();
                    if (!valid_n_max_activations.includes(n_max_activations)) {
                        n_max_activations_div.addClass("danger");
                        error_message += "Invalid selection for number of max activations to show.\n";
                        error = true;
                    }
                } else if (neuron_selection_mode === "index") {
                    neuron_index = neuron_index_input.val();
                    if (neuron_index === "") {
                        neuron_index_div.addClass("danger");
                        error_message += "Neuron selection mode is '" + neuron_selection_mode + "': you must provide neuron index.\n";
                        error = true;
                    } else {
                        neuron_index = parseInt(neuron_index);
                        if (neuron_index < 0) {
                            neuron_index_div.addClass("danger");
                            error_message += "Neuron index can't be negative.\n";
                            error = true;
                        }
                    }
                } else {
                    neuron_selection_mode_input.closest("div").addClass("danger");
                    error_message += "Invalid neuron selection mode.\n";
                    error = true;
                }
            } else {
                // if layer selection is not enabled, also neuron selection is fixed to max activations, only number of max activations to show must be read
                neuron_selection_mode = "max_activations";
                n_max_activations = n_max_activations_input.val();
                if (!valid_n_max_activations.includes(n_max_activations)) {
                    n_max_activations_div.addClass("danger");
                    error_message += "Invalid selection for number of max activations to show.\n";
                    error = true;
                }
            }
        }
        // save class selection
        else {
            classes_loaded = true;

            if (custom_model_selected) {
                if (custom_class_index === "") {
                    custom_class_index_input.closest("div").addClass("danger");
                    classes_loaded = false;
                }

                if (custom_n_classes_input.val() === "") {
                    custom_n_classes_input.closest("div").addClass("danger");
                    classes_loaded = false;
                }
            } else if (custom_model_selected === null) {
                // it means no options was chosen
                predefined_model_button.addClass("danger");
                custom_model_button.addClass("danger");
                classes_loaded = false;
            }

            if (!classes_loaded) {
                class_id_input.closest("div").addClass("danger");
                error_message += "Classes not loaded.\n";
                error = true;
            } else {
                class_id = class_id_input.val();
                if (parseInt(class_id) === -2 && reason === "visualization") {
                    class_id_input.closest("div").addClass("danger");
                    error_message += "Invalid class selection.\n";
                    error = true;
                }
            }
        }
    }

    // additional parameters check
    stddev = "";
    gradient_postprocess = "";
    augment_by_n = "";
    smoothgrad_postprocess = "";
    steps = "";
    integrated_gradients_postprocess = "";
    rule = "";
    epsilon = "";
    lrp_epsilon_bias = "";
    lrp_epsilon_IB_sequential_preset_AB_epsilon = "";
    alpha = "";
    beta = "";
    lrp_alpha_beta_bias = "";
    low = "";
    high = "";
    deep_taylor_low = "";
    deep_taylor_high = "";

    method_additional_parameters_loaded = false;

    if (reason === "visualization") {
        if (method === "random") {
            stddev = stddev_input.val();  // optional

            if (stddev !== "") {
                stddev = parseFloat(stddev);
            }

            method_additional_parameters_loaded = true;
        }
        else if (method === "gradient" || method === "gradient.baseline") {
            gradient_postprocess = gradient_postprocess_input.val();  // optional
            method_additional_parameters_loaded = true;
        }
        else if (method === "smoothgrad") {
            augment_by_n = augment_by_n_input.val();  // optional
            smoothgrad_postprocess = smoothgrad_postprocess_input.val();  // optional

            if (augment_by_n !== "") {
                augment_by_n = parseInt(augment_by_n);
            }

            method_additional_parameters_loaded = true;
        }
        else if (method === "integrated_gradients") {
            steps = steps_input.val();  // optional
            integrated_gradients_postprocess = integrated_gradients_postprocess_input.val(); // optional

            if (steps !== "") {
                steps = parseInt(steps);
            }

            method_additional_parameters_loaded = true;
        }
        else if (method === "lrp") {
            rule = lrp_rule_input.val();

            if (rule !== "") {
                if (rule === "Epsilon") {
                    epsilon = epsilon_input.val();  // optional
                    lrp_epsilon_bias = lrp_epsilon_bias_input.val();  // optional

                    if (epsilon !== "") {
                        epsilon = parseFloat(epsilon);
                    }

                    method_additional_parameters_loaded = true;
                } else if (rule === "EpsilonIgnoreBias") {
                    lrp_epsilon_IB_sequential_preset_AB_epsilon =
                        lrp_epsilon_IB_sequential_preset_AB_epsilon_input.val();  // optional

                    if (lrp_epsilon_IB_sequential_preset_AB_epsilon !== "") {
                        lrp_epsilon_IB_sequential_preset_AB_epsilon =
                            parseFloat(lrp_epsilon_IB_sequential_preset_AB_epsilon);
                    }

                    method_additional_parameters_loaded = true;
                } else if (rule === "AlphaBeta" || rule === "AlphaBetaIgnoreBias") {
                    alpha = alpha_input.val();
                    beta = beta_input.val();
                    lrp_alpha_beta_bias = lrp_alpha_beta_bias_input.val();  // optional

                    if (alpha !== "" && beta !== "") {
                        alpha = parseFloat(alpha);
                        beta = parseFloat(beta);

                        if (alpha - beta !== 1) {
                            alpha_input.closest("div").addClass("danger");
                            beta_input.closest("div").addClass("danger");
                            error_message += "&alpha; - &beta; = 1 is required.\n";
                            error = true;
                        } else {
                            method_additional_parameters_loaded = true;
                        }
                    }
                } else if (rule === "Bounded") {
                    low = low_input.val();  // optional
                    high = high_input.val();  // optional

                    if (low !== "" && high !== "") {
                        low = parseFloat(low);
                        high = parseFloat(high);

                        if (low > high) {
                            low_input.closest("div").addClass("danger");
                            high_input.closest("div").addClass("danger");
                            error_message += "Low bound can't be higher than high bound.\n";
                            error = true;
                        } else {
                            method_additional_parameters_loaded = true;
                        }
                    } else if (low !== "") {
                        low = parseFloat(low);

                        // check if valid wrt default high bound
                        if (low > 1) {
                            low_input.closest("div").addClass("danger");
                            high_input.closest("div").addClass("danger");
                            error_message += "Low bound can't be higher than high bound.\n";
                            error = true;
                        } else {
                            method_additional_parameters_loaded = true;
                        }
                    } else if (high !== "") {
                        high = parseFloat(high);

                        // check if valid wrt default low bound
                        if (high < -1) {
                            low_input.closest("div").addClass("danger");
                            high_input.closest("div").addClass("danger");
                            error_message += "Low bound can't be higher than high bound.\n";
                            error = true;
                        } else {
                            method_additional_parameters_loaded = true;
                        }
                    } else {
                        method_additional_parameters_loaded = true;
                    }
                } else {
                    method_additional_parameters_loaded = true;
                }
            }
        }
        else if (method === "lrp.epsilon") {
            epsilon = epsilon_input.val();  // optional
            lrp_epsilon_bias = lrp_epsilon_bias_input.val();  // optional

            if (epsilon !== "") {
                epsilon = parseFloat(epsilon);
            }

            method_additional_parameters_loaded = true;
        }
        else if (method === "lrp.epsilon_IB" || method === "lrp.sequential_preset_a" || method === "lrp.sequential_preset_b") {
            lrp_epsilon_IB_sequential_preset_AB_epsilon =
                lrp_epsilon_IB_sequential_preset_AB_epsilon_input.val();  // optional

            if (lrp_epsilon_IB_sequential_preset_AB_epsilon !== "") {
                lrp_epsilon_IB_sequential_preset_AB_epsilon =
                    parseFloat(lrp_epsilon_IB_sequential_preset_AB_epsilon);
            }

            method_additional_parameters_loaded = true;
        }
        else if (method === "lrp.alpha_beta") {
            alpha = alpha_input.val();
            beta = beta_input.val();
            lrp_alpha_beta_bias = lrp_alpha_beta_bias_input.val();  // optional

            if (alpha !== "" && beta !== "") {
                alpha = parseFloat(alpha);
                beta = parseFloat(beta);

                if (alpha - beta !== 1) {
                    alpha_input.closest("div").addClass("danger");
                    beta_input.closest("div").addClass("danger");
                    error_message += "&alpha; - &beta; = 1 is required.\n";
                    error = true;
                } else {
                    method_additional_parameters_loaded = true;
                }
            }
        }
        else if (method === "deep_taylor.bounded") {
            deep_taylor_low = deep_taylor_low_input.val();
            deep_taylor_high = deep_taylor_high_input.val();

            if (deep_taylor_low !== "" && deep_taylor_high !== "") {
                deep_taylor_low = parseFloat(deep_taylor_low);
                deep_taylor_high = parseFloat(deep_taylor_high);

                if (deep_taylor_low > deep_taylor_high) {
                    deep_taylor_low_input.closest("div").addClass("danger");
                    deep_taylor_high_input.closest("div").addClass("danger");
                    error_message += "Low bound can't be higher than high bound.\n";
                    error = true;
                } else {
                    method_additional_parameters_loaded = true;
                }
            }
        }
        else if (method === "pattern.net" || method === "pattern.attribution") {
            if (training_images !== "") {
                method_additional_parameters_loaded = true;
            }
        }
        else {
            // no additional parameters are required
            method_additional_parameters_loaded = true;
        }

        if (!method_additional_parameters_loaded) {
            method_selection_div.find(".additional_parameters").closest("div").addClass("danger");
            error_message += "Some required additional parameters are missing.\n";
            error = true;
        }
    }


    // if present, alert error messages and exit from visulization creation
    if (error) {
        alert(error_message);
        return false;
    }


    // if no input errors, removed any possible css signaling errors
    main_div.find(".danger").removeClass("danger");


    // data to send
    data = {
        "images": images,

        "n_models": n_models,

        "custom_model_selected": custom_model_selected,
        "custom_class_index": custom_class_index,
        "custom_n_classes": custom_n_classes,

        "final_custom_models": final_custom_models,

        "predefined_models": predefined_models,
        "mobilenet_alpha": mobilenet_alpha,
        "mobilenet_depth_multiplier": mobilenet_depth_multiplier,
        "mobilenet_dropout": mobilenet_dropout,
        "mobilenetv2_alpha": mobilenetv2_alpha,

        "method": method,
        "layer_name": layer_name,
        "neuron_selection_mode": neuron_selection_mode,
        "n_max_activations": n_max_activations,
        "neuron_index": neuron_index,
        "class_id": class_id,

        // methods additional parameters
        "stddev": stddev,
        "gradient_postprocess": gradient_postprocess,
        "augment_by_n": augment_by_n,
        "smoothgrad_postprocess": smoothgrad_postprocess,
        "steps": steps,
        "integrated_gradients_postprocess": integrated_gradients_postprocess,
        "rule": rule,
        "epsilon": epsilon,
        "lrp_epsilon_bias": lrp_epsilon_bias,
        "lrp_epsilon_IB_sequential_preset_AB_epsilon": lrp_epsilon_IB_sequential_preset_AB_epsilon,
        "alpha": alpha,
        "beta": beta,
        "lrp_alpha_beta_bias": lrp_alpha_beta_bias,
        "low": low,
        "high": high,
        "deep_taylor_low": deep_taylor_low,
        "deep_taylor_high": deep_taylor_high,
        "training_images": training_images,
    };

    return data;
}

function beforeSendAjax(suggest_button, visualize_button, main_spinner, visible_input_blocks, visible_buttons,
                        visible_refresh_icons, visible_file_upload_divs, predefined_models_div) {

    suggest_button.hide();
    visualize_button.hide();
    main_spinner.show();

    // disable all input visible in the interface for preventing errors caused by changing parameters while visualization generation is in progress
    visible_input_blocks.prop("disabled", true);
    predefined_models_div.find("input").prop("disabled", true);
    visible_file_upload_divs.find(":visible").prop("disabled", true);
    visible_file_upload_divs.find("form").css("cursor", "default");
    visible_buttons.prop("disabled", true);
    visible_refresh_icons.hide();
}

function completeAjax(suggest_button, visualize_button, main_spinner, visible_input_blocks, visible_buttons,
                      visible_refresh_icons, visible_file_upload_divs, predefined_models_div, layer_name_input,
                      layer_name_input_disabled) {

    main_spinner.hide();
    suggest_button.show();
    visualize_button.show();

    // enable again all input visible in the interface
    visible_input_blocks.prop("disabled", false);
    predefined_models_div.find("input").prop("disabled", false);
    visible_file_upload_divs.find("input").prop("disabled", false);
    visible_file_upload_divs.find("form").css("cursor", "pointer");
    visible_buttons.prop("disabled", false);
    visible_refresh_icons.show();
    layer_name_input.prop("disabled", layer_name_input_disabled);
}


function beautifyNames(key, val) {
    if (key === "method") {
        if (val === "gradient") {
            val = "Gradient / Saliency";
        } else if (val === "smoothgrad") {
            val = "SmoothGrad";
        } else if (val === "deconvnet") {
            val = "DeconvNet";
        } else if (val === "guided_backprop") {
            val = "Guided Backpropagation";
        } else if (val === "pattern.net") {
            val = "PatternNet";
        } else if (val === "gradcam") {
            val = "GradCAM";
        } else if (val === "guided_gradcam") {
            val = "Guided GradCAM";
        } else if (val === "input_t_gradient") {
            val = "Input * Gradient";
        } else if (val === "integrated_gradients") {
            val = "Integrated Gradients";
        } else if (val === "lrp") {
            val = "LRP";
        } else if (val === "lrp.z") {
            val = "LRP-z";
        } else if (val === "lrp.z_IB") {
            val = "LRP-z - IgnoreBias";
        } else if (val === "lrp.epsilon") {
            val = "LRP-&epsilon;";
        } else if (val === "lrp.epsilon_IB") {
            val = "LRP-&epsilon; - IgnoreBias";
        } else if (val === "lrp.w_square") {
            val = "LRP-wSquare";
        } else if (val === "lrp.flat") {
            val = "LRP-Flat";
        } else if (val === "lrp.alpha_beta") {
            val = "LRP-&alpha;&beta;";
        } else if (val === "lrp.alpha_2_beta_1") {
            val = "LRP-&alpha;2&beta;1";
        } else if (val === "lrp.alpha_2_beta_1_IB") {
            val = "LRP-&alpha;2&beta;1 - IgnoreBias";
        } else if (val === "lrp.alpha_1_beta_0") {
            val = "LRP-&alpha;1&beta;0";
        } else if (val === "lrp.alpha_1_beta_0_IB") {
            val = "LRP-&alpha;1&beta;0 - IgnoreBias";
        } else if (val === "lrp.z_plus") {
            val = "LRP-z+";
        } else if (val === "lrp.z_plus_fast") {
            val = "LRP-z+ Fast";
        } else if (val === "lrp.sequential_preset_a") {
            val = "LRP Sequential Preset A";
        } else if (val === "lrp.sequential_preset_b") {
            val = "LRP Sequential Preset B";
        } else if (val === "lrp.sequential_preset_a_flat") {
            val = "LRP Sequential Preset A Flat";
        } else if (val === "lrp.sequential_preset_b_flat") {
            val = "LRP Sequential Preset B Flat";
        } else if (val === "deep_taylor") {
            val = "DeepTaylor";
        } else if (val === "deep_taylor.bounded") {
            val = "DeepTaylor-Bounded";
        } else if (val === "pattern.attribution") {
            val = "PatternAttribution";
        }
    }
    else if (key === "rule") {
        if (val === "ZIgnoreBias") {
            val = "z - IgnoreBias";
        } else if (val === "Epsilon") {
            val = "&epsilon;";
        } else if (val === "EpsilonIgnoreBias") {
            val = "&epsilon; - IgnoreBias";
        } else if (val === "WSquare") {
            val = "wSquare";
        } else if (val === "AlphaBeta") {
            val = "&alpha;&beta;";
        } else if (val === "AlphaBetaIgnoreBias") {
            val = "&alpha;&beta; - IgnoreBias";
        } else if (val === "Alpha2Beta1") {
            val = "&alpha;2&beta;1";
        } else if (val === "Alpha2Beta1IgnoreBias") {
            val = "&alpha;2&beta;1 - IgnoreBias";
        } else if (val === "Alpha1Beta0") {
            val = "&alpha;1&beta;0";
        } else if (val === "Alpha1Beta0IgnoreBias") {
            val = "&alpha;1&beta;0 - IgnoreBias";
        } else if (val === "ZPlus") {
            val = "z+";
        } else if (val === "ZPlusFast") {
            val = "z+ Fast";
        }
    }

    return val;
}


function buildVisualizations(visualizations, n_visualizations, id_output_container, data, expand_src,
                             input_images_dir, input_images, output_images_dir) {

    let n_images, input_image_src, output_images, output_image_src, predictions, prediction, html, thumbnail,
        predictions_container, final_custom_models, custom_model_selected, i, k, l, general_info, class_id,
        selected_neuron, max_activations_selected, selected_class, top_n, default_top_n, n_classes, image_name_parts,
        image_name_parts_length, image_name, image_names, custom_model_name_parts, custom_model_name_parts_length,
        custom_model_name;

    // print first block
    html = "<div id='output_container_" + id_output_container + "' class='output_container main_view tabcontent'>" +
        "       <span class='close close_container'>&times;</span>" +
        "       <div class='general_info'>" +
        "           <div class='left'>" +
        "               <img class='icon expand_input_images' alt='Show/hide input images' src='" + expand_src + "'/>" +
        "               <div class='images_container'>";

    thumbnail = "           <div class='row'>";

    n_images = input_images.length;
    slideIndex[id_output_container] = 1;

    image_names = "";
    for (k = 0; k < n_images; k++) {
        input_image_src = input_images_dir + input_images[k];

        html += "           <div class='mySlides'>" +
            "                   <div class='numbertext'>" + (k + 1) + " / " + n_images + "</div>" +
            "                   <img class='image' alt='input image [" + id_output_container + "][" + k + "]' src='" + input_image_src + "'/>" +
            "               </div>";


        thumbnail += "          <div class='column'>" +
            "                       <img class='demo cursor' alt='input image [" + id_output_container + "][" + k + "]' " +
            "                           src='" + input_image_src + "' " +
            "                           onclick=\"currentSlide(" + id_output_container + ", " + (k + 1) + ")\">" +
            "                   </div>";

        image_name_parts = input_images[k].split("_");
        // last part contains nonce for making the file unique
        image_name_parts_length = image_name_parts.length - 1;
        image_name = image_name_parts[0];
        for (let j = 1; j < image_name_parts_length; j++) {
            image_name += "_" + image_name_parts[j];
        }

        // save image name
        image_names += "    <li class='image_name image_name_" + id_output_container + "_" + k + "'><b>image name</b>: " + image_name + "</li>";
    }
    thumbnail += "          </div>";

    <!-- Next and previous buttons -->
    html += "               <a class='prev' onclick=\"plusSlides(" + id_output_container + ", -1)\">&#10094;</a>" +
        "                   <a class='next' onclick=\"plusSlides(" + id_output_container + ", 1)\">&#10095;</a>";

    html += thumbnail;
    html += "           </div>" +
        "           </div>";

    html += "       <div class='right'>" +
        "               <p class='title'>Visualization</p>" +
        "               <ul>";

    general_info = [
        "n_models",

        "method",

        "stddev",
        "gradient_postprocess",
        "augment_by_n",
        "smoothgrad_postprocess",
        "steps",
        "integrated_gradients_postprocess",
        "rule",
        "epsilon",
        "lrp_epsilon_bias",
        "lrp_epsilon_IB_sequential_preset_AB_epsilon",
        "alpha",
        "beta",
        "lrp_alpha_beta_bias",
        "low",
        "high",
        "deep_taylor_low",
        "deep_taylor_high",
    ];

    html += image_names;

    custom_model_selected = data["custom_model_selected"];
    if (custom_model_selected) {
        n_classes = data["custom_n_classes"];
        if (n_classes !== "") {
            html += "       <li><b>n classes</b>: " + n_classes.toString() + "</li>";
        }
    }
    else {
        n_classes = 1000;
        html += "           <li><b>n classes</b>: " + n_classes.toString() + "</li>";
    }

    for (let key in data) {
        if (general_info.includes(key)) {
            if (data[key] !== null && data[key] !== '') {
                let val = beautifyNames(key, data[key].toString());

                if (key === "lrp_epsilon_bias" || key === "lrp_alpha_beta_bias") {
                    html += "   <li><b>bias</b>: " + val + "</li>";
                } else if (key === "alpha") {
                    html += "   <li><b>&alpha;</b>: " + val + "</li>";
                } else if (key === "beta") {
                    html += "   <li><b>&beta;</b>: " + val + "</li>";
                } else if (key === "epsilon" || key === "lrp_epsilon_IB_sequential_preset_AB_epsilon") {
                    html += "   <li><b>&epsilon;</b>: " + val + "</li>";
                } else if (key === "deep_taylor_low") {
                    html += "   <li><b>low</b>: " + val + "</li>";
                } else if (key === "deep_taylor_high") {
                    html += "   <li><b>high</b>: " + val + "</li>";
                } else {
                    html += "   <li><b>" + key.replace(/_/g, " ") + "</b>: " + val + "</li>";
                }
            }
        }
    }
    html += "           </ul>" +
        "           </div>" +
        "       </div>";

    // print visualizations
    html += "   <div class='visualizations'>";

    final_custom_models = data["final_custom_models"];
    for (i = 0; i < n_visualizations; i++) {
        output_images = visualizations[i]["output_images"];

        html += "   <div class='visualization'>" +
            "          <div class='left'>" +
            "               <div class='images_container'>";

        thumbnail = "           <div class='row'>";
        predictions_container = "       <div class='predictions_container'>";

        selected_neuron = "";
        max_activations_selected = false;
        if (data["neuron_selection_mode"] !== "") {
            selected_neuron += "    <li><b>neuron selection mode</b>: " + data["neuron_selection_mode"].toString() + "</li>";

            if (data["neuron_selection_mode"] === "index") {
                selected_neuron += "<li><b>neuron index</b>: " + data["neuron_index"].toString() + "</li>";
            }
            else if (data["neuron_selection_mode"] === "max_activations") {
                selected_neuron += "<li><b>n max activations</b>: " + data["n_max_activations"].toString() + "</li>";
                max_activations_selected = true;
            }
        }

        selected_class = "";

        predictions = visualizations[i]["predictions"];
        for (k = 0; k < n_images; k++) {
            output_image_src = output_images_dir + output_images[k];

            html += "           <div class='mySlides'>" +
                "                   <div class='numbertext'>" + (k+1) + " / " + n_images + "</div>" +
                "                   <img class='image' " +
                "                       alt='visualization [" + id_output_container + "][" + i + "][" + k + "]' " +
                "                       src='" + output_image_src + "'/>" +
                "               </div>";


            thumbnail += "          <div class='column'>" +
                "                       <img class='demo cursor'  " +
                "                           alt='visualization [" + id_output_container + "][" + i + "][" + k + "]' " +
                "                           src='" + output_image_src + "' " +
                "                           onclick=\"currentSlide(" + id_output_container + ", " + (k+1) + ")\">" +
                "                   </div>";


            if (predictions.length > 0) {
                prediction = predictions[k];
                default_top_n = n_classes < 5 ? n_classes : 5;
                if (prediction.length > 0) {
                    top_n = default_top_n;

                    predictions_container += "  <div class='predictions predictions_" + id_output_container + "_" + k + "'>" +
                        "                           <p>Model prediction:</p>" +
                        "                           <ul>";
                    for (l = 0; l < top_n; l++) {
                        predictions_container += "      <li><b>" + prediction[l]["className"] + "</b> (" +
                            prediction[l]["classId"] + ") with probability " +
                            (parseFloat(prediction[l]["score"]) * 100).toFixed(2) + "%</li>";
                    }
                    predictions_container += "      </ul>" +
                        "                       </div>";


                    if (data["class_id"] !== "") {
                        class_id = parseInt(data["class_id"]);
                        if (class_id === -1) {
                            selected_class += " <li class='selected_class selected_class_" + id_output_container + "_" + k + "'>" +
                                "                   <b>selected class</b>: " + prediction[0]["className"] + " (id: " + prediction[0]["classId"].toString() + ")" +
                                "               </li>";
                        } else {
                            for (l = 0; l < n_classes; l++) {
                                if (parseInt(prediction[l]["classId"]) === class_id) {
                                    selected_class += " <li class='selected_class selected_class_" + id_output_container + "_" + k + "'>" +
                                        "                   <b>selected class</b>: " + prediction[l]["className"] + " (id: " + class_id.toString() + ")" +
                                        "               </li>";
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if (max_activations_selected) {
                selected_neuron += "<li class='max_activations_neurons max_activations_neurons_" + id_output_container + "_" + k + "'>" +
                    "                   <b>max activations neurons: </b>: [";

                for (let l = 0; l < data["n_max_activations"]; l++) {
                    if (l < visualizations[i]["max_activations_neuron_indexes_per_image"][k].length) {
                        selected_neuron += " " + visualizations[i]["max_activations_neuron_indexes_per_image"][k][l].toString() + ",";
                    }
                    else {
                        selected_neuron += " None,";
                    }
                }
                selected_neuron += "    ] " +
                    "               </li>";
            }
        }
        thumbnail += "          </div>";
        predictions_container += "      </div>";

        <!-- Next and previous buttons -->
        html += "               <a class='prev' onclick=\"plusSlides(" + id_output_container + ", -1)\">&#10094;</a>" +
            "                   <a class='next' onclick=\"plusSlides(" + id_output_container + ", 1)\">&#10095;</a>";

        html += thumbnail;

        html += "           </div>" +
            "           </div>";

        html += "       <div class='right'>" +
            "               <div class='model_info'>" +
            "                   <ul>";

        // print informaion about model related to current visualization
        if (custom_model_selected) {
            custom_model_name_parts = final_custom_models[i].name.split("_");
            // last part contains nonce for making the file unique
            custom_model_name_parts_length = custom_model_name_parts.length - 1;
            custom_model_name = custom_model_name_parts[0];
            for (let j = 1; j < custom_model_name_parts_length; j++) {
                custom_model_name += "_" + custom_model_name_parts[j];
            }

            // print custom model name
            html += "               <li><b>model</b>: " + custom_model_name + "</li>";

            // print information about pre-processing
            if (final_custom_models[i].means === "" && final_custom_models[i].std_devs === "") {
                html += "           <li><b>pre-processing</b>: ImageNet </li>";
            } else if (final_custom_models[i].means !== "" && final_custom_models[i].std_devs === "") {
                html += "           <li><b>pre-processing</b>: subtract mean </li>";
                html += "           <li><b>custom mean</b>: " + final_custom_models[i].means.toString() + "</li>";
            } else if (final_custom_models[i].means !== "" && final_custom_models[i].std_devs !== "") {
                html += "           <li><b>pre-processing</b>: subtract mean then divide by standard deviation </li>";
                html += "           <li><b>custom mean</b>: " + final_custom_models[i].means.toString() + "</li>";
                html += "           <li><b>custom stddev</b>: " + final_custom_models[i].std_devs.toString() + "</li>";
            }
        }
        else {
            // print predefined model name
            html += "               <li><b>model</b>: " + data["predefined_models"][i] + "</li>";

            if (data["predefined_models"][i] === "MobileNet") {
                // if specified, print additional parameters for MobileNet
                if (data["mobilenet_alpha"] !== "") {
                    html += "       <li><b> MobileNet &alpha; </b>: " + data["mobilenet_alpha"].toString() + "</li>";
                }

                if (data["mobilenet_depth_multiplier"] !== "") {
                    html += "       <li><b> MobileNet depth multiplier </b>: " + data["mobilenet_depth_multiplier"].toString() + "</li>";
                }

                if (data["mobilenet_dropout"] !== "") {
                    html += "       <li><b> MobileNet dropout </b>: " + data["mobilenet_dropout"].toString() + "</li>";
                }
            }
            else if (data["predefined_models"][i] === "MobileNetV2") {
                // if specified, print additional parameters for MobileNetV2
                if (data["mobilenetv2_alpha"] !== "") {
                    html += "       <li><b> MobileNetV2 &alpha; </b>: " + data["mobilenetv2_alpha"].toString() + "</li>";
                }
            }
        }

        if (data["layer_name"] !== "") {
            html += "               <li><b>layer</b>: " + data["layer_name"].toString() + "</li>";
        }

        html += selected_neuron;

        html += selected_class;

        html += "               </ul>" +
                                predictions_container +
            "               </div>" +
            "           </div>" +
            "       </div>";
    }

    html += "   </div>" +
        "   </div>";

    return html;
}


function prepareSvg(id_output_container) {
    let html;

    html = "<div id='chart_container_" + id_output_container + "' class='output_container main_view tabcontent'>" +
        "       <span class='close close_container'>&times;</span>" +
        "       <div class='suggestion chart'>" +
        "           <div class='left'>" +
        "               <svg id='chart_" + id_output_container + "'></svg>" +
        "           </div>" +
        "           <div class='right'>" +
        "               <p class='title'>Suggestion</p>" +
        "               <p class='description'></p>" +
        "           </div>" +
        "       </div>" +
        "   </div>";

    return html;
}

function buildSvgHistogram(source, input_images_belong_to_same_class, input_images_class_id, id_output_container,
                           input_images_dir, data, suggestion, tip, suggestion_modal, suggestion_modal_content,
                           x_text) {

    let chart_container_div = $("#chart_container_" + id_output_container), n_models, basic_info_keys,
        custom_model_selected, n_classes, html, final_custom_model, custom_model_name_parts,
        custom_model_name_parts_length, custom_model_name, predefined_model;

    n_models = data["n_models"];

    // build basic info html
    basic_info_keys = [
        "n_models",
        "method",
    ];

    html = "<ul>";
    custom_model_selected = data["custom_model_selected"];
    if (custom_model_selected) {
        n_classes = data["custom_n_classes"];
        if (n_classes !== "") {
            html += "<li><b>n classes</b>: " + n_classes.toString() + "</li>";
        }
    } else {
        n_classes = 1000;
        html += "<li><b>n classes</b>: " + n_classes.toString() + "</li>";
    }

    for (let key in data) {
        if (basic_info_keys.includes(key)) {
            if (data[key] !== null && data[key] !== '') {
                let val = beautifyNames(key, data[key].toString());

                if (key === "lrp_epsilon_bias" || key === "lrp_alpha_beta_bias") {
                    html += "   <li><b>bias</b>: " + val + "</li>";
                } else if (key === "alpha") {
                    html += "   <li><b>&alpha;</b>: " + val + "</li>";
                } else if (key === "beta") {
                    html += "   <li><b>&beta;</b>: " + val + "</li>";
                } else if (key === "epsilon" || key === "lrp_epsilon_IB_sequential_preset_AB_epsilon") {
                    html += "   <li><b>&epsilon;</b>: " + val + "</li>";
                } else if (key === "deep_taylor_low") {
                    html += "   <li><b>low</b>: " + val + "</li>";
                } else if (key === "deep_taylor_high") {
                    html += "   <li><b>high</b>: " + val + "</li>";
                } else {
                    html += "   <li><b>" + key.replace(/_/g, " ") + "</b>: " + val + "</li>";
                }
            }
        }
    }

    if (n_models > 1) {
        // M models suggestion point 1 and points 4 and 5
        if (suggestion["input_images_class_name"] !== "") {
            html += "<li><b>input images' class</b>: " + suggestion["input_images_class_name"] + " (id: " + input_images_class_id + ")</li>";
        }
    }
    else {
        // 1 model, suggestion points 2 and 3
        if (custom_model_selected) {
            final_custom_model = data["final_custom_models"][0];

            custom_model_name_parts = final_custom_model.name.split("_");
            // last part contains nonce for making the file unique
            custom_model_name_parts_length = custom_model_name_parts.length - 1;
            custom_model_name = custom_model_name_parts[0];
            for (let j = 1; j < custom_model_name_parts_length; j++) {
                custom_model_name += "_" + custom_model_name_parts[j];
            }

            // print custom model name
            html += "       <li><b>model</b>: " + custom_model_name + "</li>";

            // print information about pre-processing
            if (final_custom_model.means === "" && final_custom_model.std_devs === "") {
                html += "   <li><b>pre-processing</b>: ImageNet </li>";
            }
            else if (final_custom_model.means !== "" && final_custom_model.std_devs === "") {
                html += "   <li><b>pre-processing</b>: subtract mean </li>";
                html += "   <li><b>custom mean</b>: " + final_custom_model.means.toString() + "</li>";
            }
            else if (final_custom_model.means !== "" && final_custom_model.std_devs !== "") {
                html += "   <li><b>pre-processing</b>: subtract mean then divide by standard deviation </li>";
                html += "   <li><b>custom mean</b>: " + final_custom_model.means.toString() + "</li>";
                html += "   <li><b>custom stddev</b>: " + final_custom_model.std_devs.toString() + "</li>";
            }
        }
        else {
            predefined_model = data["predefined_models"][0];

            // print predefined model name
            html += "       <li><b>model</b>: " + predefined_model + "</li>";

            if (predefined_model === "MobileNet") {
                // if specified, print additional parameters for MobileNet
                if (data["mobilenet_alpha"] !== "") {
                    html += "<li><b> MobileNet &alpha; </b>: " + data["mobilenet_alpha"].toString() + "</li>";
                }

                if (data["mobilenet_depth_multiplier"] !== "") {
                    html += "<li><b> MobileNet depth multiplier </b>: " + data["mobilenet_depth_multiplier"].toString() + "</li>";
                }

                if (data["mobilenet_dropout"] !== "") {
                    html += "<li><b> MobileNet dropout </b>: " + data["mobilenet_dropout"].toString() + "</li>";
                }
            }
            else if (predefined_model === "MobileNetV2") {
                // if specified, print additional parameters for MobileNetV2
                if (data["mobilenetv2_alpha"] !== "") {
                    html += "<li><b> MobileNetV2 &alpha; </b>: " + data["mobilenetv2_alpha"].toString() + "</li>";
                }
            }
        }
    }
    html += "</ul>";

    chart_container_div.find(".right").append(html);


    let dataset = [], j = 0, bar_padding = 5, x, y, histogram, bins, max_freq, y_ticks;

    // set the dimensions and margins of the graph
    let margin = {top: 20, right: 40, bottom: 50, left: 40},
        width = 550 - margin.left - margin.right,
        height = 400 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    let svg = d3.select("#chart_" + id_output_container)
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

    for (let val of source) {
        dataset[j++] = parseFloat(val);
    }

    // x axis
    x = d3.scaleLinear()
        .domain(d3.extent(dataset))  // domain is the extend of the dataset (from min to max value)
        .nice()  // this round the domain's interval in a more balanced way
        .range([20, width]);  // x axis will start 20 pixels after y axes, and end at the max width of the svg figure

    // set the parameters for the histogram (just the domain, leave the rest as default)
    histogram = d3.histogram()
        .domain(x.domain());

    // apply this histogram function to dataset to get the bins
    bins = histogram(dataset);

    // append the x axis to the svg figure
    if (d3.min(x.domain()) === d3.max(x.domain())) {
        svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .attr("class", "axis")
            .style("fill", "black")
            .call(d3.axisBottom(x).tickValues([parseInt(d3.min(x.domain()))]))
            .append("text")
            .attr("transform", "translate(" + width + ",0)")
            .attr("y", 20)
            .attr("dy", "1rem")
            .style("text-anchor", "end")
            .style("font-size", "1rem")
            .text(x_text);
    }
    else {
        svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .attr("class", "axis")
            .style("fill", "black")
            .call(d3.axisBottom(x).tickValues([d3.min(x.domain()), d3.max(x.domain())]))
            .append("text")
            .attr("transform", "translate(" + width + ",0)")
            .attr("y", 20)
            .attr("dy", "1rem")
            .style("text-anchor", "end")
            .style("font-size", "1rem")
            .text(x_text);
    }


    // y axis: domain is from 0 to max frequency of images in bin (max length of a bin)
    y = d3.scaleLinear()
        .range([height, 0])
        .domain([0, d3.max(bins, function (d) {
            return d.length;
        })]);

    // compute proper number of tick for the y axis in order to show only integer values (y tick corresponds to number of images in a bin)
    max_freq = d3.max(bins, function (d) {
        return d.length;
    });

    y_ticks = max_freq < 5 ? max_freq : 5;

    // append y axis to svg figure
    svg.append("g")
        .attr("class", "axis")
        .style("fill", "black")
        .call(d3.axisLeft(y).ticks(y_ticks).tickFormat(d3.format("d")))
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dy", ".71rem")
        .style("text-anchor", "end")
        .style("font-size", "1rem")
        .text("n images");

    // append the bar rectangles to the svg element
    svg.selectAll("rect")
        .data(bins)
        .enter()
        .append("rect")
        .attr("x", 3)
        .attr("transform", function (d) {
            return "translate(" + x(d.x0) + "," + y(d.length) + ")";
        })
        .attr("width", function (d) {
            if (x(d.x1) - x(d.x0) - bar_padding <= 0) {
                return 20;
            } else {
                return x(d.x1) - x(d.x0) - bar_padding;
            }
        })
        .attr("height", function (d) {
            return height - y(d.length);
        })
        .on("mouseover", function (d) {
            tip.transition()
                .duration(200)
                .style("opacity", 1);

            if (d.x0 < 1 && d.x1 < 1) {
                tip.html("<p>" + d.length + " images in bin [" + d.x0.toExponential() + ", " + d.x1.toExponential() + "]</p>" + "<p>Click to show images.</p>")
                    .style("left", (d3.event.pageX) + "px")
                    .style("top", (d3.event.pageY - 28) + "px");
            } else {
                tip.html("<p>" + d.length + " images in bin [" + d.x0.toFixed(2) + ", " + d.x1.toFixed(2) + "]</p>" + "<p>Click to show images.</p>")
                    .style("left", (d3.event.pageX) + "px")
                    .style("top", (d3.event.pageY - 28) + "px");
            }
        })
        .on("mouseout", function (d) {
            tip.transition()
                .duration(500)
                .style("opacity", 0);
        })
        .on("click", function (d) {
            let valid_indexes_for_bin = [], k = 0, index;

            for (let val of d) {
                index = source.indexOf(val);
                if (k > 0) {
                    while (valid_indexes_for_bin.includes(index)) {
                        index = source.indexOf(val, index + 1);
                    }
                }
                valid_indexes_for_bin[k++] = index;
            }

            suggestion_modal_content.html("<span id='close_suggestion_modal' class='close'>&times;</span>");
            suggestion_modal_content.append(
                buildSuggestionModalForHistogram(input_images_belong_to_same_class, input_images_class_id,
                    id_output_container, input_images_dir, data, suggestion, valid_indexes_for_bin, d)
            );

            // show the first slide for each suggestions in this container
            showSlides(id_output_container, slideIndex[id_output_container]);

            // open the modal
            suggestion_modal.style.display = "block";
        });
}

function buildSvgScatterplot(source_x, source_y, x_text, y_text, input_images_belong_to_same_class,
                             input_images_class_id, id_output_container, input_images_dir, data, suggestion, tip,
                             suggestion_modal, suggestion_modal_content) {

    let chart_container_div = $("#chart_container_" + id_output_container), basic_info_keys, custom_model_selected,
        n_classes, html;

    // build basic info html
    basic_info_keys = [
        "n_models",
        "method",
    ];

    html = "<ul>";
    custom_model_selected = data["custom_model_selected"];
    if (custom_model_selected) {
        n_classes = data["custom_n_classes"];
        if (n_classes !== "") {
            html += "<li><b>n classes</b>: " + n_classes.toString() + "</li>";
        }
    }
    else {
        n_classes = 1000;
        html += "<li><b>n classes</b>: " + n_classes.toString() + "</li>";
    }

    for (let key in data) {
        if (basic_info_keys.includes(key)) {
            if (data[key] !== null && data[key] !== '') {
                let val = beautifyNames(key, data[key].toString());

                if (key === "lrp_epsilon_bias" || key === "lrp_alpha_beta_bias") {
                    html += "   <li><b>bias</b>: " + val + "</li>";
                } else if (key === "alpha") {
                    html += "   <li><b>&alpha;</b>: " + val + "</li>";
                } else if (key === "beta") {
                    html += "   <li><b>&beta;</b>: " + val + "</li>";
                } else if (key === "epsilon" || key === "lrp_epsilon_IB_sequential_preset_AB_epsilon") {
                    html += "   <li><b>&epsilon;</b>: " + val + "</li>";
                } else if (key === "deep_taylor_low") {
                    html += "   <li><b>low</b>: " + val + "</li>";
                } else if (key === "deep_taylor_high") {
                    html += "   <li><b>high</b>: " + val + "</li>";
                } else {
                    html += "   <li><b>" + key.replace(/_/g, " ") + "</b>: " + val + "</li>";
                }
            }
        }
    }
    html += "</ul>";

    chart_container_div.find(".right").append(html);


    let dataset = [], j, x, y, max_different_classes, y_ticks;

    // set the dimensions and margins of the graph
    let margin = {top: 20, right: 30, bottom: 50, left: 40},
        width = 550 - margin.left - margin.right,
        height = 400 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    let svg = d3.select("#chart_" + id_output_container)
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

    j = 0;
    for (let val of source_x) {
        dataset[j] = {
            "x": parseFloat(val),
            "y": parseFloat(source_y[j])
        };

        j++;
    }

    // x axis
    x = d3.scaleLinear()
        .domain([0, 100])
        .range([0, width]);

    // append the x axis to the svg figure
    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .attr("class", "axis")
        .style("fill", "black")
        .call(d3.axisBottom(x))
        .append("text")
        .attr("transform", "translate(" + width + ",0)")
        .attr("y", 20)
        .attr("dy", "1rem")
        .style("text-anchor", "end")
        .style("font-size", "1rem")
        .text(x_text);

    // compute proper number of tick for the y axis in order to show only integer values (y tick corresponds to number of images in a bin)
    max_different_classes = d3.max(source_y);
    y_ticks = max_different_classes < 5 ? max_different_classes : 5;

    // y axis: domain is from 0 to max number of different predicted classes
    y = d3.scaleLinear()
        .range([height, 0])
        .domain([0, max_different_classes]);

    // append y axis to svg figure
    svg.append("g")
        .attr("class", "axis")
        .style("fill", "black")
        .call(d3.axisLeft(y).ticks(y_ticks).tickFormat(d3.format("d")))
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dy", ".71rem")
        .style("text-anchor", "end")
        .style("font-size", "1rem")
        .text(y_text);

    svg.append("g")
        .selectAll("dot")
        .data(dataset)
        .enter()
        .append("circle")
        .attr("cx", function (d) {
            return x(100*d.x.toFixed(4));
        })
        .attr("cy", function (d) {
            return y(d.y);
        })
        .attr("r", 8)
        .on("mouseover", function (d) {
            tip.transition()
                .duration(200)
                .style("opacity", 1);

            let img_index, img_src, image_name_parts, image_name_parts_length, image_name;
            img_index = source_x.indexOf(d.x);
            img_src = input_images_dir + suggestion["input_images"][img_index];

            image_name_parts = suggestion["input_images"][img_index].split("_");
            // last part contains nonce for making the file unique
            image_name_parts_length = image_name_parts.length - 1;
            image_name = image_name_parts[0];
            for (let j = 1; j < image_name_parts_length; j++) {
                image_name += "_" + image_name_parts[j];
            }

            tip.html("  <img class='image image_tip' alt='input image [" + id_output_container + "]' src='" + img_src + "'/>" +
                "       <p>" + image_name + "</p>")
                .style("left", (d3.event.pageX) + "px")
                .style("top", (d3.event.pageY - 28) + "px");
        })
        .on("mouseout", function (d) {
            tip.transition()
                .duration(500)
                .style("opacity", 0);
        })
        .on("click", function (d) {
            let img_index;
            img_index = source_x.indexOf(d.x);

            suggestion_modal_content.html("<span id='close_suggestion_modal' class='close'>&times;</span>");
            suggestion_modal_content.append(
                buildSuggestionModalForScatterplot(input_images_belong_to_same_class, input_images_class_id,
                    id_output_container, input_images_dir, data, suggestion, img_index, d)
            );

            // show the first slide for each suggestions in this container
            showSlides(id_output_container, slideIndex[id_output_container]);

            // open the modal
            suggestion_modal.style.display = "block";
        });
}

function buildSuggestion(input_images_belong_to_same_class, input_images_class_id, id_output_container,
                         input_images_dir, data, suggestion, tip, suggestion_modal, suggestion_modal_content) {

    let n_models = data["n_models"], description = "";

    if (n_models === 1) {
        if (suggestion["layer_mean_activations"] !== "") {
            if (data["layer_name"] === "") {
                description = "<p>This graph shows distribution of mean activation values at <b>last layer</b>, among the input images dataset.</p>"
            }
            else {
                description = "<p>This graph shows distribution of mean activation values at layer <b>'" +
                    data["layer_name"].toString() + "'</b>, among the input images dataset.</p>"
            }

            $("#chart_container_" + id_output_container).find(".description")
                .html(description + "<p>Click on a bin to check which of the input images belongs to it and to visualize a selection of them.</p>");

            buildSvgHistogram(suggestion["layer_mean_activations"], input_images_belong_to_same_class,
                input_images_class_id, id_output_container, input_images_dir, data, suggestion, tip, suggestion_modal,
                suggestion_modal_content, "mean activation at selected layer");
        }

        if (suggestion["neuron_activations"] !== "") {
            if (data["layer_name"] === "") {
                description = "<p>This graph shows distribution of activation values at neuron <b>{" +
                    data["neuron_index"].toString() + "} of last layer</b>, among the input images dataset.</p>";
            }
            else {
                description = "<p>This graph shows distribution of activation values at neuron <b>{" +
                    data["neuron_index"].toString() + "} of layer '" + data["layer_name"].toString() +
                    "'</b>, among the input images dataset.</p>";
            }

            $("#chart_container_" + id_output_container).find(".description")
                .html(description + "<p>Click on a bin to check which of the input images belongs to it and to visualize a selection of them.</p>");

            buildSvgHistogram(suggestion["neuron_activations"], input_images_belong_to_same_class, input_images_class_id, id_output_container,
                input_images_dir, data, suggestion, tip, suggestion_modal, suggestion_modal_content, "activation at selected neuron");
        }
    }
    else {
        if (suggestion["models_correct"] !== "") {
            $("#chart_container_" + id_output_container).find(".description")
                .html("<p>This graph shows distribution of the mean distance of the predictions of the selected models from the expected correct prediction based on the input class, among the input images dataset.</p><p>Click on a bin to check which of the input images belongs to it and to visualize a selection of them.</p>");

            buildSvgHistogram(suggestion["models_mean_distance_from_input_class"], input_images_belong_to_same_class,
                input_images_class_id, id_output_container, input_images_dir, data, suggestion, tip, suggestion_modal,
                suggestion_modal_content, "scores mean distance from input class");
        }

        if (suggestion["models_top_pred_mean_probs"] !== "") {
            $("#chart_container_" + id_output_container).find(".description")
                .html("<p>This graph shows an overview about the classification given by the input models on the input images dataset." +
                    "Each dot corresponds to an image, where the position of each one corresponds to the mean confidence of the most " +
                    "common prediction among the input models for that image, in relation to the number of different classification given " +
                    "to the same image.</p><p>Click on a dot to check which is the corresponding image and how the input models classified it.</p>");

            let x = [], y = [], k = 0, most_common_pred_mean_prob_per_img = -1;
            for (let models_top_pred_mean_probs_per_img of suggestion["models_top_pred_mean_probs"]) {
                if (models_top_pred_mean_probs_per_img.length === 1) {
                    most_common_pred_mean_prob_per_img = models_top_pred_mean_probs_per_img[0]["mean_prob"];
                }
                else {
                    models_top_pred_mean_probs_per_img.sort(function(a, b) {
                        if (b["models_index"].length - a["models_index"].length === 0) {
                            return b["mean_prob"] - a["mean_prob"];
                        }
                        else {
                            return b["models_index"].length - a["models_index"].length;
                        }
                    });

                    most_common_pred_mean_prob_per_img = models_top_pred_mean_probs_per_img[0]["mean_prob"];
                }

                x[k] = most_common_pred_mean_prob_per_img;
                y[k] = models_top_pred_mean_probs_per_img.length;
                k++;
            }

            buildSvgScatterplot(x, y, "confidence [%]",
                "n classification", input_images_belong_to_same_class, input_images_class_id,
                id_output_container, input_images_dir, data, suggestion, tip, suggestion_modal,
                suggestion_modal_content);
        }
    }
}

function buildSuggestionModalForHistogram(input_images_belong_to_same_class, input_images_class_id, id_output_container,
                                          input_images_dir, data, suggestion, valid_indexes_for_bin, d) {

    let l, html, thumbnail, input_image_src, k, basic_info_keys, basic_info, custom_model_selected, n_classes,
        input_images_visualization, image_name_parts, image_name_parts_length, image_name, image_names, n_models,
        predictions_container, prediction, top_n, default_top_n, custom_model_name_parts,
        custom_model_name_parts_length, custom_model_name, final_custom_model, predefined_model,
        layer_mean_activations, neuron_activations, models_correct, models_top_pred_mean_probs, n_images_to_show,
        index, images_list, models_mean_distance_from_input_class;

    n_models = data["n_models"];

    // build basic info html
    basic_info_keys = [
        "n_models",
        "method",
    ];

    basic_info = "";
    custom_model_selected = data["custom_model_selected"];
    if (custom_model_selected) {
        n_classes = data["custom_n_classes"];
        if (n_classes !== "") {
            basic_info += "<li><b>n classes</b>: " + n_classes.toString() + "</li>";
        }
    }
    else {
        n_classes = 1000;
        basic_info += "<li><b>n classes</b>: " + n_classes.toString() + "</li>";
    }

    for (let key in data) {
        if (basic_info_keys.includes(key)) {
            if (data[key] !== null && data[key] !== '') {
                let val = beautifyNames(key, data[key].toString());

                if (key === "lrp_epsilon_bias" || key === "lrp_alpha_beta_bias") {
                    basic_info += "   <li><b>bias</b>: " + val + "</li>";
                } else if (key === "alpha") {
                    basic_info += "   <li><b>&alpha;</b>: " + val + "</li>";
                } else if (key === "beta") {
                    basic_info += "   <li><b>&beta;</b>: " + val + "</li>";
                } else if (key === "epsilon" || key === "lrp_epsilon_IB_sequential_preset_AB_epsilon") {
                    basic_info += "   <li><b>&epsilon;</b>: " + val + "</li>";
                } else if (key === "deep_taylor_low") {
                    basic_info += "   <li><b>low</b>: " + val + "</li>";
                } else if (key === "deep_taylor_high") {
                    basic_info += "   <li><b>high</b>: " + val + "</li>";
                } else {
                    basic_info += "   <li><b>" + key.replace(/_/g, " ") + "</b>: " + val + "</li>";
                }
            }
        }
    }

    // build input images visualization html
    input_images_visualization = "" +
        "   <div class='left'>" +
        "       <div class='images_container'>";

    thumbnail = "   <div class='row'>";

    predictions_container = "";
    layer_mean_activations = "";
    neuron_activations = "";
    models_correct = "";
    models_mean_distance_from_input_class = "";
    models_top_pred_mean_probs = "";
    if (n_models === 1) {
        if (suggestion["model_predictions"] !== "") {
            predictions_container = "<div class='predictions_container'>";
        }
    }

    slideIndex[id_output_container] = 1;

    image_names = "";
    images_list = "";

    if (valid_indexes_for_bin.length >= 10) {
        n_images_to_show = 10;
    }
    else {
        n_images_to_show = valid_indexes_for_bin.length;
    }

    for (k = 0; k < valid_indexes_for_bin.length; k++) {
        index = valid_indexes_for_bin[k];

        image_name_parts = suggestion["input_images"][index].split("_");
        // last part contains nonce for making the file unique
        image_name_parts_length = image_name_parts.length - 1;
        image_name = image_name_parts[0];
        for (let j = 1; j < image_name_parts_length; j++) {
            image_name += "_" + image_name_parts[j];
        }

        if (k < n_images_to_show) {
            input_image_src = input_images_dir + suggestion["input_images"][index];
            input_images_visualization += "" +
                "       <div class='mySlides'>" +
                "           <div class='numbertext'>" + (k + 1) + " / " + n_images_to_show + "</div>" +
                "           <img class='image' alt='input image [" + id_output_container + "][" + k + "]' src='" + input_image_src + "'/>" +
                "       </div>";

            thumbnail += "  <div class='column'>" +
                "               <img class='demo cursor'" +
                "                   alt='input image [" + id_output_container + "][" + k + "]' " +
                "                   src='" + input_image_src + "' " +
                "                   onclick=\"currentSlide(" + id_output_container + ", " + (k + 1) + ")\"/>" +
                "           </div>";

            if (n_models === 1) {
                if (suggestion["model_predictions"] !== "") {
                    prediction = suggestion["model_predictions"][index];
                    default_top_n = n_classes < 5 ? n_classes : 5;
                    if (prediction.length > 0) {
                        top_n = default_top_n;

                        predictions_container += "" +
                            "                   <div class='predictions predictions_" + id_output_container + "_" + k + "'>" +
                            "                       <p>Model prediction:</p>" +
                            "                       <ul>";
                        for (l = 0; l < top_n; l++) {
                            predictions_container += "  <li><b>" + prediction[l]["className"] + "</b> (" +
                                prediction[l]["classId"] + ") with probability " +
                                (parseFloat(prediction[l]["score"]) * 100).toFixed(2) + "%</li>";
                        }
                        predictions_container += "  </ul>" +
                            "                   </div>";
                    }
                }


                if (suggestion["layer_mean_activations"] !== "") {
                    layer_mean_activations += "" +
                        "<li class='layer_mean_activations layer_mean_activations_" + id_output_container + "_" + k + "'>";

                    if (data["layer_name"] === "") {
                        layer_mean_activations += "<b>mean activation at last layer</b>: ";
                    } else {
                        layer_mean_activations += "<b>mean activation at layer '" + data["layer_name"].toString() + "'</b>: ";
                    }

                    layer_mean_activations += suggestion["layer_mean_activations"][index] + "</li>";
                }


                if (suggestion["neuron_activations"] !== "") {
                    neuron_activations += "" +
                        "<li class='neuron_activations neuron_activations_" + id_output_container + "_" + k + "'>";

                    if (data["layer_name"] === "") {
                        neuron_activations += "<b>activation of neuron {" + data["neuron_index"].toString() + "} of last layer</b>: ";
                    } else {
                        neuron_activations += "<b>activation of neuron {" + data["neuron_index"].toString() + "} of layer '" + data["layer_name"].toString() + "'</b>: ";
                    }

                    neuron_activations += suggestion["neuron_activations"][index] + "</li>";
                }
            }
            // n_models > 1
            else {
                if (suggestion["models_mean_distance_from_input_class"] !== "") {
                    models_mean_distance_from_input_class += "" +
                        "<li class='models_mean_distance_from_input_class models_mean_distance_from_input_class_" + id_output_container + "_" + k + "'>" +
                        "   <b>scores mean distance from '<i>" + suggestion["input_images_class_name"] + "</i>'</b>: " +
                        suggestion["models_mean_distance_from_input_class"][index].toExponential().toString() +
                        "</li>";
                }

                if (suggestion["models_correct"] !== "") {
                    models_correct += "" +
                        "<li class='models_correct models_correct_" + id_output_container + "_" + k + "'>" +
                        "   <b>models that predict '<i>" + suggestion["input_images_class_name"] + "</i>'</b>: [ ";

                    if (custom_model_selected) {
                        for (let l = 0; l < suggestion["models_correct"][index].length; l++) {
                            final_custom_model = data["final_custom_models"][l];

                            custom_model_name_parts = final_custom_model.name.split("_");
                            // last part contains nonce for making the file unique
                            custom_model_name_parts_length = custom_model_name_parts.length - 1;
                            custom_model_name = custom_model_name_parts[0];
                            for (let j = 1; j < custom_model_name_parts_length; j++) {
                                custom_model_name += "_" + custom_model_name_parts[j];
                            }

                            models_correct += custom_model_name + ", "
                        }
                    }
                    else {
                        for (let l = 0; l < suggestion["models_correct"][index].length; l++) {
                            predefined_model = data["predefined_models"][l];

                            models_correct += predefined_model + ", "
                        }
                    }

                    models_correct += "]</li>";
                }

                if (suggestion["models_top_pred_mean_probs"] !== "") {
                    for (let models_group of suggestion["models_top_pred_mean_probs"][index]) {
                        models_top_pred_mean_probs += "" +
                            "<li class='models_top_pred_mean_probs models_top_pred_mean_probs_" + id_output_container + "_" + k + "'>" +
                            "   model/s [ ";

                        if (custom_model_selected) {
                            for (let model_index of models_group["models_index"]) {
                                final_custom_model = data["final_custom_models"][parseInt(model_index)];

                                custom_model_name_parts = final_custom_model.name.split("_");
                                // last part contains nonce for making the file unique
                                custom_model_name_parts_length = custom_model_name_parts.length - 1;
                                custom_model_name = custom_model_name_parts[0];
                                for (let j = 1; j < custom_model_name_parts_length; j++) {
                                    custom_model_name += "_" + custom_model_name_parts[j];
                                }

                                models_top_pred_mean_probs += custom_model_name + ", "
                            }
                        }
                        else {
                            for (let model_index of models_group["models_index"]) {
                                predefined_model = data["predefined_models"][parseInt(model_index)];

                                models_top_pred_mean_probs += predefined_model + ", "
                            }
                        }

                        models_top_pred_mean_probs += "] prediction is <b>" + models_group["class_name"] + "</b> (" +
                            models_group["class_id"] + ") with mean probability " +
                            (parseFloat(models_group["mean_prob"]) * 100).toFixed(2).toString() + "%</li>";
                    }
                }
            }

            // save image name
            image_names += "<li class='image_name image_name_" + id_output_container + "_" + k + "'><b>image name</b>: " + image_name + "</li>";
        }

        images_list += "<span>" + image_name + ", </span>";
    }

    thumbnail += "  </div>";
    if (n_models === 1) {
        predictions_container += "</div>";
    }

    <!-- Next and previous buttons -->
    input_images_visualization += "" +
        "           <a class='prev' onclick=\"plusSlides(" + id_output_container + ", -1)\">&#10094;</a>" +
        "           <a class='next' onclick=\"plusSlides(" + id_output_container + ", 1)\">&#10095;</a>";

    input_images_visualization += thumbnail +
        "       </div>" +
        "   </div>";


    html = "<div id='output_container_" + id_output_container + "' class='output_container main_view'>" +
        "       <div class='suggestion'>" +
                    input_images_visualization +
        "           <div class='right'>" +
        "               <p class='title'>Suggestion</p>" +
        "               <ul>";

    if (d.x0 < 1 && d.x1 < 1) {
        html += "           <li><p>" + d.length + " images in bin [" + d.x0.toExponential() + ", " + d.x1.toExponential() + "]: { ";
    }
    else {
        html += "           <li><p>" + d.length + " images in bin [" + d.x0.toFixed(2) + ", " + d.x1.toFixed(2) + "]: { ";
    }

    html += "                   <span class='images_list'> " + images_list + " }</span></p>" +
        "                   </li>" +
                            image_names +
                            basic_info;

    if (n_models > 1) {
        // M models suggestion point 1 and points 4 and 5
        if (suggestion["input_images_class_name"] !== "") {
            html += "<li><b>input images' class</b>: " + suggestion["input_images_class_name"] + " (id: " + input_images_class_id + ")</li>";
        }

        html += models_mean_distance_from_input_class;
        html += models_correct;
        html += models_top_pred_mean_probs;
    }
    else {
        // 1 model, suggestion points 2 and 3
        if (custom_model_selected) {
            final_custom_model = data["final_custom_models"][0];

            custom_model_name_parts = final_custom_model.name.split("_");
            // last part contains nonce for making the file unique
            custom_model_name_parts_length = custom_model_name_parts.length - 1;
            custom_model_name = custom_model_name_parts[0];
            for (let j = 1; j < custom_model_name_parts_length; j++) {
                custom_model_name += "_" + custom_model_name_parts[j];
            }

            // print custom model name
            html += "       <li><b>model</b>: " + custom_model_name + "</li>";

            // print information about pre-processing
            if (final_custom_model.means === "" && final_custom_model.std_devs === "") {
                html += "   <li><b>pre-processing</b>: ImageNet </li>";
            }
            else if (final_custom_model.means !== "" && final_custom_model.std_devs === "") {
                html += "   <li><b>pre-processing</b>: subtract mean </li>";
                html += "   <li><b>custom mean</b>: " + final_custom_model.means.toString() + "</li>";
            }
            else if (final_custom_model.means !== "" && final_custom_model.std_devs !== "") {
                html += "   <li><b>pre-processing</b>: subtract mean then divide by standard deviation </li>";
                html += "   <li><b>custom mean</b>: " + final_custom_model.means.toString() + "</li>";
                html += "   <li><b>custom stddev</b>: " + final_custom_model.std_devs.toString() + "</li>";
            }
        }
        else {
            predefined_model = data["predefined_models"][0];

            // print predefined model name
            html += "       <li><b>model</b>: " + predefined_model + "</li>";

            if (predefined_model === "MobileNet") {
                // if specified, print additional parameters for MobileNet
                if (data["mobilenet_alpha"] !== "") {
                    html += "<li><b> MobileNet &alpha; </b>: " + data["mobilenet_alpha"].toString() + "</li>";
                }

                if (data["mobilenet_depth_multiplier"] !== "") {
                    html += "<li><b> MobileNet depth multiplier </b>: " + data["mobilenet_depth_multiplier"].toString() + "</li>";
                }

                if (data["mobilenet_dropout"] !== "") {
                    html += "<li><b> MobileNet dropout </b>: " + data["mobilenet_dropout"].toString() + "</li>";
                }
            }
            else if (predefined_model === "MobileNetV2") {
                // if specified, print additional parameters for MobileNetV2
                if (data["mobilenetv2_alpha"] !== "") {
                    html += "<li><b> MobileNetV2 &alpha; </b>: " + data["mobilenetv2_alpha"].toString() + "</li>";
                }
            }
        }

        html += layer_mean_activations;
        html += neuron_activations;
    }

    html += "           </ul>" +
                        predictions_container +
        "           </div>" +
        "       </div>" +
        "   </div>";

    return html;
}

function buildSuggestionModalForScatterplot(input_images_belong_to_same_class, input_images_class_id,
                                            id_output_container, input_images_dir, data, suggestion, img_index) {

    let html, thumbnail, input_image_src, basic_info_keys, basic_info, custom_model_selected, n_classes,
        input_images_visualization, image_name_parts, image_name_parts_length, image_name, image_names,
        custom_model_name_parts, custom_model_name_parts_length, custom_model_name, final_custom_model,
        predefined_model, models_top_pred_mean_probs, index;

    // build basic info html
    basic_info_keys = [
        "n_models",
        "method",
    ];

    basic_info = "";
    custom_model_selected = data["custom_model_selected"];
    if (custom_model_selected) {
        n_classes = data["custom_n_classes"];
        if (n_classes !== "") {
            basic_info += "<li><b>n classes</b>: " + n_classes.toString() + "</li>";
        }
    }
    else {
        n_classes = 1000;
        basic_info += "<li><b>n classes</b>: " + n_classes.toString() + "</li>";
    }

    for (let key in data) {
        if (basic_info_keys.includes(key)) {
            if (data[key] !== null && data[key] !== '') {
                let val = beautifyNames(key, data[key].toString());

                if (key === "lrp_epsilon_bias" || key === "lrp_alpha_beta_bias") {
                    basic_info += "   <li><b>bias</b>: " + val + "</li>";
                } else if (key === "alpha") {
                    basic_info += "   <li><b>&alpha;</b>: " + val + "</li>";
                } else if (key === "beta") {
                    basic_info += "   <li><b>&beta;</b>: " + val + "</li>";
                } else if (key === "epsilon" || key === "lrp_epsilon_IB_sequential_preset_AB_epsilon") {
                    basic_info += "   <li><b>&epsilon;</b>: " + val + "</li>";
                } else if (key === "deep_taylor_low") {
                    basic_info += "   <li><b>low</b>: " + val + "</li>";
                } else if (key === "deep_taylor_high") {
                    basic_info += "   <li><b>high</b>: " + val + "</li>";
                } else {
                    basic_info += "   <li><b>" + key.replace(/_/g, " ") + "</b>: " + val + "</li>";
                }
            }
        }
    }

    // build input images visualization html
    input_images_visualization = "" +
        "   <div class='left'>" +
        "       <div class='images_container'>";

    thumbnail = "   <div class='row'>";

    models_top_pred_mean_probs = "";

    slideIndex[id_output_container] = 1;

    image_names = "";

    index = img_index;

    input_image_src = input_images_dir + suggestion["input_images"][index];
    input_images_visualization += "" +
        "       <div class='mySlides'>" +
        "           <div class='numbertext'> 1 / 1 </div>" +
        "           <img class='image' alt='input image [" + id_output_container + "]' src='" + input_image_src + "'/>" +
        "       </div>";

    thumbnail += "  <div class='column'>" +
        "               <img class='demo cursor'" +
        "                   alt='input image [" + id_output_container + "]' " +
        "                   src='" + input_image_src + "' " +
        "                   onclick=\"currentSlide(" + id_output_container + ", 1)\"/>" +
        "           </div>";

    if (suggestion["models_top_pred_mean_probs"] !== "") {
            for (let models_group of suggestion["models_top_pred_mean_probs"][index]) {
                models_top_pred_mean_probs += "<li class='models_top_pred_mean_probs_" + id_output_container + "'>" +
                    "<p>model/s [ ";

                if (custom_model_selected) {
                    for (let model_index of models_group["models_index"]) {
                        final_custom_model = data["final_custom_models"][parseInt(model_index)];

                        custom_model_name_parts = final_custom_model.name.split("_");
                        // last part contains nonce for making the file unique
                        custom_model_name_parts_length = custom_model_name_parts.length - 1;
                        custom_model_name = custom_model_name_parts[0];
                        for (let j = 1; j < custom_model_name_parts_length; j++) {
                            custom_model_name += "_" + custom_model_name_parts[j];
                        }

                        models_top_pred_mean_probs += custom_model_name + ", "
                    }
                }
                else {
                    for (let model_index of models_group["models_index"]) {
                        predefined_model = data["predefined_models"][parseInt(model_index)];

                        models_top_pred_mean_probs += predefined_model + ", "
                    }
                }

                models_top_pred_mean_probs += "] prediction is <b>" + models_group["class_name"] + "</b> (" +
                    models_group["class_id"] + ") with mean probability " +
                    (parseFloat(models_group["mean_prob"]) * 100).toFixed(2).toString() + "%</p></li>";
            }
    }

    image_name_parts = suggestion["input_images"][index].split("_");
    // last part contains nonce for making the file unique
    image_name_parts_length = image_name_parts.length - 1;
    image_name = image_name_parts[0];
    for (let j = 1; j < image_name_parts_length; j++) {
        image_name += "_" + image_name_parts[j];
    }

    // save image name
    image_names += "<li class='image_name_" + id_output_container + "'><b>image name</b>: " + image_name + "</li>";

    thumbnail += "  </div>";

    <!-- Next and previous buttons -->
    input_images_visualization += "" +
        "           <a class='prev' onclick=\"plusSlides(" + id_output_container + ", -1)\">&#10094;</a>" +
        "           <a class='next' onclick=\"plusSlides(" + id_output_container + ", 1)\">&#10095;</a>";

    input_images_visualization += thumbnail +
        "       </div>" +
        "   </div>";

    html = "<div id='output_container_" + id_output_container + "' class='output_container main_view'>" +
        "       <div class='suggestion'>" +
                    input_images_visualization +
        "           <div class='right'>" +
        "               <p class='title'>Suggestion</p>" +
        "               <ul>";

    html += image_names +
        basic_info +
        models_top_pred_mean_probs;

    html += "           </ul>" +
        "           </div>" +
        "       </div>" +
        "   </div>";

    return html;
}


// Next/previous controls
function plusSlides(id_output_container, n) {
    showSlides(id_output_container, slideIndex[id_output_container] += n);
}

// Thumbnail image controls
function currentSlide(id_output_container, n) {
    showSlides(id_output_container, slideIndex[id_output_container] = n);
}

function showSlide(div, id_output_container, n) {
    let i, slides, dots;

    slides = div.find(".mySlides");
    dots = div.find(".demo");

    if (n > slides.length) {
        slideIndex[id_output_container] = 1
    }

    if (n < 1) {
        slideIndex[id_output_container] = slides.length;
    }

    for (i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";
    }

    for (i = 0; i < dots.length; i++) {
        dots[i].className = dots[i].className.replace(" active", "");
    }

    slides[slideIndex[id_output_container] - 1].style.display = "block";
    dots[slideIndex[id_output_container] - 1].className += " active";
}

// show an image in the slideshow and the corresponding predictions and selected class name (if any)
function showSlides(id_output_container, n) {
    let output_container_div = $("#output_container_" + id_output_container);

    output_container_div.find(".general_info").each(function(){
        showSlide($(this), id_output_container, n);

        $(this).find(".image_name").hide();
        $(this).find(".image_name_" + id_output_container + "_" + (slideIndex[id_output_container] - 1)).show();
    });

    output_container_div.find(".suggestion").each(function () {
        showSlide($(this), id_output_container, n);

        $(this).find(".image_name").hide();
        $(this).find(".image_name_" + id_output_container + "_" + (slideIndex[id_output_container] - 1)).show();

        $(this).find(".predictions").hide();
        $(this).find(".predictions_" + id_output_container + "_" + (slideIndex[id_output_container] - 1)).show();

        $(this).find(".layer_mean_activations").hide();
        $(this).find(".layer_mean_activations_" + id_output_container + "_" + (slideIndex[id_output_container] - 1)).show();

        $(this).find(".neuron_activations").hide();
        $(this).find(".neuron_activations_" + id_output_container + "_" + (slideIndex[id_output_container] - 1)).show();

        $(this).find(".models_correct").hide();
        $(this).find(".models_correct_" + id_output_container + "_" + (slideIndex[id_output_container] - 1)).show();

        $(this).find(".models_mean_distance_from_input_class").hide();
        $(this).find(".models_mean_distance_from_input_class_" + id_output_container + "_" + (slideIndex[id_output_container] - 1)).show();

        $(this).find(".models_top_pred_mean_probs").hide();
        $(this).find(".models_top_pred_mean_probs_" + id_output_container + "_" + (slideIndex[id_output_container] - 1)).show();
    });

    output_container_div.find(".visualization").each(function() {
        showSlide($(this), id_output_container, n);

        $(this).find(".predictions").hide();
        $(this).find(".predictions_" + id_output_container + "_" + (slideIndex[id_output_container] - 1)).show();

        $(this).find(".max_activations_neurons").hide();
        $(this).find(".max_activations_neurons_" + id_output_container + "_" + (slideIndex[id_output_container] - 1)).show();

        $(this).find(".selected_class").hide();
        $(this).find(".selected_class_" + id_output_container + "_" + (slideIndex[id_output_container] - 1)).show();
    });
}


// layers loading management
function ajaxLoadLayers(custom_model_selected, model_name, method_input, input, url) {
    let icon_refresh_layers = $("#layer_selection").find(".refresh");
    let layers_loader_spinner = $("#layers_loader_spinner");

    let data = {
        "custom_model_selected": custom_model_selected,
        "model_name": model_name,
        "method": method_input.val(),
    };

    return $.ajax({
        beforeSend: function () {
            $("#visualize").prop("disabled", true);
            $("#suggest").prop("disabled", true);

            input.prop("disabled", true);
            icon_refresh_layers.hide();
            layers_loader_spinner.show();
        },
        type: 'POST',
        url: url,
        data: JSON.stringify(data),
        contentType: "application/json",
        cache: false,
        processData: false,
        success: function (response) {
            if (response["status"] === "error") {
                // when a controlled error occurs
                input.closest("div").addClass("danger");
                alert(response["message"]);
            }
            else {
                // when success
                input.closest("div").removeClass("danger");
                input.prop("disabled", false);

                let layer_names = response["content"];
                input.autocomplete("option", "source", layer_names);
            }
        },
        error: function (response) {
            // when uncontrolled error occurs
        },
        complete: function () {
            layers_loader_spinner.hide();
            icon_refresh_layers.show();

            $("#visualize").prop("disabled", false);
            $("#suggest").prop("disabled", false);
        }
    });
}

function loadLayers(custom_model_selected, custom_models, predefined_models_div, predefined_model_button,
                    custom_model_button, method_input, layer_name_input, url) {

    let n_models;

    if (custom_model_selected === null) {
        // it means no options was chosen
        predefined_model_button.addClass("danger");
        custom_model_button.addClass("danger");

        return false;
    }
    else if (custom_model_selected) {
        let custom_model;

        n_models = 0;
        for (let possible_custom_model of custom_models) {
            if (possible_custom_model) {
                custom_model = possible_custom_model;
                n_models++;
            }
        }

        if (n_models === 1) {
            return ajaxLoadLayers(custom_model_selected,
                custom_model.name,
                method_input,
                layer_name_input,
                url);
        }
        else {
            return false;
        }
    }
    else {
        let predefined_models = [];

        n_models = 0;
        predefined_models_div.find("input:checked").each(function () {
            predefined_models[n_models] = $(this)[0].id;
            n_models++;
        });

        if (n_models === 1) {
            return ajaxLoadLayers(custom_model_selected,
                predefined_models[0],
                method_input,
                layer_name_input,
                url);
        }
        else {
            return false;
        }
    }
}


// classes loading management
function ajaxLoadClasses(select, url, custom_model_selected, custom_class_index, custom_n_classes, reason) {
    if (custom_model_selected && (custom_class_index === "" || custom_n_classes === "")) {
        return;
    }

    let icon_refresh_classes = $("#class_selection").find(".refresh");
    let classes_loader_spinner = $("#classes_loader_spinner");

    let data = {
        "custom_model_selected": custom_model_selected,
        "custom_class_index": custom_class_index,
        "custom_n_classes": custom_n_classes,
    };

    let i, n_classes, classes = {};
    $.ajax({
        beforeSend: function () {
            select.prop("disabled", true);

            if (reason === "visualization") {
                icon_refresh_classes.hide();
                classes_loader_spinner.show();
            }
            else {
                $("#input_images_class_modal").find("button").prop("disabled", true);
            }
        },
        type: 'POST',
        url: url,
        data: JSON.stringify(data),
        contentType: "application/json",
        cache: false,
        processData: false,
        success: function (response) {
            if (response["status"] === "error") {
                // when a controlled error occurs
                select.closest("div").addClass("danger");
                alert(response["message"]);
            } else {
                // when success
                select.closest("div").removeClass("danger");
                select.prop("disabled", false);

                if (reason === "visualization") {
                    select.append("<option class='added' value='-1' selected> Model output class</option>");
                }
                // if this method is called for loading classes for input used in suggestion, leave "Model output class" has an option does not make any sense: in case of suggestion, is the user that have to insert a specified class if all input images are belonging to it, otherwise, if input dataset is promiscuous, no class must be selected

                classes = response["content"];
                n_classes = classes.length;
                for (i = 0; i < n_classes; i++) {
                    select.append("<option class='added' value='" + i + "'>" + classes[i] + "</option>");
                }
            }
        },
        error: function (response) {
            // when uncontrolled error occurs
        },
        complete: function() {
            if (reason === "visualization") {
                classes_loader_spinner.hide();
                icon_refresh_classes.show();
            }
            else {
                $("#input_images_class_modal").find("button").prop("disabled", false);
            }
        }
    });
}

function resetClassLoading(class_id_input, custom_class_index_input, custom_n_classes_input, predefined_model_button,
                           custom_model_button) {

    custom_class_index_input.closest("div").removeClass("danger");
    custom_n_classes_input.closest("div").removeClass("danger");
    predefined_model_button.removeClass("danger");
    custom_model_button.removeClass("danger");

    class_id_input.prop("disabled", true);
    class_id_input.find(".added").remove();
    class_id_input.val("-2");
}

function loadClasses(custom_model_selected, custom_class_index_input, custom_class_index, custom_n_classes_input,
                     predefined_model_button, custom_model_button, class_id_input, url, reason="visualization") {

    resetClassLoading(class_id_input,
        custom_class_index_input, custom_n_classes_input,
        predefined_model_button, custom_model_button);

    let classes_loaded = true;

    if (custom_model_selected) {
        if (custom_class_index === "") {
            custom_class_index_input.closest("div").addClass("danger");
            classes_loaded = false;
        }

        if (custom_n_classes_input.val() === "") {
            custom_n_classes_input.closest("div").addClass("danger");
            classes_loaded = false;
        }
    } else if (custom_model_selected === null) {
        // it means no options was chosen
        predefined_model_button.addClass("danger");
        custom_model_button.addClass("danger");
        classes_loaded = false;
    }

    // if user choose predefined model, classes can be loaded in any case (always taken from imagenet)

    if (classes_loaded) {
        ajaxLoadClasses(class_id_input,
            url,
            custom_model_selected,
            custom_class_index,
            custom_n_classes_input.val(),
            reason);
    } else {
        return false;
    }
}


function openTab(button, tabName) {
    let i, tabcontent, tablinks;

    // get all elements with class="tabcontent" and hide them
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    // get all elements with class="tablinks" and remove the class "active"
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // show the current tab, and add an "active" class to the button that opened the tab
    document.getElementById(tabName).style.display = "block";
    button.addClass("active");
}