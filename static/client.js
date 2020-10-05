const selectImage = () =>{
    $("#file-chooser").click()
}

const showPicked = (input) => {
    const reader = new FileReader();
    reader.onload = e => {
        $("#img-viewer")[0].src = e.target.result;
        
    };
    reader.readAsDataURL(input.files[0]);

    
}

const analyse = () => {
    const uploadFiles = $("#file-chooser")[0].files;
    if (uploadFiles.length !==1){
        alert("Please select a file to analyze");
        return;
    }

    $("#analyse-btn")[0].innerHTML = "Analyzing..."
    const xhr = new XMLHttpRequest();
    const loc = window.location;
    xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/predict`, true)
    xhr.onerror = () => {
        alert(xhr.responseText);
    };
    xhr.onload = function(e) {
        console.log('Received')
        if (this.readyState === 4){
            const response = JSON.parse(e.target.responseText).response;
            let resultText;
            switch(response.category){
                case 'human':
                    resultText = `Hello human, \n You look like a ${response.breed} ;)`
                    break;
                case 'dog':
                    resultText = `Hey Dog. \n The predicted breed is ${response.breed}`
                    break;
                default:
                    resultText = 'Thats an unidentified object. ;). We don\'t support it yet';
            }
            $(".figure-caption")[0].innerHTML = resultText;
        }
        $("#analyse-btn")[0].innerHTML = "Analyze"
    };

    const fileData = new FormData();
    fileData.append("file", uploadFiles[0]);
    xhr.send(fileData);
}