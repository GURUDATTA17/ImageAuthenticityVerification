// // # static/js/main.js
// document.addEventListener('DOMContentLoaded', function() {
//     const dropZone = document.getElementById('drop-zone');
//     const fileInput = document.getElementById('file-input');
//     const uploadBtn = document.getElementById('upload-btn');
//     const resultsSection = document.querySelector('.results-section');

//     // Handle drag and drop events
//     ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
//         dropZone.addEventListener(eventName, preventDefaults, false);
//     });

//     function preventDefaults(e) {
//         e.preventDefault();
//         e.stopPropagation();
//     }

//     ['dragenter', 'dragover'].forEach(eventName => {
//         dropZone.addEventListener(eventName, highlight, false);
//     });

//     ['dragleave', 'drop'].forEach(eventName => {
//         dropZone.addEventListener(eventName, unhighlight, false);
//     });

//     function highlight(e) {
//         dropZone.classList.add('dragover');
//     }

//     function unhighlight(e) {
//         dropZone.classList.remove('dragover');
//     }

//     dropZone.addEventListener('drop', handleDrop, false);
    
//     function handleDrop(e) {
//         const dt = e.dataTransfer;
//         const files = dt.files;
//         handleFiles(files);
//     }

//     uploadBtn.addEventListener('click', () => {
//         fileInput.click();
//     });

//     fileInput.addEventListener('change', function() {
//         handleFiles(this.files);
//     });

//     function handleFiles(files) {
//         if (files.length > 0) {
//             const formData = new FormData();
//             formData.append('file', files[0]);

//             fetch('/upload', {
//                 method: 'POST',
//                 body: formData
//             })
//             .then(response => response.json())
//             .then(data => {
//                 if (data.error) {
//                     alert(data.error);
//                     return;
//                 }
                
//                 // Update UI with results
//                 document.getElementById('uploaded-image').src = data.filepath;
//                 document.getElementById('bihpf-result').textContent = data.bihpf_result;
//                 document.getElementById('bihpf-confidence').textContent = data.bihpf_confidence;
//                 document.getElementById('capsnet-result').textContent = data.capsnet_result;
//                 document.getElementById('capsnet-confidence').textContent = data.capsnet_confidence;
//                 document.getElementById('final-result').textContent = data.final_result;
//                 document.getElementById('final-confidence').textContent = data.final_confidence;
//                 document.getElementById('winning-method').textContent = data.winning_method;
                
//                 resultsSection.style.display = 'block';
//             })
//             .catch(error => {
//                 console.error('Error:', error);
//                 alert('An error occurred while processing the image.');
//             });
//         }
//     }
// });
document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const resultsSection = document.querySelector('.results-section');

    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('dragover');
    }

    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }

    dropZone.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    uploadBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const formData = new FormData();
            formData.append('file', files[0]);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }
                
                // Update UI with results
                document.getElementById('uploaded-image').src = data.filepath;
                document.getElementById('bihpf-result').textContent = data.bihpf_result;
                document.getElementById('bihpf-confidence').textContent = `${data.bihpf_confidence}%`;
                document.getElementById('capsnet-result').textContent = data.capsnet_result;
                document.getElementById('capsnet-confidence').textContent = `${data.capsnet_confidence}%`;
                document.getElementById('keras-result').textContent = data.keras_result;
                document.getElementById('keras-confidence').textContent = `${data.keras_confidence}%`;
                document.getElementById('final-result').textContent = data.final_result;
                document.getElementById('final-confidence').textContent = `${data.final_confidence}%`;
                document.getElementById('winning-method').textContent = data.winning_method;
                
                resultsSection.style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while processing the image.');
            });
        }
    }
});
