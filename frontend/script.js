function predictImage() {
    const fileInput = document.getElementById('imageUpload');
    const resultsDiv = document.getElementById('results');
    const bihpfResult = document.getElementById('bihpfResult');
    const capsnetResult = document.getElementById('capsnetResult');
    const finalResult = document.getElementById('finalResult');

    if (fileInput.files.length === 0) {
        alert("Please upload an image.");
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            bihpfResult.textContent = `BiHPF Prediction: ${data.bihpf.label} (Confidence: ${data.bihpf.confidence.toFixed(2)}%)`;
            capsnetResult.textContent = `CapsNet Prediction: ${data.capsnet.label} (Confidence: ${data.capsnet.confidence.toFixed(2)}%)`;
            finalResult.textContent = `Final Prediction: ${data.final.label} (Confidence: ${data.final.confidence.toFixed(2)}%)`;
            resultsDiv.style.display = 'block';
        })
        .catch((error) => {
            alert("Error during prediction: " + error.message);
        });
}
