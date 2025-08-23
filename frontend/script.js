document.getElementById('spamForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const emailText = document.getElementById('emailText').value;
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = 'Checking...';
    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: emailText })
        });
        const data = await response.json();
        if (data.prediction === 1) {
            resultDiv.textContent = 'Spam!';
            resultDiv.style.color = 'red';
        } else {
            resultDiv.textContent = 'Not Spam.';
            resultDiv.style.color = 'green';
        }
    } catch (error) {
        resultDiv.textContent = 'Error: Could not connect to server.';
        resultDiv.style.color = 'orange';
    }
});
