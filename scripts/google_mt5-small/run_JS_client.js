const inputText = "Kjo është një fjali shembull në shqip.";
const targetLanguage = "en";

fetch('http://localhost:5000/translate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ text: inputText, targetLanguage })
})
.then(response => response.json())
.then(data => {
  const translatedText = data.translatedText;
  console.log(translatedText);
})
.catch(error => {
  console.error('Error:', error);
});