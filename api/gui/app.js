document.getElementById("sendBtn").addEventListener("click", sendMessage);

document.getElementById("userInput").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        event.preventDefault(); 
        sendMessage(); 
    }
});

async function sendMessage() {
    const userInput = document.getElementById("userInput").value;
    if (!userInput) return; 

    appendMessage(userInput, 'user');

    document.getElementById("userInput").value = "";

    
    try {
        const response = await fetch("http://127.0.0.1:8000/chat/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ text: userInput }),
        });
        const data = await response.json();

        
        appendMessage(data.response, 'bot');
    } catch (error) {
        console.error("Error fetching response:", error);
        appendMessage("Sorry, something went wrong.", 'bot');
    }
}


function appendMessage(message, sender) {
    const chatBox = document.getElementById("chatBox");
    const messageElement = document.createElement("div");
    messageElement.textContent = message;
    messageElement.classList.add('message', sender);
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight; 
}
