// testChat.js
const API_URL = "http://localhost:5000/api/chat";

async function testChat() {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "Hello, can you give me a tip?" }),
    });

    const data = await response.json();
    console.log("Response from backend:", data);
  } catch (err) {
    console.error("Error:", err);
  }
}

testChat();
