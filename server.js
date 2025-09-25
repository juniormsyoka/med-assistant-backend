import express from "express";
import bodyParser from "body-parser";
import Groq from "groq-sdk";
import dotenv from "dotenv";
import cors from "cors";

dotenv.config();

const app = express();
app.use(bodyParser.json());
app.use(cors()); // ✅ allow requests from any origin (safe here)

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// Test endpoint
app.get("/api/test", (req, res) => {
  res.json({ message: "Server is working!", timestamp: new Date() });
});

// Real AI chat endpoint with streaming
app.post("/api/chat", async (req, res) => {
  const { message } = req.body;
  if (!message) return res.status(400).json({ error: "Message is required" });

  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("X-Accel-Buffering", "no"); // disable proxy buffering

  console.log("Received message:", message);

  try {
    const prompt = `You are a helpful medication assistant. Respond to this: "${message}"`;

    const stream = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant", // Groq’s fast free model
      messages: [{ role: "user", content: prompt }],
      stream: true,
    });

    for await (const chunk of stream) {
      const text = chunk.choices[0]?.delta?.content || "";
      if (text) {
        res.write(text);
      }
    }

    res.end();
  } catch (err) {
    console.error("Groq error:", err);
    res.status(500).end("Error with Groq AI service");
  }
});

// ✅ Use dynamic port for Render
const PORT = process.env.PORT || 5000;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`AI assistant backend running on port ${PORT}`);
});
