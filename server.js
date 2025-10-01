import express from "express";
import bodyParser from "body-parser";
import Tesseract from "tesseract.js";
import multer from "multer";
import Groq from "groq-sdk";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import axios from "axios";
import FormData from "form-data";
import cors from "cors";

dotenv.config();

const app = express();
app.use(bodyParser.json());
app.use(cors()); // ✅ allow requests from any origin (safe here)

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

if (!fs.existsSync("uploads")) {
  fs.mkdirSync("uploads");
}

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



// server.js (add this endpoint)
app.post("/api/insights", async (req, res) => {
  try {
    const { stats, logs } = req.body;

    const summaryPrompt = `
    You are a medication adherence coach.
    Based on these stats: ${JSON.stringify(stats)}
    and logs: ${logs.map(log => `${log.medicationName} - ${log.status}`).join(", ")}

    Write a short, encouraging summary for the patient.
    Focus on patterns (time of day, missed vs late, improvements).
    Keep it under 3 sentences.
    `;

    const completion = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [{ role: "user", content: summaryPrompt }],
    });

    const message = completion.choices[0]?.message?.content || "No insights available.";
    res.json({ insight: message });
  } catch (error) {
    console.error("Insights error:", error);
    res.status(500).json({ error: "Could not generate insights" });
  }
});





//const upload = multer({ storage: multer.memoryStorage() });


const upload = multer({ dest: "uploads/" });

app.post("/api/scan", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    console.log("📷 Received image for scanning...");

    const ocrResult = await Tesseract.recognize(req.file.buffer, "eng");
    const rawText = ocrResult.data.text.trim();

    if (!rawText) {
      return res.status(400).json({ error: "No text detected in image" });
    }

    // send to Groq
   /* const prompt = `
      Extract structured information from this prescription text:
      "${rawText}"
      Return JSON with keys: drug_name, dosage, frequency, instructions.
    `; */

    const prompt = `
You are a friendly AI medication assistant.
Here is OCR text from a prescription: """${rawText}"""

Your task:
1. Try to extract possible drug names, dosage, frequency, and instructions.
2. If parts are unclear, don't output "No clear information". Instead, say "This section is hard to read" or "unclear text".
3. Always end with a simple summary in plain English for the patient.
4. Keep it concise and reassuring.
Return your answer as a short human-readable explanation, not raw JSON.
`;

    const completion = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [{ role: "user", content: prompt }],
    });

    const aiResponse =
      completion.choices[0]?.message?.content || "No details extracted";

    let structured;
    try {
      structured = JSON.parse(aiResponse);
    } catch {
      structured = { extracted_text: aiResponse };
    }

    res.json({ text: aiResponse, structured, rawText });
  } catch (error) {
    console.error("❌ Scan error:", error);
    res.status(500).json({ error: "Failed to process prescription image" });
  }
});



app.post("/api/transcribe", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No audio file uploaded" });
    }

    const filePath = req.file.path;

    const form = new FormData();
    form.append("file", fs.createReadStream(filePath));
    form.append("model", "whisper-large-v3");
    form.append("response_format", "json");

    const transcriptionResponse = await groq.fetch(
      "POST",
      "/openai/v1/audio/transcriptions",
      {
        body: form,
        headers: form.getHeaders(),
      }
    );

    fs.unlink(filePath, (err) => {
      if (err) console.error("Failed to delete temp file:", err);
    });

    const transcript = transcriptionResponse.text || "";

    const nerPrompt = `
You are a clinical assistant. Return ONLY valid JSON containing:
{
  "symptoms": [ { "name": string, "severity": string | null, "duration": string | null, "onset": string | null, "modifiers": string | null } ],
  "summary": string
}
Here is the transcript: """${transcript}"""
If information is missing, set fields to null.
`;

    const nerCompletion = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [{ role: "user", content: nerPrompt }],
      temperature: 0,
    });

    const aiRespText = nerCompletion.choices[0]?.message?.content || "";

    let structured;
    try {
      structured = JSON.parse(aiRespText);
    } catch (e) {
      structured = { extracted_text: aiRespText };
    }

    return res.json({ transcript, structured });
  } catch (err) {
    console.error("Transcribe error:", err?.message || err);
    return res.status(500).json({ error: "Transcription failed" });
  }
});




// ✅ Use dynamic port for Render
const PORT = process.env.PORT || 5000;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`AI assistant backend running on port ${PORT}`);
});