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
import OpenAI from "openai";

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







const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const upload = multer({ storage: multer.memoryStorage() });

app.post("/api/scan", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    console.log("📷 Received image for analysis...");

    // Convert the uploaded image to base64
    const base64Image = req.file.buffer.toString("base64");
    const imageUrl = `data:${req.file.mimetype};base64,${base64Image}`;

    // Use GPT-4o (Vision) to interpret the prescription directly
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini", // or "gpt-4o" for more detailed reasoning
      messages: [
        {
          role: "system",
          content: "You are a friendly AI medication assistant.",
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: `This is a photo of a medical prescription.
Please identify:
1. Drug names, dosage, and frequency (if visible)
2. Any instructions (e.g. take with food)
3. If parts are unclear, note that they're hard to read
Then, provide a short, reassuring summary for the patient.`,
            },
            {
              type: "image_url",
              image_url: imageUrl,
            },
          ],
        },
      ],
    });

    const aiResponse = completion.choices[0]?.message?.content || "No information found.";

    res.json({
      success: true,
      text: aiResponse,
    });
  } catch (error) {
    console.error("❌ Vision scan error:", error);
    res.status(500).json({
      success: false,
      error: "Failed to process prescription image",
    });
  }
});




app.post("/api/transcribe", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No audio file uploaded" });
    }

    // Prepare the audio file buffer / form data
    const form = new FormData();
    form.append("file", req.file.buffer, {
      filename: req.file.originalname || "audio.m4a",
      contentType: req.file.mimetype || "audio/m4a",
    });
    form.append("model", "whisper-large-v3");  // or whisper-large-v3-turbo
    form.append("response_format", "json");     // default is json

    // Call Groq’s transcription endpoint
    const transcriptionResponse = await groq.fetch(
      "POST",
      "/openai/v1/audio/transcriptions",
      {
        body: form,
        headers: form.getHeaders(), // includes multipart boundary
      }
    );

    // transcriptionResponse should have { text, ... }
    const transcript = transcriptionResponse.text;

    // Now do NER extraction with Groq chat
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
  } catch  {
    console.error("Transcribe error:", err);
    return res.status(500).json({ error: "Transcription failed" });
  }
});



// ✅ Use dynamic port for Render
const PORT = process.env.PORT || 5000;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`AI assistant backend running on port ${PORT}`);
});