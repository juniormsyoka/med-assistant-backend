import express from "express";
import bodyParser from "body-parser";
import multer from "multer";
import Groq from "groq-sdk";
import dotenv from "dotenv";
import cors from "cors";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

const app = express();
app.use(bodyParser.json());
app.use(cors());

/* ===============================
   🔧 Initialize AI Clients
=================================*/
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const upload = multer({ storage: multer.memoryStorage() });

/* ===============================
   ✅ Health Check Endpoint
=================================*/
app.get("/api/test", (req, res) => {
  res.json({ message: "Server is working!", timestamp: new Date() });
});

/* ===============================
   💬 AI Chat (Groq)
=================================*/
app.post("/api/chat", async (req, res) => {
  const { message } = req.body;
  if (!message) return res.status(400).json({ error: "Message is required" });

  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("X-Accel-Buffering", "no");

  console.log("🗨️ Received message:", message);

  try {
    const prompt = `You are a helpful medical assistant. Respond concisely and clearly to this user question: "${message}"`;

    const stream = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [{ role: "user", content: prompt }],
      stream: true,
    });

    for await (const chunk of stream) {
      const text = chunk.choices[0]?.delta?.content || "";
      if (text) res.write(text);
    }
    res.end();
  } catch (err) {
    console.error("❌ Groq error:", err);
    res.status(500).end("Error with Groq AI service");
  }
});

/* ===============================
   📊 Insights (Groq Summary)
=================================*/
app.post("/api/insights", async (req, res) => {
  try {
    const { stats, logs } = req.body;

    const summaryPrompt = `
      You are a medication adherence coach.
      Based on these stats: ${JSON.stringify(stats)}
      and logs: ${logs
        .map((log) => `${log.medicationName} - ${log.status}`)
        .join(", ")}
      Write a short, encouraging summary (≤3 sentences).
      Focus on patterns, improvements, and suggestions.
    `;

    const completion = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [{ role: "user", content: summaryPrompt }],
    });

    const message =
      completion.choices[0]?.message?.content || "No insights available.";
    res.json({ insight: message });
  } catch (error) {
    console.error("Insights error:", error);
    res.status(500).json({ error: "Could not generate insights" });
  }
});

/* ===============================
   📷 Image Scan (Gemini)
=================================*/
app.post("/api/scan", upload.single("file"), async (req, res) => {
  console.log("📸 /api/scan called for image analysis");

  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }
    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({ error: "Gemini API key missing" });
    }

    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

    const prompt = `
      You are a medical assistant analyzing a prescription or medication image.
      Extract all readable text and identify:
      - Drug names and dosages
      - Frequency or duration
      - Instructions or warnings
      - If unclear or unrelated, say so clearly.
      Format:
      EXTRACTED TEXT: ...
      ANALYSIS: ...
    `;

    const imagePart = {
      inlineData: {
        mimeType: req.file.mimetype,
        data: req.file.buffer.toString("base64"),
      },
    };

    console.log("🚀 Sending image to Gemini...");
    const result = await model.generateContent([prompt, imagePart]);
    const analysis = result.response.text();

    console.log("✅ Gemini image analysis complete");
    res.json({
      text: analysis,
      analysis,
      rawText: analysis,
      success: true,
    });
  } catch (error) {
    console.error("❌ Gemini analysis error:", error);
    res.status(500).json({
      error: "Gemini analysis failed",
      details: error.message,
    });
  }
});

/* ===============================
   🎤 Voice Transcription (Groq Fallback)
=================================*/
app.post("/api/transcribe", upload.single("file"), async (req, res) => {
  try {
    if (!req.file)
      return res.status(400).json({ error: "No audio file uploaded" });

    console.log("🎤 Audio file received for transcription fallback");

    const prompt = `
      A user recorded a voice message but transcription isn't available.
      Write a warm, medical-friendly message that:
      1. Acknowledges their voice message was received.
      2. Explains that transcription features are being upgraded.
      3. Encourages them to type their message for now.
      Keep it under 2 sentences.
    `;

    const completion = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [{ role: "user", content: prompt }],
      max_tokens: 100,
    });

    let message =
      completion.choices[0]?.message?.content ||
      "Voice message received! We're upgrading our voice features.";
    message = message.replace(/^"(.*)"$/, "$1").trim();

    res.json({
      transcript: message,
      success: true,
      note: "Groq fallback (no Gemini transcription yet)",
    });
  } catch (error) {
    console.error("❌ Transcription error:", error);
    res.json({
      transcript:
        "🎤 Voice message received! We're upgrading our voice recognition. Please type your message for now.",
      success: true,
      note: "Static fallback response",
    });
  }
});

/* ===============================
   ⚙️ Server Startup
=================================*/
const PORT = process.env.PORT || 5000;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 AI Assistant backend running on port ${PORT}`);
  console.log(`💬 Chat: Groq`);
  console.log(`📷 Image analysis: Gemini`);
  console.log(`🎤 Voice fallback: Groq`);
});
