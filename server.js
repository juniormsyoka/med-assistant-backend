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

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const genAI = new GoogleGenerativeAI({ apiKey: process.env.GEMINI_API_KEY });

// Test endpoint
app.get("/api/test", (req, res) => {
  res.json({ message: "Server is working!", timestamp: new Date() });
});

// Real AI chat endpoint with streaming (still using Groq)
app.post("/api/chat", async (req, res) => {
  const { message } = req.body;
  if (!message) return res.status(400).json({ error: "Message is required" });

  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("X-Accel-Buffering", "no");

  console.log("Received message:", message);

  try {
    const prompt = `You are a helpful medication assistant. Respond to this: "${message}"`;

    const stream = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
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

// Insights endpoint (still using Groq)
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

const upload = multer({ storage: multer.memoryStorage() });

// Updated scan endpoint using Gemini
app.post("/api/scan", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    console.log("📷 Received image for scanning with Gemini...");

    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

    const prompt = `
      You are a medical assistant. Analyze this prescription or medication image and provide:

      Please extract:
      1. Medication/drug names
      2. Dosage information
      3. Frequency and instructions
      4. Any important medical notes or warnings

      If the image is unclear, not a prescription, or not medically related, please say so clearly.
      Format your response in a clear, patient-friendly way without using markdown.
      Be accurate and cautious - if you're unsure about anything, indicate the uncertainty.
    `;

    const image = {
      inlineData: {
        data: req.file.buffer.toString('base64'),
        mimeType: req.file.mimetype
      }
    };

    const result = await model.generateContent([prompt, image]);
    const response = await result.response;
    const analysis = response.text();

    res.json({ 
      text: analysis, 
      structured: { extracted_text: analysis },
      rawText: analysis 
    });
  } catch (error) {
    console.error("❌ Gemini scan error:", error);
    res.status(500).json({ error: "Failed to process medical image with Gemini" });
  }
});

// server.js - Updated transcribe endpoint using Gemini
app.post("/api/transcribe", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No audio file uploaded" });
    }

    console.log("🎤 Processing audio with Gemini...");

    const model = genAI.getGenerativeModel({ 
      model: "gemini-1.5-flash", // or "gemini-1.5-pro" for better accuracy
    });

    // Convert audio to base64
    const audioBase64 = req.file.buffer.toString('base64');
    
    const prompt = `
      Transcribe this audio message exactly as spoken. 
      Provide ONLY the raw transcription without any additional commentary, analysis, or formatting.
      
      If the audio contains medical terms, transcribe them accurately.
      If there's background noise or unclear speech, do your best to transcribe what you can hear.
      
      Transcription:
    `;

    // Create the audio part for Gemini
    const audioPart = {
      inlineData: {
        mimeType: req.file.mimetype, // e.g., 'audio/m4a', 'audio/mp4', etc.
        data: audioBase64
      }
    };

    // Generate content with both audio and text prompt
    const result = await model.generateContent([prompt, audioPart]);
    const response = await result.response;
    const transcription = response.text();

    console.log("✅ Gemini transcription result:", transcription);

    res.json({ 
      transcript: transcription,
      success: true,
      rawText: transcription
    });
    
  } catch (error) {
    console.error("❌ Gemini transcription error:", error);
    
    // More detailed error information
    let errorMessage = "Failed to transcribe audio";
    if (error.message.includes('audio format')) {
      errorMessage = "Unsupported audio format. Please try a different recording format.";
    } else if (error.message.includes('size')) {
      errorMessage = "Audio file too large. Please try a shorter recording.";
    }
    
    res.status(500).json({ 
      error: errorMessage,
      details: error.message 
    });
  }
});

// New endpoint for better audio processing (if you want to use Google Speech-to-Text)
app.post("/api/transcribe-advanced", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No audio file uploaded" });
    }

    // Here you would integrate with Google Cloud Speech-to-Text
    // Then use Gemini to analyze the medical content
    // This requires Google Cloud Speech-to-Text API credentials

    res.json({ 
      message: "Advanced audio processing endpoint - set up Google Speech-to-Text for full functionality",
      setup_required: true 
    });
  } catch (error) {
    console.error("Advanced transcribe error:", error);
    res.status(500).json({ error: "Advanced audio processing failed" });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`AI assistant backend running on port ${PORT}`);
  console.log(`📷 Image analysis: Gemini`);
  console.log(`🎤 Audio processing: Gemini`);
  console.log(`💬 Text chat: Groq`);
});