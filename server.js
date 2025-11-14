import express from "express";
import bodyParser from "body-parser";
import multer from "multer";
import Groq from "groq-sdk";
import dotenv from "dotenv";
import cors from "cors";

dotenv.config();
console.log("DEBUG: Loaded GEMINI KEY?", process.env.GEMINI_API_KEY);

const app = express();
app.use(bodyParser.json());
app.use(cors());

/* ===============================
   🔧 Initialize AI Clients
=================================*/
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const upload = multer({ storage: multer.memoryStorage() });

// Request logging middleware
app.use((req, res, next) => {
  console.log('🌐 Incoming request:', {
    method: req.method,
    url: req.url,
    origin: req.headers.origin,
    'content-type': req.headers['content-type'],
    'content-length': req.headers['content-length']
  });
  next();
});

/* ===============================
   ✅ Health Check Endpoint
=================================*/
app.get("/api/test", (req, res) => {
  res.json({ message: "Server is working!", timestamp: new Date() });
});

/* ===============================
   📋 List Available Gemini Models
=================================*/
app.get("/api/models", async (req, res) => {
  try {
    const response = await fetch(`https://generativelanguage.googleapis.com/v1/models?key=${process.env.GEMINI_API_KEY}`);
    const data = await response.json();
    
    res.json({
      success: true,
      models: data.models?.map(m => ({
        name: m.name,
        displayName: m.displayName,
        supportedMethods: m.supportedGenerationMethods
      })) || []
    });
  } catch (error) {
    res.status(500).json({
      error: "Failed to list models",
      details: error.message
    });
  }
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
   📷 Image Scan (Gemini) - UPDATED WITH LATEST MODELS
=================================*/
app.post("/api/scan", upload.single("file"), async (req, res) => {
  console.log("📸 /api/scan called");
  
  try {
    if (!req.file) {
      console.log("❌ No file uploaded");
      return res.status(400).json({ error: "No file uploaded" });
    }
    
    if (!process.env.GEMINI_API_KEY) {
      console.log("❌ Gemini API key missing");
      return res.status(500).json({ error: "Gemini API key missing" });
    }

    const mimeType = req.file.mimetype || "image/jpeg";
    console.log("📦 File details:", { 
      size: req.file.size, 
      mimeType,
      originalName: req.file.originalname
    });

    if (req.file.size === 0) {
      console.log("❌ Empty file uploaded");
      return res.status(400).json({ error: "Empty file uploaded" });
    }

    const base64Data = req.file.buffer.toString("base64");

    // Use the latest Gemini 2.5 Flash model
    const modelName = "gemini-2.5-flash";
    
    const prompt = `
      You are a medical assistant analyzing a prescription or medication image.
      Extract all readable text and identify:
      - Drug names and dosages
      - Frequency or duration  
      - Instructions or warnings
      - Patient information
      - Doctor information
      - If unclear or unrelated, say so clearly.
      
      Format your response as:
      EXTRACTED TEXT: [all text you can read]
      ANALYSIS: [your analysis of the medication information]
    `;

    console.log("🚀 Sending to Gemini 2.5 Flash...");

    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1/models/${modelName}:generateContent?key=${process.env.GEMINI_API_KEY}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: [{
            parts: [
              { text: prompt },
              {
                inlineData: {
                  mimeType: mimeType,
                  data: base64Data
                }
              }
            ]
          }],
          generationConfig: {
            temperature: 0.1,
            maxOutputTokens: 1000,
          }
        })
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("❌ Gemini API error:", errorText);
      throw new Error(`Gemini API error: ${errorText}`);
    }

    const data = await response.json();
    const text = data.candidates[0].content.parts[0].text;
    
    console.log("✅ Gemini analysis complete");
    console.log("📝 Response preview:", text.substring(0, 200) + "...");
    
    res.json({ 
      success: true,
      analysis: text,
      extractedText: text,
      debug: {
        fileSize: req.file.size,
        mimeType: mimeType,
        responseLength: text.length,
        modelUsed: modelName
      }
    });
    
  } catch (error) {
    console.error("❌ Scan error:", error.message);
    
    res.status(500).json({
      error: "Image analysis failed",
      details: error.message,
      type: "GeminiAPIError"
    });
  }
});

/* ===============================
   🔍 Test Gemini API Key
=================================*/
app.get("/api/test-gemini", async (req, res) => {
  try {
    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({ error: "Gemini API key missing" });
    }

    console.log("🔧 Testing API key:", process.env.GEMINI_API_KEY.substring(0, 10) + "...");

    // Use gemini-2.5-flash for testing
    const modelName = "gemini-2.5-flash";
    
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1/models/${modelName}:generateContent?key=${process.env.GEMINI_API_KEY}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: [{
            parts: [{ text: "Hello, respond with 'OK' if working." }]
          }]
        })
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`GenerateContent failed: ${errorText}`);
    }

    const data = await response.json();
    const responseText = data.candidates[0].content.parts[0].text;

    res.json({
      success: true,
      message: "Gemini API is working!",
      workingModel: modelName,
      response: responseText
    });

  } catch (error) {
    console.error("❌ Gemini test failed:", error.message);
    
    res.status(500).json({
      error: "Gemini API test failed",
      details: error.message
    });
  }
});

/* ===============================
   🎤 Voice Transcription - IMPROVED VERSION
=================================*/
app.post("/api/transcribe", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No audio file uploaded" });
    }

    console.log("🎤 Audio file received:", {
      size: req.file.size,
      mimetype: req.file.mimetype,
      originalname: req.file.originalname,
      bufferLength: req.file.buffer.length
    });

    // Determine the correct MIME type based on file extension
    let mimeType = req.file.mimetype;
    if (mimeType === 'application/octet-stream') {
      // Detect MIME type from filename
      if (req.file.originalname.endsWith('.m4a')) {
        mimeType = 'audio/mp4';
      } else if (req.file.originalname.endsWith('.mp3')) {
        mimeType = 'audio/mpeg';
      } else if (req.file.originalname.endsWith('.wav')) {
        mimeType = 'audio/wav';
      } else if (req.file.originalname.endsWith('.webm')) {
        mimeType = 'audio/webm';
      } else {
        mimeType = 'audio/mpeg'; // default fallback
      }
      console.log(`🔧 Corrected MIME type from ${req.file.mimetype} to ${mimeType}`);
    }

    // Try to use Gemini for transcription (for files under 4MB)
    if (req.file.size < 4 * 1024 * 1024) {
      try {
        const base64Audio = req.file.buffer.toString("base64");
        const modelName = "gemini-2.5-flash";
        
        const prompt = `
          Listen to this audio message and transcribe it accurately. 
          The user is speaking to a medical assistant about health concerns, medications, or symptoms.
          Provide a clear, verbatim transcription of everything you hear.
          If there are unclear parts, transcribe what you can and note any uncertainties.
          IMPORTANT: Respond ONLY with the transcription, no additional commentary.
        `;

        console.log("🎯 Attempting Gemini transcription...");

        const response = await fetch(
          `https://generativelanguage.googleapis.com/v1/models/${modelName}:generateContent?key=${process.env.GEMINI_API_KEY}`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              contents: [{
                parts: [
                  { text: prompt },
                  {
                    inlineData: {
                      mimeType: mimeType,
                      data: base64Audio
                    }
                  }
                ]
              }],
              generationConfig: {
                temperature: 0.1,
                maxOutputTokens: 1000,
              }
            })
          }
        );

        if (response.ok) {
          const data = await response.json();
          if (data.candidates && data.candidates[0] && data.candidates[0].content) {
            const transcript = data.candidates[0].content.parts[0].text;
            
            console.log("✅ Gemini transcription successful:", transcript.substring(0, 100) + "...");
            
            // Check if the response is actually a transcription or an error
            if (transcript && !transcript.includes("I cannot") && !transcript.includes("audio format") && transcript.length > 10) {
              return res.json({
                transcript: transcript.trim(),
                success: true,
                note: "Transcribed by Gemini",
                method: "gemini",
                audioDetails: {
                  size: req.file.size,
                  mimeType: mimeType,
                  duration: "unknown"
                }
              });
            } else {
              console.log("⚠️ Gemini returned non-transcription response:", transcript);
              throw new Error("Gemini did not provide a valid transcription");
            }
          }
        } else {
          const errorText = await response.text();
          console.log("❌ Gemini API error:", errorText);
          throw new Error(`Gemini API: ${response.status}`);
        }
      } catch (geminiError) {
        console.log("🔁 Gemini transcription failed, falling back:", geminiError.message);
      }
    } else {
      console.log("📁 File too large for Gemini transcription:", req.file.size, "bytes");
    }

    // Enhanced Groq fallback with better context
    console.log("🔄 Using enhanced Groq fallback for transcription");
    
    const prompt = `
      A user recorded a ${Math.round(req.file.size/1024)}KB voice message for a medical assistant.
      Create a SHORT, warm response that:
      1. Thanks them for the voice message
      2. Explains that voice transcription is being improved
      3. Encourages typing for immediate help
      4. Keep it to 1-2 sentences maximum
      
      Make it sound natural and helpful, not robotic.
    `;

    const completion = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [{ role: "user", content: prompt }],
      max_tokens: 80,
      temperature: 0.7,
    });

    let message = completion.choices[0]?.message?.content ||
      "Thanks for your voice message! We're working on better voice recognition. For now, typing will get you the fastest help.";
    
    // Clean up the response
    message = message
      .replace(/^"(.*)"$/, '$1')
      .replace(/^\"(.*)\"$/, '$1')
      .replace(/\.$/, '') // Remove trailing period
      .trim();

    console.log("🎯 Groq fallback response:", message);

    res.json({
      transcript: message,
      success: true,
      note: "Voice transcription upgrading - type for immediate help",
      method: "groq-fallback",
      audioDetails: {
        size: req.file.size,
        mimeType: mimeType,
        detected: "Voice message received successfully"
      }
    });

  } catch (error) {
    console.error("❌ Transcription error:", error);
    res.json({
      transcript: "Thanks for your voice message! We're currently upgrading voice features. Please type your question for immediate assistance.",
      success: true,
      note: "Static fallback - type for fastest help",
      method: "error-fallback"
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
  console.log(`📷 Image analysis: Gemini 2.5 Flash`);
  console.log(`🎤 Voice: Gemini + Groq fallback`);
  console.log(`🔧 Available at: http://localhost:${PORT}`);
});