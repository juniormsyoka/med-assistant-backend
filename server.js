import express from "express";
import bodyParser from "body-parser";
import multer from "multer";
import Groq from "groq-sdk";
import dotenv from "dotenv";
import cors from "cors";
import PatternDetector from "./services/PatternDetector.js";

dotenv.config();

// Import routes
import moodRoutes from "./routes/moodRoutes.js";

const app = express();
app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' }));
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
   🧠 MOOD ROUTES
=================================*/
app.use("/api", moodRoutes);

/* ===============================
   📋 GEMINI MODELS
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
   💬 GROQ CHAT
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

// Updated insights endpoint with enhanced context
app.post("/api/insights", async (req, res) => {
  try {
    const { stats, logs, user_profile, compliance_insights, compliance_records } = req.body;
    
    // If compliance_records are provided, use enhanced analysis
    if (compliance_records && Array.isArray(compliance_records) && compliance_records.length > 0) {
      console.log(`🧠 Using enhanced analysis for ${compliance_records.length} records`);
      
      const patternAnalysis = patternDetector.analyzeComplianceData(compliance_records);
      
      const enhancedPrompt = `
        You are a clinical AI assistant analyzing medication adherence patterns.
        
        PATIENT PROFILE:
        • ${user_profile?.name || 'User'}
        • Medications: ${patternAnalysis.medication_count || 'Unknown'}
        • Average adherence: ${stats?.adherence || 0}%
        • Records analyzed: ${compliance_records.length}
        
        DETECTED PATTERNS:
        ${patternAnalysis.patterns.map(p => `• ${p}`).join('\n') || 'No specific patterns detected'}
        
        IDENTIFIED RISK FACTORS:
        ${patternAnalysis.risk_factors.map(f => `• ${f}`).join('\n') || 'No significant risk factors'}
        
        RECENT EVENTS:
        ${logs ? logs.slice(0, 3).map(log => 
          `• ${new Date(log.createdAt).toLocaleDateString()}: ${log.medicationName} - ${log.status}`
        ).join('\n') : 'No recent events'}
        
        TASK: Provide a personalized, actionable insights summary that:
        1. Highlights 1-2 key patterns observed
        2. Suggests 1 practical improvement strategy
        3. Acknowledges positive trends if present
        4. Uses encouraging, clinical tone
        5. Keep it under 150 words
      `;
      
      const completion = await groq.chat.completions.create({
        model: "llama-3.1-8b-instant",
        messages: [{ role: "user", content: enhancedPrompt }],
        temperature: 0.3,
      });
      
      const message = completion.choices[0]?.message?.content || "No insights available.";
      
      res.json({ 
        insight: message,
        patterns: patternAnalysis.patterns,
        risk_factors: patternAnalysis.risk_factors,
        generated_at: new Date().toISOString(),
        model_used: "llama-3.1-8b-instant",
        confidence: compliance_records.length >= 10 ? "high" : "medium",
        source: "enhanced_llm_analysis",
        analysis_metadata: {
          records_analyzed: compliance_records.length,
          medication_count: patternAnalysis.medication_count
        }
      });
      
    } else {
      // Fallback to original simple insights
      console.log(`🧠 Using basic insights (no compliance records provided)`);
      
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
      
      const message = completion.choices[0]?.message?.content || "No insights available.";
      
      res.json({ 
        insight: message,
        generated_at: new Date().toISOString(),
        source: "basic_llm_analysis",
        note: "For better insights, provide compliance_records"
      });
    }
    
  } catch (error) {
    console.error("Insights error:", error);
    res.status(500).json({ 
      error: "Could not generate insights",
      insight: "Unable to generate insights at this time. Please try again later."
    });
  }
});

/* ===============================
   📷 IMAGE SCAN
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{
            parts: [
              { text: prompt },
              { inlineData: { mimeType: mimeType, data: base64Data } }
            ]
          }],
          generationConfig: { temperature: 0.1, maxOutputTokens: 1000 }
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
   🎤 VOICE TRANSCRIPTION
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

    let mimeType = req.file.mimetype;
    if (mimeType === 'application/octet-stream') {
      if (req.file.originalname.endsWith('.m4a')) mimeType = 'audio/mp4';
      else if (req.file.originalname.endsWith('.mp3')) mimeType = 'audio/mpeg';
      else if (req.file.originalname.endsWith('.wav')) mimeType = 'audio/wav';
      else if (req.file.originalname.endsWith('.webm')) mimeType = 'audio/webm';
      else mimeType = 'audio/mpeg';
      console.log(`🔧 Corrected MIME type from ${req.file.mimetype} to ${mimeType}`);
    }

    if (req.file.size < 4 * 1024 * 1024) {
      try {
        const base64Audio = req.file.buffer.toString("base64");
        const modelName = "gemini-2.5-flash";
        const prompt = `
          Listen to this audio message and transcribe it accurately. 
          The user is speaking to a medical assistant about health concerns, medications, or symptoms.
          Provide a clear, verbatim transcription of everything you hear.
          IMPORTANT: Respond ONLY with the transcription, no additional commentary.
        `;

        console.log("🎯 Attempting Gemini transcription...");
        const response = await fetch(
          `https://generativelanguage.googleapis.com/v1/models/${modelName}:generateContent?key=${process.env.GEMINI_API_KEY}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              contents: [{
                parts: [
                  { text: prompt },
                  { inlineData: { mimeType: mimeType, data: base64Audio } }
                ]
              }],
              generationConfig: { temperature: 0.1, maxOutputTokens: 1000 }
            })
          }
        );

        if (response.ok) {
          const data = await response.json();
          if (data.candidates && data.candidates[0] && data.candidates[0].content) {
            const transcript = data.candidates[0].content.parts[0].text;
            console.log("✅ Gemini transcription successful:", transcript.substring(0, 100) + "...");
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
    
    message = message
      .replace(/^"(.*)"$/, '$1')
      .replace(/^\"(.*)\"$/, '$1')
      .replace(/\.$/, '')
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
   🔍 TEST GEMINI
=================================*/
app.get("/api/test-gemini", async (req, res) => {
  try {
    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({ error: "Gemini API key missing" });
    }

    console.log("🔧 Testing API key:", process.env.GEMINI_API_KEY.substring(0, 10) + "...");
    const modelName = "gemini-2.5-flash";
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1/models/${modelName}:generateContent?key=${process.env.GEMINI_API_KEY}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: "Hello, respond with 'OK' if working." }] }]
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


// Initialize pattern detector
const patternDetector = new PatternDetector();

/* ===============================
   📊 PATTERN ANALYSIS ENDPOINT
=================================*/
app.post("/api/analyze-patterns", async (req, res) => {
  try {
    const { compliance_records, user_id, medication_ids } = req.body;
    
    if (!compliance_records || !Array.isArray(compliance_records)) {
      return res.status(400).json({
        success: false,
        message: "compliance_records array is required",
        patterns: [],
        risk_factors: []
      });
    }
    
    console.log(`📊 Analyzing ${compliance_records.length} compliance records for patterns...`);
    
    // Analyze patterns
    const analysis = patternDetector.analyzeComplianceData(compliance_records);
    
    // Add metadata
    analysis.success = true;
    analysis.analyzed_at = new Date().toISOString();
    analysis.records_analyzed = compliance_records.length;
    analysis.user_id = user_id || "anonymous";
    analysis.medication_ids = medication_ids || [];
    
    console.log(`✅ Pattern analysis complete:`, {
      patterns: analysis.patterns.length,
      risk_factors: analysis.risk_factors.length,
      confidence: compliance_records.length >= 20 ? "high" : "medium"
    });
    
    res.json(analysis);
    
  } catch (error) {
    console.error("❌ Pattern analysis error:", error);
    res.status(500).json({
      success: false,
      message: "Pattern analysis failed",
      error: error.message,
      patterns: [],
      risk_factors: []
    });
  }
});

/* ===============================
   🧠 ENHANCED INSIGHTS ENDPOINT
=================================*/
app.post("/api/enhanced-insights", async (req, res) => {
  try {
    const { compliance_records, stats, user_profile } = req.body;
    
    if (!compliance_records || !Array.isArray(compliance_records)) {
      return res.status(400).json({
        success: false,
        message: "compliance_records array is required"
      });
    }
    
    console.log(`🧠 Generating enhanced insights for ${compliance_records.length} records...`);
    
    // Step 1: Pattern analysis
    const patternAnalysis = patternDetector.analyzeComplianceData(compliance_records);
    
    // Step 2: Prepare enhanced prompt for LLM
    const enhancedPrompt = createEnhancedPrompt(
      compliance_records,
      patternAnalysis,
      stats,
      user_profile
    );
    
    // Step 3: Get LLM insights with patterns as context
    const completion = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [
        {
          role: "system",
          content: "You are a clinical AI assistant that analyzes medication adherence patterns. Provide actionable, personalized insights."
        },
        {
          role: "user",
          content: enhancedPrompt
        }
      ],
      temperature: 0.3,
      max_tokens: 300
    });
    
    const llmInsight = completion.choices[0]?.message?.content || "No insights generated.";
    
    // Step 4: Combine results
    const response = {
      success: true,
      insight: llmInsight,
      patterns: patternAnalysis.patterns,
      risk_factors: patternAnalysis.risk_factors,
      analysis_metadata: {
        records_analyzed: compliance_records.length,
        medication_count: patternAnalysis.medication_count,
        analysis_period: patternAnalysis.analysis_period,
        generated_at: new Date().toISOString()
      },
      confidence: compliance_records.length >= 20 ? "high" : "medium",
      data_quality: {
        has_enough_data: compliance_records.length >= 10,
        records_count: compliance_records.length,
        recommendation: compliance_records.length < 10 ? 
          "More data needed for personalized insights" : 
          "Based on sufficient historical data"
      }
    };
    
    console.log(`✅ Enhanced insights generated`);
    res.json(response);
    
  } catch (error) {
    console.error("❌ Enhanced insights error:", error);
    res.status(500).json({
      success: false,
      message: "Failed to generate enhanced insights",
      error: error.message,
      insight: "Unable to generate insights at this time. Please try again later."
    });
  }
});

// Helper function to create enhanced prompt
function createEnhancedPrompt(compliance_records, patternAnalysis, stats, user_profile) {
  // Calculate basic stats if not provided
  let adherenceRate = 0;
  let taken = 0;
  let missed = 0;
  
  if (stats) {
    adherenceRate = stats.adherence || 0;
  } else {
    taken = compliance_records.filter(r => r.actualAction === 'taken').length;
    missed = compliance_records.filter(r => r.actualAction === 'missed').length;
    const total = compliance_records.length;
    adherenceRate = total > 0 ? (taken / total) * 100 : 0;
  }
  
  const patternsText = patternAnalysis.patterns.length > 0 
    ? patternAnalysis.patterns.map(p => `• ${p}`).join('\n')
    : "No strong patterns detected";
    
  const risksText = patternAnalysis.risk_factors.length > 0
    ? patternAnalysis.risk_factors.map(r => `• ${r}`).join('\n')
    : "No significant risk factors identified";
  
  return `
MEDICATION ADHERENCE ANALYSIS REQUEST

PATIENT PROFILE:
• Name: ${user_profile?.name || 'User'}
• Medications tracked: ${patternAnalysis.medication_count}
• Data period: ${patternAnalysis.analysis_period.start ? new Date(patternAnalysis.analysis_period.start).toLocaleDateString() : 'Unknown'} to ${patternAnalysis.analysis_period.end ? new Date(patternAnalysis.analysis_period.end).toLocaleDateString() : 'Unknown'}

PERFORMANCE SUMMARY:
• Total records analyzed: ${compliance_records.length}
• Adherence rate: ${adherenceRate.toFixed(1)}%
• Records: ${compliance_records.length} (Taken: ${taken}, Missed: ${missed})

DETECTED PATTERNS:
${patternsText}

IDENTIFIED RISK FACTORS:
${risksText}

TASK:
Provide a personalized, actionable insight that:
1. Acknowledges 1-2 key patterns found
2. Addresses the main risk factors
3. Suggests 1 practical improvement strategy
4. Uses encouraging, clinical tone
5. Is concise (2-3 sentences max)

Focus on being helpful, not judgmental. If data is limited, acknowledge that.
  `;
}






/* ===============================
   ⚙️ SERVER STARTUP
=================================*/
const PORT = process.env.PORT || 5000;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 AI Assistant backend running on port ${PORT}`);
  console.log(`💬 Chat: Groq`);
  console.log(`📷 Image analysis: Gemini 2.5 Flash`);
  console.log(`🎤 Voice: Gemini + Groq fallback`);
  console.log(`🔧 Available at: http://localhost:${PORT}`);
  console.log(`🧠 Mood Analysis:`);
  console.log(`   POST /api/mood {text: "message", analyzeFor: "user"|"ai"}`);
  console.log(`   POST /api/mood/batch {messages: [...], conversationId: "...", userId: "..."}`);
  console.log(`   GET /api/mood/batch/status/:batchId`);
});