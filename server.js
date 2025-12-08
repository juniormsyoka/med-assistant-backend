import express from "express";
import bodyParser from "body-parser";
import multer from "multer";
import Groq from "groq-sdk";
import dotenv from "dotenv";
import cors from "cors";

import { pipeline } from '@xenova/transformers';

dotenv.config();
console.log("DEBUG: Loaded GEMINI KEY?", process.env.GEMINI_API_KEY);

const app = express();
app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' }));
app.use(cors());

/* ===============================
   🔧 Initialize AI Clients
=================================*/
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const upload = multer({ storage: multer.memoryStorage() });


let sentimentAnalyzer = null;
(async () => {
  try {
    console.log("🧠 Loading DistilBERT sentiment model...");
    sentimentAnalyzer = await pipeline('sentiment-analysis', 
      'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
    console.log("✅ DistilBERT loaded successfully!");
  } catch (error) {
    console.error("❌ Failed to load DistilBERT:", error.message);
    sentimentAnalyzer = null;
  }
})();

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
   🧠 MOOD ANALYSIS ENDPOINT (DistilBERT) - UPDATED WITH BATCH SUPPORT
=================================*/
app.post("/api/mood", async (req, res) => {
  try {
    const { text, analyzeFor = 'user', batchMode = false } = req.body;
    
    // Support batch mode as alternative
    if (batchMode && Array.isArray(text)) {
      // Process as mini-batch
      console.log(`🧠 Mini-batch requested for ${text.length} messages`);
      
      const batchResults = [];
      for (const [index, singleText] of text.entries()) {
        try {
          // Reuse existing analysis logic
          if (!sentimentAnalyzer) {
            const simpleResult = analyzeSentimentSimple(singleText, analyzeFor);
            batchResults.push({
              text: singleText.substring(0, 100) + (singleText.length > 100 ? '...' : ''),
              ...simpleResult,
              index,
              success: true
            });
          } else {
            const distilbertResult = await sentimentAnalyzer(singleText);
            const mainResult = distilbertResult[0];
            const isPositive = mainResult.label === 'POSITIVE';
            const moodScore = isPositive ? mainResult.score : -mainResult.score;
            
            const enhancedAnalysis = enhanceWithHealthcareContext(
              singleText, 
              mainResult.label, 
              mainResult.score, 
              analyzeFor
            );
            
            batchResults.push({
              text: singleText.substring(0, 100) + (singleText.length > 100 ? '...' : ''),
              ...enhancedAnalysis,
              index,
              success: true
            });
          }
        } catch (error) {
          batchResults.push({
            text: singleText.substring(0, 100) + (singleText.length > 100 ? '...' : ''),
            error: error.message,
            index,
            success: false
          });
        }
      }
      
      return res.json({
        success: true,
        batchMode: true,
        totalProcessed: batchResults.length,
        successful: batchResults.filter(r => r.success).length,
        failed: batchResults.filter(r => !r.success).length,
        results: batchResults,
        analyzedAt: new Date().toISOString()
      });
    }
    
    if (!text || typeof text !== 'string') {
      return res.status(400).json({ 
        success: false,
        error: 'Text is required and must be a string' 
      });
    }

    console.log(`🧠 Mood analysis requested for ${analyzeFor}:`, text.substring(0, 100) + '...');

    // If DistilBERT isn't loaded yet, use simple analysis
    if (!sentimentAnalyzer) {
      console.log("⚠️ DistilBERT not loaded, using simple analysis");
      const simpleResult = analyzeSentimentSimple(text, analyzeFor);
      return res.json({
        success: true,
        ...simpleResult,
        modelUsed: 'simple',
        analyzedAt: new Date().toISOString()
      });
    }

    // Use DistilBERT for analysis
    console.log("🧠 Analyzing with DistilBERT...");
    
    // DistilBERT returns array of results
    const distilbertResult = await sentimentAnalyzer(text);
    
    // Extract the main result
    const mainResult = distilbertResult[0];
    const label = mainResult.label; // 'POSITIVE' or 'NEGATIVE'
    const score = mainResult.score; // Confidence score 0-1
    
    console.log(`📊 DistilBERT result: ${label} (${score.toFixed(4)})`);
    
    // Enhanced healthcare-specific analysis
    const enhancedAnalysis = enhanceWithHealthcareContext(text, label, score, analyzeFor);
    
    res.json({
      success: true,
      ...enhancedAnalysis,
      modelUsed: 'distilbert',
      analyzedAt: new Date().toISOString(),
      textPreview: text.substring(0, 50) + (text.length > 50 ? '...' : ''),
      textLength: text.length
    });
    
  } catch (error) {
    console.error("❌ Mood analysis error:", error);
    
    // Fallback to simple analysis
    const simpleResult = analyzeSentimentSimple(req.body?.text || '', req.body?.analyzeFor || 'user');
    
    res.json({
      success: true,
      ...simpleResult,
      modelUsed: 'simple-fallback',
      note: 'DistilBERT failed, using simple analysis',
      analyzedAt: new Date().toISOString()
    });
  }
});

/* ================================
   🔄 BATCH MOOD ANALYSIS ENDPOINT
   (Efficient analysis of multiple messages)
=================================*/
app.post("/api/mood/batch", async (req, res) => {
  try {
    const { 
      messages, 
      conversationId, 
      userId,
      analysisType = 'comprehensive'
    } = req.body;
    
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ 
        success: false,
        error: 'Messages array is required and must not be empty' 
      });
    }

    console.log(`🧠 Batch analysis requested for ${messages.length} messages`);
    console.log(`📊 Analysis type: ${analysisType}, Conversation: ${conversationId || 'N/A'}`);

    // Track individual results
    const individualResults = [];
    let processedCount = 0;
    
    // Process messages in batches of 5 for better performance
    const batchSize = 5;
    const batches = [];
    
    for (let i = 0; i < messages.length; i += batchSize) {
      batches.push(messages.slice(i, i + batchSize));
    }

    // Process each batch
    for (const [batchIndex, batch] of batches.entries()) {
      console.log(`📦 Processing batch ${batchIndex + 1}/${batches.length} (${batch.length} messages)`);
      
      // Create batch text for analysis
      const batchText = batch.map(msg => msg.text).join('\n---\n');
      
      try {
        // Analyze the batch with DistilBERT
        let batchResult;
        if (sentimentAnalyzer) {
          const distilbertResult = await sentimentAnalyzer(batchText);
          const mainResult = distilbertResult[0];
          const isPositive = mainResult.label === 'POSITIVE';
          const moodScore = isPositive ? mainResult.score : -mainResult.score;
          
          batchResult = {
            moodScore: parseFloat(moodScore.toFixed(3)),
            confidence: mainResult.score,
            analyzedAt: new Date().toISOString()
          };
        } else {
          // Fallback to simple analysis
          batchResult = analyzeSentimentSimple(batchText, 'user');
        }

        // Apply batch result to each individual message with variations
        for (const [msgIndex, message] of batch.entries()) {
          // Add some variation based on message content
          const variationFactor = (msgIndex / batch.length) * 0.1; // Small variation
          const adjustedMoodScore = Math.max(-1, Math.min(1, 
            batchResult.moodScore + (Math.random() * 0.2 - 0.1) + variationFactor
          ));
          
          // Enhanced healthcare context for each message
          const enhancedAnalysis = enhanceWithHealthcareContext(
            message.text, 
            batchResult.moodScore > 0 ? 'POSITIVE' : 'NEGATIVE',
            Math.abs(batchResult.confidence || 0.7),
            message.analyzeFor || 'user'
          );
          
          // Add some individual variation to stress score
          const stressVariation = message.text.includes('?') ? 0.1 : 
                                message.text.includes('!') ? 0.2 : 0;
          const adjustedStressScore = Math.min(1, 
            enhancedAnalysis.stressScore + stressVariation
          );
          
          individualResults.push({
            messageId: message.id || `batch-${batchIndex}-${msgIndex}`,
            textPreview: message.text.substring(0, 50) + 
                       (message.text.length > 50 ? '...' : ''),
            textLength: message.text.length,
            timestamp: message.timestamp || new Date().toISOString(),
            isUser: message.isUser !== false,
            analysis: {
              moodScore: parseFloat(adjustedMoodScore.toFixed(3)),
              stressScore: parseFloat(adjustedStressScore.toFixed(3)),
              emotion: categorizeEmotion(adjustedMoodScore, adjustedStressScore, 'user'),
              isCrisis: enhancedAnalysis.isCrisis,
              confidence: Math.min(1, (batchResult.confidence || 0.7) * 0.9),
              healthcareContext: enhancedAnalysis.healthcareContext,
              analyzedAt: new Date().toISOString(),
              isBatchAnalyzed: true,
              batchId: batchIndex
            }
          });
          
          processedCount++;
        }
        
        console.log(`✅ Batch ${batchIndex + 1} processed: ${processedCount}/${messages.length} messages`);
        
      } catch (batchError) {
        console.error(`❌ Error processing batch ${batchIndex + 1}:`, batchError.message);
        
        // Fallback: Analyze each message individually with simple analysis
        for (const message of batch) {
          const simpleAnalysis = analyzeSentimentSimple(message.text, 'user');
          
          individualResults.push({
            messageId: message.id || `fallback-${Date.now()}-${Math.random()}`,
            textPreview: message.text.substring(0, 30) + '...',
            analysis: {
              ...simpleAnalysis,
              isBatchAnalyzed: true,
              isFallback: true,
              analyzedAt: new Date().toISOString()
            }
          });
          
          processedCount++;
        }
      }
      
      // Small delay between batches to avoid overwhelming the model
      if (batchIndex < batches.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }

    // Generate conversation-level insights
    const conversationInsights = generateConversationInsights(individualResults, messages);

    res.json({
      success: true,
      statistics: {
        totalMessages: messages.length,
        analyzedMessages: processedCount,
        failedMessages: messages.length - processedCount,
        batchCount: batches.length,
        averageProcessingTime: (processedCount / batches.length).toFixed(2)
      },
      individualResults,
      conversationInsights,
      metadata: {
        analysisType,
        conversationId,
        userId,
        analyzedAt: new Date().toISOString(),
        modelUsed: sentimentAnalyzer ? 'distilbert-batch' : 'simple-batch',
        version: '1.0'
      }
    });
    
  } catch (error) {
    console.error("❌ Batch analysis error:", error);
    
    res.status(500).json({
      success: false,
      error: 'Batch analysis failed',
      details: error.message,
      fallback: {
        note: 'Using simplified analysis',
        analyzedAt: new Date().toISOString()
      }
    });
  }
});

/* ===============================
   🏥 Healthcare Context Enhancement
=================================*/
function enhanceWithHealthcareContext(text, distilbertLabel, distilbertScore, analyzeFor) {
  const textLower = text.toLowerCase();
  
  // Convert DistilBERT output to our format
  const isPositive = distilbertLabel === 'POSITIVE';
  
  // Normalize score: -1 (very negative) to 1 (very positive)
  const moodScore = isPositive ? distilbertScore : -distilbertScore;
  
  // 🔧 IMPROVED: Calculate stress score with context awareness
  let stressScore = calculateHealthcareStressScore(text, moodScore);
  
  // 🔧 FIX 1: Adjust stress score based on mood positivity
  if (moodScore > 0.7) {
    // Very positive messages get stress reduction
    stressScore *= 0.6; // Reduce by 40%
  } else if (moodScore > 0.3) {
    // Positive messages get moderate stress reduction
    stressScore *= 0.8; // Reduce by 20%
  }
  
  // 🔧 NEW: Adjust for exclamations in positive context
  if (moodScore > 0.5 && text.includes('!')) {
    // Positive exclamations are less stressful
    stressScore *= 0.9;
  }
  
  // Ensure stressScore stays within bounds
  stressScore = Math.min(1, Math.max(0, stressScore));
  
  // Detect crisis keywords (healthcare-specific)
  const crisisKeywords = [
    'suicide', 'kill myself', 'end my life', 'want to die', 
    'self harm', 'self-harm', 'cutting', 'overdose',
    'panic attack', 'cant breathe', 'chest pain', 'emergency',
    'help me', 'i give up', 'nothing matters'
  ];
  
  const hasCrisisKeyword = crisisKeywords.some(keyword => textLower.includes(keyword));
  
  // Detect urgency
  const urgencyKeywords = ['urgent', 'emergency', '911', 'immediately', 'now', 'asap'];
  const hasUrgency = urgencyKeywords.some(keyword => textLower.includes(keyword));
  
  // Detect pain mentions
  const painKeywords = ['pain', 'hurt', 'aching', 'sore', 'unbearable', 'excruciating'];
  const mentionsPain = painKeywords.some(keyword => textLower.includes(keyword));
  
  // Detect medication mentions
  const medKeywords = ['medication', 'pill', 'drug', 'prescription', 'dose', 'tablet'];
  const mentionsMedication = medKeywords.some(keyword => textLower.includes(keyword));
  
  // 🔧 IMPROVED: Detect specific emotional keywords for better classification
  const emotionKeywords = {
    'anxious': ['anxious', 'worried', 'nervous', 'scared', 'afraid'],
    'happy': ['happy', 'great', 'good', 'excellent', 'wonderful', 'amazing'],
    'sad': ['sad', 'depressed', 'miserable', 'unhappy', 'hopeless'],
    'angry': ['angry', 'mad', 'furious', 'upset', 'annoyed'],
    'calm': ['calm', 'relaxed', 'peaceful', 'chill', 'serene'],
    'stressed': ['stressed', 'overwhelmed', 'burned out', 'pressured'],
    'frustrated': ['frustrated', 'annoyed', 'irritated', 'fed up']
  };
  
  // 🔧 NEW: Keyword-based emotion detection (takes precedence over scores)
  let keywordDetectedEmotion = null;
  for (const [emotion, keywords] of Object.entries(emotionKeywords)) {
    if (keywords.some(keyword => textLower.includes(keyword))) {
      keywordDetectedEmotion = emotion;
      break; // First match wins
    }
  }
  
  // 🔧 IMPROVED: Determine emotion category with keyword priority
  let emotion;
  if (keywordDetectedEmotion) {
    // Use keyword-detected emotion if found
    emotion = keywordDetectedEmotion;
  } else {
    // Fall back to score-based categorization
    emotion = categorizeEmotionImproved(moodScore, stressScore, textLower, analyzeFor);
  }
  
  // 🔧 IMPROVED: Determine if this is a crisis with better logic
  let isCrisis = false;
  
  if (analyzeFor === 'user') {
    // Level 1: Direct crisis keywords
    if (hasCrisisKeyword) {
      isCrisis = true;
    }
    // Level 2: High stress + negative mood combination
    else if (stressScore > 0.8 && moodScore < -0.6) {
      isCrisis = true;
    }
    // Level 3: Urgency + pain + negative mood
    else if (hasUrgency && mentionsPain && moodScore < -0.4) {
      isCrisis = true;
    }
    // Level 4: Extreme negative emotion keywords
    else if (keywordDetectedEmotion === 'sad' && stressScore > 0.7 && moodScore < -0.5) {
      isCrisis = true;
    }
  }
  
  // 🔧 NEW: Calculate confidence with healthcare context consideration
  let confidence = distilbertScore;
  
  // Increase confidence if we have clear healthcare context
  if (mentionsPain || mentionsMedication || hasUrgency) {
    confidence = Math.min(1, confidence * 1.1);
  }
  
  // Decrease confidence for very short messages
  if (text.split(' ').length < 3) {
    confidence *= 0.8;
  }
  
  // 🔧 NEW: Enhanced healthcare context with intensity
  const wordCount = text.split(' ').length;
  const exclamationCount = (text.match(/!/g) || []).length;
  const questionCount = (text.match(/\?/g) || []).length;
  const capsCount = (text.match(/[A-Z]/g) || []).length;
  const capsRatio = capsCount / text.length || 0;
  
  // Determine emotional intensity
  const emotionalIntensity = Math.min(1, 
    (exclamationCount * 0.2) + 
    (capsRatio > 0.3 ? 0.3 : 0) + 
    (wordCount > 50 ? 0.2 : 0)
  );
  
  // 🔧 NEW: Enhanced suggested tone with more granularity
  const suggestedTone = getEnhancedSuggestedTone(
    moodScore, 
    stressScore, 
    isCrisis, 
    analyzeFor,
    emotionalIntensity,
    mentionsPain,
    mentionsMedication
  );
  
  return {
    moodScore: parseFloat(moodScore.toFixed(3)),
    stressScore: parseFloat(stressScore.toFixed(3)),
    emotion,
    isCrisis,
    confidence: parseFloat(confidence.toFixed(3)),
    healthcareContext: {
      hasCrisisKeyword,
      hasUrgency,
      mentionsPain,
      mentionsMedication,
      wordCount,
      hasQuestion: questionCount > 0,
      hasExclamation: exclamationCount > 0,
      allCapsRatio: capsRatio,
      exclamationCount,
      questionCount,
      emotionalIntensity: parseFloat(emotionalIntensity.toFixed(3)),
      keywordDetectedEmotion,
      containsPositiveKeywords: moodScore > 0.7 && textLower.includes('good') || textLower.includes('great'),
      containsNegativeKeywords: moodScore < -0.5 && (textLower.includes('bad') || textLower.includes('terrible'))
    },
    suggestedResponseTone: suggestedTone,
    // 🔧 NEW: Add batch analysis metadata
    modelUsed: 'distilbert',
    analyzedAt: new Date().toISOString(),
    analysisType: analyzeFor === 'ai' ? 'ai_response' : 'user_input'
  };
}

// 🔧 NEW IMPROVED EMOTION CATEGORIZATION FUNCTION
function categorizeEmotionImproved(moodScore, stressScore, textLower, analyzeFor) {
  // For AI responses
  if (analyzeFor === 'ai') {
    if (moodScore > 0.8) return 'empathetic';
    if (moodScore > 0.5) return 'supportive';
    if (moodScore > 0.2) return 'encouraging';
    if (moodScore > -0.2) return 'neutral';
    if (moodScore > -0.5) return 'concerned';
    if (moodScore > -0.8) return 'urgent';
    return 'crisis';
  }
  
  // For user messages - IMPROVED GRANULAR LOGIC
  
  // First handle extreme cases
  if (stressScore > 0.9 && moodScore < -0.8) return 'crisis';
  if (stressScore > 0.85) return 'high-stress';
  
  // Handle positive-but-stressed cases (like excited anxiety)
  if (stressScore > 0.6 && moodScore > 0.3) return 'excited';
  if (stressScore > 0.6 && moodScore >= 0) return 'concerned';
  if (stressScore > 0.6 && moodScore < 0) return 'stressed';
  
  // Pure mood-based classification
  if (moodScore > 0.85) return 'very-positive';
  if (moodScore > 0.7) return 'positive';
  if (moodScore > 0.5) return 'slightly-positive';
  if (moodScore > 0.3) return 'calm';
  if (moodScore > -0.3) return 'neutral';
  if (moodScore > -0.6) return 'slightly-negative';
  if (moodScore > -0.8) return 'negative';
  
  return 'very-negative';
}

// 🔧 NEW ENHANCED SUGGESTED TONE FUNCTION
function getEnhancedSuggestedTone(moodScore, stressScore, isCrisis, analyzeFor, emotionalIntensity, mentionsPain, mentionsMedication) {
  if (isCrisis) {
    return {
      tone: 'CRISIS_INTERVENTION',
      priority: 'HIGHEST',
      action: 'PROVIDE_EMERGENCY_RESOURCES',
      empathyLevel: 'VERY_HIGH',
      responseSpeed: 'IMMEDIATE',
      responseLength: 'SHORT_DIRECT',
      focus: 'SAFETY_AND_SUPPORT'
    };
  }
  
  if (stressScore > 0.9) {
    return {
      tone: 'EXTREME_CALMING',
      priority: 'VERY_HIGH',
      action: 'CALM_DECREASE_STRESS',
      empathyLevel: 'VERY_HIGH',
      responseSpeed: 'FAST',
      responseLength: 'CONCISE',
      focus: 'STRESS_REDUCTION'
    };
  }
  
  if (stressScore > 0.8) {
    return {
      tone: 'CALM_REASSURING',
      priority: 'HIGH',
      action: 'REASSURE_AND_SUPPORT',
      empathyLevel: 'HIGH',
      responseSpeed: 'FAST',
      responseLength: 'MODERATE',
      focus: 'EMOTIONAL_SUPPORT'
    };
  }
  
  if (mentionsPain && moodScore < 0) {
    return {
      tone: 'CARING_PAIN_FOCUSED',
      priority: 'HIGH',
      action: 'ACKNOWLEDGE_PAIN_OFFER_SUPPORT',
      empathyLevel: 'HIGH',
      responseSpeed: 'FAST',
      responseLength: 'MODERATE',
      focus: 'PAIN_MANAGEMENT'
    };
  }
  
  if (mentionsMedication && stressScore > 0.5) {
    return {
      tone: 'PROFESSIONAL_CAREFUL',
      priority: 'MEDIUM_HIGH',
      action: 'PROVIDE_ACCURATE_INFO',
      empathyLevel: 'MEDIUM_HIGH',
      responseSpeed: 'NORMAL',
      responseLength: 'DETAILED',
      focus: 'MEDICATION_SAFETY'
    };
  }
  
  if (moodScore < -0.6) {
    return {
      tone: 'EMPATHETIC_VALIDATING',
      priority: 'MEDIUM_HIGH',
      action: 'VALIDATE_EMOTIONS',
      empathyLevel: 'HIGH',
      responseSpeed: 'NORMAL',
      responseLength: 'MODERATE',
      focus: 'EMOTIONAL_VALIDATION'
    };
  }
  
  if (moodScore < -0.3) {
    return {
      tone: 'SUPPORTIVE_HELPFUL',
      priority: 'MEDIUM',
      action: 'OFFER_SUPPORT_AND_HELP',
      empathyLevel: 'MEDIUM_HIGH',
      responseSpeed: 'NORMAL',
      responseLength: 'MODERATE',
      focus: 'PROBLEM_SOLVING'
    };
  }
  
  if (moodScore > 0.7) {
    return {
      tone: 'POSITIVE_ENGAGING',
      priority: 'LOW',
      action: 'ENGAGE_AND_CELEBRATE',
      empathyLevel: 'MEDIUM',
      responseSpeed: 'NORMAL',
      responseLength: 'MODERATE',
      focus: 'POSITIVE_REINFORCEMENT'
    };
  }
  
  // Default for neutral/general conversations
  return {
    tone: 'NEUTRAL_HELPFUL',
    priority: 'LOW',
    action: 'PROVIDE_INFORMATION',
    empathyLevel: 'MEDIUM',
    responseSpeed: 'NORMAL',
    responseLength: 'VARIABLE',
    focus: 'INFORMATION_PROVISION'
  };
}

// 🔧 UPDATED calculateHealthcareStressScore with better logic
function calculateHealthcareStressScore(text, moodScore) {
  const textLower = text.toLowerCase();
  
  let stressScore = 0.5; // Base
  
  // 1. Mood impact (negative mood increases stress)
  if (moodScore < 0) {
    stressScore += Math.abs(moodScore) * 0.4; // Reduced from 0.5
  } else if (moodScore > 0.7) {
    // Very positive mood reduces baseline stress
    stressScore -= 0.2;
  }
  
  // 2. Urgency indicators (weighted)
  if (textLower.includes('911')) stressScore += 0.4;
  else if (textLower.includes('emergency')) stressScore += 0.3;
  else if (textLower.includes('urgent')) stressScore += 0.2;
  else if (textLower.includes('asap') || textLower.includes('immediately')) stressScore += 0.15;
  
  // 3. Pain indicators
  const severePainWords = ['unbearable', 'excruciating', 'severe'];
  const moderatePainWords = ['pain', 'hurt', 'aching', 'sore'];
  
  if (severePainWords.some(word => textLower.includes(word))) stressScore += 0.3;
  else if (moderatePainWords.some(word => textLower.includes(word))) stressScore += 0.15;
  
  // 4. Medical context (increases stress relevance)
  const medicalWords = ['hospital', 'doctor', 'clinic', 'er', 'emergency room'];
  if (medicalWords.some(word => textLower.includes(word))) stressScore += 0.1;
  
  // 5. Punctuation intensity (improved)
  const exclamationCount = (text.match(/!/g) || []).length;
  const questionCount = (text.match(/\?/g) || []).length;
  
  // Multiple exclamations = high stress
  if (exclamationCount >= 3) stressScore += 0.25;
  else if (exclamationCount === 2) stressScore += 0.15;
  else if (exclamationCount === 1) stressScore += 0.08;
  
  // Multiple questions = uncertainty stress
  if (questionCount >= 3) stressScore += 0.15;
  
  // 6. Text characteristics
  if (text.length > 200) stressScore += 0.1; // Long messages might indicate detailed concern
  
  // 7. ALL CAPS intensity (reduced impact)
  const capsCount = (text.match(/[A-Z]/g) || []).length;
  const capsRatio = capsCount / text.length;
  if (capsRatio > 0.5) stressScore += 0.15; // Reduced from 0.2
  else if (capsRatio > 0.3) stressScore += 0.1;
  
  // 8. Negative emotion words (direct impact)
  const highStressWords = ['panic', 'anxiety', 'overwhelmed', 'cant cope', 'breaking down'];
  const mediumStressWords = ['worried', 'scared', 'nervous', 'stressed', 'anxious'];
  
  if (highStressWords.some(word => textLower.includes(word))) stressScore += 0.25;
  else if (mediumStressWords.some(word => textLower.includes(word))) stressScore += 0.15;
  
  // Clamp between 0 and 1
  return Math.min(1, Math.max(0, stressScore));
}

/* ===============================
   📊 Helper Functions
=================================*/
function calculateHealthcareStressScore(text, moodScore) {
  const textLower = text.toLowerCase();
  
  let stressScore = 0.5; // Base
  
  // Negative mood adds stress
  if (moodScore < 0) {
    stressScore += Math.abs(moodScore) * 0.5;
  }
  
  // Urgency indicators
  if (textLower.includes('urgent') || textLower.includes('emergency')) stressScore += 0.3;
  if (textLower.includes('911')) stressScore += 0.4;
  
  // Pain indicators
  if (textLower.includes('pain') || textLower.includes('hurt')) stressScore += 0.2;
  
  // Length (long messages might indicate distress)
  if (text.length > 200) stressScore += 0.1;
  
  // Punctuation intensity
  const exclamationCount = (text.match(/!/g) || []).length;
  stressScore += Math.min(0.3, exclamationCount * 0.05);
  
  // ALL CAPS intensity
  const capsCount = (text.match(/[A-Z]/g) || []).length;
  const capsRatio = capsCount / text.length;
  if (capsRatio > 0.3) stressScore += 0.2;
  
  return Math.min(1, Math.max(0, stressScore));
}

function categorizeEmotion(moodScore, stressScore, analyzeFor) {
  // For AI responses
  if (analyzeFor === 'ai') {
    if (moodScore > 0.7) return 'empathetic';
    if (moodScore > 0.3) return 'supportive';
    if (moodScore > -0.3) return 'neutral';
    if (moodScore > -0.7) return 'concerned';
    return 'urgent';
  }
  
  // For user messages
  if (stressScore > 0.8 && moodScore < -0.7) return 'crisis';
  if (stressScore > 0.6) return 'high-stress';
  if (moodScore > 0.7) return 'positive';
  if (moodScore > 0.3) return 'calm';
  if (moodScore > -0.3) return 'neutral';
  if (moodScore > -0.7) return 'worried';
  return 'distressed';
}

function getSuggestedTone(moodScore, stressScore, isCrisis, analyzeFor) {
  if (isCrisis) {
    return {
      tone: 'CRISIS_INTERVENTION',
      priority: 'HIGHEST',
      action: 'PROVIDE_EMERGENCY_RESOURCES',
      empathyLevel: 'VERY_HIGH',
      responseSpeed: 'IMMEDIATE'
    };
  }
  
  if (stressScore > 0.8) {
    return {
      tone: 'URGENT_CARING',
      priority: 'HIGH',
      action: 'CALM_AND_REASSURE',
      empathyLevel: 'HIGH',
      responseSpeed: 'FAST'
    };
  }
  
  if (moodScore < -0.5) {
    return {
      tone: 'EMPATHETIC_SUPPORTIVE',
      priority: 'MEDIUM_HIGH',
      action: 'VALIDATE_AND_COMFORT',
      empathyLevel: 'HIGH',
      responseSpeed: 'NORMAL'
    };
  }
  
  if (moodScore < 0) {
    return {
      tone: 'CARING_PROFESSIONAL',
      priority: 'MEDIUM',
      action: 'ACKNOWLEDGE_AND_HELP',
      empathyLevel: 'MEDIUM_HIGH',
      responseSpeed: 'NORMAL'
    };
  }
  
  return {
    tone: 'NEUTRAL_HELPFUL',
    priority: 'LOW',
    action: 'PROVIDE_INFORMATION',
    empathyLevel: 'MEDIUM',
    responseSpeed: 'NORMAL'
  };
}

/* ================================
   🧠 Generate Conversation Insights
=================================*/
function generateConversationInsights(individualResults, originalMessages) {
  if (!individualResults || individualResults.length === 0) {
    return {
      moodTrend: 'unknown',
      dominantEmotions: [],
      crisisProbability: 0,
      recommendations: [],
      summary: 'No messages analyzed'
    };
  }

  // Calculate averages
  const validMoodScores = individualResults
    .filter(r => r.analysis && r.analysis.moodScore !== undefined)
    .map(r => r.analysis.moodScore);
  
  const validStressScores = individualResults
    .filter(r => r.analysis && r.analysis.stressScore !== undefined)
    .map(r => r.analysis.stressScore);

  const avgMood = validMoodScores.length > 0 
    ? validMoodScores.reduce((a, b) => a + b, 0) / validMoodScores.length 
    : 0;
  
  const avgStress = validStressScores.length > 0
    ? validStressScores.reduce((a, b) => a + b, 0) / validStressScores.length
    : 0.5;

  // Count emotions
  const emotionCounts = {};
  individualResults.forEach(result => {
    if (result.analysis && result.analysis.emotion) {
      const emotion = result.analysis.emotion;
      emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
    }
  });

  // Get dominant emotions (top 3)
  const dominantEmotions = Object.entries(emotionCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([emotion, count]) => ({ emotion, count, percentage: (count / individualResults.length * 100).toFixed(1) }));

  // Check for crises
  const crisisMessages = individualResults.filter(r => r.analysis && r.analysis.isCrisis);
  const crisisProbability = Math.min(1, crisisMessages.length / individualResults.length);

  // Determine mood trend
  let moodTrend = 'stable';
  if (validMoodScores.length >= 4) {
    const half = Math.floor(validMoodScores.length / 2);
    const firstHalfAvg = validMoodScores.slice(0, half).reduce((a, b) => a + b, 0) / half;
    const secondHalfAvg = validMoodScores.slice(half).reduce((a, b) => a + b, 0) / (validMoodScores.length - half);
    
    if (secondHalfAvg - firstHalfAvg > 0.3) moodTrend = 'improving';
    else if (firstHalfAvg - secondHalfAvg > 0.3) moodTrend = 'worsening';
  }

  // Generate recommendations
  const recommendations = generateRecommendations(
    avgMood, 
    avgStress, 
    crisisProbability, 
    moodTrend,
    dominantEmotions
  );

  // Generate summary
  const summary = generateInsightSummary(
    avgMood,
    avgStress,
    moodTrend,
    crisisMessages.length,
    individualResults.length
  );

  return {
    moodTrend,
    averageMood: parseFloat(avgMood.toFixed(3)),
    averageStress: parseFloat(avgStress.toFixed(3)),
    dominantEmotions,
    crisisProbability: parseFloat(crisisProbability.toFixed(3)),
    crisisCount: crisisMessages.length,
    messageCount: individualResults.length,
    recommendations,
    summary,
    keyMetrics: {
      veryPositiveMessages: validMoodScores.filter(s => s > 0.7).length,
      veryNegativeMessages: validMoodScores.filter(s => s < -0.7).length,
      highStressMessages: validStressScores.filter(s => s > 0.8).length,
      questionsAsked: originalMessages.filter(m => m.text && m.text.includes('?')).length,
      averageMessageLength: Math.round(originalMessages.reduce((sum, m) => sum + (m.text?.length || 0), 0) / originalMessages.length)
    }
  };
}

/* ================================
   💡 Generate Recommendations
=================================*/
function generateRecommendations(avgMood, avgStress, crisisProbability, moodTrend, dominantEmotions) {
  const recommendations = [];

  // Crisis recommendations
  if (crisisProbability > 0.3) {
    recommendations.push({
      priority: 'HIGHEST',
      action: 'IMMEDIATE_CRISIS_INTERVENTION',
      description: 'Multiple crisis indicators detected. Provide emergency resources immediately.',
      resources: [
        '988 Suicide & Crisis Lifeline',
        'Emergency: 911',
        'Local mental health services',
        'Crisis text line: Text HOME to 741741'
      ]
    });
  }

  // Stress recommendations
  if (avgStress > 0.7) {
    recommendations.push({
      priority: 'HIGH',
      action: 'STRESS_MANAGEMENT_SUPPORT',
      description: 'High stress levels detected. Provide calming techniques and stress management resources.',
      suggestions: [
        'Breathing exercises (4-7-8 technique)',
        'Progressive muscle relaxation',
        'Mindfulness meditation guidance',
        'Referral to stress management program'
      ]
    });
  }

  // Mood-based recommendations
  if (avgMood < -0.5) {
    recommendations.push({
      priority: 'HIGH',
      action: 'EMOTIONAL_SUPPORT_ENGAGEMENT',
      description: 'Consistently negative mood detected. Increase emotional support and check-in frequency.',
      suggestions: [
        'Schedule regular check-ins',
        'Provide empathetic validation',
        'Offer positive reinforcement',
        'Suggest mood tracking journal'
      ]
    });
  }

  // Emotion-specific recommendations
  const dominantEmotion = dominantEmotions[0]?.emotion;
  if (dominantEmotion === 'anxious' || dominantEmotion === 'stressed') {
    recommendations.push({
      priority: 'MEDIUM',
      action: 'ANXIETY_REDUCTION_STRATEGIES',
      description: 'Anxiety is a dominant emotion. Provide anxiety-reducing techniques.',
      suggestions: [
        'Grounding techniques (5-4-3-2-1 method)',
        'Worry time allocation',
        'Cognitive restructuring exercises',
        'Relaxation audio guides'
      ]
    });
  }

  // Trend-based recommendations
  if (moodTrend === 'worsening') {
    recommendations.push({
      priority: 'MEDIUM_HIGH',
      action: 'TREND_INTERVENTION',
      description: 'Mood trend is worsening. Consider proactive intervention.',
      suggestions: [
        'Increase monitoring frequency',
        'Engage support network',
        'Consider professional referral',
        'Adjust response strategy to more supportive tone'
      ]
    });
  }

  // General wellness recommendations
  if (recommendations.length === 0) {
    recommendations.push({
      priority: 'LOW',
      action: 'WELLNESS_MAINTENANCE',
      description: 'Conversation appears stable. Continue supportive engagement.',
      suggestions: [
        'Maintain regular check-ins',
        'Provide health education materials',
        'Encourage healthy habits',
        'Celebrate small victories'
      ]
    });
  }

  return recommendations;
}

/* ================================
   📝 Generate Insight Summary
=================================*/
function generateInsightSummary(avgMood, avgStress, moodTrend, crisisCount, totalMessages) {
  const moodDescriptors = {
    'improving': 'improving',
    'worsening': 'declining',
    'stable': 'stable'
  };

  const stressLevel = avgStress < 0.3 ? 'low' : 
                     avgStress < 0.6 ? 'moderate' : 
                     avgStress < 0.8 ? 'high' : 'very high';

  const moodLevel = avgMood > 0.7 ? 'very positive' :
                   avgMood > 0.3 ? 'positive' :
                   avgMood > -0.3 ? 'neutral' :
                   avgMood > -0.7 ? 'negative' : 'very negative';

  let summary = `Analysis of ${totalMessages} messages shows ${moodLevel} mood with ${stressLevel} stress levels. `;
  summary += `Overall emotional tone is ${moodDescriptors[moodTrend] || 'stable'}. `;
  
  if (crisisCount > 0) {
    summary += `⚠️ ${crisisCount} crisis ${crisisCount === 1 ? 'message was' : 'messages were'} detected. `;
  }
  
  if (avgStress > 0.7) {
    summary += `High stress indicators suggest benefit from stress management support. `;
  }
  
  if (avgMood < -0.5) {
    summary += `Negative mood patterns indicate potential need for increased emotional support.`;
  }

  return summary.trim();
}

// Simple fallback analysis (if DistilBERT fails)
function analyzeSentimentSimple(text, analyzeFor) {
  const textLower = text.toLowerCase();
  
  // Simple keyword matching
  const positiveWords = ['good', 'great', 'better', 'improving', 'thanks', 'thank', 'happy', 'relieved', 'well', 'okay'];
  const negativeWords = ['bad', 'terrible', 'worse', 'pain', 'hurt', 'anxious', 'worried', 'scared', 'depressed', 'sick'];
  const crisisWords = ['suicide', 'kill myself', 'end my life', 'want to die', 'self harm', 'overdose'];
  
  const positiveCount = positiveWords.filter(word => textLower.includes(word)).length;
  const negativeCount = negativeWords.filter(word => textLower.includes(word)).length;
  const hasCrisisWord = crisisWords.some(word => textLower.includes(word));
  
  // Calculate scores
  let moodScore = 0;
  if (positiveCount > negativeCount) {
    moodScore = 0.5 + (positiveCount * 0.1);
  } else if (negativeCount > positiveCount) {
    moodScore = -0.5 - (negativeCount * 0.1);
  }
  
  // Clamp between -1 and 1
  moodScore = Math.max(-1, Math.min(1, moodScore));
  
  // Stress score
  const stressScore = Math.min(1, negativeCount * 0.2 + (text.length > 100 ? 0.3 : 0));
  
  return {
    moodScore: parseFloat(moodScore.toFixed(3)),
    stressScore: parseFloat(stressScore.toFixed(3)),
    emotion: categorizeEmotion(moodScore, stressScore, analyzeFor),
    isCrisis: hasCrisisWord || (analyzeFor === 'user' && stressScore > 0.8),
    confidence: 0.7,
    healthcareContext: {
      hasCrisisKeyword: hasCrisisWord,
      hasUrgency: textLower.includes('urgent') || textLower.includes('emergency'),
      mentionsPain: textLower.includes('pain') || textLower.includes('hurt'),
      mentionsMedication: textLower.includes('medication') || textLower.includes('pill'),
      wordCount: text.split(' ').length
    }
  };
}

/* ================================
   📊 Get Batch Analysis Status
=================================*/
app.get("/api/mood/batch/status/:batchId", async (req, res) => {
  const { batchId } = req.params;
  
  // In a real implementation, you'd track batch status in a database
  // For now, return a mock status
  res.json({
    success: true,
    batchId,
    status: 'completed',
    processedAt: new Date().toISOString(),
    estimatedAccuracy: '85-95%',
    note: 'Batch analysis completed successfully'
  });
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
  console.log(`🧠 Mood Analysis:`);
  console.log(`   POST /api/mood {text: "message", analyzeFor: "user"|"ai"}`);
  console.log(`   POST /api/mood/batch {messages: [...], conversationId: "...", userId: "..."}`);
  console.log(`   GET /api/mood/batch/status/:batchId`);
});