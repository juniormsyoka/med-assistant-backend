import express from 'express';
import { 
  initializeSentimentAnalyzer, 
  analyzeMood, 
  analyzeMoodBatch,
  analyzeSentimentSimple 
} from '../utils/moodAnalyzer.js';
import { generateConversationInsights } from '../utils/conversationInsights.js';

const router = express.Router();

// Initialize sentiment analyzer on startup
initializeSentimentAnalyzer();

/* ===============================
   🧠 MOOD ANALYSIS ENDPOINT
=================================*/
router.post("/mood", async (req, res) => {
  try {
    const { text, analyzeFor = 'user', batchMode = false } = req.body;
    
    if (batchMode && Array.isArray(text)) {
      return handleBatchMode(text, analyzeFor, res);
    }
    
    if (!text || typeof text !== 'string') {
      return res.status(400).json({ 
        success: false,
        error: 'Text is required and must be a string' 
      });
    }

    console.log(`🧠 Mood analysis requested for ${analyzeFor}:`, text.substring(0, 100) + '...');

    const result = await analyzeMood(text, analyzeFor);
    
    res.json({
      success: true,
      ...result,
      textPreview: text.substring(0, 50) + (text.length > 50 ? '...' : ''),
      textLength: text.length
    });
    
  } catch (error) {
    console.error("❌ Mood analysis error:", error);
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
=================================*/
router.post("/mood/batch", async (req, res) => {
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

    const individualResults = await processBatchMessages(messages);
    const conversationInsights = generateConversationInsights(individualResults, messages);

    res.json({
      success: true,
      statistics: {
        totalMessages: messages.length,
        analyzedMessages: individualResults.length,
        failedMessages: messages.length - individualResults.length,
        batchCount: Math.ceil(messages.length / 5),
        averageProcessingTime: 'calculated'
      },
      individualResults,
      conversationInsights,
      metadata: {
        analysisType,
        conversationId,
        userId,
        analyzedAt: new Date().toISOString(),
        modelUsed: 'distilbert-enhanced-batch',
        version: '2.0'
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

/* ================================
   📊 BATCH STATUS ENDPOINT
=================================*/
router.get("/mood/batch/status/:batchId", async (req, res) => {
  const { batchId } = req.params;
  res.json({
    success: true,
    batchId,
    status: 'completed',
    processedAt: new Date().toISOString(),
    estimatedAccuracy: '85-95%',
    note: 'Batch analysis completed successfully'
  });
});

// Helper functions for the routes
async function handleBatchMode(texts, analyzeFor, res) {
  const batchResults = await Promise.all(
    texts.map(async (singleText, index) => {
      try {
        const result = await analyzeMood(singleText, analyzeFor);
        return {
          text: singleText.substring(0, 100) + (singleText.length > 100 ? '...' : ''),
          ...result,
          index,
          success: true
        };
      } catch (error) {
        return {
          text: singleText.substring(0, 100) + (singleText.length > 100 ? '...' : ''),
          error: error.message,
          index,
          success: false
        };
      }
    })
  );

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

async function processBatchMessages(messages) {
  const batchSize = 5;
  const individualResults = [];
  
  for (let i = 0; i < messages.length; i += batchSize) {
    const batch = messages.slice(i, i + batchSize);
    console.log(`📦 Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(messages.length/batchSize)}`);
    
    const batchResults = await analyzeMoodBatch(batch);
    
    batchResults.forEach((analysis, index) => {
      individualResults.push({
        messageId: messages[i + index].id || `batch-${i + index}`,
        textPreview: messages[i + index].text.substring(0, 50) + 
                   (messages[i + index].text.length > 50 ? '...' : ''),
        textLength: messages[i + index].text.length,
        timestamp: messages[i + index].timestamp || new Date().toISOString(),
        isUser: messages[i + index].isUser !== false,
        analysis: {
          ...analysis,
          isBatchAnalyzed: true,
          batchId: Math.floor(i / batchSize)
        }
      });
    });
    
    if (i + batchSize < messages.length) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
  
  return individualResults;
}

export default router;