import { pipeline } from '@xenova/transformers';
import { enhanceWithHealthcareContext } from './healthcareContext.js';

let sentimentAnalyzer = null;

export async function initializeSentimentAnalyzer() {
  try {
    console.log("🧠 Loading DistilBERT sentiment model...");
    sentimentAnalyzer = await pipeline('sentiment-analysis', 
      'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
    console.log("✅ DistilBERT loaded successfully!");
    return true;
  } catch (error) {
    console.error("❌ Failed to load DistilBERT:", error.message);
    sentimentAnalyzer = null;
    return false;
  }
}

export async function analyzeMood(text, analyzeFor = 'user') {
  if (!sentimentAnalyzer) {
    return analyzeSentimentSimple(text, analyzeFor);
  }

  try {
    const distilbertResult = await sentimentAnalyzer(text);
    const mainResult = distilbertResult[0];
    return enhanceWithHealthcareContext(text, mainResult.label, mainResult.score, analyzeFor);
  } catch (error) {
    console.error("DistilBERT analysis failed:", error.message);
    return analyzeSentimentSimple(text, analyzeFor);
  }
}

export async function analyzeMoodBatch(messages) {
  if (!sentimentAnalyzer) {
    return messages.map(msg => analyzeSentimentSimple(msg.text, msg.analyzeFor || 'user'));
  }

  const results = [];
  for (const message of messages) {
    try {
      const distilbertResult = await sentimentAnalyzer(message.text);
      const mainResult = distilbertResult[0];
      results.push(enhanceWithHealthcareContext(
        message.text, 
        mainResult.label, 
        mainResult.score, 
        message.analyzeFor || 'user'
      ));
    } catch (error) {
      results.push(analyzeSentimentSimple(message.text, message.analyzeFor || 'user'));
    }
  }
  return results;
}

export function analyzeSentimentSimple(text, analyzeFor = 'user') {
  const textLower = text.toLowerCase();
  const positiveWords = ['good', 'great', 'better', 'improving', 'thanks', 'thank', 'happy', 'relieved', 'well', 'okay'];
  const negativeWords = ['bad', 'terrible', 'worse', 'pain', 'hurt', 'anxious', 'worried', 'scared', 'depressed', 'sick'];
  const crisisWords = ['suicide', 'kill myself', 'end my life', 'want to die', 'self harm', 'overdose'];
  
  const positiveCount = positiveWords.filter(word => textLower.includes(word)).length;
  const negativeCount = negativeWords.filter(word => textLower.includes(word)).length;
  const hasCrisisWord = crisisWords.some(word => textLower.includes(word));
  
  let moodScore = 0;
  if (positiveCount > negativeCount) {
    moodScore = 0.5 + (positiveCount * 0.1);
  } else if (negativeCount > positiveCount) {
    moodScore = -0.5 - (negativeCount * 0.1);
  }
  
  moodScore = Math.max(-1, Math.min(1, moodScore));
  const stressScore = Math.min(1, negativeCount * 0.2 + (text.length > 100 ? 0.3 : 0));
  
  return {
    moodScore: parseFloat(moodScore.toFixed(3)),
    stressScore: parseFloat(stressScore.toFixed(3)),
    emotion: categorizeEmotionSimple(moodScore, stressScore, analyzeFor),
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

function categorizeEmotionSimple(moodScore, stressScore, analyzeFor) {
  if (analyzeFor === 'ai') {
    if (moodScore > 0.7) return 'empathetic';
    if (moodScore > 0.3) return 'supportive';
    if (moodScore > -0.3) return 'neutral';
    if (moodScore > -0.7) return 'concerned';
    return 'urgent';
  }
  
  if (stressScore > 0.8 && moodScore < -0.7) return 'crisis';
  if (stressScore > 0.6) return 'high-stress';
  if (moodScore > 0.7) return 'positive';
  if (moodScore > 0.3) return 'calm';
  if (moodScore > -0.3) return 'neutral';
  if (moodScore > -0.7) return 'worried';
  return 'distressed';
}

export function getSentimentAnalyzerStatus() {
  return sentimentAnalyzer !== null;
}