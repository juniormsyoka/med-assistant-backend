export function enhanceWithHealthcareContext(text, distilbertLabel, distilbertScore, analyzeFor) {
  const textLower = text.toLowerCase();
  const isPositive = distilbertLabel === 'POSITIVE';
  const moodScore = isPositive ? distilbertScore : -distilbertScore;
  
  let stressScore = calculateHealthcareStressScore(text, moodScore);
  
  if (moodScore > 0.7) {
    stressScore *= 0.6;
  } else if (moodScore > 0.3) {
    stressScore *= 0.8;
  }
  
  if (moodScore > 0.5 && text.includes('!')) {
    stressScore *= 0.9;
  }
  
  stressScore = Math.min(1, Math.max(0, stressScore));
  
  const hasCrisisKeyword = hasCrisisKeywords(textLower);
  const hasUrgency = hasUrgencyKeywords(textLower);
  const mentionsPain = mentionsPainKeywords(textLower);
  const mentionsMedication = mentionsMedicationKeywords(textLower);
  
  const keywordDetectedEmotion = detectEmotionFromKeywords(textLower);
  let emotion;
  if (keywordDetectedEmotion) {
    emotion = keywordDetectedEmotion;
  } else {
    emotion = categorizeEmotionImproved(moodScore, stressScore, textLower, analyzeFor);
  }
  
  let isCrisis = detectCrisis(moodScore, stressScore, hasCrisisKeyword, hasUrgency, mentionsPain, analyzeFor, keywordDetectedEmotion);
  
  let confidence = distilbertScore;

  // Healthcare context boosts - reduced
  if (mentionsPain || mentionsMedication || hasUrgency) {
    confidence = Math.min(0.95, confidence * 1.05); // Reduced boost, cap at 0.95
  }

  // Text length adjustment - more nuanced
  const wordCount = text.split(' ').length;
  if (wordCount < 3) {
    confidence *= 0.7;
  } else if (wordCount < 6) {
    confidence *= 0.85;
  } else if (wordCount > 50) {
    confidence *= 0.9; // Very long texts might be less clear
  }

  // Add nuance based on emotional content
  if (keywordDetectedEmotion && Math.abs(moodScore) > 0.7) {
    confidence *= 0.95; // Slight reduction for extreme emotions
  }
  
  const textMetrics = calculateTextMetrics(text);
  const emotionalIntensity = calculateEmotionalIntensity(textMetrics);
  
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
      wordCount: textMetrics.wordCount,
      hasQuestion: textMetrics.questionCount > 0,
      hasExclamation: textMetrics.exclamationCount > 0,
      allCapsRatio: textMetrics.capsRatio,
      exclamationCount: textMetrics.exclamationCount,
      questionCount: textMetrics.questionCount,
      emotionalIntensity: parseFloat(emotionalIntensity.toFixed(3)),
      keywordDetectedEmotion,
      containsPositiveKeywords: moodScore > 0.7 && (textLower.includes('good') || textLower.includes('great')),
      containsNegativeKeywords: moodScore < -0.5 && (textLower.includes('bad') || textLower.includes('terrible'))
    },
    suggestedResponseTone: suggestedTone,
    modelUsed: 'distilbert',
    analyzedAt: new Date().toISOString(),
    analysisType: analyzeFor === 'ai' ? 'ai_response' : 'user_input'
  };
}

function calculateHealthcareStressScore(text, moodScore) {
  const textLower = text.toLowerCase();
  let stressScore = 0.3; // Start lower (was 0.5)
  
  // Mood impact - reduced
  if (moodScore < -0.8) {
    stressScore += 0.4;
  } else if (moodScore < -0.5) {
    stressScore += 0.25;
  } else if (moodScore < -0.2) {
    stressScore += 0.15;
  } else if (moodScore > 0.7) {
    stressScore -= 0.15; // Positive mood reduces stress
  }
  
  // Urgency - more nuanced
  if (textLower.includes('911')) stressScore += 0.3;
  else if (textLower.includes('emergency')) stressScore += 0.2;
  else if (textLower.includes('urgent')) stressScore += 0.15;
  else if (textLower.includes('asap') || textLower.includes('immediately')) stressScore += 0.1;
  
  // Pain - more nuanced
  const severePainWords = ['unbearable', 'excruciating', 'severe', 'terrible', 'awful'];
  const moderatePainWords = ['pain', 'hurt', 'aching', 'sore'];
  const mildPainWords = ['discomfort', 'uncomfortable', 'slight', 'minor'];
  
  if (severePainWords.some(word => textLower.includes(word))) stressScore += 0.3;
  else if (moderatePainWords.some(word => textLower.includes(word))) stressScore += 0.15;
  else if (mildPainWords.some(word => textLower.includes(word))) stressScore += 0.05;
  
  // Medical context
  const medicalWords = ['hospital', 'doctor', 'clinic', 'er', 'emergency room', 'appointment'];
  if (medicalWords.some(word => textLower.includes(word))) stressScore += 0.05;
  
  // Punctuation - more nuanced
  const exclamationCount = (text.match(/!/g) || []).length;
  const questionCount = (text.match(/\?/g) || []).length;
  
  if (exclamationCount >= 3) stressScore += 0.2;
  else if (exclamationCount === 2) stressScore += 0.1;
  else if (exclamationCount === 1) stressScore += 0.05;
  
  if (questionCount >= 3) stressScore += 0.1;
  else if (questionCount > 0) stressScore += 0.03;
  
  // Length - more nuanced
  if (text.length > 200) stressScore += 0.05;
  
  // Capitalization - more nuanced
  const capsCount = (text.match(/[A-Z]/g) || []).length;
  const capsRatio = capsCount / text.length;
  if (capsRatio > 0.7) stressScore += 0.15;
  else if (capsRatio > 0.5) stressScore += 0.1;
  else if (capsRatio > 0.3) stressScore += 0.05;
  
  // Emotional words - more nuanced
  const highStressWords = ['panic', 'cant cope', 'breaking down', 'suicide', 'kill myself'];
  const mediumStressWords = ['anxious', 'worried', 'scared', 'nervous', 'stressed', 'overwhelmed'];
  const lowStressWords = ['concerned', 'apprehensive', 'uneasy', 'bothered'];
  
  if (highStressWords.some(word => textLower.includes(word))) stressScore += 0.3;
  else if (mediumStressWords.some(word => textLower.includes(word))) stressScore += 0.15;
  else if (lowStressWords.some(word => textLower.includes(word))) stressScore += 0.05;
  
  return Math.min(0.95, Math.max(0.1, stressScore)); // Cap at 0.95, min 0.1
}

function hasCrisisKeywords(text) {
  const crisisKeywords = [
    'suicide', 'kill myself', 'end my life', 'want to die', 
    'self harm', 'self-harm', 'cutting', 'overdose',
    'panic attack', 'cant breathe', 'chest pain', 'emergency',
    'help me', 'i give up', 'nothing matters'
  ];
  return crisisKeywords.some(keyword => text.includes(keyword));
}

function hasUrgencyKeywords(text) {
  const urgencyKeywords = ['urgent', 'emergency', '911', 'immediately', 'now', 'asap'];
  return urgencyKeywords.some(keyword => text.includes(keyword));
}

function mentionsPainKeywords(text) {
  const painKeywords = ['pain', 'hurt', 'aching', 'sore', 'unbearable', 'excruciating'];
  return painKeywords.some(keyword => text.includes(keyword));
}

function mentionsMedicationKeywords(text) {
  const medKeywords = ['medication', 'pill', 'drug', 'prescription', 'dose', 'tablet'];
  return medKeywords.some(keyword => text.includes(keyword));
}

function detectEmotionFromKeywords(text) {
  const emotionKeywords = {
    'anxious': ['panic', 'anxiety attack', 'freaking out', 'losing control'],
    'worried': ['worried', 'concerned', 'apprehensive', 'uneasy'],
    'happy': ['happy', 'great', 'good', 'excellent', 'wonderful', 'amazing', 'fantastic'],
    'sad': ['sad', 'depressed', 'miserable', 'unhappy', 'hopeless', 'tearful'],
    'angry': ['angry', 'mad', 'furious', 'enraged', 'irate', 'upset'],
    'calm': ['calm', 'relaxed', 'peaceful', 'chill', 'serene', 'content'],
    'stressed': ['stressed', 'overwhelmed', 'burned out', 'pressured', 'swamped'],
    'frustrated': ['frustrated', 'annoyed', 'irritated', 'fed up', 'exasperated'],
    'pain': ['unbearable pain', 'excruciating', 'severe pain', 'hurt so much']
  };
  
  // Check for stronger emotions first
  for (const [emotion, keywords] of Object.entries(emotionKeywords)) {
    for (const keyword of keywords) {
      if (text.includes(keyword)) {
        // For milder emotions like "worried", require additional context
        if (emotion === 'worried' || emotion === 'concerned') {
          // Check if it's qualified as "a bit" or "slightly"
          if (text.includes('a bit worried') || text.includes('slightly worried') || 
              text.includes('little worried') || text.includes('somewhat concerned')) {
            return 'mild-concern'; // New category
          }
        }
        return emotion;
      }
    }
  }
  
  return null;
}

function categorizeEmotionImproved(moodScore, stressScore, textLower, analyzeFor) {
  if (analyzeFor === 'ai') {
    if (moodScore > 0.8) return 'empathetic';
    if (moodScore > 0.5) return 'supportive';
    if (moodScore > 0.2) return 'encouraging';
    if (moodScore > -0.2) return 'neutral';
    if (moodScore > -0.5) return 'concerned';
    if (moodScore > -0.8) return 'urgent';
    return 'crisis';
  }

  if (textLower.includes('a bit') || textLower.includes('slightly') || 
      textLower.includes('little') || textLower.includes('somewhat')) {
    if (stressScore > 0.4 && moodScore < 0) {
      return 'mild-concern';
    }
  }
  
  if (stressScore > 0.9 && moodScore < -0.8) return 'crisis';
  if (stressScore > 0.85) return 'high-stress';
  
  if (stressScore > 0.6 && moodScore > 0.3) return 'excited';
  if (stressScore > 0.6 && moodScore >= 0) return 'concerned';
  if (stressScore > 0.6 && moodScore < 0) return 'stressed';
  
  if (moodScore > 0.85) return 'very-positive';
  if (moodScore > 0.7) return 'positive';
  if (moodScore > 0.5) return 'slightly-positive';
  if (moodScore > 0.3) return 'calm';
  if (moodScore > -0.3) return 'neutral';
  if (moodScore > -0.6) return 'slightly-negative';
  if (moodScore > -0.8) return 'negative';
  
  return 'very-negative';
}

function detectCrisis(moodScore, stressScore, hasCrisisKeyword, hasUrgency, mentionsPain, analyzeFor, keywordDetectedEmotion) {
  if (analyzeFor !== 'user') return false;
  if (hasCrisisKeyword) return true;
  if (stressScore > 0.9 && moodScore < -0.8) return true;
  if (hasUrgency && mentionsPain && moodScore < -0.7) return true;
  if (keywordDetectedEmotion === 'anxious' || keywordDetectedEmotion === 'worried') {
    if (stressScore > 0.85 && moodScore < -0.75) return true;
  }
  
  if (keywordDetectedEmotion === 'sad' && stressScore > 0.85 && moodScore < -0.7) return true;
  
  return false;
}

function calculateTextMetrics(text) {
  const wordCount = text.split(' ').length;
  const exclamationCount = (text.match(/!/g) || []).length;
  const questionCount = (text.match(/\?/g) || []).length;
  const capsCount = (text.match(/[A-Z]/g) || []).length;
  const capsRatio = capsCount / text.length || 0;
  
  return { wordCount, exclamationCount, questionCount, capsRatio };
}

function calculateEmotionalIntensity(textMetrics) {
  return Math.min(1, 
    (textMetrics.exclamationCount * 0.2) + 
    (textMetrics.capsRatio > 0.3 ? 0.3 : 0) + 
    (textMetrics.wordCount > 50 ? 0.2 : 0)
  );
}

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