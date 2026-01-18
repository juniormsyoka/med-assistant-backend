export function generateConversationInsights(individualResults, originalMessages) {
  if (!individualResults || individualResults.length === 0) {
    return {
      moodTrend: 'unknown',
      dominantEmotions: [],
      crisisProbability: 0,
      recommendations: [],
      summary: 'No messages analyzed'
    };
  }

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

  const emotionCounts = {};
  individualResults.forEach(result => {
    if (result.analysis && result.analysis.emotion) {
      const emotion = result.analysis.emotion;
      emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
    }
  });

  const dominantEmotions = Object.entries(emotionCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([emotion, count]) => ({ emotion, count, percentage: (count / individualResults.length * 100).toFixed(1) }));

  const crisisMessages = individualResults.filter(r => r.analysis && r.analysis.isCrisis);
  const crisisProbability = Math.min(1, crisisMessages.length / individualResults.length);

  let moodTrend = calculateMoodTrend(validMoodScores);

  const recommendations = generateRecommendations(
    avgMood, 
    avgStress, 
    crisisProbability, 
    moodTrend,
    dominantEmotions
  );

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

function calculateMoodTrend(moodScores) {
  if (moodScores.length < 4) return 'stable';
  
  const half = Math.floor(moodScores.length / 2);
  const firstHalfAvg = moodScores.slice(0, half).reduce((a, b) => a + b, 0) / half;
  const secondHalfAvg = moodScores.slice(half).reduce((a, b) => a + b, 0) / (moodScores.length - half);
  
  if (secondHalfAvg - firstHalfAvg > 0.3) return 'improving';
  else if (firstHalfAvg - secondHalfAvg > 0.3) return 'worsening';
  
  return 'stable';
}

function generateRecommendations(avgMood, avgStress, crisisProbability, moodTrend, dominantEmotions) {
  const recommendations = [];

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