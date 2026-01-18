// Create a new file: services/PatternDetector.js
class PatternDetector {
  constructor() {
    this.weekdayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  }

  analyzeComplianceData(complianceRecords) {
    if (!complianceRecords || complianceRecords.length === 0) {
      return {
        patterns: [],
        risk_factors: [],
        summary: "Insufficient data for pattern analysis",
        medication_count: 0,
        total_records: 0,
        analysis_period: { start: null, end: null }
      };
    }

    // Convert to array if needed
    const records = Array.isArray(complianceRecords) ? complianceRecords : [complianceRecords];
    
    // Parse dates
    const parsedRecords = records.map(record => ({
      ...record,
      createdAt: new Date(record.createdAt),
      scheduledTime: record.scheduledTime ? new Date(record.scheduledTime) : null
    }));

    const patterns = this.detectPatterns(parsedRecords);
    const risk_factors = this.identifyRiskFactors(parsedRecords);

    // Find date range
    const dates = parsedRecords
      .map(r => r.createdAt)
      .filter(d => !isNaN(d.getTime()))
      .sort((a, b) => a - b);
    
    const startDate = dates.length > 0 ? dates[0] : null;
    const endDate = dates.length > 0 ? dates[dates.length - 1] : null;

    // Count unique medications
    const medicationIds = [...new Set(records.map(r => r.medicationId))];

    return {
      patterns,
      risk_factors,
      medication_count: medicationIds.length,
      total_records: records.length,
      analysis_period: {
        start: startDate ? startDate.toISOString() : null,
        end: endDate ? endDate.toISOString() : null
      }
    };
  }

  detectPatterns(records) {
    const patterns = [];

    if (records.length < 5) {
      return patterns;
    }

    // 1. Time-of-day patterns
    const hourGroups = {};
    records.forEach(record => {
      if (record.hourOfDay !== undefined) {
        const hour = record.hourOfDay;
        if (!hourGroups[hour]) {
          hourGroups[hour] = { total: 0, taken: 0 };
        }
        hourGroups[hour].total++;
        if (record.actualAction === 'taken') {
          hourGroups[hour].taken++;
        }
      }
    });

    let bestHour = null;
    let worstHour = null;
    let maxRate = 0;
    let minRate = 100;

    Object.entries(hourGroups).forEach(([hour, data]) => {
      const rate = (data.taken / data.total) * 100;
      if (rate > maxRate) {
        maxRate = rate;
        bestHour = hour;
      }
      if (rate < minRate) {
        minRate = rate;
        worstHour = hour;
      }
    });

    if (maxRate - minRate > 20 && bestHour && worstHour) {
      patterns.push(
        `Strong time-based pattern: ${bestHour}:00 has ${maxRate.toFixed(1)}% adherence, ` +
        `while ${worstHour}:00 has ${minRate.toFixed(1)}% adherence`
      );
    }

    // 2. Day-of-week patterns
    const dayGroups = {};
    records.forEach(record => {
      if (record.dayOfWeek !== undefined) {
        const day = record.dayOfWeek;
        if (!dayGroups[day]) {
          dayGroups[day] = { total: 0, taken: 0 };
        }
        dayGroups[day].total++;
        if (record.actualAction === 'taken') {
          dayGroups[day].taken++;
        }
      }
    });

    let bestDay = null;
    let worstDay = null;
    let maxDayRate = 0;
    let minDayRate = 100;

    Object.entries(dayGroups).forEach(([day, data]) => {
      const rate = (data.taken / data.total) * 100;
      if (rate > maxDayRate) {
        maxDayRate = rate;
        bestDay = day;
      }
      if (rate < minDayRate) {
        minDayRate = rate;
        worstDay = day;
      }
    });

    if (maxDayRate - minDayRate > 15 && bestDay !== null && worstDay !== null) {
      patterns.push(
        `Weekend vs weekday difference: ${this.weekdayNames[bestDay]} ` +
        `(${maxDayRate.toFixed(1)}%) vs ${this.weekdayNames[worstDay]} ` +
        `(${minDayRate.toFixed(1)}%)`
      );
    }

    // 3. Location-based patterns
    const locationGroups = {};
    records.forEach(record => {
      if (record.location) {
        if (!locationGroups[record.location]) {
          locationGroups[record.location] = { total: 0, taken: 0 };
        }
        locationGroups[record.location].total++;
        if (record.actualAction === 'taken') {
          locationGroups[record.location].taken++;
        }
      }
    });

    if (Object.keys(locationGroups).length > 1) {
      let bestLocation = null;
      let worstLocation = null;
      let maxLocationRate = 0;
      let minLocationRate = 100;

      Object.entries(locationGroups).forEach(([location, data]) => {
        const rate = (data.taken / data.total) * 100;
        if (rate > maxLocationRate) {
          maxLocationRate = rate;
          bestLocation = location;
        }
        if (rate < minLocationRate) {
          minLocationRate = rate;
          worstLocation = location;
        }
      });

      if (bestLocation && worstLocation) {
        patterns.push(
          `Location affects adherence: Best at ${bestLocation} ` +
          `(${maxLocationRate.toFixed(1)}%), worst at ${worstLocation} ` +
          `(${minLocationRate.toFixed(1)}%)`
        );
      }
    }

    // 4. Streak patterns
    const takenRecords = records.filter(r => r.actualAction === 'taken');
    if (takenRecords.length > 10) {
      takenRecords.sort((a, b) => a.createdAt - b.createdAt);
      
      // Extract unique dates
      const uniqueDates = [];
      const dateSet = new Set();
      
      takenRecords.forEach(record => {
        const dateStr = record.createdAt.toISOString().split('T')[0];
        if (!dateSet.has(dateStr)) {
          dateSet.add(dateStr);
          uniqueDates.push(dateStr);
        }
      });

      // Calculate streak
      let currentStreak = 1;
      let longestStreak = 1;

      for (let i = 1; i < uniqueDates.length; i++) {
        const prevDate = new Date(uniqueDates[i - 1]);
        const currDate = new Date(uniqueDates[i]);
        const dayDiff = (currDate - prevDate) / (1000 * 60 * 60 * 24);
        
        if (dayDiff === 1) {
          currentStreak++;
          longestStreak = Math.max(longestStreak, currentStreak);
        } else {
          currentStreak = 1;
        }
      }

      if (longestStreak >= 7) {
        patterns.push(`Impressive streak: ${longestStreak} consecutive days of adherence`);
      }
    }

    return patterns;
  }

  identifyRiskFactors(records) {
    const risk_factors = [];

    if (records.length < 10) {
      return ["Insufficient data for risk factor analysis"];
    }

    // 1. Time-based risk factors
    const hourRisk = {};
    records.forEach(record => {
      if (record.hourOfDay !== undefined) {
        const hour = record.hourOfDay;
        if (!hourRisk[hour]) {
          hourRisk[hour] = { total: 0, missed: 0 };
        }
        hourRisk[hour].total++;
        if (record.actualAction === 'missed') {
          hourRisk[hour].missed++;
        }
      }
    });

    let highestRiskHour = null;
    let highestRiskRate = 0;

    Object.entries(hourRisk).forEach(([hour, data]) => {
      const riskRate = (data.missed / data.total);
      if (riskRate > highestRiskRate) {
        highestRiskRate = riskRate;
        highestRiskHour = hour;
      }
    });

    if (highestRiskRate > 0.4 && highestRiskHour !== null) {
      risk_factors.push(`High risk time: ${highestRiskHour}:00 (${(highestRiskRate * 100).toFixed(1)}% missed)`);
    }

    // 2. Context-based risk factors
    const contextFactors = ['location', 'mood', 'batteryLevel'];
    
    contextFactors.forEach(factor => {
      const factorGroups = {};
      
      records.forEach(record => {
        const value = record[factor];
        if (value !== undefined && value !== null) {
          if (!factorGroups[value]) {
            factorGroups[value] = { total: 0, missed: 0 };
          }
          factorGroups[value].total++;
          if (record.actualAction === 'missed') {
            factorGroups[value].missed++;
          }
        }
      });

      Object.entries(factorGroups).forEach(([value, data]) => {
        const riskRate = data.missed / data.total;
        if (riskRate > 0.5 && data.total >= 3) {
          risk_factors.push(`High risk when ${factor}: ${value} (${(riskRate * 100).toFixed(1)}% missed)`);
        }
      });
    });

    // 3. Latency risk (if available)
    const takenWithLatency = records.filter(r => 
      r.actualAction === 'taken' && r.latencySeconds !== undefined
    );
    const missedWithLatency = records.filter(r => 
      r.actualAction === 'missed' && r.latencySeconds !== undefined
    );

    if (missedWithLatency.length > 0 && takenWithLatency.length > 0) {
      const avgMissLatency = missedWithLatency.reduce((sum, r) => sum + r.latencySeconds, 0) / missedWithLatency.length;
      const avgTakenLatency = takenWithLatency.reduce((sum, r) => sum + r.latencySeconds, 0) / takenWithLatency.length;
      
      if (avgMissLatency > avgTakenLatency * 2) {
        risk_factors.push(`Slow response to reminders (${avgMissLatency.toFixed(0)}s avg) increases risk`);
      }
    }

    return risk_factors.slice(0, 3); // Return top 3 risk factors
  }
}
export default PatternDetector;