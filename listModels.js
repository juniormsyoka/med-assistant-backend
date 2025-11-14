import { VertexAI } from '@google-cloud/vertexai';

// Initialize Vertex AI with your actual project ID
const vertexAI = new VertexAI({
  project: 'project-med-478209', // ✅ Your Project ID
  location: 'us-central1',
});

async function listModels() {
  try {
    console.log('🔄 Testing Vertex AI connection...');
    
    // Test with a simple model initialization
    const generativeModel = vertexAI.getGenerativeModel({
      model: 'gemini-1.0-pro',
    });

    console.log('✅ Vertex AI initialized successfully!');
    console.log('Testing with a simple prompt...');

    // Test with a simple prompt
    const prompt = 'Hello, in one sentence, tell me how AI can help with medication reminders.';
    const result = await generativeModel.generateContent(prompt);
    
    console.log('✅ Test successful! AI Response:');
    console.log(result.response.candidates[0].content.parts[0].text);
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    
    // More detailed error information
    if (error.details) {
      console.error('Error details:', error.details);
    }
  }
}

listModels();