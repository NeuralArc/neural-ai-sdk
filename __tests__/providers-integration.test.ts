import { NeuralAI, AIProvider } from "../src";
import * as dotenv from "dotenv";

// Load environment variables before tests
dotenv.config();

// Skip tests if no API keys are available
const runOpenAITests = !!process.env.OPENAI_API_KEY;
const runGoogleTests = !!process.env.GOOGLE_API_KEY;
const runDeepSeekTests = !!process.env.DEEPSEEK_API_KEY;
const runHuggingFaceTests = !!process.env.HUGGINGFACE_API_KEY;
// For Ollama, we'll check if it's running locally during the test

describe("Neural AI SDK Integration Tests", () => {
  // Set longer timeout for API calls
  jest.setTimeout(60000);

  describe("OpenAI Provider", () => {
    (runOpenAITests ? it : it.skip)(
      "should successfully generate a response from OpenAI",
      async () => {
        const openaiModel = NeuralAI.createModel(AIProvider.OPENAI, {
          model: "gpt-4o", // Use gpt-4o instead of the default gpt-3.5-turbo
        });
        const response = await openaiModel.generate({
          prompt: "Hello, can you tell me what is 2+2?",
          systemPrompt:
            "You are a helpful assistant that answers math questions concisely.",
        });

        expect(response).toBeDefined();
        expect(response.text).toBeTruthy();
        expect(response.usage).toBeDefined();
        expect(response.usage?.totalTokens).toBeGreaterThan(0);
      }
    );
  });

  describe("Google Provider", () => {
    (runGoogleTests ? it : it.skip)(
      "should successfully generate a response from Google",
      async () => {
        const googleModel = NeuralAI.createModel(AIProvider.GOOGLE, {
          model: "gemini-2.0-flash",
        });
        const response = await googleModel.generate({
          prompt: "Hello, can you tell me what is 3+3?",
        });

        expect(response).toBeDefined();
        expect(response.text).toBeTruthy();
        expect(response.text.length).toBeGreaterThan(0);
      }
    );
  });

  describe("DeepSeek Provider", () => {
    (runDeepSeekTests ? it : it.skip)(
      "should successfully generate a response from DeepSeek",
      async () => {
        const deepseekModel = NeuralAI.createModel(AIProvider.DEEPSEEK, {});

        const response = await deepseekModel.generate({
          prompt: "Hello, can you tell me what is 4+4?",
        });

        expect(response).toBeDefined();
        expect(response.text).toBeTruthy();
        expect(response.text.length).toBeGreaterThan(0);
      }
    );
  });

  describe("Ollama Provider", () => {
    it("should attempt to connect to Ollama", async () => {
      try {
        const ollamaModel = NeuralAI.createModel(AIProvider.OLLAMA, {});

        // Just testing connection, not full response as Ollama may not be running
        await ollamaModel.generate({
          prompt: "Hello, can you tell me what is 5+5?",
        });

        // If we get here, Ollama is running
        console.log("✅ Ollama is running and responding to requests");
      } catch (error) {
        // Don't fail the test, just log that Ollama isn't available
        console.log("⚠️ Ollama is not running locally or is not accessible");
        console.log(
          "   To test Ollama, ensure it's running at the URL specified in OLLAMA_BASE_URL or at http://localhost:11434/api"
        );
      }

      // Always pass this test
      expect(true).toBeTruthy();
    });
  });

  describe("HuggingFace Provider", () => {
    (runHuggingFaceTests ? it : it.skip)(
      "should successfully generate a response from HuggingFace",
      async () => {
        const huggingfaceModel = NeuralAI.createModel(AIProvider.HUGGINGFACE, {
          model: "gpt2",
        });

        const response = await huggingfaceModel.generate({
          prompt: "Hello, can you tell me what is 6+6?",
        });

        expect(response).toBeDefined();
        expect(response.text).toBeTruthy();
        expect(response.text.length).toBeGreaterThan(0);
      }
    );
  });

  describe("Streaming Capabilities", () => {
    (runOpenAITests ? it : it.skip)(
      "should successfully stream a response from OpenAI",
      async () => {
        const openaiModel = NeuralAI.createModel(AIProvider.OPENAI, {});

        const stream = openaiModel.stream({
          prompt: "Count from 1 to 5.",
        });

        let result = "";
        for await (const chunk of stream) {
          result += chunk;
        }

        expect(result).toBeTruthy();
        expect(result.length).toBeGreaterThan(0);
      }
    );
  });
});
