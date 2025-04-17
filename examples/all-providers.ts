import { NeuralAI, AIProvider } from "../src";
import * as dotenv from "dotenv";

// Load environment variables from .env file
dotenv.config();

async function main() {
  try {
    // Test each provider if API keys are available
    await testOpenAI();
    await testGoogle();
    await testDeepSeek();
    await testOllama();
    await testHuggingFace();
  } catch (error) {
    console.error("Error:", error);
  }
}

async function testOpenAI() {
  console.log("\n===== OpenAI Test =====");
  if (!process.env.OPENAI_API_KEY) {
    console.log("Skipping OpenAI test: No API key found");
    return;
  }

  try {
    const openaiModel = NeuralAI.createModel(AIProvider.OPENAI, {
      apiKey: process.env.OPENAI_API_KEY,
      model: "gpt-3.5-turbo",
    });

    console.log("Generating response...");
    const response = await openaiModel.generate({
      prompt: "What are the key benefits of using a unified AI SDK?",
      systemPrompt:
        "You are a helpful AI assistant specialized in software development.",
    });

    console.log("Response:", response.text);
    console.log("Token Usage:", response.usage);
  } catch (error) {
    console.error("OpenAI Error:", error);
  }
}

async function testGoogle() {
  console.log("\n===== Google AI Test =====");
  if (!process.env.GOOGLE_API_KEY) {
    console.log("Skipping Google test: No API key found");
    return;
  }

  try {
    const googleModel = NeuralAI.createModel(AIProvider.GOOGLE, {
      apiKey: process.env.GOOGLE_API_KEY,
      model: "gemini-pro",
    });

    console.log("Generating response...");
    const response = await googleModel.generate({
      prompt:
        "Explain how different AI models can be used for different use cases.",
    });

    console.log("Response:", response.text);
  } catch (error) {
    console.error("Google Error:", error);
  }
}

async function testDeepSeek() {
  console.log("\n===== DeepSeek Test =====");
  if (!process.env.DEEPSEEK_API_KEY) {
    console.log("Skipping DeepSeek test: No API key found");
    return;
  }

  try {
    const deepseekModel = NeuralAI.createModel(AIProvider.DEEPSEEK, {
      apiKey: process.env.DEEPSEEK_API_KEY,
      model: "deepseek-chat",
    });

    console.log("Generating response...");
    const response = await deepseekModel.generate({
      prompt:
        "Compare transformer architecture with recurrent neural networks.",
    });

    console.log("Response:", response.text);
  } catch (error) {
    console.error("DeepSeek Error:", error);
  }
}

async function testOllama() {
  console.log("\n===== Ollama Test =====");
  try {
    const ollamaModel = NeuralAI.createModel(AIProvider.OLLAMA, {
      baseURL: process.env.OLLAMA_BASE_URL || "http://localhost:11434/api",
      model: "llama2",
    });

    console.log("Generating response...");
    const response = await ollamaModel.generate({
      prompt: "What are the advantages of running AI models locally?",
    });

    console.log("Response:", response.text);
  } catch (error) {
    console.error("Ollama Error:", error);
  }
}

async function testHuggingFace() {
  console.log("\n===== HuggingFace Test =====");
  if (!process.env.HUGGINGFACE_API_KEY) {
    console.log("Skipping HuggingFace test: No API key found");
    return;
  }

  try {
    const huggingfaceModel = NeuralAI.createModel(AIProvider.HUGGINGFACE, {
      apiKey: process.env.HUGGINGFACE_API_KEY,
      model: "meta-llama/Llama-2-7b-chat-hf",
    });

    console.log("Generating response...");
    const response = await huggingfaceModel.generate({
      prompt:
        "What are some challenges in prompt engineering for different AI models?",
    });

    console.log("Response:", response.text);
  } catch (error) {
    console.error("HuggingFace Error:", error);
  }
}

main();
