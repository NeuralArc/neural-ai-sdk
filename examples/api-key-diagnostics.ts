/**
 * API Key Diagnostics Tool
 *
 * This script helps diagnose connection issues with each AI provider.
 * Run this script to test your API keys one by one.
 */
import * as dotenv from "dotenv";
import { NeuralAI, AIProvider } from "../src";
import { GoogleGenerativeAI } from "@google/generative-ai";

// Load environment variables
dotenv.config();

async function runTest() {
  console.log("🔍 Neural AI SDK API Key Diagnostics");
  console.log("===================================\n");

  // Test each provider sequentially
  await testOpenAI();
  await testGoogle();
  await testDeepSeek();
  await testHuggingFace();
  await testOllama();

  console.log("\n✨ Diagnostics completed");
}

async function testOpenAI() {
  console.log("📝 Testing OpenAI...");
  if (!process.env.OPENAI_API_KEY) {
    console.log("❌ No OpenAI API key found in environment variables");
    return;
  }

  try {
    // First check if we can just initialize the model
    const openaiModel = NeuralAI.createModel(AIProvider.OPENAI, {
      model: "gpt-3.5-turbo", // Use cheaper model for testing
    });

    console.log("✅ Successfully initialized OpenAI client");

    // Try a minimal completion to verify API key works
    try {
      console.log("   Attempting to generate a short response...");
      const response = await openaiModel.generate({
        prompt: "Hello!",
        systemPrompt: "Reply with a single word: Hi",
      });

      console.log("✅ Successfully generated response from OpenAI:");
      console.log(`   "${response.text.trim()}"`);
    } catch (error: any) {
      console.log(`❌ Failed to generate response: ${error.message}`);

      if (error.message.includes("quota")) {
        console.log("   Your OpenAI API key has exceeded its quota. Consider:");
        console.log("   1. Adding billing information to your OpenAI account");
        console.log("   2. Using a different API key");
        console.log("   3. Waiting until your quota resets");
      }
    }
  } catch (error: any) {
    console.log(`❌ Failed to initialize OpenAI client: ${error.message}`);
  }
}

async function testGoogle() {
  console.log("\n📝 Testing Google AI...");
  if (!process.env.GOOGLE_API_KEY) {
    console.log("❌ No Google API key found in environment variables");
    return;
  }

  try {
    // Use gemini-2.0-flash as the default model instead of trying to list models
    const availableModel = "gemini-2.0-flash";
    console.log(`   Using Google AI model: ${availableModel}`);

    // Try a completion
    try {
      console.log("   Attempting to generate a response...");
      const googleModel = NeuralAI.createModel(AIProvider.GOOGLE, {
        model: availableModel,
      });

      const response = await googleModel.generate({
        prompt: "What is 2+2?",
      });

      console.log("✅ Successfully generated response from Google:");
      console.log(
        `   "${response.text.substring(0, 50)}${
          response.text.length > 50 ? "..." : ""
        }"`
      );
    } catch (error: any) {
      console.log(`❌ Failed to generate response: ${error.message}`);
    }
  } catch (error: any) {
    console.log(`❌ Failed to initialize Google client: ${error.message}`);
  }
}

async function testDeepSeek() {
  console.log("\n📝 Testing DeepSeek...");
  if (!process.env.DEEPSEEK_API_KEY) {
    console.log("❌ No DeepSeek API key found in environment variables");
    return;
  }

  try {
    const deepseekModel = NeuralAI.createModel(AIProvider.DEEPSEEK, {});
    console.log("✅ Successfully initialized DeepSeek client");

    try {
      console.log("   Attempting to generate a response...");
      const response = await deepseekModel.generate({
        prompt: "What is 3+3?",
      });

      console.log("✅ Successfully generated response from DeepSeek:");
      console.log(
        `   "${response.text.substring(0, 50)}${
          response.text.length > 50 ? "..." : ""
        }"`
      );
    } catch (error: any) {
      console.log(`❌ Failed to generate response: ${error.message}`);

      if (error.message.includes("402")) {
        console.log(
          "   This appears to be a payment issue. Please check your DeepSeek account:"
        );
        console.log("   1. Ensure your account has sufficient credits");
        console.log("   2. Verify that your payment method is valid");
        console.log("   3. Check if you need to upgrade your subscription");
      }
    }
  } catch (error: any) {
    console.log(`❌ Failed to initialize DeepSeek client: ${error.message}`);
  }
}

async function testHuggingFace() {
  console.log("\n📝 Testing HuggingFace...");
  if (!process.env.HUGGINGFACE_API_KEY) {
    console.log("❌ No HuggingFace API token found in environment variables");
    return;
  }

  try {
    // Try a simple model first that most accounts should have access to
    const models = [
      "gpt2",
      "distilgpt2",
      "facebook/bart-large-cnn",
      "meta-llama/Llama-2-7b-chat-hf", // This may require special access
    ];

    for (const model of models) {
      console.log(`\n   Trying HuggingFace model: ${model}`);

      try {
        const huggingfaceModel = NeuralAI.createModel(AIProvider.HUGGINGFACE, {
          model: model,
        });

        console.log("   ✅ Successfully initialized HuggingFace client");

        const response = await huggingfaceModel.generate({
          prompt: "Hello, how are you?",
        });

        console.log("   ✅ Successfully generated response:");
        console.log(
          `      "${response.text.substring(0, 50)}${
            response.text.length > 50 ? "..." : ""
          }"`
        );
        break; // If successful, stop trying more models
      } catch (error: any) {
        console.log(`   ❌ Failed with model ${model}: ${error.message}`);
        if (error.message.includes("403")) {
          console.log(
            "      Permission denied - your token may not have access to this model"
          );
        }
      }
    }
  } catch (error: any) {
    console.log(`❌ Failed to initialize HuggingFace client: ${error.message}`);
  }
}

async function testOllama() {
  console.log("\n📝 Testing Ollama...");
  const baseURL = process.env.OLLAMA_BASE_URL || "http://localhost:11434/api";
  console.log(`   Using Ollama base URL: ${baseURL}`);

  try {
    const ollamaModel = NeuralAI.createModel(AIProvider.OLLAMA, {});
    console.log("✅ Successfully initialized Ollama client");

    try {
      console.log("   Attempting to connect to Ollama server...");
      const response = await ollamaModel.generate({
        prompt: "Hello!",
      });

      console.log("✅ Successfully generated response from Ollama:");
      console.log(
        `   "${response.text.substring(0, 50)}${
          response.text.length > 50 ? "..." : ""
        }"`
      );
    } catch (error: any) {
      console.log(`❌ Failed to connect to Ollama: ${error.message}`);
      console.log(
        "   Make sure Ollama is running locally or available at the specified URL"
      );
      console.log(`   Default URL: ${baseURL}`);
    }
  } catch (error: any) {
    console.log(`❌ Failed to initialize Ollama client: ${error.message}`);
  }
}

// Run the diagnostics
runTest().catch((error) => {
  console.error("Error running diagnostics:", error);
});
