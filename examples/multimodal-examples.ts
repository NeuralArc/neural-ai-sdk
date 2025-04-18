import { NeuralAI, AIProvider } from "../src";
import * as dotenv from "dotenv";

// Load environment variables from .env file
dotenv.config();

// Sample image URLs from the internet
const IMAGE_URLS = {
  // Nature image
  SAMPLE_IMAGE:
    "https://images.unsplash.com/photo-1500829243541-74b677fecc30?q=80&w=1000",
  // Cat image
  IMAGE1:
    "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?q=80&w=1000",
  // Dog image
  IMAGE2:
    "https://images.unsplash.com/photo-1543466835-00a7907e9de1?q=80&w=1000",
};

async function main() {
  try {
    // Test multimodal capabilities with different providers
    // await testOpenAIMultimodal();
    // await testGoogleMultimodal();
    // await testOllamaMultimodal();
    await testHuggingFaceMultimodal();

    // Examples for multiple images content
    // await testStructuredMultimodalContent();
    // await testMultiImageOllama();
  } catch (error: any) {
    console.error("Error:", error);
  }
}

async function testOpenAIMultimodal() {
  console.log("\n===== OpenAI Multimodal Test =====");
  if (!process.env.OPENAI_API_KEY) {
    console.log("Skipping OpenAI test: No API key found");
    return;
  }

  try {
    const openaiModel = NeuralAI.createModel(AIProvider.OPENAI, {
      apiKey: process.env.OPENAI_API_KEY,
      // Use a model that supports vision capabilities
      model: "gpt-4o",
    });

    console.log("Generating response from image...");
    console.log(`Using image URL: ${IMAGE_URLS.SAMPLE_IMAGE}`);

    const response = await openaiModel.generate({
      prompt: "What's in this image? Please describe it in detail.",
      image: IMAGE_URLS.SAMPLE_IMAGE,
    });

    console.log("Response:", response.text);
  } catch (error: any) {
    console.error("OpenAI Multimodal Error:", error);
  }
}

async function testGoogleMultimodal() {
  console.log("\n===== Google AI Multimodal Test =====");
  if (!process.env.GOOGLE_API_KEY) {
    console.log("Skipping Google test: No API key found");
    return;
  }

  try {
    const googleModel = NeuralAI.createModel(AIProvider.GOOGLE, {
      apiKey: process.env.GOOGLE_API_KEY,
      model: "gemini-2.0-flash",
    });

    console.log("Generating response from image...");

    // Random image from Picsum Photos service
    const imageUrl = "https://picsum.photos/800/600";
    console.log(`Using image URL: ${imageUrl}`);

    const response = await googleModel.generate({
      prompt: "Analyze this image and tell me what you see.",
      image: imageUrl,
    });

    console.log("Response:", response.text);
  } catch (error: any) {
    console.error("Google Multimodal Error:", error);
  }
}

async function testOllamaMultimodal() {
  console.log("\n===== Ollama Multimodal Test =====");

  try {
    // Create an Ollama model with a vision-capable model
    // Note: You need to have the model pulled in your Ollama instance
    // Example: ollama pull llama-3.2-vision
    const ollamaModel = NeuralAI.createModel(AIProvider.OLLAMA, {
      model: "llama-3.2-vision", // Or any other vision model
      baseURL: process.env.OLLAMA_BASE_URL || "http://localhost:11434/api",
    });

    console.log("Generating response from image with Ollama...");
    console.log(`Using image URL: ${IMAGE_URLS.SAMPLE_IMAGE}`);

    const response = await ollamaModel.generate({
      prompt: "What's in this image? Please describe it in detail.",
      image: IMAGE_URLS.SAMPLE_IMAGE,
    });

    console.log("Response:", response.text);
  } catch (error: any) {
    console.error("Ollama Multimodal Error:", error);
    console.log(
      "Note: Make sure your Ollama instance is running and has a vision-capable model installed"
    );
    console.log("You can install one with: ollama pull llama-3.2-vision");
  }
}

async function testHuggingFaceMultimodal() {
  console.log("\n===== HuggingFace Multimodal Test =====");
  if (!process.env.HUGGINGFACE_API_KEY) {
    console.log("Skipping HuggingFace test: No API key found");
    return;
  }

  try {
    // Create a HuggingFace model with a vision-capable model
    const huggingfaceModel = NeuralAI.createModel(AIProvider.HUGGINGFACE, {
      apiKey: process.env.HUGGINGFACE_API_KEY,
      // You can try different vision-capable models
      model: "llava-hf/llava-1.5-7b-hf",
    });

    console.log("Generating response from image with HuggingFace...");
    console.log(`Using image URL: ${IMAGE_URLS.IMAGE1}`);

    const response = await huggingfaceModel.generate({
      prompt: "Analyze this image and describe what you see.",
      image: IMAGE_URLS.IMAGE1,
    });

    console.log("Response:", response.text);
  } catch (error: any) {
    console.error("HuggingFace Multimodal Error:", error);
    console.log(
      "Note: Make sure you have access to the requested model in your HuggingFace account"
    );
  }
}

async function testStructuredMultimodalContent() {
  console.log("\n===== OpenAI Structured Multimodal Content Test =====");
  if (!process.env.OPENAI_API_KEY) {
    console.log("Skipping test: No OpenAI API key found");
    return;
  }

  try {
    const openaiModel = NeuralAI.createModel(AIProvider.OPENAI, {
      apiKey: process.env.OPENAI_API_KEY,
      model: "gpt-4o",
    });

    console.log("Generating response from multiple content pieces...");
    console.log(`Using images: ${IMAGE_URLS.IMAGE1} and ${IMAGE_URLS.IMAGE2}`);

    // Example with multiple content pieces (text and images)
    const response = await openaiModel.generate({
      // You can still provide a main prompt
      prompt: "Compare these two images:",
      // And use the content array for more structured input
      content: [
        {
          type: "image",
          source: IMAGE_URLS.IMAGE1,
        },
        {
          type: "text",
          text: "This is the first image.",
        },
        {
          type: "image",
          source: IMAGE_URLS.IMAGE2,
        },
        {
          type: "text",
          text: "This is the second image.",
        },
      ],
    });

    console.log("Response:", response.text);
  } catch (error: any) {
    console.error("Structured Content Error:", error);
  }
}

async function testMultiImageOllama() {
  console.log("\n===== Ollama Multiple Images Test =====");

  try {
    const ollamaModel = NeuralAI.createModel(AIProvider.OLLAMA, {
      model: "llama-3.2-vision", // Or any other vision model that supports multiple images
      baseURL: process.env.OLLAMA_BASE_URL || "http://localhost:11434/api",
    });

    console.log("Generating response from multiple images with Ollama...");
    console.log(`Using images: ${IMAGE_URLS.IMAGE1} and ${IMAGE_URLS.IMAGE2}`);

    const response = await ollamaModel.generate({
      prompt:
        "Compare these two images and tell me the differences between the cat and dog:",
      content: [
        {
          type: "image",
          source: IMAGE_URLS.IMAGE1,
        },
        {
          type: "text",
          text: "This is a cat.",
        },
        {
          type: "image",
          source: IMAGE_URLS.IMAGE2,
        },
        {
          type: "text",
          text: "This is a dog.",
        },
      ],
    });

    console.log("Response:", response.text);
  } catch (error: any) {
    console.error("Ollama Multiple Images Error:", error);
    if (error.message.includes("doesn't support multimodal inputs")) {
      console.log(
        "The selected model doesn't appear to support multimodal inputs."
      );
      console.log("Try using a different vision-capable model.");
    } else if (error.message.toLowerCase().includes("multiple images")) {
      console.log(
        "Note: Not all vision models support multiple images in a single request"
      );
    }
  }
}

main();
