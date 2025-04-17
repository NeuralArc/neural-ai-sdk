import { NeuralAI, AIProvider } from "../src";

async function main() {
  try {
    // Example with OpenAI
    console.log("OpenAI Example:");
    const openaiModel = NeuralAI.createModel(AIProvider.OPENAI, {
      apiKey: process.env.OPENAI_API_KEY || "your-openai-api-key",
      model: "gpt-3.5-turbo",
    });

    const openaiResponse = await openaiModel.generate({
      prompt: "Explain what a neural network is in 3 sentences.",
      systemPrompt:
        "You are a helpful AI assistant that provides concise explanations.",
    });

    console.log("OpenAI Response:", openaiResponse.text);
    console.log("Token Usage:", openaiResponse.usage);
    console.log("\n----------------------------\n");

    // Example with Google
    console.log("Google AI Example:");
    const googleModel = NeuralAI.createModel(AIProvider.GOOGLE, {
      apiKey: process.env.GOOGLE_API_KEY || "your-google-api-key",
      model: "gemini-pro",
    });

    const googleResponse = await googleModel.generate({
      prompt: "Explain what machine learning is in 3 sentences.",
    });

    console.log("Google Response:", googleResponse.text);
    console.log("\n----------------------------\n");

    // Example with Streaming
    console.log("Streaming Example:");
    console.log("Streaming response: ");

    const stream = openaiModel.stream({
      prompt: "Write a short haiku about artificial intelligence.",
    });

    process.stdout.write("Response: ");
    for await (const chunk of stream) {
      process.stdout.write(chunk);
    }
    console.log("\n");
  } catch (error) {
    console.error("Error:", error);
  }
}

main();
