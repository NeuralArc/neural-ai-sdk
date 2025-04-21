import { AIProvider, AIModelRequest, FunctionDefinition } from "../src/types";
import { NeuralAI } from "../src";
import dotenv from "dotenv";

// Load environment variables from .env file
dotenv.config();

// Define function definitions
const weatherFunction: FunctionDefinition = {
  name: "getWeather",
  description: "Get the current weather for a location",
  parameters: {
    type: "object",
    properties: {
      location: {
        type: "string",
        description: "The city and state, e.g. San Francisco, CA",
      },
      unit: {
        type: "string",
        enum: ["celsius", "fahrenheit"],
        description: "The temperature unit to use",
      },
    },
    required: ["location"],
  },
};

const calculatorFunction: FunctionDefinition = {
  name: "calculator",
  description: "Perform a mathematical calculation",
  parameters: {
    type: "object",
    properties: {
      operation: {
        type: "string",
        enum: ["add", "subtract", "multiply", "divide"],
        description: "The operation to perform",
      },
      a: {
        type: "number",
        description: "The first operand",
      },
      b: {
        type: "number",
        description: "The second operand",
      },
    },
    required: ["operation", "a", "b"],
  },
};

// Function implementations
function handleWeatherFunction(args: any) {
  try {
    const { location, unit = "celsius" } =
      typeof args === "string" ? JSON.parse(args) : args;
    console.log(`🌤️ Getting weather for ${location} in ${unit}`);
    return {
      location,
      temperature: unit === "celsius" ? 22 : 72,
      condition: "Sunny",
      unit,
    };
  } catch (error) {
    console.error("Error parsing weather arguments:", error);
    console.error("Raw arguments:", args);
    return { error: "Failed to parse arguments" };
  }
}

function handleCalculatorFunction(args: any) {
  try {
    const { operation, a, b } =
      typeof args === "string" ? JSON.parse(args) : args;
    console.log(`🧮 Calculating ${a} ${operation} ${b}`);

    let result;
    switch (operation) {
      case "add":
        result = a + b;
        break;
      case "subtract":
        result = a - b;
        break;
      case "multiply":
        result = a * b;
        break;
      case "divide":
        result = a / b;
        break;
      default:
        throw new Error(`Unknown operation: ${operation}`);
    }

    return { result };
  } catch (error) {
    console.error("Error parsing calculator arguments:", error);
    console.error("Raw arguments:", args);
    return { error: "Failed to parse arguments" };
  }
}

// Test function calling with a specific provider
async function testProviderFunctionCalling(
  provider: AIProvider,
  modelName?: string
) {
  console.log(`\n🔍 Testing ${provider} provider with function calling`);
  console.log("=".repeat(50));

  try {
    // Create model with appropriate configuration for the provider
    const modelConfig: any = {
      temperature: 0.2, // Lower temperature for more deterministic function calling
    };

    // Set appropriate model name for each provider
    if (modelName) {
      modelConfig.model = modelName;
    } else {
      // Default models that are known to work well with function calling
      switch (provider) {
        case AIProvider.OPENAI:
          modelConfig.model = "gpt-3.5-turbo";
          break;
        case AIProvider.GOOGLE:
          modelConfig.model = "gemini-2.0-flash";
          break;
        case AIProvider.DEEPSEEK:
          modelConfig.model = "deepseek-chat";
          break;
        case AIProvider.OLLAMA:
          modelConfig.model = "llama2:7b"; // Needs to be installed in Ollama
          break;
        case AIProvider.HUGGINGFACE:
          modelConfig.model = "meta-llama/Llama-2-70b-chat-hf";
          break;
      }
    }

    console.log(`Using model: ${modelConfig.model || "default"}`);
    const model = NeuralAI.createModel(provider, modelConfig);

    // Test Auto Function Call
    console.log("\n📋 Testing Auto Function Call (Weather)");
    const weatherRequest: AIModelRequest = {
      prompt: "What's the current weather in Tokyo?",
      functions: [weatherFunction, calculatorFunction],
      functionCall: "auto", // Let the model decide
    };

    console.log(`Request: "${weatherRequest.prompt}"`);
    const weatherResponse = await model.generate(weatherRequest);

    console.log(`Response text: "${weatherResponse.text}"`);
    if (
      weatherResponse.functionCalls &&
      weatherResponse.functionCalls.length > 0
    ) {
      console.log("Function calls detected:");

      for (const call of weatherResponse.functionCalls) {
        console.log(`- Function: ${call.name}`);
        console.log(`  Arguments: ${call.arguments}`);

        if (call.name === "getWeather") {
          const result = handleWeatherFunction(call.arguments);
          console.log(`  Result: ${JSON.stringify(result, null, 2)}`);
        } else if (call.name === "calculator") {
          const result = handleCalculatorFunction(call.arguments);
          console.log(`  Result: ${JSON.stringify(result, null, 2)}`);
        }
      }
    } else {
      console.log("❌ No function calls detected");
    }

    // Test Specific Function Call
    console.log("\n📋 Testing Specific Function Call (Calculator)");
    const calcRequest: AIModelRequest = {
      prompt: "Calculate 123 times 456",
      functions: [weatherFunction, calculatorFunction],
      functionCall: { name: "calculator" }, // Force calculator function
    };

    console.log(`Request: "${calcRequest.prompt}"`);
    const calcResponse = await model.generate(calcRequest);

    console.log(`Response text: "${calcResponse.text}"`);
    if (calcResponse.functionCalls && calcResponse.functionCalls.length > 0) {
      console.log("Function calls detected:");

      for (const call of calcResponse.functionCalls) {
        console.log(`- Function: ${call.name}`);
        console.log(`  Arguments: ${call.arguments}`);

        if (call.name === "calculator") {
          const result = handleCalculatorFunction(call.arguments);
          console.log(`  Result: ${JSON.stringify(result, null, 2)}`);
        }
      }
    } else {
      console.log("❌ No function calls detected");
    }

    return true;
  } catch (error: any) {
    console.error(`❌ Error with ${provider}: ${error.message}`);
    if (error.response?.data) {
      console.error("API response data:", error.response.data);
    }
    return false;
  }
}

async function main() {
  console.log("🧠 Neural AI SDK - Function Calling Tests");
  console.log("======================================");

  // Get provider from command line if provided
  const args = process.argv.slice(2);
  const requestedProvider = args[0]?.toLowerCase();
  const modelName = args[1];

  let providersToTest: AIProvider[] = [];

  // Determine which providers to test
  if (requestedProvider) {
    switch (requestedProvider) {
      case "openai":
        providersToTest = [AIProvider.OPENAI];
        break;
      case "google":
        providersToTest = [AIProvider.GOOGLE];
        break;
      case "deepseek":
        providersToTest = [AIProvider.DEEPSEEK];
        break;
      case "ollama":
        providersToTest = [AIProvider.OLLAMA];
        break;
      case "huggingface":
        providersToTest = [AIProvider.HUGGINGFACE];
        break;
      case "all":
        providersToTest = [
          AIProvider.OPENAI,
          AIProvider.GOOGLE,
          AIProvider.DEEPSEEK,
          AIProvider.OLLAMA,
          AIProvider.HUGGINGFACE,
        ];
        break;
      default:
        console.error(`Unknown provider: ${requestedProvider}`);
        console.log(
          "Available providers: openai, google, deepseek, ollama, huggingface, all"
        );
        process.exit(1);
    }
  } else {
    // Default to OpenAI as it has the most mature function calling support
    providersToTest = [AIProvider.OPENAI];
  }

  console.log(`Testing providers: ${providersToTest.join(", ")}`);

  // Track successful providers
  const results: Record<string, boolean> = {};

  // Test each provider
  for (const provider of providersToTest) {
    results[provider] = await testProviderFunctionCalling(provider, modelName);
  }

  // Summary
  console.log("\n\n📊 Function Calling Test Results Summary");
  console.log("=====================================");
  for (const [provider, success] of Object.entries(results)) {
    console.log(`${provider}: ${success ? "✅ Passed" : "❌ Failed"}`);
  }
}

// Check if this file is being run directly
if (require.main === module) {
  main().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
  });
}
