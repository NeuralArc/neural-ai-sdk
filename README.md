# Neural AI SDK

A unified JavaScript/TypeScript SDK for interacting with various AI LLM providers. This SDK allows you to integrate multiple AI models from different organizations through a single consistent interface.

## Supported AI Providers

- OpenAI (GPT models)
- Google (Gemini models)
- DeepSeek
- Ollama (local models)
- HuggingFace

## Installation

```bash
npm install neural-ai-sdk
```

## Usage

### Basic Example

```typescript
import { NeuralAI, AIProvider } from "neural-ai-sdk";

// Create an OpenAI model
const openaiModel = NeuralAI.createModel(AIProvider.OPENAI, {
  apiKey: "your-openai-api-key",
  model: "gpt-4",
});

// Generate a response
async function generateResponse() {
  const response = await openaiModel.generate({
    prompt: "What is artificial intelligence?",
    systemPrompt: "You are a helpful AI assistant.",
  });

  console.log(response.text);
}

generateResponse();
```

### Using Streaming

```typescript
import { NeuralAI, AIProvider } from "neural-ai-sdk";

// Create a Google model
const googleModel = NeuralAI.createModel(AIProvider.GOOGLE, {
  apiKey: "your-google-api-key",
  model: "gemini-pro",
});

// Stream a response
async function streamResponse() {
  const stream = googleModel.stream({
    prompt: "Write a short story about AI.",
  });

  for await (const chunk of stream) {
    process.stdout.write(chunk);
  }
}

streamResponse();
```

### Working With Different Providers

```typescript
import { NeuralAI, AIProvider } from "neural-ai-sdk";

// Create Ollama model (for local inference)
const ollamaModel = NeuralAI.createModel(AIProvider.OLLAMA, {
  baseURL: "http://localhost:11434/api", // Default Ollama API URL
  model: "llama2",
});

// Create HuggingFace model
const huggingfaceModel = NeuralAI.createModel(AIProvider.HUGGINGFACE, {
  apiKey: "your-huggingface-token",
  model: "meta-llama/Llama-2-7b-chat-hf",
});

// Create DeepSeek model
const deepseekModel = NeuralAI.createModel(AIProvider.DEEPSEEK, {
  apiKey: "your-deepseek-api-key",
  model: "deepseek-chat",
});
```

## Configuration Options

All models accept the following configuration options:

| Option        | Description                                                                   |
| ------------- | ----------------------------------------------------------------------------- |
| `apiKey`      | API key for authentication (required for most providers except Ollama)        |
| `baseURL`     | Base URL for the API (optional, defaults to provider's standard API endpoint) |
| `model`       | The model to use (optional, each provider has a default)                      |
| `temperature` | Controls randomness (0.0 to 1.0)                                              |
| `maxTokens`   | Maximum number of tokens to generate                                          |
| `topP`        | Nucleus sampling parameter                                                    |

## Using Request Options

You can provide options at the time of the request that override the model's default configuration:

```typescript
const response = await openaiModel.generate({
  prompt: "Explain quantum computing",
  options: {
    temperature: 0.7,
    maxTokens: 500,
  },
});
```

## Advanced Usage

### Access Raw API Responses

Each response includes a `raw` property with the full response data from the provider:

```typescript
const response = await openaiModel.generate({
  prompt: "Summarize machine learning",
});

// Access the raw response data
console.log(response.raw);
```

### Response Usage Information

When available, you can access token usage information:

```typescript
const response = await openaiModel.generate({
  prompt: "Explain neural networks",
});

console.log(`Prompt tokens: ${response.usage?.promptTokens}`);
console.log(`Completion tokens: ${response.usage?.completionTokens}`);
console.log(`Total tokens: ${response.usage?.totalTokens}`);
```

## License

MIT
