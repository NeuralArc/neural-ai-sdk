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
  apiKey: "your-openai-api-key", // Optional if OPENAI_API_KEY environment variable is set
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

### Environment Variables Support

You can provide API keys and base URLs through environment variables instead of directly in code:

```typescript
// No need to provide API keys in code if they're set as environment variables
const openaiModel = NeuralAI.createModel(AIProvider.OPENAI, {
  model: "gpt-4",
});

const googleModel = NeuralAI.createModel(AIProvider.GOOGLE, {
  model: "gemini-pro",
});
```

Available environment variables:

| Provider    | API Key Variable      | Base URL Variable (optional) |
| ----------- | --------------------- | ---------------------------- |
| OpenAI      | `OPENAI_API_KEY`      | -                            |
| Google      | `GOOGLE_API_KEY`      | -                            |
| DeepSeek    | `DEEPSEEK_API_KEY`    | `DEEPSEEK_BASE_URL`          |
| HuggingFace | `HUGGINGFACE_API_KEY` | `HUGGINGFACE_BASE_URL`       |
| Ollama      | -                     | `OLLAMA_BASE_URL`            |

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
  // baseURL is optional if OLLAMA_BASE_URL environment variable is set
  model: "llama2",
});

// Create HuggingFace model
const huggingfaceModel = NeuralAI.createModel(AIProvider.HUGGINGFACE, {
  // apiKey is optional if HUGGINGFACE_API_KEY environment variable is set
  model: "meta-llama/Llama-2-7b-chat-hf",
});

// Create DeepSeek model
const deepseekModel = NeuralAI.createModel(AIProvider.DEEPSEEK, {
  // apiKey is optional if DEEPSEEK_API_KEY environment variable is set
  model: "deepseek-chat",
});
```

## Environment Configuration

You can set up environment variables by:

1. Creating a `.env` file in your project root
2. Setting environment variables in your deployment platform
3. Setting them in your system environment

Example `.env` file:

```
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here
OLLAMA_BASE_URL=http://localhost:11434/api
```

Make sure to load environment variables from your `.env` file using a package like `dotenv`:

```javascript
require("dotenv").config();
```

## Configuration Options

All models accept the following configuration options:

| Option        | Description                                                                    |
| ------------- | ------------------------------------------------------------------------------ |
| `apiKey`      | API key for authentication (optional if set as environment variable)           |
| `baseURL`     | Base URL for the API (optional, uses environment variable or default endpoint) |
| `model`       | The model to use (optional, each provider has a default)                       |
| `temperature` | Controls randomness (0.0 to 1.0)                                               |
| `maxTokens`   | Maximum number of tokens to generate                                           |
| `topP`        | Nucleus sampling parameter                                                     |

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
