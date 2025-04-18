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

### Automatic Environment Variables Support

The SDK automatically loads environment variables from `.env` files when imported, so you don't need to manually configure dotenv. Simply create a `.env` file in your project root, and the API keys will be automatically detected:

```typescript
// No need to provide API keys in code if they're set in .env files
// No need to manually call require('dotenv').config()
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

### Using Multimodal Capabilities

The SDK supports multimodal capabilities for providers with vision-capable models. You can pass images to any model - the SDK will attempt to process them appropriately and provide helpful error messages if the model doesn't support vision inputs.

#### Simple Image + Text Example

```typescript
import { NeuralAI, AIProvider } from "neural-ai-sdk";

// Create an OpenAI model with vision capabilities
const openaiModel = NeuralAI.createModel(AIProvider.OPENAI, {
  model: "gpt-4o", // Model that supports vision
});

// Process an image with a text prompt
async function analyzeImage() {
  const response = await openaiModel.generate({
    prompt: "What's in this image? Please describe it in detail.",
    // The image can be a URL, local file path, or Buffer
    image: "https://example.com/image.jpg",
  });

  console.log(response.text);
}

analyzeImage();
```

#### Using Multiple Images

For more complex scenarios with multiple images or mixed content:

```typescript
import { NeuralAI, AIProvider } from "neural-ai-sdk";

// Create a Google model with multimodal support
const googleModel = NeuralAI.createModel(AIProvider.GOOGLE, {
  model: "gemini-2.0-flash",
});

async function compareImages() {
  const response = await googleModel.generate({
    prompt: "Compare these two images and tell me the differences:",
    content: [
      {
        type: "image",
        source: "https://example.com/image1.jpg",
      },
      {
        type: "text",
        text: "This is the first image.",
      },
      {
        type: "image",
        source: "https://example.com/image2.jpg",
      },
      {
        type: "text",
        text: "This is the second image.",
      },
    ],
  });

  console.log(response.text);
}

compareImages();
```

#### Supported Image Sources

The SDK handles various image sources:

- **URLs**: `"https://example.com/image.jpg"`
- **Local file paths**: `"/path/to/local/image.jpg"`
- **Buffers**: Direct image data as a Buffer object

The SDK automatically handles:

- Base64 encoding
- MIME type detection
- Image formatting for each provider's API

#### Multimodal Support Across Providers

All providers can attempt to process images - the SDK will automatically handle errors gracefully if a specific model doesn't support multimodal inputs.

| Provider    | Common Vision-Capable Models                        |
| ----------- | --------------------------------------------------- |
| OpenAI      | gpt-4o, gpt-4-vision                                |
| Google      | gemini-2.0-flash                                    |
| Ollama      | llama-3.2-vision, llama3-vision, bakllava, llava    |
| HuggingFace | llava, cogvlm, idefics, instructblip                |
| DeepSeek    | (Check provider documentation for supported models) |

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

The SDK automatically loads environment variables from `.env` files when imported, so you don't need to manually configure dotenv.

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

### Multimodal Streaming

You can also stream responses from multimodal prompts:

```typescript
import { NeuralAI, AIProvider } from "neural-ai-sdk";

const model = NeuralAI.createModel(AIProvider.OPENAI, {
  model: "gpt-4o",
});

async function streamImageAnalysis() {
  const stream = model.stream({
    prompt: "Describe this image in detail:",
    image: "https://example.com/image.jpg",
  });

  for await (const chunk of stream) {
    process.stdout.write(chunk);
  }
}

streamImageAnalysis();
```

## License

MIT
