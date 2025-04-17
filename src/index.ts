// Type exports
export {
  AIProvider,
  type AIModelConfig,
  type AIModelRequest,
  type AIModelResponse,
  type AIModel,
} from "./types";

// Model implementations
export { OpenAIModel } from "./models/openai-model";
export { GoogleModel } from "./models/google-model";
export { DeepSeekModel } from "./models/deepseek-model";
export { OllamaModel } from "./models/ollama-model";
export { HuggingFaceModel } from "./models/huggingface-model";

// Factory class for easier model creation
import { AIProvider, AIModelConfig } from "./types";
import { OpenAIModel } from "./models/openai-model";
import { GoogleModel } from "./models/google-model";
import { DeepSeekModel } from "./models/deepseek-model";
import { OllamaModel } from "./models/ollama-model";
import { HuggingFaceModel } from "./models/huggingface-model";

export class NeuralAI {
  /**
   * Create an AI model instance based on the provider and configuration
   * @param provider The AI provider to use
   * @param config Configuration for the AI model
   * @returns An instance of the specified AI model
   */
  static createModel(provider: AIProvider, config: AIModelConfig) {
    switch (provider) {
      case AIProvider.OPENAI:
        return new OpenAIModel(config);
      case AIProvider.GOOGLE:
        return new GoogleModel(config);
      case AIProvider.DEEPSEEK:
        return new DeepSeekModel(config);
      case AIProvider.OLLAMA:
        return new OllamaModel(config);
      case AIProvider.HUGGINGFACE:
        return new HuggingFaceModel(config);
      default:
        throw new Error(`Unsupported AI provider: ${provider}`);
    }
  }
}
