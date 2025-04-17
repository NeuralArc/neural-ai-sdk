export interface AIModelConfig {
  apiKey?: string;
  baseURL?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
}

export enum AIProvider {
  OPENAI = "openai",
  GOOGLE = "google",
  DEEPSEEK = "deepseek",
  OLLAMA = "ollama",
  HUGGINGFACE = "huggingface",
}

export interface AIModelResponse {
  text: string;
  usage?: {
    promptTokens?: number;
    completionTokens?: number;
    totalTokens?: number;
  };
  raw?: any;
}

export interface AIModelRequest {
  prompt: string;
  systemPrompt?: string;
  options?: Partial<AIModelConfig>;
}

export interface AIModel {
  provider: AIProvider;
  generate(request: AIModelRequest): Promise<AIModelResponse>;
  stream(request: AIModelRequest): AsyncGenerator<string, void, unknown>;
}
