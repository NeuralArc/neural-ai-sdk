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

// Define content types for multimodal support
export type ContentType = "text" | "image";

export interface TextContent {
  type: "text";
  text: string;
}

export interface ImageContent {
  type: "image";
  source: string | Buffer; // URL, local path, or Buffer
}

export type Content = TextContent | ImageContent;

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
  // Add multimodal content support
  content?: Content[];
  // For simple image input (convenience method)
  image?: string | Buffer;
}

export interface AIModel {
  provider: AIProvider;
  generate(request: AIModelRequest): Promise<AIModelResponse>;
  stream(request: AIModelRequest): AsyncGenerator<string, void, unknown>;
}
