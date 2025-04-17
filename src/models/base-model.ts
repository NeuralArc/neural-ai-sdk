import {
  AIModel,
  AIModelConfig,
  AIModelRequest,
  AIModelResponse,
  AIProvider,
} from "../types";

export abstract class BaseModel implements AIModel {
  protected config: AIModelConfig;
  abstract provider: AIProvider;

  constructor(config: AIModelConfig) {
    this.config = config;
  }

  abstract generate(request: AIModelRequest): Promise<AIModelResponse>;
  abstract stream(
    request: AIModelRequest
  ): AsyncGenerator<string, void, unknown>;

  protected mergeConfig(options?: Partial<AIModelConfig>): AIModelConfig {
    return {
      ...this.config,
      ...(options || {}),
    };
  }
}
