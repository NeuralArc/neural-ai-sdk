import axios from "axios";
import {
  AIModelConfig,
  AIModelRequest,
  AIModelResponse,
  AIProvider,
} from "../types";
import { BaseModel } from "./base-model";

export class OllamaModel extends BaseModel {
  readonly provider = AIProvider.OLLAMA;
  private baseURL: string;

  constructor(config: AIModelConfig) {
    super(config);
    this.baseURL = config.baseURL || "http://localhost:11434/api";
  }

  async generate(request: AIModelRequest): Promise<AIModelResponse> {
    const config = this.mergeConfig(request.options);

    let prompt = request.prompt;

    // Add system prompt if provided
    if (request.systemPrompt) {
      prompt = `${request.systemPrompt}\n\n${prompt}`;
    }

    const response = await axios.post(`${this.baseURL}/generate`, {
      model: config.model || "llama2",
      prompt,
      temperature: config.temperature,
      num_predict: config.maxTokens,
      top_p: config.topP,
    });

    return {
      text: response.data.response,
      usage: {
        promptTokens: response.data.prompt_eval_count,
        completionTokens: response.data.eval_count,
        totalTokens: response.data.prompt_eval_count + response.data.eval_count,
      },
      raw: response.data,
    };
  }

  async *stream(
    request: AIModelRequest
  ): AsyncGenerator<string, void, unknown> {
    const config = this.mergeConfig(request.options);

    let prompt = request.prompt;

    if (request.systemPrompt) {
      prompt = `${request.systemPrompt}\n\n${prompt}`;
    }

    const response = await axios.post(
      `${this.baseURL}/generate`,
      {
        model: config.model || "llama2",
        prompt,
        temperature: config.temperature,
        num_predict: config.maxTokens,
        top_p: config.topP,
        stream: true,
      },
      {
        responseType: "stream",
      }
    );

    const reader = response.data;

    for await (const chunk of reader) {
      const lines = chunk.toString().split("\n").filter(Boolean);

      for (const line of lines) {
        try {
          const parsed = JSON.parse(line);
          if (parsed.response) {
            yield parsed.response;
          }
        } catch (error) {
          console.error("Error parsing Ollama stream data:", error);
        }
      }
    }
  }
}
