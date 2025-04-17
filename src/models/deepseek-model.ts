import axios from "axios";
import {
  AIModelConfig,
  AIModelRequest,
  AIModelResponse,
  AIProvider,
} from "../types";
import { BaseModel } from "./base-model";

export class DeepSeekModel extends BaseModel {
  readonly provider = AIProvider.DEEPSEEK;
  private baseURL: string;

  constructor(config: AIModelConfig) {
    super(config);
    if (!config.apiKey) {
      throw new Error("DeepSeek API key is required");
    }

    this.baseURL = config.baseURL || "https://api.deepseek.com/v1";
  }

  async generate(request: AIModelRequest): Promise<AIModelResponse> {
    const config = this.mergeConfig(request.options);

    const messages = [];

    if (request.systemPrompt) {
      messages.push({
        role: "system",
        content: request.systemPrompt,
      });
    }

    messages.push({
      role: "user",
      content: request.prompt,
    });

    const response = await axios.post(
      `${this.baseURL}/chat/completions`,
      {
        model: config.model || "deepseek-chat",
        messages,
        temperature: config.temperature,
        max_tokens: config.maxTokens,
        top_p: config.topP,
      },
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${config.apiKey}`,
        },
      }
    );

    return {
      text: response.data.choices[0].message.content,
      usage: {
        promptTokens: response.data.usage?.prompt_tokens,
        completionTokens: response.data.usage?.completion_tokens,
        totalTokens: response.data.usage?.total_tokens,
      },
      raw: response.data,
    };
  }

  async *stream(
    request: AIModelRequest
  ): AsyncGenerator<string, void, unknown> {
    const config = this.mergeConfig(request.options);

    const messages = [];

    if (request.systemPrompt) {
      messages.push({
        role: "system",
        content: request.systemPrompt,
      });
    }

    messages.push({
      role: "user",
      content: request.prompt,
    });

    const response = await axios.post(
      `${this.baseURL}/chat/completions`,
      {
        model: config.model || "deepseek-chat",
        messages,
        temperature: config.temperature,
        max_tokens: config.maxTokens,
        top_p: config.topP,
        stream: true,
      },
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${config.apiKey}`,
        },
        responseType: "stream",
      }
    );

    const reader = response.data;

    for await (const chunk of reader) {
      const lines = chunk.toString().split("\n").filter(Boolean);

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") continue;

          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices[0]?.delta?.content;
            if (content) {
              yield content;
            }
          } catch (error) {
            console.error("Error parsing DeepSeek stream data:", error);
          }
        }
      }
    }
  }
}
