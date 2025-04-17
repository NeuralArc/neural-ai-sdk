import { OpenAI } from "openai";
import {
  AIModelConfig,
  AIModelRequest,
  AIModelResponse,
  AIProvider,
} from "../types";
import { BaseModel } from "./base-model";

export class OpenAIModel extends BaseModel {
  readonly provider = AIProvider.OPENAI;
  private client: OpenAI;

  constructor(config: AIModelConfig) {
    super(config);
    if (!config.apiKey) {
      throw new Error("OpenAI API key is required");
    }

    this.client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL,
    });
  }

  async generate(request: AIModelRequest): Promise<AIModelResponse> {
    const config = this.mergeConfig(request.options);

    const messages = [];

    // Add system prompt if provided
    if (request.systemPrompt) {
      messages.push({
        role: "system" as const,
        content: request.systemPrompt,
      });
    }

    // Add user prompt
    messages.push({
      role: "user" as const,
      content: request.prompt,
    });

    const response = await this.client.chat.completions.create({
      model: config.model || "gpt-3.5-turbo",
      messages,
      temperature: config.temperature,
      max_tokens: config.maxTokens,
      top_p: config.topP,
    });

    return {
      text: response.choices[0].message.content || "",
      usage: {
        promptTokens: response.usage?.prompt_tokens,
        completionTokens: response.usage?.completion_tokens,
        totalTokens: response.usage?.total_tokens,
      },
      raw: response,
    };
  }

  async *stream(
    request: AIModelRequest
  ): AsyncGenerator<string, void, unknown> {
    const config = this.mergeConfig(request.options);

    const messages = [];

    if (request.systemPrompt) {
      messages.push({
        role: "system" as const,
        content: request.systemPrompt,
      });
    }

    messages.push({
      role: "user" as const,
      content: request.prompt,
    });

    const stream = await this.client.chat.completions.create({
      model: config.model || "gpt-3.5-turbo",
      messages,
      temperature: config.temperature,
      max_tokens: config.maxTokens,
      top_p: config.topP,
      stream: true,
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || "";
      if (content) {
        yield content;
      }
    }
  }
}
