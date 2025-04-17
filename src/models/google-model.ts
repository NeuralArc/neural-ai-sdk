import { GoogleGenerativeAI, GenerativeModel } from "@google/generative-ai";
import {
  AIModelConfig,
  AIModelRequest,
  AIModelResponse,
  AIProvider,
} from "../types";
import { BaseModel } from "./base-model";
import { getApiKey } from "../utils";

export class GoogleModel extends BaseModel {
  readonly provider = AIProvider.GOOGLE;
  private client: GoogleGenerativeAI;

  constructor(config: AIModelConfig) {
    super(config);
    const apiKey = getApiKey(config.apiKey, "GOOGLE_API_KEY", "Google");
    this.client = new GoogleGenerativeAI(apiKey);
  }

  async generate(request: AIModelRequest): Promise<AIModelResponse> {
    const config = this.mergeConfig(request.options);
    const model = this.client.getGenerativeModel({
      model: config.model || "gemini-2.0-flash", // Updated default model
      generationConfig: {
        temperature: config.temperature,
        maxOutputTokens: config.maxTokens,
        topP: config.topP,
      },
    });

    const prompt = this.formatPrompt(request);
    const result = await model.generateContent(prompt);
    const response = result.response;

    return {
      text: response.text(),
      raw: response,
    };
  }

  async *stream(
    request: AIModelRequest
  ): AsyncGenerator<string, void, unknown> {
    const config = this.mergeConfig(request.options);
    const model = this.client.getGenerativeModel({
      model: config.model || "gemini-2.0-flash", // Updated default model
      generationConfig: {
        temperature: config.temperature,
        maxOutputTokens: config.maxTokens,
        topP: config.topP,
      },
    });

    const prompt = this.formatPrompt(request);
    const result = await model.generateContentStream(prompt);

    for await (const chunk of result.stream) {
      const text = chunk.text();
      if (text) {
        yield text;
      }
    }
  }

  private formatPrompt(request: AIModelRequest): string[] {
    const parts: string[] = [];

    if (request.systemPrompt) {
      parts.push(request.systemPrompt);
    }

    parts.push(request.prompt);

    return parts;
  }
}
