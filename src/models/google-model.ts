import { GoogleGenerativeAI, GenerativeModel } from "@google/generative-ai";
import {
  AIModelConfig,
  AIModelRequest,
  AIModelResponse,
  AIProvider,
} from "../types";
import { BaseModel } from "./base-model";
import { getApiKey } from "../utils";
import { processImage } from "../utils/image-utils";

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

    const content = await this.formatMultiModalContent(request);
    const result = await model.generateContent(content);
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

    const content = await this.formatMultiModalContent(request);
    const result = await model.generateContentStream(content);

    for await (const chunk of result.stream) {
      const text = chunk.text();
      if (text) {
        yield text;
      }
    }
  }

  /**
   * Format content for Google's Gemini API, handling both text and images
   */
  private async formatMultiModalContent(
    request: AIModelRequest
  ): Promise<any[]> {
    const parts: any[] = [];

    // Add system prompt if provided
    if (request.systemPrompt) {
      parts.push({ text: request.systemPrompt });
    }

    // Add main prompt text
    if (request.prompt) {
      parts.push({ text: request.prompt });
    }

    // Process structured content array if provided
    if (request.content) {
      for (const item of request.content) {
        if (item.type === "text") {
          parts.push({ text: item.text });
        } else if (item.type === "image") {
          // Process image and add to parts
          const { base64, mimeType } = await processImage(item.source);
          parts.push({
            inlineData: {
              data: base64,
              mimeType: mimeType,
            },
          });
        }
      }
    }

    // Process single image if provided via convenience property
    if (request.image) {
      const { base64, mimeType } = await processImage(request.image);
      parts.push({
        inlineData: {
          data: base64,
          mimeType: mimeType,
        },
      });
    }

    return parts;
  }
}
