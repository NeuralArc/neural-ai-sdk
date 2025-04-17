import axios from "axios";
import {
  AIModelConfig,
  AIModelRequest,
  AIModelResponse,
  AIProvider,
} from "../types";
import { BaseModel } from "./base-model";
import { getBaseUrl } from "../utils";
import { processImage } from "../utils/image-utils";

export class OllamaModel extends BaseModel {
  readonly provider = AIProvider.OLLAMA;
  private baseURL: string;

  constructor(config: AIModelConfig) {
    super(config);
    this.baseURL = getBaseUrl(
      config.baseURL,
      "OLLAMA_BASE_URL",
      "http://localhost:11434/api"
    );
  }

  async generate(request: AIModelRequest): Promise<AIModelResponse> {
    const config = this.mergeConfig(request.options);

    try {
      const payload = await this.createRequestPayload(request, config);
      const response = await axios.post(`${this.baseURL}/generate`, payload);

      return {
        text: response.data.response,
        usage: {
          promptTokens: response.data.prompt_eval_count,
          completionTokens: response.data.eval_count,
          totalTokens:
            response.data.prompt_eval_count + response.data.eval_count,
        },
        raw: response.data,
      };
    } catch (error: any) {
      // Enhance error message if it appears to be related to multimodal support
      if (
        error.response?.status === 400 &&
        (request.image || request.content) &&
        (error.response?.data?.error?.includes("image") ||
          error.response?.data?.error?.includes("multimodal") ||
          error.response?.data?.error?.includes("vision"))
      ) {
        throw new Error(
          `The model "${
            config.model || "default"
          }" doesn't support multimodal inputs. Try a vision-capable model like "llama-3.2-vision" or "llava". Original error: ${
            error.message
          }`
        );
      }
      throw error;
    }
  }

  async *stream(
    request: AIModelRequest
  ): AsyncGenerator<string, void, unknown> {
    const config = this.mergeConfig(request.options);

    try {
      const payload = await this.createRequestPayload(request, config, true);

      const response = await axios.post(`${this.baseURL}/generate`, payload, {
        responseType: "stream",
      });

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
    } catch (error: any) {
      // Enhance error message if it appears to be related to multimodal support
      if (
        error.response?.status === 400 &&
        (request.image || request.content) &&
        (error.response?.data?.error?.includes("image") ||
          error.response?.data?.error?.includes("multimodal") ||
          error.response?.data?.error?.includes("vision"))
      ) {
        throw new Error(
          `The model "${
            config.model || "default"
          }" doesn't support multimodal inputs. Try a vision-capable model like "llama-3.2-vision" or "llava". Original error: ${
            error.message
          }`
        );
      }
      throw error;
    }
  }

  /**
   * Creates the request payload for Ollama, handling multimodal content if provided
   */
  private async createRequestPayload(
    request: AIModelRequest,
    config: AIModelConfig,
    isStream: boolean = false
  ): Promise<any> {
    // Base payload
    const payload: any = {
      model: config.model || "llama2",
      temperature: config.temperature,
      num_predict: config.maxTokens,
      top_p: config.topP,
    };

    // Handle streaming
    if (isStream) {
      payload.stream = true;
    }

    // If there are any image inputs, use the messages format
    if (
      request.image ||
      (request.content && request.content.some((item) => item.type === "image"))
    ) {
      // Create a messages array for multimodal models (similar to OpenAI format)
      const messages = [];

      // Add system prompt if provided
      if (request.systemPrompt) {
        messages.push({
          role: "system",
          content: request.systemPrompt,
        });
      }

      // Create a user message with potentially multiple content parts
      const userMessage: any = { role: "user", content: [] };

      // Add the main prompt as text content
      if (request.prompt) {
        userMessage.content.push({
          type: "text",
          text: request.prompt,
        });
      }

      // Process structured content if available
      if (request.content) {
        for (const item of request.content) {
          if (item.type === "text") {
            userMessage.content.push({
              type: "text",
              text: item.text,
            });
          } else if (item.type === "image") {
            const { base64, mimeType } = await processImage(item.source);
            userMessage.content.push({
              type: "image",
              image: {
                data: base64,
                mimeType: mimeType,
              },
            });
          }
        }
      }

      // Handle the convenience image property
      if (request.image) {
        const { base64, mimeType } = await processImage(request.image);
        userMessage.content.push({
          type: "image",
          image: {
            data: base64,
            mimeType: mimeType,
          },
        });
      }

      // Add the user message
      messages.push(userMessage);

      // Set the messages in the payload
      payload.messages = messages;
    } else {
      // Traditional text-only format
      let prompt = request.prompt;

      // Add system prompt if provided
      if (request.systemPrompt) {
        prompt = `${request.systemPrompt}\n\n${prompt}`;
      }

      payload.prompt = prompt;
    }

    return payload;
  }
}
