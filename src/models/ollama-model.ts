import axios from "axios";
import {
  AIModelConfig,
  AIModelRequest,
  AIModelResponse,
  AIProvider,
  FunctionCall,
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
      // Create and modify the payload for function calling
      const payload = await this.createRequestPayload(request, config);

      const response = await axios.post(`${this.baseURL}/generate`, payload, {
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.data || !response.data.response) {
        throw new Error("Invalid response from Ollama API");
      }

      // Try to extract function calls from the response
      const functionCalls = this.extractFunctionCallsFromText(
        response.data.response,
        request
      );

      return {
        text: response.data.response,
        usage: {
          promptTokens: response.data.prompt_eval_count,
          completionTokens: response.data.eval_count,
          totalTokens:
            response.data.prompt_eval_count + response.data.eval_count,
        },
        functionCalls,
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
        headers: {
          "Content-Type": "application/json",
        },
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
   * Extract function calls from text using various patterns
   */
  private extractFunctionCallsFromText(
    text: string,
    currentRequest: AIModelRequest
  ): FunctionCall[] | undefined {
    if (!text) return undefined;

    try {
      // Try multiple patterns for function calls

      // Pattern 1: JSON format with name and arguments
      // E.g., {"name": "getWeather", "arguments": {"location": "Tokyo"}}
      const jsonRegex =
        /\{[\s\n]*"name"[\s\n]*:[\s\n]*"([^"]+)"[\s\n]*,[\s\n]*"arguments"[\s\n]*:[\s\n]*([\s\S]*?)\}/g;
      const jsonMatches = [...text.matchAll(jsonRegex)];
      if (jsonMatches.length > 0) {
        return jsonMatches.map((match) => {
          try {
            // Try to parse the arguments as JSON
            const argsText = match[2];
            let args;
            try {
              args = JSON.parse(argsText);
              return {
                name: match[1],
                arguments: JSON.stringify(args),
              };
            } catch (e) {
              // If parsing fails, use the raw text
              return {
                name: match[1],
                arguments: argsText,
              };
            }
          } catch (e) {
            console.warn("Error parsing function call:", e);
            return {
              name: match[1],
              arguments: "{}",
            };
          }
        });
      }

      // Pattern 2: Function call pattern: functionName({"key": "value"})
      const functionRegex = /([a-zA-Z0-9_]+)\s*\(\s*(\{[\s\S]*?\})\s*\)/g;
      const functionMatches = [...text.matchAll(functionRegex)];
      if (functionMatches.length > 0) {
        return functionMatches.map((match) => ({
          name: match[1],
          arguments: match[2],
        }));
      }

      // Pattern 3: Look for more specific calculator patterns
      if (
        currentRequest.functionCall &&
        typeof currentRequest.functionCall === "object" &&
        currentRequest.functionCall.name === "calculator"
      ) {
        const calculatorRegex =
          /"?operation"?\s*:\s*"?([^",\s]+)"?,\s*"?a"?\s*:\s*(\d+),\s*"?b"?\s*:\s*(\d+)/;
        const calculatorMatch = text.match(calculatorRegex);
        if (calculatorMatch) {
          const operation = calculatorMatch[1];
          const a = parseInt(calculatorMatch[2]);
          const b = parseInt(calculatorMatch[3]);
          return [
            {
              name: "calculator",
              arguments: JSON.stringify({ operation, a, b }),
            },
          ];
        }
      }

      // Pattern 4: Look for more specific weather patterns
      if (
        currentRequest.functionCall &&
        typeof currentRequest.functionCall === "object" &&
        currentRequest.functionCall.name === "getWeather"
      ) {
        const weatherRegex =
          /"?location"?\s*:\s*"([^"]+)"(?:,\s*"?unit"?\s*:\s*"([^"]+)")?/;
        const weatherMatch = text.match(weatherRegex);
        if (weatherMatch) {
          const location = weatherMatch[1];
          const unit = weatherMatch[2] || "celsius";
          return [
            {
              name: "getWeather",
              arguments: JSON.stringify({ location, unit }),
            },
          ];
        }
      }
    } catch (e) {
      console.warn("Error in extractFunctionCallsFromText:", e);
    }

    return undefined;
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
      model: config.model || "llama3", // Updated default to a model that better supports function calling
      temperature: config.temperature,
      num_predict: config.maxTokens,
      top_p: config.topP,
    };

    // Handle streaming
    if (isStream) {
      payload.stream = true;
    }

    // Check if we should use chat format (messages array) or text format
    const useMessagesFormat =
      request.image ||
      (request.content &&
        request.content.some((item) => item.type === "image")) ||
      (request.functions && request.functions.length > 0);

    if (useMessagesFormat) {
      // Modern message-based format for Ollama
      const messages = [];

      // Add system prompt if provided
      if (request.systemPrompt) {
        messages.push({
          role: "system",
          content: request.systemPrompt,
        });
      }

      // Create user message content parts
      let userContent = [];

      // Add main text content
      if (request.prompt) {
        userContent.push({
          type: "text",
          text: request.prompt,
        });
      }

      // Add structured content
      if (request.content) {
        for (const item of request.content) {
          if (item.type === "text") {
            userContent.push({
              type: "text",
              text: item.text,
            });
          } else if (item.type === "image") {
            const { base64, mimeType } = await processImage(item.source);
            userContent.push({
              type: "image",
              image: {
                data: base64,
                mimeType: mimeType,
              },
            });
          }
        }
      }

      // Add simple image if provided
      if (request.image) {
        const { base64, mimeType } = await processImage(request.image);
        userContent.push({
          type: "image",
          image: {
            data: base64,
            mimeType: mimeType,
          },
        });
      }

      // Create the user message
      let userMessage: any = {
        role: "user",
      };

      // If we have a single text content, use string format
      if (userContent.length === 1 && userContent[0].type === "text") {
        userMessage.content = userContent[0].text;
      } else {
        userMessage.content = userContent;
      }

      messages.push(userMessage);

      // Add function calling data to system prompt
      if (request.functions && request.functions.length > 0) {
        // Create a system prompt for function calling
        let functionSystemPrompt = request.systemPrompt || "";

        // Add function definitions as JSON
        functionSystemPrompt += `\n\nAvailable functions:\n\`\`\`json\n${JSON.stringify(
          request.functions,
          null,
          2
        )}\n\`\`\`\n\n`;

        // Add instruction based on functionCall setting
        if (typeof request.functionCall === "object") {
          functionSystemPrompt += `You must call the function: ${request.functionCall.name}.\n`;
          functionSystemPrompt += `Format your response as a function call using this exact format:\n`;
          functionSystemPrompt += `{"name": "${request.functionCall.name}", "arguments": {...}}\n`;
        } else if (request.functionCall === "auto") {
          functionSystemPrompt += `Call one of these functions if appropriate for the user's request.\n`;
          functionSystemPrompt += `Format your response as a function call using this exact format:\n`;
          functionSystemPrompt += `{"name": "functionName", "arguments": {...}}\n`;
        }

        // Replace or add the system message
        if (messages.length > 0 && messages[0].role === "system") {
          messages[0].content = functionSystemPrompt;
        } else {
          messages.unshift({
            role: "system",
            content: functionSystemPrompt,
          });
        }
      }

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
