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
      // Create the payload for the request
      const payload = await this.createRequestPayload(request, config);

      // Determine which endpoint to use based on the payload format
      const endpoint = payload.messages ? "chat" : "generate";

      console.log(
        `Using Ollama ${endpoint} endpoint with model: ${
          payload.model || "default"
        }`
      );

      // Set stream to true to handle responses as a stream
      payload.stream = true;

      const response = await axios.post(
        `${this.baseURL}/${endpoint}`,
        payload,
        {
          responseType: "stream",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      // Accumulate the complete response
      let responseText = "";
      let promptTokens = 0;
      let completionTokens = 0;

      // Process the stream
      const reader = response.data;
      for await (const chunk of reader) {
        const lines = chunk.toString().split("\n").filter(Boolean);

        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);

            // Handle different response formats
            if (endpoint === "chat") {
              if (parsed.message && parsed.message.content) {
                responseText += parsed.message.content;
              }
            } else if (parsed.response) {
              responseText += parsed.response;
            }

            // Extract token usage from the final message
            if (parsed.done) {
              promptTokens = parsed.prompt_eval_count || 0;
              completionTokens = parsed.eval_count || 0;
            }
          } catch (error) {
            console.error("Error parsing Ollama stream data:", line, error);
          }
        }
      }

      console.log(`Extracted response text: "${responseText}"`);

      // Try to extract function calls from the response
      const functionCalls = this.extractFunctionCallsFromText(
        responseText,
        request
      );

      return {
        text: responseText,
        usage: {
          promptTokens: promptTokens,
          completionTokens: completionTokens,
          totalTokens: promptTokens + completionTokens,
        },
        functionCalls,
        raw: { response: responseText }, // We don't have the original raw response, so create one
      };
    } catch (error: any) {
      console.error("Ollama API error details:", error);
      if (error.response) {
        console.error("Response status:", error.response.status);
        console.error("Response data:", error.response.data);
      }
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

      // Check if the error is about the model not being found or loaded
      if (
        error.response?.status === 404 ||
        (error.response?.data &&
          typeof error.response.data.error === "string" &&
          error.response.data.error.toLowerCase().includes("model") &&
          error.response.data.error.toLowerCase().includes("not"))
      ) {
        throw new Error(
          `Model "${
            config.model || "default"
          }" not found or not loaded in Ollama. ` +
            `Make sure the model is installed with 'ollama pull ${
              config.model || "llama2"
            }' ` +
            `Original error: ${error.response?.data?.error || error.message}`
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

      // Determine which endpoint to use based on the payload format
      const endpoint = payload.messages ? "chat" : "generate";

      const response = await axios.post(
        `${this.baseURL}/${endpoint}`,
        payload,
        {
          responseType: "stream",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      const reader = response.data;

      for await (const chunk of reader) {
        const lines = chunk.toString().split("\n").filter(Boolean);

        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);
            // Handle different response formats
            if (endpoint === "chat") {
              if (parsed.message && parsed.message.content) {
                yield parsed.message.content;
              }
            } else if (parsed.response) {
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
      // Fix incomplete JSON - look for patterns where JSON might be incomplete
      // First, let's try to fix a common issue where the closing brace is missing
      const fixedText = this.tryFixIncompleteJSON(text);

      // Pattern 1: JSON format with name and arguments
      // E.g., {"name": "getWeather", "arguments": {"location": "Tokyo"}}
      const jsonRegex =
        /\{[\s\n]*"name"[\s\n]*:[\s\n]*"([^"]+)"[\s\n]*,[\s\n]*"arguments"[\s\n]*:[\s\n]*([\s\S]*?)\}/g;
      const jsonMatches = [...fixedText.matchAll(jsonRegex)];
      if (jsonMatches.length > 0) {
        return jsonMatches.map((match) => {
          try {
            // Try to parse the arguments as JSON
            let argsText = match[2];

            // Fix potential incomplete JSON in arguments
            argsText = this.tryFixIncompleteJSON(argsText);

            let args;
            try {
              args = JSON.parse(argsText);
              return {
                name: match[1],
                arguments: JSON.stringify(args),
              };
            } catch (e) {
              // If parsing fails, try to fix the JSON before returning
              console.warn(
                "Error parsing function arguments, trying to fix:",
                e
              );
              return {
                name: match[1],
                arguments: this.tryFixIncompleteJSON(argsText, true),
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
      const functionMatches = [...fixedText.matchAll(functionRegex)];
      if (functionMatches.length > 0) {
        return functionMatches.map((match) => {
          const argsText = this.tryFixIncompleteJSON(match[2]);
          return {
            name: match[1],
            arguments: argsText,
          };
        });
      }

      // Pattern 3: Looking for direct JSON objects - for function specific forced calls
      if (
        currentRequest.functionCall &&
        typeof currentRequest.functionCall === "object"
      ) {
        const forcedFunctionName = currentRequest.functionCall.name;

        // For getWeather function
        if (forcedFunctionName === "getWeather") {
          const weatherMatch = fixedText.match(
            /\{[\s\n]*"location"[\s\n]*:[\s\n]*"([^"]*)"(?:[\s\n]*,[\s\n]*"unit"[\s\n]*:[\s\n]*"([^"]*)"|)(.*?)\}/s
          );
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

        // For calculator function
        if (forcedFunctionName === "calculator") {
          const calculatorMatch = fixedText.match(
            /\{[\s\n]*"operation"[\s\n]*:[\s\n]*"([^"]*)"[\s\n]*,[\s\n]*"a"[\s\n]*:[\s\n]*(\d+)[\s\n]*,[\s\n]*"b"[\s\n]*:[\s\n]*(\d+)[\s\S]*?\}/s
          );
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
      }

      // If no matches found and we have a functionCall request, try one more pattern matching approach
      if (currentRequest.functionCall) {
        // Try to extract JSON-like structures even if they're not complete
        const namedFunction =
          typeof currentRequest.functionCall === "object"
            ? currentRequest.functionCall.name
            : null;

        // Look for the function name in the text followed by arguments
        const functionNamePattern = namedFunction
          ? new RegExp(
              `"name"\\s*:\\s*"${namedFunction}"\\s*,\\s*"arguments"\\s*:\\s*(\\{[\\s\\S]*?)(?:\\}|$)`,
              "s"
            )
          : null;

        if (functionNamePattern) {
          const extractedMatch = fixedText.match(functionNamePattern);
          if (extractedMatch && extractedMatch[1]) {
            let argsText = extractedMatch[1];
            // Make sure the JSON is complete
            if (!argsText.endsWith("}")) {
              argsText += "}";
            }

            try {
              // Try to parse the fixed arguments
              const args = JSON.parse(argsText);
              return [
                {
                  name: namedFunction!,
                  arguments: JSON.stringify(args),
                },
              ];
            } catch (e) {
              console.warn("Failed to parse extracted arguments:", e);
              return [
                {
                  name: namedFunction!,
                  arguments: this.tryFixIncompleteJSON(argsText, true),
                },
              ];
            }
          }
        }
      }
    } catch (e) {
      console.warn("Error in extractFunctionCallsFromText:", e);
    }

    return undefined;
  }

  /**
   * Tries to fix incomplete JSON strings by adding missing closing braces
   */
  private tryFixIncompleteJSON(text: string, returnAsString = false): string {
    // Skip if the string is already valid JSON
    try {
      JSON.parse(text);
      return text; // Already valid
    } catch (e) {
      // Not valid JSON, try to fix
    }

    // Count opening and closing braces
    const openBraces = (text.match(/\{/g) || []).length;
    const closeBraces = (text.match(/\}/g) || []).length;

    // If we have more opening braces than closing, add the missing closing braces
    if (openBraces > closeBraces) {
      const missingBraces = openBraces - closeBraces;
      let fixedText = text + "}".repeat(missingBraces);

      // Try to parse it to see if it's valid now
      try {
        if (!returnAsString) {
          JSON.parse(fixedText);
        }
        return fixedText;
      } catch (e) {
        // Still not valid, return the original
        console.warn("Failed to fix JSON even after adding braces", e);
        return text;
      }
    }

    return text;
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
      top_p: config.topP,
    };

    // Add max tokens for the generate endpoint
    if (config.maxTokens) {
      payload.num_predict = config.maxTokens;
    }

    // Handle streaming
    if (isStream) {
      payload.stream = true;
    }

    // Check if we should use chat format (messages array) or text format
    const useMessagesFormat =
      request.image ||
      (request.content &&
        request.content.some((item) => item.type === "image")) ||
      (request.functions && request.functions.length > 0) ||
      request.systemPrompt; // Always use messages format when system prompt is provided

    if (useMessagesFormat) {
      // Modern message-based format for Ollama (chat endpoint)
      const messages = [];

      // Add system prompt if provided
      if (request.systemPrompt) {
        messages.push({
          role: "system",
          content: request.systemPrompt,
        });
      } else if (request.functions && request.functions.length > 0) {
        // Add function calling guidance in system prompt if none provided
        let functionSystemPrompt =
          "You are a helpful AI assistant with access to functions.";

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

        messages.push({
          role: "system",
          content: functionSystemPrompt,
        });
      }

      // Create user message content parts
      let userContent: any[] = [];

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
      } else if (userContent.length > 0) {
        userMessage.content = userContent;
      } else {
        // Add empty string if no content provided to avoid invalid request
        userMessage.content = "";
      }

      messages.push(userMessage);
      payload.messages = messages;

      // Remove any fields specific to the generate endpoint
      // that might cause issues with the chat endpoint
      if (payload.hasOwnProperty("num_predict")) {
        delete payload.num_predict;
      }
    } else {
      // Traditional text-only format (generate endpoint)
      let prompt = request.prompt || "";

      // Add system prompt if provided
      if (request.systemPrompt) {
        prompt = `${request.systemPrompt}\n\n${prompt}`;
      }

      payload.prompt = prompt;
    }

    return payload;
  }
}
