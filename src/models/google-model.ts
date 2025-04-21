import { GoogleGenerativeAI, GenerativeModel } from "@google/generative-ai";
import {
  AIModelConfig,
  AIModelRequest,
  AIModelResponse,
  AIProvider,
  FunctionCall,
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

    // Create base model configuration with Gemini 2.0 models
    const modelConfig: any = {
      model: config.model || "gemini-2.0-flash", // Using 2.0 models as default
      generationConfig: {
        temperature: config.temperature,
        maxOutputTokens: config.maxTokens,
        topP: config.topP,
      },
    };

    const model = this.client.getGenerativeModel(modelConfig);

    // Handle function calling through prompt engineering for Gemini 2.0 models
    // as native function calling may not be fully supported in the same way
    if (request.functions && request.functions.length > 0) {
      try {
        // Create an enhanced prompt with function definitions
        let enhancedPrompt = request.prompt || "";

        if (request.systemPrompt) {
          enhancedPrompt = `${request.systemPrompt}\n\n${enhancedPrompt}`;
        }

        // Add function definitions to the prompt
        enhancedPrompt += `\n\nYou have access to the following functions:\n\`\`\`json\n${JSON.stringify(
          request.functions,
          null,
          2
        )}\n\`\`\`\n\n`;

        // Add specific instructions based on function call mode
        if (typeof request.functionCall === "object") {
          enhancedPrompt += `You MUST use the function: ${request.functionCall.name}\n`;
          enhancedPrompt += `Format your response as a function call using JSON in this exact format:\n`;
          enhancedPrompt += `{"name": "${request.functionCall.name}", "arguments": {...}}\n`;
          enhancedPrompt += `Don't include any explanations, just output the function call.\n`;
        } else if (request.functionCall === "auto") {
          enhancedPrompt += `If appropriate for the request, call one of these functions.\n`;
          enhancedPrompt += `Format your response as a function call using JSON in this exact format:\n`;
          enhancedPrompt += `{"name": "functionName", "arguments": {...}}\n`;
        }

        const result = await model.generateContent(enhancedPrompt);
        const response = result.response;
        const text = response.text();

        // Extract function calls from the response text
        const functionCalls = this.extractFunctionCallsFromText(text);

        return {
          text,
          functionCalls,
          raw: response,
        };
      } catch (error) {
        console.warn(
          "Function calling with prompt engineering failed for Gemini, falling back to text-only",
          error
        );
        // Fall back to regular text generation if prompt engineering fails
      }
    }

    // Regular text generation without function calling
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
    // For streaming, we'll keep it simpler as function calling with streaming
    // has additional complexities
    const config = this.mergeConfig(request.options);

    const model = this.client.getGenerativeModel({
      model: config.model || "gemini-2.0-flash", // Using 2.0 models as default
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
   * Extract function calls from text using regex patterns since we're using prompt engineering
   */
  private extractFunctionCallsFromText(
    text: string
  ): FunctionCall[] | undefined {
    try {
      if (!text) return undefined;

      const functionCalls = [];

      // Pattern 1: JSON object with name and arguments
      // Try full regex match first
      const jsonRegex =
        /\{[\s\n]*"name"[\s\n]*:[\s\n]*"([^"]+)"[\s\n]*,[\s\n]*"arguments"[\s\n]*:[\s\n]*(\{.*?\})\s*\}/gs;
      let match;

      while ((match = jsonRegex.exec(text)) !== null) {
        try {
          const name = match[1];
          const args = match[2];

          functionCalls.push({
            name,
            arguments: args,
          });
        } catch (e) {
          console.warn("Error parsing function call:", e);
        }
      }

      // Pattern 2: Markdown code blocks that might contain JSON
      const markdownRegex = /```(?:json)?\s*\n\s*(\{[\s\S]*?\})\s*\n```/gs;
      while ((match = markdownRegex.exec(text)) !== null) {
        try {
          const jsonBlock = match[1].trim();
          // Try to parse the JSON
          const jsonObj = JSON.parse(jsonBlock);

          if (jsonObj.name && (jsonObj.arguments || jsonObj.args)) {
            functionCalls.push({
              name: jsonObj.name,
              arguments: JSON.stringify(jsonObj.arguments || jsonObj.args),
            });
          }
        } catch (e) {
          // JSON might be malformed, try more aggressive parsing
          const nameMatch = match[1].match(/"name"\s*:\s*"([^"]+)"/);
          const argsMatch = match[1].match(/"arguments"\s*:\s*(\{[^}]*\})/);

          if (nameMatch && argsMatch) {
            try {
              // Try to fix and parse the arguments
              const argumentsStr = argsMatch[1].replace(/,\s*$/, "");
              const fixedArgs =
                argumentsStr + (argumentsStr.endsWith("}") ? "" : "}");

              functionCalls.push({
                name: nameMatch[1],
                arguments: fixedArgs,
              });
            } catch (e) {
              console.warn("Failed to fix JSON format:", e);
            }
          }
        }
      }

      // Pattern 3: If still no matches, try looser regex
      if (functionCalls.length === 0) {
        // Extract function name and args separately with more permissive patterns
        const nameMatch = text.match(/"name"\s*:\s*"([^"]+)"/);
        if (nameMatch) {
          const name = nameMatch[1];

          // Find arguments block, accounting for potential closing bracket issues
          const argsRegex = /"arguments"\s*:\s*(\{[^{]*?(?:\}|$))/;
          const argsMatch = text.match(argsRegex);

          if (argsMatch) {
            let args = argsMatch[1].trim();

            // Fix common JSON formatting issues
            if (!args.endsWith("}")) {
              args += "}";
            }

            // Clean up trailing commas which cause JSON.parse to fail
            args = args.replace(/,\s*}/g, "}");

            functionCalls.push({
              name,
              arguments: args,
            });
          }
        }
      }

      return functionCalls.length > 0 ? functionCalls : undefined;
    } catch (error) {
      console.warn("Error extracting function calls from text:", error);
      return undefined;
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
