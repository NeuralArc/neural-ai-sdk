import axios from "axios";
import {
  AIModelConfig,
  AIModelRequest,
  AIModelResponse,
  AIProvider,
  FunctionCall,
} from "../types";
import { BaseModel } from "./base-model";
import { getApiKey, getBaseUrl } from "../utils";
import { processImage } from "../utils/image-utils";

export class HuggingFaceModel extends BaseModel {
  readonly provider = AIProvider.HUGGINGFACE;
  private baseURL: string;

  constructor(config: AIModelConfig) {
    super(config);
    const apiKey = getApiKey(
      config.apiKey,
      "HUGGINGFACE_API_KEY",
      "HuggingFace"
    );
    this.baseURL = getBaseUrl(
      config.baseURL,
      "HUGGINGFACE_BASE_URL",
      "https://api-inference.huggingface.co/models"
    );
  }

  async generate(request: AIModelRequest): Promise<AIModelResponse> {
    const config = this.mergeConfig(request.options);
    // Use a more accessible default model that doesn't require special permissions
    const model = config.model || "mistralai/Mistral-7B-Instruct-v0.2";

    try {
      // Try multimodal approach if images are present
      if (
        request.image ||
        (request.content &&
          request.content.some((item) => item.type === "image"))
      ) {
        return await this.generateWithImages(request, config, model);
      } else {
        return await this.generateTextOnly(request, config, model);
      }
    } catch (error: any) {
      // Enhance error messages for multimodal related errors
      if (
        (request.image || request.content) &&
        (error.response?.data?.includes("Content-Type") ||
          error.response?.status === 415 ||
          error.response?.data?.error?.includes("image") ||
          error.message?.includes("multimodal") ||
          error.message?.toLowerCase().includes("vision") ||
          error.message?.toLowerCase().includes("unsupported"))
      ) {
        let errorMessage = `Model "${model}" doesn't appear to support multimodal inputs properly.`;

        // Add more specific guidance based on error details
        if (error.response?.status === 415) {
          errorMessage +=
            " The model may require a different format for image inputs.";
        }

        // Include original error message for debugging
        errorMessage += ` Original error: ${
          error.response?.data || error.message
        }`;

        // Suggest known working models
        errorMessage +=
          " Try a different vision-capable model like 'llava-hf/llava-1.5-7b-hf' or check HuggingFace's documentation for this specific model.";

        throw new Error(errorMessage);
      } else if (error.response?.status === 403) {
        // Handle permission errors more specifically
        throw new Error(
          `Permission denied for model "${model}". Try using a different model with public access. Error: ${
            error.response?.data || error.message
          }`
        );
      }
      throw error;
    }
  }

  /**
   * Generate a text-only response
   */
  private async generateTextOnly(
    request: AIModelRequest,
    config: AIModelConfig,
    model: string
  ): Promise<AIModelResponse> {
    let fullPrompt = request.prompt;

    if (request.systemPrompt) {
      fullPrompt = `${request.systemPrompt}\n\n${fullPrompt}`;
    }

    // If functions are provided, enhance the prompt to handle function calling
    if (request.functions && request.functions.length > 0) {
      // Format the functions in a way the model can understand
      fullPrompt += `\n\nAVAILABLE FUNCTIONS:\n${JSON.stringify(
        request.functions,
        null,
        2
      )}\n\n`;

      // Add guidance based on function call setting
      if (typeof request.functionCall === "object") {
        fullPrompt += `You must call the function: ${request.functionCall.name}.\n`;
        fullPrompt += `Format your answer as a function call using JSON, like this:\n`;
        fullPrompt += `{"name": "${request.functionCall.name}", "arguments": {...}}\n`;
        fullPrompt += `Don't include any explanations, just output the function call.\n`;
      } else if (request.functionCall === "auto") {
        fullPrompt += `Call one of the available functions if appropriate. Format the function call as JSON, like this:\n`;
        fullPrompt += `{"name": "functionName", "arguments": {...}}\n`;
      }
    }

    const payload = {
      inputs: fullPrompt,
      parameters: {
        temperature: config.temperature || 0.1,
        max_new_tokens: config.maxTokens || 500,
        top_p: config.topP || 0.9,
        return_full_text: false,
      },
    };

    try {
      const response = await axios.post(`${this.baseURL}/${model}`, payload, {
        headers: {
          Authorization: `Bearer ${
            config.apiKey ||
            getApiKey(config.apiKey, "HUGGINGFACE_API_KEY", "HuggingFace")
          }`,
          "Content-Type": "application/json",
        },
      });

      // Parse the response
      let text = "";
      if (Array.isArray(response.data)) {
        text = response.data[0]?.generated_text || "";
      } else if (response.data.generated_text) {
        text = response.data.generated_text;
      } else {
        text = JSON.stringify(response.data);
      }

      // Extract function calls from the response
      const functionCalls = this.extractFunctionCallsFromText(text);

      return {
        text,
        functionCalls,
        raw: response.data,
      };
    } catch (error) {
      console.error("Error generating with HuggingFace model:", error);
      throw error;
    }
  }

  /**
   * Generate a response using multimodal inputs (text + images)
   */
  private async generateWithImages(
    request: AIModelRequest,
    config: AIModelConfig,
    model: string
  ): Promise<AIModelResponse> {
    // Some HF models expect different input formats, try various formats one by one
    const errors = [];

    // Format 1: Nested inputs object with text and image
    try {
      return await this.generateWithNestedFormat(request, config, model);
    } catch (error: any) {
      errors.push(error);
    }

    // Format 2: Plain inputs with string prompt and image in the main object
    try {
      return await this.generateWithFlatFormat(request, config, model);
    } catch (error: any) {
      errors.push(error);
    }

    // Format 3: Try multipart form data as last resort
    try {
      return await this.generateWithMultipartForm(request, config, model);
    } catch (error: any) {
      errors.push(error);
    }

    // If we get here, all formats failed, throw an enhanced error
    const errorMessage = `Model "${model}" doesn't appear to support multimodal inputs in any of the attempted formats. Try a different vision-capable model like 'llava-hf/llava-1.5-7b-hf'. Errors: ${errors
      .map((e) => e.message || e)
      .join("; ")}`;
    throw new Error(errorMessage);
  }

  /**
   * Extract function calls from text using pattern matching
   * This is more robust than the previous implementation and handles various formats
   */
  private extractFunctionCallsFromText(
    text: string
  ): FunctionCall[] | undefined {
    if (!text) return undefined;

    try {
      const functionCalls = [];

      // Pattern 1: Clean JSON function call format
      // Example: {"name": "functionName", "arguments": {...}}
      const jsonRegex =
        /\{[\s\n]*"name"[\s\n]*:[\s\n]*"([^"]+)"[\s\n]*,[\s\n]*"arguments"[\s\n]*:[\s\n]*([\s\S]*?)\}/g;
      let match;
      while ((match = jsonRegex.exec(text)) !== null) {
        try {
          // Try to parse the arguments part as JSON
          const name = match[1];
          let args = match[2].trim();

          // Check if args is already a valid JSON string
          try {
            JSON.parse(args);
            functionCalls.push({
              name,
              arguments: args,
            });
          } catch (e) {
            // If not valid JSON, try to extract the JSON object
            const argsMatch = args.match(/\{[\s\S]*\}/);
            if (argsMatch) {
              functionCalls.push({
                name,
                arguments: argsMatch[0],
              });
            } else {
              functionCalls.push({
                name,
                arguments: "{}",
              });
            }
          }
        } catch (e) {
          console.warn("Error parsing function call:", e);
        }
      }

      // Pattern 2: Function-like syntax
      // Example: functionName({param1: "value", param2: 123})
      const functionRegex = /([a-zA-Z0-9_]+)\s*\(\s*(\{[\s\S]*?\})\s*\)/g;
      while ((match = functionRegex.exec(text)) !== null) {
        functionCalls.push({
          name: match[1],
          arguments: match[2],
        });
      }

      // Pattern 3: Markdown code block with JSON
      // Example: ```json\n{"name": "functionName", "arguments": {...}}\n```
      const markdownRegex = /```(?:json)?\s*\n\s*(\{[\s\S]*?\})\s*\n```/g;
      while ((match = markdownRegex.exec(text)) !== null) {
        try {
          const jsonObj = JSON.parse(match[1]);
          if (jsonObj.name && (jsonObj.arguments || jsonObj.args)) {
            functionCalls.push({
              name: jsonObj.name,
              arguments: JSON.stringify(jsonObj.arguments || jsonObj.args),
            });
          }
        } catch (e) {
          // Ignore parse errors for markdown blocks
        }
      }

      return functionCalls.length > 0 ? functionCalls : undefined;
    } catch (e) {
      console.warn("Error in extractFunctionCallsFromText:", e);
      return undefined;
    }
  }

  /**
   * Try generating with nested inputs format (common in newer models)
   */
  private async generateWithNestedFormat(
    request: AIModelRequest,
    config: AIModelConfig,
    model: string
  ): Promise<AIModelResponse> {
    const prompt = request.systemPrompt
      ? `${request.systemPrompt}\n\n${request.prompt}`
      : request.prompt;

    // Handle function calling by adding function definitions to the prompt
    let enhancedPrompt = prompt;
    if (request.functions && request.functions.length > 0) {
      const functionText = JSON.stringify(
        { functions: request.functions },
        null,
        2
      );
      enhancedPrompt = `${enhancedPrompt}\n\nAvailable functions:\n\`\`\`json\n${functionText}\n\`\`\`\n\n`;

      if (typeof request.functionCall === "object") {
        enhancedPrompt += `Please call the function: ${request.functionCall.name}\n\n`;
      } else if (request.functionCall === "auto") {
        enhancedPrompt += "Call the appropriate function if needed.\n\n";
      }
    }

    let payload: any = {
      inputs: {
        text: enhancedPrompt,
      },
      parameters: {
        temperature: config.temperature || 0.1,
        max_new_tokens: config.maxTokens || 500,
        top_p: config.topP || 0.9,
        return_full_text: false,
      },
    };

    // Process the convenience 'image' property
    if (request.image) {
      const { base64 } = await processImage(request.image);
      payload.inputs.image = base64;
    }

    // Process content array if provided
    if (request.content) {
      // Initialize images array if multiple images
      const hasMultipleImages =
        request.content.filter((item) => item.type === "image").length > 1;
      if (hasMultipleImages) {
        payload.inputs.images = [];
      }

      for (const item of request.content) {
        if (item.type === "image") {
          const { base64 } = await processImage(item.source);
          if (hasMultipleImages) {
            payload.inputs.images.push(base64);
          } else {
            payload.inputs.image = base64;
          }
        }
        // Text content is already included in the prompt
      }
    }

    const response = await axios.post(`${this.baseURL}/${model}`, payload, {
      headers: {
        Authorization: `Bearer ${
          config.apiKey ||
          getApiKey(config.apiKey, "HUGGINGFACE_API_KEY", "HuggingFace")
        }`,
        "Content-Type": "application/json",
      },
    });

    // Parse response with function call extraction
    return this.parseResponseWithFunctionCalls(response);
  }

  /**
   * Try generating with flat inputs format (common in some models)
   */
  private async generateWithFlatFormat(
    request: AIModelRequest,
    config: AIModelConfig,
    model: string
  ): Promise<AIModelResponse> {
    const prompt = request.systemPrompt
      ? `${request.systemPrompt}\n\n${request.prompt}`
      : request.prompt;

    // Handle function calling by adding function definitions to the prompt
    let enhancedPrompt = prompt;
    if (request.functions && request.functions.length > 0) {
      const functionText = JSON.stringify(
        { functions: request.functions },
        null,
        2
      );
      enhancedPrompt = `${enhancedPrompt}\n\nAvailable functions:\n\`\`\`json\n${functionText}\n\`\`\`\n\n`;

      if (typeof request.functionCall === "object") {
        enhancedPrompt += `Please call the function: ${request.functionCall.name}\n\n`;
      } else if (request.functionCall === "auto") {
        enhancedPrompt += "Call the appropriate function if needed.\n\n";
      }
    }

    // Some models expect a flat structure with inputs as a string
    let payload: any = {
      inputs: enhancedPrompt,
      parameters: {
        temperature: config.temperature || 0.1,
        max_new_tokens: config.maxTokens || 500,
        top_p: config.topP || 0.9,
        return_full_text: false,
      },
    };

    // For single image, add it directly to the payload
    if (request.image) {
      const { base64 } = await processImage(request.image);
      payload.image = base64; // At top level, not in inputs
    }

    // Process only the first image from content if available and no direct image
    if (!request.image && request.content) {
      const imageContent = request.content.find(
        (item) => item.type === "image"
      );
      if (imageContent) {
        const { base64 } = await processImage(imageContent.source);
        payload.image = base64; // At top level, not in inputs
      }
    }

    const response = await axios.post(`${this.baseURL}/${model}`, payload, {
      headers: {
        Authorization: `Bearer ${
          config.apiKey ||
          getApiKey(config.apiKey, "HUGGINGFACE_API_KEY", "HuggingFace")
        }`,
        "Content-Type": "application/json",
      },
    });

    // Parse response with function call extraction
    return this.parseResponseWithFunctionCalls(response);
  }

  /**
   * Helper to parse HuggingFace response with function call extraction
   */
  private parseResponseWithFunctionCalls(response: any): AIModelResponse {
    let text = "";
    if (Array.isArray(response.data)) {
      text = response.data[0]?.generated_text || "";
    } else if (response.data.generated_text) {
      text = response.data.generated_text;
    } else if (typeof response.data === "string") {
      text = response.data;
    } else {
      text = JSON.stringify(response.data);
    }

    // Extract function calls from the response text
    const functionCalls = this.extractFunctionCallsFromText(text);

    return {
      text,
      functionCalls,
      raw: response.data,
    };
  }

  /**
   * Fallback method that uses multipart/form-data for older HuggingFace models
   */
  private async generateWithMultipartForm(
    request: AIModelRequest,
    config: AIModelConfig,
    model: string
  ): Promise<AIModelResponse> {
    // Create a multipart form-data payload for multimodal models
    const formData = new FormData();

    // Add text prompt with function definitions
    const prompt = request.systemPrompt
      ? `${request.systemPrompt}\n\n${request.prompt}`
      : request.prompt;

    // Handle function calling by adding function definitions to the prompt
    let enhancedPrompt = prompt;
    if (request.functions && request.functions.length > 0) {
      const functionText = JSON.stringify(
        { functions: request.functions },
        null,
        2
      );
      enhancedPrompt = `${enhancedPrompt}\n\nAvailable functions:\n\`\`\`json\n${functionText}\n\`\`\`\n\n`;

      if (typeof request.functionCall === "object") {
        enhancedPrompt += `Please call the function: ${request.functionCall.name}\n\n`;
      } else if (request.functionCall === "auto") {
        enhancedPrompt += "Call the appropriate function if needed.\n\n";
      }
    }

    formData.append("text", enhancedPrompt);

    // Process the convenience 'image' property
    if (request.image) {
      const { base64 } = await processImage(request.image);
      const imageBlob = this.base64ToBlob(base64);
      formData.append("image", imageBlob, "image.jpg");
    }

    // Process content array if provided
    if (request.content) {
      let imageIndex = 0;
      for (const item of request.content) {
        if (item.type === "image") {
          const { base64 } = await processImage(item.source);
          const imageBlob = this.base64ToBlob(base64);
          formData.append(
            `image_${imageIndex}`,
            imageBlob,
            `image_${imageIndex}.jpg`
          );
          imageIndex++;
        }
        // Text content is already included in the prompt
      }
    }

    // Add model parameters
    if (config.temperature) {
      formData.append("temperature", config.temperature.toString());
    }
    if (config.maxTokens) {
      formData.append("max_new_tokens", config.maxTokens.toString());
    }
    if (config.topP) {
      formData.append("top_p", config.topP.toString());
    }

    const response = await axios.post(`${this.baseURL}/${model}`, formData, {
      headers: {
        Authorization: `Bearer ${
          config.apiKey ||
          getApiKey(config.apiKey, "HUGGINGFACE_API_KEY", "HuggingFace")
        }`,
        "Content-Type": "multipart/form-data",
      },
    });

    // Parse response based on return format
    let text = "";
    if (Array.isArray(response.data)) {
      text = response.data[0]?.generated_text || "";
    } else if (response.data.generated_text) {
      text = response.data.generated_text;
    } else if (typeof response.data === "string") {
      text = response.data;
    } else {
      text = JSON.stringify(response.data);
    }

    // Extract function calls from the response text
    const functionCalls = this.extractFunctionCallsFromText(text);

    return {
      text,
      functionCalls,
      raw: response.data,
    };
  }

  async *stream(
    request: AIModelRequest
  ): AsyncGenerator<string, void, unknown> {
    try {
      // HuggingFace Inference API doesn't natively support streaming for all models
      // We'll implement a basic chunking on top of the non-streaming API
      const response = await this.generate(request);

      // Simple chunking for demonstration purposes
      const chunkSize = 10;
      const text = response.text;

      for (let i = 0; i < text.length; i += chunkSize) {
        const chunk = text.slice(i, i + chunkSize);
        yield chunk;

        // Add a small delay to simulate streaming
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
    } catch (error: any) {
      // Rethrow with enhanced error message
      throw error;
    }
  }

  /**
   * Convert a base64 string to a Blob object
   */
  private base64ToBlob(base64: string): Blob {
    const byteString = atob(base64);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);

    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ab], { type: "image/jpeg" });
  }
}
