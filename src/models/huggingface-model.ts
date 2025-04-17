import axios from "axios";
import {
  AIModelConfig,
  AIModelRequest,
  AIModelResponse,
  AIProvider,
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
    const model = config.model || "meta-llama/Llama-2-7b-chat-hf";

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

    const payload = {
      inputs: fullPrompt,
      parameters: {
        temperature: config.temperature,
        max_new_tokens: config.maxTokens,
        top_p: config.topP,
        return_full_text: false,
      },
    };

    const response = await axios.post(`${this.baseURL}/${model}`, payload, {
      headers: {
        Authorization: `Bearer ${
          config.apiKey ||
          getApiKey(config.apiKey, "HUGGINGFACE_API_KEY", "HuggingFace")
        }`,
        "Content-Type": "application/json",
      },
    });

    // HuggingFace can return different formats depending on the model
    let text = "";
    if (Array.isArray(response.data)) {
      text = response.data[0]?.generated_text || "";
    } else if (response.data.generated_text) {
      text = response.data.generated_text;
    } else {
      text = JSON.stringify(response.data);
    }

    return {
      text,
      raw: response.data,
    };
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

    let payload: any = {
      inputs: {
        text: prompt,
      },
      parameters: {
        temperature: config.temperature,
        max_new_tokens: config.maxTokens,
        top_p: config.topP,
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

    // Parse response
    return this.parseResponse(response);
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

    // Some models expect a flat structure with inputs as a string
    let payload: any = {
      inputs: prompt,
      parameters: {
        temperature: config.temperature,
        max_new_tokens: config.maxTokens,
        top_p: config.topP,
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

    // Parse response
    return this.parseResponse(response);
  }

  /**
   * Helper to parse HuggingFace response in various formats
   */
  private parseResponse(response: any): AIModelResponse {
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

    return {
      text,
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

    // Add text prompt
    const prompt = request.systemPrompt
      ? `${request.systemPrompt}\n\n${request.prompt}`
      : request.prompt;

    formData.append("text", prompt);

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

    return {
      text,
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
