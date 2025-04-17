import axios from "axios";
import {
  AIModelConfig,
  AIModelRequest,
  AIModelResponse,
  AIProvider,
} from "../types";
import { BaseModel } from "./base-model";
import { getApiKey, getBaseUrl } from "../utils";

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

  async *stream(
    request: AIModelRequest
  ): AsyncGenerator<string, void, unknown> {
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
  }
}
