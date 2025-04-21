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

export class DeepSeekModel extends BaseModel {
  readonly provider = AIProvider.DEEPSEEK;
  private baseURL: string;

  constructor(config: AIModelConfig) {
    super(config);
    const apiKey = getApiKey(config.apiKey, "DEEPSEEK_API_KEY", "DeepSeek");
    this.baseURL = getBaseUrl(
      config.baseURL,
      "DEEPSEEK_BASE_URL",
      "https://api.deepseek.com/v1"
    );
  }

  async generate(request: AIModelRequest): Promise<AIModelResponse> {
    const config = this.mergeConfig(request.options);

    const messages = [];

    if (request.systemPrompt) {
      messages.push({
        role: "system",
        content: request.systemPrompt,
      });
    }

    messages.push({
      role: "user",
      content: request.prompt,
    });

    // Prepare request payload
    const payload: any = {
      model: config.model || "deepseek-chat",
      messages,
      temperature: config.temperature,
      max_tokens: config.maxTokens,
      top_p: config.topP,
    };

    // Add function calling support if functions are provided
    if (request.functions && request.functions.length > 0) {
      payload.functions = request.functions;

      // Handle function call configuration
      if (request.functionCall) {
        if (request.functionCall === "auto") {
          payload.function_call = "auto";
        } else if (request.functionCall === "none") {
          payload.function_call = "none";
        } else if (typeof request.functionCall === "object") {
          payload.function_call = { name: request.functionCall.name };
        }
      }
    }

    const response = await axios.post(
      `${this.baseURL}/chat/completions`,
      payload,
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${
            config.apiKey ||
            getApiKey(config.apiKey, "DEEPSEEK_API_KEY", "DeepSeek")
          }`,
        },
      }
    );

    // Process function calls if any
    const functionCalls = this.processFunctionCalls(response.data);

    return {
      text: response.data.choices[0].message.content,
      usage: {
        promptTokens: response.data.usage?.prompt_tokens,
        completionTokens: response.data.usage?.completion_tokens,
        totalTokens: response.data.usage?.total_tokens,
      },
      functionCalls,
      raw: response.data,
    };
  }

  async *stream(
    request: AIModelRequest
  ): AsyncGenerator<string, void, unknown> {
    const config = this.mergeConfig(request.options);

    const messages = [];

    if (request.systemPrompt) {
      messages.push({
        role: "system",
        content: request.systemPrompt,
      });
    }

    messages.push({
      role: "user",
      content: request.prompt,
    });

    // Prepare request payload
    const payload: any = {
      model: config.model || "deepseek-chat",
      messages,
      temperature: config.temperature,
      max_tokens: config.maxTokens,
      top_p: config.topP,
      stream: true,
    };

    // Add function calling support if functions are provided
    if (request.functions && request.functions.length > 0) {
      payload.functions = request.functions;

      // Handle function call configuration
      if (request.functionCall) {
        if (request.functionCall === "auto") {
          payload.function_call = "auto";
        } else if (request.functionCall === "none") {
          payload.function_call = "none";
        } else if (typeof request.functionCall === "object") {
          payload.function_call = { name: request.functionCall.name };
        }
      }
    }

    const response = await axios.post(
      `${this.baseURL}/chat/completions`,
      payload,
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${
            config.apiKey ||
            getApiKey(config.apiKey, "DEEPSEEK_API_KEY", "DeepSeek")
          }`,
        },
        responseType: "stream",
      }
    );

    const reader = response.data;

    for await (const chunk of reader) {
      const lines = chunk.toString().split("\n").filter(Boolean);

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") continue;

          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices[0]?.delta?.content;
            if (content) {
              yield content;
            }
          } catch (error) {
            console.error("Error parsing DeepSeek stream data:", error);
          }
        }
      }
    }
  }

  /**
   * Process function calls from DeepSeek API response
   */
  private processFunctionCalls(response: any): FunctionCall[] | undefined {
    if (!response.choices?.[0]?.message?.function_call) {
      return undefined;
    }

    const functionCall = response.choices[0].message.function_call;
    return [
      {
        name: functionCall.name,
        arguments: functionCall.arguments,
      },
    ];
  }
}
