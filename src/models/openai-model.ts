import { OpenAI } from "openai";
import {
  AIModelConfig,
  AIModelRequest,
  AIModelResponse,
  AIProvider,
  Content,
  FunctionDefinition,
  FunctionCall,
} from "../types";
import { BaseModel } from "./base-model";
import { getApiKey } from "../utils";
import { processImage } from "../utils/image-utils";

export class OpenAIModel extends BaseModel {
  readonly provider = AIProvider.OPENAI;
  private client: OpenAI;

  constructor(config: AIModelConfig) {
    super(config);
    const apiKey = getApiKey(config.apiKey, "OPENAI_API_KEY", "OpenAI");

    this.client = new OpenAI({
      apiKey: apiKey,
      baseURL: config.baseURL,
    });
  }

  async generate(request: AIModelRequest): Promise<AIModelResponse> {
    const config = this.mergeConfig(request.options);

    // Process messages for OpenAI API
    const messages = await this.formatMessages(request);

    // Prepare function calling if requested
    const functionOptions = this.prepareFunctionCalling(request);

    const response = await this.client.chat.completions.create({
      model: config.model || "gpt-3.5-turbo",
      messages,
      temperature: config.temperature,
      max_tokens: config.maxTokens,
      top_p: config.topP,
      ...functionOptions,
    });

    // Process function calls if any are present
    const functionCalls = this.processFunctionCalls(response);

    return {
      text: response.choices[0].message.content || "",
      usage: {
        promptTokens: response.usage?.prompt_tokens,
        completionTokens: response.usage?.completion_tokens,
        totalTokens: response.usage?.total_tokens,
      },
      functionCalls,
      raw: response,
    };
  }

  async *stream(
    request: AIModelRequest
  ): AsyncGenerator<string, void, unknown> {
    const config = this.mergeConfig(request.options);

    // Process messages for OpenAI API
    const messages = await this.formatMessages(request);

    // Prepare function calling if requested
    const functionOptions = this.prepareFunctionCalling(request);

    const response = await this.client.chat.completions.create({
      model: config.model || "gpt-3.5-turbo",
      messages,
      temperature: config.temperature,
      max_tokens: config.maxTokens,
      top_p: config.topP,
      stream: true,
      ...functionOptions,
    });

    // Using a more compatible approach with Stream API
    const stream = response as unknown as AsyncIterable<any>;
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || "";
      if (content) {
        yield content;
      }
    }
  }

  /**
   * Prepare OpenAI function calling options
   */
  private prepareFunctionCalling(request: AIModelRequest): any {
    if (!request.functions || request.functions.length === 0) {
      return {};
    }

    // Transform our function definitions to OpenAI's format
    const tools = request.functions.map((func) => ({
      type: "function",
      function: {
        name: func.name,
        description: func.description,
        parameters: func.parameters,
      },
    }));

    // Handle tool_choice (functionCall in our API)
    let tool_choice = undefined;
    if (request.functionCall) {
      if (request.functionCall === "auto") {
        tool_choice = "auto";
      } else if (request.functionCall === "none") {
        tool_choice = "none";
      } else if (typeof request.functionCall === "object") {
        tool_choice = {
          type: "function",
          function: { name: request.functionCall.name },
        };
      }
    }

    return {
      tools,
      tool_choice,
    };
  }

  /**
   * Process function calls from OpenAI response
   */
  private processFunctionCalls(response: any): FunctionCall[] | undefined {
    const toolCalls = response.choices[0]?.message?.tool_calls;

    if (!toolCalls || toolCalls.length === 0) {
      return undefined;
    }

    return toolCalls
      .filter((call: any) => call.type === "function")
      .map((call: any) => ({
        name: call.function.name,
        arguments: call.function.arguments,
      }));
  }

  /**
   * Format messages for OpenAI API, including handling multimodal content
   */
  private async formatMessages(request: AIModelRequest): Promise<any[]> {
    const messages = [];

    // Add system prompt if provided
    if (request.systemPrompt) {
      messages.push({
        role: "system" as const,
        content: request.systemPrompt,
      });
    }

    // Handle multimodal content
    if (request.content || request.image) {
      const content = [];

      // Add the text prompt
      if (request.prompt) {
        content.push({ type: "text", text: request.prompt });
      }

      // Add any structured content
      if (request.content) {
        for (const item of request.content) {
          if (item.type === "text") {
            content.push({ type: "text", text: item.text });
          } else if (item.type === "image") {
            const { base64, mimeType } = await processImage(item.source);
            content.push({
              type: "image_url",
              image_url: {
                url: `data:${mimeType};base64,${base64}`,
              },
            });
          }
        }
      }

      // Add single image if provided via the convenience property
      if (request.image) {
        const { base64, mimeType } = await processImage(request.image);
        content.push({
          type: "image_url",
          image_url: {
            url: `data:${mimeType};base64,${base64}`,
          },
        });
      }

      messages.push({
        role: "user" as const,
        content,
      });
    } else {
      // Traditional text-only message
      messages.push({
        role: "user" as const,
        content: request.prompt,
      });
    }

    return messages;
  }
}
