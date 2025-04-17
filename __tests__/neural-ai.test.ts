import {
  NeuralAI,
  AIProvider,
  OpenAIModel,
  GoogleModel,
  DeepSeekModel,
  OllamaModel,
  HuggingFaceModel,
} from "../src";

describe("NeuralAI", () => {
  describe("createModel", () => {
    it("should create an OpenAI model instance", () => {
      const config = { apiKey: "test-api-key" };
      const model = NeuralAI.createModel(AIProvider.OPENAI, config);
      expect(model).toBeInstanceOf(OpenAIModel);
      expect(model.provider).toBe(AIProvider.OPENAI);
    });

    it("should create a Google model instance", () => {
      const config = { apiKey: "test-api-key" };
      const model = NeuralAI.createModel(AIProvider.GOOGLE, config);
      expect(model).toBeInstanceOf(GoogleModel);
      expect(model.provider).toBe(AIProvider.GOOGLE);
    });

    it("should create a DeepSeek model instance", () => {
      const config = { apiKey: "test-api-key" };
      const model = NeuralAI.createModel(AIProvider.DEEPSEEK, config);
      expect(model).toBeInstanceOf(DeepSeekModel);
      expect(model.provider).toBe(AIProvider.DEEPSEEK);
    });

    it("should create an Ollama model instance", () => {
      const config = {};
      const model = NeuralAI.createModel(AIProvider.OLLAMA, config);
      expect(model).toBeInstanceOf(OllamaModel);
      expect(model.provider).toBe(AIProvider.OLLAMA);
    });

    it("should create a HuggingFace model instance", () => {
      const config = { apiKey: "test-api-key" };
      const model = NeuralAI.createModel(AIProvider.HUGGINGFACE, config);
      expect(model).toBeInstanceOf(HuggingFaceModel);
      expect(model.provider).toBe(AIProvider.HUGGINGFACE);
    });

    it("should throw an error for an unsupported provider", () => {
      const config = { apiKey: "test-api-key" };
      expect(() => {
        // @ts-ignore - Testing invalid provider
        NeuralAI.createModel("unsupported-provider", config);
      }).toThrow("Unsupported AI provider: unsupported-provider");
    });
  });
});
