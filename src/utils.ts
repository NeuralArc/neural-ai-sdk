/**
 * Utilities for the Neural AI SDK
 */

/**
 * Get an API key from config or environment variables
 * @param configKey The API key from the config object
 * @param envVarName The name of the environment variable to check
 * @param providerName The name of the AI provider (for error messages)
 * @returns The API key if found, throws an error otherwise
 */
export function getApiKey(
  configKey: string | undefined,
  envVarName: string,
  providerName: string
): string {
  // First check if the API key is provided in the config
  if (configKey) {
    return configKey;
  }

  // Then check environment variables
  const envKey = process.env[envVarName];
  if (envKey) {
    return envKey;
  }

  // If no API key is found, throw a helpful error message
  throw new Error(
    `${providerName} API key is required.\n` +
      `Please provide it via the 'apiKey' option or set the ${envVarName} environment variable.\n` +
      `Example:\n` +
      `- In code: NeuralAI.createModel(AIProvider.${providerName.toUpperCase()}, { apiKey: "your-api-key" })\n` +
      `- In .env: ${envVarName}=your-api-key`
  );
}

/**
 * Get a base URL from config or environment variables
 * @param configUrl The URL from the config object
 * @param envVarName The name of the environment variable to check
 * @param defaultUrl The default URL to use if not provided
 * @returns The base URL
 */
export function getBaseUrl(
  configUrl: string | undefined,
  envVarName: string,
  defaultUrl: string
): string {
  if (configUrl) {
    return configUrl;
  }

  return process.env[envVarName] || defaultUrl;
}
