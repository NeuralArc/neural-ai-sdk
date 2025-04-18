/**
 * Utilities for the Neural AI SDK
 */

/**
 * Try to load environment variables from .env files
 * This is done automatically when the module is imported
 */
export function loadEnvVariables(): void {
  try {
    // Only require dotenv if it's available
    // This avoids errors if the user hasn't installed dotenv
    const dotenv = require('dotenv');
    
    // Load from .env file in the project root by default
    dotenv.config();
    
    // Also try to load from any parent directories to support monorepos
    // and projects where the .env file might be in a different location
    dotenv.config({ path: '../../.env' });
    dotenv.config({ path: '../.env' });
  } catch (error) {
    // Silent fail if dotenv is not available
    // This is intentional to not break the module if dotenv is not installed
  }
}

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
