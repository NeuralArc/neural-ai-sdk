import fs from "fs";
import path from "path";
import axios from "axios";

/**
 * Checks if a string is a valid URL
 */
export function isUrl(str: string): boolean {
  try {
    const url = new URL(str);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

/**
 * Checks if a string is a valid file path
 */
export function isFilePath(str: string): boolean {
  try {
    return fs.existsSync(str) && fs.statSync(str).isFile();
  } catch {
    return false;
  }
}

/**
 * Converts an image to base64 from various sources
 * @param source - URL, file path, or Buffer
 * @returns Promise with base64 encoded image
 */
export async function imageToBase64(source: string | Buffer): Promise<string> {
  // If source is already a Buffer
  if (Buffer.isBuffer(source)) {
    return source.toString("base64");
  }

  // If source is a URL
  if (isUrl(source)) {
    try {
      const response = await axios.get(source, { responseType: "arraybuffer" });
      const buffer = Buffer.from(response.data, "binary");
      return buffer.toString("base64");
    } catch (error: any) {
      throw new Error(`Failed to fetch image from URL: ${error.message}`);
    }
  }

  // If source is a file path
  if (isFilePath(source)) {
    try {
      const buffer = fs.readFileSync(source);
      return buffer.toString("base64");
    } catch (error: any) {
      throw new Error(`Failed to read image file: ${error.message}`);
    }
  }

  throw new Error("Invalid image source. Must be URL, file path, or Buffer");
}

/**
 * Determines the MIME type based on file extension
 * @param filePath - Path to the file or URL
 */
export function getMimeType(filePath: string): string {
  if (!filePath) return "image/jpeg"; // Default

  const ext = path.extname(filePath).toLowerCase();

  switch (ext) {
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".png":
      return "image/png";
    case ".gif":
      return "image/gif";
    case ".webp":
      return "image/webp";
    case ".bmp":
      return "image/bmp";
    case ".svg":
      return "image/svg+xml";
    default:
      return "image/jpeg"; // Default to JPEG
  }
}

/**
 * Processes an image source and returns data needed for API requests
 */
export async function processImage(
  source: string | Buffer
): Promise<{ base64: string; mimeType: string }> {
  const base64 = await imageToBase64(source);
  const mimeType =
    typeof source === "string" ? getMimeType(source) : "image/jpeg";

  return { base64, mimeType };
}
