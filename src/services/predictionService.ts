/**
 * API service for next-word prediction backend
 */

// Configuration
const CONFIG = {
  API_BASE_URL: "http://localhost:8000",
  DEFAULT_TOP_K: 5,
  REQUEST_TIMEOUT: 10000, // 10 seconds
} as const;

export interface Prediction {
  word: string;
  probability: number;
  confidence: number;
}

export interface PredictionRequest {
  text: string;
  top_k?: number;
}

export interface PredictionResponse {
  predictions: Prediction[];
  input_text: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  vocab_loaded: boolean;
  vocab_size: number;
}

export interface ModelInfo {
  architecture: string;
  vocab_size: number;
  embedding_dim: number | string;
  hidden_dim: number | string;
  num_layers: number | string;
  sample_vocabulary: string[];
  model_parameters: number;
}

export interface AnalysisResponse {
  input_text: string;
  word_tokens: string[];
  token_ids: number[];
  sequence_length: number;
  embedding_shape: number[];
  lstm_output_shape: number[];
  predictions: Prediction[];
  total_vocabulary_considered: number;
}

class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

export class PredictionService {
  private static async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${CONFIG.API_BASE_URL}${endpoint}`;

    // Add timeout to the request
    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      CONFIG.REQUEST_TIMEOUT
    );

    try {
      const response = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
        signal: controller.signal,
        ...options,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new ApiError(
          errorData.detail || `HTTP error! status: ${response.status}`,
          response.status
        );
      }

      return response.json();
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof ApiError) {
        throw error;
      }

      // Handle timeout
      if (error instanceof DOMException && error.name === "AbortError") {
        throw new ApiError(
          "Request timeout. The server is taking too long to respond."
        );
      }

      // Handle network errors
      if (error instanceof TypeError && error.message.includes("fetch")) {
        throw new ApiError(
          "Unable to connect to the prediction service. Please make sure the backend is running on http://localhost:8000"
        );
      }

      throw new ApiError("An unexpected error occurred");
    }
  }

  static async predict(
    request: PredictionRequest
  ): Promise<PredictionResponse> {
    // Validate input
    if (!request.text || !request.text.trim()) {
      throw new ApiError("Input text cannot be empty");
    }

    const requestData = {
      text: request.text.trim(),
      top_k: request.top_k || CONFIG.DEFAULT_TOP_K,
    };

    return this.makeRequest<PredictionResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(requestData),
    });
  }

  static async healthCheck(): Promise<HealthResponse> {
    return this.makeRequest<HealthResponse>("/health");
  }

  static async getModelInfo(): Promise<ModelInfo> {
    return this.makeRequest<ModelInfo>("/model-info");
  }

  static async analyzeText(
    request: PredictionRequest
  ): Promise<AnalysisResponse> {
    // Validate input
    if (!request.text || !request.text.trim()) {
      throw new ApiError("Input text cannot be empty");
    }

    const requestData = {
      text: request.text.trim(),
      top_k: request.top_k || CONFIG.DEFAULT_TOP_K,
    };

    return this.makeRequest<AnalysisResponse>("/analyze", {
      method: "POST",
      body: JSON.stringify(requestData),
    });
  }

  static async ping(): Promise<{ message: string }> {
    return this.makeRequest<{ message: string }>("/");
  }

  // Utility method to check if the service is available
  static async isServiceAvailable(): Promise<boolean> {
    try {
      await this.ping();
      return true;
    } catch {
      return false;
    }
  }
}

export default PredictionService;
