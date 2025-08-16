import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import PredictionService, {
  type PredictionResponse,
  type AnalysisResponse,
  type PredictionRequest,
} from "@/services/predictionService";

export const usePrediction = () => {
  const [predictions, setPredictions] = useState<string[]>([]);
  const [analysisData, setAnalysisData] = useState<AnalysisResponse | null>(
    null
  );

  const predictMutation = useMutation({
    mutationFn: async (text: string) => {
      const request: PredictionRequest = { text, top_k: 5 };
      // Run both prediction and analysis in parallel
      const [predictionResult, analysisResult] = await Promise.all([
        PredictionService.predict(request),
        PredictionService.analyzeText(request),
      ]);
      return { prediction: predictionResult, analysis: analysisResult };
    },
    onSuccess: (data) => {
      setPredictions(data.prediction.predictions.map((p) => p.word));
      setAnalysisData(data.analysis);
    },
    onError: (error) => {
      console.error("Prediction failed:", error);
      setPredictions([]);
      setAnalysisData(null);
    },
  });

  const clearPredictions = () => {
    setPredictions([]);
    setAnalysisData(null);
  };

  return {
    predictions,
    analysisData,
    predictMutation,
    clearPredictions,
    isLoading: predictMutation.isPending,
    error: predictMutation.error?.message,
  };
};

export default usePrediction;
