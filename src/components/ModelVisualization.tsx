import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Brain, Layers, Activity } from "lucide-react";
import { type AnalysisResponse } from "@/services/predictionService";

interface ModelVisualizationProps {
  inputText: string;
  predictions: string[];
  isLoading: boolean;
  analysisData?: AnalysisResponse | null;
}

export const ModelVisualization = ({
  inputText,
  predictions,
  isLoading,
  analysisData,
}: ModelVisualizationProps) => {
  // Use analysisData tokens if available, otherwise fallback to simple split
  const tokens =
    analysisData?.word_tokens ||
    inputText
      .toLowerCase()
      .split(" ")
      .filter((word) => word.length > 0);

  // Convert predictions to the format expected by the Progress component
  const predictionData =
    analysisData?.predictions ||
    predictions.map((word, index) => ({
      word,
      probability: 1 / (index + 1), // Fallback probability
      confidence: Math.max(0, 100 - index * 20), // Fallback confidence
    }));

  return (
    <Card className="shadow-lg border-blue-200">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-blue-600" />
          Model Internal Visualization
        </CardTitle>
        <CardDescription>
          See how the AI processes your input step by step
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Input Tokenization */}
        <div>
          <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
            <Layers className="w-4 h-4" />
            1. Input Tokenization
          </h4>
          <div className="flex flex-wrap gap-2 p-3 bg-gray-50 rounded-lg">
            {tokens.length > 0 ? (
              tokens.map((token, index) => (
                <Badge key={index} variant="outline" className="font-mono">
                  {token}
                </Badge>
              ))
            ) : (
              <span className="text-gray-500 text-sm">
                Enter text to see tokenization
              </span>
            )}
          </div>
          {analysisData && (
            <div className="mt-2 text-xs text-gray-600">
              Token IDs: [{analysisData.token_ids.join(", ")}] | Sequence
              Length: {analysisData.sequence_length}
            </div>
          )}
        </div>

        {/* Model Processing */}
        <div>
          <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
            <Activity className="w-4 h-4" />
            2. Neural Network Processing
          </h4>
          <div className="space-y-3">
            <ProcessingLayer
              name="Embedding Layer"
              description={
                analysisData
                  ? `Converts words to ${
                      analysisData.embedding_shape[1]
                    }D vectors (${analysisData.embedding_shape.join(" × ")})`
                  : "Converts words to numerical vectors"
              }
              isActive={tokens.length > 0}
              isLoading={isLoading}
            />
            <ProcessingLayer
              name="LSTM Layer"
              description={
                analysisData
                  ? `Processes sequence with output shape ${analysisData.lstm_output_shape.join(
                      " × "
                    )}`
                  : "Processes sequence and captures context"
              }
              isActive={tokens.length > 0}
              isLoading={isLoading}
            />
            <ProcessingLayer
              name="Output Layer"
              description={
                analysisData
                  ? `Generates probabilities over ${analysisData.total_vocabulary_considered} vocabulary words`
                  : "Generates probability distribution over vocabulary"
              }
              isActive={tokens.length > 0}
              isLoading={isLoading}
            />
          </div>
        </div>

        {/* Prediction Probabilities */}
        {predictionData.length > 0 && (
          <div>
            <h4 className="text-sm font-medium mb-2">
              3. Prediction Probabilities
            </h4>
            <div className="space-y-2">
              {predictionData.slice(0, 5).map((prediction, index) => (
                <div key={index} className="flex items-center gap-3">
                  <Badge variant="secondary" className="w-12 text-center">
                    #{index + 1}
                  </Badge>
                  <span className="font-mono font-medium min-w-[80px]">
                    {prediction.word}
                  </span>
                  <Progress
                    value={prediction.confidence}
                    className="flex-1 h-2"
                  />
                  <span className="text-sm text-gray-500 min-w-[40px]">
                    {prediction.confidence.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

interface ProcessingLayerProps {
  name: string;
  description: string;
  isActive: boolean;
  isLoading: boolean;
}

const ProcessingLayer = ({
  name,
  description,
  isActive,
  isLoading,
}: ProcessingLayerProps) => (
  <div
    className={`p-3 rounded-lg border transition-colors ${
      isActive ? "bg-blue-50 border-blue-200" : "bg-gray-50 border-gray-200"
    }`}
  >
    <div className="flex items-center justify-between">
      <div>
        <div className="font-medium text-sm">{name}</div>
        <div className="text-xs text-gray-600">{description}</div>
      </div>
      <div className="flex items-center gap-2">
        {isLoading && isActive && (
          <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
        )}
        <div
          className={`w-3 h-3 rounded-full ${
            isActive ? "bg-green-500" : "bg-gray-300"
          }`}
        ></div>
      </div>
    </div>
  </div>
);
