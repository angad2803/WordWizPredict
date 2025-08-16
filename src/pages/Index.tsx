import { useState } from "react";
import { Header } from "@/components/Header";
import { PredictionInput } from "@/components/PredictionInput";
import { PredictionResults } from "@/components/PredictionResults";
import { ErrorAlert } from "@/components/ErrorAlert";
import { InfoSection } from "@/components/InfoSection";
import { ModelVisualization } from "@/components/ModelVisualization";
import { SentenceSuggestions } from "@/components/SentenceSuggestions";
import OptunaVisualization from "@/components/OptunaVisualization";
import { usePrediction } from "@/hooks/usePrediction";

const Index = () => {
  const [inputText, setInputText] = useState("");
  const {
    predictions,
    analysisData,
    predictMutation,
    clearPredictions,
    isLoading,
    error,
  } = usePrediction();

  const handleSubmit = (text: string) => {
    predictMutation.mutate(text);
  };

  const handlePredictionClick = (word: string) => {
    setInputText((prev) => prev + " " + word);
    clearPredictions();
  };

  const handleInputChange = (text: string) => {
    setInputText(text);
    // Clear predictions when user starts typing again
    if (predictions.length > 0) {
      clearPredictions();
    }
  };

  const handleSentenceSelect = (sentence: string) => {
    setInputText(sentence);
    clearPredictions();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto">
        <Header />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Left Column - Input and Results */}
          <div className="space-y-6">
            <PredictionInput
              inputText={inputText}
              onInputChange={handleInputChange}
              onSubmit={handleSubmit}
              isLoading={isLoading}
            />

            <ErrorAlert error={error} />

            <PredictionResults
              predictions={predictions}
              onPredictionClick={handlePredictionClick}
            />
          </div>

          {/* Right Column - Visualizations and Suggestions */}
          <div className="space-y-6">
            <ModelVisualization
              inputText={inputText}
              predictions={predictions}
              isLoading={isLoading}
              analysisData={analysisData}
            />

            <SentenceSuggestions onSentenceSelect={handleSentenceSelect} />
          </div>
        </div>

        {/* Full Width Section - Optuna Optimization */}
        <div className="mb-6">
          <OptunaVisualization />
        </div>

        {/* Bottom Section - Info */}
        <InfoSection />
      </div>
    </div>
  );
};

export default Index;
