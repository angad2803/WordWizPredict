import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Loader2, Send } from "lucide-react";

interface PredictionInputProps {
  inputText: string;
  onInputChange: (text: string) => void;
  onSubmit: (text: string) => void;
  isLoading: boolean;
}

export const PredictionInput = ({
  inputText,
  onInputChange,
  onSubmit,
  isLoading,
}: PredictionInputProps) => {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim()) {
      onSubmit(inputText.trim());
    }
  };

  return (
    <Card className="shadow-lg">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Send className="w-5 h-5" />
          Type Your Sentence
        </CardTitle>
        <CardDescription>
          Enter a partial sentence and get AI-powered word predictions
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex gap-2">
            <Input
              type="text"
              placeholder="Start typing your sentence..."
              value={inputText}
              onChange={(e) => onInputChange(e.target.value)}
              className="flex-1 text-lg"
              disabled={isLoading}
              autoFocus
            />
            <Button
              type="submit"
              disabled={!inputText.trim() || isLoading}
              className="px-6"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                "Predict"
              )}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
};
