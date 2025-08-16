import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface PredictionResultsProps {
  predictions: string[];
  onPredictionClick: (word: string) => void;
}

export const PredictionResults = ({
  predictions,
  onPredictionClick,
}: PredictionResultsProps) => {
  if (predictions.length === 0) return null;

  return (
    <Card className="shadow-lg">
      <CardHeader>
        <CardTitle>Predicted Next Words</CardTitle>
        <CardDescription>
          Click on any prediction to add it to your sentence
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3">
          {predictions.map((word, index) => (
            <PredictionItem
              key={index}
              word={word}
              rank={index + 1}
              onClick={() => onPredictionClick(word)}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

interface PredictionItemProps {
  word: string;
  rank: number;
  onClick: () => void;
}

const PredictionItem = ({ word, rank, onClick }: PredictionItemProps) => (
  <div
    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors group"
    onClick={onClick}
  >
    <div className="flex items-center gap-3">
      <Badge variant="secondary" className="font-mono">
        #{rank}
      </Badge>
      <span className="text-lg font-medium text-gray-900 group-hover:text-indigo-600 transition-colors">
        {word}
      </span>
    </div>
    <div className="text-right">
      <div className="text-sm font-medium text-indigo-600">Click to add</div>
    </div>
  </div>
);
