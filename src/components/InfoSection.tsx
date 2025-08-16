import { Card, CardContent } from "@/components/ui/card";
import { Brain } from "lucide-react";

export const InfoSection = () => (
  <Card className="border-dashed border-2 border-gray-300">
    <CardContent className="pt-6">
      <div className="text-center text-gray-500">
        <Brain className="w-12 h-12 mx-auto mb-3 text-gray-400" />
        <h3 className="font-medium mb-2">How it works</h3>
        <p className="text-sm leading-relaxed">
          Our AI model analyzes your partial sentence and predicts the most
          likely next words based on patterns learned from training data. The
          confidence score shows how certain the model is about each prediction.
        </p>
      </div>
    </CardContent>
  </Card>
);
