import { Brain } from "lucide-react";

export const Header = () => (
  <div className="text-center mb-8 pt-8">
    <div className="flex items-center justify-center gap-3 mb-4">
      <Brain className="w-8 h-8 text-indigo-600" />
      <h1 className="text-4xl font-bold text-gray-900">WordWiz</h1>
    </div>
    <p className="text-xl text-gray-600">AI-Powered Next Word Prediction</p>
    <p className="text-sm text-gray-500 mt-2">
      Start typing a sentence and let our AI predict what comes next
    </p>
  </div>
);
