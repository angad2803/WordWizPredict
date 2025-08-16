import { useState, useEffect, useCallback } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { BookOpen, RefreshCw, ArrowRight } from "lucide-react";

interface SentenceSuggestionsProps {
  onSentenceSelect: (sentence: string) => void;
}

interface SentenceSuggestion {
  id: number;
  text: string;
  category: string;
  description: string;
}

interface SuggestionCardProps {
  suggestion: SentenceSuggestion;
  onSelect: (text: string) => void;
  categoryColor: string;
}

// Short prompts perfect for next-word prediction
const PROMPT_LIBRARY = [
  {
    text: "what is data",
    category: "curriculum",
    description: "Data science topics",
  },
  { text: "the course fee", category: "payment", description: "Pricing info" },
  {
    text: "machine learning",
    category: "curriculum",
    description: "ML topics",
  },
  { text: "what if i", category: "sessions", description: "Session queries" },
  {
    text: "where can i",
    category: "schedule",
    description: "Location questions",
  },
  {
    text: "python programming",
    category: "curriculum",
    description: "Programming topics",
  },
  {
    text: "can i do",
    category: "eligibility",
    description: "Eligibility check",
  },
  { text: "the program", category: "general", description: "Program info" },
  {
    text: "deep learning",
    category: "curriculum",
    description: "Advanced topics",
  },
  { text: "live session", category: "sessions", description: "Class format" },
  {
    text: "placement assistance",
    category: "placement",
    description: "Job support",
  },
  {
    text: "monthly subscription",
    category: "subscription",
    description: "Payment plans",
  },
  { text: "refund policy", category: "refund", description: "Refund terms" },
  {
    text: "completion certificate",
    category: "certificate",
    description: "Certification",
  },
  { text: "sql for", category: "curriculum", description: "Database topics" },
  {
    text: "portfolio building",
    category: "placement",
    description: "Career prep",
  },
];

const CATEGORY_COLORS: Record<string, string> = {
  payment: "bg-green-100 text-green-800",
  curriculum: "bg-purple-100 text-purple-800",
  sessions: "bg-orange-100 text-orange-800",
  schedule: "bg-indigo-100 text-indigo-800",
  eligibility: "bg-cyan-100 text-cyan-800",
  general: "bg-slate-100 text-slate-800",
  placement: "bg-amber-100 text-amber-800",
  subscription: "bg-teal-100 text-teal-800",
  refund: "bg-red-100 text-red-800",
  certificate: "bg-emerald-100 text-emerald-800",
};

export const SentenceSuggestions = ({
  onSentenceSelect,
}: SentenceSuggestionsProps) => {
  const [suggestions, setSuggestions] = useState<SentenceSuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const loadSuggestions = useCallback(() => {
    setIsLoading(true);
    setTimeout(() => {
      // Randomly select 8 short prompts
      const shuffled = [...PROMPT_LIBRARY].sort(() => 0.5 - Math.random());
      const selected = shuffled.slice(0, 8).map((prompt, index) => ({
        id: index,
        text: prompt.text,
        category: prompt.category,
        description: prompt.description,
      }));
      setSuggestions(selected);
      setIsLoading(false);
    }, 300);
  }, []);

  useEffect(() => {
    loadSuggestions();
  }, [loadSuggestions]);

  const getCategoryColor = (category: string) =>
    CATEGORY_COLORS[category] || "bg-gray-100 text-gray-800";

  return (
    <Card className="shadow-lg border-blue-200">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-blue-600" />
              Quick Prediction Prompts
            </CardTitle>
            <CardDescription>
              Try these short prompts to see next-word predictions
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={loadSuggestions}
            disabled={isLoading}
            className="flex items-center gap-2"
          >
            <RefreshCw
              className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`}
            />
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3">
          {isLoading
            ? // Loading skeletons
              Array.from({ length: 6 }).map((_, index) => (
                <div
                  key={index}
                  className="p-3 bg-gray-100 rounded-lg animate-pulse"
                >
                  <div className="h-4 bg-gray-300 rounded w-3/4 mb-2"></div>
                  <div className="h-3 bg-gray-300 rounded w-1/2"></div>
                </div>
              ))
            : suggestions.map((suggestion) => (
                <SuggestionCard
                  key={suggestion.id}
                  suggestion={suggestion}
                  onSelect={onSentenceSelect}
                  categoryColor={getCategoryColor(suggestion.category)}
                />
              ))}
        </div>
        <div className="mt-4 pt-3 border-t">
          <Badge variant="outline" className="text-xs">
            💡 Based on your Data Science Mentorship Program dataset
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
};

const SuggestionCard = ({
  suggestion,
  onSelect,
  categoryColor,
}: SuggestionCardProps) => (
  <div
    className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors group border"
    onClick={() => onSelect(suggestion.text)}
  >
    <div className="flex items-center justify-between">
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-gray-900 group-hover:text-green-600 transition-colors">
            "{suggestion.text}"
          </span>
          <ArrowRight className="w-4 h-4 text-gray-400 group-hover:text-green-600 transition-colors" />
        </div>
        <div className="flex items-center gap-2">
          <Badge className={`text-xs ${categoryColor}`}>
            {suggestion.category}
          </Badge>
          <span className="text-xs text-gray-500">
            {suggestion.description}
          </span>
        </div>
      </div>
    </div>
  </div>
);
