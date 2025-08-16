import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
} from "recharts";
import { TrendingUp, Target, Settings, RefreshCw } from "lucide-react";

interface OptimizationTrial {
  trial_id: number;
  value: number;
  params: {
    learning_rate: number;
    hidden_dim: number;
    embedding_dim: number;
    batch_size: number;
    dropout_rate: number;
  };
  state: "COMPLETE" | "PRUNED" | "FAIL";
  datetime_start: string;
  duration: number;
}

interface TooltipPayload {
  dataKey: string;
  value: number;
  color: string;
}

interface OptunaVisualizationProps {
  className?: string;
}

const OptunaVisualization: React.FC<OptunaVisualizationProps> = ({
  className,
}) => {
  const [trials, setTrials] = useState<OptimizationTrial[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [bestTrial, setBestTrial] = useState<OptimizationTrial | null>(null);

  // Generate mock Optuna optimization data
  const generateOptimizationData = () => {
    setIsLoading(true);

    // Simulate optimization trials
    const mockTrials: OptimizationTrial[] = [];
    const baseDate = new Date("2024-08-15");

    for (let i = 0; i < 50; i++) {
      const learning_rate = Math.random() * 0.01 + 0.0001; // 0.0001 to 0.0101
      const hidden_dim = Math.floor(Math.random() * 200) + 50; // 50 to 250
      const embedding_dim = Math.floor(Math.random() * 150) + 50; // 50 to 200
      const batch_size = [16, 32, 64, 128][Math.floor(Math.random() * 4)];
      const dropout_rate = Math.random() * 0.5; // 0 to 0.5

      // Simulate realistic loss values based on hyperparameters
      let loss = 2.5;
      loss -= (Math.log10(learning_rate) + 3) * 0.1; // Better with optimal learning rate
      loss -= (hidden_dim - 150) * 0.001; // Better with more hidden units (to a point)
      loss -= (embedding_dim - 100) * 0.001; // Better with more embedding dims
      loss += dropout_rate * 0.3; // Too much dropout hurts
      loss += Math.random() * 0.5 - 0.25; // Add noise

      const trial: OptimizationTrial = {
        trial_id: i,
        value: Math.max(0.8, loss), // Minimum loss of 0.8
        params: {
          learning_rate,
          hidden_dim,
          embedding_dim,
          batch_size,
          dropout_rate,
        },
        state:
          Math.random() > 0.1
            ? "COMPLETE"
            : Math.random() > 0.5
            ? "PRUNED"
            : "FAIL",
        datetime_start: new Date(baseDate.getTime() + i * 60000).toISOString(),
        duration: Math.random() * 300 + 30, // 30-330 seconds
      };

      mockTrials.push(trial);
    }

    // Sort by value to find best trial
    const completedTrials = mockTrials.filter((t) => t.state === "COMPLETE");
    const best = completedTrials.reduce((prev, current) =>
      prev.value < current.value ? prev : current
    );

    setTimeout(() => {
      setTrials(mockTrials);
      setBestTrial(best);
      setIsLoading(false);
    }, 1000);
  };

  useEffect(() => {
    generateOptimizationData();
  }, []);

  // Prepare data for different visualizations
  const optimizationHistory = trials.map((trial, index) => ({
    trial: trial.trial_id,
    loss: trial.value,
    cumulative_best: trials
      .slice(0, index + 1)
      .filter((t) => t.state === "COMPLETE")
      .reduce((min, t) => Math.min(min, t.value), Infinity),
    state: trial.state,
  }));

  const parameterImportance = [
    { name: "Learning Rate", importance: 0.35, color: "#8884d8" },
    { name: "Hidden Dim", importance: 0.25, color: "#82ca9d" },
    { name: "Embedding Dim", importance: 0.2, color: "#ffc658" },
    { name: "Dropout Rate", importance: 0.15, color: "#ff7300" },
    { name: "Batch Size", importance: 0.05, color: "#8dd1e1" },
  ];

  const scatterData = trials
    .filter((t) => t.state === "COMPLETE")
    .map((trial) => ({
      learning_rate: trial.params.learning_rate,
      loss: trial.value,
      hidden_dim: trial.params.hidden_dim,
      embedding_dim: trial.params.embedding_dim,
      trial_id: trial.trial_id,
    }));

  const CustomTooltip = ({
    active,
    payload,
    label,
  }: {
    active?: boolean;
    payload?: TooltipPayload[];
    label?: string | number;
  }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border rounded-lg shadow-lg">
          <p className="font-medium">{`Trial ${label}`}</p>
          {payload.map(
            (
              entry: { dataKey: string; value: number; color: string },
              index: number
            ) => (
              <p key={index} style={{ color: entry.color }}>
                {`${entry.dataKey}: ${entry.value.toFixed(4)}`}
              </p>
            )
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5 text-purple-500" />
              LSTM Hyperparameter Optimization
            </CardTitle>
            <CardDescription>
              Optuna-style optimization results for your next-word prediction
              model
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={generateOptimizationData}
            disabled={isLoading}
            className="flex items-center gap-2"
          >
            <RefreshCw
              className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`}
            />
            New Study
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {bestTrial && (
          <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-green-600" />
              <span className="font-medium text-green-700">Best Trial</span>
              <Badge variant="secondary">Trial #{bestTrial.trial_id}</Badge>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
              <div>
                <span className="text-muted-foreground">Loss:</span>
                <span className="ml-2 font-mono font-medium">
                  {bestTrial.value.toFixed(4)}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Learning Rate:</span>
                <span className="ml-2 font-mono font-medium">
                  {bestTrial.params.learning_rate.toFixed(5)}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Hidden Dim:</span>
                <span className="ml-2 font-mono font-medium">
                  {bestTrial.params.hidden_dim}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Embedding Dim:</span>
                <span className="ml-2 font-mono font-medium">
                  {bestTrial.params.embedding_dim}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Batch Size:</span>
                <span className="ml-2 font-mono font-medium">
                  {bestTrial.params.batch_size}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Dropout:</span>
                <span className="ml-2 font-mono font-medium">
                  {bestTrial.params.dropout_rate.toFixed(3)}
                </span>
              </div>
            </div>
          </div>
        )}

        <Tabs defaultValue="history" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="history">Optimization History</TabsTrigger>
            <TabsTrigger value="importance">Parameter Importance</TabsTrigger>
            <TabsTrigger value="scatter">Parameter Relationships</TabsTrigger>
          </TabsList>

          <TabsContent value="history" className="space-y-4">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={optimizationHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="trial" />
                  <YAxis />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="#8884d8"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    name="Trial Loss"
                  />
                  <Line
                    type="monotone"
                    dataKey="cumulative_best"
                    stroke="#82ca9d"
                    strokeWidth={3}
                    name="Best So Far"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-sm text-muted-foreground">
              Blue line shows individual trial losses, green line shows the best
              loss achieved so far.
            </p>
          </TabsContent>

          <TabsContent value="importance" className="space-y-4">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={parameterImportance} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 1]} />
                  <YAxis dataKey="name" type="category" width={100} />
                  <Tooltip
                    formatter={(value: number) => [
                      `${(value * 100).toFixed(1)}%`,
                      "Importance",
                    ]}
                  />
                  <Bar dataKey="importance" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-sm text-muted-foreground">
              Parameter importance based on impact on model performance.
              Learning rate has the highest impact.
            </p>
          </TabsContent>

          <TabsContent value="scatter" className="space-y-4">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart data={scatterData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="learning_rate"
                    name="Learning Rate"
                    type="number"
                    scale="log"
                    domain={["dataMin", "dataMax"]}
                  />
                  <YAxis dataKey="loss" name="Loss" />
                  <Tooltip
                    cursor={{ strokeDasharray: "3 3" }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-white p-3 border rounded-lg shadow-lg">
                            <p className="font-medium">{`Trial ${data.trial_id}`}</p>
                            <p>{`Learning Rate: ${data.learning_rate.toFixed(
                              5
                            )}`}</p>
                            <p>{`Loss: ${data.loss.toFixed(4)}`}</p>
                            <p>{`Hidden Dim: ${data.hidden_dim}`}</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Scatter dataKey="loss" fill="#8884d8" fillOpacity={0.6} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <p className="text-sm text-muted-foreground">
              Relationship between learning rate and model loss. Lower learning
              rates generally perform better.
            </p>
          </TabsContent>
        </Tabs>

        <div className="mt-4 flex items-center gap-2 text-xs text-muted-foreground">
          <Settings className="w-3 h-3" />
          <span>Optimization study: {trials.length} trials completed</span>
          <Badge variant="outline" className="text-xs">
            {trials.filter((t) => t.state === "COMPLETE").length} successful
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
};

export default OptunaVisualization;
