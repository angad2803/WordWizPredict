import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

interface ErrorAlertProps {
  error: string | null;
}

export const ErrorAlert = ({ error }: ErrorAlertProps) => {
  if (!error) return null;

  return (
    <Alert variant="destructive">
      <AlertCircle className="h-4 w-4" />
      <AlertDescription>
        {error ||
          "Failed to get predictions. Make sure the backend server is running on http://localhost:8000"}
      </AlertDescription>
    </Alert>
  );
};
