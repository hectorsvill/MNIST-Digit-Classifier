// To run this script directly:
// 1. Save it as main.go
// 2. Ensure the target API (e.g., http://127.0.0.1:5000) is running and accepts image uploads.
// 3. Have a 28x28 grayscale PNG image ready (e.g., digit.png).
// 4. Execute: go run main.go --image=path/to/your/digit.png [other flags]
// Example: go run main.go --image=digit.png --train --label=2
// Example: go run main.go --apiurl=http://other-host:5001 --image=another_digit.png

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"mime/multipart" // Required for multipart form data
	"net/http"
	"os"
	"path/filepath" // To get filename from path
	"time"
)

// --- Structs for JSON unmarshalling (Responses) ---
// Request structs are removed as we are not sending JSON bodies for predict/learn

type HealthResponse struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

type PredictResponse struct {
	Prediction int     `json:"prediction"`
	Confidence float64 `json:"confidence"`
}

// Assuming a structure for the learn response based on the Python project
type LearnResponse struct {
	Message           string  `json:"message"`
	LabelProvided     int     `json:"label_provided"`
	LossOnExample     float64 `json:"loss_on_example"`
	AccuracyOnExample float64 `json:"accuracy_on_example"`
}

// --- Global HTTP Client ---
var httpClient = &http.Client{
	Timeout: 15 * time.Second, // Slightly longer timeout for potential uploads
}

// --- API Interaction Functions ---

func checkHealth(apiURL string) (*HealthResponse, error) {
	url := fmt.Sprintf("%s/health", apiURL)
	resp, err := httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to send GET request to %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API health check failed with status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var healthResp HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&healthResp); err != nil {
		return nil, fmt.Errorf("failed to decode health response JSON: %w", err)
	}
	return &healthResp, nil
}

// predictDigit now takes an image path instead of features
func predictDigit(apiURL string, imagePath string) (*PredictResponse, error) {
	url := fmt.Sprintf("%s/predict", apiURL)

	// Create buffer and multipart writer
	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)

	// Open the image file
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open image file '%s': %w", imagePath, err)
	}
	defer file.Close()

	// Create form file part
	part, err := writer.CreateFormFile("image", filepath.Base(imagePath)) // Use "image" as the field name
	if err != nil {
		return nil, fmt.Errorf("failed to create form file part: %w", err)
	}

	// Copy file contents to the part
	_, err = io.Copy(part, file)
	if err != nil {
		return nil, fmt.Errorf("failed to copy image data to form part: %w", err)
	}

	// IMPORTANT: Close the writer to finalize the body and write boundary
	err = writer.Close()
	if err != nil {
		return nil, fmt.Errorf("failed to close multipart writer: %w", err)
	}

	// Create the request
	req, err := http.NewRequest("POST", url, &requestBody) // Use the buffer
	if err != nil {
		return nil, fmt.Errorf("failed to create POST request: %w", err)
	}
	// Set the content type header *from the writer*
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Send the request
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send POST request to %s: %w", url, err)
	}
	defer resp.Body.Close()

	// Handle response
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("predict request failed with status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var predictResp PredictResponse
	if err := json.NewDecoder(resp.Body).Decode(&predictResp); err != nil {
		return nil, fmt.Errorf("failed to decode predict response JSON: %w", err)
	}
	return &predictResp, nil
}

// trainModel now takes image path and label, no iterations needed for API
func trainModel(apiURL string, imagePath string, label int) (*LearnResponse, error) {
	url := fmt.Sprintf("%s/learn", apiURL)

	// Create buffer and multipart writer
	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)

	// --- Add Image File Part ---
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open image file '%s': %w", imagePath, err)
	}
	defer file.Close()

	part, err := writer.CreateFormFile("image", filepath.Base(imagePath))
	if err != nil {
		return nil, fmt.Errorf("failed to create form file part for learn: %w", err)
	}
	_, err = io.Copy(part, file)
	if err != nil {
		return nil, fmt.Errorf("failed to copy image data to form part for learn: %w", err)
	}
	// --- End Add Image File Part ---

	// --- Add Label Form Field Part ---
	err = writer.WriteField("label", fmt.Sprintf("%d", label)) // Send label as string field
	if err != nil {
		return nil, fmt.Errorf("failed to write label field: %w", err)
	}
	// --- End Add Label Form Field Part ---

	// Close writer
	err = writer.Close()
	if err != nil {
		return nil, fmt.Errorf("failed to close multipart writer for learn: %w", err)
	}

	// Create request
	req, err := http.NewRequest("POST", url, &requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create POST request for learn: %w", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Send request
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send POST request to %s: %w", url, err)
	}
	defer resp.Body.Close()

	// Handle response
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("learn request failed with status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var learnResp LearnResponse
	if err := json.NewDecoder(resp.Body).Decode(&learnResp); err != nil {
		return nil, fmt.Errorf("failed to decode learn response JSON: %w", err)
	}
	return &learnResp, nil
}

// --- Main Execution ---

func main() {
	// --- Command Line Flags ---
	apiURL := flag.String("apiurl", "http://127.0.0.1:5000", "Base URL of the prediction API")
	// Image path is now required unless you add default logic
	imagePath := flag.String("image", "", "Path to the 28x28 grayscale image file (required)")
	train := flag.Bool("train", false, "Perform training using the specified image and label")
	trainLabel := flag.Int("label", -1, "Correct label for the image when training (required if --train is true)")
	// Iterations flag removed as it's not sent in the multipart request anymore
	flag.Parse()

	// Validate required image path flag
	if *imagePath == "" {
		log.Fatal("Error: Image path must be provided using the --image flag.")
		// flag.Usage() // Optionally print usage instructions
		// os.Exit(1)
	}

	log.Printf("Using API URL: %s", *apiURL)
	log.Printf("Using image: %s", *imagePath)

	// --- Check Health ---
	log.Println("Checking API health...")
	healthStatus, err := checkHealth(*apiURL)
	if err != nil {
		log.Fatalf("Health check failed: %v", err)
	}
	log.Printf("API Health Status: %+v", *healthStatus)
	if healthStatus.Status != "ok" {
		log.Printf("Warning: API status is not 'ok': %s", healthStatus.Message)
	}

	// --- Predict Digit ---
	log.Println("Predicting digit from image file...")
	prediction, err := predictDigit(*apiURL, *imagePath)
	if err != nil {
		log.Fatalf("Prediction failed: %v", err)
	}
	log.Printf("Prediction Result: Digit=%d, Confidence=%.4f", prediction.Prediction, prediction.Confidence)

	// --- Train Model (Conditional) ---
	if *train {
		log.Println("Training requested...")
		if *trainLabel < 0 || *trainLabel > 9 {
			log.Fatalf("Invalid label provided for training: %d. Must be between 0 and 9.", *trainLabel)
		}

		log.Printf("Training with image '%s', Label=%d", *imagePath, *trainLabel)
		// Call trainModel without iterations
		trainResponse, err := trainModel(*apiURL, *imagePath, *trainLabel)
		if err != nil {
			log.Fatalf("Training failed: %v", err)
		}
		log.Printf("Training Response: %+v", *trainResponse)
	} else {
		log.Println("Training not requested (use --train flag to enable).")
	}

	log.Println("Script finished.")
}




// How to Run:

// Save the code as main.go.

// Make sure your updated Python Flask API server (accepting image uploads) is running.

// Have a 28x28 grayscale PNG image file ready (e.g., digit.png).

// Open your terminal in the directory where you saved main.go.

// Execute using go run:

// Predict only (Image path is now required):

// go run main.go --image=digit.png
// Use code with caution.
// Bash
// Predict and Train:

// go run main.go --image=digit.png --train --label=2
// Use code with caution.
// Bash
// Use a different API URL:

// go run main.go --apiurl=http://192.168.1.100:5000 --image=another_digit.png
// Use code with caution.
// Bash
// Key Changes:

// Removed image generation code (createDigitImageData).

// Removed image processing imports (image, image/color, _ "image/png").

// Added imports for mime/multipart, path/filepath.

// Modified predictDigit and trainModel to accept imagePath instead of features.

// Implemented multipart/form-data request body creation in predictDigit and trainModel using mime/multipart.

// trainModel now adds the label using writer.WriteField.

// Removed the -iterations flag and its usage.

// Made the -image flag required.

// Updated log messages.

