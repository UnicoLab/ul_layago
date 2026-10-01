package cli

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	layago "github.com/UnicoLab/ul_layago"
)

type Predictor interface {
	Predict(context.Context, layago.Request) (*layago.Response, error)
}

// Serve emits one response or structured error per JSON line and keeps stdout machine-readable.
func Serve(ctx context.Context, input io.Reader, output io.Writer, predictor Predictor, timeout time.Duration) error {
	if timeout <= 0 {
		return errors.New("request timeout must be positive")
	}
	scanner := bufio.NewScanner(input)
	scanner.Buffer(make([]byte, 64<<10), 1<<20)
	encoder := json.NewEncoder(output)
	for scanner.Scan() {
		if err := ctx.Err(); err != nil {
			return err
		}
		if len(bytes.TrimSpace(scanner.Bytes())) == 0 {
			continue
		}
		var request layago.Request
		decoder := json.NewDecoder(bytes.NewReader(scanner.Bytes()))
		decoder.DisallowUnknownFields()
		err := decoder.Decode(&request)
		if err == nil && bytes.TrimSpace(scanner.Bytes())[0] != '{' {
			err = errors.New("expected a JSON request object")
		}
		if err == nil {
			var extra any
			if trailing := decoder.Decode(&extra); trailing != io.EOF {
				err = errors.New("expected exactly one JSON object per line")
			}
		}
		if err != nil {
			if err := encoder.Encode(map[string]any{"contract": layago.PredictionContract, "id": request.ID, "error": map[string]string{"code": "invalid_request", "message": err.Error()}}); err != nil {
				return err
			}
			continue
		}
		requestCtx, cancel := context.WithTimeout(ctx, timeout)
		response, err := predictor.Predict(requestCtx, request)
		cancel()
		if err != nil {
			code := "prediction_failed"
			if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
				code = "cancelled"
			}
			if errors.Is(err, layago.ErrTruncation) {
				code = "input_truncated"
			}
			if err := encoder.Encode(map[string]any{"contract": layago.PredictionContract, "id": request.ID, "error": map[string]string{"code": code, "message": err.Error()}}); err != nil {
				return err
			}
		} else if err := encoder.Encode(response); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read JSON lines (maximum 1 MiB per line): %w", err)
	}
	return nil
}
