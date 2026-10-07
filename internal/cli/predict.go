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
	return serveRequests(ctx, input, output, predictor.Predict, func(r layago.Request) string { return r.ID }, timeout)
}

// ServeAdapters requires an explicit adapter_id on every line. Unknown selectors
// are errors; no head is selected by arrival order or a hidden default.
func ServeAdapters(ctx context.Context, input io.Reader, output io.Writer, set *layago.AdapterSet, timeout time.Duration) error {
	return serveRequests(ctx, input, output, func(ctx context.Context, r layago.AdapterRequest) (*layago.Response, error) {
		return set.Predict(ctx, r.AdapterID, r.Request)
	}, func(r layago.AdapterRequest) string { return r.ID }, timeout)
}

func serveRequests[T any](ctx context.Context, input io.Reader, output io.Writer, predict func(context.Context, T) (*layago.Response, error), id func(T) string, timeout time.Duration) error {
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
		var request T
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
			if err := encoder.Encode(map[string]any{"contract": layago.PredictionContract, "id": id(request), "error": map[string]string{"code": "invalid_request", "message": err.Error()}}); err != nil {
				return err
			}
			continue
		}
		requestCtx, cancel := context.WithTimeout(ctx, timeout)
		response, err := predict(requestCtx, request)
		cancel()
		if err != nil {
			code := "prediction_failed"
			if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
				code = "cancelled"
			}
			if errors.Is(err, layago.ErrTruncation) {
				code = "input_truncated"
			}
			if err := encoder.Encode(map[string]any{"contract": layago.PredictionContract, "id": id(request), "error": map[string]string{"code": code, "message": err.Error()}}); err != nil {
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
