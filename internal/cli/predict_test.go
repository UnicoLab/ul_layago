package cli

import (
	"context"
	"encoding/json"
	"errors"
	layago "github.com/UnicoLab/ul_layago"
	"io"
	"strings"
	"testing"
	"time"
)

type stub func(context.Context, layago.Request) (*layago.Response, error)

func (f stub) Predict(ctx context.Context, r layago.Request) (*layago.Response, error) {
	return f(ctx, r)
}
func TestJSONLinesErrorsDoNotLoseFollowingRequests(t *testing.T) {
	p := stub(func(ctx context.Context, r layago.Request) (*layago.Response, error) {
		if r.ID == "long" {
			return nil, layago.ErrTruncation
		}
		return &layago.Response{Contract: layago.PredictionContract, ID: r.ID}, nil
	})
	input := "{broken\n{\"id\":\"extra\",\"unsupported\":true}\n{} {}\n{\"id\":\"long\"}\n{\"id\":\"good\"}\n"
	var out strings.Builder
	if err := Serve(context.Background(), strings.NewReader(input), &out, p, time.Second); err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 5 {
		t.Fatal(out.String())
	}
	for i, line := range lines {
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatal(err)
		}
		if i < 4 && row["error"] == nil {
			t.Fatal(row)
		}
		if i == 4 && row["id"] != "good" {
			t.Fatal(row)
		}
	}
}

type brokenWriter struct{}

func (brokenWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }
func TestDeadlineAndIOFailure(t *testing.T) {
	p := stub(func(ctx context.Context, r layago.Request) (*layago.Response, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	})
	var out strings.Builder
	if err := Serve(context.Background(), strings.NewReader("{}\n"), &out, p, time.Millisecond); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "cancelled") {
		t.Fatal(out.String())
	}
	if err := Serve(context.Background(), strings.NewReader("broken\n"), brokenWriter{}, p, time.Second); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatal(err)
	}
	if err := Serve(context.Background(), strings.NewReader(strings.Repeat("x", 1<<20)), &out, p, time.Second); err == nil {
		t.Fatal("accepted oversized line")
	}
}
