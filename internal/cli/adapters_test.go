package cli

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	layago "github.com/UnicoLab/ul_layago"
	"github.com/UnicoLab/ul_layago/tensor"
)

func TestAdapterJSONLinesRequireExplicitSelectorAndPreserveFollowingRequests(t *testing.T) {
	set, err := layago.OpenAdapters(layago.AdapterSetOptions{Base: "../../testdata/parity/checkpoint", Artifacts: map[string]string{"q": "../../testdata/artifact/tiny-q/float32"}, Workers: 1, Kernel: tensor.KernelGeneric})
	if err != nil {
		t.Fatal(err)
	}
	input := `{"id":"missing","state":"Hello","questions":[{"id":"q","task":"q_metric_choice","options":[{"key":"a"},{"key":"b"}]}]}
{"id":"unknown","adapter_id":"other","state":"Hello","questions":[{"id":"q","task":"q_metric_choice","options":[{"key":"a"},{"key":"b"}]}]}
{"id":"good","adapter_id":"q","state":"Hello","questions":[{"id":"q","task":"q_metric_choice","options":[{"key":"a"},{"key":"b"}]}]}
`
	var out strings.Builder
	if err := ServeAdapters(context.Background(), strings.NewReader(input), &out, set, time.Second); err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 3 {
		t.Fatal(out.String())
	}
	for i, line := range lines {
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatal(err)
		}
		if i < 2 && row["error"] == nil {
			t.Fatal("silently selected adapter")
		}
		if i == 2 && (row["model"] == nil || row["model"].(map[string]any)["adapter_id"] != "q") {
			t.Fatal(row)
		}
	}
	var single strings.Builder
	if err := Serve(context.Background(), strings.NewReader(`{"adapter_id":"q"}`+"\n"), &single, stub(func(context.Context, layago.Request) (*layago.Response, error) {
		t.Fatal("single mode accepted selector")
		return nil, nil
	}), time.Second); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(single.String(), "invalid_request") {
		t.Fatal(single.String())
	}
}
