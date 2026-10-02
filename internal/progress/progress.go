// Package progress publishes durable progress counters for external supervisors.
package progress

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type Writer struct {
	mu       sync.Mutex
	path     string
	sequence int
	stage    string
	step     int
}

func New(path string) *Writer { return &Writer{path: path, step: -1} }
func (w *Writer) Advance(stage string, step int) error {
	if w.path == "" {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if stage == w.stage && step == w.step {
		return nil
	}
	w.sequence++
	data, err := json.Marshal(map[string]any{"contract": "ul.layago-progress/v1", "sequence": w.sequence, "stage": stage, "step": step, "updated_at": time.Now().UTC().Format(time.RFC3339Nano)})
	if err != nil {
		return err
	}
	if err = os.MkdirAll(filepath.Dir(w.path), 0700); err != nil {
		return err
	}
	file, err := os.CreateTemp(filepath.Dir(w.path), ".progress-*")
	if err != nil {
		return err
	}
	name := file.Name()
	defer os.Remove(name)
	if _, err = file.Write(append(data, '\n')); err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if err = os.Rename(name, w.path); err != nil {
		return fmt.Errorf("progress: %w", err)
	}
	// Append history only after the atomic status update is durable.
	history, err := os.OpenFile(w.path+".jsonl", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	_, err = history.Write(append(data, '\n'))
	if err == nil {
		err = history.Sync()
	}
	closeErr = history.Close()
	if err == nil {
		err = closeErr
	}
	if err == nil {
		w.stage, w.step = stage, step
	}
	return err
}
