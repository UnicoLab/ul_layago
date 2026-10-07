// Command layago serves typed decisions from local weights using JSON lines over stdin/stdout.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"runtime/debug"
	"strings"
	"syscall"
	"time"

	layago "github.com/UnicoLab/ul_layago"
	"github.com/UnicoLab/ul_layago/internal/cli"
	"github.com/UnicoLab/ul_layago/tensor"
)

type adapterFlags map[string]string

func (a *adapterFlags) String() string { return "id=directory (repeatable)" }
func (a *adapterFlags) Set(value string) error {
	id, dir, ok := strings.Cut(value, "=")
	if !ok || strings.TrimSpace(id) == "" || dir == "" {
		return fmt.Errorf("adapter must be id=directory")
	}
	if *a == nil {
		*a = adapterFlags{}
	}
	if _, ok := (*a)[id]; ok {
		return fmt.Errorf("duplicate adapter ID %q", id)
	}
	(*a)[id] = dir
	return nil
}

func main() {
	var adapters adapterFlags
	flag.Var(&adapters, "adapter", "Head adapter id=artifact-directory; repeatable, exclusive with checkpoint/artifact")
	cacheBytes := flag.Int64("cache-bytes", 0, "Bounded full-sequence encoder cache for adapter mode; 0 disables")
	checkpoint := flag.String("checkpoint", "", "Local pretrained Laya checkpoint")
	artifact := flag.String("artifact", "", "Local trained head artifact (exclusive with checkpoint)")
	base := flag.String("base", "", "Exact base encoder for artifact inference")
	workers := flag.Int("workers", 0, "Intra-operation workers; 0 uses GOMAXPROCS")
	kernelName := flag.String("kernel", "auto", "auto, generic, arm64_neon or amd64_avx2")
	timeout := flag.Duration("timeout", 30*time.Second, "Per-request timeout")
	memoryLimit := flag.Int64("memory-limit", 4<<30, "Soft Go memory budget in bytes (not an OS RSS cap)")
	version := flag.Bool("version", false, "Print package version without loading weights")
	flag.Parse()
	if *version {
		fmt.Println(layago.Version)
		return
	}
	if flag.NArg() != 0 || *timeout <= 0 || *memoryLimit <= 0 {
		fmt.Fprintln(os.Stderr, "unexpected positional arguments or nonpositive timeout")
		os.Exit(2)
	}
	debug.SetMemoryLimit(*memoryLimit)
	kernel, err := tensor.ParseKernel(*kernelName)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if len(adapters) > 0 {
		if *checkpoint != "" || *artifact != "" {
			fmt.Fprintln(os.Stderr, "adapter mode is exclusive with checkpoint/artifact")
			os.Exit(2)
		}
		set, err := layago.OpenAdapters(layago.AdapterSetOptions{Base: *base, Artifacts: adapters, Workers: *workers, Kernel: kernel, CacheBytes: *cacheBytes})
		if err == nil {
			err = cli.ServeAdapters(ctx, os.Stdin, os.Stdout, set, *timeout)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if *cacheBytes != 0 {
		fmt.Fprintln(os.Stderr, "cache-bytes requires adapter mode")
		os.Exit(2)
	}
	predictor, err := layago.Open(layago.OpenOptions{Checkpoint: *checkpoint, Artifact: *artifact, Base: *base, Workers: *workers, Kernel: kernel})
	if err == nil {
		err = cli.Serve(ctx, os.Stdin, os.Stdout, predictor, *timeout)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
