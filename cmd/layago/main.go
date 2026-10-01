// Command layago serves typed decisions from local weights using JSON lines over stdin/stdout.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"runtime/debug"
	"syscall"
	"time"

	layago "github.com/UnicoLab/ul_layago"
	"github.com/UnicoLab/ul_layago/internal/cli"
	"github.com/UnicoLab/ul_layago/tensor"
)

func main() {
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
	predictor, err := layago.Open(layago.OpenOptions{Checkpoint: *checkpoint, Artifact: *artifact, Base: *base, Workers: *workers, Kernel: kernel})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if err := cli.Serve(ctx, os.Stdin, os.Stdout, predictor, *timeout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
