package tensor

import (
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"runtime/metrics"
	"slices"
	"sync/atomic"
	"testing"
	"time"
)

// stopMetric is the runtime's stop-the-world stopping latency distribution
// (time from deciding to stop the world until every P has stopped).
const stopMetric = "/sched/pauses/stopping/gc:seconds"

func readStopHistogram(t testing.TB) *metrics.Float64Histogram {
	s := []metrics.Sample{{Name: stopMetric}}
	metrics.Read(s)
	if s[0].Value.Kind() != metrics.KindFloat64Histogram {
		t.Skipf("%s is not available in this Go runtime", stopMetric)
	}
	return s[0].Value.Float64Histogram()
}

// histogramQuantile returns the q-quantile (in ms, bucket upper bound) of
// the observations added between two reads of the same histogram.
func histogramQuantile(before, after *metrics.Float64Histogram, q float64) (float64, uint64) {
	var total uint64
	counts := make([]uint64, len(after.Counts))
	for i := range counts {
		counts[i] = after.Counts[i] - before.Counts[i]
		total += counts[i]
	}
	need := max(1, uint64(math.Ceil(q*float64(total))))
	var seen uint64
	for i, c := range counts {
		seen += c
		if seen >= need {
			hi := after.Buckets[i+1]
			if math.IsInf(hi, 1) {
				hi = after.Buckets[i]
			}
			return hi * 1e3, total
		}
	}
	return math.NaN(), total
}

// Go assembly cannot be preempted asynchronously, so a goroutine inside the
// kernels stops for the scheduler or a stop-the-world only at a Go
// preemption point. This test runs single-tile passes whose K is 512 kernel
// calls long (several milliseconds per tile) while another goroutine forces
// garbage collections, and requires the median stopping latency to stay
// below an eighth of the tile time: that holds only if every kernel call,
// not just every tile, is a preemption point (callF32/callQ8). Without those
// wrappers the median is about half a tile.
//
// It measures wall time on a shared scheduler and allocates about 180 MB,
// so it runs only when LAYAGO_PREEMPT_TEST=1; the kernel receipt records
// the same evidence, and TestEveryAssemblyCallIsAPreemptionPoint holds the
// code structure it depends on in the default suite.
func TestAssemblyKernelsStopPromptlyForTheRuntime(t *testing.T) {
	if os.Getenv("LAYAGO_PREEMPT_TEST") == "" {
		t.Skip("timing test: set LAYAGO_PREEMPT_TEST=1 to run it")
	}
	kernels := asmKernels(t)
	if len(kernels) == 0 {
		t.Skip("no assembly kernel")
	}
	prev := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(prev)
	r := rand.New(rand.NewPCG(23, 24))
	for _, kern := range kernels {
		for _, int8 := range []bool{false, true} {
			mr, nr := kern.Tile()
			k := 512 * kChunk
			l, _ := NewLinear(nr, k, randomSlice(r, nr*k), nil)
			if int8 {
				l.Quantize()
			}
			x := randomSlice(r, mr*k)
			dst := make([]float32, mr*nr)
			par := Parallel{Workers: 1, Kernel: kern}
			var passes []float64
			for i := 0; i < 7; i++ {
				start := time.Now()
				l.Forward(dst, x, mr, par)
				passes = append(passes, float64(time.Since(start).Microseconds())/1000)
			}
			slices.Sort(passes)
			tileMs := passes[len(passes)/2]
			var stop atomic.Bool
			done := make(chan struct{})
			go func() {
				defer close(done)
				for !stop.Load() {
					l.Forward(dst, x, mr, par)
				}
			}()
			time.Sleep(10 * time.Millisecond)
			before := readStopHistogram(t)
			for i := 0; i < 200; i++ {
				runtime.GC()
				time.Sleep(time.Duration(r.IntN(2000)) * time.Microsecond)
			}
			after := readStopHistogram(t)
			stop.Store(true)
			<-done
			p50, n := histogramQuantile(before, after, 0.5)
			p90, _ := histogramQuantile(before, after, 0.9)
			t.Logf("%s int8=%v: one tile of %d calls takes %.2f ms; stop-the-world stopping over %d stops: p50 %.3f ms, p90 %.3f ms",
				kern, int8, k/kChunk, tileMs, n, p50, p90)
			if p50 > tileMs/8 {
				t.Errorf("%s int8=%v: median stopping latency %.3f ms is not far below the %.2f ms tile: kernel calls are not preemption points", kern, int8, p50, tileMs)
			}
		}
	}
}
