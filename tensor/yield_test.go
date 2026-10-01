package tensor

import (
	"math/rand/v2"
	"os"
	"runtime"
	"slices"
	"sync/atomic"
	"testing"
	"time"
)

// withYield swaps the yield hook and threshold for one test.
func withYield(t *testing.T, after time.Duration, hook func()) {
	t.Helper()
	prevAfter, prevYield := yieldAfter, yield
	yieldAfter, yield = after, hook
	t.Cleanup(func() { yieldAfter, yield = prevAfter, prevYield })
}

// Run yields between tasks once yieldAfter has passed, on the serial path
// and in every worker, never inside a task, and not at all while the
// threshold is not reached.
func TestRunYieldsBetweenTasks(t *testing.T) {
	var yields atomic.Int64
	var inTask atomic.Int64
	hook := func() {
		if inTask.Load() != 0 {
			t.Error("yielded inside a task")
		}
		yields.Add(1)
	}
	for _, workers := range []int{1, 4} {
		withYield(t, 0, hook)
		yields.Store(0)
		Parallel{Workers: workers}.Run(37, func(int) {})
		if got := yields.Load(); got != 37 {
			t.Errorf("workers=%d, threshold 0: %d yields after 37 tasks, want one after each", workers, got)
		}
		withYield(t, time.Hour, hook)
		yields.Store(0)
		Parallel{Workers: workers}.Run(37, func(int) {})
		if got := yields.Load(); got != 0 {
			t.Errorf("workers=%d, threshold 1h: %d yields", workers, got)
		}
	}
	// A task longer than the threshold is followed by a yield, a short one
	// is not: the serial path, 3 tasks of which only the middle one is slow.
	withYield(t, 5*time.Millisecond, hook)
	yields.Store(0)
	Parallel{Workers: 1}.Run(3, func(i int) {
		inTask.Add(1)
		if i == 1 {
			time.Sleep(20 * time.Millisecond)
		}
		inTask.Add(-1)
	})
	if got := yields.Load(); got != 1 {
		t.Errorf("one slow task among three: %d yields, want 1", got)
	}
}

// At GOMAXPROCS=1 a goroutine sleeping 1 ms at a time wakes late by however
// long the inference goroutine keeps the only processor. This runs one
// synthetic encoder layer at 64 tokens (LayerNorm, the 2304x768 and
// 768x1152 projections, GELU) back to back on the best kernel with the
// yields on (yieldAfter 0.5 ms) and off, and with no work at all (the idle
// floor: the OS timer and, on a shared host, waiting for a core),
// interleaved over three rounds. It requires the median lateness with
// yields to be at most 1 ms, far below the runtime's own preemption (about
// 10 ms), and, where the idle floor shows the host can resolve it, the
// design's p99 <= 2 ms. Wall-clock and host dependent, so opt-in like
// TestAssemblyKernelsStopPromptlyForTheRuntime: LAYAGO_PREEMPT_TEST=1.
func TestYieldsBoundTickerLatenessAtGOMAXPROCS1(t *testing.T) {
	if os.Getenv("LAYAGO_PREEMPT_TEST") == "" {
		t.Skip("timing test: set LAYAGO_PREEMPT_TEST=1 to run it")
	}
	r := rand.New(rand.NewPCG(31, 32))
	const m, d, inter = 64, 768, 1152
	qkv, _ := NewLinear(3*d, d, randomSlice(r, 3*d*d), nil)
	down, _ := NewLinear(d, inter, randomSlice(r, d*inter), nil)
	x := randomSlice(r, m*d)
	h := make([]float32, m*d)
	y := make([]float32, m*3*d)
	act := randomSlice(r, m*inter)
	out := make([]float32, m*d)
	norm := make([]float32, d)
	for i := range norm {
		norm[i] = 1
	}
	par := Parallel{Workers: 1}
	layer := func() {
		LayerNorm(h, x, m, d, norm, nil, 1e-5, par)
		qkv.Forward(y, h, m, par)
		GELUInPlace(act, par)
		down.Forward(out, act, m, par)
	}
	// lateness returns p50 and p99 (ms) of the ticker while work runs for
	// one second; nil work leaves the processor idle.
	lateness := func(work func()) [2]float64 {
		prev := runtime.GOMAXPROCS(1)
		defer runtime.GOMAXPROCS(prev)
		var lat []float64
		stop := make(chan struct{})
		done := make(chan struct{})
		go func() {
			defer close(done)
			for {
				select {
				case <-stop:
					return
				default:
				}
				want := time.Now().Add(time.Millisecond)
				time.Sleep(time.Millisecond)
				lat = append(lat, float64(time.Since(want).Microseconds())/1000)
			}
		}()
		if work == nil {
			time.Sleep(time.Second)
		} else {
			for end := time.Now().Add(time.Second); time.Now().Before(end); {
				work()
			}
		}
		close(stop)
		<-done
		slices.Sort(lat)
		return [2]float64{lat[len(lat)/2], lat[min(len(lat)-1, len(lat)*99/100)]}
	}
	start := time.Now()
	layer()
	t.Logf("one synthetic layer: %.2f ms on %s", float64(time.Since(start).Microseconds())/1000, par.EffectiveKernel())
	var idle, on, off [][2]float64
	for round := 0; round < 3; round++ {
		idle = append(idle, lateness(nil))
		withYield(t, 500*time.Microsecond, runtime.Gosched)
		on = append(on, lateness(layer))
		withYield(t, time.Hour, runtime.Gosched)
		off = append(off, lateness(layer))
	}
	med := func(rows [][2]float64, q int) float64 {
		v := []float64{rows[0][q], rows[1][q], rows[2][q]}
		slices.Sort(v)
		return v[1]
	}
	t.Logf("ticker lateness at GOMAXPROCS=1, ms, median of 3 rounds: idle p50 %.2f p99 %.2f; yields on p50 %.2f p99 %.2f; yields off p50 %.2f p99 %.2f",
		med(idle, 0), med(idle, 1), med(on, 0), med(on, 1), med(off, 0), med(off, 1))
	if med(on, 0) > 1 {
		t.Errorf("with yields the ticker is late by %.2f ms at p50, want <= 1 ms", med(on, 0))
	}
	// The design's p99 <= 2 ms is measurable only where the host itself
	// wakes an idle sleeper on time: on a shared machine the OS deschedules
	// the busy thread for tens of milliseconds, which no yield can help.
	if med(idle, 1) > 1 {
		t.Logf("not judging p99: the idle floor's p99 is %.2f ms, so this host cannot resolve 2 ms (run it on a dedicated host, such as the gate's)", med(idle, 1))
	} else if med(on, 1) > 2 {
		t.Errorf("with yields the ticker is late by %.2f ms at p99, want <= 2 ms (idle floor %.2f ms)", med(on, 1), med(idle, 1))
	}
}

// BenchmarkRunYieldOverhead is the 64-token Wqkv projection on one worker
// with the yields at their default threshold and effectively off, for an
// A/B of their cost (a clock read per task, a Gosched per 0.5 ms).
func BenchmarkRunYieldOverhead(b *testing.B) {
	r := rand.New(rand.NewPCG(9, 10))
	const m, out, in = 64, 2304, 768
	x := randomSlice(r, m*in)
	l, _ := NewLinear(out, in, randomSlice(r, out*in), nil)
	dst := make([]float32, m*out)
	par := Parallel{Workers: 1}
	l.Prepack(par.EffectiveKernel())
	for _, c := range []struct {
		name  string
		after time.Duration
	}{{"yields=0.5ms", yieldAfter}, {"yields=off", time.Hour}} {
		b.Run(c.name, func(b *testing.B) {
			prev := yieldAfter
			yieldAfter = c.after
			defer func() { yieldAfter = prev }()
			for b.Loop() {
				l.Forward(dst, x, m, par)
			}
			b.ReportMetric(float64(2*m*out*in)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
		})
	}
}
