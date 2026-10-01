package train

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"time"

	"github.com/UnicoLab/ul_layago/safetensors"
	"github.com/UnicoLab/ul_layago/tensor"
)

// CacheContract names the on-disk feature cache layout: a store of trunk
// features content-addressed by each item's token ids and markers, shared
// by every dataset view encoded with the same trunk, kernel and precision.
// A larger nested subsample, another split or another run over the same
// sequences encodes only the sequences the store does not hold yet.
const CacheContract = "hive.laya-feature-cache/v3"

// CacheSpec controls where and how features are cached.
type CacheSpec struct {
	// Root holds one store directory per store key. It must be outside the
	// repository: a real store is hundreds of megabytes.
	Root string
	// DType is "f32" (default, bit-exact) or "f16" (half the size; features
	// then differ from an end-to-end forward by float16 rounding).
	DType string
	// ShardSize is the number of sequences per new shard file (default 256).
	ShardSize int
	// Workers is the number of batches encoded at once (default GOMAXPROCS);
	// IntraOp is the intra-op worker count of each forward (default 1) and
	// Batch the number of sequences stacked per forward (default 8).
	Workers, IntraOp, Batch int
	Kernel                  tensor.Kernel
	// Progress, when set, is called after every new shard.
	Progress func(CacheProgress)
}

// CacheProgress reports a cache build in flight. Total counts the view's
// items and ExamplesDone those whose features are stored (reused or
// encoded); Encode is the number of sequences this build encodes and
// Encoded how many it has encoded.
type CacheProgress struct {
	ShardsDone, Shards   int
	ExamplesDone, Total  int
	Encoded, Encode      int
	Elapsed              time.Duration
	ExamplesPerSecond    float64
	ResumedShards        int
	LastShardExamplesSec float64
}

// SkippedExample is an item the cache did not featurise.
type SkippedExample struct {
	ID     string `json:"id"`
	Reason string `json:"reason"`
}

// Throughput reports how fast the trunk ran while building a cache.
type Throughput struct {
	Examples          int     `json:"examples_encoded"`
	Tokens            int     `json:"tokens_encoded"`
	Seconds           float64 `json:"wall_seconds"`
	ExamplesPerSecond float64 `json:"examples_per_second"`
	TokensPerSecond   float64 `json:"tokens_per_second"`
	Workers           int     `json:"workers"`
	IntraOp           int     `json:"intra_op_workers"`
	Batch             int     `json:"sequences_per_forward"`
	// ResumedShards counts the store shards holding sequences the view
	// reused, ReusedSequences those sequences.
	ResumedShards   int    `json:"resumed_shards"`
	ReusedSequences int    `json:"reused_sequences"`
	Kernel          string `json:"kernel"`
	GOMAXPROCS      int    `json:"gomaxprocs"`
	// CPUSeconds is the process CPU time of the build; on a shared host
	// TokensPerCPUSecond is the steadier rate, since wall time also
	// measures the queue for a core.
	CPUSeconds         float64 `json:"process_cpu_seconds"`
	TokensPerCPUSecond float64 `json:"tokens_per_cpu_second"`
	LoadAverageStart   string  `json:"load_average_at_start"`
	LoadAverageEnd     string  `json:"load_average_at_end"`
}

// CacheTrunk is everything about the trunk the cached features depend on.
type CacheTrunk struct {
	ModelID         string `json:"model_id"`
	WeightsSHA256   string `json:"weights_sha256"`
	ConfigSHA256    string `json:"config_sha256"`
	TokenizerSHA256 string `json:"tokenizer_sha256"`
	TensorPrefix    string `json:"tensor_prefix"`
	LayersTotal     int    `json:"layers_total"`
	LayersUsed      int    `json:"layers_used"`
	HiddenSize      int    `json:"hidden_size"`
	NumericPath     string `json:"numeric_path"`
	MaxLen          int    `json:"max_len"`
	HeadMaxLen      int    `json:"head_max_len"`
}

func cacheTrunk(id TrunkIdentity) CacheTrunk {
	return CacheTrunk{ModelID: id.ModelID, WeightsSHA256: id.WeightsSHA256, ConfigSHA256: id.ConfigSHA256,
		TokenizerSHA256: id.TokenizerSHA256, TensorPrefix: id.TensorPrefix, LayersTotal: id.LayersTotal, LayersUsed: id.LayersUsed,
		HiddenSize: id.HiddenSize, NumericPath: id.NumericPath, MaxLen: id.MaxLen, HeadMaxLen: id.HeadMaxLen}
}

// StoreInfo is store.json: everything a stored feature depends on besides
// its own sequence.
type StoreInfo struct {
	Contract string     `json:"contract"`
	Key      string     `json:"cache_id"`
	Trunk    CacheTrunk `json:"trunk"`
	Kernel   string     `json:"kernel"`
	DType    string     `json:"dtype"`
}

// CacheInfo describes one view of the store: the items of one prepared
// dataset, written to views/<sequence digest>.json.
type CacheInfo struct {
	Contract      string `json:"contract"`
	Key           string `json:"cache_id"`
	DatasetDigest string `json:"dataset_digest"`
	Instance      string `json:"instance"`
	// SequenceDigest covers every item's id, token ids and markers, so a
	// change of rendering, instructions, options or subsample is another
	// view; the store keeps the sequences views share once.
	SequenceDigest string           `json:"sequence_digest"`
	Trunk          CacheTrunk       `json:"trunk"`
	Kernel         string           `json:"kernel"`
	DType          string           `json:"dtype"`
	ShardSize      int              `json:"shard_size"`
	Examples       int              `json:"examples"`
	Sequences      int              `json:"distinct_sequences"`
	Shards         int              `json:"store_shards"`
	StoreSequences int              `json:"store_sequences"`
	Skipped        []SkippedExample `json:"skipped"`
	SeqLenP50      int              `json:"sequence_tokens_p50"`
	SeqLenP95      int              `json:"sequence_tokens_p95"`
	SeqLenMax      int              `json:"sequence_tokens_max"`
	SeqLenMean     float64          `json:"sequence_tokens_mean"`
	Throughput     Throughput       `json:"last_build_throughput"`
}

// Record is one item's cached features.
type Record struct {
	ID     string
	SeqLen int
	Features
}

// Cache is a feature store bound to the items of one prepared dataset.
type Cache struct {
	Dir   string
	Store StoreInfo
	Info  CacheInfo
	// view maps each scorable item's id to its sequence key; index maps a
	// stored sequence key to its shard and position.
	view  map[string]string
	index map[string]storedSeq
	// Encoded and Reused count the view's distinct sequences the last
	// BuildCache encoded and found stored (Info.Throughput keeps the
	// throughput of the build that last encoded any).
	Encoded, Reused int
}

type storedSeq struct {
	shard, pos int
}

// CacheKey derives the store directory name from everything a stored
// feature depends on besides its sequence: the trunk's weights, config and
// tokenizer digests, layers used, numeric path and sequence limits, the
// kernel and the stored precision.
func CacheKey(trunk CacheTrunk, kernel, dtype string) string {
	idJSON, _ := json.Marshal(trunk)
	h := sha256.New()
	for _, part := range []string{CacheContract, string(idJSON), kernel, dtype} {
		h.Write([]byte(part))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))[:24]
}

// SequenceKey is the store's key of one sequence: its token ids and marker
// positions, all a frozen trunk's features depend on besides the trunk.
func SequenceKey(ids []int32, markers []int) string {
	h := sha256.New()
	var b [4]byte
	put := func(v uint32) {
		binary.LittleEndian.PutUint32(b[:], v)
		h.Write(b[:])
	}
	put(uint32(len(ids))) //nolint:gosec // G115: sequence lengths are bounded by max_len
	for _, id := range ids {
		put(uint32(id)) //nolint:gosec // G115: two's-complement encoding
	}
	put(uint32(len(markers))) //nolint:gosec // G115: option counts are small
	for _, m := range markers {
		put(uint32(m)) //nolint:gosec // G115: marker positions are bounded by max_len
	}
	return hex.EncodeToString(h.Sum(nil))[:32]
}

func (s CacheSpec) withDefaults() (CacheSpec, error) {
	if s.DType == "" {
		s.DType = "f32"
	}
	if s.DType != "f32" && s.DType != "f16" {
		return s, fmt.Errorf("train: feature dtype %q is not f32 or f16", s.DType)
	}
	if s.ShardSize <= 0 {
		s.ShardSize = 256
	}
	if s.Workers <= 0 {
		s.Workers = runtime.GOMAXPROCS(0)
	}
	if s.IntraOp <= 0 {
		s.IntraOp = 1
	}
	if s.Batch <= 0 {
		s.Batch = 8
	}
	if s.Root == "" {
		return s, errors.New("train: cache root is required")
	}
	return s, nil
}

// viewPath is where a view's CacheInfo is written.
func viewPath(dir, sequenceDigest string) string {
	return filepath.Join(dir, "views", sequenceDigest[:min(len(sequenceDigest), 24)]+".json")
}

// BuildCache runs the frozen trunk once per distinct scorable sequence of p
// that the store does not hold yet, and binds the store to p. Unscorable
// items (markers cut by the sequence limits) are recorded as skipped, never
// encoded. The store is append-only: new sequences go to new shards, each
// written atomically, so an interrupted build resumes where it stopped and
// a larger nested subsample encodes only what it adds. One build writes a
// store at a time (an advisory lock on the store directory, taken only when
// there is something to encode).
func BuildCache(ctx context.Context, trunk *Trunk, p *Prepared, spec CacheSpec) (*Cache, error) {
	spec, err := spec.withDefaults()
	if err != nil {
		return nil, err
	}
	par := tensor.Parallel{Workers: spec.IntraOp, Kernel: spec.Kernel}
	kernel := kernelName(par)
	ct := cacheTrunk(trunk.Identity)
	key := CacheKey(ct, kernel, spec.DType)
	dir := filepath.Join(spec.Root, key)
	if err := os.MkdirAll(filepath.Join(dir, "views"), 0o750); err != nil {
		return nil, err
	}
	store := StoreInfo{Contract: CacheContract, Key: key, Trunk: ct, Kernel: kernel, DType: spec.DType}
	c := &Cache{Dir: dir, Store: store}
	// A view whose every sequence is already stored reads the store without
	// its lock: shards are only ever added, each by an atomic rename, so
	// training can read a store while another build appends to it.
	prev, err := readStoreInfo(dir)
	existing := err == nil && prev == store
	if existing {
		if err := c.scan(false); err != nil {
			return nil, err
		}
	}
	c.bindItems(p, spec.ShardSize)
	pl := c.plan(p)
	if !existing || len(pl.todo) > 0 {
		unlock, err := lockStore(dir)
		if err != nil {
			return nil, err
		}
		defer unlock()
		switch prev, err := readStoreInfo(dir); {
		case err == nil && prev != store:
			return nil, fmt.Errorf("train: store %s holds %+v, not %+v", dir, prev, store)
		case errors.Is(err, os.ErrNotExist):
			raw, err := json.MarshalIndent(store, "", "  ")
			if err != nil {
				return nil, err
			}
			if err := writeAtomic(filepath.Join(dir, "store.json"), append(raw, '\n')); err != nil {
				return nil, err
			}
		case err != nil:
			return nil, err
		}
		if err := c.scan(true); err != nil {
			return nil, err
		}
		pl = c.plan(p)
	}
	todo, reused, reusedShards, waiting := pl.todo, pl.reused, pl.reusedShards, pl.waiting
	shards := (len(todo) + spec.ShardSize - 1) / spec.ShardSize
	next := c.nextShard()
	started, cpu0, load0 := time.Now(), processCPU(), LoadAverage()
	var tokens, encoded int
	for s := 0; s < shards; s++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		chunk := todo[s*spec.ShardSize : min(len(todo), (s+1)*spec.ShardSize)]
		shardStart := time.Now()
		keys := make([]string, len(chunk))
		seqs := make([][]int32, len(chunk))
		mks := make([][]int, len(chunk))
		for i, it := range chunk {
			keys[i], seqs[i], mks[i] = it.key, it.ids, it.markers
		}
		toks, err := c.writeShard(ctx, trunk, next+s, keys, seqs, mks, spec, par)
		if err != nil {
			return nil, fmt.Errorf("shard %d: %w", next+s, err)
		}
		for i, k := range keys {
			c.index[k] = storedSeq{next + s, i}
		}
		tokens += toks
		encoded += len(chunk)
		if spec.Progress != nil {
			el := time.Since(started)
			spec.Progress(CacheProgress{
				ShardsDone: s + 1, Shards: shards, ExamplesDone: len(p.Items) - waiting + encoded*waiting/max(1, len(todo)),
				Total: len(p.Items), Encoded: encoded, Encode: len(todo), Elapsed: el,
				ExamplesPerSecond: float64(encoded) / el.Seconds(), ResumedShards: len(reusedShards),
				LastShardExamplesSec: float64(len(chunk)) / time.Since(shardStart).Seconds(),
			})
		}
	}
	secs := time.Since(started).Seconds()
	cpu := (processCPU() - cpu0).Seconds()
	th := Throughput{
		Examples: encoded, Tokens: tokens, Seconds: secs, Workers: spec.Workers, IntraOp: spec.IntraOp, Batch: spec.Batch,
		ResumedShards: len(reusedShards), ReusedSequences: reused, Kernel: kernel, GOMAXPROCS: runtime.GOMAXPROCS(0),
		CPUSeconds: cpu, LoadAverageStart: load0, LoadAverageEnd: LoadAverage(),
	}
	if secs > 0 {
		th.ExamplesPerSecond = float64(encoded) / secs
		th.TokensPerSecond = float64(tokens) / secs
	}
	if cpu > 0 {
		th.TokensPerCPUSecond = float64(tokens) / cpu
	}
	// A build that encoded nothing measured nothing: keep the throughput
	// the view last recorded.
	if encoded == 0 {
		if prev, err := readView(dir, p.SequenceDigest); err == nil && prev.Throughput.Examples > 0 {
			th = prev.Throughput
		}
	}
	c.Info.Throughput, c.Encoded, c.Reused = th, encoded, reused
	c.Info.Shards, c.Info.StoreSequences = c.nextShard(), len(c.index)
	raw, err := json.MarshalIndent(c.Info, "", "  ")
	if err != nil {
		return nil, err
	}
	if err := writeAtomic(viewPath(dir, p.SequenceDigest), append(raw, '\n')); err != nil {
		return nil, err
	}
	return c, nil
}

// pendingSeq is one sequence a build must encode.
type pendingSeq struct {
	key     string
	ids     []int32
	markers []int
}

// buildPlan is what a build of a bound view must encode (each sequence
// once, in item order) and what it reuses.
type buildPlan struct {
	todo         []pendingSeq
	reused       int
	reusedShards map[int]bool
	waiting      int
}

func (c *Cache) plan(p *Prepared) buildPlan {
	pl := buildPlan{reusedShards: map[int]bool{}}
	seen := map[string]bool{}
	for _, it := range p.Items {
		if !it.Scorable {
			continue
		}
		k := c.view[it.Ex.ID]
		s, stored := c.index[k]
		if !stored {
			pl.waiting++
		}
		if seen[k] {
			continue
		}
		seen[k] = true
		if stored {
			pl.reused++
			pl.reusedShards[s.shard] = true
			continue
		}
		pl.todo = append(pl.todo, pendingSeq{k, it.IDs, it.Markers})
	}
	return pl
}

// OpenCache opens a store directory for reading; Bind then attaches the
// items of a prepared dataset.
func OpenCache(dir string) (*Cache, error) {
	store, err := readStoreInfo(dir)
	if err != nil {
		return nil, err
	}
	if store.Contract != CacheContract {
		return nil, fmt.Errorf("train: cache contract %q, want %q", store.Contract, CacheContract)
	}
	c := &Cache{Dir: dir, Store: store}
	if err := c.scan(false); err != nil {
		return nil, err
	}
	return c, nil
}

// Bind attaches the items of p to the store. It fails when a scorable
// item's sequence is not stored. The view's recorded build throughput and
// shard size are kept.
func (c *Cache) Bind(p *Prepared) error {
	c.bindItems(p, 0)
	for _, it := range p.Items {
		if !it.Scorable {
			continue
		}
		if _, ok := c.index[c.view[it.Ex.ID]]; !ok {
			return fmt.Errorf("train: store %s has no features for %s (build them with the same options first)", c.Store.Key, it.Ex.ID)
		}
	}
	if prev, err := readView(c.Dir, p.SequenceDigest); err == nil {
		c.Info.Throughput, c.Info.ShardSize = prev.Throughput, prev.ShardSize
	}
	c.Info.Shards, c.Info.StoreSequences = c.nextShard(), len(c.index)
	return nil
}

// bindItems computes the view: every scorable item's sequence key, the
// skipped items and the sequence-length statistics.
func (c *Cache) bindItems(p *Prepared, shardSize int) {
	s := c.Store
	c.view = map[string]string{}
	c.Info = CacheInfo{Contract: CacheContract, Key: s.Key, DatasetDigest: p.Dataset.Digest(), Instance: p.Instance,
		SequenceDigest: p.SequenceDigest, Trunk: s.Trunk, Kernel: s.Kernel, DType: s.DType, ShardSize: shardSize, Examples: len(p.Items)}
	var lens []int
	distinct := map[string]bool{}
	for _, it := range p.Items {
		if !it.Scorable {
			c.Info.Skipped = append(c.Info.Skipped, SkippedExample{ID: it.Ex.ID, Reason: fmt.Sprintf(
				"option markers truncated: %d of %d fit max_len=%d head_max_len=%d", len(it.Markers), it.Ex.NumOptions(),
				s.Trunk.MaxLen, s.Trunk.HeadMaxLen)})
			continue
		}
		k := SequenceKey(it.IDs, it.Markers)
		c.view[it.Ex.ID] = k
		distinct[k] = true
		lens = append(lens, len(it.IDs))
	}
	c.Info.Sequences = len(distinct)
	slices.Sort(lens)
	if len(lens) > 0 {
		c.Info.SeqLenP50 = lens[percentileIndex(len(lens), 0.5)]
		c.Info.SeqLenP95 = lens[percentileIndex(len(lens), 0.95)]
		c.Info.SeqLenMax = lens[len(lens)-1]
		sum := 0
		for _, l := range lens {
			sum += l
		}
		c.Info.SeqLenMean = math.Round(float64(sum)/float64(len(lens))*100) / 100
	}
}

func readStoreInfo(dir string) (StoreInfo, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "store.json")) //nolint:gosec // G304: cache directory chosen by the operator
	if err != nil {
		return StoreInfo{}, err
	}
	var s StoreInfo
	if err := json.Unmarshal(raw, &s); err != nil {
		return StoreInfo{}, fmt.Errorf("train: %s/store.json: %w", dir, err)
	}
	return s, nil
}

func readView(dir, sequenceDigest string) (CacheInfo, error) {
	raw, err := os.ReadFile(viewPath(dir, sequenceDigest)) //nolint:gosec // G304: inside the store
	if err != nil {
		return CacheInfo{}, err
	}
	var info CacheInfo
	err = json.Unmarshal(raw, &info)
	return info, err
}

// shardNumbers lists the store's shard numbers in order.
func shardNumbers(dir string) ([]int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []int
	for _, e := range entries {
		var n int
		if _, err := fmt.Sscanf(e.Name(), "shard-%d.safetensors", &n); err == nil && e.Name() == filepath.Base(shardPath(dir, n)) {
			out = append(out, n)
		}
	}
	slices.Sort(out)
	return out, nil
}

// scan indexes every stored sequence. A shard that does not parse or
// belongs to another store is left out; when repair is set (a build
// holding the store's lock) it is removed, so its sequences are encoded
// again into a new shard.
func (c *Cache) scan(repair bool) error {
	c.index = map[string]storedSeq{}
	nums, err := shardNumbers(c.Dir)
	if err != nil {
		return err
	}
	for _, n := range nums {
		meta, f, err := readShard(shardPath(c.Dir, n))
		if err == nil {
			_ = f.Close()
		}
		if err != nil || meta.Contract != CacheContract || meta.Key != c.Store.Key || meta.Shard != n {
			if repair {
				if err := os.Remove(shardPath(c.Dir, n)); err != nil {
					return err
				}
			}
			continue
		}
		for i, k := range meta.Seqs {
			if _, dup := c.index[k]; !dup {
				c.index[k] = storedSeq{n, i}
			}
		}
	}
	return nil
}

// nextShard is the number after the highest shard on disk.
func (c *Cache) nextShard() int {
	nums, err := shardNumbers(c.Dir)
	if err != nil || len(nums) == 0 {
		return 0
	}
	return nums[len(nums)-1] + 1
}

// percentileIndex is the nearest-rank index of quantile q in n sorted values.
func percentileIndex(n int, q float64) int {
	return max(0, min(n-1, int(math.Ceil(q*float64(n)))-1))
}

func shardPath(dir string, s int) string {
	return filepath.Join(dir, fmt.Sprintf("shard-%05d.safetensors", s))
}

type shardMeta struct {
	Contract string   `json:"contract"`
	Key      string   `json:"key"`
	Shard    int      `json:"shard"`
	Seqs     []string `json:"sequences"`
}

// writeShard encodes sequences into a new shard. Batches are encoded in
// parallel and gathered in order; a sequence's features do not depend on
// its batch, so the shard's bytes do not depend on Workers or Batch.
func (c *Cache) writeShard(ctx context.Context, trunk *Trunk, s int, keys []string, seqs [][]int32, mks [][]int, spec CacheSpec,
	par tensor.Parallel) (int, error) {
	tokens := 0
	for _, ids := range seqs {
		tokens += len(ids)
	}
	d := trunk.Encoder.Config.HiddenSize
	nb := (len(seqs) + spec.Batch - 1) / spec.Batch
	feats := make([][]Features, nb)
	errs := make([]error, nb)
	tensor.Parallel{Workers: spec.Workers}.Run(nb, func(b int) {
		lo, hi := b*spec.Batch, min(len(seqs), (b+1)*spec.Batch)
		feats[b], errs[b] = trunk.Features(ctx, seqs[lo:hi], mks[lo:hi], par)
	})
	var pooled, markers []float32
	ks := make([]int32, 0, len(seqs))
	lens := make([]int32, 0, len(seqs))
	for b := range feats {
		if errs[b] != nil {
			return 0, errs[b]
		}
		for i, f := range feats[b] {
			pooled = append(pooled, f.Pooled...)
			markers = append(markers, f.Markers...)
			ks = append(ks, int32(f.K))                           //nolint:gosec // G115: option counts are small
			lens = append(lens, int32(len(seqs[b*spec.Batch+i]))) //nolint:gosec // G115: sequence lengths are bounded by max_len
		}
	}
	metaJSON, err := json.Marshal(shardMeta{Contract: CacheContract, Key: c.Store.Key, Shard: s, Seqs: keys})
	if err != nil {
		return 0, err
	}
	pt, err := featureTensor("pooled", []int{len(seqs), d}, pooled, spec.DType)
	if err != nil {
		return 0, err
	}
	mt, err := featureTensor("markers", []int{len(markers) / max(1, d), d}, markers, spec.DType)
	if err != nil {
		return 0, err
	}
	var buf bytes.Buffer
	err = safetensors.Write(&buf, []safetensors.NamedTensor{pt, mt, int32Tensor("k", ks), int32Tensor("seq_len", lens)},
		map[string]string{"meta": string(metaJSON)})
	if err != nil {
		return 0, err
	}
	return tokens, writeAtomic(shardPath(c.Dir, s), buf.Bytes())
}

func int32Tensor(name string, v []int32) safetensors.NamedTensor {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(x)) //nolint:gosec // G115: two's-complement encoding
	}
	return safetensors.NamedTensor{Name: name, DType: safetensors.I32, Shape: []int{len(v)}, Data: b}
}

func featureTensor(name string, shape []int, v []float32, dtype string) (safetensors.NamedTensor, error) {
	if dtype == "f32" {
		return safetensors.Float32Tensor(name, shape, v), nil
	}
	b := make([]byte, 2*len(v))
	for i, x := range v {
		h, ok := Float32ToFloat16(x)
		if !ok {
			return safetensors.NamedTensor{}, fmt.Errorf("train: feature %v overflows float16; use the f32 cache", x)
		}
		binary.LittleEndian.PutUint16(b[2*i:], h)
	}
	return safetensors.NamedTensor{Name: name, DType: safetensors.F16, Shape: shape, Data: b}, nil
}

// Float32ToFloat16 rounds to the nearest IEEE half (ties to even). ok is
// false when a finite value overflows to infinity.
func Float32ToFloat16(f float32) (uint16, bool) {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int((bits >> 23) & 0xff)
	mant := bits & 0x7fffff
	switch {
	case exp == 0xff: // inf or nan
		if mant != 0 {
			return sign | 0x7e00, true
		}
		return sign | 0x7c00, true
	case exp == 0 && mant == 0:
		return sign, true
	}
	e := exp - 127 + 15
	if e >= 0x1f {
		return sign | 0x7c00, false
	}
	if e <= 0 {
		// Subnormal half: shift the implicit-one mantissa right.
		if e < -10 {
			return sign, true
		}
		m := mant | 0x800000
		shift := uint(14 - e)
		half := uint32(1) << (shift - 1)
		r := m >> shift
		rem := m & ((uint32(1) << shift) - 1)
		if rem > half || (rem == half && r&1 == 1) {
			r++
		}
		return sign | uint16(r), true //nolint:gosec // G115: r <= 0x400
	}
	r := mant >> 13
	rem := mant & 0x1fff
	out := uint32(e)<<10 | r //nolint:gosec // G115: e in 1..30
	if rem > 0x1000 || (rem == 0x1000 && r&1 == 1) {
		out++ // may carry into the exponent, which is the correct rounding
	}
	if out >= 0x7c00 {
		return sign | 0x7c00, false
	}
	return sign | uint16(out), true //nolint:gosec // G115: out < 0x7c00
}

func writeAtomic(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func readShard(path string) (shardMeta, *safetensors.File, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return shardMeta{}, nil, err
	}
	var meta shardMeta
	if err := json.Unmarshal([]byte(f.Header.Metadata["meta"]), &meta); err != nil {
		_ = f.Close()
		return shardMeta{}, nil, fmt.Errorf("train: shard %s metadata: %w", path, err)
	}
	return meta, f, nil
}

// Records loads the features of every scorable item the cache is bound to,
// keyed by item id. Items that share a sequence share its features.
func (c *Cache) Records() (map[string]*Record, error) {
	if c.view == nil {
		return nil, errors.New("train: the cache is not bound to a prepared dataset")
	}
	byShard := map[int]map[string][]string{}
	for id, k := range c.view {
		s, ok := c.index[k]
		if !ok {
			return nil, fmt.Errorf("train: store %s has no features for %s", c.Store.Key, id)
		}
		if byShard[s.shard] == nil {
			byShard[s.shard] = map[string][]string{}
		}
		byShard[s.shard][k] = append(byShard[s.shard][k], id)
	}
	out := make(map[string]*Record, len(c.view))
	nums := make([]int, 0, len(byShard))
	for n := range byShard {
		nums = append(nums, n)
	}
	slices.Sort(nums)
	for _, n := range nums {
		want := byShard[n]
		err := c.readShardRecords(n, func(key string, f Features, seqLen int) {
			for _, id := range want[key] {
				out[id] = &Record{ID: id, SeqLen: seqLen, Features: f}
			}
		})
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// readShardRecords calls fn for every sequence of shard n that the index
// resolves to that shard and position.
func (c *Cache) readShardRecords(n int, fn func(key string, f Features, seqLen int)) error {
	meta, f, err := readShard(shardPath(c.Dir, n))
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	if meta.Key != c.Store.Key || meta.Shard != n {
		return fmt.Errorf("train: shard %d belongs to store %s shard %d", n, meta.Key, meta.Shard)
	}
	tensors := map[string]safetensors.Tensor{}
	for _, name := range []string{"pooled", "markers", "k", "seq_len"} {
		t, err := f.Tensor(name)
		if err != nil {
			return err
		}
		tensors[name] = t
	}
	pooled, err := tensors["pooled"].Float32()
	if err != nil {
		return err
	}
	markers, err := tensors["markers"].Float32()
	if err != nil {
		return err
	}
	ks, lens := int32s(tensors["k"]), int32s(tensors["seq_len"])
	d := c.Store.Trunk.HiddenSize
	if len(ks) != len(meta.Seqs) || len(lens) != len(meta.Seqs) || len(pooled) != len(meta.Seqs)*d {
		return fmt.Errorf("train: shard %d tensors do not match its %d sequences", n, len(meta.Seqs))
	}
	off := 0
	for i, key := range meta.Seqs {
		k := int(ks[i])
		if (off+k)*d > len(markers) {
			return fmt.Errorf("train: shard %d markers are short", n)
		}
		if s, ok := c.index[key]; ok && s.shard == n && s.pos == i {
			fn(key, Features{
				Pooled:  pooled[i*d : (i+1)*d : (i+1)*d],
				Markers: markers[off*d : (off+k)*d : (off+k)*d], K: k,
			}, int(lens[i]))
		}
		off += k
	}
	return nil
}

func int32s(t safetensors.Tensor) []int32 {
	out := make([]int32, len(t.Data)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(t.Data[4*i:])) //nolint:gosec // G115: two's-complement decoding
	}
	return out
}
