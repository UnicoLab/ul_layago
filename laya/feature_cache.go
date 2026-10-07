package laya

import (
	"container/list"
	"crypto/sha256"
	"encoding/binary"
	"runtime"
	"slices"
	"sync"

	"github.com/UnicoLab/ul_layago/tensor"
)

// CacheStats is a snapshot of a base's optional in-memory full-sequence cache.
type CacheStats struct {
	BudgetBytes, Bytes      int64
	Entries                 int
	Hits, Misses, Evictions uint64
}
type featureKey [32]byte
type featureEntry struct {
	key    featureKey
	hidden []float32
	bytes  int64
}
type featureCache struct {
	mu      sync.Mutex
	entries map[featureKey]*list.Element
	lru     list.List
	state   CacheStats
}

func newFeatureCache(budget int64) *featureCache {
	if budget == 0 {
		return nil
	}
	return &featureCache{entries: map[featureKey]*list.Element{}, state: CacheStats{BudgetBytes: budget}}
}
func sequenceKey(ids []int32, p tensor.Parallel) featureKey {
	h := sha256.New()
	h.Write([]byte(runtime.GOARCH + "/" + p.EffectiveKernel().String()))
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(len(ids)))
	h.Write(buf[:])
	for _, id := range ids {
		binary.LittleEndian.PutUint32(buf[:4], uint32(id))
		h.Write(buf[:4])
	}
	var key featureKey
	copy(key[:], h.Sum(nil))
	return key
}
func (c *featureCache) get(key featureKey) ([]float32, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.entries[key]
	if !ok {
		c.state.Misses++
		return nil, false
	}
	c.state.Hits++
	c.lru.MoveToFront(entry)
	return slices.Clone(entry.Value.(featureEntry).hidden), true
}
func (c *featureCache) put(key featureKey, hidden []float32) {
	bytes := int64(len(hidden))*4 + 128
	c.mu.Lock()
	defer c.mu.Unlock()
	if bytes > c.state.BudgetBytes {
		return
	}
	if entry, ok := c.entries[key]; ok {
		c.lru.MoveToFront(entry)
		return
	}
	for c.state.Bytes+bytes > c.state.BudgetBytes {
		last := c.lru.Back()
		e := last.Value.(featureEntry)
		delete(c.entries, e.key)
		c.lru.Remove(last)
		c.state.Bytes -= e.bytes
		c.state.Evictions++
	}
	c.entries[key] = c.lru.PushFront(featureEntry{key: key, hidden: slices.Clone(hidden), bytes: bytes})
	c.state.Bytes += bytes
}
func (c *featureCache) stats() CacheStats {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := c.state
	out.Entries = len(c.entries)
	return out
}
