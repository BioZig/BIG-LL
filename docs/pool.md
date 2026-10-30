# Component Documentation: Static Thread Pool (`pool.zig`)

## 1. Specification Overview

The `pool.zig` module provides `StaticThreadPool`, a zero-allocation, cache-aligned persistent worker pool engineered for high-throughput evolutionary computing loops on modern symmetrical and asymmetrical multi-core processors.

## 2. Architectural Invariants

### Persistent Worker Lifecycle
Worker threads are spawned once during `init()` and joined during `deinit()`. Ephemeral thread spawning inside generational step loops is strictly prohibited ($O(0)$ runtime heap allocation and zero OS thread creation overhead per generation).

### 64-Byte Cache Line Alignment and False Sharing Immunity
Each worker descriptor is declared with explicit hardware cache alignment:
```zig
command: u32 align(64) = 0,
done: u32 = 1,
```
Command flags (`command`) and completion flags (`done`) reside on isolated 64-byte L1 data cache lines across cores. This eliminates cache coherence invalidation ping-pong and bus locks during high-frequency dispatch cycles.

### Lock-Free Signaling Protocol
Workers idle-wait using acquire-release atomic load/store semantics without kernel futex round-trips:
* Idle loop:
  $$\text{spin on } @\text{atomicLoad}(u32, \&\text{worker.command}, .acquire)$$
* Main thread task publication: Writes task context and function pointers, followed by atomic release:
  $$@\text{atomicStore}(u32, \&\text{worker.command}, 1, .release)$$
* Worker completion publication:
  $$@\text{atomicStore}(u32, \&\text{worker.done}, 1, .release)$$
* Direct primary dispatch: The calling thread executes worker slot 0 directly, eliminating dispatch latency for the first partition.
* Zero runtime heap allocations during `dispatch()`.

## 3. SoC Affinity and Core Heterogeneity (Apple Silicon & ARM big.LITTLE)

### Kernel QoS Class Pinning
Modern commercial SoCs (Apple M-series, Snapdragon X Elite, Intel Alder/Raptor Lake) feature asymmetric core clusters consisting of high-frequency Performance (P) cores and low-power Efficiency (E) cores. In unconstrained OS scheduling, parallel worker threads risk dispatch to E-cores, causing straggler effects where a 3x slower thread stalls the entire synchronized generation.

To guarantee execution on high-throughput execution units, `pool.zig` invokes native platform QoS configuration upon thread initialization:
```zig
pub fn setHighPerformanceAffinity() void {
    if (builtin.os.tag == .macos or builtin.os.tag == .ios or builtin.os.tag == .watchos or builtin.os.tag == .tvos) {
        _ = pthread_set_qos_class_self_np(0x21, 0); // QOS_CLASS_USER_INTERACTIVE
    }
}
```
* **Target:** `QOS_CLASS_USER_INTERACTIVE` (`0x21`).
* **Kernel Dispatch Policy:** Signals Darwin kernel thread scheduler to bind workers strictly to Firestorm / Avalanche P-core clusters, bypassing Icestorm / Blizzard E-cores and stabilizing throughput variance.
