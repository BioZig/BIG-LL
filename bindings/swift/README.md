# BigLL Swift Package

Native Swift bindings for BIG-LL (Bitsliced Information-Geometric Linkage Learning) on Apple Silicon.

## Overview

`BigLL` exposes the zero-allocation BIG-LL evolutionary kernel directly to Swift runtimes on macOS via a C-ABI bridge. Host evaluation functions execute natively in Swift while chromosome bitboards, in-register linkage learning, and hardware SIMD operations are managed by the Zig / Metal engine.

## Requirements

* macOS 13.0+ (ARM64 Apple Silicon)
* Swift 5.9+ / Xcode 15+
* Built `libbig_ll.dylib` (from project root: `zig build`)

## Usage

```swift
import BigLL

let bitLength = 256
let population = 128
let generations = 50

let solver = BigLLSolver(chromosomeBits: bitLength, populationSize: population, seed: 42) { words in
    // Compute objective evaluation directly in Swift
    var ones = 0
    for w in words {
        ones += w.nonzeroBitCount
    }
    return Double(ones)
}

solver.run(generations: generations)

print("Best Fitness: \(solver.bestFitness)")
print("Best Genes: \(solver.bestGenes)")
```

## Running the Showcase

From `bindings/swift`:

```bash
swift run BigLLExample
swift test
```
