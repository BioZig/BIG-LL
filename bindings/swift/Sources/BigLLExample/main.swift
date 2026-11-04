import Foundation
import BigLL

print("=========================================================")
print("  BIG-LL Swift Native Integration Showcase (Apple Silicon)")
print("=========================================================")

let bitLength = 256
let population = 128
let generations = 50

print("Initializing BigLLSolver:")
print("  Chromosome Dimension: \(bitLength) bits (4 x 64-bit words)")
print("  Population Size:      \(population) individuals")
print("  Generations:          \(generations) epochs")

let start = DispatchTime.now()

let solver = BigLLSolver(chromosomeBits: bitLength, populationSize: population, seed: 1337) { words in
    // Hardware-accelerated popcount across 64-bit words
    var ones = 0
    for w in words {
        ones += w.nonzeroBitCount
    }
    return Double(ones)
}

print("Running evolutionary loop...")
solver.run(generations: generations)

let end = DispatchTime.now()
let elapsedNanoseconds = end.uptimeNanoseconds - start.uptimeNanoseconds
let elapsedMs = Double(elapsedNanoseconds) / 1_000_000.0

let bestFit = solver.bestFitness
let genes = solver.bestGenes

print("\nOptimization Complete:")
print("  Best Fitness:     \(bestFit) / \(bitLength).0")
print("  Generation:       \(solver.currentGeneration)")
print("  Elapsed Time:     \(String(format: "%.2f", elapsedMs)) ms")
print("  Best Words (Hex): \(genes.map { String(format: "0x%016llX", $0) }.joined(separator: " "))")
print("=========================================================")
