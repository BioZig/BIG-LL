import Foundation
import CBigLL

public typealias EvaluationFunction = ([UInt64]) -> Double

public final class BigLLSolver {
    private var handle: big_ll_handle_t?
    private var boxPointer: UnsafeMutableRawPointer?
    public let chromosomeBits: Int
    public let populationSize: Int

    public init(
        chromosomeBits: Int,
        populationSize: Int,
        seed: UInt64 = 42,
        evaluator: @escaping EvaluationFunction
    ) {
        self.chromosomeBits = chromosomeBits
        self.populationSize = populationSize

        let wrapper = Box(evaluator)
        let unmanaged = Unmanaged.passRetained(wrapper)
        let contextPtr = unmanaged.toOpaque()
        self.boxPointer = contextPtr

        let cEval: big_ll_eval_fn = { genesPtr, wordCount, userData in
            guard let userData = userData, let genesPtr = genesPtr else { return 0.0 }
            let box = Unmanaged<Box<EvaluationFunction>>.fromOpaque(userData).takeUnretainedValue()
            let buffer = UnsafeBufferPointer(start: genesPtr, count: wordCount)
            return box.value(Array(buffer))
        }

        self.handle = big_ll_create(
            chromosomeBits,
            populationSize,
            seed,
            cEval,
            contextPtr
        )
    }

    deinit {
        if let handle = handle {
            big_ll_destroy(handle)
        }
        if let ptr = boxPointer {
            Unmanaged<Box<EvaluationFunction>>.fromOpaque(ptr).release()
        }
    }

    @discardableResult
    public func step() -> Bool {
        guard let handle = handle else { return false }
        return big_ll_step(handle) == 0
    }

    @discardableResult
    public func run(generations: Int) -> Bool {
        guard let handle = handle else { return false }
        return big_ll_run(handle, generations) == 0
    }

    public var bestFitness: Double {
        guard let handle = handle else { return 0.0 }
        return big_ll_get_best_fitness(handle)
    }

    public var currentGeneration: Int {
        guard let handle = handle else { return 0 }
        return big_ll_get_generation(handle)
    }

    public var bestGenes: [UInt64] {
        guard let handle = handle else { return [] }
        let words = big_ll_get_word_count(handle)
        var buffer = [UInt64](repeating: 0, count: words)
        _ = big_ll_get_best_genes(handle, &buffer, words)
        return buffer
    }
}

private final class Box<T> {
    let value: T
    init(_ value: T) {
        self.value = value
    }
}
