import XCTest
@testable import BigLL

final class BigLLTests: XCTestCase {
    func testOneMaxConvergence() {
        let bitLength = 128
        let population = 128
        let generations = 25

        let solver = BigLLSolver(chromosomeBits: bitLength, populationSize: population, seed: 42) { words in
            var ones = 0
            for w in words {
                ones += w.nonzeroBitCount
            }
            return Double(ones)
        }

        let initialFitness = solver.bestFitness
        XCTAssertGreaterThanOrEqual(initialFitness, 0.0)

        let success = solver.run(generations: generations)
        XCTAssertTrue(success)
        XCTAssertEqual(solver.currentGeneration, generations)

        let finalFitness = solver.bestFitness
        XCTAssertGreaterThan(finalFitness, initialFitness)
        XCTAssertLessThanOrEqual(finalFitness, Double(bitLength))

        let genes = solver.bestGenes
        XCTAssertEqual(genes.count, (bitLength + 63) / 64)
    }
}
