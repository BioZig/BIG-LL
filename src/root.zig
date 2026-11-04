pub const types = @import("core/types.zig");
pub const prng = @import("core/prng.zig");
pub const linkage = @import("core/linkage.zig");
pub const crossover = @import("core/crossover.zig");
pub const dedup = @import("core/dedup.zig");
pub const engine = @import("core/engine.zig");
pub const island = @import("core/island.zig");
pub const pool = @import("core/pool.zig");
pub const pareto = @import("core/pareto.zig");
pub const shm = @import("ipc/shm.zig");
pub const c_api = @import("ipc/c_api.zig");
comptime {
    _ = c_api;
}
pub const distributed = @import("ipc/distributed.zig");
pub const data_format = @import("data/format.zig");
pub const data_loader = @import("data/loader.zig");
pub const data_generator = @import("data/generator.zig");
pub const data_importer = @import("data/importer.zig");

pub const gray_code = @import("adapters/gray_code.zig");
pub const random_key = @import("adapters/random_key.zig");
pub const minlp = @import("adapters/minlp.zig");
pub const boolean_set = @import("adapters/boolean_set.zig");

pub const airfoil = @import("domains/airfoil.zig");
pub const logistics = @import("domains/logistics.zig");
pub const power_grid = @import("domains/power_grid.zig");
pub const bioinformatics = @import("domains/bioinformatics.zig");
pub const finance = @import("domains/finance.zig");
pub const telecom = @import("domains/telecom.zig");

pub const Chromosome = types.Chromosome;
pub const Population = types.Population;
pub const DynamicChromosome = types.DynamicChromosome;
pub const DynamicPopulation = types.DynamicPopulation;
pub const Xoshiro256 = prng.Xoshiro256;
pub const LinkageEngine = linkage.LinkageEngine;
pub const DynamicLinkageEngine = linkage.DynamicLinkageEngine;
pub const CrossoverEngine = crossover.CrossoverEngine;
pub const DynamicCrossoverEngine = crossover.DynamicCrossoverEngine;
pub const L1Dedup = dedup.L1Dedup;
pub const AdaptiveDedup = dedup.AdaptiveDedup;
pub const GeneticEngine = engine.GeneticEngine;
pub const GeneticEngineCustom = engine.GeneticEngineCustom;
pub const DynamicGeneticEngine = engine.DynamicGeneticEngine;
pub const EngineOptions = engine.EngineOptions;
pub const IslandEngine = island.IslandEngine;
pub const StaticThreadPool = pool.StaticThreadPool;
pub const ParetoChromosome = pareto.ParetoChromosome;
pub const ParetoEngine = pareto.ParetoEngine;
pub const dominates = pareto.dominates;

pub const GrayDecoder = gray_code.GrayDecoder;
pub const RandomKeyDecoder = random_key.RandomKeyDecoder;
pub const MinlpDecoder = minlp.MinlpDecoder;
pub const AugmentedLagrangianConstraint = minlp.AugmentedLagrangianConstraint;
pub const BooleanSetDecoder = boolean_set.BooleanSetDecoder;

pub const AirfoilDomain = airfoil.AirfoilDomain;
pub const LogisticsDomain = logistics.LogisticsDomain;
pub const PowerGridDomain = power_grid.PowerGridDomain;
pub const BioinformaticsDomain = bioinformatics.BioinformaticsDomain;
pub const FinanceDomain = finance.FinanceDomain;
pub const TelecomDomain = telecom.TelecomDomain;

pub const ParameterizedLogisticsDomain = logistics.ParameterizedLogisticsDomain;
pub const ParameterizedPowerGridDomain = power_grid.ParameterizedPowerGridDomain;
pub const ParameterizedFinanceDomain = finance.ParameterizedFinanceDomain;
pub const ParameterizedBioinformaticsDomain = bioinformatics.ParameterizedBioinformaticsDomain;
pub const ParameterizedTelecomDomain = telecom.ParameterizedTelecomDomain;

pub const SharedMemoryRegion = shm.SharedMemoryRegion;
pub const SharedPopulationHeader = shm.SharedPopulationHeader;
pub const BigLlEvalFn = c_api.BigLlEvalFn;
pub const BigLlOptions = c_api.BigLlOptions;
pub const createEngine = c_api.createEngine;

pub const BigDHeader = data_format.BigDHeader;
pub const DomainTag = data_format.DomainTag;
pub const BigDInstance = data_loader.BigDInstance;
pub const MemoryMappedInstance = data_loader.MemoryMappedInstance;
pub const StaticBufferInstance = data_loader.StaticBufferInstance;

pub const BlockChromosome = types.BlockChromosome;
pub const BITS_PER_BLOCK = types.BITS_PER_BLOCK;
pub const WORDS_PER_BLOCK = types.WORDS_PER_BLOCK;
pub const BYTES_PER_BLOCK = types.BYTES_PER_BLOCK;

pub const DistributedTransport = distributed.DistributedTransport;
pub const TransportType = distributed.TransportType;
pub const FrameHeader = distributed.FrameHeader;
pub const NetworkMailbox = distributed.NetworkMailbox;

// Apple Silicon Dedicated Metal GPU Batch Acceleration
pub const metal = @import("metal/metal_engine.zig");
pub const MetalEngine = metal.MetalEngine;


