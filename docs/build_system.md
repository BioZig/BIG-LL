# Component Documentation: Build System (`build.zig`)

## 1. Specification

The build system targets Zig `0.16.0-dev` and orchestrates native vectorization, module exports, CLI compilation, and parallel test runners.

### Targets & Steps

1. **`big_ll` (Module):**
   * Root source: `src/root.zig`.
   * Standard library package importable by consumers, tests, and domain adapters.

2. **`big_ll` (Executable):**
   * Root source: `src/main.zig`.
   * Compiled binary installed to `zig-out/bin/big_ll`.

3. **`test` (Step):**
   * Executes all unit test suites concurrently (`tests/test_types.zig`, `tests/test_prng.zig`, etc.).
   * Runs directly against the monomorphized `big_ll` module.

### Command Invocations

* Build binary:
  ```bash
  zig build -Doptimize=ReleaseFast
  ```
* Run test suite:
  ```bash
  zig build test
  ```
* Execute CLI:
  ```bash
  zig build run -- [args]
  ```
