// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "BigLL",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(name: "BigLL", targets: ["BigLL"]),
        .executable(name: "BigLLExample", targets: ["BigLLExample"])
    ],
    targets: [
        .target(
            name: "CBigLL",
            path: "Sources/CBigLL",
            publicHeadersPath: "include"
        ),
        .target(
            name: "BigLL",
            dependencies: ["CBigLL"],
            linkerSettings: [
                .unsafeFlags([
                    "-L../../zig-out/lib",
                    "-lbig_ll",
                    "-Xlinker", "-rpath", "-Xlinker", "@loader_path/../../../../zig-out/lib",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../../zig-out/lib",
                    "-Xlinker", "-rpath", "-Xlinker", "../../zig-out/lib"
                ])
            ]
        ),
        .executableTarget(
            name: "BigLLExample",
            dependencies: ["BigLL"]
        ),
        .testTarget(
            name: "BigLLTests",
            dependencies: ["BigLL"]
        )
    ]
)
