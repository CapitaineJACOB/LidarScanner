import Foundation
import Metal
import MetalKit
import ARKit
import simd

final class PointCloudRenderer: NSObject, MTKViewDelegate, ARSessionDelegate {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private weak var mtkView: MTKView?
    private let state: ScannerState

    private let session = ARSession()
    private var textureCache: CVMetalTextureCache!

    private var computePipeline: MTLComputePipelineState!
    private var renderPipeline: MTLRenderPipelineState!
    private var depthStencilState: MTLDepthStencilState!

    private var vertexBuffer: MTLBuffer?
    private var gridWidth = 0
    private var gridHeight = 0

    var highConfidenceOnly: Bool = true
    var colorMode: ColorMode = .depth

    private var viewportSize = CGSize(width: 1, height: 1)
    private let inFlightSemaphore = DispatchSemaphore(value: 3)

    init(metalDevice: MTLDevice, view: MTKView, state: ScannerState) {
        self.device = metalDevice
        self.commandQueue = metalDevice.makeCommandQueue()!
        self.mtkView = view
        self.state = state
        super.init()

        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
        buildPipelines()
        session.delegate = self
    }

    func start() {
        guard ARWorldTrackingConfiguration.isSupported else { return }
        let config = ARWorldTrackingConfiguration()
        config.environmentTexturing = .none
        config.planeDetection = []

        if type(of: config).supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        // Sur iPhone 17 Pro le LiDAR fournit aussi la profondeur "smoothée"
        if type(of: config).supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        }
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    // MARK: - Pipelines

    private func buildPipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Impossible de charger la librairie Metal par défaut")
        }

        let unprojectFn = library.makeFunction(name: "unprojectDepth")!
        computePipeline = try! device.makeComputePipelineState(function: unprojectFn)

        let vertexFn = library.makeFunction(name: "pointCloudVertex")!
        let fragmentFn = library.makeFunction(name: "pointCloudFragment")!

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFn
        pipelineDescriptor.fragmentFunction = fragmentFn
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = false

        renderPipeline = try! device.makeRenderPipelineState(descriptor: pipelineDescriptor)

        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .less
        depthDescriptor.isDepthWriteEnabled = true
        depthStencilState = device.makeDepthStencilState(descriptor: depthDescriptor)
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else { return }

        let depthMap = depthData.depthMap
        let confidenceMap = depthData.confidenceMap ?? makeDefaultConfidenceMap(matching: depthMap)

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)

        if gridWidth != width || gridHeight != height || vertexBuffer == nil {
            gridWidth = width
            gridHeight = height
            let count = width * height
            vertexBuffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Float>>.stride * 2,
                                              options: .storageModeShared)
        }

        guard let depthTexture = makeTexture(from: depthMap, pixelFormat: .r32Float),
              let confidenceTexture = makeTexture(from: confidenceMap, pixelFormat: .r8Uint),
              let yTexture = makeTexture(from: frame.capturedImage, plane: 0, pixelFormat: .r8Unorm),
              let cbcrTexture = makeTexture(from: frame.capturedImage, plane: 1, pixelFormat: .rg8Unorm),
              let vertexBuffer = vertexBuffer,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return }

        var uniforms = PointCloudUniforms()
        uniforms.localToWorld = frame.camera.transform
        let intrinsics = frame.camera.intrinsics
        uniforms.cameraIntrinsicsInversed = intrinsics.inverse
        uniforms.cameraResolution = SIMD2<Float>(Float(frame.camera.imageResolution.width),
                                                  Float(frame.camera.imageResolution.height))
        uniforms.viewProjectionMatrix = matrix_identity_float4x4 // recalculé au moment du rendu
        uniforms.colorMode = 0
        uniforms.highConfidenceOnly = 0
        uniforms.minDepth = 0.15
        uniforms.maxDepth = 6.0
        uniforms.gridWidth = Int32(width)
        uniforms.gridHeight = Int32(height)

        encoder.setComputePipelineState(computePipeline)
        encoder.setTexture(depthTexture, index: 0)
        encoder.setTexture(confidenceTexture, index: 1)
        encoder.setTexture(yTexture, index: 2)
        encoder.setTexture(cbcrTexture, index: 3)
        encoder.setBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setBytes(&uniforms, length: MemoryLayout<PointCloudUniforms>.stride, index: 1)

        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(width: (width + 7) / 8, height: (height + 7) / 8, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        commandBuffer.commit()

        DispatchQueue.main.async { [weak self] in
            self?.state.pointCount = width * height
            self?.state.minDepth = 0.15
            self?.state.maxDepth = 6.0
        }
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        viewportSize = size
    }

    func draw(in view: MTKView) {
        guard let frame = session.currentFrame,
              let vertexBuffer = vertexBuffer,
              let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer()
        else { return }

        _ = inFlightSemaphore.wait(timeout: .distantFuture)
        commandBuffer.addCompletedHandler { [weak self] _ in self?.inFlightSemaphore.signal() }

        let viewMatrix = frame.camera.viewMatrix(for: .portrait)
        let projMatrix = frame.camera.projectionMatrix(for: .portrait,
                                                         viewportSize: viewportSize,
                                                         zNear: 0.05, zFar: 20)
        var uniforms = PointCloudUniforms()
        uniforms.viewProjectionMatrix = projMatrix * viewMatrix
        uniforms.colorMode = Int32(colorModeIndex())
        uniforms.highConfidenceOnly = highConfidenceOnly ? 1 : 0
        uniforms.minDepth = 0.15
        uniforms.maxDepth = 6.0
        uniforms.gridWidth = Int32(gridWidth)
        uniforms.gridHeight = Int32(gridHeight)

        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }
        encoder.setRenderPipelineState(renderPipeline)
        encoder.setDepthStencilState(depthStencilState)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<PointCloudUniforms>.stride, index: 1)
        encoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: gridWidth * gridHeight)
        encoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    private func colorModeIndex() -> Int {
        switch colorMode {
        case .depth: return 0
        case .confidence: return 1
        case .rgb: return 2
        }
    }

    // MARK: - Helpers

    private func makeTexture(from pixelBuffer: CVPixelBuffer, pixelFormat: MTLPixelFormat) -> MTLTexture? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil, textureCache, pixelBuffer, nil, pixelFormat, width, height, 0, &cvTexture)
        guard status == kCVReturnSuccess, let cvTexture = cvTexture else { return nil }
        return CVMetalTextureGetTexture(cvTexture)
    }

    private func makeTexture(from pixelBuffer: CVPixelBuffer, plane: Int, pixelFormat: MTLPixelFormat) -> MTLTexture? {
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, plane)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, plane)
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil, textureCache, pixelBuffer, nil, pixelFormat, width, height, plane, &cvTexture)
        guard status == kCVReturnSuccess, let cvTexture = cvTexture else { return nil }
        return CVMetalTextureGetTexture(cvTexture)
    }

    private func makeDefaultConfidenceMap(matching depthMap: CVPixelBuffer) -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_OneComponent8, nil, &buffer)
        if let buffer = buffer {
            CVPixelBufferLockBaseAddress(buffer, [])
            if let base = CVPixelBufferGetBaseAddress(buffer) {
                memset(base, 2, CVPixelBufferGetBytesPerRow(buffer) * height)
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
        }
        return buffer!
    }
}

// Pont Swift vers la struct C définie dans ShaderTypes.h
struct PointCloudUniforms {
    var viewProjectionMatrix = matrix_identity_float4x4
    var localToWorld = matrix_identity_float4x4
    var cameraIntrinsicsInversed = matrix_identity_float3x3
    var cameraResolution = SIMD2<Float>(1, 1)
    var colorMode: Int32 = 0
    var highConfidenceOnly: Int32 = 1
    var minDepth: Float = 0.15
    var maxDepth: Float = 6.0
    var gridWidth: Int32 = 0
    var gridHeight: Int32 = 0
}
