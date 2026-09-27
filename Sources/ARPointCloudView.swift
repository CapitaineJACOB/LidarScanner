import SwiftUI
import MetalKit
import ARKit

struct ARPointCloudView: UIViewRepresentable {
    @ObservedObject var state: ScannerState

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.backgroundColor = .black
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.contentScaleFactor = UIScreen.main.scale
        view.preferredFramesPerSecond = 60

        let renderer = PointCloudRenderer(metalDevice: view.device!, view: view, state: state)
        context.coordinator.renderer = renderer
        view.delegate = renderer
        renderer.start()
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.renderer?.highConfidenceOnly = state.highConfidenceOnly
        context.coordinator.renderer?.colorMode = state.colorMode
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var renderer: PointCloudRenderer?
    }
}
