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
        guard let renderer = context.coordinator.renderer else { return }
        renderer.highConfidenceOnly = state.highConfidenceOnly
        renderer.colorMode = state.colorMode
        renderer.isCapturing = state.isCapturing
        renderer.pointSizeScale = Float(state.pointSizeScale)
        renderer.fadeEnabled = state.fadeEnabled
        renderer.fadeDurationSeconds = Float(state.fadeDuration)
        renderer.shapeMode = state.shapeMode

        if state.resetRequested {
            renderer.reset()
            DispatchQueue.main.async {
                state.resetRequested = false
            }
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var renderer: PointCloudRenderer?
    }
}
