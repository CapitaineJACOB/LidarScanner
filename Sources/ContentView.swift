import SwiftUI

struct ContentView: View {
    @StateObject private var state = ScannerState()

    var body: some View {
        ZStack(alignment: .bottom) {
            ARPointCloudView(state: state)
                .ignoresSafeArea()

            VStack(spacing: 18) {
                HStack {
                    Label("\(state.pointCount)", systemImage: "circle.grid.3x3.fill")
                        .font(.system(size: 15, weight: .semibold, design: .monospaced))
                    Spacer()
                    Button {
                        state.resetRequested = true
                    } label: {
                        Image(systemName: "arrow.counterclockwise.circle.fill")
                            .font(.system(size: 20))
                    }
                    Button {
                        state.showSettings = true
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                            .font(.system(size: 18))
                    }
                }
                .foregroundColor(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))

                VStack(spacing: 8) {
                    CaptureButton(isCapturing: $state.isCapturing)
                    Text(state.isCapturing ? "Scan en cours…" : "Maintenir pour scanner")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(.white.opacity(0.85))
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 40)
        }
        .background(Color.black)
        .sheet(isPresented: $state.showSettings) {
            SettingsView(state: state)
        }
    }
}

struct CaptureButton: View {
    @Binding var isCapturing: Bool

    var body: some View {
        Circle()
            .fill(isCapturing ? Color.green : Color.white)
            .frame(width: 84, height: 84)
            .overlay(
                Image(systemName: "dot.radiowaves.left.and.right")
                    .font(.system(size: 30, weight: .semibold))
                    .foregroundColor(.black)
            )
            .scaleEffect(isCapturing ? 0.9 : 1.0)
            .animation(.easeOut(duration: 0.12), value: isCapturing)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        if !isCapturing { isCapturing = true }
                    }
                    .onEnded { _ in
                        isCapturing = false
                    }
            )
    }
}

struct SettingsView: View {
    @ObservedObject var state: ScannerState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            Form {
                Section("Affichage") {
                    Picker("Couleur", selection: $state.colorMode) {
                        Text("Profondeur").tag(ColorMode.depth)
                        Text("Confiance").tag(ColorMode.confidence)
                        Text("Caméra").tag(ColorMode.rgb)
                    }
                    Picker("Forme des points", selection: $state.shapeMode) {
                        Text("Points ronds").tag(PointShape.dot)
                        Text("Traits horizontaux").tag(PointShape.horizontal)
                    }
                    Toggle("Confiance haute uniquement", isOn: $state.highConfidenceOnly)
                }

                Section("Taille des points") {
                    Slider(value: $state.pointSizeScale, in: 0.5...4.0, step: 0.1)
                    Text(String(format: "%.1fx", state.pointSizeScale))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Section("Disparition des anciens points") {
                    Toggle("Activer le fondu", isOn: $state.fadeEnabled)
                    Slider(value: $state.fadeDuration, in: 1...20, step: 0.5)
                        .disabled(!state.fadeEnabled)
                    Text(String(format: "%.1f secondes avant disparition", state.fadeDuration))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Section {
                    Button(role: .destructive) {
                        state.resetRequested = true
                    } label: {
                        Text("Réinitialiser le nuage de points")
                    }
                }
            }
            .navigationTitle("Réglages")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Fermer") { dismiss() }
                }
            }
        }
    }
}

enum ColorMode {
    case depth, confidence, rgb
}

enum PointShape {
    case dot, horizontal
}

final class ScannerState: ObservableObject {
    @Published var pointCount: Int = 0
    @Published var minDepth: Float = 0
    @Published var maxDepth: Float = 0
    @Published var highConfidenceOnly: Bool = false
    @Published var colorMode: ColorMode = .depth
    @Published var resetRequested: Bool = false
    @Published var isCapturing: Bool = false
    @Published var showSettings: Bool = false
    @Published var pointSizeScale: Double = 1.6
    @Published var fadeEnabled: Bool = true
    @Published var fadeDuration: Double = 6.0
    @Published var shapeMode: PointShape = .dot
}
