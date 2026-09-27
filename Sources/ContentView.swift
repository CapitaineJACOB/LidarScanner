import SwiftUI

struct ContentView: View {
    @StateObject private var state = ScannerState()

    var body: some View {
        ZStack(alignment: .bottom) {
            ARPointCloudView(state: state)
                .ignoresSafeArea()

            VStack(spacing: 12) {
                HStack {
                    Label("\(state.pointCount)", systemImage: "circle.grid.3x3.fill")
                        .font(.system(size: 15, weight: .semibold, design: .monospaced))
                    Spacer()
                    Text(String(format: "%.2f m - %.2f m", state.minDepth, state.maxDepth))
                        .font(.system(size: 13, weight: .medium, design: .monospaced))
                    Button {
                        state.resetRequested = true
                    } label: {
                        Image(systemName: "arrow.counterclockwise.circle.fill")
                            .font(.system(size: 20))
                    }
                }
                .foregroundColor(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))

                HStack(spacing: 10) {
                    Toggle(isOn: $state.highConfidenceOnly) {
                        Text("Confiance haute uniquement")
                            .font(.system(size: 13, weight: .medium))
                    }
                    .toggleStyle(.switch)
                    .tint(.green)
                }
                .foregroundColor(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))

                Picker("Couleur", selection: $state.colorMode) {
                    Text("Profondeur").tag(ColorMode.depth)
                    Text("Confiance").tag(ColorMode.confidence)
                    Text("Caméra").tag(ColorMode.rgb)
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 40)
        }
        .background(Color.black)
    }
}

enum ColorMode {
    case depth, confidence, rgb
}

final class ScannerState: ObservableObject {
    @Published var pointCount: Int = 0
    @Published var minDepth: Float = 0
    @Published var maxDepth: Float = 0
    @Published var highConfidenceOnly: Bool = true
    @Published var colorMode: ColorMode = .depth
    @Published var resetRequested: Bool = false
}
