import SwiftUI

@main
struct LidarScannerApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .ignoresSafeArea()
                .statusBar(hidden: true)
                .persistentSystemOverlays(.hidden)
        }
    }
}
