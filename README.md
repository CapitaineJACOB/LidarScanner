# LidarScanner

App iOS (SwiftUI + ARKit + Metal) qui affiche en temps réel le vrai nuage de points du scanner LiDAR
(profondeur `ARFrame.sceneDepth`, déprojetée en 3D et rendue point par point).

Ce dossier ne contient PAS de fichier `.xcodeproj` : il est généré automatiquement par
`xcodegen` à partir de `project.yml`, aussi bien en local que dans le workflow GitHub Actions
(`.github/workflows/build.yml`), qui compile un `.ipa` non signé sur un runner macOS gratuit.

Voir les instructions pas-à-pas données par Claude pour :
1. mettre ce dossier sur GitHub sans ligne de commande
2. lancer le build via l'onglet Actions
3. installer le `.ipa` obtenu avec AltServer (Windows) + AltStore
