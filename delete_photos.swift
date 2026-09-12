// Deletes (moves to Recently Deleted) Photos assets by local identifier prefix (uuid),
// using PhotoKit directly instead of the buggy Photos.app AppleScript `delete` command.
// Usage: swift delete_photos.swift UUID1 UUID2 ...
import Photos
import Foundation

let uuids = Array(CommandLine.arguments.dropFirst())
if uuids.isEmpty {
    print("No uuids provided.")
    exit(0)
}

let semaphore = DispatchSemaphore(value: 0)

func run() {
    let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
    if status != .authorized {
        PHPhotoLibrary.requestAuthorization(for: .readWrite) { newStatus in
            if newStatus == .authorized {
                performDelete()
            } else {
                print("Not authorized to access Photos (status: \(newStatus.rawValue))")
                semaphore.signal()
            }
        }
    } else {
        performDelete()
    }
}

func performDelete() {
    var assets: [PHAsset] = []
    var missing: [String] = []
    for uuid in uuids {
        let fetchOptions = PHFetchOptions()
        let matches = PHAsset.fetchAssets(withLocalIdentifiers: [uuid + "/L0/001"], options: fetchOptions)
        if matches.count > 0 {
            assets.append(matches.object(at: 0))
        } else {
            // fall back to scanning all assets for a matching prefix (in case the suffix differs)
            var found: PHAsset? = nil
            let all = PHAsset.fetchAssets(with: nil)
            all.enumerateObjects { asset, _, stop in
                if asset.localIdentifier.hasPrefix(uuid) {
                    found = asset
                    stop.pointee = true
                }
            }
            if let f = found {
                assets.append(f)
            } else {
                missing.append(uuid)
            }
        }
    }

    if !missing.isEmpty {
        print("Could not find \(missing.count) asset(s): \(missing)")
    }

    guard !assets.isEmpty else {
        print("Nothing to delete.")
        print("FREED_BYTES:0")
        semaphore.signal()
        return
    }

    var totalBytes: Int64 = 0
    for asset in assets {
        let resources = PHAssetResource.assetResources(for: asset)
        for resource in resources {
            if let size = resource.value(forKey: "fileSize") as? Int64 {
                totalBytes += size
            }
        }
    }

    PHPhotoLibrary.shared().performChanges({
        PHAssetChangeRequest.deleteAssets(assets as NSArray)
    }) { success, error in
        if success {
            print("Deleted \(assets.count) asset(s).")
            print("FREED_BYTES:\(totalBytes)")
        } else {
            print("Delete failed: \(error?.localizedDescription ?? "unknown error")")
            print("FREED_BYTES:0")
        }
        semaphore.signal()
    }
}

run()
semaphore.wait()
