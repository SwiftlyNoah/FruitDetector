import SwiftUI

struct ContentView: View {
    @State private var image: Image = Image("default")
    @State private var showingImagePicker = false
    @State private var inputImage: UIImage? = nil
    @State private var recognizedFruit: String = "N/A"
    
    var body: some View {
        NavigationStack {
            List {
                Section {
                    image
                        .resizable()
                        .scaledToFill()
                        .frame(height: 300)
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .frame(height: 308)
                }
                
                Section {
                    HStack {
                        Text("Recognized Fruit:")
                        
                        Spacer()
                        
                        Text(recognizedFruit)
                    }
                    
                    Button("Take Photo") {
                        self.showingImagePicker = true
                    }
                }
            }
            .sheet(isPresented: $showingImagePicker, onDismiss: loadImage) {
                ImagePicker(image: self.$inputImage)
            }
            .navigationTitle("Fruit Detector")
        }
    }
    
    func loadImage() {
        guard let inputImage = inputImage else { return }

        // Resize the image to fit the model's input size (150x150)
        if let resizedImage = inputImage.resize(to: CGSize(width: 250, height: 200)),
           let pixelBuffer = resizedImage.toCVPixelBuffer() {
            image = Image(uiImage: inputImage)
            do {
                let model = try FruitDetector()
                let input = FruitDetectorInput(input_1: pixelBuffer)
                let output = try model.prediction(input: input)
                // Process the output
                let result = output.Identity // or output.IdentityShapedArray
                var maxIdx = 0
                for i in 0..<result.count {
                    if result[maxIdx].floatValue < result[i].floatValue {
                        maxIdx = i
                    }
                }
                recognizedFruit = ["Apple", "Banana", "Avacado", "Cherry", "Kiwi", "Mango", "Orange", "Pineapple", "Strawberries", "Watermelon"][maxIdx]
                
            } catch {
                print("Error: \(error)")
            }

        }
    }
}

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?

    func makeUIViewController(context: UIViewControllerRepresentableContext<ImagePicker>) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: UIViewControllerRepresentableContext<ImagePicker>) {

    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        var parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }

            picker.dismiss(animated: true)
        }
    }
}

import UIKit
import CoreML

extension UIImage {
    func resize(to newSize: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        self.draw(in: CGRect(origin: .zero, size: newSize))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage
    }
    
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(self.size.width), Int(self.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard status == kCVReturnSuccess else { return nil }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData, width: Int(self.size.width), height: Int(self.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else { return nil }
        
        context.translateBy(x: 0, y: self.size.height)
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height), blendMode: .copy, alpha: 1.0) // Add this
        UIGraphicsPopContext()
        
        // Normalize pixel values to [0,1]
        if let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer!) {
            let bufferWidth = CVPixelBufferGetWidth(pixelBuffer!)
            let bufferHeight = CVPixelBufferGetHeight(pixelBuffer!)
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer!)
            let byteBuffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            for y in 0..<bufferHeight {
                var pixel = byteBuffer + y * bytesPerRow
                for _ in 0..<bufferWidth {
                    pixel.pointee = UInt8(Float(pixel.pointee) / 255.0) // Normalize pixel values
                    pixel += 1
                }
            }
        }
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        return pixelBuffer
    }
}
