///// Copyright (c) 2023 Kodeco Inc.
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
///
/// Notwithstanding the foregoing, you may not use, copy, modify, merge, publish,
/// distribute, sublicense, create a derivative work, and/or sell copies of the
/// Software in any work that is designed, intended, or marketed for pedagogical or
/// instructional purposes related to programming, coding, application development,
/// or information technology.  Permission for such use, copying, modification,
/// merger, publication, distribution, sublicensing, creation of derivative works,
/// or sale is expressly withheld.
///
/// This project and source code may use libraries or frameworks that are
/// released under various Open-Source licenses. Use of those libraries and
/// frameworks are governed by their own individual licenses.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.

import MetalKit

class Renderer: NSObject {
  static var device: MTLDevice!
  static var commandQueue: MTLCommandQueue!
  static var library: MTLLibrary!

  var densityPipelineState: MTLRenderPipelineState!
  var velocityPipelineState: MTLRenderPipelineState!
  var pressurePipelineState: MTLRenderPipelineState!
  var divergencePipelineState: MTLRenderPipelineState!
  var vorticityPipelineState: MTLRenderPipelineState!

  // Ping-pong textures
  var velocityTex: MTLTexture!
  var velocityPrevTex: MTLTexture!
  var densityTex: MTLTexture!
  var densityPrevTex: MTLTexture!
  var pressureTex: MTLTexture!
  var divergenceTex: MTLTexture!
  var vorticityTex: MTLTexture!

  // Compute Pipelines
  var advectState: MTLComputePipelineState!
  var jacobiState: MTLComputePipelineState!
  var divergenceState: MTLComputePipelineState!
  var gradientState: MTLComputePipelineState!
  var boundaryState: MTLComputePipelineState!
  var splatState: MTLComputePipelineState!
  var dissipateState: MTLComputePipelineState!
  var vorticityState: MTLComputePipelineState!
  var vorticityForceState: MTLComputePipelineState!

  let gridSize = 128

  var interactPoints: [CGPoint] = []
  var displayMode: DisplayMode = .density

  init(metalView: MTKView) {
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else {
      fatalError("GPU not available")
    }
    Self.device = device
    Self.commandQueue = commandQueue
    metalView.device = device

    let library = device.makeDefaultLibrary()!
    Self.library = library

    super.init()

    func makeTexture(format: MTLPixelFormat) -> MTLTexture {
      let desc = MTLTextureDescriptor()
      desc.textureType = .type3D
      desc.pixelFormat = format
      desc.width = gridSize
      desc.height = gridSize
      desc.depth = gridSize
      desc.mipmapLevelCount = 1
      desc.usage = [.shaderRead, .shaderWrite]
      desc.storageMode = .private
      return device.makeTexture(descriptor: desc)!
    }

    velocityTex     = makeTexture(format: .rgba32Float)
    velocityPrevTex = makeTexture(format: .rgba32Float)
    densityTex      = makeTexture(format: .r32Float)
    densityPrevTex  = makeTexture(format: .r32Float)
    pressureTex     = makeTexture(format: .r32Float)
    divergenceTex   = makeTexture(format: .r32Float)
    vorticityTex    = makeTexture(format: .rgba32Float)
  
    func makeComputePipeline(name: String) -> MTLComputePipelineState {
      let function = library.makeFunction(name: name)!
      return try! device.makeComputePipelineState(function: function)
    }

    advectState    = makeComputePipeline(name: "advect_kernel")
    jacobiState    = makeComputePipeline(name: "jacobi_kernel")
    divergenceState = makeComputePipeline(name: "divergence_kernel")
    gradientState  = makeComputePipeline(name: "gradient_kernel")
    boundaryState  = makeComputePipeline(name: "boundary_kernel")
    splatState     = makeComputePipeline(name: "splat_kernel")
    dissipateState = makeComputePipeline(name: "dissipate_kernel")
    vorticityState = makeComputePipeline(name: "vorticity_kernel")
    vorticityForceState = makeComputePipeline(name: "vorticity_force_kernel")

    func makeRenderPipeline(fragmentName: String) -> MTLRenderPipelineState {
      let pipelineDescriptor = MTLRenderPipelineDescriptor()
      pipelineDescriptor.vertexFunction = library.makeFunction(name: "vertex_main")
      pipelineDescriptor.fragmentFunction = library.makeFunction(name: fragmentName)
      pipelineDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
      return try! device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    densityPipelineState    = makeRenderPipeline(fragmentName: "fragment_main")
    velocityPipelineState   = makeRenderPipeline(fragmentName: "fragment_velocity")
    pressurePipelineState   = makeRenderPipeline(fragmentName: "fragment_pressure")
    divergencePipelineState = makeRenderPipeline(fragmentName: "fragment_divergence")
    vorticityPipelineState  = makeRenderPipeline(fragmentName: "fragment_vorticity")

    metalView.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.05, alpha: 1)
    metalView.delegate = self
  }
}

extension Renderer: MTKViewDelegate {
  func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

  func draw(in view: MTKView) {
    guard let drawable = view.currentDrawable,
          let descriptor = view.currentRenderPassDescriptor,
          let commandBuffer = Self.commandQueue.makeCommandBuffer(),
          let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
      return
    }

    func dispatch(_ state: MTLComputePipelineState, textures: [MTLTexture]) {
      computeEncoder.setComputePipelineState(state)
      for (i, tex) in textures.enumerated() {
        computeEncoder.setTexture(tex, index: i)
      }
      let tgSize = MTLSize(width: 8, height: 8, depth: 8)
      let gridSize = MTLSize(width: self.gridSize, height: self.gridSize, depth: self.gridSize)
      computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    }

    var dt: Float       = 0.1
    var rdx: Float      = 1.0
    var halfrdx: Float  = 0.5
    var scaleNeg: Float = -1.0
    var scalePos: Float = 1.0
    var decay: Float    = 0.999
    var radius: Float   = 20
    var epsilon: Float  = 0.1 // Vorticity scale

    // 1. Process Interactions
    if !interactPoints.isEmpty {
      for pt in interactPoints {
        var point = SIMD3<Float>(Float(pt.x), Float(pt.y), Float(gridSize) / 2.0)
        var colorDens = SIMD4<Float>(400.0, 0, 0, 0)

        // Splat Density
        computeEncoder.setBytes(&point, length: MemoryLayout<SIMD3<Float>>.size, index: 0)
        computeEncoder.setBytes(&colorDens, length: MemoryLayout<SIMD4<Float>>.size, index: 1)
        computeEncoder.setBytes(&radius, length: MemoryLayout<Float>.size, index: 2)
        dispatch(splatState, textures: [densityTex, densityPrevTex])
        Swift.swap(&densityTex, &densityPrevTex)

        // Splat Velocity
        var colorVel = SIMD4<Float>(Float.random(in: -40...40), -100.0, 0.0, 0.0)
        computeEncoder.setBytes(&point, length: MemoryLayout<SIMD3<Float>>.size, index: 0)
        computeEncoder.setBytes(&colorVel, length: MemoryLayout<SIMD4<Float>>.size, index: 1)
        computeEncoder.setBytes(&radius, length: MemoryLayout<Float>.size, index: 2)
        dispatch(splatState, textures: [velocityTex, velocityPrevTex])
        Swift.swap(&velocityTex, &velocityPrevTex)
      }
      interactPoints.removeAll()
    }

    // 2. Advect Velocity
    computeEncoder.setBytes(&dt, length: MemoryLayout<Float>.size, index: 0)
    computeEncoder.setBytes(&rdx, length: MemoryLayout<Float>.size, index: 1)
    dispatch(advectState, textures: [velocityTex, velocityTex, velocityPrevTex])
    Swift.swap(&velocityTex, &velocityPrevTex)

    // Boundary Velocity
    computeEncoder.setBytes(&scaleNeg, length: MemoryLayout<Float>.size, index: 0)
    dispatch(boundaryState, textures: [velocityTex, velocityPrevTex])
    Swift.swap(&velocityTex, &velocityPrevTex)

    // 3. Divergence
    computeEncoder.setBytes(&halfrdx, length: MemoryLayout<Float>.size, index: 0)
    dispatch(divergenceState, textures: [velocityTex, divergenceTex])

    // 3.1 Vorticity Confinement
    computeEncoder.setBytes(&halfrdx, length: MemoryLayout<Float>.size, index: 0)
    dispatch(vorticityState, textures: [velocityTex, vorticityTex])

    computeEncoder.setBytes(&halfrdx, length: MemoryLayout<Float>.size, index: 0)
    computeEncoder.setBytes(&dt, length: MemoryLayout<Float>.size, index: 1)
    computeEncoder.setBytes(&epsilon, length: MemoryLayout<Float>.size, index: 2)
    dispatch(vorticityForceState, textures: [vorticityTex, velocityTex, velocityPrevTex])
    Swift.swap(&velocityTex, &velocityPrevTex)

    // 4. Pressure Projection (Jacobi)
    var alpha: Float = -1.0
    var rBeta: Float = 1.0 / 6.0
    for _ in 0..<30 {
      computeEncoder.setBytes(&alpha, length: MemoryLayout<Float>.size, index: 0)
      computeEncoder.setBytes(&rBeta, length: MemoryLayout<Float>.size, index: 1)
      dispatch(jacobiState, textures: [pressureTex, divergenceTex, densityPrevTex])
      Swift.swap(&pressureTex, &densityPrevTex)

      // Boundary Pressure
      computeEncoder.setBytes(&scalePos, length: MemoryLayout<Float>.size, index: 0)
      dispatch(boundaryState, textures: [pressureTex, densityPrevTex])
      Swift.swap(&pressureTex, &densityPrevTex)
    }

    // 5. Subtract Gradient
    computeEncoder.setBytes(&halfrdx, length: MemoryLayout<Float>.size, index: 0)
    dispatch(gradientState, textures: [pressureTex, velocityTex, velocityPrevTex])
    Swift.swap(&velocityTex, &velocityPrevTex)

    // Boundary Velocity
    computeEncoder.setBytes(&scaleNeg, length: MemoryLayout<Float>.size, index: 0)
    dispatch(boundaryState, textures: [velocityTex, velocityPrevTex])
    Swift.swap(&velocityTex, &velocityPrevTex)

    // 6. Advect Density
    computeEncoder.setBytes(&dt, length: MemoryLayout<Float>.size, index: 0)
    computeEncoder.setBytes(&rdx, length: MemoryLayout<Float>.size, index: 1)
    dispatch(advectState, textures: [velocityTex, densityTex, densityPrevTex])
    Swift.swap(&densityTex, &densityPrevTex)

    // 7. Dissipate Density and Velocity
    computeEncoder.setBytes(&decay, length: MemoryLayout<Float>.size, index: 0)
    dispatch(dissipateState, textures: [velocityTex, velocityPrevTex])
    Swift.swap(&velocityTex, &velocityPrevTex)

    computeEncoder.setBytes(&decay, length: MemoryLayout<Float>.size, index: 0)
    dispatch(dissipateState, textures: [densityTex, densityPrevTex])
    Swift.swap(&densityTex, &densityPrevTex)

    computeEncoder.endEncoding()

    // 8. Display
    if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) {
      switch displayMode {
      case .density:
        renderEncoder.setRenderPipelineState(densityPipelineState)
        renderEncoder.setFragmentTexture(densityTex, index: 0)
      case .velocity:
        renderEncoder.setRenderPipelineState(velocityPipelineState)
        renderEncoder.setFragmentTexture(velocityTex, index: 0)
      case .pressure:
        renderEncoder.setRenderPipelineState(pressurePipelineState)
        renderEncoder.setFragmentTexture(pressureTex, index: 0)
      case .divergence:
        renderEncoder.setRenderPipelineState(divergencePipelineState)
        renderEncoder.setFragmentTexture(divergenceTex, index: 0)
      case .vorticity:
        renderEncoder.setRenderPipelineState(vorticityPipelineState)
        renderEncoder.setFragmentTexture(vorticityTex, index: 0)
      }
      var slice: Float = 0.5
      renderEncoder.setFragmentBytes(&slice, length: MemoryLayout<Float>.size, index: 0)
      renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
      renderEncoder.endEncoding()
    }

    commandBuffer.present(drawable)
    commandBuffer.commit()
  }

  func handleInteraction(at location: CGPoint, in size: CGSize) {
    let normalizedX = Float(location.x / size.width)
    let normalizedY = Float(location.y / size.height)

    let cx = normalizedX * Float(gridSize)
    let cy = normalizedY * Float(gridSize)

    interactPoints.append(CGPoint(x: CGFloat(cx), y: CGFloat(cy)))
  }
}
