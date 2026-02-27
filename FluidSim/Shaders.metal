#include <metal_stdlib>
using namespace metal;

struct VertexOut {
  float4 position [[position]];
  float2 texCoords;
};

// 1. Vertex Shader: Fullscreen quad
vertex VertexOut vertex_main(uint vertexID [[vertex_id]]) {
  float2 positions[6] = {
    float2(-1, -1), float2(1, -1), float2(-1, 1),
    float2(1, -1),  float2(1, 1),  float2(-1, 1)
  };
  float2 texCoords[6] = {
    float2(0, 1), float2(1, 1), float2(0, 0),
    float2(1, 1), float2(1, 0), float2(0, 0)
  };

  VertexOut out;
  out.position  = float4(positions[vertexID], 0.0, 1.0);
  out.texCoords = texCoords[vertexID];
  return out;
}

// 2. Fragment Shader: Density (fire/smoke style)
fragment float4 fragment_main(VertexOut in [[stage_in]],
                              texture2d<float> texture [[texture(0)]]) {
  constexpr sampler s(address::clamp_to_edge, filter::linear);
  float density = texture.sample(s, in.texCoords).r;

  float3 color = mix(float3(0.05, 0.05, 0.05), float3(0.1, 0.6, 1.0), min(density / 100.0, 1.0));
  color += mix(float3(0.0), float3(1.0, 1.0, 1.0), min(max(density - 100.0, 0.0) / 100.0, 1.0));

  return float4(color, 1.0);
}

// 3. Fragment Shader: Velocity (RG mapped to direction/magnitude)
fragment float4 fragment_velocity(VertexOut in [[stage_in]],
                                  texture2d<float> texture [[texture(0)]]) {
  constexpr sampler s(address::clamp_to_edge, filter::linear);
  float2 vel = texture.sample(s, in.texCoords).xy;

  float3 color = float3(vel.x * 0.01 + 0.5, vel.y * 0.01 + 0.5, 0.5);
  return float4(color, 1.0);
}

// 4. Fragment Shader: Pressure (negative=blue, positive=red)
fragment float4 fragment_pressure(VertexOut in [[stage_in]],
                                  texture2d<float> texture [[texture(0)]]) {
  constexpr sampler s(address::clamp_to_edge, filter::linear);
  float pressure = texture.sample(s, in.texCoords).r;

  float3 color = float3(max(pressure * 0.5, 0.0), 0.0, max(-pressure * 0.5, 0.0));
  return float4(color, 1.0);
}

// 5. Fragment Shader: Divergence (green/magenta debug)
fragment float4 fragment_divergence(VertexOut in [[stage_in]],
                                    texture2d<float> texture [[texture(0)]]) {
  constexpr sampler s(address::clamp_to_edge, filter::linear);
  float div = texture.sample(s, in.texCoords).r;

  float3 color = float3(max(div * 5.0, 0.0), max(-div * 5.0, 0.0), max(div * 5.0, 0.0));
  return float4(color, 1.0);
}

// ======== FLUID SOLVER KERNELS ========

// 1. Advection
kernel void advect_kernel(
  texture2d<float, access::sample> velocityTexture [[texture(0)]],
  texture2d<float, access::sample> quantityTexture [[texture(1)]],
  texture2d<float, access::write>  destTexture     [[texture(2)]],
  constant float& dt  [[buffer(0)]],
  constant float& rdx [[buffer(1)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= destTexture.get_width() || gid.y >= destTexture.get_height()) return;

  float2 coords = float2(gid) + 0.5;
  constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);

  float2 vel = velocityTexture.sample(s, coords).xy;
  float2 pos = coords - dt * rdx * vel;

  float4 advectedQty = quantityTexture.sample(s, pos);
  destTexture.write(advectedQty, gid);
}

// 2. Jacobi Iteration (Diffusion and Pressure Projection)
kernel void jacobi_kernel(
  texture2d<float, access::read>  xTexture   [[texture(0)]],
  texture2d<float, access::read>  bTexture   [[texture(1)]],
  texture2d<float, access::write> destTexture [[texture(2)]],
  constant float& alpha [[buffer(0)]],
  constant float& rBeta [[buffer(1)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= destTexture.get_width() || gid.y >= destTexture.get_height()) return;

  uint w = destTexture.get_width();
  uint h = destTexture.get_height();

  uint left   = (gid.x > 0)     ? gid.x - 1 : 0;
  uint right  = (gid.x < w - 1) ? gid.x + 1 : w - 1;
  uint top    = (gid.y > 0)     ? gid.y - 1 : 0;
  uint bottom = (gid.y < h - 1) ? gid.y + 1 : h - 1;

  float4 xL = xTexture.read(uint2(left,  gid.y));
  float4 xR = xTexture.read(uint2(right, gid.y));
  float4 xT = xTexture.read(uint2(gid.x, top));
  float4 xB = xTexture.read(uint2(gid.x, bottom));
  float4 bC = bTexture.read(gid);

  float4 xNew = (xL + xR + xT + xB + alpha * bC) * rBeta;
  destTexture.write(xNew, gid);
}

// 3. Divergence
kernel void divergence_kernel(
  texture2d<float, access::read>  velocityTexture [[texture(0)]],
  texture2d<float, access::write> divTexture      [[texture(1)]],
  constant float& halfrdx [[buffer(0)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= divTexture.get_width() || gid.y >= divTexture.get_height()) return;

  uint w = divTexture.get_width();
  uint h = divTexture.get_height();

  uint left   = (gid.x > 0)     ? gid.x - 1 : 0;
  uint right  = (gid.x < w - 1) ? gid.x + 1 : w - 1;
  uint top    = (gid.y > 0)     ? gid.y - 1 : 0;
  uint bottom = (gid.y < h - 1) ? gid.y + 1 : h - 1;

  float2 wL = velocityTexture.read(uint2(left,  gid.y)).xy;
  float2 wR = velocityTexture.read(uint2(right, gid.y)).xy;
  float2 wT = velocityTexture.read(uint2(gid.x, top)).xy;
  float2 wB = velocityTexture.read(uint2(gid.x, bottom)).xy;

  float div = halfrdx * ((wR.x - wL.x) + (wB.y - wT.y));
  divTexture.write(float4(div, 0.0, 0.0, 0.0), gid);
}

// 4. Vorticity Confinement: Phase 1 (Compute Curl)
kernel void vorticity_kernel(
  texture2d<float, access::read>  velocityTexture  [[texture(0)]],
  texture2d<float, access::write> vorticityTexture [[texture(1)]],
  constant float& halfrdx [[buffer(0)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= vorticityTexture.get_width() || gid.y >= vorticityTexture.get_height()) return;

  uint w = vorticityTexture.get_width();
  uint h = vorticityTexture.get_height();

  uint left   = (gid.x > 0)     ? gid.x - 1 : 0;
  uint right  = (gid.x < w - 1) ? gid.x + 1 : w - 1;
  uint top    = (gid.y > 0)     ? gid.y - 1 : 0;
  uint bottom = (gid.y < h - 1) ? gid.y + 1 : h - 1;

  float2 wL = velocityTexture.read(uint2(left,  gid.y)).xy;
  float2 wR = velocityTexture.read(uint2(right, gid.y)).xy;
  float2 wT = velocityTexture.read(uint2(gid.x, top)).xy;
  float2 wB = velocityTexture.read(uint2(gid.x, bottom)).xy;

  // curl = (dv/dx) - (du/dy)
  float curl = halfrdx * ((wR.y - wL.y) - (wB.x - wT.x));
  vorticityTexture.write(float4(curl, 0.0, 0.0, 0.0), gid);
}

// 5. Vorticity Confinement: Phase 2 (Apply Force)
kernel void vorticity_force_kernel(
  texture2d<float, access::read>  vorticityTexture [[texture(0)]],
  texture2d<float, access::read>  velocityTexture  [[texture(1)]],
  texture2d<float, access::write> destTexture      [[texture(2)]],
  constant float& halfrdx [[buffer(0)]],
  constant float& dt      [[buffer(1)]],
  constant float& epsilon [[buffer(2)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= destTexture.get_width() || gid.y >= destTexture.get_height()) return;

  uint w = destTexture.get_width();
  uint h = destTexture.get_height();

  uint left   = (gid.x > 0)     ? gid.x - 1 : 0;
  uint right  = (gid.x < w - 1) ? gid.x + 1 : w - 1;
  uint top    = (gid.y > 0)     ? gid.y - 1 : 0;
  uint bottom = (gid.y < h - 1) ? gid.y + 1 : h - 1;

  float vC = vorticityTexture.read(gid).r;
  float vL = vorticityTexture.read(uint2(left,  gid.y)).r;
  float vR = vorticityTexture.read(uint2(right, gid.y)).r;
  float vT = vorticityTexture.read(uint2(gid.x, top)).r;
  float vB = vorticityTexture.read(uint2(gid.x, bottom)).r;

  // Vorticity gradient: eta = grad(|omega|)
  float2 eta = halfrdx * float2(abs(vR) - abs(vL), abs(vB) - abs(vT));

  float lengthEta = length(eta);
  float2 N = lengthEta < 0.0001 ? float2(0) : eta / lengthEta;

  // Confinement force: epsilon * (N x omega)
  float2 force = float2(N.y * vC, -N.x * vC);

  float2 vel = velocityTexture.read(gid).xy;
  vel += epsilon * dt * force;

  destTexture.write(float4(vel.x, vel.y, 0.0, 0.0), gid);
}

// 6. Subtract Gradient
kernel void gradient_kernel(
  texture2d<float, access::read>  pressureTexture  [[texture(0)]],
  texture2d<float, access::read>  velocityTexture  [[texture(1)]],
  texture2d<float, access::write> destTexture      [[texture(2)]],
  constant float& halfrdx [[buffer(0)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= destTexture.get_width() || gid.y >= destTexture.get_height()) return;

  uint w = destTexture.get_width();
  uint h = destTexture.get_height();

  uint left   = (gid.x > 0)     ? gid.x - 1 : 0;
  uint right  = (gid.x < w - 1) ? gid.x + 1 : w - 1;
  uint top    = (gid.y > 0)     ? gid.y - 1 : 0;
  uint bottom = (gid.y < h - 1) ? gid.y + 1 : h - 1;

  float pL = pressureTexture.read(uint2(left,  gid.y)).r;
  float pR = pressureTexture.read(uint2(right, gid.y)).r;
  float pT = pressureTexture.read(uint2(gid.x, top)).r;
  float pB = pressureTexture.read(uint2(gid.x, bottom)).r;

  float2 v = velocityTexture.read(gid).xy;
  v -= halfrdx * float2(pR - pL, pB - pT);

  destTexture.write(float4(v.x, v.y, 0.0, 0.0), gid);
}

// 7. Boundary Conditions
kernel void boundary_kernel(
  texture2d<float, access::read>  xTexture    [[texture(0)]],
  texture2d<float, access::write> destTexture [[texture(1)]],
  constant float& scale [[buffer(0)]],
  uint2 gid [[thread_position_in_grid]])
{
  uint w = destTexture.get_width();
  uint h = destTexture.get_height();

  if (gid.x >= w || gid.y >= h) return;

  bool isBoundary = (gid.x == 0 || gid.x == w - 1 || gid.y == 0 || gid.y == h - 1);

  if (isBoundary) {
    int2 offset = int2(0, 0);
    if (gid.x == 0)     offset.x =  1;
    if (gid.x == w - 1) offset.x = -1;
    if (gid.y == 0)     offset.y =  1;
    if (gid.y == h - 1) offset.y = -1;

    float4 val = xTexture.read(uint2(int2(gid) + offset));
    destTexture.write(scale * val, gid);
  } else {
    destTexture.write(xTexture.read(gid), gid);
  }
}

// 8. Splat (Injection)
kernel void splat_kernel(
  texture2d<float, access::read>  inputTexture  [[texture(0)]],
  texture2d<float, access::write> outputTexture [[texture(1)]],
  constant float2& point  [[buffer(0)]],
  constant float4& color  [[buffer(1)]],
  constant float&  radius [[buffer(2)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;

  float2 p = float2(gid) - point;
  float d  = exp(-dot(p, p) / radius);

  float4 inVal = inputTexture.read(gid);
  outputTexture.write(inVal + color * d, gid);
}

// 9. Dissipate (Decay)
kernel void dissipate_kernel(
  texture2d<float, access::read>  inputTexture  [[texture(0)]],
  texture2d<float, access::write> outputTexture [[texture(1)]],
  constant float& rate [[buffer(0)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;

  float4 val = inputTexture.read(gid);
  outputTexture.write(val * rate, gid);
}
