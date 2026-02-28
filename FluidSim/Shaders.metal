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
                              texture3d<float> texture [[texture(0)]],
                              constant float& slice [[buffer(0)]]) {
  constexpr sampler s(address::clamp_to_edge, filter::linear);
  float3 cords = float3(in.texCoords, slice);
  float density = texture.sample(s, cords).r;

  float3 color = mix(float3(0.05, 0.05, 0.05), float3(0.1, 0.6, 1.0), min(density / 100.0, 1.0));
  color += mix(float3(0.0), float3(1.0, 1.0, 1.0), min(max(density - 100.0, 0.0) / 100.0, 1.0));

  return float4(color, 1.0);
}

// 3. Fragment Shader: Velocity (RG mapped to direction/magnitude)
fragment float4 fragment_velocity(VertexOut in [[stage_in]],
                                  texture3d<float> texture [[texture(0)]],
                                  constant float& slice [[buffer(0)]]) {
  constexpr sampler s(address::clamp_to_edge, filter::linear);
  float3 cords = float3(in.texCoords, slice);
  float3 vel = texture.sample(s, cords).xyz;

  float3 color = float3(vel.x * 0.01 + 0.5, vel.y * 0.01 + 0.5, vel.z * 0.01 + 0.5);
  return float4(color, 1.0);
}

// 4. Fragment Shader: Pressure (negative=blue, positive=red)
fragment float4 fragment_pressure(VertexOut in [[stage_in]],
                                  texture3d<float> texture [[texture(0)]],
                                  constant float& slice [[buffer(0)]]) {
  constexpr sampler s(address::clamp_to_edge, filter::linear);
  float3 cords = float3(in.texCoords, slice);
  float pressure = texture.sample(s, cords).r;

  float3 color = float3(max(pressure * 0.5, 0.0), 0.0, max(-pressure * 0.5, 0.0));
  return float4(color, 1.0);
}

// 5. Fragment Shader: Divergence (green/magenta debug)
fragment float4 fragment_divergence(VertexOut in [[stage_in]],
                                    texture3d<float> texture [[texture(0)]],
                                    constant float& slice [[buffer(0)]]) {
  constexpr sampler s(address::clamp_to_edge, filter::linear);
  float3 cords = float3(in.texCoords, slice);
  float div = texture.sample(s, cords).r;

  float3 color = float3(max(div * 5.0, 0.0), max(-div * 5.0, 0.0), max(div * 5.0, 0.0));
  return float4(color, 1.0);
}

// 6. Fragment Shader: Vorticity (blue/yellow heat map)
fragment float4 fragment_vorticity(VertexOut in [[stage_in]],
                                   texture3d<float> texture [[texture(0)]],
                                   constant float& slice [[buffer(0)]]) {
  constexpr sampler s(address::clamp_to_edge, filter::linear);
  float3 cords = float3(in.texCoords, slice);
  float3 curl = texture.sample(s, cords).xyz;
  float magnitude = length(curl);

  // Map magnitude to blue/yellow
  float3 color = mix(float3(0, 0, 1), float3(1, 1, 0), clamp(magnitude * 0.1, 0.0, 1.0));
  return float4(color, 1.0);
}

// ======== FLUID SOLVER KERNELS ========

/// 1. Advection
kernel void advect_kernel(
  texture3d<float, access::sample> velocityTexture [[texture(0)]],
  texture3d<float, access::sample> quantityTexture [[texture(1)]],
  texture3d<float, access::write>  destTexture     [[texture(2)]],
  constant float& dt  [[buffer(0)]],
  constant float& rdx [[buffer(1)]],
  uint3 gid [[thread_position_in_grid]])
{
  if (gid.x >= destTexture.get_width()  ||
      gid.y >= destTexture.get_height() ||
      gid.z >= destTexture.get_depth()) return;

  float3 coords = float3(gid) + 0.5;
  constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);

  float3 vel = velocityTexture.sample(s, coords).xyz;
  float3 pos = coords - dt * rdx * vel;

  float4 advectedQty = quantityTexture.sample(s, pos);
  destTexture.write(advectedQty, gid);
}

// 2. Jacobi Iteration
kernel void jacobi_kernel(
  texture3d<float, access::read>  xTexture    [[texture(0)]],
  texture3d<float, access::read>  bTexture    [[texture(1)]],
  texture3d<float, access::write> destTexture [[texture(2)]],
  constant float& alpha [[buffer(0)]],
  constant float& rBeta [[buffer(1)]],
  uint3 gid [[thread_position_in_grid]])
{
  uint w = destTexture.get_width();
  uint h = destTexture.get_height();
  uint d = destTexture.get_depth();

  if (gid.x >= w || gid.y >= h || gid.z >= d) return;

  uint left   = (gid.x > 0)     ? gid.x - 1 : 0;
  uint right  = (gid.x < w - 1) ? gid.x + 1 : w - 1;
  uint top    = (gid.y > 0)     ? gid.y - 1 : 0;
  uint bottom = (gid.y < h - 1) ? gid.y + 1 : h - 1;
  uint front  = (gid.z > 0)     ? gid.z - 1 : 0;
  uint back   = (gid.z < d - 1) ? gid.z + 1 : d - 1;

  float4 xL = xTexture.read(uint3(left,   gid.y, gid.z));
  float4 xR = xTexture.read(uint3(right,  gid.y, gid.z));
  float4 xT = xTexture.read(uint3(gid.x,  top,   gid.z));
  float4 xB = xTexture.read(uint3(gid.x,  bottom, gid.z));
  float4 xF = xTexture.read(uint3(gid.x,  gid.y, front));
  float4 xBk = xTexture.read(uint3(gid.x, gid.y, back));
  float4 bC = bTexture.read(gid);

  float4 xNew = (xL + xR + xT + xB + xF + xBk + alpha * bC) * rBeta;
  destTexture.write(xNew, gid);
}

// 3. Divergence
kernel void divergence_kernel(
  texture3d<float, access::read>  velocityTexture [[texture(0)]],
  texture3d<float, access::write> divTexture      [[texture(1)]],
  constant float& halfrdx [[buffer(0)]],
  uint3 gid [[thread_position_in_grid]])
{
  uint w = divTexture.get_width();
  uint h = divTexture.get_height();
  uint d = divTexture.get_depth();

  if (gid.x >= w || gid.y >= h || gid.z >= d) return;

  uint left   = (gid.x > 0)     ? gid.x - 1 : 0;
  uint right  = (gid.x < w - 1) ? gid.x + 1 : w - 1;
  uint top    = (gid.y > 0)     ? gid.y - 1 : 0;
  uint bottom = (gid.y < h - 1) ? gid.y + 1 : h - 1;
  uint front  = (gid.z > 0)     ? gid.z - 1 : 0;
  uint back   = (gid.z < d - 1) ? gid.z + 1 : d - 1;

  float3 wL = velocityTexture.read(uint3(left,  gid.y, gid.z)).xyz;
  float3 wR = velocityTexture.read(uint3(right, gid.y, gid.z)).xyz;
  float3 wT = velocityTexture.read(uint3(gid.x, top,   gid.z)).xyz;
  float3 wB = velocityTexture.read(uint3(gid.x, bottom, gid.z)).xyz;
  float3 wF = velocityTexture.read(uint3(gid.x, gid.y, front)).xyz;
  float3 wBk = velocityTexture.read(uint3(gid.x, gid.y, back)).xyz;

  // div = du/dx + dv/dy + dw/dz
  float div = halfrdx * ((wR.x - wL.x) + (wB.y - wT.y) + (wBk.z - wF.z));
  divTexture.write(float4(div, 0, 0, 0), gid);
}

// 4. Gradient (Subtract Pressure)
kernel void gradient_kernel(
  texture3d<float, access::read>  pressureTexture [[texture(0)]],
  texture3d<float, access::read>  velocityTexture [[texture(1)]],
  texture3d<float, access::write> destTexture     [[texture(2)]],
  constant float& halfrdx [[buffer(0)]],
  uint3 gid [[thread_position_in_grid]])
{
  uint w = destTexture.get_width();
  uint h = destTexture.get_height();
  uint d = destTexture.get_depth();

  if (gid.x >= w || gid.y >= h || gid.z >= d) return;

  uint left   = (gid.x > 0)     ? gid.x - 1 : 0;
  uint right  = (gid.x < w - 1) ? gid.x + 1 : w - 1;
  uint top    = (gid.y > 0)     ? gid.y - 1 : 0;
  uint bottom = (gid.y < h - 1) ? gid.y + 1 : h - 1;
  uint front  = (gid.z > 0)     ? gid.z - 1 : 0;
  uint back   = (gid.z < d - 1) ? gid.z + 1 : d - 1;

  float pL  = pressureTexture.read(uint3(left,  gid.y, gid.z)).r;
  float pR  = pressureTexture.read(uint3(right, gid.y, gid.z)).r;
  float pT  = pressureTexture.read(uint3(gid.x, top,   gid.z)).r;
  float pB  = pressureTexture.read(uint3(gid.x, bottom, gid.z)).r;
  float pF  = pressureTexture.read(uint3(gid.x, gid.y, front)).r;
  float pBk = pressureTexture.read(uint3(gid.x, gid.y, back)).r;

  float3 v = velocityTexture.read(gid).xyz;
  v -= halfrdx * float3(pR - pL, pB - pT, pBk - pF);

  destTexture.write(float4(v, 0.0), gid);
}

// 5. Boundary Conditions
kernel void boundary_kernel(
  texture3d<float, access::read>  xTexture    [[texture(0)]],
  texture3d<float, access::write> destTexture [[texture(1)]],
  constant float& scale [[buffer(0)]],
  uint3 gid [[thread_position_in_grid]])
{
  uint w = destTexture.get_width();
  uint h = destTexture.get_height();
  uint d = destTexture.get_depth();

  if (gid.x >= w || gid.y >= h || gid.z >= d) return;

  bool isBoundary = (gid.x == 0 || gid.x == w - 1 ||
                     gid.y == 0 || gid.y == h - 1 ||
                     gid.z == 0 || gid.z == d - 1);

  if (isBoundary) {
    int3 offset = int3(0, 0, 0);
    if (gid.x == 0)     offset.x =  1;
    if (gid.x == w - 1) offset.x = -1;
    if (gid.y == 0)     offset.y =  1;
    if (gid.y == h - 1) offset.y = -1;
    if (gid.z == 0)     offset.z =  1;
    if (gid.z == d - 1) offset.z = -1;

    float4 val = xTexture.read(uint3(int3(gid) + offset));
    destTexture.write(scale * val, gid);
  } else {
    destTexture.write(xTexture.read(gid), gid);
  }
}

// 6. Vorticity: Phase 1 (Compute Curl)
kernel void vorticity_kernel(
  texture3d<float, access::read>  velocityTexture  [[texture(0)]],
  texture3d<float, access::write> vorticityTexture [[texture(1)]],
  constant float& halfrdx [[buffer(0)]],
  uint3 gid [[thread_position_in_grid]])
{
  uint w = vorticityTexture.get_width();
  uint h = vorticityTexture.get_height();
  uint d = vorticityTexture.get_depth();

  if (gid.x >= w || gid.y >= h || gid.z >= d) return;

  uint left   = (gid.x > 0)     ? gid.x - 1 : 0;
  uint right  = (gid.x < w - 1) ? gid.x + 1 : w - 1;
  uint top    = (gid.y > 0)     ? gid.y - 1 : 0;
  uint bottom = (gid.y < h - 1) ? gid.y + 1 : h - 1;
  uint front  = (gid.z > 0)     ? gid.z - 1 : 0;
  uint back   = (gid.z < d - 1) ? gid.z + 1 : d - 1;

  float3 wL  = velocityTexture.read(uint3(left,  gid.y, gid.z)).xyz;
  float3 wR  = velocityTexture.read(uint3(right, gid.y, gid.z)).xyz;
  float3 wT  = velocityTexture.read(uint3(gid.x, top,   gid.z)).xyz;
  float3 wB  = velocityTexture.read(uint3(gid.x, bottom, gid.z)).xyz;
  float3 wF  = velocityTexture.read(uint3(gid.x, gid.y, front)).xyz;
  float3 wBk = velocityTexture.read(uint3(gid.x, gid.y, back)).xyz;

  // curl = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy)
  float3 curl = halfrdx * float3(
    (wB.z  - wT.z)  - (wBk.y - wF.y),
    (wBk.x - wF.x)  - (wR.z  - wL.z),
    (wR.y  - wL.y)  - (wB.x  - wT.x)
  );

  vorticityTexture.write(float4(curl, 0.0), gid);
}

// 7. Vorticity: Phase 2 (Apply Force)
kernel void vorticity_force_kernel(
  texture3d<float, access::read>  vorticityTexture [[texture(0)]],
  texture3d<float, access::read>  velocityTexture  [[texture(1)]],
  texture3d<float, access::write> destTexture      [[texture(2)]],
  constant float& halfrdx [[buffer(0)]],
  constant float& dt      [[buffer(1)]],
  constant float& epsilon [[buffer(2)]],
  uint3 gid [[thread_position_in_grid]])
{
  uint w = destTexture.get_width();
  uint h = destTexture.get_height();
  uint d = destTexture.get_depth();

  if (gid.x >= w || gid.y >= h || gid.z >= d) return;

  uint left   = (gid.x > 0)     ? gid.x - 1 : 0;
  uint right  = (gid.x < w - 1) ? gid.x + 1 : w - 1;
  uint top    = (gid.y > 0)     ? gid.y - 1 : 0;
  uint bottom = (gid.y < h - 1) ? gid.y + 1 : h - 1;
  uint front  = (gid.z > 0)     ? gid.z - 1 : 0;
  uint back   = (gid.z < d - 1) ? gid.z + 1 : d - 1;

  float3 vC  = vorticityTexture.read(gid).xyz;
  float3 vL  = vorticityTexture.read(uint3(left,  gid.y, gid.z)).xyz;
  float3 vR  = vorticityTexture.read(uint3(right, gid.y, gid.z)).xyz;
  float3 vT  = vorticityTexture.read(uint3(gid.x, top,   gid.z)).xyz;
  float3 vB  = vorticityTexture.read(uint3(gid.x, bottom, gid.z)).xyz;
  float3 vF  = vorticityTexture.read(uint3(gid.x, gid.y, front)).xyz;
  float3 vBk = vorticityTexture.read(uint3(gid.x, gid.y, back)).xyz;

  // Gradiente da magnitude do curl
  float3 eta = halfrdx * float3(
    length(vR)  - length(vL),
    length(vB)  - length(vT),
    length(vBk) - length(vF)
  );

  float lenEta = length(eta);
  float3 N = lenEta < 0.0001 ? float3(0) : eta / lenEta;

  // Força = epsilon * (N × curl)
  float3 force = cross(N, vC);

  float3 vel = velocityTexture.read(gid).xyz;
  vel += epsilon * dt * force;

  destTexture.write(float4(vel, 0.0), gid);
}

// 8. Splat
kernel void splat_kernel(
  texture3d<float, access::read>  inputTexture  [[texture(0)]],
  texture3d<float, access::write> outputTexture [[texture(1)]],
  constant float3& point  [[buffer(0)]],  
  constant float4& color  [[buffer(1)]],
  constant float&  radius [[buffer(2)]],
  uint3 gid [[thread_position_in_grid]])
{
  if (gid.x >= outputTexture.get_width()  ||
      gid.y >= outputTexture.get_height() ||
      gid.z >= outputTexture.get_depth()) return;

  float3 p = float3(gid) - point;
  float d  = exp(-dot(p, p) / radius);

  float4 inVal = inputTexture.read(gid);
  outputTexture.write(inVal + color * d, gid);
}

// 9. Dissipate
kernel void dissipate_kernel(
  texture3d<float, access::read>  inputTexture  [[texture(0)]],
  texture3d<float, access::write> outputTexture [[texture(1)]],
  constant float& rate [[buffer(0)]],
  uint3 gid [[thread_position_in_grid]])
{
  if (gid.x >= outputTexture.get_width()  ||
      gid.y >= outputTexture.get_height() ||
      gid.z >= outputTexture.get_depth()) return;

  outputTexture.write(inputTexture.read(gid) * rate, gid);
}
