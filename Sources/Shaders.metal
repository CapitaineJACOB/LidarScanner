#include <metal_stdlib>
#include <metal_atomic>
#include "ShaderTypes.h"
using namespace metal;

struct ParticleVertex {
    float4 position; // xyz = position monde, w = confiance (0,1,2)
    float4 color;    // couleur précalculée (depth ou confiance) ou rgb caméra
};

// MARK: - Compute : déprojection de chaque pixel de profondeur en point 3D,
// ajouté (append) au buffer accumulé partagé plutôt que remplacé chaque frame.

kernel void unprojectDepth(
    texture2d<float, access::read> depthTexture [[texture(0)]],
    texture2d<uint, access::read> confidenceTexture [[texture(1)]],
    texture2d<float, access::sample> capturedImageTextureY [[texture(2)]],
    texture2d<float, access::sample> capturedImageTextureCbCr [[texture(3)]],
    device ParticleVertex *output [[buffer(0)]],
    constant PointCloudUniforms &uniforms [[buffer(1)]],
    device atomic_uint *pointCounter [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(uniforms.gridWidth) || gid.y >= uint(uniforms.gridHeight)) { return; }

    float depth = depthTexture.read(gid).r;
    uint confidence = confidenceTexture.read(gid).r;

    if (depth < uniforms.minDepth || depth > uniforms.maxDepth || depth <= 0.0) { return; }

    // Pixel homogène -> rayon caméra -> point 3D caméra (mètres)
    float3 pixel = float3(float(gid.x), float(gid.y), 1.0);
    float3 cameraRay = uniforms.cameraIntrinsicsInversed * pixel;
    float3 cameraPoint = cameraRay * depth;

    // Caméra -> monde
    float4 worldPoint = uniforms.localToWorld * float4(cameraPoint, 1.0);

    // Couleur depuis l'image caméra (YCbCr -> RGB), pour le mode "Caméra"
    float2 uv = float2(gid) / uniforms.cameraResolution;
    constexpr sampler colorSampler(mag_filter::linear, min_filter::linear);
    float y = capturedImageTextureY.sample(colorSampler, uv).r;
    float2 cbcr = capturedImageTextureCbCr.sample(colorSampler, uv).rg - float2(0.5, 0.5);
    float3 rgb = float3(
        y + 1.402 * cbcr.y,
        y - 0.344 * cbcr.x - 0.714 * cbcr.y,
        y + 1.772 * cbcr.x
    );

    // On réserve une place dans le buffer accumulé (append atomique)
    uint writeIndex = atomic_fetch_add_explicit(pointCounter, 1u, memory_order_relaxed);
    if (writeIndex >= uint(uniforms.maxPoints)) { return; }

    output[writeIndex].position = float4(worldPoint.xyz, float(confidence));
    output[writeIndex].color = float4(clamp(rgb, 0.0, 1.0), 1.0);
}

// MARK: - Rendu du nuage de points accumulé

struct VertexOut {
    float4 clipPosition [[position]];
    float4 color;
    float pointSize [[point_size]];
};

float3 depthToColor(float depth, float minDepth, float maxDepth) {
    float t = clamp((depth - minDepth) / max(maxDepth - minDepth, 0.001), 0.0, 1.0);
    float3 c1 = float3(0.05, 0.05, 0.9);
    float3 c2 = float3(0.1, 0.9, 0.2);
    float3 c3 = float3(0.95, 0.1, 0.1);
    if (t < 0.5) { return mix(c1, c2, t * 2.0); }
    return mix(c2, c3, (t - 0.5) * 2.0);
}

float3 confidenceToColor(float confidence) {
    if (confidence >= 2.0) { return float3(0.15, 1.0, 0.3); }
    if (confidence >= 1.0) { return float3(1.0, 0.8, 0.1); }
    return float3(0.9, 0.15, 0.15);
}

vertex VertexOut pointCloudVertex(
    const device ParticleVertex *vertices [[buffer(0)]],
    constant PointCloudUniforms &uniforms [[buffer(1)]],
    uint vertexID [[vertex_id]])
{
    VertexOut out;
    ParticleVertex v = vertices[vertexID];
    float confidence = v.position.w;

    if (uniforms.highConfidenceOnly != 0 && confidence < 2.0) {
        out.clipPosition = float4(0, 0, -10, 1);
        out.color = float4(0);
        out.pointSize = 0.0;
        return out;
    }

    out.clipPosition = uniforms.viewProjectionMatrix * float4(v.position.xyz, 1.0);
    out.pointSize = clamp(1400.0 / max(out.clipPosition.w, 0.01), 1.2, 6.0);

    if (uniforms.colorMode == 1) {
        out.color = float4(confidenceToColor(confidence), 0.9);
    } else if (uniforms.colorMode == 2) {
        out.color = float4(v.color.rgb, 0.9);
    } else {
        float dist = length(v.position.xyz);
        out.color = float4(depthToColor(dist, 0.2, 5.0), 0.9);
    }
    return out;
}

fragment float4 pointCloudFragment(VertexOut in [[stage_in]],
                                    float2 pointCoord [[point_coord]])
{
    float dist = length(pointCoord - float2(0.5));
    if (dist > 0.5) { discard_fragment(); }
    // léger fondu vers les bords du point pour un rendu "scintillant"
    float edgeFade = 1.0 - smoothstep(0.3, 0.5, dist);
    return float4(in.color.rgb, in.color.a * edgeFade);
}
