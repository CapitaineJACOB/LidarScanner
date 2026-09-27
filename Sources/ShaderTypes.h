#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

struct PointCloudUniforms {
    matrix_float4x4 viewProjectionMatrix;
    matrix_float4x4 localToWorld;
    matrix_float3x3 cameraIntrinsicsInversed;
    simd_float2 cameraResolution;

    int colorMode;      // 0 = depth, 1 = confidence, 2 = rgb
    int highConfidenceOnly;
    float minDepth;
    float maxDepth;
    int gridWidth;
    int gridHeight;
    int maxPoints;
};

#endif
