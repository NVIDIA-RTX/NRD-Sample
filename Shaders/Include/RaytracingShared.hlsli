
NRI_RESOURCE( RaytracingAccelerationStructure, gWorldTlas, t, 0, SET_RAY_TRACING );
NRI_RESOURCE( RaytracingAccelerationStructure, gLightTlas, t, 1, SET_RAY_TRACING );
NRI_RESOURCE( StructuredBuffer<InstanceData>, gIn_InstanceData, t, 2, SET_RAY_TRACING );
NRI_RESOURCE( StructuredBuffer<PrimitiveData>, gIn_PrimitiveData, t, 3, SET_RAY_TRACING );
NRI_RESOURCE( StructuredBuffer<MorphedPrimitivePrevPositions>, gIn_MorphedPrimitivePrevPositions, t, 4, SET_RAY_TRACING );
NRI_RESOURCE( Texture2D<float4>, gIn_Textures[], t, 5, SET_RAY_TRACING );

NRI_RESOURCE( RWStructuredBuffer<uint64_t>, gInOut_SharcHashEntriesBuffer, u, 0, SET_SHARC );
NRI_RESOURCE( RWStructuredBuffer<uint>, gInOut_SharcHashCopyOffsetBuffer, u, 1, SET_SHARC );
NRI_RESOURCE( RWStructuredBuffer<uint4>, gInOut_SharcVoxelDataBuffer, u, 2, SET_SHARC );
NRI_RESOURCE( RWStructuredBuffer<uint4>, gInOut_SharcVoxelDataBufferPrev, u, 3, SET_SHARC );

#if( USE_STOCHASTIC_SAMPLING == 1 )
    #define TEX_SAMPLER gNearestMipmapNearestSampler
#else
    #define TEX_SAMPLER gLinearMipmapLinearSampler
#endif

#if( USE_LOAD == 1 )
    #define SAMPLE( coords ) Load( int3( coords ) )
#else
    #define SAMPLE( coords ) SampleLevel( TEX_SAMPLER, coords.xy, coords.z )
#endif

#include "HairBRDF.hlsli"

struct GeometryProps
{
    float3 X;
    float3 Xprev;
    float3 V;
    float4 T;
    float3 N;
    float2 uv;
    float mip;
    float hitT;
    float curvature;
    uint textureOffsetAndFlags;
    uint instanceIndex;

    float3 GetXoffset( float3 offsetDir, float amount = PT_BOUNCE_RAY_OFFSET )
    {
        float viewZ = Geometry::AffineTransform( gWorldToView, X ).z;
        amount *= gUnproject * lerp( abs( viewZ ), 1.0, abs( gOrthoMode ) );

        return X + offsetDir * max( amount, 0.00001 );
    }

    bool Has( uint flag )
    { return ( textureOffsetAndFlags & ( flag << FLAG_FIRST_BIT ) ) != 0; }

    uint GetBaseTexture( )
    { return textureOffsetAndFlags & NON_FLAG_MASK; }

    float3 GetForcedEmissionColor( )
    { return ( ( textureOffsetAndFlags >> 2 ) & 0x1 ) ? float3( 1.0, 0.0, 0.0 ) : float3( 0.0, 1.0, 0.0 ); }

    bool IsSky( )
    { return hitT == INF; }
};

float2 GetConeAngleFromAngularRadius( float mip, float tanConeAngle )
{
    // In any case, we are limited by the output resolution
    tanConeAngle = max( tanConeAngle, gTanPixelAngularRadius );

    return float2( mip, tanConeAngle );
}

float2 GetConeAngleFromRoughness( float mip, float roughness )
{
    float tanConeAngle = roughness * roughness * 0.05; // TODO: tweaked to be accurate and give perf boost

    return GetConeAngleFromAngularRadius( mip, tanConeAngle );
}

float2 STF_Bilinear( float2 uv, float2 texSize )
{
    Filtering::Bilinear f = Filtering::GetBilinearFilter( uv, texSize );

    float2 rnd = Rng::Hash::GetFloat2( );
    f.origin += step( rnd, f.weights );

    return f.origin / texSize;
}

float3 GetSamplingCoords( uint textureIndex, float2 uv, float mip, int mode )
{
    float2 texSize;
    gIn_Textures[ NonUniformResourceIndex( textureIndex ) ].GetDimensions( texSize.x, texSize.y ); // TODO: if I only had it as a constant...

    // Recalculate for the current texture
    float mipNum = log2( max( texSize.x, texSize.y ) );
    mip += mipNum - MAX_MIP_LEVEL;
    if( mode == MIP_VISIBILITY )
    {
        // We must avoid using lower mips because it can lead to significant increase in AHS invocations. Mips lower than 128x128 are skipped!
        mip = min( mip, mipNum - 7.0 );
    }
    else
        mip += gMipBias * ( mode == MIP_LESS_SHARP ? 0.5 : 1.0 );
    mip = clamp( mip, 0.0, mipNum - 1.0 );

    #if( USE_STOCHASTIC_SAMPLING == 1 )
        mip = floor( mip ) + step( Rng::Hash::GetFloat( ), frac( mip ) );
    #elif( USE_LOAD == 1 )
        mip = round( mip );
    #endif

    texSize *= exp2( -mip );

    // Uv coordinates
    #if( USE_STOCHASTIC_SAMPLING == 1 )
        uv = STF_Bilinear( uv, texSize );
    #endif

    #if( USE_LOAD == 1 )
        uv = frac( uv ) * texSize;
    #endif

    return float3( uv, mip );
}

//====================================================================================================================================
// TRACER
//====================================================================================================================================

#define CheckNonOpaqueTriangle( rayQuery, mipAndCone ) \
    { \
        /* Instance */ \
        uint instanceIndex = rayQuery.CandidateInstanceID( ) + rayQuery.CandidateGeometryIndex( ); \
        InstanceData instanceData = gIn_InstanceData[ instanceIndex ]; \
        \
        /* Transform */ \
        float3x3 mObjectToWorld = ( float3x3 )rayQuery.CandidateObjectToWorld3x4( ); \
        float3x4 mOverloaded = float3x4( instanceData.mOverloadedMatrix0, instanceData.mOverloadedMatrix1, instanceData.mOverloadedMatrix2 ); \
        if( instanceData.textureOffsetAndFlags & ( FLAG_STATIC << FLAG_FIRST_BIT ) ) \
            mObjectToWorld = ( float3x3 )mOverloaded; \
        \
        float flip = Math::Sign( instanceData.scale ) * ( rayQuery.CandidateTriangleFrontFace( ) ? -1.0 : 1.0 ); \
        \
        /* Primitive */ \
        uint primitiveIndex = instanceData.primitiveOffset + rayQuery.CandidatePrimitiveIndex( ); \
        PrimitiveData primitiveData = gIn_PrimitiveData[ primitiveIndex ]; \
        \
        float worldArea = primitiveData.worldArea * instanceData.scale * instanceData.scale; \
        \
        /* Barycentrics */ \
        float3 barycentrics; \
        barycentrics.yz = rayQuery.CandidateTriangleBarycentrics( ); \
        barycentrics.x = 1.0 - barycentrics.y - barycentrics.z; \
        \
        /* Uv */ \
        float2 uv = barycentrics.x * primitiveData.uv0 + barycentrics.y * primitiveData.uv1 + barycentrics.z * primitiveData.uv2; \
        \
        /* Normal */ \
        float3 n0 = Packing::DecodeUnitVector( primitiveData.n0, true ); \
        float3 n1 = Packing::DecodeUnitVector( primitiveData.n1, true ); \
        float3 n2 = Packing::DecodeUnitVector( primitiveData.n2, true ); \
        \
        float3 N = barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2; \
        N = Geometry::RotateVector( mObjectToWorld, N ); \
        N = normalize( N * flip ); \
        \
        /* Mip level */ \
        float NoRay = abs( dot( rayQuery.WorldRayDirection( ), N ) ); \
        float a = rayQuery.CandidateTriangleRayT( ); \
        a *= mipAndCone.y; \
        a *= Math::PositiveRcp( NoRay ); \
        a *= sqrt( primitiveData.uvArea / worldArea ); \
        \
        float mip = log2( a ); \
        mip += MAX_MIP_LEVEL; \
        mip = max( mip, 0.0 ); \
        mip += mipAndCone.x; \
        \
        /* Alpha test */ \
        uint baseTexture = ( instanceData.textureOffsetAndFlags & NON_FLAG_MASK ) + 0; \
        float3 coords = GetSamplingCoords( baseTexture, uv, mip, MIP_VISIBILITY ); \
        float alpha = gIn_Textures[ baseTexture ].SAMPLE( coords ).w; \
        \
        if( alpha > 0.5 ) \
            rayQuery.CommitNonOpaqueTriangleHit( ); \
    }

bool CastVisibilityRay_AnyHit( float3 origin, float3 direction, float Tmin, float Tmax, float2 mipAndCone, RaytracingAccelerationStructure accelerationStructure, uint instanceInclusionMask, uint rayFlags )
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = Tmin;
    rayDesc.TMax = Tmax;

    RayQuery< RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH > rayQuery;
    rayQuery.TraceRayInline( accelerationStructure, rayFlags, instanceInclusionMask, rayDesc );

    while( rayQuery.Proceed( ) )
        CheckNonOpaqueTriangle( rayQuery, mipAndCone );

    return rayQuery.CommittedStatus( ) == COMMITTED_NOTHING;
}

float CastVisibilityRay_ClosestHit( float3 origin, float3 direction, float Tmin, float Tmax, float2 mipAndCone, RaytracingAccelerationStructure accelerationStructure, uint instanceInclusionMask, uint rayFlags )
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = Tmin;
    rayDesc.TMax = Tmax;

    RayQuery< RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES > rayQuery;
    rayQuery.TraceRayInline( accelerationStructure, rayFlags, instanceInclusionMask, rayDesc );

    while( rayQuery.Proceed( ) )
        CheckNonOpaqueTriangle( rayQuery, mipAndCone );

    return rayQuery.CommittedStatus( ) == COMMITTED_NOTHING ? INF : rayQuery.CommittedRayT( );
}

GeometryProps CastRay( float3 origin, float3 direction, float Tmin, float Tmax, float2 mipAndCone, RaytracingAccelerationStructure accelerationStructure, uint instanceInclusionMask, uint rayFlags )
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = Tmin;
    rayDesc.TMax = Tmax;

    RayQuery< RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES > rayQuery;
    rayQuery.TraceRayInline( accelerationStructure, rayFlags, instanceInclusionMask, rayDesc );

    while( rayQuery.Proceed( ) )
        CheckNonOpaqueTriangle( rayQuery, mipAndCone );

    // TODO: reuse data if committed == candidate ( use T to check )
    GeometryProps props = ( GeometryProps )0;
    props.mip = mipAndCone.x;

    if( rayQuery.CommittedStatus( ) == COMMITTED_NOTHING )
    {
        props.hitT = INF;
        props.X = origin + direction * props.hitT;
        props.Xprev = props.X;
    }
    else
    {
        props.hitT = rayQuery.CommittedRayT( );

        // Instance
        uint instanceIndex = rayQuery.CommittedInstanceID( ) + rayQuery.CommittedGeometryIndex( );
        props.instanceIndex = instanceIndex;

        InstanceData instanceData = gIn_InstanceData[ instanceIndex ];

        // Texture offset and flags
        props.textureOffsetAndFlags = instanceData.textureOffsetAndFlags;

        // Transform
        float3x3 mObjectToWorld = ( float3x3 )rayQuery.CommittedObjectToWorld3x4( );
        float3x4 mOverloaded = float3x4( instanceData.mOverloadedMatrix0, instanceData.mOverloadedMatrix1, instanceData.mOverloadedMatrix2 ); \

        if( props.Has( FLAG_STATIC ) )
            mObjectToWorld = ( float3x3 )mOverloaded;

        float flip = Math::Sign( instanceData.scale ) * ( rayQuery.CommittedTriangleFrontFace( ) ? -1.0 : 1.0 );

        // Primitive
        uint primitiveIndex = instanceData.primitiveOffset + rayQuery.CommittedPrimitiveIndex( );
        PrimitiveData primitiveData = gIn_PrimitiveData[ primitiveIndex ];

        float worldArea = primitiveData.worldArea * instanceData.scale * instanceData.scale;

        // Barycentrics
        float3 barycentrics;
        barycentrics.yz = rayQuery.CommittedTriangleBarycentrics( );
        barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

        // Normal
        float3 n0 = Packing::DecodeUnitVector( primitiveData.n0, true );
        float3 n1 = Packing::DecodeUnitVector( primitiveData.n1, true );
        float3 n2 = Packing::DecodeUnitVector( primitiveData.n2, true );

        float3 N = barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2;
        N = Geometry::RotateVector( mObjectToWorld, N );
        N = normalize( N * flip );
        props.N = -N; // TODO: why negated?

        // Curvature
        float dnSq0 = Math::LengthSquared( n0 - n1 );
        float dnSq1 = Math::LengthSquared( n1 - n2 );
        float dnSq2 = Math::LengthSquared( n2 - n0 );
        float dnSq = max( dnSq0, max( dnSq1, dnSq2 ) );
        props.curvature = sqrt( dnSq / worldArea );

        // Mip level
        float NoRay = abs( dot( direction, props.N ) );
        float a = props.hitT * mipAndCone.y;
        a *= Math::PositiveRcp( NoRay );
        a *= sqrt( primitiveData.uvArea / worldArea );

        float mip = log2( a );
        mip += MAX_MIP_LEVEL;
        mip = max( mip, 0.0 );
        props.mip += mip;

        // Uv
        props.uv = barycentrics.x * primitiveData.uv0 + barycentrics.y * primitiveData.uv1 + barycentrics.z * primitiveData.uv2;

        // Tangent
        float3 t0 = Packing::DecodeUnitVector( primitiveData.t0, true );
        float3 t1 = Packing::DecodeUnitVector( primitiveData.t1, true );
        float3 t2 = Packing::DecodeUnitVector( primitiveData.t2, true );

        float3 T = barycentrics.x * t0 + barycentrics.y * t1 + barycentrics.z * t2;
        T = Geometry::RotateVector( mObjectToWorld, T );
        T = normalize( T );
        props.T = float4( T, primitiveData.bitangentSign_unused.x );

        props.X = origin + direction * props.hitT;
        if( props.Has( FLAG_DEFORMABLE ) )
        {
            MorphedPrimitivePrevPositions prev = gIn_MorphedPrimitivePrevPositions[ instanceData.morphedPrimitiveOffset + rayQuery.CommittedPrimitiveIndex( ) ];

            float3 XprevLocal = barycentrics.x * prev.pos0.xyz + barycentrics.y * prev.pos1.xyz + barycentrics.z * prev.pos2.xyz;
            props.Xprev = Geometry::AffineTransform( mOverloaded, XprevLocal );
        }
        else if( !props.Has( FLAG_STATIC ) )
            props.Xprev = Geometry::AffineTransform( mOverloaded, props.X );
        else
            props.Xprev = props.X;
    }

    props.V = -direction;

    return props;
}

//====================================================================================================================================
// MATERIAL PROPERTIES
//====================================================================================================================================

struct MaterialProps
{
    float3 Ldirect; // unshadowed
    float3 Lemi;
    float3 N;
    float3 T;
    float3 baseColor;
    float roughness;
    float metalness;
    float curvature;
};

MaterialProps GetMaterialProps( GeometryProps geometryProps, bool viewIndependentLightingModel = false )
{
    MaterialProps props = ( MaterialProps )0;

    float3 Csky = GetSkyIntensity( -geometryProps.V );

    [branch]
    if( geometryProps.IsSky( ) )
    {
        props.Lemi = Csky;

        return props;
    }

    uint baseTexture = geometryProps.GetBaseTexture( );
    InstanceData instanceData = gIn_InstanceData[ geometryProps.instanceIndex ];

    // Base color
    float3 coords = GetSamplingCoords( baseTexture, geometryProps.uv, geometryProps.mip, MIP_SHARP );
    float4 color = gIn_Textures[ NonUniformResourceIndex( baseTexture ) ].SAMPLE( coords );
    color.xyz *= instanceData.baseColorAndMetalnessScale.xyz;
    color.xyz *= geometryProps.Has( FLAG_TRANSPARENT ) ? 1.0 : Math::PositiveRcp( color.w ); // Correct handling of BC1 with pre-multiplied alpha
    float3 baseColor = saturate( color.xyz );

    // Roughness and metalness
    coords = GetSamplingCoords( baseTexture + 1, geometryProps.uv, geometryProps.mip, MIP_SHARP );
    float3 materialProps = gIn_Textures[ NonUniformResourceIndex( baseTexture + 1 ) ].SAMPLE( coords ).xyz;
    float roughness = saturate( materialProps.y * instanceData.emissionAndRoughnessScale.w );
    float metalness = saturate( materialProps.z * instanceData.baseColorAndMetalnessScale.w );

    // Normal
    coords = GetSamplingCoords( baseTexture + 2, geometryProps.uv, geometryProps.mip, MIP_LESS_SHARP );
    float2 packedNormal = gIn_Textures[ NonUniformResourceIndex( baseTexture + 2 ) ].SAMPLE( coords ).xy;
    float3 N = gUseNormalMap ? Geometry::TransformLocalNormal( packedNormal, geometryProps.T, geometryProps.N ) : geometryProps.N;
    float3 T = geometryProps.T.xyz;

    // Estimate curvature
    float viewZ = Geometry::AffineTransform( gWorldToView, geometryProps.X ).z;
    float pixelSize = gUnproject * lerp( abs( viewZ ), 1.0, abs( gOrthoMode ) );
    float localCurvature = length( Geometry::UnpackLocalNormal( packedNormal ).xy ) * float( gUseNormalMap );
    localCurvature /= pixelSize;

    // Emission
    coords = GetSamplingCoords( baseTexture + 3, geometryProps.uv, geometryProps.mip, MIP_VISIBILITY );
    float3 Lemi = gIn_Textures[ NonUniformResourceIndex( baseTexture + 3 ) ].SAMPLE( coords ).xyz;
    Lemi *= instanceData.emissionAndRoughnessScale.xyz;
    Lemi *= ( baseColor + 0.01 ) / ( max( baseColor, max( baseColor, baseColor ) ) + 0.01 );

    [flatten]
    if( geometryProps.Has( FLAG_FORCED_EMISSION ) )
    {
        Lemi = geometryProps.GetForcedEmissionColor( );
        baseColor = 0.0;
    }

    Lemi *= gEmissionIntensity;

    // Material overrides
    [flatten]
    if( gForcedMaterial == MATERIAL_GYPSUM )
    {
        roughness = 1.0;
        baseColor = 0.5;
        metalness = 0.0;
    }
    else if( gForcedMaterial == MATERIAL_COBALT )
    {
        roughness = pow( saturate( baseColor.x * baseColor.y * baseColor.z ), 0.33333 );
        baseColor = float3( 0.672411, 0.637331, 0.585456 );
        metalness = 1.0;

        #if( USE_ANOTHER_COBALT == 1 )
            roughness = pow( saturate( roughness - 0.1 ), 0.25 ) * 0.3 + 0.07;
        #endif
    }

    if( geometryProps.Has( FLAG_HAIR ) )
    {
        roughness = gHairBetas.x;
        metalness = gHairBetas.y;
        baseColor = gHairBaseColor.xyz;
    }
    else
    {
        metalness = gMetalnessOverride == 0.0 ? metalness : gMetalnessOverride;
        roughness = gRoughnessOverride == 0.0 ? roughness : gRoughnessOverride;
    }

    #if( USE_PUDDLES == 1 )
        roughness *= Math::SmoothStep( 0.6, 0.8, length( frac( geometryProps.uv ) * 2.0 - 1.0 ) );
    #endif

    #if( USE_RANDOMIZED_ROUGHNESS == 1 )
        float2 noise = ( frac( sin( dot( geometryProps.uv, float2( 12.9898, 78.233 ) * 2.0 ) ) * 43758.5453 ) );
        float noise01 = abs( noise.x + noise.y ) * 0.5;
        roughness *= 1.0 + ( noise01 * 2.0 - 1.0 ) * 0.25;
    #endif

    roughness = saturate( roughness );
    metalness = saturate( metalness );

    // Transform to diffuse material if emission is here
    float emissionLevel = Color::Luminance( Lemi );
    emissionLevel = saturate( emissionLevel * 50.0 );

    metalness = lerp( metalness, 0.0, emissionLevel );
    roughness = lerp( roughness, 1.0, emissionLevel );

    // Direct lighting ( no shadow )
    float3 Ldirect = 0;
    float NoL = saturate( dot( geometryProps.N, gSunDirection.xyz ) );
    float shadow = geometryProps.Has( FLAG_HAIR ) ? float( NoL > 0.0 ) : Math::SmoothStep( 0.03, 0.1, NoL );

    [branch]
    if( shadow != 0.0 )
    {
        float3 Csun = GetSunIntensity( gSunDirection.xyz );

        if( geometryProps.Has( FLAG_HAIR ) )
        {
            float3x3 mLocalBasis = HairGetBasis( N, T );
            float3 vLocal = Geometry::RotateVector( mLocalBasis, geometryProps.V );

            HairSurfaceData hairSd = ( HairSurfaceData )0;
            hairSd.N = float3( 0, 0, 1 );
            hairSd.T = float3( 1, 0, 0 );
            hairSd.V = vLocal;

            HairData hairData = ( HairData )0;
            hairData.baseColor = baseColor;
            hairData.betaM = roughness;
            hairData.betaN = metalness;

            HairContext hairBrdf = HairContextInit( hairSd, hairData );

            float3 sunLocal = Geometry::RotateVector( mLocalBasis, gSunDirection.xyz );
            float3 throughput = HairEval( hairBrdf, vLocal, sunLocal );

            Ldirect = Csun * throughput;
        }
        else
        {
            // Pseudo sky importance sampling
            float3 Cimp = lerp( Csky, Csun, Math::SmoothStep( 0.0, 0.2, roughness ) );
            Cimp *= Math::SmoothStep( -0.01, 0.05, gSunDirection.z );

            // Extract materials
            float3 albedo, Rf0;
            BRDF::ConvertBaseColorMetalnessToAlbedoRf0( baseColor.xyz, metalness, albedo, Rf0 );

            // Apply lighting
            if( viewIndependentLightingModel )
            {
                // Very simple "diffuse-like" model
                float m = roughness * roughness;
                float3 C = albedo * Csun + Rf0 * m * Cimp;
                float Kdiff = NoL / Math::Pi( 1.0 );

                Ldirect = Kdiff * C;
            }
            else
            {
                float3 Cdiff, Cspec;
                BRDF::DirectLighting( N, gSunDirection.xyz, geometryProps.V, Rf0, roughness, Cdiff, Cspec );

                Ldirect = Cdiff * albedo * Csun + Cspec * Cimp;
            }
        }

        Ldirect *= shadow;
    }

    // Output
    props.Ldirect = Ldirect;
    props.Lemi = Lemi;
    props.N = N;
    props.T = T;
    props.baseColor = baseColor;
    props.roughness = roughness;
    props.metalness = metalness;
    props.curvature = geometryProps.curvature + localCurvature;

    return props;
}

//====================================================================================================================================
// MISC
//====================================================================================================================================

float GetDeltaEventRay( GeometryProps geometryProps, bool isReflection, float eta, out float3 Xoffset, out float3 ray )
{
    if( isReflection )
        ray = reflect( -geometryProps.V, geometryProps.N );
    else
    {
        float3 I = -geometryProps.V;
        float NoI = dot( geometryProps.N, I );
        float k = max( 1.0 - eta * eta * ( 1.0 - NoI * NoI ), 0.0 );

        ray = normalize( eta * I - ( eta * NoI + sqrt( k ) ) * geometryProps.N );
        eta = 1.0 / eta;
    }

    float amount = geometryProps.Has( FLAG_TRANSPARENT ) ? PT_GLASS_RAY_OFFSET : PT_BOUNCE_RAY_OFFSET;
    float s = Math::Sign( dot( ray, geometryProps.N ) );

    Xoffset = geometryProps.GetXoffset( geometryProps.N * s, amount );

    return eta;
}

bool IsDelta( MaterialProps materialProps )
{
    return materialProps.roughness < 0.044 && ( materialProps.metalness > 0.941 || Color::Luminance( materialProps.baseColor ) < 0.005 ); // TODO: tweaked for some content?
}

#define SKIP_SOFT_SHADOWS 0x1
#define SKIP_EMISSIVE 0x2

float3 GetShadowedLighting( GeometryProps geometryProps, MaterialProps materialProps, uint flags = 0 )
{
    const uint instanceInclusionMask = FLAG_NON_TRANSPARENT; // Default shadow rays must ignore transparency // TODO: what about translucency?
    const uint rayFlags = 0;

    float3 L = materialProps.Ldirect;

    if( Color::Luminance( L ) != 0 && !gDisableShadowsAndEnableImportanceSampling )
    {
        float2 rnd = Rng::Hash::GetFloat2( );
        rnd = ImportanceSampling::Cosine::GetRay( rnd ).xy;
        rnd *= gTanSunAngularRadius;
        rnd *= float( ( flags & SKIP_SOFT_SHADOWS ) == 0 );

        float3 sunDirection = normalize( gSunBasisX.xyz * rnd.x + gSunBasisY.xyz * rnd.y + gSunDirection.xyz );

        float2 mipAndCone = GetConeAngleFromAngularRadius( geometryProps.mip, gTanSunAngularRadius );
        float3 Xoffset = geometryProps.GetXoffset( sunDirection, PT_SHADOW_RAY_OFFSET );
        L *= CastVisibilityRay_AnyHit( Xoffset, sunDirection, 0.0, INF, mipAndCone, gWorldTlas, instanceInclusionMask, rayFlags );
    }

    if( ( flags & SKIP_EMISSIVE ) == 0 )
        L += materialProps.Lemi;

    return NRD_MODE < OCCLUSION ? L : 0.0;
}

float EstimateDiffuseProbability( GeometryProps geometryProps, MaterialProps materialProps, bool useMagicBoost = false )
{
    // IMPORTANT: can't be used for hair tracing, but applicable in other hair related calculations
    float3 albedo, Rf0;
    BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

    float NoV = abs( dot( materialProps.N, geometryProps.V ) );
    float3 Fenv = BRDF::EnvironmentTerm_Rtg( Rf0, NoV, materialProps.roughness );

    float lumSpec = Color::Luminance( Fenv );
    float lumDiff = Color::Luminance( albedo * ( 1.0 - Fenv ) );

    float diffProb = lumDiff / max( lumDiff + lumSpec, NRD_EPS );

    // Boost diffussiness ( aka diffuse-like behavior ) if roughness is high
    if( useMagicBoost )
        diffProb = lerp( diffProb, 1.0, GetSpecMagicCurve( materialProps.roughness ) );

    // Clamp probability to a sane range. High energy fireflies are very undesired. They can be get rid of only
    // if the number of accumulated samples exeeds 100-500. NRD accumulates for not more than 30 frames only
    float diffProbClamped = clamp( diffProb, 1.0 / PT_MAX_FIREFLY_RELATIVE_INTENSITY, 1.0 - 1.0 / PT_MAX_FIREFLY_RELATIVE_INTENSITY );

    [flatten]
    if( diffProb < PT_EVIL_TWIN_LOBE_TOLERANCE )
        return 0.0; // no diffuse materials are common ( metals )
    else if( diffProb > 1.0 - PT_EVIL_TWIN_LOBE_TOLERANCE )
        return 1.0; // no specular materials are uncommon ( broken material model? )
    else
        return diffProbClamped;
}

float ReprojectIrradiance(
    bool isPrevFrame, bool isRefraction,
    Texture2D<float3> texDiff, Texture2D<float4> texSpecViewZ,
    GeometryProps geometryProps, uint2 pixelPos,
    out float3 Ldiff, out float3 Lspec
)
{
    // Get UV and ignore back projection
    float2 uv = Geometry::GetScreenUv( isPrevFrame ? gWorldToClipPrev : gWorldToClip, geometryProps.X, true ) - gJitter;

    float2 rescale = ( isPrevFrame ? gRectSizePrev : gRectSize ) * gInvRenderSize;
    float4 data = texSpecViewZ.SampleLevel( gNearestSampler, uv * rescale, 0 );
    float prevViewZ = abs( data.w ) / FP16_VIEWZ_SCALE;

    // Initial state
    float weight = 1.0;
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;

    // Relaxed checks for refractions
    float viewZ = abs( Geometry::AffineTransform( isPrevFrame ? gWorldToViewPrev : gWorldToView, geometryProps.X ).z );
    float err = ( viewZ - prevViewZ ) * Math::PositiveRcp( max( viewZ, prevViewZ ) );

    if( isRefraction )
    {
        // Confidence - viewZ ( PSR makes prevViewZ further than the original primary surface )
        weight *= Math::LinearStep( 0.01, 0.005, saturate( err ) );

        // Fade-out on screen edges ( hard )
        weight *= all( saturate( uv ) == uv );
    }
    else
    {
        // Confidence - viewZ
        weight *= Math::LinearStep( 0.01, 0.005, abs( err ) );

        // Fade-out on screen edges ( soft )
        float2 f = Math::LinearStep( 0.0, 0.1, uv ) * Math::LinearStep( 1.0, 0.9, uv );
        weight *= f.x * f.y;

        // Confidence - ignore back-facing
        // Instead of storing previous normal we can store previous NoL, if signs do not match we hit the surface from the opposite side
        float NoL = dot( geometryProps.N, gSunDirection.xyz );
        weight *= float( NoL * Math::Sign( data.w ) > 0.0 );

        // Confidence - ignore too short rays
        float2 uv = Geometry::GetScreenUv( gWorldToClip, geometryProps.X, true ) - gJitter;
        float d = length( ( uv - pixelUv ) * gRectSize );
        weight *= Math::LinearStep( 1.0, 3.0, d );
    }

    // Ignore sky
    weight *= float( !geometryProps.IsSky( ) );

    // Use only if radiance is on the screen
    weight *= float( gOnScreen < SHOW_AMBIENT_OCCLUSION );

    // Add global confidence
    if( isPrevFrame )
        weight *= gPrevFrameConfidence; // see C++ code for details

    // Read data
    Ldiff = texDiff.SampleLevel( gNearestSampler, uv * rescale, 0 );
    Lspec = data.xyz;

    // Avoid NANs
    [flatten]
    if( any( isnan( Ldiff ) | isinf( Ldiff ) | isnan( Lspec ) | isinf( Lspec ) ) || NRD_MODE >= OCCLUSION )
    {
        Ldiff = 0;
        Lspec = 0;
        weight = 0;
    }

    // Avoid really bad reprojection
    float f = saturate( weight / 0.001 );
    Ldiff *= f;
    Lspec *= f;

    return weight;
}
