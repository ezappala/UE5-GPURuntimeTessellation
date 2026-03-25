// Licensed under the MIT License. See LICENSE file in the project root.

#include "GPUTessellationComponent.h"
#include "GPUTessellationMeshBuilder.h"
#include "GPUTessellationSceneProxy.h"
#include "PrimitiveSceneProxy.h"
#include "RenderingThread.h"
#include "Engine/TextureRenderTarget.h"
#include "Engine/World.h"
#include "GameFramework/PlayerController.h"
#include "Materials/MaterialInterface.h"

#if WITH_EDITOR
#include "Editor.h"
#include "EditorViewportClient.h"
#endif

UGPUTessellationComponent::UGPUTessellationComponent(
    const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer),
    LastCameraPosition(FVector::ZeroVector),
    CurrentResolution(32, 32) {
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = true;

#if WITH_EDITOR
    // Enable ticking in editor so LOD works in viewport
    bTickInEditor = true;
#endif

    // Set default bounds
    Bounds = FBoxSphereBounds(FBox(FVector(-500, -500, -100), FVector(500, 500, 100)));

    // Enable shadow casting
    bCastDynamicShadow = true;
    bCastStaticShadow = false;
    bAffectDynamicIndirectLighting = true;
    bAffectDistanceFieldLighting = true;
}

FPrimitiveSceneProxy* UGPUTessellationComponent::CreateSceneProxy() {
    if (TessellationSettings.TessellationFactor > 0.0f) {
        return new FGPUTessellationSceneProxy(this);
    }
    return nullptr;
}

FBoxSphereBounds UGPUTessellationComponent::CalcBounds(const FTransform& LocalToWorld) const {
    // Calculate bounds based on plane size and displacement (plane is XZ, Y is up)
    const float HalfSizeX = TessellationSettings.PlaneSizeX * 0.5f;
    const float HalfSizeZ = TessellationSettings.PlaneSizeY * 0.5f;
    // PlaneSizeY is actually Z dimension
    const float MaxDisplacement = TessellationSettings.DisplacementIntensity + FMath::Abs(
        TessellationSettings.DisplacementOffset);

    // Check for zero or near-zero scale which would make bounds invalid
    const FVector Scale3D = LocalToWorld.GetScale3D();
    constexpr float MinScale = 0.001f;
    if (FMath::IsNearlyZero(Scale3D.X, MinScale) ||
        FMath::IsNearlyZero(Scale3D.Y, MinScale) ||
        FMath::IsNearlyZero(Scale3D.Z, MinScale)) {
        // This is an error condition - always log as Warning
        if (bEnableDebugLogging) {
            UE_LOG(
                LogTemp,
                Warning,
                TEXT(
                    "GPUTessellation: CalcBounds - ZERO OR NEAR-ZERO SCALE DETECTED: %s - Using identity scale"
                ),
                *Scale3D.ToString());
        }
        // Use a transform with identity scale
        FTransform FixedTransform = LocalToWorld;
        FixedTransform.SetScale3D(FVector::OneVector);

        const FBox LocalBox(
            FVector(-HalfSizeX, -MaxDisplacement, -HalfSizeZ),
            FVector(HalfSizeX, MaxDisplacement, HalfSizeZ)
        );

        return FBoxSphereBounds(LocalBox).TransformBy(FixedTransform);
    }

    const FBox LocalBox(
        FVector(-HalfSizeX, -MaxDisplacement, -HalfSizeZ),
        FVector(HalfSizeX, MaxDisplacement, HalfSizeZ)
    );

    const FBoxSphereBounds Result = FBoxSphereBounds(LocalBox).TransformBy(LocalToWorld);

    // Throttled logging (max once every 2 seconds)
    if (bEnableDebugLogging) {
        const double CurrentTime = FPlatformTime::Seconds();
        if (CurrentTime - LastLogTime >= 2.0) {
            LastLogTime = CurrentTime;
            UE_LOG(
                LogTemp,
                Log,
                TEXT(
                    "GPUTessellation: CalcBounds - PlaneSizeX:%.1f PlaneSizeZ:%.1f MaxDisp:%.1f Scale:%s Result:%s"
                ),
                TessellationSettings.PlaneSizeX,
                TessellationSettings.PlaneSizeY,
                MaxDisplacement,
                *Scale3D.ToString(),
                *Result.ToString());
        }
    }

    return Result;
}

void UGPUTessellationComponent::GetUsedMaterials(
    TArray<UMaterialInterface*>& OutMaterials,
    bool bGetDebugMaterials
) const { if (Material) { OutMaterials.AddUnique(Material); } }

int32 UGPUTessellationComponent::GetNumMaterials() const { return Material ? 1 : 0; }

UMaterialInterface* UGPUTessellationComponent::GetMaterial(const int32 ElementIndex) const {
    return ElementIndex == 0 ? Material : nullptr;
}

void UGPUTessellationComponent::TickComponent(
    const float DeltaTime,
    const ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction
) {
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (!bAutoUpdate) { return; }

    // Update LOD based on selected mode
    switch (TessellationSettings.LODMode) {
    case EGPUTessellationLODMode::DistanceBased: {
        // Initialize LOD system on first update
        static bool bInitialized = false;
        if (!bInitialized) {
            // Initialize from MaxTessellationFactor (LOD range max)
            CurrentLODLevel = static_cast<float>(TessellationSettings.MaxTessellationFactor);
            LastAppliedTessFactor = TessellationSettings.MaxTessellationFactor;
            bInitialized = true;

            if (bEnableDebugLogging) {
                UE_LOG(
                    LogTemp,
                    Warning,
                    TEXT("GPUTessellation: LOD Initialized - Max Factor: %d, Min Factor: %d"),
                    TessellationSettings.MaxTessellationFactor,
                    TessellationSettings.MinTessellationFactor);
            }
        }
        UpdateDistanceBasedLOD(DeltaTime);
        break;
    }

    case EGPUTessellationLODMode::DistanceBasedDiscrete: {
        UpdateDiscreteLOD(DeltaTime);
        break;
    }

    case EGPUTessellationLODMode::DistanceBasedPatches: {
        UpdatePatchBasedLOD(DeltaTime);
        break;
    }

    case EGPUTessellationLODMode::DensityTexture: UpdateDensityBasedLOD(DeltaTime);
        break;

    case EGPUTessellationLODMode::Disabled:
    default:
        // No LOD - use TessellationFactor directly via CalculateGridResolution()
        break;
    }

    // Check if any textures are render targets (dynamic textures that update every frame)
    // If so, force mesh regeneration every frame to reflect the changes (with optional FPS limiting)
    if (bAutoUpdateRenderTargets) {
        bool bHasRenderTarget = false;

        if (DisplacementTexture && DisplacementTexture->IsA<UTextureRenderTarget>()) {
            bHasRenderTarget = true;
        }
        if (SubtractTexture && SubtractTexture->IsA<UTextureRenderTarget>()) {
            bHasRenderTarget = true;
        }
        if (NormalMapTexture && NormalMapTexture->IsA<UTextureRenderTarget>()) {
            bHasRenderTarget = true;
        }

        // Force update when using render targets, with optional FPS limiting
        if (bHasRenderTarget) {
            bool bShouldUpdate = true;

            // Apply FPS limiting if specified (0 = unlimited)
            if (RenderTargetUpdateFPS > 0) {
                const double CurrentTime = FPlatformTime::Seconds();
                const double MinTimeBetweenUpdates = 1.0 / static_cast<double>(
                    RenderTargetUpdateFPS);

                if (CurrentTime - LastRenderTargetUpdateTime < MinTimeBetweenUpdates) {
                    bShouldUpdate = false;
                } else { LastRenderTargetUpdateTime = CurrentTime; }
            }

            if (bShouldUpdate) { MarkRenderStateDirty(); }
        }
    }
}

void UGPUTessellationComponent::OnRegister() {
    Super::OnRegister();

    // Update bounds before scene proxy creation
    UpdateBounds();

    // Initial mesh generation
    if (bAutoUpdate) { UpdateTessellatedMesh(); }
}

void UGPUTessellationComponent::OnUnregister() { Super::OnUnregister(); }

#if WITH_EDITOR
void UGPUTessellationComponent::PostEditChangeProperty(
    FPropertyChangedEvent& PropertyChangedEvent) {
    Super::PostEditChangeProperty(PropertyChangedEvent);

    // Update mesh when properties change
    if (PropertyChangedEvent.Property != nullptr) { MarkRenderStateDirty(); }
}
#endif

void UGPUTessellationComponent::UpdateTessellatedMesh() { MarkRenderStateDirty(); }

void UGPUTessellationComponent::SetDisplacementTexture(UTexture* InTexture) {
    DisplacementTexture = InTexture;
    UpdateTessellatedMesh();
}

void UGPUTessellationComponent::SetSubtractTexture(UTexture* InTexture) {
    SubtractTexture = InTexture;
    UpdateTessellatedMesh();
}

void UGPUTessellationComponent::SetNormalMapTexture(UTexture* InTexture) {
    NormalMapTexture = InTexture;
    UpdateTessellatedMesh();
}

void UGPUTessellationComponent::SetMaterial(
    const int32 ElementIndex,
    UMaterialInterface* InMaterial) {
    if (ElementIndex == 0) {
        Material = InMaterial;
        MarkRenderStateDirty();
    }
}

void UGPUTessellationComponent::UpdateSettings(const FGPUTessellationSettings& NewSettings) {
    TessellationSettings = NewSettings;
    UpdateTessellatedMesh();
}

FIntPoint UGPUTessellationComponent::GetTessellationResolution() const {
    return CalculateGridResolution();
}

int32 UGPUTessellationComponent::GetVertexCount() const {
    const FIntPoint Res = CalculateGridResolution();
    return Res.X * Res.Y;
}

int32 UGPUTessellationComponent::GetTriangleCount() const {
    const FIntPoint Res = CalculateGridResolution();
    return (Res.X - 1) * (Res.Y - 1) * 2;
}

void UGPUTessellationComponent::MarkRenderStateDirty() { Super::MarkRenderStateDirty(); }

FIntPoint UGPUTessellationComponent::CalculateGridResolution() const {
    // Calculate resolution based on tessellation factor
    // When LOD is enabled, use the calculated LOD factor; otherwise use user's TessellationFactor
    const int32 EffectiveTessellationFactor = TessellationSettings.LODMode !=
        EGPUTessellationLODMode::Disabled
        ? LastAppliedTessFactor
        : TessellationSettings.TessellationFactor;

    int32 Resolution = EffectiveTessellationFactor * 4;
    Resolution = FMath::Clamp(Resolution, 4, 1024); // Max 1024 to support tessellation up to 256

    // Make it a multiple of 8 for better compute shader performance
    Resolution = FMath::DivideAndRoundUp(Resolution, 8) * 8;

    CurrentResolution = FIntPoint(Resolution, Resolution);
    return CurrentResolution;
}

void UGPUTessellationComponent::UpdateDistanceBasedLOD(const float DeltaTime) {
    // Get camera position - works in both editor and game mode
    const UWorld* World = GetWorld();
    if (World == nullptr) { return; }

    FVector CameraPos = FVector::ZeroVector;
    bool bFoundCamera = false;

    // Try to get camera from player controller (game mode)
    if (const APlayerController* PC = World->GetFirstPlayerController()) {
        FVector ViewLocation;
        FRotator ViewRotation;
        PC->GetPlayerViewPoint(ViewLocation, ViewRotation);
        CameraPos = ViewLocation;
        bFoundCamera = true;
    }
#if WITH_EDITOR
    // In editor, use editor viewport camera
    else if (GEditor != nullptr && GEditor->GetActiveViewport() != nullptr) {
        const FViewport* Viewport = GEditor->GetActiveViewport();
        const FEditorViewportClient* ViewportClient = static_cast<FEditorViewportClient*>(Viewport->
            GetClient());
        if (ViewportClient != nullptr) {
            CameraPos = ViewportClient->GetViewLocation();
            bFoundCamera = true;
        }
    }
#endif

    if (!bFoundCamera) {
        if (bEnableDebugLogging) {
            UE_LOG(LogTemp, Warning, TEXT("GPUTessellation LOD: NO CAMERA FOUND!"));
        }
        return;
    }

    // Calculate distance (to pivot or bounds based on setting)
    FVector ComponentPos;
    const float Distance = CalculateDistanceToCamera(CameraPos, ComponentPos);

    // Account for component scale - larger objects should use LOD at proportionally larger distances
    // Use the maximum scale component to represent overall size
    const FVector Scale3D = GetComponentScale();
    const float MaxScale = FMath::Max3(
        FMath::Abs(Scale3D.X),
        FMath::Abs(Scale3D.Y),
        FMath::Abs(Scale3D.Z));

    // Scale LOD distances by the component scale
    // This makes LOD distances work consistently regardless of actor scale
    // Example: Actor at scale 100 will use LOD distances 100x larger
    const float ScaledMinDistance = TessellationSettings.MinTessellationDistance * MaxScale;
    const float ScaledMaxDistance = TessellationSettings.MaxTessellationDistance * MaxScale;

    // Store camera position for tracking changes
    const float CameraMovement = FVector::Dist(CameraPos, LastCameraPosition);
    LastCameraPosition = CameraPos;

    // Calculate target LOD factor based on distance (using scaled distances)
    const int32 TargetTessFactor = CalculateLODFactorScaled(
        Distance,
        ScaledMinDistance,
        ScaledMaxDistance);

    // Debug logging (throttled) - show LOD calculation every 2 seconds
    if (bEnableDebugLogging) {
        const double CurrentTime = FPlatformTime::Seconds();
        if (CurrentTime - LastLogTime >= 2.0) {
            LastLogTime = CurrentTime;

            // Calculate which zone we're in (using scaled distances)
            FString DistanceZone;
            if (Distance <= ScaledMinDistance) {
                DistanceZone = TEXT("NEAR (Max Tessellation)");
            } else if (Distance >=
                ScaledMaxDistance) { DistanceZone = TEXT("FAR (Min Tessellation)"); } else {
                const float DistanceRange = ScaledMaxDistance - ScaledMinDistance;
                const float DistanceInRange = Distance - ScaledMinDistance;
                const float Percentage = DistanceInRange / DistanceRange * 100.0f;
                DistanceZone = FString::Printf(
                    TEXT("TRANSITION (%.1f%% through range)"),
                    Percentage);
            }

            UE_LOG(LogTemp, Warning, TEXT("GPUTessellation LOD Status:"));
            UE_LOG(
                LogTemp,
                Warning,
                TEXT("  Camera: %s (moved %.1f since last frame)"),
                *CameraPos.ToString(),
                CameraMovement);
            UE_LOG(
                LogTemp,
                Warning,
                TEXT("  Component: %s, Scale: %.2f (max component)"),
                *ComponentPos.ToString(),
                MaxScale);
            UE_LOG(
                LogTemp,
                Warning,
                TEXT("  Distance: %.1f units (%.1f meters) - %s"),
                Distance,
                Distance / 100.0f,
                *DistanceZone);
            UE_LOG(
                LogTemp,
                Warning,
                TEXT("  Distance Range (scaled): %.1f to %.1f (base: %.1f to %.1f, scale: %.2fx)"),
                ScaledMinDistance,
                ScaledMaxDistance,
                TessellationSettings.MinTessellationDistance,
                TessellationSettings.MaxTessellationDistance,
                MaxScale);
            UE_LOG(
                LogTemp,
                Warning,
                TEXT("  Target LOD: %d, Current: %.1f, Applied: %d"),
                TargetTessFactor,
                CurrentLODLevel,
                LastAppliedTessFactor);
            UE_LOG(
                LogTemp,
                Warning,
                TEXT("  Factor Range: %d (max) to %d (min)"),
                TessellationSettings.MaxTessellationFactor,
                TessellationSettings.MinTessellationFactor);
            UE_LOG(
                LogTemp,
                Warning,
                TEXT("  User TessellationFactor: %d (NOT modified by LOD)"),
                TessellationSettings.TessellationFactor);
            UE_LOG(
                LogTemp,
                Warning,
                TEXT("  Mode: %s, DeltaTime: %.4f"),
                World->WorldType == EWorldType::Editor ? TEXT("Editor") : TEXT("Game"),
                DeltaTime);
        }
    }

    // Smooth interpolation for transitions
    if (TessellationSettings.LODTransitionSpeed > 0.0f) {
        CurrentLODLevel = FMath::FInterpTo(
            CurrentLODLevel,
            static_cast<float>(TargetTessFactor),
            DeltaTime,
            TessellationSettings.LODTransitionSpeed
        );
    } else { CurrentLODLevel = static_cast<float>(TargetTessFactor); }

    const int32 NewTessFactor = FMath::RoundToInt(CurrentLODLevel);

    // Apply hysteresis to prevent oscillation
    if (FMath::Abs(NewTessFactor - LastAppliedTessFactor) > TessellationSettings.LODHysteresis) {
        // LOD changed significantly - apply it (store for grid resolution calculation)
        if (bEnableDebugLogging) {
            UE_LOG(LogTemp, Warning, TEXT("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"));
            UE_LOG(LogTemp, Warning, TEXT("GPUTessellation: LOD TRANSITION"));
            UE_LOG(
                LogTemp,
                Warning,
                TEXT("  Change: %d -> %d (diff: %d, hysteresis: %d)"),
                LastAppliedTessFactor,
                NewTessFactor,
                FMath::Abs(NewTessFactor - LastAppliedTessFactor),
                TessellationSettings.LODHysteresis);
            UE_LOG(
                LogTemp,
                Warning,
                TEXT("  Distance: %.1f units (%.1f meters)"),
                Distance,
                Distance / 100.0f);
            UE_LOG(LogTemp, Warning, TEXT("  Camera: %s"), *CameraPos.ToString());
            UE_LOG(LogTemp, Warning, TEXT("  Component: %s"), *ComponentPos.ToString());
            UE_LOG(
                LogTemp,
                Warning,
                TEXT("  TessellationFactor preserved: %d"),
                TessellationSettings.TessellationFactor);
            UE_LOG(LogTemp, Warning, TEXT("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"));
        }

        // CRITICAL: Store LOD factor separately - DO NOT modify user's TessellationFactor!
        LastAppliedTessFactor = NewTessFactor;
        MarkRenderStateDirty();
    }
}

void UGPUTessellationComponent::UpdateDiscreteLOD(float DeltaTime) {
    // Get camera position
    const UWorld* World = GetWorld();
    if (World == nullptr) { return; }

    FVector CameraPos = FVector::ZeroVector;
    bool bFoundCamera = false;

    // Try to get camera from player controller (game mode)
    if (const APlayerController* PC = World->GetFirstPlayerController()) {
        FVector ViewLocation;
        FRotator ViewRotation;
        PC->GetPlayerViewPoint(ViewLocation, ViewRotation);
        CameraPos = ViewLocation;
        bFoundCamera = true;
    }
#if WITH_EDITOR
    // In editor, use editor viewport camera
    else if (GEditor != nullptr && GEditor->GetActiveViewport() != nullptr) {
        const FViewport* Viewport = GEditor->GetActiveViewport();
        const FEditorViewportClient* ViewportClient = static_cast<FEditorViewportClient*>(Viewport->
            GetClient());
        if (ViewportClient != nullptr) {
            CameraPos = ViewportClient->GetViewLocation();
            bFoundCamera = true;
        }
    }
#endif

    if (!bFoundCamera) { return; }

    // Calculate distance (to pivot or bounds)
    FVector ComponentPos;
    const float Distance = CalculateDistanceToCamera(CameraPos, ComponentPos);

    // Account for component scale
    const FVector Scale3D = GetComponentScale();
    const float MaxScale = FMath::Max3(
        FMath::Abs(Scale3D.X),
        FMath::Abs(Scale3D.Y),
        FMath::Abs(Scale3D.Z));
    const float ScaledDistance = Distance / MaxScale;

    // Determine which discrete level to use based on distance thresholds
    int32 TargetTessFactor = 4; // Default to lowest

    if (TessellationSettings.DiscreteLODLevels.Num() > 0) {
        // Start with highest quality (first level) and work down
        TargetTessFactor = StaticCast<int32>(TessellationSettings.DiscreteLODLevels[0]);

        // Check each distance threshold
        for (int32 i = 0; i < TessellationSettings.DiscreteLODDistances.Num() && i <
             TessellationSettings.
             DiscreteLODLevels.Num(); ++i) {
            if (ScaledDistance > TessellationSettings.DiscreteLODDistances[i]) {
                // Beyond this threshold, use next lower level if available
                if (i + 1 < TessellationSettings.DiscreteLODLevels.Num()) {
                    TargetTessFactor = StaticCast<int32>(
                        TessellationSettings.DiscreteLODLevels[i + 1]);
                }
            } else {
                // Within threshold, use current level
                TargetTessFactor = StaticCast<int32>(TessellationSettings.DiscreteLODLevels[i]);
                break;
            }
        }
    }

    // Convert enum to actual tessellation factor
    switch (StaticCast<EGPUTessellationPatchLevel>(TargetTessFactor)) {
    case EGPUTessellationPatchLevel::Patch_4: TargetTessFactor = 4;
        break;
    case EGPUTessellationPatchLevel::Patch_8: TargetTessFactor = 8;
        break;
    case EGPUTessellationPatchLevel::Patch_16: TargetTessFactor = 16;
        break;
    case EGPUTessellationPatchLevel::Patch_32: TargetTessFactor = 32;
        break;
    case EGPUTessellationPatchLevel::Patch_64: TargetTessFactor = 64;
        break;
    case EGPUTessellationPatchLevel::Patch_128: TargetTessFactor = 128;
        break;
    default: TargetTessFactor = 16;
        break;
    }

    // Apply hysteresis to prevent oscillation
    const int32 Difference = FMath::Abs(TargetTessFactor - LastAppliedTessFactor);
    if (Difference >= TessellationSettings.LODHysteresis) {
        LastAppliedTessFactor = TargetTessFactor;
        CurrentLODLevel = static_cast<float>(TargetTessFactor);
        MarkRenderStateDirty();

        if (bEnableDebugLogging) {
            UE_LOG(
                LogTemp,
                Warning,
                TEXT("GPUTessellation Discrete LOD: Distance=%.1f (scaled=%.1f), Level=%d"),
                Distance,
                ScaledDistance,
                TargetTessFactor);
        }
    }
}

void UGPUTessellationComponent::UpdatePatchBasedLOD(float DeltaTime) {
    // Spatial patch system generates patches with per-patch LOD based on camera distance
    // Track camera position and send updates to scene proxy when camera moves significantly

    // Get camera position (same logic as other LOD modes)
    FVector CameraPos = FVector::ZeroVector;
    bool bFoundCamera = false;

    if (const UWorld* World = GetWorld()) {
        if (const APlayerController* PC = World->GetFirstPlayerController()) {
            FVector ViewLocation;
            FRotator ViewRotation;
            PC->GetPlayerViewPoint(ViewLocation, ViewRotation);
            CameraPos = ViewLocation;
            bFoundCamera = true;
        }
    }

#if WITH_EDITOR
    // In editor, use editor viewport camera
    if (!bFoundCamera && GEditor != nullptr && GEditor->GetActiveViewport() != nullptr) {
        const FViewport* Viewport = GEditor->GetActiveViewport();
        const FEditorViewportClient* ViewportClient = static_cast<FEditorViewportClient*>(Viewport->
            GetClient());
        if (ViewportClient != nullptr) {
            CameraPos = ViewportClient->GetViewLocation();
            bFoundCamera = true;
        }
    }
#endif

    if (!bFoundCamera) {
        if (bEnableDebugLogging) {
            static double LastWarningTime = 0.0;
            const double CurrentTime = FPlatformTime::Seconds();
            if (CurrentTime - LastWarningTime >= 5.0) {
                LastWarningTime = CurrentTime;
                UE_LOG(LogTemp, Warning, TEXT("GPUTessellation Patch LOD: NO CAMERA FOUND!"));
            }
        }
        return;
    }

    // Check if camera moved significantly (threshold to avoid constant updates)
    const float CameraMovement = FVector::Dist(CameraPos, LastCameraPosition);
    constexpr float UpdateThreshold = 100.0f;
    // Update if camera moved more than 100 units (1 meter)

    // Scale threshold by component scale for larger objects
    const FVector Scale3D = GetComponentScale();
    const float MaxScale = FMath::Max3(
        FMath::Abs(Scale3D.X),
        FMath::Abs(Scale3D.Y),
        FMath::Abs(Scale3D.Z));
    const float ScaledThreshold = UpdateThreshold * MaxScale;

    // Check if patch configuration changed
    if (LastPatchCountX != TessellationSettings.PatchCountX ||
        LastPatchCountY != TessellationSettings.PatchCountY ||
        CameraMovement > ScaledThreshold) {
        LastPatchCountX = TessellationSettings.PatchCountX;
        LastPatchCountY = TessellationSettings.PatchCountY;
        LastCameraPosition = CameraPos;

        // Send camera position to scene proxy for patch regeneration
        SendRenderDynamicData_Concurrent();

        if (bEnableDebugLogging) {
            UE_LOG(
                LogTemp,
                Warning,
                TEXT(
                    "GPUTessellation Patch LOD: Camera moved %.1f units (threshold %.1f) - "
                    "Updating patches with camera at: %s"
                ),
                CameraMovement,
                ScaledThreshold,
                *CameraPos.ToString());
        }
    }
}

void UGPUTessellationComponent::UpdateDensityBasedLOD(const float DeltaTime) {
    // Density texture LOD would require sampling the texture on CPU
    // For now, fall back to distance-based LOD
    // Full implementation would require reading density at component location
    UpdateDistanceBasedLOD(DeltaTime);
}

float UGPUTessellationComponent::CalculateDistanceToCamera(
    const FVector& CameraPos,
    FVector& OutComponentPos) const {
    OutComponentPos = GetComponentLocation();

    if (!TessellationSettings.bUseDistanceToBounds) {
        // Simple: distance to pivot point
        return FVector::Dist(OutComponentPos, CameraPos);
    }

    // Calculate distance to closest point on plane bounds
    // The plane is in local XZ space, so we need to transform camera to local space
    const FTransform ComponentTransform = GetComponentTransform();
    const FVector LocalCameraPos = ComponentTransform.InverseTransformPosition(CameraPos);

    // Plane size (local space)
    const float HalfSizeX = TessellationSettings.PlaneSizeX * 0.5f;
    const float HalfSizeZ = TessellationSettings.PlaneSizeY * 0.5f; // PlaneSizeY is Z dimension

    // Clamp camera position to plane bounds (in local space)
    const float ClampedX = FMath::Clamp(LocalCameraPos.X, -HalfSizeX, HalfSizeX);
    const float ClampedZ = FMath::Clamp(LocalCameraPos.Z, -HalfSizeZ, HalfSizeZ);

    // For Y (height), we can use 0 (plane surface) or account for displacement
    // Using 0 for simplicity - could add max displacement height if needed
    constexpr float ClampedY = 0.0f;

    const FVector ClosestPointLocal(ClampedX, ClampedY, ClampedZ);
    const FVector ClosestPointWorld = ComponentTransform.TransformPosition(ClosestPointLocal);

    // Return distance to closest point on plane
    return FVector::Dist(ClosestPointWorld, CameraPos);
}

void UGPUTessellationComponent::SendRenderDynamicData_Concurrent() {
    // Send camera position to scene proxy for patch regeneration
    // This follows the Unreal pattern used by other dynamic mesh components

    if (SceneProxy != nullptr) {
        // Create dynamic data with current camera position
        FGPUTessellationDynamicData* DynamicData = new FGPUTessellationDynamicData();
        DynamicData->CameraPosition = LastCameraPosition;
        DynamicData->LocalToWorld = GetComponentTransform().ToMatrixWithScale();

        // Send to scene proxy on render thread
        FGPUTessellationSceneProxy* TessSceneProxy = static_cast<FGPUTessellationSceneProxy*>(
            SceneProxy);
        ENQUEUE_RENDER_COMMAND(SendGPUTessellationDynamicData)(
            [TessSceneProxy, DynamicData]([[maybe_unused]] FRHICommandListImmediate& RHICmdList) {
                TessSceneProxy->UpdateDynamicData_RenderThread(DynamicData);
            });
    }
}

int32 UGPUTessellationComponent::CalculateLODFactorScaled(
    const float Distance,
    const float ScaledMinDistance,
    const float ScaledMaxDistance
) const {
    // Distance-based falloff with min and max distance ranges
    // Distance < MinDistance: Use MaxTessellationFactor (high detail when close)
    // Distance between Min and Max: Lerp from MaxTessellationFactor to MinTessellationFactor
    // Distance > MaxDistance: Use MinTessellationFactor (low detail when far)

    // Uses scaled distances passed in (already adjusted for component scale)
    const float MinDist = ScaledMinDistance;
    float MaxDist = ScaledMaxDistance;

    // Ensure min < max
    if (MinDist >= MaxDist) {
        MaxDist = MinDist + 1000.0f; // Failsafe
    }

    // Calculate interpolation factor
    float t;
    if (Distance <= MinDist) {
        t = 0.0f; // Full max tessellation
    } else if (Distance >= MaxDist) {
        t = 1.0f; // Full min tessellation
    } else {
        // Interpolate between min and max distance
        t = (Distance - MinDist) / (MaxDist - MinDist);

        // Apply smooth curve (ease-in-out) for more natural transitions
        t = t * t * (3.0f - 2.0f * t); // Smoothstep
    }

    const float LerpedFactor = FMath::Lerp(
        static_cast<float>(TessellationSettings.MaxTessellationFactor),
        // Close distance
        static_cast<float>(TessellationSettings.MinTessellationFactor),
        // Far distance
        t
    );

    return FMath::Clamp(FMath::RoundToInt(LerpedFactor), 1, 256);
}

int32 UGPUTessellationComponent::CalculateLODFactor(const float Distance) const {
    // Legacy method - use CalculateLODFactorScaled instead for scale-aware LOD
    // Distance-based falloff with min and max distance ranges
    // Distance < MinDistance: Use MaxTessellationFactor (high detail when close)
    // Distance between Min and Max: Lerp from MaxTessellationFactor to MinTessellationFactor
    // Distance > MaxDistance: Use MinTessellationFactor (low detail when far)

    const float MinDist = TessellationSettings.MinTessellationDistance;
    float MaxDist = TessellationSettings.MaxTessellationDistance;

    // Ensure min < max
    if (MinDist >= MaxDist) {
        MaxDist = MinDist + 1000.0f; // Failsafe
    }

    // Calculate interpolation factor
    float t;
    if (Distance <= MinDist) {
        t = 0.0f; // Full max tessellation
    } else if (Distance >= MaxDist) {
        t = 1.0f; // Full min tessellation
    } else {
        // Interpolate between min and max distance
        t = (Distance - MinDist) / (MaxDist - MinDist);

        // Apply smooth curve (ease-in-out) for more natural transitions
        t = t * t * (3.0f - 2.0f * t); // Smoothstep
    }

    const float LerpedFactor = FMath::Lerp(
        static_cast<float>(TessellationSettings.MaxTessellationFactor),
        // Close distance
        static_cast<float>(TessellationSettings.MinTessellationFactor),
        // Far distance
        t
    );

    return FMath::Clamp(FMath::RoundToInt(LerpedFactor), 1, 256);
}
