using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Transforms;

public struct SimulationProperties
    : IComponentData
{
    public float gravitationalConstant;
}

[BurstCompile]
[UpdateAfter(typeof(SpawnSystem))]
public partial struct PlanetSystem : ISystem
{
    private EntityQuery planetsQuery;
    private Entity simulationPropertiesEntity;

    [BurstCompile]
    public void OnCreate(ref SystemState state)
    {
        planetsQuery = SystemAPI.QueryBuilder().WithAll<LocalTransform, Velocity, Mass, Force>().Build();

        simulationPropertiesEntity = state.EntityManager.CreateSingleton<SimulationProperties>(
            new SimulationProperties { gravitationalConstant = 1f},
            "simulationProperties"
            );
        state.RequireForUpdate<Execute>();
        state.RequireForUpdate(planetsQuery);
    }

    [BurstCompile]
    public void OnDestroy(ref SystemState state)
    {
    }

    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        state.Dependency.Complete();

        var simulationProperties = state.EntityManager.GetComponentData<SimulationProperties>(simulationPropertiesEntity);

        var planetPositionHandle = SystemAPI.GetComponentTypeHandle<LocalTransform>();
        var planetVelocityHandle = SystemAPI.GetComponentTypeHandle<Velocity>();
        var planetMassHandle = SystemAPI.GetComponentTypeHandle<Mass>();
        var planetForceHandle = SystemAPI.GetComponentTypeHandle<Force>();

        planetPositionHandle.Update(ref state);
        planetVelocityHandle.Update(ref state);
        planetMassHandle.Update(ref state);
        planetForceHandle.Update(ref state);

        var world = state.WorldUnmanaged;

        var octreeStorage = new NativeList<Octree<BarnesHut.NodeData, BarnesHut.InternalNodeData>.Node>(world.UpdateAllocator.Handle);
        var octreeDataStorage = new NativeList<BarnesHut.NodeData>(world.UpdateAllocator.Handle);
        var octreeDataPositionsStorage = new NativeList<float3>(world.UpdateAllocator.Handle);
        var octreeInternalDataStorage = new NativeList<BarnesHut.InternalNodeData>(world.UpdateAllocator.Handle);
        var archetypeChunks = planetsQuery.ToArchetypeChunkArray(Allocator.TempJob);
        var baseEntityIndexArray = planetsQuery.CalculateBaseEntityIndexArray(Allocator.TempJob);
        var entitiesCount = planetsQuery.CalculateEntityCount();
        var positions = new NativeArray<float3>(entitiesCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var masses = new NativeArray<float>(entitiesCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        var octreeConstructionJobData = new BarnesHut.BarnesHutOctreeConstruction()
        {
            octreeNodesStorage = octreeStorage,
            octreeNodesDataPositionsStorage = octreeDataPositionsStorage,
            octreeNodesDataStorage = octreeDataStorage,
            octreeNodesInternalDataStorage = octreeInternalDataStorage,
            archetypeChunks = archetypeChunks,
            baseEntityIndexArray = baseEntityIndexArray,
            positions = positions,
            masses = masses,
            localTransformHandle = planetPositionHandle,
            massHandle = planetMassHandle,
            center = float3.zero,
            extent = 1500.0f,
        };
        state.Dependency = octreeConstructionJobData.Schedule(state.Dependency);
        state.Dependency.Complete();

        var evaluateJobData = new BarnesHut.BarnesHutEvaluate
        {
            octreeNodesStorage = octreeStorage.AsArray(),
            octreeNodesDataPositionsStorage = octreeDataPositionsStorage.AsArray(),
            octreeNodesDataStorage = octreeDataStorage.AsArray(),
            octreeNodesInternalDataStorage = octreeInternalDataStorage.AsArray(),
            planetPositionHandle = planetPositionHandle,
            planetMassHandle = planetMassHandle,
            planetVelocityHandle = planetVelocityHandle,
            planetForceHandle = planetForceHandle,
            threshold = 0.5f,
            gravitationalConstant = simulationProperties.gravitationalConstant,
        };
        state.Dependency = evaluateJobData.ScheduleParallel(planetsQuery, state.Dependency);

        var finalizeJobData = new BarnesHut.BarnesHutFinalize
        {
            planetPositionHandle = planetPositionHandle,
            planetVelocityHandle = planetVelocityHandle,
            planetMassHandle = planetMassHandle,
            planetForceHandle = planetForceHandle,
            deltaTime = SystemAPI.Time.DeltaTime,
            deltaTime2 = SystemAPI.Time.DeltaTime * SystemAPI.Time.DeltaTime,
        };
        state.Dependency = finalizeJobData.ScheduleParallel(planetsQuery, state.Dependency);
        state.CompleteDependency();

        octreeStorage.Dispose();
        octreeDataStorage.Dispose();
        octreeDataPositionsStorage.Dispose();
        octreeInternalDataStorage.Dispose();
        archetypeChunks.Dispose();
        baseEntityIndexArray.Dispose();
        positions.Dispose();
        masses.Dispose();
    }
}