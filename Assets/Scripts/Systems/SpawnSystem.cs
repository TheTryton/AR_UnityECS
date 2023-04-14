using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using Unity.Transforms;

public struct SpawnerID
    : ISharedComponentData
{
    public int id;
}

[BurstCompile]
[UpdateBefore(typeof(PlanetSystem))]
public partial struct SpawnSystem
    : ISystem
{
    private EntityQuery spawnerQuery;
    private EntityQuery planetQuery;
    private ComponentLookup<LocalTransform> localToWorldLookup;
    private ComponentLookup<Velocity> velocityLookup;
    private ComponentLookup<Center> centerLookup;
    private ComponentLookup<Mass> massLookup;
    private ComponentLookup<Force> forceLookup;
    private ComponentLookup<ForceSymmetrical> forceSymmetricalLookup;

    public void OnCreate(ref SystemState state)
    {
        spawnerQuery = SystemAPI.QueryBuilder().WithAll<PlanetSpawner, LocalTransform>().Build();
        planetQuery = SystemAPI.QueryBuilder().WithAll<LocalTransform, Velocity, Mass, SpawnerID>().Build();
        localToWorldLookup = SystemAPI.GetComponentLookup<LocalTransform>();
        velocityLookup = SystemAPI.GetComponentLookup<Velocity>();
        centerLookup = SystemAPI.GetComponentLookup<Center>();
        massLookup = SystemAPI.GetComponentLookup<Mass>();
        forceLookup = SystemAPI.GetComponentLookup<Force>();
        forceSymmetricalLookup = SystemAPI.GetComponentLookup<ForceSymmetrical>();

        state.RequireForUpdate(spawnerQuery);
    }

    public void OnDestroy(ref SystemState state)
    {
    }

    public void OnUpdate(ref SystemState state)
    {
        var world = state.WorldUnmanaged;

        var planetSpawners = spawnerQuery.ToComponentDataArray<PlanetSpawner>(Allocator.Temp);

        using (var markedForDestruction = new NativeList<Entity>(128, world.UpdateAllocator.Handle))
        {
            for (int i = 0; i < planetSpawners.Length; i++)
            {
                planetQuery.SetSharedComponentFilter(new SpawnerID { id = planetSpawners[i].id });

                var entities = planetQuery.ToEntityArray(Allocator.Temp);
            
                    for (int ei = 0; ei < entities.Length; ei++)
                    {
                        var localTransform = state.EntityManager.GetComponentData<LocalTransform>(entities[ei]);
                        var center = state.EntityManager.GetComponentData<Center>(entities[ei]);
                        if (length(localTransform.Position - center.center) > planetSpawners[i].outerRadius)
                            markedForDestruction.Add(entities[ei]);
                    }
            }
            state.EntityManager.DestroyEntity(markedForDestruction.AsArray());
        }

        localToWorldLookup.Update(ref state);
        velocityLookup.Update(ref state);
        centerLookup.Update(ref state);
        massLookup.Update(ref state);
        forceLookup.Update(ref state);

        var localToWorlds = spawnerQuery.ToComponentDataArray<LocalTransform>(Allocator.Temp);
        
        for (int i = 0; i < planetSpawners.Length; i++)
        {
            planetQuery.SetSharedComponentFilter(new SpawnerID { id = planetSpawners[i].id });

            var planetsCount = planetQuery.CalculateEntityCount();

            if(planetsCount <= planetSpawners[i].count)
            {
                var planetEntities = CollectionHelper.CreateNativeArray<Entity, RewindableAllocator>(planetSpawners[i].count - planetsCount, ref world.UpdateAllocator);
                state.EntityManager.Instantiate(planetSpawners[i].prefab, planetEntities);
                state.EntityManager.AddComponent(planetEntities, new ComponentTypeSet(
                    new ComponentType[]{
                        ComponentType.ReadWrite<LocalTransform>(),
                        ComponentType.ReadWrite<Velocity>(),
                        ComponentType.ReadWrite<Center>(),
                        ComponentType.ReadWrite<Mass>(),
                        ComponentType.ReadWrite<Force>(),
                        ComponentType.ReadWrite<ForceSymmetrical>()
                    }
                    ));
                state.EntityManager.AddSharedComponent(planetEntities, new SpawnerID { id = planetSpawners[i].id });

                var job = new SetPlanetLocalToWorld
                {
                    localToWorldFromEntity = localToWorldLookup,
                    velocityFromEntity = velocityLookup,
                    centerFromEntity = centerLookup,
                    massFromEntity = massLookup,
                    forceFromEntity = forceLookup,
                    forceSymmetricalFromEntity = forceSymmetricalLookup,
                    entities = planetEntities,
                    center = localToWorlds[i].Position,
                    minInitialVelocity = planetSpawners[i].minInitialVelocity,
                    maxInitialVelocity = planetSpawners[i].maxInitialVelocity,
                    minInitialMass = planetSpawners[i].minInitialMass,
                    maxInitialMass = planetSpawners[i].maxInitialMass,
                    radius = planetSpawners[i].radius
                };
                state.Dependency = job.Schedule(planetSpawners[i].count - planetsCount, 128, state.Dependency);
            }
        }

        state.CompleteDependency();
    }

    [BurstCompile]
    public struct SetPlanetLocalToWorld
        : IJobParallelFor
    {
        [NativeDisableContainerSafetyRestriction]
        [NativeDisableParallelForRestriction]
        public ComponentLookup<LocalTransform> localToWorldFromEntity;
        [NativeDisableContainerSafetyRestriction]
        [NativeDisableParallelForRestriction]
        public ComponentLookup<Velocity> velocityFromEntity;
        [NativeDisableContainerSafetyRestriction]
        [NativeDisableParallelForRestriction]
        public ComponentLookup<Center> centerFromEntity;
        [NativeDisableContainerSafetyRestriction]
        [NativeDisableParallelForRestriction]
        public ComponentLookup<Mass> massFromEntity;
        [NativeDisableContainerSafetyRestriction]
        [NativeDisableParallelForRestriction]
        public ComponentLookup<Force> forceFromEntity;
        [NativeDisableContainerSafetyRestriction]
        [NativeDisableParallelForRestriction]
        public ComponentLookup<ForceSymmetrical> forceSymmetricalFromEntity;

        public NativeArray<Entity> entities;
        public float3 center;
        public float minInitialVelocity;
        public float maxInitialVelocity;
        public float minInitialMass;
        public float maxInitialMass;
        public float radius;

        public void Execute(int i)
        {
            var entity = entities[i];
            var random = new Random(((uint)(entity.Index + i + 1) * 0x9F6ABC1));
            var mass = random.NextFloat(minInitialMass, maxInitialMass);
            var r = random.NextFloat(0, radius);
            var a = random.NextFloat(0, PI);
            var b = random.NextFloat(0, 2 * PI);
            var pos = float3(
                r * sin(a) * cos(b),
                r * sin(a) * sin(b),
                r * cos(a)
                ) + center;
            // position
            {
                var localToWorld = LocalTransform.FromPositionRotationScale(
                    pos,
                    Unity.Mathematics.quaternion.identity,
                    mass
                    );
                localToWorldFromEntity[entity] = localToWorld;
            }
            // velocity
            {
                var vel = random.NextFloat3Direction();
                var velMag = random.NextFloat(minInitialVelocity, maxInitialVelocity);
                velocityFromEntity[entity] = new Velocity
                {
                    velocity = vel * velMag
                };
            }
            // center
            {
                centerFromEntity[entity] = new Center
                {
                    center = center
                };
            }
            // mass
            {
                massFromEntity[entity] = new Mass
                {
                    mass = mass
                };
            }
            // force
            {
                forceFromEntity[entity] = new Force
                {
                    force = Unity.Mathematics.float3.zero
                };
            }
            // force symmetrical
            {
                forceSymmetricalFromEntity[entity] = new ForceSymmetrical
                {
                    force = Unity.Mathematics.float3.zero
                };
            }
        }
    }
}
