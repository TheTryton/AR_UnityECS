using Unity.Entities;
using Unity.Mathematics;

public struct Mass
    : IComponentData
{
    public float mass;
}

public struct Center
    : IComponentData
{
    public float3 center;
}

public struct Velocity
    : IComponentData
{
    public float3 velocity;
}

public struct Force
    : IComponentData
{
    public float3 force;
}

public struct ForceSymmetrical
    : IComponentData
{
    public float3 force;
}
