using System.Collections;
using System.Collections.Generic;
using Unity.Entities;
using UnityEngine;

[AddComponentMenu("Planet Spawner")]
public class PlanetSpawnerAuthoring
    : MonoBehaviour
{
    public GameObject prefab;
    public float radius;
    public float outerRadius;
    public float minInitialVelocity;
    public float maxInitialVelocity;
    public float minInitialMass;
    public float maxInitialMass;
    public int count;

    public class PlanetSpawnerAuthoringBaker : Baker<PlanetSpawnerAuthoring>
    {
        public override void Bake(PlanetSpawnerAuthoring authoring)
        {
            AddComponent(new PlanetSpawner
            {
                prefab = GetEntity(authoring.prefab),
                radius = authoring.radius,
                outerRadius = authoring.outerRadius,
                minInitialVelocity = authoring.minInitialVelocity,
                maxInitialVelocity = authoring.maxInitialVelocity,
                minInitialMass = authoring.minInitialMass,
                maxInitialMass = authoring.maxInitialMass,
                count = authoring.count,
                id = authoring.GetInstanceID()
            });
        }
    }

    public void OnDrawGizmos()
    {
        Gizmos.color = new Color(0.0f, 1.0f, 0.0f, 0.5f);
        Gizmos.DrawSphere(transform.position, radius);
        Gizmos.color = new Color(1.0f, 0.0f, 0.0f, 0.5f);
        Gizmos.DrawSphere(transform.position, outerRadius);
    }
}

public struct PlanetSpawner
    : IComponentData
{
    public int id;
    public Entity prefab;
    public float radius;
    public float outerRadius;
    public float minInitialVelocity;
    public float maxInitialVelocity;
    public float minInitialMass;
    public float maxInitialMass;
    public int count;
}
