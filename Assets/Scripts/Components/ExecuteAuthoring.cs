using UnityEngine;
using Unity.Entities;

[AddComponentMenu("Execute")]
public class ExecuteAuthoring : MonoBehaviour
{
    public class Baker : Baker<ExecuteAuthoring>
    {
        public override void Bake(ExecuteAuthoring authoring)
        {
            AddComponent<Execute>();
        }
    }
}

public struct Execute : IComponentData
{
}
