using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;
using UnityEngine;

public static class Octree<T, IT>
        where T : unmanaged
        where IT : unmanaged
{
    public struct Node
    {
        public const int maxDataPerNode = 16;
        public static readonly int3 childOffset = math.int3(1, 2, 4);

        // Spatial Data

        public readonly float3 center;
        public readonly float extent;

        // Data Access Pointers

        private int firstNodeDataIndex;
        private int nodeDataLength;

        // Hierachy Access Pointers

        private int indexInStorage;
        private int firstChildIndexInStorage;

        private Node(
             float3 center, float extent,
             int indexInStorage,
             int firstNodeDataIndex
             )
        {
            this.center = center;
            this.extent = extent;

            this.firstNodeDataIndex = firstNodeDataIndex;
            this.nodeDataLength = 0;

            this.indexInStorage = indexInStorage;
            this.firstChildIndexInStorage = -1;
        }

        public bool IsLeaf
            => firstChildIndexInStorage == -1;

        public bool IsEmpty
            => nodeDataLength == 0;

        public bool IsInternal
            => firstChildIndexInStorage != -1;

        public int DataElementsCount
            => nodeDataLength;

        public int IndexInStorage
            => indexInStorage;

        public int FirstChildIndexInStorage
            => firstChildIndexInStorage;

        [BurstCompile]
        public NativeSlice<Node> GetChildren(NativeSlice<Node> allNodes)
            => allNodes.Slice(firstChildIndexInStorage, 8);

        [BurstCompile]
        public NativeSlice<float3> GetDataPositions(NativeSlice<float3> allNodesDataPositions)
            => allNodesDataPositions.Slice(firstNodeDataIndex, nodeDataLength);

        [BurstCompile]
        public NativeSlice<T> GetData(NativeSlice<T> allNodesData)
            => allNodesData.Slice(firstNodeDataIndex, nodeDataLength);

        [BurstCompile]
        public NativeSlice<IT> GetInternalData(NativeSlice<IT> allNodesInternalData)
            => allNodesInternalData.Slice(firstNodeDataIndex, nodeDataLength);

        [BurstCompile]
        private int GetChildIndexOffset(float3 position)
        {
            bool3 relativePosition = position > center;
            return math.csum(math.int3(relativePosition) * childOffset);
        }

        [BurstCompile]
        public int GetChildIndex(float3 position)
        {
            bool3 relativePosition = position > center;
            int childIndex = math.csum(math.int3(relativePosition) * childOffset);
            return firstChildIndexInStorage + childOffset[childIndex];
        }

        [BurstCompile]
        public bool Contains(float3 position)
        {
            var low = center - extent;
            var high = center + extent;
            return math.all(low < position) && math.all(position <= high);
        }

        [BurstCompile]
        public void InsertData(
            float3 position, T data,
            NativeList<Node> allNodes,
            NativeList<T> allNodesData,
            NativeList<IT> allNodesInternalData,
            NativeList<float3> allNodesDataPositions,
            int internalNodeDataSize = 0
            )
        {
#if DEBUG
            if (!Contains(position))
                throw new System.ArgumentOutOfRangeException(nameof(position), "Position is out of octree node bounds!");
#endif

            // if INTERNAL node
            if (IsInternal)
            {
                var childIndex = firstChildIndexInStorage + GetChildIndexOffset(position);
                var child = allNodes[childIndex];

                child.InsertData(
                            position, data,
                            allNodes, allNodesData, allNodesInternalData, allNodesDataPositions,
                            internalNodeDataSize
                            ); // commits automatically
            }
            // if data element fits into this node (LEAF)
            else if (nodeDataLength < maxDataPerNode)
            {
                // add it
                allNodesData[firstNodeDataIndex + nodeDataLength] = data;
                allNodesDataPositions[firstNodeDataIndex + nodeDataLength] = position;
                // and update this node state
                nodeDataLength += 1;
                allNodes[indexInStorage] = this;
            }
            // else (SPLIT LEAF) - we need to do a subdivision
            else
            {
                // CREATE SUBNODES

                var newNodesStorageIndexStart = allNodes.Length;

                {
                    // 0 means -x/-y/-z
                    // 1 means x/y/z

                    float newExtent = extent / 2;
                    float3 poffset = math.float3(newExtent);
                    float3 noffset = -poffset;

                    float3 center000 = center + math.select(noffset, poffset, math.bool3(false, false, false));
                    float3 center100 = center + math.select(noffset, poffset, math.bool3(true, false, false));
                    float3 center010 = center + math.select(noffset, poffset, math.bool3(false, true, false));
                    float3 center110 = center + math.select(noffset, poffset, math.bool3(true, true, false));
                    float3 center001 = center + math.select(noffset, poffset, math.bool3(false, false, true));
                    float3 center101 = center + math.select(noffset, poffset, math.bool3(true, false, true));
                    float3 center011 = center + math.select(noffset, poffset, math.bool3(false, true, true));
                    float3 center111 = center + math.select(noffset, poffset, math.bool3(true, true, true));

                    var newNodesDataStorageIndexStart = allNodesData.Length;

                    allNodes.ResizeUninitialized(allNodes.Length + 8);
                    allNodesData.ResizeUninitialized(allNodesData.Length + 7 * maxDataPerNode); // 7 - we are reusing this node data slice
                    allNodesDataPositions.ResizeUninitialized(allNodesDataPositions.Length + 7 * maxDataPerNode); // 7 - we are reusing this node data slice

                    var newNodesSlice = allNodes.AsArray().Slice(newNodesStorageIndexStart, 8);

                    newNodesSlice[0] = new Node(center000, newExtent, newNodesStorageIndexStart, firstNodeDataIndex);
                    newNodesSlice[1] = new Node(center100, newExtent, newNodesStorageIndexStart + 1, newNodesDataStorageIndexStart);
                    newNodesSlice[2] = new Node(center010, newExtent, newNodesStorageIndexStart + 2, newNodesDataStorageIndexStart + 1 * maxDataPerNode);
                    newNodesSlice[3] = new Node(center110, newExtent, newNodesStorageIndexStart + 3, newNodesDataStorageIndexStart + 2 * maxDataPerNode);
                    newNodesSlice[4] = new Node(center001, newExtent, newNodesStorageIndexStart + 4, newNodesDataStorageIndexStart + 3 * maxDataPerNode);
                    newNodesSlice[5] = new Node(center101, newExtent, newNodesStorageIndexStart + 5, newNodesDataStorageIndexStart + 4 * maxDataPerNode);
                    newNodesSlice[6] = new Node(center011, newExtent, newNodesStorageIndexStart + 6, newNodesDataStorageIndexStart + 5 * maxDataPerNode);
                    newNodesSlice[7] = new Node(center111, newExtent, newNodesStorageIndexStart + 7, newNodesDataStorageIndexStart + 6 * maxDataPerNode);
                }
                // ASSIGN CHILDREN STORAGE INDEX
                {
                    firstChildIndexInStorage = newNodesStorageIndexStart;
                }
                // TRANSFER DATA
                {
                    // move current node data to temporary slice
                    // as we are reusing memory allocated for this node in first child
                    var transferTemporaryDataSlice = allNodesData.AsArray().Slice(0, maxDataPerNode);
                    var transferTemporaryDataPositionsSlice = allNodesDataPositions.AsArray().Slice(0, maxDataPerNode);

                    var transferedDataSlice = allNodesData.AsArray().Slice(firstNodeDataIndex, maxDataPerNode);
                    var transferedDataPositionsSlice = allNodesDataPositions.AsArray().Slice(firstNodeDataIndex, maxDataPerNode);

                    transferTemporaryDataSlice.CopyFrom(transferedDataSlice);
                    transferTemporaryDataPositionsSlice.CopyFrom(transferedDataPositionsSlice);

                    // mark this node as INTERNAL node

                    nodeDataLength = -1;

                    // transfer each data element to subnodes

                    for (int i = 0; i < transferTemporaryDataSlice.Length; i++)
                    {
                        var transferedData = transferTemporaryDataSlice[i];
                        var transferedDataPosition = transferTemporaryDataPositionsSlice[i];

                        var childIndex = firstChildIndexInStorage + GetChildIndexOffset(transferedDataPosition);
                        var child = allNodes[childIndex];

                        child.InsertData(
                            transferedDataPosition, transferedData,
                            allNodes, allNodesData, allNodesInternalData, allNodesDataPositions
                            ); // commits automatically
                    }
                }
                // CREATE INTERNAL DATA
                {
                    if (internalNodeDataSize == 0)
                    {
                        firstNodeDataIndex = -1;
                        nodeDataLength = 0;
                    }
                    else
                    {
                        firstNodeDataIndex = allNodesInternalData.Length;
                        allNodesInternalData.ResizeUninitialized(allNodesInternalData.Length + internalNodeDataSize);
                        nodeDataLength = internalNodeDataSize;
                    }
                }
                // COMMIT ALL NODE CHANGES
                {
                    allNodes[indexInStorage] = this;
                }
            }
        }

        public static Node CreateRoot(
            float3 center, float extent,
            NativeList<Node> allNodes,
            NativeList<T> allNodesData,
            NativeList<float3> allNodesPositionData
            )
        {
            allNodesData.ResizeUninitialized(maxDataPerNode * 2);
            allNodesPositionData.ResizeUninitialized(maxDataPerNode * 2);

            var rootNode = new Node(
                center, extent,
                0, maxDataPerNode
                );
            allNodes.Add(rootNode);
            return rootNode;
        }
    }

    private unsafe struct Stack
    {
        private const int MAX_STACK_SIZE = 1024;
        private fixed int stack[MAX_STACK_SIZE];
        private int stackPointer;

        [BurstCompile]
        public void Push(int element)
        {
#if DEBUG
            if (stackPointer == MAX_STACK_SIZE)
                throw new System.StackOverflowException();
#endif
            stack[stackPointer++] = element;
        }

        public bool Empty
            => stackPointer == 0;

        [BurstCompile]
        public int Pop()
        {
#if DEBUG
            if (stackPointer == 0)
                throw new System.StackOverflowException();
#endif
            return stack[--stackPointer];
        }
    }

    private static int CalculateDepth(Node currentNode, NativeSlice<Node> allNodes)
    {
        if (currentNode.IsInternal)
        {
            int maxDepth = 0;
            foreach (var child in currentNode.GetChildren(allNodes))
                maxDepth = math.max(maxDepth, CalculateDepth(child, allNodes));
            return maxDepth;
        }
        else
        {
            return 0;
        }
    }

    public static int CalculateDepth(NativeSlice<Node> allNodes)
        => CalculateDepth(allNodes[0], allNodes);

    private static int CalculateDataCount(Node currentNode, NativeSlice<Node> allNodes)
    {
        if (currentNode.IsInternal)
        {
            int maxDepth = 0;
            foreach (var child in currentNode.GetChildren(allNodes))
                maxDepth = math.max(maxDepth, CalculateDepth(child, allNodes));
            return maxDepth;
        }
        else
        {
            return currentNode.DataElementsCount;
        }
    }

    public static int CalculateDataCount(NativeSlice<Node> allNodes)
        => CalculateDataCount(allNodes[0], allNodes);
}

public static class BarnesHut
{
    public struct NodeData
    {
        public float mass;
        public float3 centerOfMass;
    }

    public struct InternalNodeData
    {
        public float3 centerOfMass;
        public float mass;
    }

    private static void CalculateCentersOfMass(
        ref Octree<NodeData, InternalNodeData>.Node currentNode,
        NativeSlice<Octree<NodeData, InternalNodeData>.Node> allNodes, NativeSlice<float3> allNodesPositionData,
        NativeSlice<NodeData> allNodesData, NativeSlice<InternalNodeData> allNodesInternalData
        )
    {
        if (currentNode.IsLeaf)
        {
            var data = currentNode.GetData(allNodesData);
            var dataPositions = currentNode.GetDataPositions(allNodesPositionData);

            float totalMass = 0;
            float3 centerOfMass = float3.zero;

            for (int di = 0; di < dataPositions.Length; di++)
            {
                var dataMass = data[di].mass;
                var dataCenterOfMass = dataPositions[di];
                totalMass += dataMass;
                centerOfMass += dataCenterOfMass * dataMass;
            }

            centerOfMass /= totalMass;

            data[0] = new NodeData
            {
                mass = data[0].mass,
                centerOfMass = centerOfMass,
            };
        }
        else
        {
            float totalMass = 0;
            float3 centerOfMass = float3.zero;

            var children = currentNode.GetChildren(allNodes);
            for (int i = 0; i < children.Length; i++)
            {
                var child = children[i];

                if (child.IsLeaf)
                {
                    var childData = child.GetData(allNodesData);
                    var childDataPositions = child.GetDataPositions(allNodesPositionData);
                    for (int di = 0; di < childDataPositions.Length; di++)
                    {
                        var childMass = childData[di].mass;
                        var childCenterOfMass = childDataPositions[di];
                        totalMass += childMass;
                        centerOfMass += childCenterOfMass * childMass;
                    }
                }
                else
                {
                    CalculateCentersOfMass(ref child, allNodes, allNodesPositionData, allNodesData, allNodesInternalData);
                    var childMass = child.GetInternalData(allNodesInternalData)[0].mass;
                    var childCenterOfMass = child.GetInternalData(allNodesInternalData)[0].centerOfMass;
                    totalMass += childMass;
                    centerOfMass += childCenterOfMass * childMass;
                }

            }
            centerOfMass /= totalMass;

            var currentNodeInternalData = currentNode.GetInternalData(allNodesInternalData);
            currentNodeInternalData[0] = new InternalNodeData
            {
                mass = totalMass,
                centerOfMass = centerOfMass,
            };
        }
    }

    private static float3 CalculateForce(
            float mass0,
            float mass1,
            float3 delta, float distance
            )
    {
        var distance3 = distance * distance * distance;
        var accelerationNoMass = delta / distance3;
        return mass0 * mass1 * accelerationNoMass;
    }

    private static float3 CalculateTotalForce(
        float3 massPosition, float mass,
        Octree<NodeData, InternalNodeData>.Node currentNode,
        float3 currentNodeCenterOfMass, float currentNodeTotalMass,
        NativeSlice<Octree<NodeData, InternalNodeData>.Node> allNodes,
        NativeSlice<float3> allNodesDataPositions,
        NativeSlice<NodeData> allNodesData,
        NativeSlice<InternalNodeData> allNodesInternalData,
        float threshold
        )
    {
        var delta = currentNodeCenterOfMass - massPosition;
        var distance = math.length(delta);
        var distanceIsZero = math.abs(distance) < float.Epsilon;
        if (distanceIsZero)
            return float3.zero;

        var sd = currentNode.extent / distance;

        if (sd < threshold)
        {
            return CalculateForce(mass, currentNodeTotalMass, delta, distance);
        }
        else
        {
            if (currentNode.IsEmpty)
                return float3.zero;

            float3 totalForce = float3.zero;

            foreach(var child in currentNode.GetChildren(allNodes))
            {
                if (child.IsInternal)
                {
                    float childMass = child.GetInternalData(allNodesInternalData)[0].mass;
                    float3 childCenterOfMass = child.GetInternalData(allNodesInternalData)[0].centerOfMass;

                    totalForce += CalculateTotalForce(
                        massPosition, mass, child, childCenterOfMass, childMass,
                        allNodes, allNodesDataPositions, allNodesData, allNodesInternalData,
                        threshold
                        );
                }
                else
                {
                    var childElementData = child.GetData(allNodesData);
                    var childElementPositions = child.GetDataPositions(allNodesDataPositions);
                    for (int i = 0; i < childElementData.Length; i++)
                    {
                        var childDelta = childElementPositions[i] - massPosition;
                        var childDistance = math.length(childDelta);
                        var childDistanceIsZero = math.abs(childDistance) < float.Epsilon;
                        if (childDistanceIsZero)
                            continue;

                        totalForce += CalculateForce(mass, childElementData[i].mass, childDelta, childDistance);
                    }
                }
            }

            return totalForce;
        }
    }

    private unsafe struct Stack
    {
        private const int MAX_STACK_SIZE = 1024;
        private fixed int stack[MAX_STACK_SIZE];
        private int stackPointer;

        [BurstCompile]
        public void Push(int element)
        {
#if DEBUG
            if (stackPointer == MAX_STACK_SIZE)
                throw new System.StackOverflowException();
#endif
            stack[stackPointer++] = element;
        }

        public bool Empty
            => stackPointer == 0;

        [BurstCompile]
        public int Pop()
        {
#if DEBUG
            if (stackPointer == 0)
                throw new System.StackOverflowException();
#endif
            return stack[--stackPointer];
        }
    }

    public unsafe static float3 CalculateTotalForce(
        float3 massPosition, float mass,
        NativeSlice<Octree<NodeData, InternalNodeData>.Node> allNodes,
        NativeSlice<float3> allNodesDataPositions,
        NativeSlice<NodeData> allNodesData,
        NativeSlice<InternalNodeData> allNodesInternalData,
        float threshold, float gravitationalConstant
        )
    {
        float3 totalForce = float3.zero;

        unsafe
        {
            Stack stack = new Stack();

            stack.Push(0);

            while (!stack.Empty)
            {
                int currentNodeIndex = stack.Pop();

                var currentNode = allNodes[currentNodeIndex];

                float currentNodeTotalMass;
                float3 currentNodeCenterOfMass;
                if (currentNode.IsInternal)
                {
                    var internalData = currentNode.GetInternalData(allNodesInternalData);
                    currentNodeTotalMass = internalData[0].mass;
                    currentNodeCenterOfMass = internalData[0].centerOfMass;
                }
                else if (currentNode.IsEmpty)
                    continue;
                else
                {
                    var data = currentNode.GetData(allNodesData);
                    currentNodeTotalMass = data[0].mass;
                    currentNodeCenterOfMass = data[0].centerOfMass;
                }

                var delta = currentNodeCenterOfMass - massPosition;
                var distance = math.length(delta);
                var distanceIsZero = math.abs(distance) < float.Epsilon;
                if (distanceIsZero)
                    continue;

                var sd = currentNode.extent / distance;

                if (sd < threshold)
                {
                    totalForce += CalculateForce(mass, currentNodeTotalMass, delta, distance);
                }
                else
                {
                    if (currentNode.IsInternal)
                    {
                        for (int i = 0; i < 8; i++)
                            stack.Push(currentNode.FirstChildIndexInStorage + i);
                    }
                    else
                    {
                        var elementData = currentNode.GetData(allNodesData);
                        var elementPositions = currentNode.GetDataPositions(allNodesDataPositions);
                        for (int i = 0; i < elementData.Length; i++)
                        {
                            var elementDelta = elementPositions[i] - massPosition;
                            var elementDistance = math.length(elementDelta);
                            var elementDistanceIsZero = math.abs(elementDistance) < float.Epsilon;
                            if (elementDistanceIsZero)
                                continue;

                            totalForce += CalculateForce(mass, elementData[i].mass, elementDelta, elementDistance);
                        }
                    }
                }
            }
        }

        return gravitationalConstant * totalForce;
    }

    public static void Create(
           NativeList<Octree<NodeData, InternalNodeData>.Node> allNodes, NativeList<float3> allNodesPositionData,
           NativeList<NodeData> allNodesData, NativeList<InternalNodeData> allNodesInternalData,
           NativeSlice<float3> positions, NativeSlice<float> masses,
           float3 topLevelCenter, float topLevelExtent
           )
    {
#if DEBUG
        if (positions.Length != masses.Length)
            throw new System.ArgumentException("Positions and masses arrays must be of the same size!", $"{nameof(positions)}, {nameof(masses)}");
        if (positions.Length == 0)
            throw new System.ArgumentException("Positions and masses arrays contain at least 1 element!", $"{nameof(positions)}, {nameof(masses)}");
#endif
        Octree<NodeData, InternalNodeData>.Node.CreateRoot(
            topLevelCenter, topLevelExtent,
            allNodes, allNodesData, allNodesPositionData
            );

        var low = topLevelCenter - topLevelExtent;
        var high = topLevelCenter + topLevelExtent;

        for (int i = 0; i < positions.Length; i++)
        {
            var position = positions[i];
            var mass = masses[i];

            if (math.any(math.isnan(position)) || math.any(low >= position) || math.any(high < position))
                continue;

            allNodes[0].InsertData(
                position, new NodeData { mass = mass },
                allNodes, allNodesData, allNodesInternalData,
                allNodesPositionData, 1
                );
        }

        var rootNode = allNodes[0];
        CalculateCentersOfMass(ref rootNode, allNodes.AsArray(), allNodesPositionData.AsArray(), allNodesData.AsArray(), allNodesInternalData.AsArray());
    }

    [BurstCompile]
    public struct BarnesHutOctreeConstruction
        : IJob
    {
        public NativeList<Octree<NodeData, InternalNodeData>.Node> octreeNodesStorage;
        public NativeList<float3> octreeNodesDataPositionsStorage;
        public NativeList<NodeData> octreeNodesDataStorage;
        public NativeList<InternalNodeData> octreeNodesInternalDataStorage;

        public NativeArray<ArchetypeChunk> archetypeChunks;
        public NativeArray<int> baseEntityIndexArray;
        public NativeArray<float3> positions;
        public NativeArray<float> masses;
        public ComponentTypeHandle<LocalTransform> localTransformHandle;
        public ComponentTypeHandle<Mass> massHandle;
        public float3 center;
        public float extent;

        [BurstCompile]
        public void Execute()
        {
            for (int ai = 0; ai < archetypeChunks.Length; ai++)
            {
                var chunk = archetypeChunks[ai];
                var chunkPositions = chunk.GetNativeArray(ref localTransformHandle);
                var chunkMasses = chunk.GetNativeArray(ref massHandle);

                for (int ci = 0, chunkSize = chunk.Count; ci < chunkSize; ci++)
                {
                    int entityIndexInQuery = baseEntityIndexArray[ai] + ci;
                    positions[entityIndexInQuery] = chunkPositions[ci].Position;
                    masses[entityIndexInQuery] = chunkMasses[ci].mass;
                }
            }

            Create(
                octreeNodesStorage,
                octreeNodesDataPositionsStorage, octreeNodesDataStorage,
                octreeNodesInternalDataStorage,
                positions, masses,
                center, extent
                );
        }
    }

    [BurstCompile]
    public struct BarnesHutEvaluate
        : IJobChunk
    {
        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<Octree<NodeData, InternalNodeData>.Node> octreeNodesStorage;
        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<float3> octreeNodesDataPositionsStorage;
        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<NodeData> octreeNodesDataStorage;
        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<InternalNodeData> octreeNodesInternalDataStorage;

        public ComponentTypeHandle<LocalTransform> planetPositionHandle;
        public ComponentTypeHandle<Mass> planetMassHandle;
        public ComponentTypeHandle<Velocity> planetVelocityHandle;
        public ComponentTypeHandle<Force> planetForceHandle;
        public float threshold;
        public float gravitationalConstant;

        public void Execute(in ArchetypeChunk chunk, int unfilteredChunkIndex, bool useEnabledMask, in v128 chunkEnabledMask)
        {
            var planetPositions = chunk.GetNativeArray(ref planetPositionHandle);
            var planetMasses = chunk.GetNativeArray(ref planetMassHandle);
            var planetForces = chunk.GetNativeArray(ref planetForceHandle);

            for (int i = 0, chunkSize = chunk.Count; i < chunkSize; i++)
            {
                var totalForce = CalculateTotalForce(
                    planetPositions[i].Position, planetMasses[i].mass,
                    octreeNodesStorage, octreeNodesDataPositionsStorage, octreeNodesDataStorage, octreeNodesInternalDataStorage,
                    threshold, gravitationalConstant
                    );
                planetForces[i] = new Force
                {
                    force = totalForce
                };
            }
        }
    }

    [BurstCompile]
    public struct BarnesHutFinalize
        : IJobChunk
    {
        public ComponentTypeHandle<LocalTransform> planetPositionHandle;
        public ComponentTypeHandle<Velocity> planetVelocityHandle;
        public ComponentTypeHandle<Mass> planetMassHandle;
        public ComponentTypeHandle<Force> planetForceHandle;
        public float deltaTime;
        public float deltaTime2;

        public void Execute(in ArchetypeChunk chunk, int unfilteredChunkIndex, bool useEnabledMask, in v128 chunkEnabledMask)
        {
            var planetPositions = chunk.GetNativeArray(ref planetPositionHandle);
            var planetVelocities = chunk.GetNativeArray(ref planetVelocityHandle);
            var planetMasses = chunk.GetNativeArray(ref planetMassHandle);
            var planetForces = chunk.GetNativeArray(ref planetForceHandle);

            for (int i = 0, chunkSize = chunk.Count; i < chunkSize; i++)
            {
                var totalForce = planetForces[i].force;
                var totalAcceleration = totalForce / planetMasses[i].mass;

                planetPositions[i] =
                    planetPositions[i].Translate(
                        planetVelocities[i].velocity * deltaTime +
                        totalAcceleration * deltaTime2 / 2
                    );
                planetVelocities[i] = new Velocity
                {
                    velocity = planetVelocities[i].velocity + totalAcceleration * deltaTime
                };
            }
        }
    }
}

namespace BarnesHutOld
{
    [BurstCompile]
    public struct BHOctreeNodeData
    {
        public float3 centerOfMass;
        public float totalMass;
    }

    [BurstCompile]
    public unsafe struct BHOctreeNode
    {
        public static readonly int3 childIndexMapping = math.int3(1, 2, 4);

        // Spatial octree data
        public float3 center;
        public float extent;

        public float3 centerOfMass;
        public float totalMass;

        public int storageIndex;
        public int massesCount;

        public fixed int childIndices[8];


        public BHOctreeNode(
            float3 center, float extent,
            int storageIndex
            )
        {
            this.center = center;
            this.extent = extent;
            centerOfMass = float3.zero;
            this.totalMass = 0;
            this.storageIndex = storageIndex;
            this.massesCount = 0;
        }

        public bool IsLeaf
            => massesCount <= 1;

        public bool IsEmpty
            => massesCount == 0;

        [BurstCompile]
        public int GetChildIndex(float3 position)
        {
#if DEBUG
            if (IsLeaf)
                throw new System.InvalidOperationException("This BHOctree is a leaf node! It doesn't have any children nodes!");
#endif
            bool3 relativePosition = position > center;
            int childIndex = math.csum(math.int3(relativePosition) * childIndexMapping);
            return childIndices[childIndex];
        }

        [BurstCompile]
        public void Subdivide(int newNodesStorageIndexStart, NativeSlice<BHOctreeNode> childNodes)
        {
#if DEBUG
            if (IsEmpty)
                throw new System.InvalidOperationException("This BHOctree is empty! It doesn't require subdivision!");
#endif
            // CREATE SUBNODES

            // 0 means -x/-y/-z
            // 1 means x/y/z

            float newExtent = extent / 2;
            float3 poffset = math.float3(newExtent);
            float3 noffset = -poffset;

            float3 center000 = center + math.select(noffset, poffset, math.bool3(false, false, false));
            float3 center100 = center + math.select(noffset, poffset, math.bool3(true, false, false));
            float3 center010 = center + math.select(noffset, poffset, math.bool3(false, true, false));
            float3 center110 = center + math.select(noffset, poffset, math.bool3(true, true, false));
            float3 center001 = center + math.select(noffset, poffset, math.bool3(false, false, true));
            float3 center101 = center + math.select(noffset, poffset, math.bool3(true, false, true));
            float3 center011 = center + math.select(noffset, poffset, math.bool3(false, true, true));
            float3 center111 = center + math.select(noffset, poffset, math.bool3(true, true, true));

            childNodes[0] = new BHOctreeNode(center000, newExtent, newNodesStorageIndexStart);
            childNodes[1] = new BHOctreeNode(center100, newExtent, newNodesStorageIndexStart + 1);
            childNodes[2] = new BHOctreeNode(center010, newExtent, newNodesStorageIndexStart + 2);
            childNodes[3] = new BHOctreeNode(center110, newExtent, newNodesStorageIndexStart + 3);
            childNodes[4] = new BHOctreeNode(center001, newExtent, newNodesStorageIndexStart + 4);
            childNodes[5] = new BHOctreeNode(center101, newExtent, newNodesStorageIndexStart + 5);
            childNodes[6] = new BHOctreeNode(center011, newExtent, newNodesStorageIndexStart + 6);
            childNodes[7] = new BHOctreeNode(center111, newExtent, newNodesStorageIndexStart + 7);
            childIndices[0] = newNodesStorageIndexStart;
            childIndices[1] = newNodesStorageIndexStart + 1;
            childIndices[2] = newNodesStorageIndexStart + 2;
            childIndices[3] = newNodesStorageIndexStart + 3;
            childIndices[4] = newNodesStorageIndexStart + 4;
            childIndices[5] = newNodesStorageIndexStart + 5;
            childIndices[6] = newNodesStorageIndexStart + 6;
            childIndices[7] = newNodesStorageIndexStart + 7;

            // MARK NON LEAF NODE

            massesCount = 2;

            // PLACE MASS OF CURRENT NODE IN CHILDREN

            float3 massPosition = centerOfMass;
            float mass = totalMass;
            // center of mass of this node will be calculated in final pass
            centerOfMass = float3.zero;
            totalMass = 0;

            var childIndex = GetChildIndex(massPosition);
            var sliceIndex = childIndex - newNodesStorageIndexStart;
            var childNode = childNodes[sliceIndex];
            childNode.PlaceMass(massPosition, mass);
            childNodes[sliceIndex] = childNode;
        }

        [BurstCompile]
        public void PlaceMass(float3 massPosition, float mass)
        {
#if DEBUG
            float3 low = center - extent;
            float3 high = center + extent;
            if (math.any(massPosition < low) || math.any(massPosition > high))
                throw new System.InvalidOperationException("Added mass is outside the bounds of this BHOctree node!");
            if (!IsEmpty)
                throw new System.InvalidOperationException("This BHOctreeNode is not empty! Subdivide this node in order to place new mass!");
#endif
            centerOfMass = massPosition;
            totalMass = mass;

            massesCount = 1;
        }

        [BurstCompile]
        public void CalculateCenterOfMass(NativeSlice<BHOctreeNode> allNodes)
        {
            if (IsLeaf)
                return;

            float totalMass = 0;
            float3 centerOfMass = float3.zero;
            int massesCount = 0;

            for (int i = 0; i < 8; i++)
            {
                var node = allNodes[childIndices[i]];
                node.CalculateCenterOfMass(allNodes);
                allNodes[childIndices[i]] = node;

                totalMass += allNodes[childIndices[i]].totalMass;
                centerOfMass += allNodes[childIndices[i]].centerOfMass;
                massesCount += allNodes[childIndices[i]].massesCount;
            }

            centerOfMass /= totalMass;

            this.centerOfMass = centerOfMass;
            this.totalMass = totalMass;
            this.massesCount = massesCount;
        }
    }

    [BurstCompile]
    public struct BHOctree
    {
        private static float3 CalculateForce(
            float mass0,
            float mass1,
            float3 delta, float distance
            )
        {
            var distance3 = distance * distance * distance;
            var accelerationNoMass = delta / distance3;
            return mass0 * mass1 * accelerationNoMass;
        }

        private static float3 CalculateTotalForce(
            float3 massPosition, float mass,
            int currentNodeIndex,
            NativeArray<BHOctreeNode> allNodes,
            float threshold
            )
        {
            var delta = allNodes[currentNodeIndex].centerOfMass - massPosition;
            var distance = math.length(delta);
            var distanceIsZero = math.abs(distance) < float.Epsilon;
            if (distanceIsZero)
                return float3.zero;

            var sd = allNodes[currentNodeIndex].extent / distance;

            if (sd < threshold)
            {
                return CalculateForce(mass, allNodes[currentNodeIndex].totalMass, delta, distance);
            }
            else
            {
                if (allNodes[currentNodeIndex].IsEmpty)
                    return float3.zero;

                float3 totalForce = float3.zero;

                for (int i = 0; i < 8; i++)
                {
                    int childIndex = -1;
                    unsafe
                    {
                        var currentNode = allNodes[currentNodeIndex];
                        childIndex = currentNode.childIndices[i];
                    }
                    totalForce += CalculateTotalForce(massPosition, mass, childIndex, allNodes, threshold);
                }

                return totalForce;
            }
        }

        private unsafe struct Stack
        {
            private const int MAX_STACK_SIZE = 1024;
            private fixed int stack[MAX_STACK_SIZE];
            private int stackPointer;

            [BurstCompile]
            public void Push(int element)
            {
#if DEBUG
                if (stackPointer == MAX_STACK_SIZE)
                    throw new System.StackOverflowException();
#endif
                stack[stackPointer++] = element;
            }

            public bool Empty
                => stackPointer == 0;

            [BurstCompile]
            public int Pop()
            {
#if DEBUG
                if (stackPointer == 0)
                    throw new System.StackOverflowException();
#endif
                return stack[--stackPointer];
            }
        }

        public unsafe static float3 CalculateTotalForce(
            float3 massPosition, float mass,
            NativeArray<BHOctreeNode> allNodes,
            float threshold, float gravitationalConstant
            )
        {
            float3 totalForce = float3.zero;

            unsafe
            {
                Stack stack = new Stack();

                stack.Push(0);

                while (!stack.Empty)
                {
                    int currentNodeIndex = stack.Pop();

                    var delta = allNodes[currentNodeIndex].centerOfMass - massPosition;
                    var distance = math.length(delta);
                    var distanceIsZero = math.abs(distance) < float.Epsilon;
                    if (distanceIsZero)
                        continue;

                    var sd = allNodes[currentNodeIndex].extent / distance;

                    if (sd < threshold)
                    {
                        totalForce += CalculateForce(mass, allNodes[currentNodeIndex].totalMass, delta, distance);
                    }
                    else
                    {
                        if (allNodes[currentNodeIndex].IsLeaf)
                        {
                            totalForce += CalculateForce(mass, allNodes[currentNodeIndex].totalMass, delta, distance);
                        }
                        else if (!allNodes[currentNodeIndex].IsEmpty)
                        {
                            var currentNode = allNodes[currentNodeIndex];

                            var c0 = currentNode.childIndices[0];
                            var c1 = currentNode.childIndices[1];
                            var c2 = currentNode.childIndices[2];
                            var c3 = currentNode.childIndices[3];
                            var c4 = currentNode.childIndices[4];
                            var c5 = currentNode.childIndices[5];
                            var c6 = currentNode.childIndices[6];
                            var c7 = currentNode.childIndices[7];

                            for (int i = 7; i >= 0; i--)
                            {
                                stack.Push(currentNode.childIndices[i]);
                            }
                        }
                    }
                }
            }

            return gravitationalConstant * totalForce;// CalculateTotalForce(massPosition, mass, 0, allNodes, threshold);
        }

        private static int GetMaxDepth(
            BHOctreeNode currentNode,
            NativeArray<BHOctreeNode> allNodes
            )
        {
            if (currentNode.IsLeaf)
                return 0;
            int maxDepth = 0;
            for (int i = 0; i < 8; i++)
            {
                unsafe
                {
                    maxDepth = math.max(maxDepth, GetMaxDepth(allNodes[currentNode.childIndices[i]], allNodes) + 1);
                }
            }
            return maxDepth;
        }

        public static int GetMaxDepth(
            NativeArray<BHOctreeNode> allNodes
            )
        {
            return GetMaxDepth(allNodes[0], allNodes);
        }

        public static void Create(
            NativeList<BHOctreeNode> allNodes,
            NativeSlice<float3> positions, NativeSlice<float> masses,
            float3 center, float extent
            )
        {
#if DEBUG
            if (positions.Length != masses.Length)
                throw new System.ArgumentException("Positions and masses arrays must be of the same size!", $"{nameof(positions)}, {nameof(masses)}");
            if (positions.Length == 0)
                throw new System.ArgumentException("Positions and masses arrays contain at least 1 element!", $"{nameof(positions)}, {nameof(masses)}");
#endif
            BHOctreeNode rootNode = new BHOctreeNode(
                center, extent, allNodes.Length
                );
            rootNode.PlaceMass(positions[0], masses[0]);
            allNodes.Add(rootNode);

            for (int i = 1; i < positions.Length; i++)
            {
                var position = positions[i];
                var mass = masses[i];

                int currentNodeIndex = 0;

                // move through the octree
                for (;
                    !allNodes[currentNodeIndex].IsLeaf;
                    currentNodeIndex = allNodes[currentNodeIndex].GetChildIndex(position)
                    ) ;

                while (!allNodes[currentNodeIndex].IsEmpty)
                {
                    // make space in list
                    var baseIndex = allNodes.Length;
                    allNodes.Resize(allNodes.Length + 8, NativeArrayOptions.UninitializedMemory);
                    // subdivide
                    var node = allNodes[currentNodeIndex];
                    node.Subdivide(baseIndex, allNodes.AsArray().Slice(baseIndex, 8));
                    allNodes[currentNodeIndex] = node;
                    // move to new node
                    currentNodeIndex = allNodes[currentNodeIndex].GetChildIndex(position);
                }

                // place new mass
                {
                    var node = allNodes[currentNodeIndex];
                    node.PlaceMass(position, mass);
                    allNodes[currentNodeIndex] = node;
                }
            }

            rootNode = allNodes[0];
            rootNode.CalculateCenterOfMass(allNodes.AsArray());
            allNodes[0] = rootNode;
        }
    }

    [BurstCompile]
    public struct BarnesHutOctreeConstruction
        : IJob
    {
        public NativeList<BHOctreeNode> bhOctreeNodesStorage;
        public NativeArray<ArchetypeChunk> archetypeChunks;
        public NativeArray<int> baseEntityIndexArray;
        public NativeArray<float3> positions;
        public NativeArray<float> masses;
        public ComponentTypeHandle<LocalTransform> localTransformHandle;
        public ComponentTypeHandle<Mass> massHandle;
        public float3 center;
        public float extent;

        [BurstCompile]
        public void Execute()
        {
            for (int ai = 0; ai < archetypeChunks.Length; ai++)
            {
                var chunk = archetypeChunks[ai];
                var chunkPositions = chunk.GetNativeArray(ref localTransformHandle);
                var chunkMasses = chunk.GetNativeArray(ref massHandle);

                for (int ci = 0, chunkSize = chunk.Count; ci < chunkSize; ci++)
                {
                    int entityIndexInQuery = baseEntityIndexArray[ai] + ci;
                    positions[entityIndexInQuery] = chunkPositions[ci].Position;
                    masses[entityIndexInQuery] = chunkMasses[ci].mass;
                }
            }

            BHOctree.Create(
                bhOctreeNodesStorage,
                positions, masses,
                center, extent
                );
        }
    }

    [BurstCompile]
    public struct BarnesHutEvaluate
        : IJobChunk
    {
        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<BHOctreeNode> bhOctreeNodesStorage;
        public ComponentTypeHandle<LocalTransform> planetPositionHandle;
        public ComponentTypeHandle<Mass> planetMassHandle;
        public ComponentTypeHandle<Velocity> planetVelocityHandle;
        public ComponentTypeHandle<Force> planetForceHandle;
        public float threshold;
        public float gravitationalConstant;

        public void Execute(in ArchetypeChunk chunk, int unfilteredChunkIndex, bool useEnabledMask, in v128 chunkEnabledMask)
        {
            var planetPositions = chunk.GetNativeArray(ref planetPositionHandle);
            var planetMasses = chunk.GetNativeArray(ref planetMassHandle);
            var planetForces = chunk.GetNativeArray(ref planetForceHandle);

            for (int i = 0, chunkSize = chunk.Count; i < chunkSize; i++)
            {
                var totalForce = BHOctree.CalculateTotalForce(planetPositions[i].Position, planetMasses[i].mass, bhOctreeNodesStorage, threshold, gravitationalConstant);
                planetForces[i] = new Force
                {
                    force = totalForce
                };
            }
        }
    }

    [BurstCompile]
    public struct BarnesHutFinalize
        : IJobChunk
    {
        public ComponentTypeHandle<LocalTransform> planetPositionHandle;
        public ComponentTypeHandle<Velocity> planetVelocityHandle;
        public ComponentTypeHandle<Mass> planetMassHandle;
        public ComponentTypeHandle<Force> planetForceHandle;
        public float deltaTime;
        public float deltaTime2;

        public void Execute(in ArchetypeChunk chunk, int unfilteredChunkIndex, bool useEnabledMask, in v128 chunkEnabledMask)
        {
            var planetPositions = chunk.GetNativeArray(ref planetPositionHandle);
            var planetVelocities = chunk.GetNativeArray(ref planetVelocityHandle);
            var planetMasses = chunk.GetNativeArray(ref planetMassHandle);
            var planetForces = chunk.GetNativeArray(ref planetForceHandle);

            for (int i = 0, chunkSize = chunk.Count; i < chunkSize; i++)
            {
                var totalForce = planetForces[i].force;
                var totalAcceleration = totalForce / planetMasses[i].mass;

                planetPositions[i] =
                    planetPositions[i].Translate(
                        planetVelocities[i].velocity * deltaTime +
                        totalAcceleration * deltaTime2 / 2
                    );
                planetVelocities[i] = new Velocity
                {
                    velocity = planetVelocities[i].velocity + totalAcceleration * deltaTime
                };
            }
        }
    }
}

