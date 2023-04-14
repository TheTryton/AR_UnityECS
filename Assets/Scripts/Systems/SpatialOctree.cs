using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
/*
/// <summary>
/// 
/// </summary>
/// <typeparam name="LDT">Leaf data</typeparam>
/// <typeparam name="IDT">Internal data</typeparam>
[BurstCompile(OptimizeFor = OptimizeFor.Performance)]
public struct SpatialOctree<LDT, IDT>
    : IDisposable
    where LDT : unmanaged
    where IDT : unmanaged
{
    public const int maxElementsPerLeaf = 128;

    private NativeParallelHashMap<uint, Node> topLevelNodes;
    private UnsafeAppendBuffer leafData;
    private int leafDataEndIndex;
    private int leafDataEndCommitedIndex;
    private NativeParallelHashMap<uint, IntPtr> nodeLeafData;
    private readonly float topLevelChunkExtent;

    /// <summary>
    /// Returns hash for spatial index of top level node
    /// </summary>
    /// <param name="position">Element position in 3D space</param>
    /// <returns>Top level hash value of 3D space position</returns>
    private uint HashTopLevel(int3 topLevelChunkIndex)
        => math.hash(topLevelChunkIndex);

    private int3 TopLevelChunkIndex(float3 position)
    {
        var selection = position < 0;
        var high = math.int3(position / topLevelChunkExtent);
        var low = high - 1;
        return math.select(high, low, selection);
    }

    private uint Hash(float3 position)
        => math.hash(position);

    private unsafe IntPtr AppendLeafData(int count)
    {
        var dummyV = stackalloc LDT[count];

        var sliceStartIndex = Interlocked.Add(ref leafDataEndIndex, count);
        var sliceEndIndex = sliceStartIndex + count;
        while (sliceEndIndex != leafDataEndCommitedIndex) ; //spin lock
        // spin lock acquired
        leafData.Add(dummyV, count * sizeof(LDT)); // just append count elements of data to the append buffer
        // spin lock release
        leafDataEndCommitedIndex = sliceEndIndex;

        // now return the pointer to allocated append buffer slice 
        return new IntPtr((LDT*)leafData.Ptr + sliceStartIndex);
    }

    [StructLayout(LayoutKind.Explicit)]
    private struct Node
    {
        private float3 center;
        private float extent;


        private bool leaf;
        private uint hash;

        public Node(float3 center, float extent, uint hash)
        {
            this.center = center;
            this.extent = extent;
            leaf = true;
            this.hash = hash;
        }
    }

    public SpatialOctree(float topLevelChunkExtent, int topLevelSpacePreallocation, int leafDataInitialCapacity, AllocatorManager.AllocatorHandle allocatorHandle)
    {
        topLevelNodes = new NativeParallelHashMap<int, Node>(topLevelSpacePreallocation, allocatorHandle);
        leafData = new UnsafeAppendBuffer(leafDataInitialCapacity, UnsafeUtility.AlignOf<LDT>(), allocatorHandle);
        leafDataEndIndex = 0;
        nodeLeafData = new NativeParallelHashMap<int, NativeSlice<LDT>>(leafDataInitialCapacity, allocatorHandle);
        this.topLevelChunkExtent = topLevelChunkExtent;
    }

    public void Insert(float3 position, LDT leafData)
    {
        var topLevelChunkIndex = TopLevelChunkIndex(position);
        var topLevelHash = HashTopLevel(topLevelChunkIndex);

        // create the node
        var nodeCenter = math.float3(topLevelChunkIndex) * topLevelChunkExtent / 2;
        // allocate leafData
        IntPtr leafDataPtr;
        unsafe
        {
            leafDataPtr = AppendLeafData(maxElementsPerLeaf);
        }
        var node = new Node(nodeCenter, topLevelChunkExtent, topLevelHash);

        // find if top level chunk exists
        // insert new top level node
        if (topLevelNodes.TryAdd(topLevelHash, node))
        {
            // fill in remaining

            
        }
        else
        {
            if(topLevelNodes.TryGetValue(topLevelHash, out node))
            {

            }

            
        }
    }

    
    public void Dispose()
    {
        topLevelNodes.Dispose();
    }
}
*/