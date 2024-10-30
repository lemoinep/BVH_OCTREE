
#pragma clang diagnostic push

#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"


#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <optional>
#include <random>
#include <cfloat>
#include <stdexcept>

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hipblas.h"
#include "hipsolver.h"
#include "hipblas-export.h"

#include <roctx.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/count.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/hip/vector.h>
#include <thrust/partition.h>

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>



 //Links Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#define MAX_TRIANGLES_PER_NODE 8
#define MAX_TRIANGLES_PER_LEAF 8
#define MAX_DEPTH 100 // Maximum depth of the tree
#define MAX_STACK_SIZE 1024
#define MAX_NODES_PER_LEVEL 1000

//#define MAX_DEPTH 10  // Maximum depth of the tree
//#define MAX_TRIANGLES_PER_LEAF 10  // Maximum number of triangles per sheet
//#define MAX_NODES_PER_LEVEL 1000000  // Maximum number of nodes per level
//#define MAX_TRIANGLES_PER_NODE 1000  // Maximum number of triangles per node

// NOTA: The octree is a hierarchical data structure used to speed up the process of ray tracing in complex 3D scenes. 
// An octree recursively subdivides the 3D space into eight equal subvolumes (or voxels). 
// Each node in the tree represents a cubic volume of the space. The main advantage of the octree is that it allows for fast 
// and efficient ray traversal through the scene. What is tested in this program development....




#define SWAP(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)

// Function to make a dot
__host__ __device__
float dot(const float3& a, const float3& b);

// Function to make a cross
__host__ __device__
float3 cross(const float3& a, const float3& b);

// Function to return a length
__host__ __device__
float length(const float3& v);

// Function to normalize
__host__ __device__
float3 normalize(const float3& v);


struct Vec3 {
    float x, y, z;

    __host__ __device__
    Vec3 operator-(const Vec3& v) const {
        return {x - v.x, y - v.y, z - v.z};
    }

    __host__ __device__
    Vec3 operator+(const Vec3& v) const {
        return {x + v.x, y + v.y, z + v.z};
    }

    __host__ __device__
    Vec3 operator*(float scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }

     __host__ __device__
        Vec3 operator*(const Vec3& other) const {
        return Vec3(x * other.x, y * other.y, z * other.z);
    }

    __host__ __device__
    float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __host__ __device__
    Vec3 cross(const Vec3& v) const {
        return {
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        };
    }

    __host__ __device__
    Vec3 () : x(0), y(0), z(0) {}

    __host__ __device__
    Vec3 (float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__
    Vec3 init(float x, float y, float z) {
        Vec3 v;
        v.x = x;
        v.y = y;
        v.z = z;
        return v;
    }

    __host__ __device__
    Vec3 float3ToVec3(const float3& f) {
        return Vec3(f.x, f.y, f.z);
    }

    __host__ __device__
    void normalize() {
        float length = sqrt(x * x + y * y + z * z);
        if (length > 0) {
            x /= length;
            y /= length;
            z /= length;
        }
    }
};

__device__ inline float3 elementwise_min(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ inline float3 elementwise_max(const float3& a, const float3& b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}


// Function to make a dot
__host__ __device__
float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Function to make a cross
__host__ __device__
float3 cross(const float3& a, const float3& b) {
    return float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Function to return a length
__host__ __device__
float length(const float3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Function to normalize
__host__ __device__
float3 normalize(const float3& v) {
    float len = length(v);
    if (len > 0) {
        return float3(v.x / len, v.y / len, v.z / len);
    }
    return v;
}

// Function to write a float3
__host__ __device__
void print_float3(const float3& v) {
    printf("%f %f %f\n",v.x,v.y,v.z);
}


__host__ __device__ Vec3 min(const Vec3& a, const Vec3& b) {
	return Vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __device__ Vec3 max(const Vec3& a, const Vec3& b) {
	return Vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}


__host__ __device__ Vec3 cross(const Vec3& a, const Vec3& b) {
	return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__host__ __device__ float dot(const Vec3& a, const Vec3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}


float3 toFloat3(const Vec3& v) { return {v.x, v.y, v.z}; }



struct OctreeTriangle {
    float3 vertices[3];
    int id;

    __device__ float3 getNormal() const {
        return normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
    }
};


struct OctreeRay {
    float3 origin;
    float3 direction;
};


struct HitOctreeRay {
    float distanceResults;
    int hitResults;
    int idResults;
    float3 intersectionPoint;
};



struct OctreeAABB {
    float3 min;
    float3 max;

    __device__ bool intersect(const OctreeRay& ray) const {
        
        float3 invDir = make_float3(1.0f,1.0f,1.0f) / ray.direction;
        float tmin = (min.x - ray.origin.x) * invDir.x;
        float tmax = (max.x - ray.origin.x) * invDir.x;
        if (invDir.x < 0.0f) {
            SWAP(float,tmin, tmax);
        }
        float tymin = (min.y - ray.origin.y) * invDir.y;
        float tymax = (max.y - ray.origin.y) * invDir.y;
        if (invDir.y < 0.0f) {
            SWAP(float,tymin, tymax);
        }
        if ((tmin > tymax) || (tymin > tmax)) {
            return false;
        }
        if (tymin > tmin) {
            tmin = tymin;
        }
        if (tymax < tmax) {
            tmax = tymax;
        }

        float tzmin = (min.z - ray.origin.z) * invDir.z;
        float tzmax = (max.z - ray.origin.z) * invDir.z;

        if (invDir.z < 0.0f) {
            SWAP(float,tzmin, tzmax);
        }

        if ((tmin > tzmax) || (tzmin > tmax)) {
            return false;
        }

        return true;
    }
};

struct OctreeNode {
    OctreeAABB bbox;
    OctreeNode* children[8];
    int triangleCount;
    OctreeTriangle triangles[MAX_TRIANGLES_PER_NODE];
    bool isLeaf;

    __device__ OctreeNode() : triangleCount(0), isLeaf(false) {
        for (int i = 0; i < 8; ++i) {
            children[i] = nullptr;
        }
    }

    __device__ ~OctreeNode() {
        for (int i = 0; i < 8; ++i) {
            if (children[i] != nullptr) {
                delete children[i];
            }
        }
    }
};



__device__ OctreeAABB computeBoundingBox(OctreeTriangle* triangles, int triangleCount) {
    OctreeAABB bbox;
    bbox.min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    bbox.max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = 0; i < triangleCount; ++i) {
        for (int j = 0; j < 3; ++j) {
            bbox.min = elementwise_min(bbox.min, triangles[i].vertices[j]);
            bbox.max = elementwise_max(bbox.max, triangles[i].vertices[j]);
        }
    }
    return bbox;
}


__device__ int distributeTriangles(const OctreeAABB& bbox, OctreeTriangle* triangles, int triangleCount, OctreeTriangle* childTriangles, int octantIndex) {
    float3 midPoint = 0.5f * (bbox.min + bbox.max);
    int childCount = 0;

    for (int i = 0; i < triangleCount; ++i) {
        OctreeTriangle& tri = triangles[i];
        OctreeAABB triBBox = computeBoundingBox(&tri, 1); 

        // Check if the triangle is in the corresponding octant
        bool inOctant = true;
        if (octantIndex & 1) inOctant &= triBBox.max.x > midPoint.x; // Right
        else inOctant &= triBBox.min.x < midPoint.x; // Left

        if (octantIndex & 2) inOctant &= triBBox.max.y > midPoint.y; // Up
        else inOctant &= triBBox.min.y < midPoint.y; // Down

        if (octantIndex & 4) inOctant &= triBBox.max.z > midPoint.z; // Front
        else inOctant &= triBBox.min.z < midPoint.z; // Back

        if (inOctant && childCount < MAX_TRIANGLES_PER_NODE) {
            childTriangles[childCount++] = tri; // Add the triangle to the octant
        }
    }

    return childCount;
}











__device__ bool rayAABBIntersection(const OctreeRay& ray, const OctreeAABB& aabb) {
    float3 invDir = make_float3(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
    float3 tMin = (aabb.min - ray.origin) * invDir;
    float3 tMax = (aabb.max - ray.origin) * invDir;
    
    float3 t1, t2;
    t1.x = fminf(tMin.x, tMax.x);
    t1.y = fminf(tMin.y, tMax.y);
    t1.z = fminf(tMin.z, tMax.z);
    
    t2.x = fmaxf(tMin.x, tMax.x);
    t2.y = fmaxf(tMin.y, tMax.y);
    t2.z = fmaxf(tMin.z, tMax.z);
    
    float tNear = fmaxf(fmaxf(t1.x, t1.y), t1.z);
    float tFar = fminf(fminf(t2.x, t2.y), t2.z);
    
    return tNear <= tFar && tFar > 0;
}

__device__ bool rayTriangleIntersection(const OctreeRay& ray, const OctreeTriangle& triangle, float& t,float3& intersectionPoint) {
    float3 edge1 = triangle.vertices[1] - triangle.vertices[0];
    float3 edge2 = triangle.vertices[2] - triangle.vertices[0];

    float3 h = cross(ray.direction, edge2);
    float a = dot(edge1, h);

    if (a > -1e-8 && a < 1e-8) {
        return false;
    }

    float f = 1.0f / a;
    float3 s = ray.origin - triangle.vertices[0];
    float u = f * dot(s, h);


    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    float3 q = cross(s, edge1);
    float v = f * dot(ray.direction, q);


    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    t = f * dot(edge2, q);

    if (t > 1e-6) {
        intersectionPoint = ray.origin + t * ray.direction;
    }
    else
    {
        intersectionPoint =make_float3(INFINITY, INFINITY, INFINITY);
    }

     return t > 1e-8; 
}


__device__ bool rayAABBIntersection(const OctreeRay& ray, const OctreeAABB& aabb, float& tEntry, float& tExit) {
    float3 invDir = make_float3(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
    float3 tMin = (aabb.min - ray.origin) * invDir;
    float3 tMax = (aabb.max - ray.origin) * invDir;
    
    float3 t1, t2;
    t1.x = fminf(tMin.x, tMax.x);
    t1.y = fminf(tMin.y, tMax.y);
    t1.z = fminf(tMin.z, tMax.z);
    
    t2.x = fmaxf(tMin.x, tMax.x);
    t2.y = fmaxf(tMin.y, tMax.y);
    t2.z = fmaxf(tMin.z, tMax.z);
    
    tEntry = fmaxf(fmaxf(t1.x, t1.y), t1.z);
    tExit = fminf(fminf(t2.x, t2.y), t2.z);
    
    return tEntry <= tExit && tExit > 0;
}

__device__ bool rayAABBIntersection(const OctreeRay& ray, const OctreeAABB& aabb, float& tMin) {
    float3 invDir = make_float3(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
    float3 t0 = (aabb.min - ray.origin) * invDir;
    float3 t1 = (aabb.max - ray.origin) * invDir;
    
    float3 tmin;
    tmin.x = fminf(t0.x, t1.x);
    tmin.y = fminf(t0.y, t1.y);
    tmin.z = fminf(t0.z, t1.z);
    float3 tmax;
    tmax.x = fmaxf(t0.x, t1.x);
    tmax.y = fmaxf(t0.y, t1.y);
    tmax.z = fmaxf(t0.z, t1.z);
    
    float tNear = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
    float tFar = fminf(fminf(tmax.x, tmax.y), tmax.z);
    
    if (tNear <= tFar && tFar > 0) {
        tMin = tNear > 0 ? tNear : tFar;
        return true;
    }
    
    return false;
}




__device__ bool traverseOctreeIterative(OctreeNode* root, const OctreeRay& ray, float& tMin, OctreeTriangle& hitTriangle,HitOctreeRay& hr) {
    
    bool isView=true; isView=false;
    
    if (isView) printf("[traverseOctree]\n");

    // Stack to store nodes to visit
    struct StackEntry {
        OctreeNode* node;
        float tEntry;
    };
    StackEntry stack[MAX_STACK_SIZE];
    int stackSize = 0;


    float rootTEntry;
    if (!rayAABBIntersection(ray, root->bbox, rootTEntry)) {
        return false;  // No intersection with the root
    }
    stack[stackSize++] = {root, rootTEntry};

    bool hit = false;

    while (stackSize > 0) {
        StackEntry current = stack[--stackSize];
        OctreeNode* currentNode = current.node;
        float nodeEntry = current.tEntry;

        if (isView) printf("Visiting node: isLeaf = %s, triangleCount = %d\n", currentNode->isLeaf ? "true" : "false", currentNode->triangleCount);

        if (nodeEntry >= tMin) {
            continue;  // No need to check this node or its children
        }

        if (currentNode->isLeaf) {
            //printf("isLeaf\n");
            for (int i = 0; i < currentNode->triangleCount; ++i) {
                if (isView) {
                    printf("in leaf %i %f\n",currentNode->triangleCount,tMin);
                    printf("Triangle %d: (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)\n", i,
                        currentNode->triangles[i].vertices[0].x, currentNode->triangles[i].vertices[0].y, currentNode->triangles[i].vertices[0].z,
                        currentNode->triangles[i].vertices[1].x, currentNode->triangles[i].vertices[1].y, currentNode->triangles[i].vertices[1].z,
                        currentNode->triangles[i].vertices[2].x, currentNode->triangles[i].vertices[2].y, currentNode->triangles[i].vertices[2].z);
                }
                

                float t;
                float3 intersectionPointT;
                if (rayTriangleIntersection(ray, currentNode->triangles[i], t,intersectionPointT) ) {
                    if (t < tMin) {
                        tMin = t;
                        hitTriangle = currentNode->triangles[i];
                        hit = true;
                        if (isView) printf("Hit triangle at t = %f\n", t);
                        hr.hitResults = i;
                        hr.distanceResults=fabs(t);  //distance
                        hr.intersectionPoint=intersectionPointT;
                        hr.idResults=int(currentNode->triangles[i].id);
                    }

                }
            }
        } else {
            for (int i = 0; i < 8; ++i) {
                if (currentNode->children[i] != nullptr) {
                    float childTEntry;
                    if (rayAABBIntersection(ray, currentNode->children[i]->bbox, childTEntry) &&
                        childTEntry < tMin) {
                        stack[stackSize++] = {currentNode->children[i], childTEntry};
                    }
                }
            }

            // Sort the stack by ascending tEntry
            for (int i = stackSize - 1; i > 0; --i) {
                for (int j = 0; j < i; ++j) {
                    if (stack[j].tEntry > stack[j+1].tEntry) {
                        StackEntry temp = stack[j];
                        stack[j] = stack[j+1];
                        stack[j+1] = temp;
                    }
                }
            }
        }
    }

    return hit;
}




__device__ OctreeRay generateOctreeRay(int idx,int width,int height) {
    float3 origin = make_float3(0.0f, 0.0f, 2.0f); 
    float3 direction = make_float3(
        (idx % width) / (float) width - 0.5f, 
        (idx / width) / (float) height - 0.5f,
        -1.0f 
    );
    return {origin, normalize(direction)}; 
}



__global__ void rayTracingKernelTraverse(OctreeNode* d_octree,HitOctreeRay* d_HitRays,int width,int height) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return; 
        OctreeRay ray = generateOctreeRay(idx,width,height);        
        d_HitRays[idx].hitResults = -1;
        d_HitRays[idx].distanceResults = INFINITY; //distance
        d_HitRays[idx].intersectionPoint=make_float3(INFINITY, INFINITY, INFINITY);
        d_HitRays[idx].idResults = -1;
        float tMin = INFINITY;
        OctreeTriangle hitTriangle;
        bool hit = traverseOctreeIterative(d_octree,ray, tMin, hitTriangle,d_HitRays[idx]);

        if (hit) {
            printf("Hit dist :  %f\n", tMin);
        } 
        else 
        {
            printf("No hit\n");
        }
}


__device__ OctreeAABB computeChildBBox(const OctreeAABB& parentBBox, int childIndex) {
    OctreeAABB childBBox;
    float3 center = make_float3(
        (parentBBox.min.x + parentBBox.max.x) * 0.5f,
        (parentBBox.min.y + parentBBox.max.y) * 0.5f,
        (parentBBox.min.z + parentBBox.max.z) * 0.5f
    );

    childBBox.min = parentBBox.min;
    childBBox.max = parentBBox.max;

    if (childIndex & 1) childBBox.min.x = center.x; else childBBox.max.x = center.x;
    if (childIndex & 2) childBBox.min.y = center.y; else childBBox.max.y = center.y;
    if (childIndex & 4) childBBox.min.z = center.z; else childBBox.max.z = center.z;

    return childBBox;
}


__device__ void buildOctreeByLevel(OctreeNode* root, OctreeTriangle* triangles, int triangleCount) {
    // Initialize the root
    root->bbox = computeBoundingBox(triangles, triangleCount);
    root->triangleCount = triangleCount;
    root->isLeaf = false;

    bool isView=true;
 
    if (isView) {
        printf("root box min <%f %f %f>\n", root->bbox.min.x, root->bbox.min.y, root->bbox.min.z);
        printf("root box max <%f %f %f>\n", root->bbox.max.x, root->bbox.max.y, root->bbox.max.z);
    }

    // Queues to store nodes to be processed at each level
    OctreeNode* currentLevelNodes[MAX_NODES_PER_LEVEL];
    OctreeNode* nextLevelNodes[MAX_NODES_PER_LEVEL];
    int currentLevelSize = 1;
    int nextLevelSize = 0;

    currentLevelNodes[0] = root;

    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        if (isView) printf("Processing depth %d with %d nodes\n", depth, currentLevelSize);
        for (int i = 0; i < currentLevelSize; i++) {
            OctreeNode* node = currentLevelNodes[i];
            if (isView) printf("Node %d at depth %d: triangleCount = %d\n", i, depth, node->triangleCount);

            if (node->triangleCount <= MAX_TRIANGLES_PER_LEAF || depth == MAX_DEPTH - 1) {
                node->isLeaf = true;
                // Copy the triangles into the leaf node
                int copyCount = min(node->triangleCount, MAX_TRIANGLES_PER_NODE);
                for (int t = 0; t < copyCount; t++) {
                    node->triangles[t] = triangles[t];
                }
                node->triangleCount = copyCount;
                if (isView) printf("Leaf node created with %d triangles\n", node->triangleCount);
                continue;
            }

            // Subdivide the node
            for (int j = 0; j < 8; j++) {
                node->children[j] = new OctreeNode();
                OctreeTriangle childTriangles[MAX_TRIANGLES_PER_NODE];
                int childCount = distributeTriangles(node->bbox, triangles, node->triangleCount, childTriangles, j);
                
                if (childCount > 0) {
                    node->children[j]->triangleCount = childCount;
                    node->children[j]->bbox = computeChildBBox(node->bbox, j);
                    
                    // Copy triangles to child
                    int copyCount = min(childCount, MAX_TRIANGLES_PER_NODE);
                    for (int t = 0; t < copyCount; t++) {
                        node->children[j]->triangles[t] = childTriangles[t];
                    }
                    node->children[j]->triangleCount = copyCount;
                    
                    if (nextLevelSize < MAX_NODES_PER_LEVEL) {
                        nextLevelNodes[nextLevelSize++] = node->children[j];
                        if (isView) printf("Child %d created with %d triangles\n", j, copyCount);
                    } else {
                        // Handle queue overflow
                        if (isView) printf("Warning: Max nodes per level reached. Child %d skipped.\n", j);
                        delete node->children[j];
                        node->children[j] = nullptr;
                    }
                } else {
                    delete node->children[j];
                    node->children[j] = nullptr;
                }
            }

            if (isView) {
                printf("Node box min <%f %f %f> max <%f %f %f>\n", 
                    node->bbox.min.x, node->bbox.min.y, node->bbox.min.z,
                    node->bbox.max.x, node->bbox.max.y, node->bbox.max.z);
            }
        }

        // Prepare the next level
        if (nextLevelSize == 0) {
            if (isView) printf("Construction completed at depth %d\n", depth);
            break;
        }
        
        memcpy(currentLevelNodes, nextLevelNodes, nextLevelSize * sizeof(OctreeNode*));
        currentLevelSize = nextLevelSize;
        nextLevelSize = 0;
    }
}




__global__ void buildOctreeKernel(OctreeNode* node, OctreeTriangle* triangles, int triangleCount, int depth) {
    buildOctreeByLevel(node, triangles, triangleCount);
}


void buildPicturRayTracing(OctreeTriangle* triangles, int triangleCount, int width, int height) {

    std::chrono::steady_clock::time_point t_begin_0,t_begin_1;
    std::chrono::steady_clock::time_point t_end_0,t_end_1;
    long int t_laps;

    t_begin_0 = std::chrono::steady_clock::now();  
    //int maxNodes=100;
    OctreeNode* d_octree;
    hipMalloc(&d_octree, sizeof(OctreeNode));
    t_end_0 = std::chrono::steady_clock::now(); 

    t_begin_0 = std::chrono::steady_clock::now();  
    buildOctreeKernel<<<dim3(1),dim3(1)>>>(d_octree, triangles, triangleCount,0);
    hipDeviceSynchronize();
    t_end_0 = std::chrono::steady_clock::now();  

    int nbRay = width * height;
    HitOctreeRay* d_HitRays;
    hipMalloc(&d_HitRays, nbRay*sizeof(HitOctreeRay));

    t_begin_1 = std::chrono::steady_clock::now();  
    dim3 blockSize(256);
    dim3 numBlocks((nbRay + blockSize.x - 1) / blockSize.x);


    rayTracingKernelTraverse<<<numBlocks, blockSize>>>(
        d_octree, 
        d_HitRays,
        width, 
        height
    );

    hipDeviceSynchronize();

    t_end_1 = std::chrono::steady_clock::now();  

    std::vector<HitOctreeRay> h_HitRays(nbRay);    
    hipMemcpy(h_HitRays.data(), d_HitRays, nbRay*sizeof(HitOctreeRay), hipMemcpyDeviceToHost);

    std::cout<<"\n";
    std::cout<<"Debriefing\n";
    std::cout<<"\n";
    for (int i=0;i<nbRay;i++)
    {
        //std::cout<<"["<<i<<"] "<<h_HitRays[i].idResults<<"\n";
        if (h_HitRays[i].idResults!=-1)
        {
            std::cout<<"["<<i<<"] "<<h_HitRays[i].hitResults<<" "
            <<h_HitRays[i].distanceResults<<" "
            <<h_HitRays[i].idResults
            <<" <"<<h_HitRays[i].intersectionPoint.x<<","<<h_HitRays[i].intersectionPoint.y<<","<<h_HitRays[i].intersectionPoint.z<<">"
            <<"\n";
        }
    }

    t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end_0 - t_begin_0).count();
    std::cout << "[INFO]: Elapsed microseconds inside OCTREE : "<<t_laps<< " us\n";

    t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end_1 - t_begin_1).count();
    std::cout << "[INFO]: Elapsed microseconds inside Ray Tracing Octree : "<<t_laps<< " us\n";

    hipFree(d_octree);
    hipFree(d_HitRays);
}



bool loadOBJOctreeTriangle(const std::string& filename, thrust::host_vector<OctreeTriangle>& triangles,const int& id) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::vector<float3> vertices;
    std::string line;
    bool isView=false;
    while (std::getline(file, line)) {
        if (line[0] == 'v') {
            float x, y, z;
            sscanf(line.c_str(), "v %f %f %f", &x, &y, &z);
            vertices.push_back(make_float3(x, y, z));
            if (isView) std::cout <<"v=<" << x <<","<<y<<","<<z<< ">\n";
        } else if (line[0] == 'f') {
            unsigned int i1, i2, i3;
            sscanf(line.c_str(), "f %u %u %u", &i1, &i2, &i3);
            if (isView) std::cout <<"f=<" << i1 <<","<<i2<<","<<i3<< ">\n";
            triangles.push_back({vertices[i1-1], vertices[i2-1], vertices[i3-1]});
            triangles.back().id=id;
        }
    }
    return true;
}

void Test001()
{
    std::string filename;
    filename = "Triangle2Cube.obj";
    filename = "Test.obj";

    int width=3;
    int height=width;
    int triangleCount = 5; 

    thrust::host_vector<OctreeTriangle> h_triangles(triangleCount);
    h_triangles[0] = { make_float3(-1.0f, -1.0f, -5.0f), make_float3(1.0f, -1.0f, -5.0f), make_float3(0.0f, 1.0f, -5.0f),0 };
    h_triangles[1] = { make_float3(-2.0f, -2.0f, -6.0f), make_float3(2.0f, -2.0f, -6.0f), make_float3(0.0f, 2.0f, -6.0f),1 };
    h_triangles[2] = { make_float3(-1.5f, -1.5f, -4.0f), make_float3(1.5f, -1.5f, -4.0f), make_float3(0.0f, 1.5f, -4.0f),2 };
    h_triangles[3] = { make_float3(-1.2f, -1.2f, -7.0f), make_float3(1.2f, -1.2f, -7.0f), make_float3(0.0f, 1.2f, -7.0f),3};
    h_triangles[4] = { make_float3(-1.8f, -1.8f, -8.0f), make_float3(1.8f, -1.8f, -8.0f), make_float3(0.0f, 1.8f, -8.0f),4 };

    triangleCount = h_triangles.size();
    std::cout<<"Nb Triangles="<<triangleCount<<"\n";

    // Copy Triangle host to device
    OctreeTriangle* d_triangles;
    hipMalloc(&d_triangles, sizeof(OctreeTriangle) * triangleCount);
    hipMemcpy(d_triangles, h_triangles.data(), sizeof(OctreeTriangle) * triangleCount, hipMemcpyHostToDevice);

    buildPicturRayTracing(d_triangles, triangleCount, width, height);

    hipFree(d_triangles);
}


void Test002()
{
     
    std::string filename;
    filename = "Triangle2Cube.obj";
    filename = "Test.obj";

    int width=3;
    int height=width;
    int triangleCount = 5; 

    thrust::host_vector<OctreeTriangle> h_triangles;
    loadOBJOctreeTriangle("Test.obj", h_triangles,0);
    triangleCount = h_triangles.size();
    std::cout<<"Nb Triangles="<<triangleCount<<"\n";

    // Copy Triangle host to device
    OctreeTriangle* d_triangles;
    hipMalloc(&d_triangles, sizeof(OctreeTriangle) * triangleCount);
    hipMemcpy(d_triangles, h_triangles.data(), sizeof(OctreeTriangle) * triangleCount, hipMemcpyHostToDevice);

    buildPicturRayTracing(d_triangles, triangleCount, width, height);

    hipFree(d_triangles);
}


int main(){

    Test001();
    std::cout<<"=========================================\n";
    Test002();
    std::cout<<"=========================================\n";
    
    return 0;
}



#pragma clang diagnostic pop



