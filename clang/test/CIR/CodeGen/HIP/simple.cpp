#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x hip -fhip-new-launch-api \
// RUN:            -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:              -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// Attribute for global_fn
// CIR-HOST: [[Kernel:#[a-zA-Z_0-9]+]] = {{.*}}#cir.cuda_kernel_name<_Z9global_fnv>{{.*}}

// This should emit as a normal C++ function.
__host__ void host_fn(int *a, int *b, int *c) {}

// CIR: cir.func @_Z7host_fnPiS_S_

__global__ void global_fn() {}
// CIR-HOST: @_Z24__device_stub__global_fnv(){{.*}}extra([[Kernel]])
// CIR-DEVICE: @_Z9global_fnv
