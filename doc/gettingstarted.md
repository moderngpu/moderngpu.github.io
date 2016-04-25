# Getting started

moderngpu 2.0 is designed for CUDA 7.5+. It runs on all generations of NVIDIA GPUs from sm_20 (Fermi) and up. 

The reference environment is 64-bit Linux with g++ 4.9.3 with nvcc 7.5.17. This library and code using it might not compile on Microsoft Visual Studio 2013 because of symbol length limitations in that compiler. Windows users should try upgrading to CUDA 8.0 and using Visual Studio 2015.

## Cloning the source

Clone the source from the moderngpu github [repository](https://www.github.com/moderngpu/moderngpu).

From the command line, 
```
git clone https://www.github.com/moderngpu/moderngpu [your_directory]
```
to clone the master branch of the repository into _your_directory_.

## Compiling

moderngpu uses some advanced opt-in features of the CUDA compiler. You may need to set these flags when building your application:
* **-I [your_directory]/src** Add moderngpu to your project's include path. Provides access to the library's header files under the `moderngpu/` path, eg `#include <moderngpu/transform.hxx>`.

* **-std=c++11** C++11 features are used extensively.

* **--expt-extended-lambda** enables device-tagged lambdas.

* **-use_fast_math** enables the fast CUDA math library. It won't give bit-identical results with arithmetic run on your host processor, but numerical apps are greatly accelerated by its inclusion.

* **-Xptxas="-v"** enables verbose reporting of PTX assembly. If your kernel uses more than 0 bytes of local memory, your code is probably doing something wrong.

* **-lineinfo** tracks kernel line numbers. `cuda-memcheck` and the CUDA Visual Profiler give more intelligible results when this option is used.

* **-gencode arch=compute_xx,code=compute_xx** generates PTX for architecture sm_xx. May be asserted multiple times to take advantage of architecture-specific tunings and intrinsics. PTX is forward compatible, but must be JIT compiled by the CUDA runtime to SASS before device code is launched. -gencode may be specified multiple times to target different architectures.

* **-gencode arch=compute_xx,code=sm_xx** generates SASS for architecture sm_xx. SASS is more space-efficient than PTX and doesn't require JIT compilation, but it's only forward-compatible within the same major architecture. That is, sm_35 devices can execute sm_30 SASS, but sm_5x devices cannot.

## Testing the library

Test your installation by compiling this simple program.

```cpp
#include <moderngpu/transform.hxx>

using namespace mgpu;

int main(int argc, char** argv) {
  // The context encapsulates things like an allocator and a stream.
  // By default it prints device info to the console.
  standard_context_t context;

  // Launch five threads to greet us.
  transform([]MGPU_DEVICE(int index) {
    printf("Hello GPU from thread %d\n", index);
  }, 5, context);

  // Synchronize on the context's stream to send the output to the console.
  context.synchronize();

  return 0;
}
```

If the library is installed at _../moderngpu_, compile with this line:
```
$ nvcc \
  -std=c++11 \
  --expt-extended-lambda \
  -gencode arch=compute_20,code=compute_20 \
  -I ../moderngpu/src \
  -o hello \
  hello.cu
```

If all goes well, the program should produce output similar to this:
```
$ ./hello
GeForce GTX 980 Ti : 1190.000 Mhz   (Ordinal 0)
22 SMs enabled. Compute Capability sm_52
FreeMem:   5837MB   TotalMem:   6140MB   64-bit pointers.
Mem Clock: 3505.000 Mhz x 384 bits   (336.5 GB/s)
ECC Disabled


Hello GPU from thread 0
Hello GPU from thread 1
Hello GPU from thread 2
Hello GPU from thread 3
Hello GPU from thread 4
```
If you see output like this:
```
$ ./hello
terminate called after throwing an instance of 'mgpu::cuda_exception_t'
  what():  invalid device function
Aborted
```
then you likely didn't build your executable with options that are compatible with the architecture of your device. For instance, building with **-gencode arch=compute_20,code=sm_20** will generate a binary that produces this output when run on a Maxwell device, because Maxwell cannot run Fermi SASS. Either generate Maxwell SASS with **-gencode arch=compute_52,code=sm_52** or PTX for any earlier architecture with **-gencode arch=compute_35,code=compute_35** to make a Maxwell-compatible binary.

## Looking at the generated code

Programming the GPU is often about meta-programming the compiler (or, depending on your level of commitment, foregoing the conveniences of a compiler and using an [assembler](https://github.com/NervanaSystems/maxas)). The easiest way to confirm that the compiler is doing what you want it to do is to look at the disassembly. Use **-gencode arch=compute_xx,code=sm_xx** to generate SASS for the architecture of interest, and then use **cuobjdump** to dump the disassembly:
```
$ nvcc \
  -std=c++11 \
  --expt-extended-lambda \
  -gencode arch=compute_52,code=sm_52 \
  -I ../moderngpu/src -o hello \
  hello.cu
$ cuobjdump -sass hello
        code for sm_52
                Function : _ZN4mgpu16launch_box_cta_kINS_15launch_params_tIL...
        .headerflags    @"EF_CUDA_SM52 EF_CUDA_PTX_SM(EF_CUDA_SM52)"
                                                           /* 0x001c5800fe0007f6 */ 
/*0008*/               MOV R1, c[0x0][0x20];               /* 0x4c98078000870001 */ 
/*0010*/     {         IADD32I R1, R1, -0x8;               /* 0x1c0fffffff870101 */ 
/*0018*/               S2R R3, SR_CTAID.X;        }        /* 0xf0c8000002570003 */ 
                                                           /* 0x083fd800e7e007f0 */ 
/*0028*/     {         LOP.OR R6, R1, c[0x0][0x4];         /* 0x4c47020000170106 */ 
/*0030*/               S2R R4, SR_TID.X;        }          /* 0xf0c8000002170004 */ 
/*0038*/               IADD32I R0, R3.reuse, 0x1;          /* 0x1c00000000170300 */ 
                                                           /* 0x005fd400fe2007f6 */ 
/*0048*/               SHL R0, R0, 0x7;                    /* 0x3848000000770000 */ 
/*0050*/               IMNMX R2, R0, c[0x0][0x140], PT;    /* 0x4c21038005070002 */ 
/*0058*/               LOP32I.AND R0, R4, 0x7f;            /* 0x0400000007f70400 */ 
                                                           /* 0x001ff400fda007e6 */ 
/*0068*/               ISCADD R2, -R3, R2, 0x7;            /* 0x5c1a038000270302 */ 
/*0070*/               ISETP.GE.AND P0, PT, R0, R2, PT;    /* 0x5b6d038000270007 */ 
/*0078*/           @P0 EXIT;                               /* 0xe30000000000000f */ 
                                                           /* 0x001fc000fe4007f1 */ 
/*0088*/               ISCADD R0, R3, R0, 0x7;             /* 0x5c18038000070300 */ 
/*0090*/               LOP32I.AND R2, R6, 0xffffff;        /* 0x04000ffffff70602 */ 
/*0098*/     {         MOV32I R4, 0x0;                     /* 0x010000000007f004 */ 
/*00a8*/               STL [R2], R0;        }              /* 0x001fd800fe2000f1 */ 
                                                           /* 0xef54000000070200 */ 
/*00b0*/               MOV32I R5, 0x0;                     /* 0x010000000007f005 */ 
/*00b8*/               MOV R7, RZ;                         /* 0x5c9807800ff70007 */ 
                                                           /* 0x001ffc00ffe00ffd */ 
/*00c8*/               JCAL 0x0;                           /* 0xe220000000000040 */ 
/*00d0*/               EXIT;                               /* 0xe30000000007000f */ 
/*00d8*/               BRA 0xd8;                           /* 0xe2400fffff87000f */ 
                                                           /* 0x001f8000fc0007e0 */ 
/*00e8*/               NOP;                                /* 0x50b0000000070f00 */ 
/*00f0*/               NOP;                                /* 0x50b0000000070f00 */ 
/*00f8*/               NOP;                                /* 0x50b0000000070f00 */ 
```

Periodic inspection of disassembly can help confirm that your C++ abstractions are having the intended effect. It may also helps catch compiler absurdities before your software is distributed to billions of end users.

## Diagnosing misbehavior

If illegal memory instructions are causing your program to crash or to fail to perform as expected, **cuda-memcheck** can help diagnose the issue.

**bad_stores.cu**
```cpp
#include <moderngpu/transform.hxx>

using namespace mgpu;

int main(int argc, char** argv) {
  standard_context_t context;

  mem_t<int> x(100, context);
  int* x_data = x.data();

  // Write past the end of x's data.
  transform([=]MGPU_DEVICE(int index) {
    x_data[1000 + index] = index * index;
  }, 128, context);

  // Synchronize on the context's stream to send the output to the console.
  context.synchronize();  

  return 0;
}
```
In this example we allocate 100 integers of device storage but start writing at 1000 elements past the start of the array. cuda-memcheck has at least an equal chance of finding the exact line and instruction that is causing the program misbehavior as it does of hanging your X session:
```
$ nvcc \
  -std=c++11 --expt-extended-lambda -lineinfo \
  -gencode arch=compute_52,code=sm_52 
  -I ../moderngpu/src 
  -o bad_stores 
  bad_stores.cu

$ cuda-memcheck bad_stores

========= Invalid __global__ write of size 4
=========     at 0x000000e8 in /home/sean/projects/foo/bad_stores.cu:13:
_ZN4mgpu16launch_box_cta_kINS_15launch_params_tILi128ELi1ELi1ELi...
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x706389c40 is out of bounds
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 (cuLaunchKernel + 0x2c5) [0x472225]
=========     Host Frame:bad_stores [0x180a1]
=========     Host Frame:bad_stores [0x328b3]
=========     Host Frame:bad_stores [0x2f43]
=========     Host Frame:bad_stores [0x2e0d]
...
```
In this case, cuda-memcheck delivers precisely the goods. There is indeed an invalid global write of size 4 (our `index * index` integer). The tool even tells us that it's at bad_stores.cu line 13.
```
$ cuobjdump -sass bad_stores
        code for sm_52
                Function : _ZN4mgpu16launch_box_cta_kINS_15launch_params_tIL...
        .headerflags    @"EF_CUDA_SM52 EF_CUDA_PTX_SM(EF_CUDA_SM52)"
                                                            /* 0x001cfc00e22007f6 */
/*0008*/               MOV R1, c[0x0][0x20];                /* 0x4c98078000870001 */
/*0010*/               S2R R3, SR_CTAID.X;                  /* 0xf0c8000002570003 */
/*0018*/               S2R R4, SR_TID.X;                    /* 0xf0c8000002170004 */
                                                            /* 0x001fc400fec20ff6 */
/*0028*/               IADD32I R0, R3.reuse, 0x1;           /* 0x1c00000000170300 */
/*0030*/               SHL R0, R0, 0x7;                     /* 0x3848000000770000 */
/*0038*/               IMNMX R2, R0, c[0x0][0x140], PT;     /* 0x4c21038005070002 */
                                                            /* 0x001fb400fcc017f5 */
/*0048*/               LOP32I.AND R0, R4, 0x7f;             /* 0x0400000007f70400 */
/*0050*/               ISCADD R2, -R3, R2, 0x7;             /* 0x5c1a038000270302 */
/*0058*/               ISETP.GE.AND P0, PT, R0, R2, PT;     /* 0x5b6d038000270007 */
                                                            /* 0x081fd800fec007fd */
/*0068*/           @P0 EXIT;                                /* 0xe30000000000000f */
/*0070*/               ISCADD R0, R3, R0, 0x7;              /* 0x5c18038000070300 */
/*0078*/               IADD32I R2, R0.reuse, 0x2710;        /* 0x1c00000271070002 */
                                                            /* 0x001fc400fea207f1 */
/*0088*/               SHL R3, R2.reuse, 0x2;               /* 0x3848000000270203 */
/*0090*/               SHR R5, R2, 0x1e;                    /* 0x3829000001e70205 */
/*0098*/               IADD R4.CC, R3, c[0x0][0x148];       /* 0x4c10800005270304 */
                                                            /* 0x001fc840fe8007e1 */
/*00a8*/               XMAD R2, R0, R0, RZ;                 /* 0x5b007f8000070002 */
/*00b0*/               XMAD.MRG R3, R0.reuse, R0.H1, RZ;    /* 0x5b007fa800070003 */
/*00b8*/               IADD.X R5, R5, c[0x0][0x14c];        /* 0x4c10080005370505 */
                                                            /* 0x001fc800fe6007f1 */
/*00c8*/               XMAD.PSL.CBCC R0, R0.H1, R3.H1, R2;  /* 0x5b30011800370000 */
/*00d0*/               MOV R2, R4;                          /* 0x5c98078000470002 */
/*00d8*/               MOV R3, R5;                          /* 0x5c98078000570003 */
                                                            /* 0x001ffc00ffe000f1 */
/*00e8*/               STG.E [R2], R0;                      /* 0xeedc200000070200 */
/*00f0*/               EXIT;                                /* 0xe30000000007000f */
/*00f8*/               BRA 0xf8;                            /* 0xe2400fffff87000f */
```
cuda-memcheck also identifies the illegal write at address 0x000000e8 in the program code. Because we used `-gencode arch=compute_52,code=sm_52` to generate Maxwell SASS, we can use `cuobjdump -sass` to display its disassembly. Here we see `STG.E [R2], R0` at program counter 00e8. This is indeed our errant instruction.

cuda-memcheck can be an excellent tool early on in your debugging process. You'll invariably revert to printf debugging for things more serious, but this tool provides a less scatter-shot way to get started.