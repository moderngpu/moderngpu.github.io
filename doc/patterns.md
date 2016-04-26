# Patterns and behaviors

moderngpu exposes functionality through templated _pattern_ functions. The user specializes these patterns with concrete _behaviors_, typically written as C++ lambdas. The pattern launches a kernel, computes a _context_ for each work-item, and invokes the behavior on each work-item, providing the corresponding context.

moderngpu provides patterns useful for many operations:

1. **cta_launch** - Launch a grid of cooperative thread arrays (CTAs) and pass the behavior function the index of the thread (threadIdx.x) and CTA (blockIdx.x).
2. **cta_transform** - Launch a grid of CTAs, but size them according to the number of work-items that can be processed given the architecture's specified _launch box_.
3. **transform** - A non-cooperative method. Invoke the behavior once for each work-item.
4. **transform_reduce** - Call a behavior once for each work-item and recursively reduce the return values with a user-provided reducer. This pattern enables array-wide sum, max, and min operations.
5. **transform_scan** - Like `transform_reduce`, but computes a reduction for each interval from the start of the array to each element. For an addition operator this is the [prefix sum](https://en.wikipedia.org/wiki/Prefix_sum) operation. 
6. **transform_lbs** - A vectorized and load-balanced `transform` implemented using _load-balancing search_. The caller specifies the geometry of the problem with a segments descriptor array and the behavior is invoked for each work-item, providing both the ID of the segment the work-item belongs to and its rank within the segment. This is the signature pattern of moderngpu.
7. **lbs_segreduce** - Fold together all values in each segment and return one reduction value per segment. The behavior is invoked with the segment ID and rank of each work-item. This pattern makes for consistent performance for simultaneous processing of many irregularly-shaped problems.
8. **transform_compact** - An efficient two-pass pattern for conditionally selecting and compacting elements of an array.
9. **lbs_workcreate** - Dynamic work-creation accelerated with load-balancing search. This is a two-pass pattern. On the upsweep the pattern returns the number of output work-items to stream for each input work-item. On the downsweep, the pattern encodes parameters for each work-creating segment. This pattern solves many problems that CUDA Dynamic Parallelism was intended to solve, but with exact load-balancing and requiring no special hardware mechanisms.

moderngpu also includes traditional bulk-synchronous parallel general-purpose functions. Most of these can be adapted to the pattern-behavior model with the use of _lambda iterators_, which wrap lambda behavior functions in the interface of pointers.

* **reduce** and **scan** are the iterator-oriented equivalents of `transform_reduce` and `transform_scan`.
* **merge**, **bulk_remove** and **bulk_insert** are general-purpose array construction functions.
* **mergesort** is a basic array-wide sort. **segmented_sort** is an advanced mergesort that sorts keys and values within segments defined by a segments descriptor array. Thanks to a novel short-circuiting feature, `segmented_sort` actually improves in performance as the number of segments to sort increases.
* **sorted_search** is a vectorized sorted search. A binary search looks for a needle in a sorted haystack. The vectorized sorted search looks for an array of sorted needles in a sorted haystack. A problem with _n_ needles and _m_ haystack items requires _O(n log m) operations to binary search but only _O(n + m) operations to sorted search. This is a critical function for implementing database operations.
* **inner_join** is a relational join operator. It's a demonstration of the power of combining vectorized sorted search with the load-balancing search and useful in its own right.

## Patterns and arguments

**no_capture.cu**
```cpp
#include <moderngpu/transform.hxx>

using namespace mgpu;

int main(int argc, char** argv) {
  standard_context_t context;

  transform([]MGPU_DEVICE(int index) {
    printf("Hello from thread %d\n", index);
  }, 10, context);
  context.synchronize();

  return 0;
}
```
```
Hello from thread 0
Hello from thread 1
Hello from thread 2
Hello from thread 3
Hello from thread 4
Hello from thread 5
Hello from thread 6
Hello from thread 7
Hello from thread 8
Hello from thread 9
```
This simple example defines a behavior function which takes one argument and prints it to the console. It is combined with the `transform` pattern, which invokes it once on each of the 10 input work-items. The behavior is a device-tagged lambda. We must mark it with `MGPU_DEVICE` (or `__device__`) after the capture list `[]` and before the arguments list `(int index)`. Under CUDA 7.5 you must compile with `--expt-extended-lambda` to enable device-tagged lambdas.

**lamdda_capture.cu**
```cpp
#include <moderngpu/transform.hxx>

using namespace mgpu;

int main(int argc, char** argv) {
  standard_context_t context;

  // Define an exponential argument on the host.
  float a = 22.102, b = 1.815f;

  // Allocate device memory.
  int count = 1000;
  float* output;
  cudaMalloc((void**)&output, sizeof(float) * count);
  
  transform([=]MGPU_DEVICE(int index) {
    output[index] = a * exp(b * index);
  }, count, context);
  
  // Do something with the output.
  // xxx

  cudaFree(output);
  return 0;
}
```
This example computes `a * exp(b * index)` into device memory for 1000 indices. It uses lambda closure to capture three arguments from the host: coefficients `a` and `b` and the pointer to device memory `output`. Use the capture list `[=]` to capture any arguments from the host by value. Because host and device codes reside in different address spaces, arguments cannot be captured (or passed) by reference.

## Variadic arguments

Lambda capture is the most convenient but not the only way to pass arguments from the host into kernels. 

**variadic_parameters.cu**
```cpp
#include <moderngpu/transform.hxx>

using namespace mgpu;

int main(int argc, char** argv) {
  standard_context_t context;

  typedef launch_params_t<128, 16> launch_t;

  // Define the behavior function. It takes an int* pointer 
  // after the required (tid, cta) indices. This pointer will
  // be passed as the last argument to cta_launch.
  auto f = [=]MGPU_DEVICE(int tid, int cta, int* cta_temp) {
    // Each CTA does something special on cta_temp.
    if(!tid) cta_temp[cta] = cta;
  };

  // Compute occupancy.
  int occ = occupancy<launch_t>(f, context, (int*)nullptr);

  // Allocate one item per CTA.
  mem_t<int> cta_temp(occ, context);

  // Invoke the pattern and pass it the cta_temp pointer variadically. 
  cta_launch<launch_t>(
    f,                 // Handle of the behavior function.
    occ,               // Number of CTAs to launch.
    context,           // Reference to context object. Holds the stream.
    cta_temp.data()    // Parameters to pass to the behavior.
  );

  return 0;
}
```
Most patterns support passing additional arguments to the behavior function variadically. These parameters are passed at the end of patterns like `transform`, `cta_launch` and `transform_lbs`, copied to the device, and passed again to the behavior function on the device side. 

Why provide this second mechanism of parameter passing? Sometimes lambda capture cannot be used. _Occupancy_ is the maximum number of CTAs that can be executed concurrently on the device. The occupancy of a kernel depends on the definition of that kernel, as different kernels use different amounts of SM resources, which are only available in finite quantities. 

Let's say we want to allocate memory based on occupancy and pass this memory into a kernel. We need to define the kernel to compute the occupancy, then allocate memory given the occupancy, then pass the allocated memory into the kernel. This isn't possible with lambda capture, as the pointer to this allocated memory needs to be available for the lambda definition, on which the occupancy calculator depends.

Although this is a fair motivating case for the use of variadic parameters, in some cases they have a tremendous advantage over captured parameters. 

## Restricted arguments

Parameters that are passed by lambda capture are handled directly by the CUDA compiler; moderngpu doesn't get its hands on those. Parameters passed variadically, however, are available for transformation by the library.

moderngpu tags all pointer-type variadic pattern arguments with the **\_\_restrict\_\_** qualifier. \_\_restrict\_\_ is an assertion by the programmer that all qualified pointers reference non-overlapping memory, that is, the pointers are not _aliased_. When pointers are aliased, the compiler issues cannot re-order store instructions, and this leads to sequential dependencies which may increase latency. 

By tagging all pointers with \_\_restrict\_\_, the compiler can _overlap_ loads by factoring them to the top, issuing each load before the preceding one has returned its data. Overlapping of IO increases _instruction-level parallelism_ (ILP) and decreases kernel latency. 

In this example we call the `transform` pattern with a block size of 128 threads and a grain size of 4. That is, for a full block, 4 work-items are processed per thread. Using a large grain size opens the possibility for scheduling overlapped loads and realizing high ILP. 
```cpp
  // Pass by capture.
  transform<128, 4>([=]MGPU_DEVICE(int index) {
    output[index] = 2 * input[index];
  }, 0, context);
```
```
/*01b0*/               LDG.E R0, [R2];                      /* 0xeed4200000070200 */
/*01b8*/               SHL R6, R0, 0x1;                     /* 0x3848000000170006 */
                                                            /* 0x011fc8000e4001f2 */
/*01c8*/               STG.E [R4], R6;                      /* 0xeedc200000070406 */
/*01d0*/               LDG.E R0, [R2+0x200];                /* 0xeed4200020070200 */
/*01d8*/               SHL R7, R0, 0x1;                     /* 0x3848000000170007 */
                                                            /* 0x011f88000e4001f2 */
/*01e8*/               STG.E [R4+0x200], R7;                /* 0xeedc200020070407 */
/*01f0*/               LDG.E R0, [R2+0x400];                /* 0xeed4200040070200 */
/*01f8*/               SHL R9, R0, 0x1;                     /* 0x3848000000170009 */
                                                            /* 0x011fc8000e4001f2 */
/*0208*/               STG.E [R4+0x400], R9;                /* 0xeedc200040070409 */
/*0210*/               LDG.E R0, [R2+0x600];                /* 0xeed4200060070200 */
/*0218*/               SHL R10, R0, 0x1;                    /* 0x384800000017000a */
                                                            /* 0x001ffc00ffe001f1 */
/*0228*/               STG.E [R4+0x600], R10;               /* 0xeedc20006007040a */
```
In the first usage of `transform`, the input and output pointers are passed by lambda capture. The moderngpu library has no chance to tag these with the \_\_restrict\_\_ qualifier, and so the compiler assumes pointer aliasing and evaluates the behavior function four times in sequence. The SM waits hundreds of cycles for each LDG instruction to return data, so it can shift it and store it back out. Unless occupancy is high, kernels with sequential load/store pairs are very often latency limited.
```cpp
  // Pass by argument. Pointers are treated as restricted.
  transform<128, 4>([]MGPU_DEVICE(int index, const int* input, int* output) {
    output[index] = 2 * input[index];
  }, 0, context, input, output);
```  
```
/*01b0*/               LDG.E.CI R6, [R2];                   /* 0xeed4a00000070206 */
/*01b8*/               LDG.E.CI R0, [R2+0x200];             /* 0xeed4a00020070200 */
                                                            /* 0x011fc800162000b1 */
/*01c8*/               LDG.E.CI R7, [R2+0x400];             /* 0xeed4a00040070207 */
/*01d0*/               LDG.E.CI R8, [R2+0x600];             /* 0xeed4a00060070208 */
/*01d8*/               SHL R6, R6, 0x1;                     /* 0x3848000000170606 */
                                                            /* 0x041fc0003e4087f0 */
/*01e8*/     {         SHL R9, R0, 0x1;                     /* 0x3848000000170009 */
/*01f0*/               STG.E [R4], R6;        }             /* 0xeedc200000070406 */
/*01f8*/     {         SHL R10, R8, 0x1;                    /* 0x384800000017080a */
/*0208*/               STG.E [R4+0x200], R9;        }       /* 0x0007c800fc4001f1 */
                                                            /* 0xeedc200020070409 */
/*0210*/               SHL R7, R7, 0x1;                     /* 0x3848000000170707 */
/*0218*/               STG.E [R4+0x400], R7;                /* 0xeedc200040070407 */
                                                            /* 0x001ffc00ffe001f1 */
/*0228*/               STG.E [R4+0x600], R10;               /* 0xeedc20006007040a */
```
The second usage of `transform` passes `input` and `output` into the `transform` pattern, and the pattern tags them with the \_\_restrict\_\_ qualifier passes them back to the lambda. Because a guarantee has been made that the pointers do not alias, the compiler factors the load instructions to the top, issuing all four at once. If a load takes 400 cycles, we now only wait for that duration once, rather than waiting for it four times sequentially.

The use of restricted pointers may have downsides, but these are mostly hypothetical. Because more values are loaded before being stored back out, register pressure is higher and occupancy may suffer. Still, the increase in ILP almost certainly outweighs the slight loss in thread-level parallelism resulting from diminished occupancy.

## Restricted arguments with advanced patterns

**interval_move** is an advanced operation with a simple implementation. It's a simple behavior for the _load-balancing search_ pattern **transform_lbs**, which performs a load-balanced and vectorized set of array copy operations. Each segment has a length specified by the segments descriptor array. Each segment also has a starting array in the output array (its _scatter_ offset) starting offset in the input array (its _gather_ offset). Because all work-items (i.e. elements to be copied) in each segment share the same scatter and gather offsets, and those offsets are encoded exactly once per segment, they can be loaded automatically in an optimized way by the load-balancing search pattern, bound together into a tuple, and passed to the behavior function. The behavior then adds the rank of the work-item (i.e. the offset of the element within the segment) to the segment's gather and scatter offsets and loads and stores to them, respectively.

```cpp
template<typename launch_arg_t = empty_t, 
  typename input_it, typename segments_it, typename scatter_it,
  typename gather_it, typename output_it>
void interval_move1(input_it input, int count, segments_it segments,
  int num_segments, scatter_it scatter, gather_it gather, output_it output, 
  context_t& context) {

  transform_lbs<launch_arg_t>(
    [=]MGPU_DEVICE(int index, int seg, int rank, tuple<int, int> desc) {
      output[get<0>(desc) + rank] = input[get<1>(desc) + rank];
    }, 
    count, segments, num_segments, make_tuple(scatter, gather), context
  );
}
```
```
/*13d0*/                   LDG.E R0, [R16];                 /* 0xeed4200000071000 */
/*13d8*/                   STG.E [R6], R0;                  /* 0xeedc200000070600 */
                                                            /* 0x0012c8087e400272 */
/*13e8*/                   LDG.E R2, [R18];                 /* 0xeed4200000071202 */
/*13f0*/                   STG.E [R8], R2;                  /* 0xeedc200000070802 */
/*13f8*/                   LDG.E R3, [R10];                 /* 0xeed4200000070a03 */
                                                            /* 0x0403c420164105fd */
/*1408*/                   STG.E [R4], R3;                  /* 0xeedc200000070403 */
/*1410*/                   LDG.E R12, [R12];                /* 0xeed4200000070c0c */
/*1418*/                   STG.E [R14], R12;                /* 0xeedc200000070e0c */
```

Because of the expressiveness of `transform_lbs`, the interval move function is implemented with a one-line behavior. This version uses lambda capture to access the input and output pointers. Although the results of this function are undefined when the input and output pointers reference overlapping memory (it would have to be written very carefully to give deterministic results), the compiler still generates code assuming pointer aliasing rules apply. Using a grain size of four, we see the four load/store pairs in the program's disassembly. This is still good code generation--much logic has been factored away from the IO and hoisted to the top of the code. But still, this sequence generates unnecessary dependencies, and the function will not perform as well as it could.

```cpp
template<typename launch_arg_t = empty_t, 
  typename input_it, typename segments_it, typename scatter_it,
  typename gather_it, typename output_it>
void interval_move2(input_it input, int count, segments_it segments,
  int num_segments, scatter_it scatter, gather_it gather, output_it output, 
  context_t& context) {

  transform_lbs<launch_arg_t>(
    []MGPU_DEVICE(int index, int seg, int rank, tuple<int, int> desc,
      input_it input, output_it output) {
      output[get<0>(desc) + rank] = input[get<1>(desc) + rank];
    }, 
    count, segments, num_segments, make_tuple(scatter, gather), context,
    input, output
  );
}
```
```
/*13d0*/                   LDG.E.CI R0, [R16];              /* 0xeed4a00000071000 */
/*13d8*/                   LDG.E.CI R2, [R18];              /* 0xeed4a00000071202 */
                                                            /* 0x010fc800162002b1 */
/*13e8*/                   LDG.E.CI R3, [R10];              /* 0xeed4a00000070a03 */
/*13f0*/                   LDG.E.CI R12, [R12];             /* 0xeed4a00000070c0c */
/*13f8*/                   STG.E [R6], R0;                  /* 0xeedc200000070600 */
                                                            /* 0x0003c800fe6084f1 */
/*1408*/                   STG.E [R8], R2;                  /* 0xeedc200000070802 */
/*1410*/                   DEPBAR.LE SB5, 0x1;              /* 0xf0f0000034170000 */
/*1418*/                   STG.E [R4], R3;                  /* 0xeedc200000070403 */
                                                            /* 0x001ffc00ffe100f1 */
/*1428*/                   STG.E [R14], R12;                /* 0xeedc200000070e0c */
```
This implementation differs only that the input and output parameters are passed variadically to `transform_lbs`. The compiler can now re-order the memory operations, which it does, overlapping all four loads to the input array.

Always look for opportunities to overlap loads from global memory. The moderngpu patterns aggressively overlap for data generated and consumed internally, and the automatic \_\_restrict\_\_ promotion on variadically-passed pointer arguments allows access to behavior data to be similarly optimized.
