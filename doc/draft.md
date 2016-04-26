


Each pattern works with its arguments to define a deterministic _network_ of operations. A simple pattern like [`transform`](#transform) has a simple network: it maps the global ID of each thread to the `index` argument of the provided behavior. 

## High-level patterns

[`transform_reduce`](#transform_reduce) defines a more complicated network, calling the user's reduction operator recursively on the results of reduced tiles of inputs. The structure of the network is determined by the count of the reduction but is independent of the user-supplied behaviors.

[`transform_lbs`](#transform_lbs) is an advanced pattern. By using dynamic searches it provides the behavior function with critical context (the segment and rank of the work-item within the segment) for each work-item while load-balancing work-items across the GPU. This network, while still deterministic, changes structure with the user's choice of segmentation.

## Defining behaviors

The simplest pattern is `mgpu::transform`.

**`transform.hxx`**
```cpp
template<
  int nt = 128,         // Number of threads in the thread block.
  int vt = 1,           // Grain size for each thread.
  typename func_t,      // Function called for each work-item.
  typename... args_t    // Auxiliary arguments.
>
void transform(func_t f, size_t count, context_t& context, args_t... args);
```

The user provides a behavior functor or lambda implementing `void operator()(int index)`. This behavior is invoked once per work-item, which range from 0 to `count`. This is a standard _map_ pattern, and is included in every general--purpose CUDA library. 

#### Example
The behavior uses C++ lambda closure to capture the arguments it needs from the surrounding scope. Lambdas must be tagged with `MGPU_DEVICE` (a macro for `__device__`) when they are created in host code and invoked on device code.
```cpp
// Square each element in x and store it into y.
void square_all(const float* x, int count, float* y, context_t& context) {
  auto k = [=]MGPU_DEVICE(int index) {
    y[index] = x[index] * x[index];
  };
  transform(k, count, context);
}
```
For compactness we can also define the behavior lambda without assigning it to a variable.
```cpp
void square_all(const float* x, int count, float* y, context_t& context) {
  transform([=]MGPU_DEVICE(int index) {
    y[index] = x[index] * x[index];
  }, count, context);
}
```
`transform` has a trivial implementation but is nonetheless helpful. It simply launches one thread per work-item and passes 
```cpp 
index = threadIdx.x + blockDim.x * blockIdx.x
```
to the behavior function. 

#### Example
Consider a conversion from AOS (array of struct) to SOA (struct of array). We use the _transform_ pattern to invoke the _behavior_ for each element. The behavior simply constructs a struct on the four array elements and stores it to the result array.
```cpp
struct vec_t {
  float x, y, z, w;
};
void arrays_to_struct(const float* x, const float* y, const float* z, 
  const float* w, int count, vec_t* v, context_t& context) {
  transform([=]MGPU_DEVICE(int index) {
    v[index] = vec_t { x[index], y[index], z[index], w[index] };
  }, count, context);
}
```
This behavior seems straight-forward, but we can use it with almost every pattern, providing an ability to specialize very complicated kernels with little effort and perfect clarity.

## transform_reduce
_Reduction_ is the next pattern that all CUDA libraries implement. This one contains some logic--a network that adds together all inputs in parallel. `mgpu::reduce` is specialized on two behaviors: a function that, given an index, returns the value of the item to be reduced; and a function that reduces two argument values into one.

**`kernel_reduce.hxx`**
```cpp
template<
  typename launch_arg_t = empty_t,  // optional launch_box overrides default.
  typename func_t,         // implements type_t operator()(int index).
  typename output_it,      // iterator to output. one output element.
  typename op_t            // reduction operator implements
                           //   type_t operator()(type_t left, type_t right).
>
void transform_reduce(func_t f, int count, output_it reduction, op_t op, 
  context_t& context);
```

#### Example
This sample shows both behaviors for reduce. The _transform_ behavior returns the value to be reduced given an index. In this case, the product of an element from `a` with an element from `b`. The _reducer_ behavior folds together two argument values. In this case it returns the sum. The final scalar reduction is stored into `c`.
```cpp
// Add up the element-wise products of a and b and store into c.
void dot_product(const float* a, const float* b, float* c, int count,
  context_t& context) {
  auto transformer = [=]MGPU_DEVICE(int index) {
    return a[index] * b[index];
  };
  auto reducer = []MGPU_DEVICE(float lhs, float rhs) -> float {
    return lhs + rhs;
  };
  reduce(transformer, count, c, reducer, context);
}
```
Addition is a very common reducer, so a _functor_ type `mgpu::plus_t<>` is defined as a convenience. `multiplies_t<>`, `maximum_t<>` and `minimum_t<>` are also provided in **`operators.hxx`** for use with `reduce`. 
```cpp
// Add up the element-wise products of a and b and store into c.
void dot_product(const float* a, const float* b, float* c, int count,
  context_t& context) {
  reduce([=]MGPU_DEVICE(int index) {
    return a[index] * b[index];
  }, count, c, plus_t<float>(), context);
}
```

## transform_scan

## transform_segreduce
Many libraries have a form of segmented reduction pattern, but nothing else provides as much for general segmented algorithms as moderngpu. 

**`kernel_segreduce.hxx`**
```cpp
template<
  typename launch_arg_t = empty_t, // provides constants (nt, vt, vt0).
  typename func_t,         // implements type_t operator()(int index).
  typename segments_it,    // segments-descriptor array.
                           //   specificies starting offset of each segment.
  typename output_it,      // output iterator. one output per segment.
  typename op_t,           // reduction operator implements
                           //   type_t operator()(type_t a, type_t b)
  typename type_t
>
void transform_segreduce(func_t f, int count, segments_it segments, 
  int num_segments, output_it output, op_t op, type_t init, 
  context_t& context);
```


## transform_lbs

**`kernel_load_balance.hxx`**
```cpp
template<
  typename launch_arg_t = empty_t, // provides (nt, vt, vt0)
  typename func_t,         // load-balancing search callback implements
                           //   void operator()(int index,   // work-item
                           //                   int seg,     // segment ID
                           //                   int rank,    // rank within seg
                           //                   tuple<...> cached_values).
  typename segments_it,    // segments-descriptor array.
  typename tpl_t           // tuple<> of iterators for caching loads.
>
void transform_lbs(func_t f, int count, segments_it segments, 
  int num_segments, tpl_t caching_iterators, context_t& context);

// version of transform_lbs without caching iterators
template<
  typename launch_arg_t = empty_t, 
  typename func_t,         // load-balancing search callback implements
                           //   void operator()(int index, int seg, int rank).
  typename segments_it
>
void transform_lbs(func_t f, int count, segments_it segments, 
  int num_segments, context_t& context);  
```


## lbs_segreduce

**`kernel_segreduce.hxx`**
```cpp
template<
  typename launch_arg_t = empty_t, // provides (nt, vt, vt0)
  typename func_t,         // load-balancing search callback implements
                           //   type_t operator()(int index,   // work-item
                           //                     int seg,     // segment ID
                           //                     int rank,    // rank within seg
                           //                     tuple<...> cached_values).
  typename segments_it,    // segments-descriptor array.
  typename tpl_t,          // tuple<> of iterators for caching loads.
  typename output_it,      // output iterator. one output per segment.
  typename op_t,           // reduction operator implements
                           //   type_t operator()(type_t a, type_t b).
  typename type_t
>
void lbs_segreduce(func_t f, int count, segments_it segments,
  int num_segments, tpl_t caching_iterators, output_it output, op_t op,
  type_t init, context_t& context);

// version of lbs_segreduce without caching iterators.
template<
  typename launch_arg_t = empty_t, 
  typename func_t,         // load-balancing search callback implements
                           //   type_t operator()(int index, int seg, int rank).
  typename segments_it, 
  typename output_it, 
  typename op_t,
  typename type_t
>
void lbs_segreduce(func_t f, int count, segments_it segments,
  int num_segments, output_it output, op_t op, type_t init, 
  context_t& context);
```

```cpp
void segmented_stddev(const float* x, int count, const int* segments,
  int num_segments, float* c, context_t& context) {

  // Find the total value of each segment.
  mem_t<float> totals(num_segments, context);
  segreduce(x, count, segments, num_segments, totals.data(), plus_t<float>(),
    0.0f, context);

  // Define an iterator that turns the segment total into the segment mean.
  float* totals_data = totals.data();
  auto mean = make_load_iterator<float>([=]MGPU_DEVICE(int index) -> float {
    float total = totals_data[index];
    int size = segments[index + 1] - segments[index];
    if(size > 0) total /= size;
    return total;
  });

  // Find the sum of squared differences in each segment.
  // We use a caching iterator to apply the segment size normalization once
  // per CTA and to broadcast the result to each thread that needs the 
  // mean.
  lbs_segreduce([=]MGPU_DEVICE(int index, int seg, int rank, tuple<float> u) {
    // u is a tuple holding the mean of the segment.
    return sq(x[index] - get<0>(u));
  }, count, segments, num_segments, make_tuple(mean), y, 
    plus_t<float>(), 0.0f, context);

  // Divide the sum of squared differences by the size of each segment
  // and sqrt for the standard deviation.
  transform([=]MGPU_DEVICE(int index) {
    int size = segments[index + 1] - segments[index];
    if(size > 0) y[index] = sqrt(y[index] / size);
  }, count, context);
}
```

## Debugging kernels

