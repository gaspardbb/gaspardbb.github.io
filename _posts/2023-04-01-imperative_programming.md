---
layout: distill
title: From Python to Julia
description: Imperative vs declarative programming languages for scientific computing
giscus_comments: false
date: 2023-03-30


authors:
  - name: Gaspard Beugnot

# bibliography: 2018-12-22-distill.bib

toc:
  - name: My use of Python and why I switched 
  - name: Enters Julia
  - name: Is an imperative language really the solution?
  - name: Conclusion
    subsections:
    - name: What should you choose?
---

Python is a highly prevalent programming language, widely known for its user-friendly syntax and extensive library ecosystem, particularly for machine learning and web development. Because of this remarkable combination, a large part of the Python community does not have a computer science background. Personally, as an applied mathematics practitioner, I adopted Python for course projects about five years ago, primarily as a means to implement algorithms derived from academic literature. At that time, I did not question the underlying qualities of this tool - its design principles, limitations, how it actually _worked_.

This discussion is for Python users who don't have a strong background in computer science but often use Python and its scientific computing libraries. It aims to provide some basic programming concepts while also highlighting the language's limitations. I will explain why I opted for a different programming language and the lessons I learned, particularly in machine learning implementation.

In my opinion, reflecting on our programming practices provides a foundation for better coding approaches that result in enhanced productivity. Moreover, it helps us understand how the tools we use can affect the way we conduct our research and work. Indeed, programming languages are tools that reflect the way knowledge is organized through user-contributed libraries. Given the ever more complex science we produce, this is a subject I'm passionate about. 

_The text features collapsible boxes containing more advanced concepts. They are not necessary to understand my main message, feel free to skip them!_


## My use of Python and why I switched 

Python has a lot of amazing libraries that make it really powerful, like Numpy for scientific computing, Scikit-learn for machine learning, and Pytorch for automatic differentiation and GPU computing.  It's pretty incredible that you can get these tools for free and use them to make complex models with just a few lines of code, even without much computer knowledge. And I'm not just talking about the newest and fanciest language models (even though Llama, Meta's contender to GPT4, is just [~200LOC long to describe](https://github.com/facebookresearch/llama/blob/main/llama/model.py "Llama's repo")). Just look at [scikit-learn example's page](https://scikit-learn.org/stable/auto_examples/index.html "Scikit Learn page"), which showcases dozens of ML algorithms that you can run in a few lines of code.

Yet the more I understood it, the more I became annoyed by some part of the language; I decided to switch when I had to deal with polynomials, in an ongoing project aiming at minimizing them globally with an optimality certificate -- a project in the line of [Lasserre's Sum of Square hierarchy](https://en.wikipedia.org/wiki/Sum-of-squares_optimization "SoS on Wikipedia") and their recent extension, [Kernel Sum of Square](https://arxiv.org/abs/2012.11978 "Kernel SoS on arXiv"). Consider for instance polynomial evaluation. Basically, you can define a polynomial $$ f $$ in $$ \mathbb{R}^d $$ as a collection of $$ n $$ coefficients of degree $$ p_i \in \mathbb{N}^d $$, with $$ i \in (1, n) $$. That is, 

$$
\forall x \in \mathbb{R}^d, ~~ f(x) = \sum_{i=1}^n \alpha_i \prod_{k=1}^d x_k^{p_{ik}}
$$

To evaluate this function in Python, you could use broadcasting; you would define an array $$ p $$ of size $$ (n, d) $$ and do something like 

```Python
np.sum(alpha * np.prod(x ** p, axis=1), axis=0)  # f(x)
```

{% details What is broadcasting? %}
Broadcasting describes how NumPy (or other tensor librairies such as PyTorch or Jax) *vectorizes* operations on arrays. A simple illustration of vectorization is the following. 
When I started Python (we've all been beginners once right), I wrote ``for`` loops to iterate through my arrays, and I was happy with that. For instance, to compute 

$$
\sum_{i=1}^n f(x_i)^2
$$

I would do 

```python
s = 0
for i in range(n)
    s += f(x[i]) ** 2
```

Then my code took a very long time to process modest image batch of $$ 100 \times 100 $$ pixels, and I learned that iterating element by element on Numpy arrays was a terrible practice. A much more efficient way to compute the sum is to write 

```python
s = np.sum(f(x) ** 2)
```

That is what [vectorization](https://en.wikipedia.org/wiki/Array_programming "Vectorization on Wikipedia") is about: instead of processing elements one at a time, we write operations on *all* the elements of an array at once. This results in much faster computation as the code uses efficient C librairies instead of Python native ones. In the code snippet above, $$ x \mapsto f(x)^2 $$ is applied on all the elements of `x` at once. 

Indeed, Python is an interpreted language which is designed for ease of use rather than speed. Yet, good performances can be achieved by calling code in C, a much more powerful (hence difficult to master) language. Hopefully, awesome libraries like Numpy enables easy prototyping while preserving great C-like performances. It comes as a cost, such as not being able to iterate through an array element by element. Using Python without those libraries is totally unrealistic for real project (take the results of [such study](https://greenlab.di.uminho.pt/wp-content/uploads/2017/09/paperSLE.pdf "What are the most environmental-friendly PL?") with a pinch of salt).
{% enddetails  %}

However, this solution is unsatisfactory for at least 2 reasons:

1. The most efficient and stable way to evaluate a polynomial is through [Horner's method](https://en.wikipedia.org/wiki/Horner%27s_method "Horner's method on wikipedia"), which uses a recursion which cannot be implemented efficiently in Python. _Admittedly, this is not a big issue for a code used in research_.
2. Most importantly, we might want to express this polynomial in another basis. This require another representation that can't be efficiently implemented with Numpy.

{% details Not convinced? More on polynomial representation %}
We may want to define $$ p $$ in a basis $$ B_j $$ of polynomial of degree $$ j $$ in $$ \mathbb{R} $$. $$ f $$ is then written 

$$
\forall x \in \mathbb{R}^d, ~~ f(x) = \sum_{i=1}^n \alpha_i \prod_{k=1}^d \sum_{j=0}^{p_{ik}} b_{ikj} B_j(x_k)
$$

Implementing this _efficiently_ in Python is far from easy. Indeed, broadcasting acts on arrays, and arrays have the same size in all dimension. Here, the coefficients $$ b_{ikj} $$ have size $$ n, d $$ for the first two dimensions $$ i, k $$. However, the third dimension has size $$ p_{ik} $$ which is not constant across all the $$ (i, k) $$. A workaround would be to define $$ b $$ as an array of size $$ (n, d, p_{\max}) $$ where $$ p_{\max} = \max_{i, k} p_{ik} $$, and pad $$ b $$ with zeros. However, this allocates unnecessary memory. 

To obtain an efficient implementation, one observes that if $$ B_j $$ is an orthogonal polynomial, it satisfies a [2nd ordre relation](https://en.wikipedia.org/wiki/Favard%27s_theorem "Favard's Theorem") of the form 

$$ 
B_j = \alpha_j(x) B_{j-1} + \beta_j B_{j-2}.
$$

Ordinary examples of orthogonal polynomial includes the canonical basis $$ B_j = x^j $$, the [Chebychev series](https://en.wikipedia.org/wiki/Chebyshev_polynomials "Chebychev polynomials") (very useful for function approximation), the [Hermite basis](https://en.wikipedia.org/wiki/Hermite_polynomials "Hermite polynomials") (used e.g. as eigenfunction of the Fourier Transform), etc. 

Such relation enables a very efficient and stable evaluation of 

$$
   \sum_{j=0}^{p} b_{j} B_j(x)
$$ 

with Clenshaw's algorithm, whose Honner's method is a special case.

On top of that, these 2nd order relation enables to easily change the basis of $$ f $$. For instance, we can recenter a polynomial on the ball of center $$ m $$ and radius $$ r $$ by considering $$ x \mapsto f((x-m)/r) $$ in the basis $$ B_j((x-m)/r) $$.

Anyways, it was my use case and switching to Julia made all this implementation very easy. 
{% enddetails  %}

## Enters Julia
[Julia](https://julialang.org/ "Julia's webpage")'s main aspect is being a compiled language which feels like an interpreted language. Compiled languages are languages which go through a two steps process: they are first translated into machine code then executed; on the other hand, interpreted languages are executed directly. Compiled languages are faster, while interpreted languages are easier to prototype with. I won't go into more details about the distinction between interpreted and compiled language (and some might argue that [compiled and interpreted languages is two ways of saying tomato](https://tratt.net/laurie/blog/2023/compiled_and_interpreted_languages_two_ways_of_saying_tomato.html "Tratt's blog")), but Julia is designed to find the best balance for researchers who need both speed for heavy numerical computations and quick prototyping to test models. Julia achieves this by using "just in time" compilation, which means that each function is compiled the first time it's called.

Coming from Python, one of the thing I love most about Julia is being able to write as many `for` loops as I want without having to worry about broadcasting. For instance, evaluating efficiently the polynomial I mentioned before is straightforward. Just write `for` loops. 

```julia
"Evaluates a multidimensional polynomial"
function (f::MDCanonicalExpanded{T})(x::AbstractVector{T}) where {T}
    r = zero(T)
    for (k, aₖ) ∈ enumerate(coefficients(f))
        ∏ = one(T)
        for i ∈ 1:dim(f)
            # realpolys(f)[k, i] contains all the coefficients b_ki
            ∏ *= clenshaw(realpolys(f)[k, i], x[i])
        end
        r += aₖ * ∏
    end
    r
end
```

Another great thing about Julia is that you don't have to deal with the "_two languages barrier_". In Python, when you work with moderately complex programs, you often have to rely on external libraries written in C because of Python's poor performance. Although these libraries often exist thanks to the large community, if you want to change something in the library or understand how it works better, you need to know C, which can be frustrating.
In contrast, since pure Julia code is performant, you can usually see the source code for almost everything. In other words, Julia relies less on external libraries for scientific computation. For instance, you can see the implementation of common functions such as the trigonometric functions `sin` and `cos`. Have you ever wonder how your computer evaluates them? Well, simply through Taylor expansion. To evaluate `sin(x)`, Julia [firsts checks the domain](https://github.com/JuliaLang/julia/blob/17cfb8e65ead377bf1b4598d8a9869144142c84e/base/special/trig.jl#L29 "Julia's source code") of `x`, then [uses a polynomial approximation](https://github.com/JuliaLang/julia/blob/17cfb8e65ead377bf1b4598d8a9869144142c84e/base/special/trig.jl#L69 "Julia's source code") which is valid up to machine precision on $$ [\pi/4, \pi/4] $$. I reproduce the code here:

```julia
# Coefficients in 13th order polynomial approximation on [0; π/4]
#     sin(x) ≈ x + S1*x³ + S2*x⁵ + S3*x⁷ + S4*x⁹ + S5*x¹¹ + S6*x¹³
# D for double, S for sin, number is the order of x-1
const DS1 = -1.66666666666666324348e-01
const DS2 = 8.33333333332248946124e-03
const DS3 = -1.98412698298579493134e-04
const DS4 = 2.75573137070700676789e-06
const DS5 = -2.50507602534068634195e-08
const DS6 = 1.58969099521155010221e-10

@inline function sin_kernel(y::Float64)
    y² =  y*y
    y⁴ =  y²*y²
    r  =  @horner(y², DS2, DS3, DS4) + y²*y⁴*@horner(y², DS5, DS6)
    y³ =  y²*y
    y+y³*(DS1+y²*r)
end
```
I find it amazing not to be limited to using libraries written in another language. To some extent, scientific Python is all about manipulating libraries written in C, which is not the case for Julia, where everything is written in Julia. For instance, you don't have to set aside an algorithm simply because the computation of the `erf` function is too costly, or because you want to use the [Fadeeva function](https://en.wikipedia.org/wiki/Faddeeva_function "Fadeeva function on wikipedia") which is implemented almost nowhere. You can simply compute a Taylor approximation and write it in Julia, and _specialize it_ to your use case. In Python, such low-level manipulation requires C code and are not accessible to users who don't speak C. All that illustrates the two languages-barrier. 

Another useful aspect, but admittedly of minor importance is unicode support in Julia. That is, you can write code such as 
```julia
xₜ₊₁ = xₜ - η * ∇f(xₜ)
```
and that is perfectly fine. This provides a more direct conversion from math formula to computer code. 

For instance, Julia was great for polynomial evaluation I already mentioned, but also for other complex function evaluation involving multiple nested loops, or simply rejection sampling. This can be done in Python with broadcasting, but it's much easier in Julia! This sample an element from `proba` which is not in `set`.
```julia
function samplewithrejection(proba, Ω)
    ntries, maxtries = 0, 1000
    while true
        s = sample1from(proba)
        sample ∉ Ω && return s
        ((ntries += 1) > maxtries) && break
    end
    error("Reached max number of rejection sampling (1000)")
end
```

All this was great, but as my comprehension of Julia grew I started encountering real issues with the language.

{% details More details about Julia %}
Julia is relatively easy to master, especially if you're at ease with Numpy or any other tensor library. The paradigm Julia relies on – [multiple dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY&themeRefresh=1 "A video on multiple dispatch") – ends up being quite intuitive, enabling a whole range of coding practice from object oriented to functional. The fact that function [do not automatically broadcast](https://docs.julialang.org/en/v1/manual/mathematical-operations/#man-dot-operators "Julia's docs") is a great choice in my opinion: in Julia, if `x` is an array, `exp(x)` returns an error. Vectorization is performed when a dot `.` is added, i.e. `exp.(x)` applies `exp` element wise to `x`. After a while, it seems natural to distinguish between the two and `np.exp(x)` feels weird, even more as `exp(x)` could refer to the [operator exponential](https://en.wikipedia.org/wiki/Matrix_exponential "Matrix exponential on Wikipedia") if `x` is a square matrix.

One thing I took a while to master is [type stability](https://docs.julialang.org/en/v1/manual/performance-tips/#Write-%22type-stable%22-functions "Julia's docs") which can be quite hard to debug. 
{% enddetails  %}

## Is an imperative language really the solution?

I was very happy writing all the `for` loops I needed until I learned that… There are not _that_ fast. Indeed, to quantify the speed of a program one does not stop at whether the underlying language is compiled or interpreted, nor to which target it compiles to; optimization is still crucial for precise and frequently executed operations. I was initially excited to have the freedom to write any algorithm in Julia, but soon realized that my naive implementations weren't always as fast as they could be.

Let me give you one example, for which I won't go into too much details. Say you want to perform matrix multiplication, i.e.
$$
A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}, ~~ C_{ij} = \sum_{k = 1}^n A_{ik} B_{kj}, ~~ (i, j) \in (1,m) \times (m, p),
$$

which you do writing a `for` loop, e.g.

```julia
for i ∈ 1:m, j ∈ 1:p
    s = zero(T)  # T is the type (e.g. float32) of the arrays
    for k ∈ 1:n
        s += A[i, k] * B[k, j]
    end
    C[i, j] .= s
end
```

Naturally, crafting the loop requires more effort than utilizing the `C = A * B` shortcut, but let us set aside that notion for the sake of this discussion.
Does the resulting program perform as quickly as invoking `C = A * B`? I previously believed that it was approximately so, but that is not the case. Not even close. The loop's slower performance stems from the fact that the specific structure of the loop enables many optimizations, such as arranging data in a manner that optimizes the microprocessor's allocation of ressources. Packages such as [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl "LoopVectorization.jl on GitHub") try to alleviate this issue, but that is another layer of knowledge required to achieve a flawless implementation. However, using `C = A * B` allows the compiler to recognize that it is executing a matrix multiplication and call the [`gemm` routine](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3 "gemm on Wikipedia"), a function that has been heavily optimized over the past few decades.

This example is somewhat contrived, but it illustrates the fact that Numpy's broadcasting requirements compel its users to write better code. It moves us away from a term that was coined on [HackerNews](https://news.ycombinator.com/item?id=33968572 "HackerNews thread") called "Potato Programming" ([one potato, two potato, three potato](https://en.wikipedia.org/wiki/One_potato,_two_potato "reference on Wikipedia")), which refers to algorithms that handle individual elements one at a time, prohibiting the compiler from parallelizing computations and thereby hurting performance. Conversely, vectorization inherently describes a sequence of operations performed on all the elements in an array. This is especially true when using GPU programming, which can produce immense speedups but necessitates the "all elements at once" methodology. **Eventhough some aspects of NumPy are not practical, it forces the programmer to embrace better coding practice**. By avoiding Python `for` loops, because they are so slow, we write efficient code _no matter the programming language we use_.

A practical example is computing pairwise distances. If $$ X \in \mathbb{R}^{n \times d} $$ is a matrix containing $$ n $$ samples described by $$ d $$ features, we might wish to compute the distance between sample $$ i $$ and $$ j $$, that is $$ \lVert X_{:, i} - X_{:, j} \rVert^2 $$ for all $$ i, j \in (1, n) $$. A broadcasted approach would look like `np.sum(np.abs2(X - X.T), axis=1)`. This formulation computes twice as many operations as a loop-based approach (since $$ d_{ij} = d_{ji} $$), but it is ten times faster in a typical implementation.

{% details Code sample for computing pairwise distances %}
I spent the same time to write each function. I simply checked that they compiled and that they were type stable. I obtained<d-footnote>In Julia, arrays are stored in memory in column-major order, hence here `X` is the transpose of the previous example.</d-footnote>:
```julia
# Those function write the pairwise distances in the array R
pw_broadcasted!(R, X) = (R .= sum(abs2.(reshape(X, 1, size(X)...) .- X'), dims=2)[:, 1, :])
pw_for!(R, X) = begin
    @inbounds for i ∈ axes(X, 2)
        for j ∈ i:size(X, 2)
            d = sum(abs2.(X[:, i] - X[:, j]))[]
            R[i, j] = d
            R[j, i] = d
        end
    end
end

using BenchmarkTools
d, n = 10, 100
X = randn(d, n)
R = zeros(eltype(X), n, n)

# @btime pw_broadcasted!(R, X)
#   59.458 μs (14 allocations: 937.88 KiB)

# julia> @btime pw_for!(R, X)
#   560.125 μs (20200 allocations: 2.77 MiB)
```
The loop version can probably be improved but again, they are equal in the time I spent writing them. 
{% enddetails  %}

A relevant concept is the distinction between imperative and declarative programming styles. Imperative programming involves explicitly specifying all the changes made to the program's state, while declarative programming involves describing what needs to be done and relying on existing implementations of simpler functions to carry out the task. A good example in my opinion is the "[Einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html#numpy.einsum "Einsum on Numpy's doc")" function in NumPy, which is based on [Einstein notations](https://en.wikipedia.org/wiki/Einstein_notation "Einsum on Wikipedia"). With Einsum, the user simply writes out the computation they want to perform without worrying about the implementation details. Typically, the matrix multiplication example from earlier can be written as `np.einsum('ik,kj', A, B)` (check [Tullio.jl](https://github.com/mcabbott/Tullio.jl "Tullio.jl on GitHub") for Julia). I find these tools amazing for a few reasons:

* First of all, they are easy to read: we directly know what is happening in mathematical terms, which can be difficult with loops and abominable with broadcasting;
* They can be extended to all sorts of transformation, see e.g. [Einops](https://github.com/arogozhnikov/einops "Einops on GitHub");
* They can be optimized to rearrange the terms in order to make the least possible computation, see e.g. [opt_einops](https://optimized-einsum.readthedocs.io/en/stable/ "opt_einops documentation");
* They exemplify the paradigm which aims at delegating complicated computation to more optimized routines. It is aligned with this quote of [Rust documentation](https://blog.rust-lang.org/2015/05/11/traits.html "Rust's documentation") "C++ implementations obey the zero-overhead principle: What you don't use, you don't pay for. And further: What you do use, you couldn't hand code any better." [Stroustrup]. 

That's when I realized that I was happy with a language that provided low-level control over operations, but I _needed_ librairies with a sturdy set of function I could call without fear of being suboptimal. Actually, all the tensor libraries I came across actually push their user _not_ to iterate through the arrays, for the performance reasons I mentioned. E.g. the [`ndarray` crate in Rust](https://github.com/rust-ndarray/ndarray#readme "ndarray on GitHub") states "Prefer higher order methods and arithmetic operations on arrays first, then iteration, and as a last priority using indexed algorithms."

## Conclusion 

That's what this post is about: I came to Julia expecting to optimize perfectly any programs I wanted. However, I came to realize that achieving such a feat requires an immense amount of knowledge, or at least it is extremely difficult to compete with preexisting highly specialized routines. I believe this is a direct consequence of the evolution of science in the past 70 years: science has become too complex for one to be able to grasp all the spectrum from its core principles to its final implementation. I doubt we will see another Albert Einstein or any other scientific genius who could, by their own, drastically change the way science is done. 
In other words, science has entered a stage of extreme specialization, where one must delegate some parts of their research to other fields in order to push the limits of knowledge. This requires the creation of appropriate tools to facilitate the sharing of science and knowledge<d-footnote>I'm no epistemologist, this statement might be wrong or someone has probably thought about this better before but well, that's what a blog post is about.</d-footnote>.

In conclusion, we must strike a balance between not being constrained by the tools we use (for instance, I desire the ability to efficiently implement polynomials) while also navigating an increasingly complex scientific landscape that necessitates using and trusting tools we have not personally developed. On top of that, the difference we highlighted between two coding paradigms (imperative vs. declarative) has huge consequences on the performances of automatic differentiation, a core tenet of deep learning. But that's a story for another post! 

***

### What should you choose? 

I thought that imperative languages such as C or Julia were necessary for serious scientific computing, but I'm not convinced anymore. If you are proficient with Numpy and the rest, there is no harm sticking with it. While not as flexible as other languages, those are incredible libraries, very efficient and promoting good coding practices. But if you are tired of messing up with broadcasting, Julia is easy to master and has lot to offer. Just remember not to abuse loops, or do it wisely! And you will likely learn a lot in the process 😉

*** 

*Thanks to [Célia Escribe](https://www.linkedin.com/in/celiaescribe/?originalSubdomain=fr "LinkedIn page") for her valuable feedback!*