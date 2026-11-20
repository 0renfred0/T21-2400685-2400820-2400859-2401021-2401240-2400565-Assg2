# Question 2 (Team T21)

Notation: the image has `m` rows and `n` columns, and `e[i,j]` is the energy of pixel `(i,j)`.
We number rows from 1 here. In the code they start from 0.

## 2(a)(i) Why the number of seams explodes

Think of building a seam one row at a time. In the top row we can start anywhere, so that is `n`
choices. After that, each step goes to the pixel straight below or one to the left or right. In
the middle of the image that is three options. At the left or right edge it is two, because one
of the neighbours is missing. Since `n > 1` there are always at least two options, so with `m`
rows we get at least `n * 2^(m-1)` seams. For `n >= 2` that is at least `2^m`. (If we count the
three-way choices instead, we get an upper bound of `n * 3^(m-1)`, so it really is exponential
either way.)

We also checked this by brute force with `all_vertical_seams` in `tests/helpers.py`. Here are the
counts for a few widths:

| n \ m | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|-------|---|---|---|---|---|---|---|---|
| 2 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| 3 | 3 | 7 | 17 | 41 | 99 | 239 | 577 | 1393 |
| 5 | 5 | 13 | 35 | 95 | 259 | 707 | 1931 | 5275 |

For `n = 2` it is exactly `2^m`, which shows the bound can't be improved. Our test
`test_number_of_seams_grows_at_least_exponentially` in `tests/test_seams.py` checks the same thing.
Counting this way is also a tiny DP: the seams ending at a pixel are the sum of the seams ending
at its three parents. Using that, a 100 x 100 image has a 50-digit number of seams. Trying all of
them is hopeless, which is why we need dynamic programming.

## 2(a)(ii) How the problem breaks into subproblems

Let `M[i,j]` be the cheapest total energy of any seam that starts in the top row and ends at
`(i,j)`. Then:

```
M[1, j] = e[1, j]
M[i, j] = e[i, j] + min( M[i-1, j-1], M[i-1, j], M[i-1, j+1] )      (skip columns that don't exist)
```

The cheapest seam in the whole image costs `min_j M[m, j]`, and we find the seam itself by
remembering which parent won in each cell and walking back up from the best cell in the last row.

**Why this is right.** Suppose the cheapest seam into `(i,j)` came through the pixel `(i-1,j')`
above it. The part of the seam above that pixel has to be the cheapest seam into `(i-1,j')`. If
there were a cheaper one, we could swap it in. The seam would still be connected, since the step
into `(i,j)` hasn't changed, and the total would drop, which contradicts "cheapest". So we only
ever need the best answer for each parent, not every seam that reaches it.

**Why it's fast.** Each `M[i-1,j]` is used by up to three cells in the next row. A plain
recursion would recompute it again and again (about `3^m` steps). With DP we fill each of the
`m * n` cells once, with three comparisons per cell, so the whole thing is `O(m n)`.
`find_vertical_seam_dp` does exactly this. It only keeps the previous row of costs and stores the
parent choices as small `int8` offsets. Horizontal seams are the same problem on the transposed
image.

Another way to see it: the pixels form a directed acyclic graph and a seam is the shortest path from
the top to the bottom. Our row-by-row DP is that shortest-path computation done in topological
order. We also implemented graph cut (item 4) as a different way to get the same optimal seam, but
it is much slower.

## 2(b) Would divide and conquer speed up the Sobel step?

Not in terms of work. Each output pixel needs 9 multiply-adds from its own 3x3 neighbourhood, which
is a constant amount, and every one of the `m n` pixels needs an output. So the cost is
`Theta(m n)` no matter what. If we split the image in half, filter both halves with a one-row
overlap, and stick them together, we get `T(N) = 2T(N/2) + O(1) = Theta(N)`. That is the same cost.

Splitting did help a bit in practice though. We tried it with row blocks (the output was
identical to filtering the whole image, single thread, on our laptop):

| Image | Whole image | 4 blocks | 16 blocks |
|-------|-------------|----------|-----------|
| 300 x 480 | 3.5 ms | 3.0 ms | 2.0 ms |
| 1080 x 1920 | 66 ms | 53 ms | 57 ms |
| 2048 x 2048 | 142 ms | 122 ms | 108 ms |

That is roughly 1.2x to 2x faster. The reason is cache: small blocks fit in it, while our
shift-and-add filter makes large temporary arrays for a big image. So it is a constant-factor win,
not a better big-O. The blocks are also independent, so they could be run on separate threads.

Divide and conquer really does win for large kernels. A direct `k x k` convolution costs
`Theta(m n k^2)`, and the FFT (itself divide and conquer) costs `Theta(m n log(m n))` for any `k`. On
a 512 x 512 image we saw the FFT overtake direct convolution at around `k = 30` (at `k = 63` it was
16 ms against 284 ms). Sobel is only 3x3, so that doesn't help us.

Two things would help more, and we have not implemented either of them. First, after a seam is
removed only a few pixels per row change their energy, so updating just those would cost `O(m)`
instead of `O(m n)`. In our `castle.png` run (144 seams, 1.34 s in total) the energy maps took
0.50 s and the DP took 0.53 s, so this is a real saving. Second, Sobel is separable, so two 1-D
passes need 6 multiplies per pixel instead of 9.

**In short:** divide and conquer doesn't make the 3x3 filter asymptotically cheaper, but tiling
gives a modest speed-up, and updating the energy only near the seam is the better idea.
