# Logarithmic-Time Internal Pattern Matching on Compressed Strings
This project is an implementation of the operations described in [Logarithmic-Time Internal Pattern Matching in Compressed and Dynamic Texts [SPIRE'2024]](https://link.springer.com/chapter/10.1007/978-3-031-72200-4_8). The paper will also be available on [arXiv](arxiv.org).
The implementation is written in Rust and focuses on the compressed setting, although the operations make no assumptions about the origin of the data structure.

## How to cite the paper
Published version from SPIRE 2024:

```bib
@InProceedings{10.1007/978-3-031-72200-4_8,
author="Duyster, Anouk
and Kociumaka, Tomasz",
editor="Lipt{\'a}k, Zsuzsanna
and Moura, Edleno
and Figueroa, Karina
and Baeza-Yates, Ricardo",
title="Logarithmic-Time Internal Pattern Matching Queries in Compressed and Dynamic Texts",
booktitle="String Processing and Information Retrieval",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="102--117"
}

```

## How to use this project

This project supports operations on restricted recompression grammars.

It also provides an (unoptimized) operation to compress an arbitrary `String` into a `DeltaFragment`, which is our implementation of a recompression grammar.


### Create a compressed string

If you have some `String`, you can compress it.
This project does not support manipulating the compressed string.

```rs
// s : String
let df : DeltaFragment = DeltaFragment::from(s);
```

This returns a restricted recompression grammar, with height logarithmic in string length w.h.p.

This operation is slow: linear in string length. It also may require a lot of memory during construction of the data structure.
The good news is that it only needs to be done once.

### Random access
returns the character at an arbitrary index in the string.

```rs
// df : DeltaFragment
let c : char = df.access(3);
```

Takes logarithmic time.

### Extract a fragment

A fragment is a substring of the original text. The created struct references the original data structure.

```rs
// df : DeltaFragment of length 20
let x = df.extract(2,12); // now x is a DeltaFragment of length 10
let y = x.extract(0,10); // x==y
```

This operation is fast: It takes constant time.

### Run an LCE query

```rs
// df : DeltaFragment
let x = df.extract(1,3);
let y = df.extract(5,15);
let res = x.lce(y);
```

This operation is fast: logarithmic in string length. It especially does not scale with the length of the fragments.

### Run an IPM query

```rs
// df : DeltaFragment
let x = df.extract(1,10);
let y = df.extract(5,15);
let res : Option<ArithmeticProgression> = x.ipm(y);
```

This operation is fast: logarithmic in string length. It especially does not scale with the length of the fragments.


## What are IPM queries
Given a string `s`, and two substrings `x` and `y` in `s` s.t. `2*x.len() > y.len()` an IPM query asks for all occurences of `x` in `y`.

Observe that all occurrences must overlap. 

Therefore the result can be represented in the form "the first occurrence starts at index 3 and then there are 2 more occurrences with an offset of 5". This is represented as `Some(ArithmeticProgression { a: 3, max_i: 4, g: 5})`.

## Documentation of the Data Structure
We represent the text as a context-free grammar that allows for rules of the form `A->BC` and `D->E^k`. Substrings are represented as fragments.

Any fragment represented by our data structure stores

* the root of the grammar (for the entire text)
* the index in the text where the represented substring starts (inclusive)
* the index in the text where the represented substring ends (exclusive)

All indices are 0-based.
