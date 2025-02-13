//! This code is the implementation for the paper "Logarithmic-Time Internal Pattern Matching for Compressed and Dynamic texts".
//! It contains the trait for the PILLAR model and an implementation using recompression RLSLPs.

#![feature(linked_list_cursors)] // this should only work with the nightly toolchain

use std::fmt::Debug;

mod common;
pub mod restricted_recompression;
pub mod naive;


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArithmeticProgression {
  a: u64,
  g: u64,
  max_i: u64, // exclusive
}

pub trait PillarModelFragment {
  fn from_randomized(s: String) -> Self;

  fn len(&self) -> u64;

  // returns T[i]
  fn access(&self, i: u64) -> char; // panics on invalid access

  // return a fragment for T[i..j)
  fn extract(&self, i: u64, j: u64) -> Self;

  // return the longest common prefix of the two strings
  fn lcp(&self, other: &Self) -> u64;

  // return the longest common prefix of the two strings
  fn lcp_r(&self, other: &Self) -> u64;

  // the "to" values are exclusive
  // panics when other.len >= 2 * self.len
  // self is X, other is Y
  fn ipm(&self, other: &Self) -> Option<ArithmeticProgression>;
}
