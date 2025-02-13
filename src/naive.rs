use common::{combine_progressions, pattern_matching};

use crate::*;

pub struct NaiveFragment {
  pub content: String
}

impl PillarModelFragment for NaiveFragment {
  fn from_randomized(s: String) -> Self {
    Self { content: s }
  }

  fn len(&self) -> u64 {
    self.content.len() as u64
  }

  fn access(&self, i: u64) -> char {
    self.content.chars().nth(i as usize).unwrap()
  }

  fn extract(&self, i: u64, j: u64) -> Self {
    let s = self.content.to_string();
    let mut iter = s.char_indices();
    let (start,_) = iter.nth(i as usize).unwrap();
    let (end, _) = iter.nth((j-i-1) as usize).unwrap();
    let substring = &s[start..end];

    Self { content: String::from(substring) }
  }

  fn lcp(&self, _other: &Self) -> u64 {
    todo!()
  }

  fn lcp_r(&self, _other: &Self) -> u64 {
    todo!()
  }

  fn ipm(&self, other: &Self) -> Option<ArithmeticProgression> {
    combine_progressions(
      pattern_matching(&self.content.chars().collect::<Vec<_>>(), &other.content.chars().collect::<Vec<_>>())
    )
  }
}

#[cfg(test)]
mod tests {
  use test_case::test_case;
  use super::*;

  #[test_case("aaaaa",0,4,0,5, Some(ArithmeticProgression { a: 0, g: 1, max_i: 2}))]
  #[test_case("aaaaa",0,5,0,4, None)]
  fn simple(s: &str, x_i: usize, x_j: usize, y_i: usize, y_j: usize, expected: Option<ArithmeticProgression>) {
    let y = NaiveFragment::from_randomized(s[y_i..y_j].into());
    let x = NaiveFragment::from_randomized(s[x_i..x_j].into());
    assert_eq!(expected, x.ipm(&y));
  }
}
