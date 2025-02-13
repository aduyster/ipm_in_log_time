use crate::*;
use kmp;
use std::rc::Rc;

// arguments
// * a partial solution (begun arith prog)
// * the occurence to add
// * length of the pattern
// returns
// * if existing, a finished arith prog
// * the begun arith prog
// runtime O(1)
fn add_to_result(partial: Option<ArithmeticProgression>, occurrence: usize, len: usize) -> (Option<ArithmeticProgression>, ArithmeticProgression) {
  if let Some(mut curr) = partial {
    if curr.max_i == 1 && curr.a + len as u64 > occurrence as u64 {
      // so far, `curr` is a single-occurence arithmetic progression. add the next one here
      curr.max_i = 2;
      curr.g = occurrence as u64 - curr.a;
      return (None, curr);
    } else if curr.a + curr.max_i*curr.g == occurrence as u64 {
      // in this case, `curr` already contains multiple occurences. add one more
      curr.max_i += 1;
      return (None, curr);
    } else {
      return (Some(curr), ArithmeticProgression { a: occurrence as u64, g:0, max_i: 1 });
    }
  } else {
    return (None, ArithmeticProgression { a: occurrence as u64, g:0, max_i: 1 });
  }
}

// computes pattern matching for plaintext pattern and text
// searches the pattern in the text
// return arithmetic progressions of the pattern in the text
// runtime: linear in text length
// size of the returned vec (if it was collected) <= |text|/|pattern|
pub fn pattern_matching<A>(pattern: &[A], text: &[A]) -> impl Iterator<Item=ArithmeticProgression>
where A: Eq
{
  if pattern.len() == 0 {
    return vec![ArithmeticProgression { a: 0, g: 1, max_i: text.len() as u64 }].into_iter()
  }

  let lsp_array = kmp::kmp_table(pattern);

  let mut res = vec![];
  let mut current: Option<ArithmeticProgression> = None;

  let mut text_index = 0;
  let mut pattern_index = 0;

  while text_index < text.len() {
    if text[text_index] == pattern[pattern_index] {
      text_index += 1;
      pattern_index += 1;

      if pattern_index == pattern.len() {
        let (done, begun) = add_to_result(current, text_index - pattern_index, pattern.len());
        if let Some(solution) = done { res.push(solution); }
        current = Some(begun);
        pattern_index = lsp_array[pattern_index-1];
      }
    } else {
      if pattern_index > 0 {
        pattern_index = lsp_array[pattern_index-1]
      } else {
        text_index += 1;
      }
    }
  }

  if let Some(solution) = current { res.push(solution); }
  res.into_iter()
}

// computes pattern matching for run-length-encoded pattern and text
// searches the pattern in the text
// assumes that there is no ...(i,a)(j,a)... in pattern or text (but instead, those are always collapsed to ...(i+j,a)...)
// return arithmetic progressions of the pattern in the text
// runtime: O(|rle(text)|)
// size of the returned vec <= min( |rle(text)|/|rle(pattern)| , |text|/|pattern| )
// returns hit index in rle(text)
pub fn rle_pattern_matching<A>(pattern: &[(u64,A)], text: &[(u64,A)]) -> impl Iterator<Item=ArithmeticProgression>
where A: Eq
{
  if pattern.len() > text.len() {
    return vec![].into_iter();
  }
  return match pattern.len() {
    0 => {
      let textlen = text.into_iter().map(|(n,_)| n).sum();
      vec![ArithmeticProgression { a:0, max_i:textlen, g:1 } ].into_iter()
    },
    1 => {
      let mut res = vec![];
      let (pattern_run,pattern_symb) = &pattern[0];
      let mut counter = 0;
      for (run,symb) in text {
        if symb==pattern_symb && run >= pattern_run {
          let max_i = run-pattern_run+1;
          res.push(ArithmeticProgression { a: counter, g: if max_i==1 {0} else {1}, max_i });
        }
        counter += 1;
      }
      res.into_iter()
    },
    2 => {
      let (first_run,first_symb) = &pattern[0];
      let (second_run,second_symb) = &pattern[1];

      let mut res = vec![];
      let mut counter = 0;

      let mut flag = false;

      for (run,symb) in text {
        if ! flag {
          if symb==first_symb && run >= first_run {
            flag = true;
          }
        } else {
          if symb==second_symb && run >= second_run {
            res.push(ArithmeticProgression { a: counter-1, max_i: 1, g: 0});
          }
          flag = false;
        }
        counter += 1;
      }
      res.into_iter()
    },
    _ => { // |rle(pattern)| >= 3
      let mut res = vec![];

      if pattern.len() > text.len() {
        return vec![].into_iter();
      }

      let candidates = pattern_matching(
        &pattern[1..pattern.len()-1],
        &text[1..text.len()-1]
      );

      for mut candidate in candidates {
        // middle, compare begin
        {
          let (i,A) = &pattern[0];
          let (j,B) = &pattern[candidate.g as usize];

          if j<i || A!=B {
            // begin doesnt fit, keep only first occ
            if candidate.max_i<=1 {
              continue;
            }
            candidate.max_i = 1;
          }
        }

        // middle, compare end
        {
          let (i,A) = &pattern[pattern.len()-1];
          let (j,B) = &pattern[pattern.len()-1- candidate.g as usize];

          if j<i || A!=B {
            // end doesn't fit, keep only last occ
            if candidate.max_i<=1{
              continue;
            }
            candidate.a += (candidate.max_i-1)*candidate.g;
            candidate.max_i = 1;
          }
        }

        // first
        {
          let (i,A) = &pattern[0];
          let (j,B) = &text[candidate.a as usize];

          if j<i || A!=B {
            if candidate.max_i <= 1{
              continue;
            }
            candidate.a += candidate.g;
            candidate.max_i -= 1;
          }
        }

        // last
        if candidate.max_i > 0 {
          let (i,A) = &pattern[pattern.len()-1];
          let (j,B) = &text[candidate.a as usize + candidate.g as usize * (candidate.max_i-1) as usize + pattern.len()-1]; // there must be a more elegant way

          if j<i || A!=B {
            if candidate.max_i<=1 {
              continue;
            }
            candidate.max_i -= 1;
          }
        }

        res.push(candidate);
      }
      res.into_iter()
    }
  };
}

// takes an iterator of arithmetic progressions which
// * is sorted by a
// * all g are equivalent
// returns None on empty input
// returns a single ArithmeticProgression corresponding to all given occurences.
// Panics when that is not possible
pub fn combine_progressions(mut input: impl Iterator<Item=ArithmeticProgression>) -> Option<ArithmeticProgression> {
  return if let Some(first) = input.next() {
    let mut res = first.clone();

    for ap in input {
      if res.max_i == 1 && ap.max_i == 1 {
        res.max_i = 2;
        res.g = ap.a - res.a;
      } else if res.max_i == 1 && res.a == ap.a - ap.g {
        res.max_i = ap.max_i + 1;
        res.g = ap.g;
      } else if ap.max_i == 1 && ap.a == res.a + res.g * res.max_i {
        res.max_i = res.max_i + 1;
      } else if ap.g == res.g && ap.a == res.a + res.g * res.max_i {
        res.max_i = res.max_i + ap.max_i;
      } else {
        panic!();
      }
    }
    Some(res)
  } else {
    None
  };
}

pub fn merge_runs<A>(input: Vec<(u64, Rc<A>)>) -> Vec<(u64, Rc<A>)> {
  let mut res: Vec<(u64, Rc<A>)> = vec![];
  if input.is_empty() {
    return res;
  }
  let mut cur_k: u64 = 0;
  let mut cur_a = input[0].1.clone();
  for (k, a) in input {
    if k > 0 && !Rc::ptr_eq(&a, &cur_a) {
      if cur_k > 0 {
        res.push((cur_k, cur_a));
      }
      (cur_k, cur_a) = (0, a);
    }
    cur_k += k;
  }
  if cur_k > 0 {
    res.push((cur_k, cur_a));
  }
  return res;
}

#[cfg(test)]
mod tests {
  use super::*;
  use test_case::test_case;

  #[test_case("aa","aaa", vec![ArithmeticProgression {a:0,g:1,max_i:2}])]
  #[test_case("aa","aaaBBC",vec![ArithmeticProgression {a:0,g:1, max_i:2 }])]
  #[test_case("bab","babababab",vec![ArithmeticProgression {a:0, g:2, max_i: 4}])]
  #[test_case("","aaa",vec![ArithmeticProgression {a:0, g:1, max_i: 3}])]
  #[test_case("a","bb",vec![] ; "no occ")]
  #[test_case("aa","a",vec![] ; "pattern too long")]
  fn test_pattern_matching(pattern: &str, text: &str, expected: Vec<ArithmeticProgression>) {
    assert_eq!(
        expected,
        pattern_matching(pattern.as_bytes(),text.as_bytes()).collect::<Vec<_>>()
    );
  }

  #[test_case(
    vec![(2,'a')], vec![(17,'a'),(3,'b'),(1,'1'),(2,'a')],
    vec![ArithmeticProgression {a:0, g:1, max_i:16}, ArithmeticProgression {a:3, g:0, max_i: 1}]
    ; "single_prog_pattern"
  )]
  #[test_case(
    vec![(1,'a'),(3,'b')], vec![(17,'a'),(3,'b'),(1,'a'),(4,'b'),(2,'a'),(2,'b'),(2,'a'),(2,'b')],
    vec![ArithmeticProgression {a:0, g:0, max_i: 1}, ArithmeticProgression {a:2, g:0, max_i: 1}]
    ; "two"
  )]
  #[test_case(
    vec![(1,'a'),(1,'b'),(1,'a'),(1,'b'),(1,'a')], vec![(3,'a'),(1,'b'),(1,'a'),(1,'b'),(1,'a'),(1,'b'),(1,'a'),(1,'b'),(1,'a'),(1,'b'),(1,'a')],
    vec![ArithmeticProgression { a:0, g:2, max_i: 4}]
    ; "multi"
  )]
  #[test_case(
    vec![(1,'b'),(1,'c'),(1,'d')],vec![(2,'a'),(1,'b'),(1,'c'),(1,'d'),(1,'e')],
    vec![ArithmeticProgression { a:1, g:0, max_i: 1}]
    ; "single hit"
  )]
  fn test_rle_pattern_matching(pattern: Vec<(u64,char)>, text: Vec<(u64,char)>, expected: Vec<ArithmeticProgression>) {
    assert_eq!(
      expected,
      rle_pattern_matching(&pattern, &text).collect::<Vec<_>>()
    );
  }

  #[test_case(None, vec![])]
  #[test_case(Some(ArithmeticProgression{a:0,g:1,max_i:10}), vec![ArithmeticProgression{a:0,g:1,max_i:3}, ArithmeticProgression{a:3, g:1, max_i:7}])]
  fn good_combine_progressions(expected: Option<ArithmeticProgression>, input: Vec<ArithmeticProgression>) {
    assert_eq!(expected, combine_progressions(input.into_iter()));
  }

  #[should_panic]
  #[test_case(vec![ArithmeticProgression{a:0,g:1,max_i:7}, ArithmeticProgression{a:3,g:2,max_i:2}])]
  fn bad_combine_progressions(input: Vec<ArithmeticProgression>) {
    combine_progressions(input.into_iter());
  }
}
