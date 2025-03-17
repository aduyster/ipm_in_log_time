use crate::{
  ArithmeticProgression,
  PillarModelFragment,
  common,
};
use rand::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

use std::{
  collections::{HashMap, HashSet},
  rc::Rc,
};

mod popped_sequence;
use popped_sequence::{
  PSNode,
  PoppedSequence,
};

mod parsetreenodes;
use parsetreenodes::{
  ParseTreeNode,
  UncompressedParseTreeNode,
};

mod lce;
use lce::lce;

mod ipm;
use ipm::ParseTreeSlice;

// Why Rc: https://users.rust-lang.org/t/confused-between-box-rc-cell-arc/10946 (in the future, we might choose Arc if we use parallel algorithms)
// todo: je nach level gibt es nur block und run!
#[derive(Debug, Eq, Hash)]
enum Production {
  Leaf(char),
  Pair(Rc<Symbol>, Rc<Symbol>),
  Run(Rc<Symbol>, u64),
}

impl PartialEq for Production {
  fn eq(&self, other: &Self) -> bool {
    return match (self, other) {
      (Production::Leaf(a), Production::Leaf(b)) => a == b,
      (Production::Pair(a, b), Production::Pair(c, d)) => {
        Rc::<Symbol>::ptr_eq(a, c) && Rc::<Symbol>::ptr_eq(b, d)
      }
      (Production::Run(a, i), Production::Run(b, j)) => Rc::<Symbol>::ptr_eq(a, b) && i == j,
      _ => false,
    };
  }
}

/// A restricted recompression RLSLP symbol
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Symbol {
  /// The right-hand side of the production
  rhs: Production,
  /// The expansion length
  pub length: u64,
  /// The lowest level this symbol is in (guaranteed to be the same level for all occurrences within a restricted recompression RLSLP)
  level: u16,
}

impl Symbol {
  pub const fn new_leaf(c: char) -> Self {
    Symbol {
      length: 1,
      level: 0,
      rhs: Production::Leaf(c),
    }
  }

  pub fn new_pair(a: Rc<Symbol>, b: Rc<Symbol>, level: u16) -> Self {
    Symbol {
      length: a.length + b.length,
      level,
      rhs: Production::Pair(a, b),
    }
  }

  pub fn new_run(a: Rc<Symbol>, k: u64, level: u16) -> Self {
    Symbol {
      length: a.length * k,
      level,
      rhs: Production::Run(a, k),
    }
  }

  /// returns the
  /// * leaf index respective to that child
  /// * and the child child index
  fn leaf_index_child(&self, index: u64) -> (u64, u64) {
    if index >= self.length {
      panic!("invalid index!")
    }
    match &self.rhs {
      Production::Leaf(_) => panic!("leaves don't have children!"),
      Production::Pair(a, _) => {
        if index < a.length {
          return (index, 0);
        } else {
          return (index - a.length, 1);
        }
      }
      Production::Run(a, _) => {
        return (index % a.length, index / a.length);
      }
    }
  }

  /// Returns the *index*-th symbol of the production
  fn child(&self, index: u64) -> Rc<Symbol> {
    debug_assert!(index < self.degree());
    return match &self.rhs {
      Production::Leaf(_) => panic!(),
      Production::Pair(a, b) => ([a, b][index as usize]).clone(),
      Production::Run(a, _) => a.clone(),
    };
  }

  /// Return the total expansion length of the first *index* children
  fn boundary(&self, index: u64) -> u64 {
    return match &self.rhs {
      Production::Leaf(_) => {
        debug_assert!(index == 0);
        0
      }
      Production::Pair(a, b) => {
        debug_assert!(index <= 2);
        [0, a.length, a.length + b.length][index as usize]
      }
      Production::Run(a, k) => {
        debug_assert!(index <= *k);
        index * a.length
      }
    };
  }

  /// returns the number of children
  fn degree(&self) -> u64 {
    return match &self.rhs {
      Production::Leaf(_) => 0,
      Production::Pair(_, _) => 2,
      Production::Run(_, k) => *k,
    };
  }

  /// Expands self up to level *level* (level = 0 for full expansion), and append the result into res
  fn expand_above(self: &Rc<Self>, level: u16, res: &mut Vec<Rc<Symbol>>) {
    if self.level <= level {
      res.push(self.clone());
    } else {
      for i in 0..self.degree() {
        self.child(i).expand_above(level, res);
      }
    }
  }
}


#[derive(Debug)]
pub struct DeltaFragment {
  // roots: Vec<Rc<Symbol>>, // the roots of a fragment! (for the entire tree, there is only one; otherwise, at most O(#rounds))
  // todo: the delta level, how to do this now that the rootlevel may contain things below? but is that even an issue?
  pub root: Rc<Symbol>,
  pub from: u64,
  pub to: u64,
}

fn compute_paused_length(level: u16) -> u64 {
  // (8/7)^((k+1)/2 -1)
  // this is a prelim implentation and probably fails due to rounding errors for large numbers...
  ((8.0 / 7.0) as f64).powf(((level + 1) / 2) as f64) as u64
}

impl DeltaFragment {
  pub fn layers(&self) -> u16 {
    self.root.level
  }

  // returns
  // * `level`: the level from the popped sequence whose elements where taken
  // * `proxy_pattern`: in run-length encoded form
  // * `start_of_pp_in_self`: the index in the text s.t. X[i..?] = exp(proxy_pattern)
  fn proxy_pattern(
    &self,
  ) -> ParseTreeSlice {
    // Compute the level \ell
    let ps = PoppedSequence::from(self);
    let max_level = ps.left.len() as u16;
    let mut level = max_level;
    let mut len: u16 = 0;
    let mut q: Vec<Vec<Rc<Symbol>>> = vec![vec![]; (level + 1) as usize];
    while level >= len {
      level = level - 1;
      for x in [&ps.left[level as usize], &ps.right[level as usize]] {
        match x.to_run() {
          None => {}
          Some((k, a)) => {
            for _ in 0..k {
              q[a.level as usize].push(a.clone());
              len = len + 1;
              if len > level {
                break;
              }
            }
          }
        }
      }
      let to_expand = q.pop().unwrap();
      for a in to_expand {
        match &a.rhs {
          Production::Leaf(_) => {}
          Production::Pair(b, c) => {
            q[b.level as usize].push(b.clone());
            q[c.level as usize].push(c.clone());
            len = len + 1;
            if len > level {
              break;
            }
          }
          Production::Run(b, k) => {
            len = len - 1;
            for _ in 0..*k {
              q[b.level as usize].push(b.clone());
              len = len + 1;
              if len > level {
                break;
              }
            }
          }
        }
      }
    }

    // Construct \hX_{\ell+1}
    let mut ps_left: Vec<PSNode> = ps.left[(level+1) as usize ..max_level as usize].iter().cloned().collect();
    let mut ps_right: Vec<PSNode> = ps.right[(level+1) as usize..max_level as usize].iter().cloned().collect();
    ps_right.reverse();
    ps_left.append(&mut ps_right);

    let mut x_parent: Vec<Rc<Symbol>> = vec![];

    for x in ps_left {
      match x.to_run() {
        None => {}
        Some((k, a)) => {
          for _ in 0..k {
            a.expand_above(level+1, &mut x_parent);
          }
        }
      }
    }

    // Retrieve rle(\hX_\ell)
    let mut res: Vec<(u64,Rc<Symbol>)> = vec![];
    if let Some((k, a)) = ps.left[level as usize].to_run() {
      res.push((k,a));
    }
    for a in x_parent {
      if a.level <= level {
        res.push((1,a));
      } else {
        match &a.rhs {
          Production::Leaf(_) => panic!(),
          Production::Pair(b,c) => {
            res.push((1,b.clone()));
            res.push((1,c.clone()));
          }
          Production::Run(b,e) => res.push((*e,b.clone()))
        }
      }
    }
    if let Some((k, a)) = ps.right[level as usize].to_run() {
      res.push((k,a));
    }

    let prefix_len = ps.left[0..level as usize].iter().map(|psnode| psnode.length()).sum::<u64>();
    ParseTreeSlice { level, content: res, prefix_len }
  }

  fn proxy_text(
    &self,
    level: u16,
    max_par_len: u64, // maximum number of distinct parents (shrink size) on each side of the center
                max_len: u64, // maximum number of symbols at each side of the center
  ) -> ParseTreeSlice {
    let mut middle = UncompressedParseTreeNode::leaf(self.root.clone(), self.from + self.len() / 2);
    while middle.level < level {
      middle = middle.parent();
    }
    let middle_from = middle.from();
    let middle_to = middle.to();
    let mut right = ipm::walk(&middle, true, max_len, max_par_len, self.to - middle_from);
    let mut left = ipm::walk(&middle, false, max_len, max_par_len, middle_to - self.from);

    let left_len = left.iter().map(|A| A.0 * A.1.length).sum::<u64>();
    let prefix_len = middle_to - left_len - self.from;

    if right.is_empty() && left.is_empty() {
      return ParseTreeSlice { content: vec![], prefix_len: 0, level };
    }
    if left.is_empty() {
      right[0].0 = right[0].0 - 1;
    } else {
      left[0].0 = left[0].0 - 1;
    }

    left.reverse();
    left.append(&mut right);

    ParseTreeSlice { content: common::merge_runs(left), prefix_len, level }
  }
}

impl PillarModelFragment for DeltaFragment {
  fn from_randomized(s: String) -> Self {
    let mut rng = StdRng::seed_from_u64(2);
    let mut current_level: Vec<Rc<Symbol>> = {
      let mut leaves = HashMap::<char, Rc<Symbol>>::new();

      s.chars()
      .map(|c| Rc::clone(leaves.entry(c).or_insert(Rc::new(Symbol::new_leaf(c)))))
      .collect()
    };
    let mut next_level = vec![];

    let mut level = 0;
    while current_level.len() > 1 {
      let paused_length = compute_paused_length(level);

      if level % 2 == 1 {
        let mut left = HashSet::<Rc<Symbol>>::new();
        let mut right = HashSet::<Rc<Symbol>>::new();
        let mut last_left = false;
        let mut symbols_for_next_level = HashMap::<(Rc<Symbol>, Rc<Symbol>), Rc<Symbol>>::new();

        for node in current_level {
          // do pair compression

          if node.length <= paused_length {
            // find the side of the current
            let is_left = if left.contains(&node) {
              true
            } else if right.contains(&node) {
              false
            } else {
              let is_left = rng.gen_bool(0.5);
              if is_left {
                left.insert(Rc::clone(&node));
              } else {
                right.insert(Rc::clone(&node));
              }
              is_left
            };

            // replace if necessary
            if last_left && !is_left {
              // last was left, current is right
              let l = next_level.pop().unwrap();
              let r = node;

              let key = (Rc::clone(&l), Rc::clone(&r));
              let n : &Rc<Symbol> = symbols_for_next_level.entry(key).or_insert(Rc::new(Symbol::new_pair(
                Rc::clone(&l),
                                                                                                         Rc::clone(&r),
                                                                                                         level + 1,
              )));
              next_level.push(Rc::clone(n));
              last_left = false;
            } else {
              last_left = is_left;
              next_level.push(node);
            }
          } else {
            last_left = false;
            next_level.push(node);
          }
        }
        current_level = core::mem::take(&mut next_level);
      } else {
        // do block compression
        let mut symbols_for_next_level = HashMap::<(Rc<Symbol>, u64), Rc<Symbol>>::new();
        let mut last_symbol = current_level[0].clone();
        let mut next_level = vec![];
        let mut counter = 1;

        for node in &current_level[1..] {
          if &last_symbol == node {
            counter += 1;
          } else {
            if counter > 1 {
              let key = (Rc::clone(&last_symbol), counter);
              let new_node = symbols_for_next_level.entry(key).or_insert(Rc::new(Symbol::new_run(
                last_symbol,
                counter,
                level + 1,
              )));

              next_level.push(Rc::clone(new_node));
            } else {
              next_level.push(last_symbol);
            }

            counter = 1;
            last_symbol = node.clone();
          }
        }

        if counter > 1 {
          let key = (Rc::clone(&last_symbol), counter);
          let new_node = symbols_for_next_level.entry(key).or_insert(Rc::new(Symbol::new_run(
            Rc::clone(&last_symbol),
                                                                                             counter,
                                                                                             level + 1,
          )));

          next_level.push(Rc::clone(new_node));
        } else {
          next_level.push(last_symbol);
        }
        current_level = core::mem::take(&mut next_level);
      }
      level += 1;
      // dbg!(&current_level.clone());
    }

    let root: Rc<Symbol> = current_level[0].clone();
    DeltaFragment {
      root: Rc::clone(&root),
      from: 0,
      to: root.length,
    }
  }

  fn len(&self) -> u64 {
    self.to - self.from
  }

  // returns T[i] in O(log n)
  fn access(&self, i: u64) -> char {
    let mut node = Rc::clone(&self.root);
    let mut index = self.from + i;

    while node.level > 0 {
      let (i, child_index) = node.leaf_index_child(index);
      index = i;
      node = node.child(child_index);
    }

    return if let Production::Leaf(c) = node.rhs {
      c
    } else {
      unreachable!("Invalid index");
    };
  }

  fn extract(&self, from: u64, to: u64) -> Self {
    assert!(self.from+to <= self.to);
    DeltaFragment {
      root: Rc::clone(&self.root),
      from: self.from + from,
      to: self.from + to,
    }
  }

  fn lcp(&self, other: &Self) -> u64 {
    if std::cmp::min(self.len(), other.len()) == 0 { return 0; }

    let full: u64 = lce(
      self.root.clone(),
                        self.from,
                        other.root.clone(),
                        other.from,
                        true,
    );


    let scope = std::cmp::min(self.to - self.from, other.to - other.from);
    return std::cmp::min(full, scope);
  }

  fn lcp_r(&self, other: &Self) -> u64 {
    if other.to == 0 || self.to == 0 { return 0; } //can't have a backwards pointer starting at 0 due to index shenanigangs :/

    let full: u64 = lce(
      self.root.clone(),
                        self.to,
                        other.root.clone(),
                        other.to,
                        false,
    );
    let scope = std::cmp::min(self.to - self.from, other.to - other.from);
    return std::cmp::min(full, scope);
  }

  // self is X, other is Y
  fn ipm(&self, other: &Self) -> Option<ArithmeticProgression> {
    let proxy_pattern = self.proxy_pattern();
    let proxy_text = other.proxy_text(proxy_pattern.level, 2*(proxy_pattern.level as u64)+3, proxy_pattern.len()+(proxy_pattern.level as u64));

    let end_of_proxy_pattern_in_x = proxy_pattern.prefix_len + proxy_pattern.expanded_len();
    let proxy_pattern_suffix_len = self.len() - end_of_proxy_pattern_in_x;

    // find candidates, reduce them part by part
    common::combine_progressions(
      common::rle_pattern_matching(&proxy_pattern.content, &proxy_text.content)
      .filter_map( |mut candidate| {
        candidate.g = ipm::compute_g(candidate.g, &proxy_pattern.content);
        candidate.a = ipm::compute_a(&mut candidate, &proxy_pattern, &proxy_text)?;

        if candidate.max_i > 1 {
          debug_assert!(candidate.g!=0);

          let u = self.extract(std::cmp::min(end_of_proxy_pattern_in_x, self.len()), self.len())            .lcp(&self.extract(std::cmp::min(end_of_proxy_pattern_in_x - candidate.g,self.len()), self.len()));
          let ubar = self.extract(0, proxy_pattern.prefix_len).lcp_r(&self.extract(0, proxy_pattern.prefix_len + candidate.g));

          let paper_a = candidate.a + end_of_proxy_pattern_in_x + (candidate.max_i-1)*candidate.g;
          let v = other.extract(std::cmp::min(paper_a,other.len()), other.len()).lcp(& other.extract(std::cmp::min(paper_a-candidate.g,other.len()), other.len()));
          let paper_abar = candidate.a+proxy_pattern.prefix_len;
          let vbar = other.extract(0, paper_abar).lcp_r(&other.extract(0, paper_abar + candidate.g));

          candidate = ipm::validate_occurence(
            candidate,
            proxy_pattern.prefix_len,
            proxy_pattern_suffix_len,
            u,
            ubar,
            v,
            vbar,
          )?;
        }

        return if candidate.max_i == 1 {
          candidate.g = 0;
          let hit = &other.extract(std::cmp::min(other.len(), candidate.a),std::cmp::min(candidate.a + self.len(),other.len()));

          if self.lcp(&hit) == self.len() {
            Some(candidate)
          } else {
            None
          }
        } else {
          Some(candidate)
        };
      }
      )
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::rc::Rc;
  use rand::distributions::Uniform;
  use test_case::test_case;
  use crate::naive::*;

  fn vec_compare<A: std::fmt::Debug + PartialEq>(v1: Vec<A>, v2: Vec<A>) {
    assert_eq!(v1.len(), v2.len());
    for i in 0..v1.len() {
      assert_eq!(v1[i], v2[i]);
    }
  }

  #[test_case(0, "abcde", 0,3,1,3 ; "none")]
  #[test_case(1, "aaaa", 1,3,2,3 ; "minirun1")]
  #[test_case(3, "baaaaa", 0,3,0,6 ; "minirun2")]
  #[test_case(3, "aaaaabc", 2,6,0,6 ; "normal")]
  #[test_case(3, "aaaa", 0,4,1,4 ; "run only")]
  #[test_case(0, "abc", 0,3,1,3 ; "dont use run")]
  #[test_case(2, "aaaaabc", 1,7,3,7 ; "run later")]
  #[test_case(4, "aaaaaaaaaabc", 1,7,3,7 ; "interrupted run")]
  #[test_case(0, "aaaaaaaaaaaaaaaa", 1,1,3,3 ; "empty")]
  fn test_lcp(expected: u64, s: &str, from1: u64, to1: u64, from2: u64, to2: u64) {
    let deltatree = DeltaFragment::from_randomized(String::from(s));
    let f1 = deltatree.extract(from1, to1);
    let f2 = deltatree.extract(from2, to2);
    assert_eq!(expected, f1.lcp(&f2));
    assert_eq!(expected, f2.lcp(&f1));
  }

  #[test_case(0, "cba", 0,3,0,2 ; "none")]
  #[test_case(1, "aa", 0,1,1,2 ; "minirun1")]
  #[test_case(3, "aaaaab", 3,6,0,6  ; "minirun2")]
  #[test_case(3, "cbaaaa", 0,6,0,5 ; "normal")]
  #[test_case(3, "aaaa", 0,4,0,3 ; "run only")]
  #[test_case(0, "cba", 0,3,0,2 ; "dont use run")]
  #[test_case(2, "cbaaaaa", 0,7,0,4 ; "run later")]
  #[test_case(0, "aaa", 1,1,3,3 ; "empty")]
  fn test_lcp_r(expected: u64, s: &str, from1: u64, to1: u64, from2: u64, to2: u64) {
    let deltatree = DeltaFragment::from_randomized(String::from(s));
    let f1 = deltatree.extract(from1, to1);
    let f2 = deltatree.extract(from2, to2);
    assert_eq!(expected, f1.lcp_r(&f2));
    assert_eq!(expected, f2.lcp_r(&f1));
  }

  #[test]
  fn test_proxy_pattern_normal() {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let c = Rc::new(Symbol::new_leaf('c'));
    let d = Rc::new(Symbol::new_leaf('d'));
    let e = Rc::new(Symbol::new_leaf('e'));

    let A = Rc::new(Symbol::new_pair(a, b.clone(), 1));
    let C = Rc::new(Symbol::new_pair(c.clone(), d.clone(), 1));
    let D = Rc::new(Symbol::new_pair(A.clone(), C.clone(), 2));
    let E = Rc::new(Symbol::new_pair(D.clone(), e.clone(), 3));

    let df = DeltaFragment {
      root: E.clone(),
      from: 0,
      to: E.length,
    };

    let pp1 = ParseTreeSlice { level: 0, content: vec![(1, b.clone()), (1, c.clone()),(1, d.clone()),(1, e.clone())], prefix_len: 0};
    assert_eq!(pp1, df.extract(1, 5).proxy_pattern());
    let pp2 = ParseTreeSlice { level: 0, content: vec![(1, b.clone()), (1, c.clone())], prefix_len: 0};
    assert_eq!(pp2, df.extract(1, 3).proxy_pattern());
  }

  #[test]
  fn test_proxy_pattern_high() {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let c = Rc::new(Symbol::new_leaf('c'));
    let d = Rc::new(Symbol::new_leaf('d'));
    let e = Rc::new(Symbol::new_leaf('e'));

    let A = Rc::new(Symbol::new_pair(a, b.clone(), 1));
    let C = Rc::new(Symbol::new_pair(c.clone(), d, 1));
    let D = Rc::new(Symbol::new_pair(A.clone(), C.clone(), 2));
    let E = Rc::new(Symbol::new_pair(D, e.clone(), 3));

    let df = DeltaFragment {
      root: E.clone(),
      from: 0,
      to: E.length,
    };

    let pp1 = ParseTreeSlice { level: 1, content: vec![(1, A.clone()), (1, C.clone())], prefix_len: 0 };

    assert_eq!(pp1, df.proxy_pattern());
    assert_eq!(pp1, df.extract(0, 4).proxy_pattern());
  }

  #[test]
  fn test_proxy_pattern_runs() {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let A = Rc::new(Symbol::new_run(a.clone(), 4, 1));
    let B = Rc::new(Symbol::new_run(a.clone(), 2, 1));
    let C = Rc::new(Symbol::new_pair(b.clone(), A.clone(), 2));
    let D = Rc::new(Symbol::new_pair(b.clone(), B.clone(), 2));
    let E = Rc::new(Symbol::new_pair(D.clone(), b.clone(), 3));
    let F = Rc::new(Symbol::new_pair(C.clone(), E, 4));

    let pp1 = vec![(3, a.clone())];
    let pp2 = vec![(3, a.clone())];
    let pp5 = vec![(1, b.clone()), (2, a.clone())];

    let df = DeltaFragment {
      root: F.clone(),
      from: 0,
      to: F.length,
    };

    assert_eq!(ParseTreeSlice { level: 0, content: pp1, prefix_len: 0}, df.extract(1, 4).proxy_pattern());
    assert_eq!(ParseTreeSlice { level: 0, content: pp2, prefix_len: 0}, df.extract(2, 5).proxy_pattern());
    assert_eq!(ParseTreeSlice { level: 0, content: vec![(4, a.clone()), (1, b.clone()), (2, a.clone())], prefix_len: 0},
               df.extract(1, 8).proxy_pattern()
    );
    assert_eq!(
      ParseTreeSlice { level: 0, content: vec![(3, a.clone()), (1, b.clone()), (2, a.clone())], prefix_len: 0},
               df.extract(2, 8).proxy_pattern()
    );
    assert_eq!(ParseTreeSlice { level: 0, content: pp5, prefix_len: 0}, df.extract(5, 8).proxy_pattern());
    assert_eq!(ParseTreeSlice { level: 0, content: vec![(4, a.clone()), (1, b.clone()), (1, a.clone())], prefix_len: 0}, df.extract(1, 7).proxy_pattern());
    assert_eq!(ParseTreeSlice{level: 1, content: vec![(1, A.clone()), (1, b.clone())], prefix_len: 1}, df.extract(0, 7).proxy_pattern());
    assert_eq!(ParseTreeSlice{level: 1, content: vec![(1, b.clone()), (1, B.clone())], prefix_len: 3}, df.extract(2, 9).proxy_pattern());
  }

  #[test]
  fn test_proxy_text1() {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let c = Rc::new(Symbol::new_leaf('c'));
    let d = Rc::new(Symbol::new_leaf('d'));
    let e = Rc::new(Symbol::new_leaf('e'));
    let f = Rc::new(Symbol::new_leaf('f'));

    let all = vec![
      (1, a.clone()),
      (1, b.clone()),
      (1, c.clone()),
      (1, d.clone()),
      (1, e.clone()),
      (1, f.clone()),
    ];
    let G = Rc::new(Symbol::new_pair(a.clone(), b.clone(), 1));
    let H = Rc::new(Symbol::new_pair(c.clone(), d.clone(), 1));
    let I = Rc::new(Symbol::new_pair(e.clone(), f.clone(), 1));
    let J = Rc::new(Symbol::new_pair(G, H, 2));
    let K = Rc::new(Symbol::new_pair(J, I, 3));

    let df = DeltaFragment {
      root: K.clone(),
      from: 0,
      to: K.length,
    };

    assert_eq!(ParseTreeSlice { content: all.clone(), prefix_len: 0, level: 0}, df.proxy_text(0, 2, 4));
    assert_eq!(ParseTreeSlice { content: all.clone(), prefix_len: 0, level: 0}, df.proxy_text(0, 4, 4));
    assert_eq!(ParseTreeSlice { content: all.clone(), prefix_len: 0, level: 0}, df.proxy_text(0, 2, 10));
    assert_eq!(ParseTreeSlice { content: vec![(1, d.clone())], prefix_len: 3, level: 0}, df.proxy_text(0, 1, 1));
    assert_eq!(ParseTreeSlice { content: vec![(1, c.clone()),(1, d.clone())], prefix_len: 2, level: 0}, df.proxy_text(0, 1, 2));
    assert_eq!(ParseTreeSlice { content: vec![(1, c.clone()),(1, d.clone()), (1, e.clone())], prefix_len: 2, level: 0}, df.proxy_text(0, 2, 2));
    assert_eq!(ParseTreeSlice { content: vec![(1, b.clone()), (1, c.clone()),(1, d.clone()), (1, e.clone()),(1, f.clone())], prefix_len: 1, level: 0}, df.proxy_text(0, 2, 3));
  }

  #[test]
  fn test_proxy_text2() {
    let al = Rc::new(Symbol::new_leaf('a'));
    let bl = Rc::new(Symbol::new_leaf('b'));

    let little_runs = vec![
      (1, al.clone()),
      (1, bl.clone()),
      (1, al.clone()),
      (1, bl.clone()),
      (1, al.clone()),
      (1, bl.clone()),
      (2, al.clone()),
    ];

    let a = Rc::new(Symbol::new_run(al.clone(), 2, 1));
    let c = Rc::new(Symbol::new_pair(al.clone(), bl.clone(), 2));
    let d = Rc::new(Symbol::new_pair(a.clone(), bl.clone(), 2));
    let e = Rc::new(Symbol::new_run(c.clone(), 4, 3));
    let f = Rc::new(Symbol::new_pair(d.clone(), e.clone(), 4));
    let g = Rc::new(Symbol::new_pair(c.clone(), al.clone(), 4));
    let h = Rc::new(Symbol::new_pair(d.clone(), g.clone(), 6));
    let j = Rc::new(Symbol::new_pair(f.clone(), h.clone(), 8));

    let df = DeltaFragment {
      root: j.clone(),
      from: 0,
      to: j.length,
    };

    assert_eq!(ParseTreeSlice{ content: little_runs, prefix_len: 5, level: 0 }, df.proxy_text(0, 4, 16));
  }

  #[test]
  fn test_long_run_proxy_text() {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let m = Rc::new(Symbol::new_leaf('d'));
    let c = Rc::new(Symbol::new_pair(a, b, 1));
    let d = Rc::new(Symbol::new_run(c.clone(),17,2));
    let e = Rc::new(Symbol::new_run(c.clone(),23,2));
    let f = Rc::new(Symbol::new_pair(d, m, 3));
    let g = Rc::new(Symbol::new_pair(f, e, 4));

    let df = DeltaFragment { root: g.clone(), from: 0, to: g.clone().length };
    let y = df.extract(36, 80);
    dbg!(&y);
    assert_eq!(ParseTreeSlice { content: vec![(21,c.clone())], prefix_len: 1, level: 1}, y.proxy_text(1, 17, 34));
  }

  // if we want proxy text to be same as paper, this would be the way
  #[test]
  fn test_short_proxy_text() {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let c = Rc::new(Symbol::new_leaf('c'));
    let d = Rc::new(Symbol::new_leaf('d'));
    let e = Rc::new(Symbol::new_leaf('e'));

    let A = Rc::new(Symbol::new_pair(a.clone(), b.clone(), 1));
    let C = Rc::new(Symbol::new_pair(c.clone(), d.clone(), 1));
    let D = Rc::new(Symbol::new_pair(A.clone(), C.clone(), 2));
    let E = Rc::new(Symbol::new_pair(D, e.clone(), 3));

    let df = DeltaFragment {
      root: E.clone(),
      from: 0,
      to: E.length,
    };

    assert_eq!(
      ParseTreeSlice { content: vec![(1,A.clone()),(1, C.clone()), (1, e.clone())], prefix_len: 0, level: 1},
               df.proxy_text(1,2,4)
    );
    assert_eq!(ParseTreeSlice { content: vec![(1, C.clone())], prefix_len: 2, level: 1},df.proxy_text(1,1,1));
    assert_eq!(ParseTreeSlice { content: vec![(1, A.clone()),(1, C.clone())],prefix_len:0, level:1},df.proxy_text(1,1,2));
    assert_eq!(ParseTreeSlice { content: vec![(1, A.clone()),(1, C.clone()), (1, e.clone())],prefix_len: 0, level: 1},df.proxy_text(1,2,2));

  }

  #[test_case(Some(ArithmeticProgression { a:0, max_i: 1, g: 0}), (0,5), (0,5))]
  #[test_case(Some(ArithmeticProgression { a:0, max_i: 1, g: 0}), (0,4), (0,5))]
  #[test_case(Some(ArithmeticProgression { a:1, max_i: 1, g: 0}), (1,4), (0,5))]
  #[test_case(None, (0,5), (0,4))]
  fn test_ipm_unique(expected: Option<ArithmeticProgression>, X_indices: (u64, u64), Y_indices: (u64, u64)) {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let c = Rc::new(Symbol::new_leaf('c'));
    let d = Rc::new(Symbol::new_leaf('d'));
    let e = Rc::new(Symbol::new_leaf('e'));

    let A = Rc::new(Symbol::new_pair(a, b.clone(), 1));
    let C = Rc::new(Symbol::new_pair(c.clone(), d, 1));
    let D = Rc::new(Symbol::new_pair(A.clone(), C.clone(), 2));
    let E = Rc::new(Symbol::new_pair(D, e.clone(), 3));

    let df = DeltaFragment {
      root: E.clone(),
      from: 0,
      to: E.length,
    };

    assert_eq!(expected, df.extract(X_indices.0, X_indices.1).ipm(&df.extract(Y_indices.0, Y_indices.1)));
  }

  #[test]
  fn test_ipm_unique_runs() {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let C = Rc::new(Symbol::new_run(a, 10, 1));
    let D = Rc::new(Symbol::new_run(b, 10, 1));
    let E = Rc::new(Symbol::new_pair(C, D, 2));


    let df = DeltaFragment {
      root: E.clone(),
      from: 0,
      to: E.length,
    };

    let dfr = DeltaFragment::from_randomized("aaaaaaaaaabbbbbbbbbb".to_string());

    assert_eq!(Some(ArithmeticProgression{a: 5, g:0, max_i: 1}), df.extract(5, 15).ipm(&df.extract(0,19)));
    assert_eq!(Some(ArithmeticProgression{a: 5, g:0, max_i: 1}), dfr.extract(5, 15).ipm(&dfr.extract(0,19)));
  }

  #[test_case(1)]
  #[test_case(9)]
  #[test_case(50)]
  fn test_ipm_run_higher(k: u64) {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let c = Rc::new(Symbol::new_leaf('c'));
    let d = Rc::new(Symbol::new_leaf('d'));
    let ab = Rc::new(Symbol::new_pair(a.clone(), b.clone(),1));
    let cd = Rc::new(Symbol::new_pair(c.clone(), d.clone(),1));
    let left= Rc::new(Symbol::new_run(ab.clone(), 2*k, 2));
    let right = Rc::new(Symbol::new_run(cd, 2*k, 2));
    let root = Rc::new(Symbol::new_pair(left, right, 3));
    let df = DeltaFragment {
      root: root.clone(),
      from: 0,
      to: 8*k
    };

    let text = "ab".repeat(2*k as usize) + &"cd".repeat(2*k as usize);


    let x = df.extract(2*k, 6*k);
    let y= df.extract(0, 8*k-2);
    assert_eq!(Some(ArithmeticProgression{a: 2*k, g:0, max_i: 1}), x.ipm(&y));

    let dfr = DeltaFragment::from_randomized(text);
    let xr = dfr.extract(2*k, 6*k);
    let yr = dfr.extract(0, 8*k-2);
    assert_eq!(Some(ArithmeticProgression{a: 2*k, g:0, max_i: 1}), xr.ipm(&yr));
  }

  #[test]
  fn test_ipm_higher_run() {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let m = Rc::new(Symbol::new_leaf('d'));
    let c = Rc::new(Symbol::new_pair(a, b, 1));
    let d = Rc::new(Symbol::new_run(c.clone(),17,2));
    let e = Rc::new(Symbol::new_run(c.clone(),23,2));
    let f = Rc::new(Symbol::new_pair(d, m, 3));
    let g = Rc::new(Symbol::new_pair(f, e, 4));

    let df = DeltaFragment { root: g.clone(), from: 0, to: g.clone().length };
    let x = df.extract(0, 34);
    let y = df.extract(36, 80);

    assert_eq!(Some(ArithmeticProgression { a: 1, g: 2, max_i: 5}), x.ipm(&y));
  }

  fn rand_subseq(rng: &mut StdRng, g: usize) -> String {
    (0..g).map(|_| if rng.gen_bool(0.5) { "a" } else { "b" }).collect::<Vec<_>>().join("")
  }

  #[test]
  fn test_ipm_automated() {
    for seed in 0..1000 {
      let mut rng = StdRng::seed_from_u64(seed);
      let g = 9;
      let period = rand_subseq(&mut rng, g);

      let xstr = (0..10).map(|_| period.clone()).collect::<Vec<_>>().join("");
      let rand_y = (0..1000).map(|i| if i<=5 || rng.gen_bool(0.5) { period.clone() } else { rand_subseq(&mut rng, g)}).collect::<Vec<_>>().join("");
      let hitting_y = (0..100).map( |i| if rng.gen_bool(0.8) {period.clone()} else {rand_subseq(&mut rng, g)}).collect::<Vec<_>>().join("");
      let text = vec![xstr,rand_y,hitting_y].join("");

      let df = DeltaFragment::from_randomized(text.clone());
      let nf = NaiveFragment::from_randomized(text.clone());
      assert_eq!(text.clone().len() as u64, df.len());
      assert_eq!(text.clone().len() as u64, nf.len());

      for o in 0..100 {
        let mut rngo = StdRng::seed_from_u64(o);
        dbg!(&(seed,o));
        // pick some random values
        let xoffset = rngo.sample(Uniform::new(0,g as u64 -1));
        let xlen = rngo.sample(Uniform::new(1,9*g as u64));
        let yoffset = g as u64 *10 + rngo.sample(Uniform::new(0,990*g as u64-2*xlen));
        let ylen = rngo.sample(Uniform::new(xlen,std::cmp::max(xlen+1,2*xlen-1)));
        assert!(2*xlen>ylen);
        assert!(xlen <= ylen);
        assert!(xoffset+xlen < text.len() as u64);
        assert!(yoffset+ylen < text.len() as u64);

        let dx = df.extract(xoffset, xoffset+xlen);
        let nx = nf.extract(xoffset, xoffset+xlen);
        let dy = df.extract(yoffset, yoffset+ylen);
        let ny = nf.extract(yoffset, yoffset+ylen);

        let naive_res  = nx.ipm(&ny);
        let faster_res = dx.ipm(&dy);

        assert_eq!(naive_res, faster_res);
      }
    }
  }

  #[test_case(57, 97)]
  #[test_case(454,81)]
  fn test_ipm_from_seed(text_seed: u64, xy_seed: u64) {
    let mut rng = StdRng::seed_from_u64(text_seed);
    let g = 9;
    let period = rand_subseq(&mut rng, g);

    let xstr = (0..10).map(|_| period.clone()).collect::<Vec<_>>().join("");
    let rand_y = (0..1000).map(|i| if i<=5 || rng.gen_bool(0.5) { period.clone() } else { rand_subseq(&mut rng, g)}).collect::<Vec<_>>().join("");
    let hitting_y = (0..100).map( |i| if rng.gen_bool(0.8) {period.clone()} else {rand_subseq(&mut rng, g)}).collect::<Vec<_>>().join("");
    let text = vec![xstr,rand_y,hitting_y].join("");

    let df = DeltaFragment::from_randomized(text.clone());
    let nf = NaiveFragment::from_randomized(text.clone());
    assert_eq!(text.clone().len() as u64, df.len());
    assert_eq!(text.clone().len() as u64, nf.len());

    let mut rngo = StdRng::seed_from_u64(xy_seed);
    // pick some random values
    let xoffset = rngo.sample(Uniform::new(0,g as u64 -1));
    let xlen = rngo.sample(Uniform::new(1,9*g as u64));
    let yoffset = g as u64 *10 + rngo.sample(Uniform::new(0,990*g as u64-2*xlen));
    let ylen = rngo.sample(Uniform::new(xlen,std::cmp::max(xlen+1,2*xlen-1)));
    assert!(2*xlen>ylen);
    assert!(xlen <= ylen);
    assert!(xoffset+xlen < text.len() as u64);
    assert!(yoffset+ylen < text.len() as u64);

    let dx = df.extract(xoffset, xoffset+xlen);
    let nx = nf.extract(xoffset, xoffset+xlen);
    let dy = df.extract(yoffset, yoffset+ylen);
    let ny = nf.extract(yoffset, yoffset+ylen);

    let naive_res  = nx.ipm(&ny);
    dbg!(&naive_res);
    let faster_res = dx.ipm(&dy);

    dbg!(&text[xoffset as usize..xoffset as usize+xlen as usize]);
    dbg!(&text[yoffset as usize..yoffset as usize+ylen as usize]);
    assert_eq!(naive_res, faster_res);
  }

  #[test_case(1)]
  #[test_case(2)]
  #[test_case(9)]
  #[test_case(50)]
  fn bad_test(k : u64) {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let c = Rc::new(Symbol::new_leaf('c'));
    let d = Rc::new(Symbol::new_leaf('d'));
    let apow = Rc::new(Symbol::new_run(a.clone(), k, 1));
    let dpow = Rc::new(Symbol::new_run(d.clone(), k, 1));
    let bc = Rc::new(Symbol::new_pair(b.clone(), c.clone(),2));
    let bcpow= Rc::new(Symbol::new_run(bc.clone(), k, 3));
    let abcpow = Rc::new(Symbol::new_pair(apow.clone(), bcpow.clone(), 4));
    let root = Rc::new(Symbol::new_pair(abcpow.clone(), dpow.clone(), 6));
    let df = DeltaFragment {
      root: root.clone(),
      from: 0,
      to: 4*k
    };
    assert_eq!(root.length, 4*k);

    let text = "a".repeat(k as usize) + &"bc".repeat(k as usize) + &"d".repeat(k as usize);


    let x = df.extract(k-1, 3*k+1);
    let y= df.extract(0, 4*k);
    assert_eq!(Some(ArithmeticProgression{a: k-1, g:0, max_i: 1}), x.ipm(&y));

    let dfr = DeltaFragment::from_randomized(text);
    let xr = dfr.extract(k-1, 3*k+1);
    let yr = dfr.extract(0, 4*k);
    assert_eq!(Some(ArithmeticProgression{a: k-1, g:0, max_i: 1}), xr.ipm(&yr));
  }

  #[test]
  fn simple_measure() {
    fn sturmian_prefix(exponent: usize, length: usize) -> String {
      let mut a = String::from("b");
      let mut b = String::from("a");

      while b.len() < length {
        a = b.repeat(exponent)+&a;
        core::mem::swap(&mut a, &mut b);
      }

      b[..length].to_string()
    }

    fn has_candidate(x: DeltaFragment, y: DeltaFragment) -> bool {
      let proxy_pattern = x.proxy_pattern();
      let proxy_text = y.proxy_text(proxy_pattern.level, 2*(proxy_pattern.level as u64)+3, proxy_pattern.len()+(proxy_pattern.level as u64));

      for candidate in common::rle_pattern_matching(&proxy_pattern.content, &proxy_text.content) {
        if candidate.max_i > 0 {
          return true;
        }
      }
      false
    }

    let a = sturmian_prefix(3, 1000000);

    let dfa = DeltaFragment::from_randomized(a);

    let mut rngo = StdRng::from_entropy();

    let mut unsuccess_total = 0;
    let mut unsuccess_has_candidate = 0;

    for _ in 0..10000 {
      let xlen_dist  = Uniform::new(2,dfa.len()/2);
      let xlen : u64 = rngo.sample(xlen_dist);
      let ylen = xlen*2-1;

      let ystart = rngo.sample(Uniform::new(0,dfa.len()-ylen));
      let xstart = rngo.sample(Uniform::new(0,dfa.len()-xlen));

      let y = dfa.extract(ystart, ystart+ylen);
      let x = dfa.extract(xstart,xstart+xlen);

      if x.ipm(&y).is_none() {
        unsuccess_total += 1;
        if has_candidate(x, y) {
          unsuccess_has_candidate += 1;
        }
      }
    }

    // about the same number of success...
    dbg!(unsuccess_total);
    dbg!(unsuccess_has_candidate);

    // assert!(false); // necessary to enable print
  }
}
