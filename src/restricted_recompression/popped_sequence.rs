use super::*;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum PSNode {
  Run(u64, Rc<Symbol>),
  Single(Rc<Symbol>),
  None, // taking nothing from a level (bc directly in next level above the parent covers both)
}

impl PSNode {
  pub fn length(&self) -> u64 {
    return match self {
      PSNode::Single(x) => x.length,
      PSNode::None => 0,
      PSNode::Run(u, x) => u * x.length,
    };
  }

  pub fn to_run(&self) -> Option<(u64, Rc<Symbol>)> {
    return match self {
      PSNode::Single(x) => Some((1, x.clone())),
      PSNode::None => None,
      PSNode::Run(u, x) => Some((*u, x.clone())),
    };
  }
}

// each list has one entry per level;
// first entry is level 0 (leaf), highest entry is last
// left.len() == right.len() must always be true (add a Nonce if it isn't)
#[derive(Debug, PartialEq, Eq)]
pub struct PoppedSequence {
  pub left: Vec<PSNode>,
  pub right: Vec<PSNode>,
}

impl PoppedSequence {
  pub fn from(df: &DeltaFragment) -> Self {
    let mut left = UncompressedParseTreeNode::leaf(df.root.clone(), df.from);
    let mut right = UncompressedParseTreeNode::leaf(df.root.clone(), df.to - 1);

    let mut left_vec: Vec<PSNode> = vec![];
    let mut right_vec: Vec<PSNode> = vec![];

    loop {
      let left_parent = left.parent();
      let left_index = left.my_index();
      let left_special = left_index == 0 && left_parent.is_pair();
      let right_parent = right.parent();
      let right_index = right.my_index();
      let right_special = right_index == 1 && right_parent.is_pair();

      if left_parent == right_parent && (!left_special || !right_special) {
        left_vec.push(left_parent.ps_node(left_index, right_index));
        right_vec.push(PSNode::None);
        break;
      }
      if left_special {
        left_vec.push(PSNode::None);
        left = left_parent;
      } else {
        left_vec.push(left_parent.ps_node(left_index, left_parent.degree() - 1));
        left = left_parent.next(true).unwrap();
      }
      if right_special {
        right_vec.push(PSNode::None);
        right = right_parent;
      } else {
        right_vec.push(right_parent.ps_node(0, right_index));
        if left == right_parent {
          break;
        }
        right = right_parent.next(false).unwrap();
      }
    }
    return Self {
      left: left_vec,
      right: right_vec,
    };
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::rc::Rc;
  use test_case::test_case;

  #[test]
  fn test_popped_sequence() {
    // 1) create a grammar (hard code the tree here)
    let al = Rc::new(Symbol::new_leaf('a'));
    let bl = Rc::new(Symbol::new_leaf('b'));

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

    // 2) hardcode the ps of multiple (different?) substrings
    let ps = PoppedSequence {
      left: vec![
        PSNode::Single(al.clone()),
        PSNode::Single(bl.clone()),
        PSNode::Single(c),
      ],
      right: vec![
        PSNode::Single(bl.clone()),
        PSNode::Single(al.clone()),
        PSNode::None,
      ],
    };

    // 3) test them against all occurences
    let f1 = df.extract(1, 7);
    let f2 = df.extract(3, 9);
    let f3 = df.extract(5, 11);

    assert_eq!(ps, PoppedSequence::from(&f1));
    assert_eq!(ps, PoppedSequence::from(&f2));
    assert_eq!(ps, PoppedSequence::from(&f3));
  }

  #[test]
  fn test_popped_seq_normal() {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let c = Rc::new(Symbol::new_leaf('c'));
    let d = Rc::new(Symbol::new_leaf('d'));
    let e = Rc::new(Symbol::new_leaf('e'));

    let A = Rc::new(Symbol::new_pair(a, b.clone(), 1));
    let C = Rc::new(Symbol::new_pair(c.clone(), d, 1));
    let D = Rc::new(Symbol::new_pair(A, C.clone(), 2));
    let E = Rc::new(Symbol::new_pair(D, e.clone(), 3));

    let df = DeltaFragment {
      root: E.clone(),
      from: 0,
      to: E.length,
    };

    let ps1 = PoppedSequence {
      left: vec![PSNode::Single(b.clone()), PSNode::Single(C.clone())],
      right: vec![PSNode::Single(e), PSNode::None],
    };
    assert_eq!(ps1, PoppedSequence::from(&df.extract(1, df.len())));
    let ps2 = PoppedSequence {
      left: vec![PSNode::Single(b)],
      right: vec![PSNode::Single(c)],
    };
    assert_eq!(ps2, PoppedSequence::from(&df.extract(1, 3)));
  }

  #[test]
  fn test_popped_seq_high() {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let c = Rc::new(Symbol::new_leaf('c'));
    let d = Rc::new(Symbol::new_leaf('d'));
    let e = Rc::new(Symbol::new_leaf('e'));

    let A = Rc::new(Symbol::new_pair(a, b.clone(), 1));
    let C = Rc::new(Symbol::new_pair(c.clone(), d, 1));
    let D = Rc::new(Symbol::new_pair(A.clone(), C.clone(), 2));
    let E = Rc::new(Symbol::new_pair(D.clone(), e.clone(), 3));

    let df = DeltaFragment {
      root: E.clone(),
      from: 0,
      to: E.length,
    };

    let ps1 = PoppedSequence {
      left: vec![PSNode::None, PSNode::None, PSNode::Single(D.clone())],
      right: vec![PSNode::Single(e.clone()), PSNode::None, PSNode::None],
    };
    let ps2 = PoppedSequence {
      left: vec![PSNode::Single(b.clone()), PSNode::Single(C.clone())],
      right: vec![PSNode::None, PSNode::None],
    };
    assert_eq!(ps1, PoppedSequence::from(&df));
    assert_eq!(ps2, PoppedSequence::from(&df.extract(1, 4)));
  }

  #[test]
  fn mini_ps_run() {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let A = Rc::new(Symbol::new_run(a.clone(), 2, 1));
    let B = Rc::new(Symbol::new_run(b.clone(), 2, 1));
    let df = DeltaFragment {
      root: Rc::new(Symbol::new_pair(A.clone(), B.clone(), 2)),
      from: 0,
      to: 4,
    };

    assert_eq!(
      PoppedSequence {
        left: vec![PSNode::Run(2, a.clone())],
               right: vec![PSNode::Run(2, b.clone())]
      },
      PoppedSequence::from(&df)
    );
  }

  #[test]
  fn test_popped_seq_runs() {
    let a = Rc::new(Symbol::new_leaf('a'));
    let b = Rc::new(Symbol::new_leaf('b'));
    let A = Rc::new(Symbol::new_run(a.clone(), 4, 1));
    let B = Rc::new(Symbol::new_run(a.clone(), 2, 1));
    let C = Rc::new(Symbol::new_pair(b.clone(), A.clone(), 2));
    let D = Rc::new(Symbol::new_pair(b.clone(), B.clone(), 2));
    let E = Rc::new(Symbol::new_pair(D.clone(), b.clone(), 3));
    let F = Rc::new(Symbol::new_pair(C.clone(), E, 4));

    let df = DeltaFragment {
      root: F.clone(),
      from: 0,
      to: F.length,
    };

    let ps1 = PoppedSequence {
      left: vec![PSNode::Run(3, a.clone())],
      right: vec![PSNode::None],
    };
    let ps2 = PoppedSequence {
      left: vec![PSNode::Run(3, a.clone())],
      right: vec![PSNode::None],
    };
    let ps3 = PoppedSequence {
      left: vec![PSNode::Run(4, a.clone()), PSNode::Single(b.clone())],
      right: vec![PSNode::Run(2, a.clone()), PSNode::None],
    };
    let ps4 = PoppedSequence {
      left: vec![PSNode::Run(3, a.clone()), PSNode::Single(b.clone())],
      right: vec![PSNode::Run(2, a.clone()), PSNode::None],
    };
    let ps5 = PoppedSequence {
      left: vec![PSNode::Single(b.clone())],
      right: vec![PSNode::Run(2, a.clone())],
    };
    let ps6 = PoppedSequence {
      left: vec![PSNode::Run(4, a.clone()), PSNode::Single(b.clone())],
      right: vec![PSNode::Single(a.clone()), PSNode::None],
    };

    assert_eq!(ps1, PoppedSequence::from(&df.extract(1, 4)));
    assert_eq!(ps2, PoppedSequence::from(&df.extract(2, 5)));
    assert_eq!(ps3, PoppedSequence::from(&df.extract(1, 8)));
    assert_eq!(ps4, PoppedSequence::from(&df.extract(2, 8)));
    assert_eq!(ps5, PoppedSequence::from(&df.extract(5, 8)));
    assert_eq!(ps6, PoppedSequence::from(&df.extract(1, 7)));
  }
}
