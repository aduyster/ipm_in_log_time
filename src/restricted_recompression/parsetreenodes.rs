use crate::restricted_recompression::*;


/// A node of the parse tree
#[derive(Debug)]
pub struct ParseTreeNode {
  /// the position where node's expansion starts
  from: u64,
  /// the symbol associated with the node
  pub symbol: Rc<Symbol>,
  /// the parent of the node (None if the node is the root)
  parent: Option<Rc<ParseTreeNode>>,
}

impl PartialEq for ParseTreeNode {
  fn eq(&self, other: &Self) -> bool {
    Rc::<Symbol>::ptr_eq(&self.symbol, &other.symbol) && self.from == other.from
  }
}

impl ParseTreeNode {
  /// Returns the root of the parse *symbol*
  pub fn root(symbol: &Rc<Symbol>) -> Self {
    ParseTreeNode {
      from: 0,
      symbol: symbol.clone(),
      parent: None,
    }
  }

  /// Returns the *index*-th child of the node
  fn child(self: &Rc<Self>, index: u64) -> Rc<ParseTreeNode> {
    return Rc::new(ParseTreeNode {
      from: self.from + self.symbol.boundary(index),
                   symbol: self.symbol.child(index),
                   parent: Some(self.clone()),
    });
  }

  /// Returns the index of the child whose expansion contains the *leaf_index*-th character
  fn index(&self, leaf_index: u64) -> u64 {
    debug_assert!(self.from <= leaf_index);
    debug_assert!(leaf_index < self.from + self.symbol.length);
    self.symbol.leaf_index_child(leaf_index - self.from).1
  }

  /// Count the number of siblings in the *forward* direction
  pub fn count_next_siblings(&self, forward: bool) -> u64 {
    match &self.parent {
      None => return 0,
      Some(parent) => {
        let my_index = parent.index(self.from);
        let parent_degree = parent.symbol.degree();
        if forward {
          return parent_degree - my_index - 1;
        } else {
          return my_index;
        }
      }
    }
  }

  /// Returns the sibling *steps* away in the *forward* direction. Current node if steps = 0.
  pub fn sibling(&self, steps: u64, forward: bool) -> Rc<ParseTreeNode> {
    match &self.parent {
      None => panic!("Too many steps"),
      Some(parent) => {
        let my_index = parent.index(self.from);
        if forward {
          return parent.child(my_index + steps);
        } else {
          return parent.child(my_index - steps);
        }
      }
    }
  }

  /// Returns the highest node whose expansion start right after/before the expansion of self.
  /// None if self represents a suffix/prefix.
  pub fn next(&self, forward: bool) -> Option<Rc<ParseTreeNode>> {
    if self.count_next_siblings(forward) > 0 {
      return Some(self.sibling(1, forward));
    }
    match &self.parent {
      None => return None,
      Some(parent) => return parent.next(forward),
    }
  }

  /// Returns the leftmost child (if forward = true) or the rightmost child (otherwise)
  pub fn first_child(self: &Rc<Self>, forward: bool) -> Rc<ParseTreeNode> {
    let degree = self.symbol.degree();
    if forward {
      return self.child(0);
    } else {
      return self.child(degree - 1);
    }
  }

  /// Descends to the highest node whose expansion starts/ends at position leaf_index
  pub fn descend(self: &Rc<Self>, leaf_index: u64, forward: bool) -> Rc<ParseTreeNode> {
    if forward {
      if self.from == leaf_index {
        return self.clone();
      } else {
        let idx = self.index(leaf_index);
        return self.child(idx).descend(leaf_index, forward);
      }
    } else {
      if self.from + self.symbol.length == leaf_index {
        return self.clone();
      } else {
        let idx = self.index(leaf_index - 1);
        return self.child(idx).descend(leaf_index, forward);
      }
    }
  }

  /// Descends to the leaf representing position leaf_index
  fn leaf(self: Rc<Self>, leaf_index: u64) -> Rc<ParseTreeNode> {
    if self.symbol.length == 1 {
      return self;
    } else {
      let idx = self.index(leaf_index);
      return self.child(idx).leaf(leaf_index);
    }
  }
}
/// A node of the uncompressed parse tree
pub struct UncompressedParseTreeNode {
  pub level: u16,
  node: Rc<ParseTreeNode>,
}

impl PartialEq for UncompressedParseTreeNode {
  fn eq(&self, other: &Self) -> bool {
    return self.level == other.level && self.node.from == other.node.from;
  }
}

impl std::fmt::Debug for UncompressedParseTreeNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "UParseTreeNode with symbol {:p} at level {} covering interval [{}..{})",
      self.symbol(),
           self.level,
           self.from(),
           self.to()
    )
  }
}

impl UncompressedParseTreeNode {
  fn root(root: Rc<Symbol>) -> Self {
    UncompressedParseTreeNode {
      level: root.level,
      node: Rc::new(ParseTreeNode::root(&root)),
    }
  }

  pub fn leaf(root: Rc<Symbol>, leaf_index: u64) -> Self {
    UncompressedParseTreeNode {
      level: 0,
      node: Rc::new(ParseTreeNode::root(&root)).leaf(leaf_index),
    }
  }

  fn has_proper_children(&self) -> bool {
    return self.level == self.node.symbol.level;
  }

  pub fn length(&self) -> u64 {
    return self.node.symbol.length;
  }

  pub fn from(&self) -> u64 {
    return self.node.from;
  }

  pub fn to(&self) -> u64 {
    return self.node.from + self.length();
  }

  pub fn symbol(&self) -> Rc<Symbol> {
    return self.node.symbol.clone();
  }

  fn has_proper_parent(&self) -> bool {
    match &self.node.parent {
      None => false,
      Some(parent) => self.level + 1 == parent.symbol.level,
    }
  }

  pub fn parent(&self) -> UncompressedParseTreeNode {
    let node = if self.has_proper_parent() {
      match &self.node.parent {
        None => panic!(),
        Some(parent) => parent.clone(),
      }
    } else {
      self.node.clone()
    };
    return UncompressedParseTreeNode {
      level: self.level + 1,
      node: node,
    };
  }

  fn child(&self, index: u64) -> UncompressedParseTreeNode {
    let node = if !self.has_proper_children() {
      self.node.clone()
    } else {
      self.node.clone().child(index)
    };
    return UncompressedParseTreeNode {
      level: self.level - 1,
      node: node,
    };
  }

  pub fn next(&self, forward: bool) -> Option<UncompressedParseTreeNode> {
    return match self.node.next(forward) {
      None => None,
      Some(next_node) => {
        let mut target = next_node;
        while target.symbol.level > self.level {
          target = target.first_child(forward)
        }
        Some(UncompressedParseTreeNode {
          level: self.level,
          node: target,
        })
      }
    };
  }

  pub fn sibling(&self, steps: u64, forward: bool) -> UncompressedParseTreeNode {
    if self.has_proper_parent() {
      return UncompressedParseTreeNode {
        node: self.node.sibling(steps, forward),
        level: self.level,
      };
    } else {
      return UncompressedParseTreeNode {
        node: self.node.clone(),
        level: self.level,
      };
    }
  }

  pub fn my_index(&self) -> u64 {
    if self.has_proper_parent() {
      return self.node.parent.clone().unwrap().index(self.node.from);
    } else {
      return 0;
    }
  }

  pub fn degree(&self) -> u64 {
    if !self.has_proper_children() {
      return 1;
    } else {
      return self.node.symbol.degree();
    }
  }

  pub fn count_next_siblings(&self, forward: bool) -> u64 {
    if self.has_proper_parent() {
      self.node.count_next_siblings(forward)
    } else {
      0
    }
  }

  pub fn is_pair(&self) -> bool {
    if !self.has_proper_children() {
      false
    } else {
      match self.symbol().rhs {
        Production::Pair(_, _) => true,
        _ => false,
      }
    }
  }

  pub fn ps_node(&self, left: u64, right: u64) -> PSNode {
    if right == left {
      return PSNode::Single(self.child(left).symbol());
    } else {
      return PSNode::Run(right - left + 1, self.child(left).symbol());
    }
  }
}
