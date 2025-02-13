use super::*;

fn lce_internal(
  p: Option<Rc<ParseTreeNode>>,
  q: Option<Rc<ParseTreeNode>>,
  forward: bool,
) -> u64 {
  match (p, q) {
    (None, _) => return 0,
    (_, None) => return 0,
    (Some(P), Some(Q)) => {
      let plen = P.symbol.length;
      let qlen = Q.symbol.length;
      if plen > qlen {
        return lce_internal(Some(P.first_child(forward)), Some(Q), forward);
      }
      if plen < qlen {
        return lce_internal(Some(P), Some(Q.first_child(forward)), forward);
      }
      if !Rc::ptr_eq(&P.symbol, &Q.symbol) {
        if plen == 1 && qlen == 1 {
          return 0;
        } else {
          return lce_internal(
            Some(P.first_child(forward)),
                              Some(Q.first_child(forward)),
                              forward,
          );
        }
      }
      let next = std::cmp::min(
        P.count_next_siblings(forward),
                               Q.count_next_siblings(forward),
      );
      if next == 0 {
        return plen + lce_internal(P.next(forward), Q.next(forward), forward);
      } else {
        return plen * next
        + lce_internal(
          Some(P.sibling(next, forward)),
                       Some(Q.sibling(next, forward)),
                       forward,
        );
      }
    }
  }
}

pub fn lce(A: Rc<Symbol>, i: u64, B: Rc<Symbol>, j: u64, forward: bool) -> u64 {
  let P = Rc::new(ParseTreeNode::root(&A)).descend(i, forward);
  let Q = Rc::new(ParseTreeNode::root(&B)).descend(j, forward);
  return lce_internal(Some(P), Some(Q), forward);
}
