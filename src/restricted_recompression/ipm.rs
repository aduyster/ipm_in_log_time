use super::*;

pub fn validate_occurence(
  mut candidate: ArithmeticProgression,
  start_of_pp_in_x: u64,
  tail_len_pp: u64,
  u:    u64,
  ubar: u64,
  v:    u64,
  vbar: u64,
)
-> Option<ArithmeticProgression>
{
  if ubar == start_of_pp_in_x && u == tail_len_pp {
    let min_period = (ubar.saturating_sub(vbar) + candidate.g -1)/candidate.g;
    let cutoff_periods = (u.saturating_sub(v) + candidate.g -1)/candidate.g;

    if cutoff_periods + min_period >= candidate.max_i { return None; }
    let max_period = candidate.max_i - cutoff_periods;

    candidate.a += candidate.g * min_period;
    candidate.max_i = max_period - min_period;
  } else {
    if ubar < start_of_pp_in_x {
      if vbar > ubar { return None; }
      candidate.a += ubar-vbar;
    } else {
      if u+1<v { return None; }
      candidate.a += (candidate.max_i-1)*candidate.g +v -u;
    }
    candidate.max_i = 1;
  }

  // single check is done outside
  return if candidate.max_i >= 1 {
    Some(candidate)
  } else {
    None
  };
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParseTreeSlice {
  pub level: u16,
  pub prefix_len: u64,
  pub content: Vec<(u64,Rc<Symbol>)>
}

impl ParseTreeSlice {
  pub fn expanded_len(&self) -> u64 {
    self.content
    .iter()
    .map(|(r, symb)| r * symb.length)
    .sum()
  }

  pub fn len(&self) -> u64 {
    self.content
    .iter()
    .map(|(r, symb)| r * symb.length)
    .sum()
  }

  pub fn complete_run_sizes(&self) -> Vec<u64> {
    let mut pt_complete_run_sizes = vec![0];
    for index in 1..self.content.len() {
      pt_complete_run_sizes.push(pt_complete_run_sizes[index-1] + self.content[index-1].0 * self.content[index-1].1.length);
    }
    pt_complete_run_sizes
  }
}

/// helper function for proxy text construction
pub fn walk(
  start: &UncompressedParseTreeNode,
  forward: bool,
    max_len: u64,
    max_par_len: u64,
    max_exp_len: u64,
) -> Vec<(u64, Rc<Symbol>)> {
  let mut res: Vec<(u64, Rc<Symbol>)> = vec![];
  let mut len: u64 = 0;
  let mut par_len: u64 = 0;
  let mut exp_len: u64 = 0;
  let mut cur = start.sibling(0, forward);
  while par_len < max_par_len {
    let nxt = cur.count_next_siblings(forward);
    let limit = std::cmp::min(max_len - len, (max_exp_len - exp_len) / cur.length());
    let steps = std::cmp::min(limit, nxt);
    if steps > 0 {
      len = len + steps;
      exp_len = exp_len + steps * cur.length();
      res.push((steps, cur.symbol()));
      cur = cur.sibling(steps, forward);
    } else if limit > 0 {
      par_len = par_len + 1;
      len = len + 1;
      exp_len = exp_len + cur.length();
      res.push((1, cur.symbol()));
      match cur.next(forward) {
        None => return res,
        Some(next) => cur = next,
      }
    } else {
      return res;
    }
  }
  return res;
}

pub fn compute_g (g : u64, proxy_pattern: &Vec<(u64, Rc<Symbol>)>) -> u64 {
  return if g == 0 {
    0
  } else if proxy_pattern.len() > 1 && &proxy_pattern[g as usize].1 == &proxy_pattern[0].1 && proxy_pattern[g as usize].0 > proxy_pattern[0].0 {
    let offset = proxy_pattern[0].1.length * (proxy_pattern[g as usize].0 - proxy_pattern[0].0);
    proxy_pattern[0..g as usize].iter().map(|(r,symb)| r * symb.length).sum::<u64>() + offset
  } else if proxy_pattern.len() > 1 {
    proxy_pattern[0..g as usize].iter().map(|(r,symb)| r * symb.length).sum::<u64>()
  } else {
    proxy_pattern[0].1.length
  };
}

pub fn compute_a (candidate : &mut ArithmeticProgression, proxy_pattern: &ParseTreeSlice, proxy_text: &ParseTreeSlice) -> Option<u64> {
  let start_of_candidate_in_y = if proxy_pattern.content.len() > 1 {
    let offset_from_first_hit_run = (proxy_text.content[candidate.a as usize].0 - proxy_pattern.content[0].0) * proxy_pattern.content[0].1.length;

    let upper = proxy_text.prefix_len + proxy_text.complete_run_sizes()[candidate.a as usize] + offset_from_first_hit_run;
    if upper >= proxy_pattern.prefix_len {
      upper - proxy_pattern.prefix_len
    } else if candidate.g != 0 {
      let i = (proxy_pattern.prefix_len + candidate.g - upper -1) / candidate.g;
      if candidate.max_i < i+1 {
        return None;
      }
      candidate.max_i -= i;
      upper + i * candidate.g - proxy_pattern.prefix_len
    } else {
      return None;
    }
  } else {
    if proxy_pattern.prefix_len > proxy_text.prefix_len + proxy_text.complete_run_sizes()[candidate.a as usize] {
      if candidate.g == 0 { return None; }

      let min_period = (proxy_pattern.prefix_len - proxy_text.prefix_len - proxy_text.complete_run_sizes()[candidate.a as usize] + candidate.g-1) / candidate.g;

      if candidate.max_i <= min_period { return None; }
      candidate.max_i -= min_period;

      proxy_text.prefix_len + proxy_text.complete_run_sizes()[candidate.a as usize] + min_period * candidate.g - proxy_pattern.prefix_len
    } else {
      proxy_text.prefix_len + proxy_text.complete_run_sizes()[candidate.a as usize] - proxy_pattern.prefix_len
    }
  };
  Some(start_of_candidate_in_y)
}
