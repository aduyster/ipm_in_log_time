use std::{
  collections::{HashMap, HashSet}, hint::black_box, rc::Rc
};
use rand::{
  distributions::Uniform,
  rngs::StdRng,
  Rng, SeedableRng
};

use criterion::{
  criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion
};

use pillar_model::{
  restricted_recompression::{
    DeltaFragment,
    Symbol
  },
  PillarModelFragment,
};

// Docs: https://bheisler.github.io/criterion.rs/book/user_guide/comparing_functions.html

fn sturmian_prefix(exponent: usize, length: usize) -> String {
  let mut a = String::from("b");
  let mut b = String::from("a");

  while b.len() < length {
    a = b.repeat(exponent)+&a;
    core::mem::swap(&mut a, &mut b);
  }

  b[..length].to_string()
}


// copy from the `from_randomized` function in Deltafragment, just without any pausing
fn construct_fragment_without_pausing(s: String) -> DeltaFragment {
  let mut rng = StdRng::from_entropy();
  let mut current_level: Vec<Rc<Symbol>> = {
    let mut leaves = HashMap::<char, Rc<Symbol>>::new();

    s.chars()
    .map(|c| Rc::clone(leaves.entry(c).or_insert(Rc::new(Symbol::new_leaf(c)))))
    .collect()
  };
  let mut next_level = vec![];

  let mut level = 0;
  while current_level.len() > 1 {
    // let paused_length = compute_paused_length(level);

    if level % 2 == 1 {
      let mut left = HashSet::<Rc<Symbol>>::new();
      let mut right = HashSet::<Rc<Symbol>>::new();
      let mut last_left = false;
      let mut symbols_for_next_level = HashMap::<(Rc<Symbol>, Rc<Symbol>), Rc<Symbol>>::new();

      for node in current_level {
        // do pair compression

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

fn create_vals(rngo: &mut StdRng, fragment: &DeltaFragment) -> (DeltaFragment, DeltaFragment) {
  let xlen_dist  = Uniform::new(1,fragment.len()/2);
  let xlen : u64 = rngo.sample(xlen_dist);
  let ylen = xlen*2-1;

  let ystart = rngo.sample(Uniform::new(0,fragment.len()-ylen));
  let xstart = rngo.sample(Uniform::new(0,fragment.len()-xlen));

  let y = fragment.extract(ystart, ystart+ylen);
  let x = fragment.extract(xstart,xstart+xlen);
  (x,y)
}

fn generate_input(exponent: usize, pausing: bool, power: usize) -> DeltaFragment {
  let len = 2_usize.pow(power as u32);
  let text = sturmian_prefix(exponent, len);
  if pausing {
    DeltaFragment::from_randomized(text)
  } else {
    construct_fragment_without_pausing(text)
  }
}

fn len_bench<const IS_SUCCESS_FORCED : bool, const SUCCESS_VAL : bool>(b : &mut criterion::Bencher, fragment: &DeltaFragment) { // has to be &ARGTYPE since this function is called by bench_with_input
  let mut rngo = StdRng::from_entropy();

  b.iter_batched(
    || {
      let (mut x, mut y) = create_vals(&mut rngo, &fragment);

      if IS_SUCCESS_FORCED {
        while SUCCESS_VAL != x.ipm(&y).is_some() {
          (x,y) = create_vals(&mut rngo, &fragment);
        }
      }
      (x,y)
    },
    |(x,y)| {
      black_box(black_box(x).ipm(black_box(&y)));
    },
    BatchSize::SmallInput
  )
}

fn bench_by_exp(c : &mut Criterion) {
  let groupname = "IPM Queries on Sturmian Prefixes";
  let mut group = c.benchmark_group(groupname);

  for power in POWERS.step_by(STEP_SIZE) {
    group.bench_with_input(BenchmarkId::new("sturmian power: 1", power), &{ let input = generate_input(1,PAUSING_ALGO,power); input}, len_bench::<false,false>);
    group.bench_with_input(BenchmarkId::new("sturmian power: 3", power), &{ let input = generate_input(3,PAUSING_ALGO,power); input}, len_bench::<false,false>);
    group.bench_with_input(BenchmarkId::new("sturmian power: 5", power), &{ let input = generate_input(5,PAUSING_ALGO,power); input}, len_bench::<false,false>);
  }

  group.finish();
}

fn bench_by_success(c : &mut Criterion) {
  let groupname = "Success of IPM Queries";
  let mut group = c.benchmark_group(groupname);

  for power in POWERS.step_by(STEP_SIZE) {
    group.bench_with_input(BenchmarkId::new("ipm successfull", power), &{ let input = generate_input(3,PAUSING_ALGO, power); input}, len_bench::<true,true>);
    group.bench_with_input(BenchmarkId::new("ipm unsuccessfull", power), &{ let input = generate_input(3,PAUSING_ALGO, power); input}, len_bench::<true,false>);
    //group.bench_with_input(BenchmarkId::new("any ipm", power), &{ let input = generate_input(3,PAUSING_ALGO,power); input}, len_bench::<false,false>);
  }

  group.finish();
}

fn bench_by_pausing(c : &mut Criterion) {
  let groupname = "IPM Queries on (non)-restricted recompression RLSLPs";
  let mut group = c.benchmark_group(groupname);

  for power in POWERS.step_by(STEP_SIZE) {
    group.bench_with_input(BenchmarkId::new("restricted", &power), &{ let input = generate_input(3,PAUSING_ALGO,power); input}, len_bench::<false,false>);
    group.bench_with_input(BenchmarkId::new("non-restricted", &power), &{ let input = generate_input(3,NO_PAUSING_ALGO,power); input}, len_bench::<false,false>);
  }

  group.finish();
}

// complex types don't work for generics... (i.e. using Option<bool> for forced (non)success, or directly giving the create function)
const PAUSING_ALGO : bool = true;
const NO_PAUSING_ALGO : bool = false;
const POWERS : std::ops::Range<usize> = 4..33; // fails on values <=1
const STEP_SIZE : usize = 1;

criterion_group!(ipm_by_exp,
                 bench_by_exp
);
criterion_group!(ipm_by_success,
                 bench_by_success
);
criterion_group!(ipm_by_pausing,
                 bench_by_pausing
);
criterion_main!(
  // ipm_by_exp, // no idea what we are measuring this for
  ipm_by_pausing,
  ipm_by_success,
);
