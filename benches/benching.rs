use std::hint::black_box;
use rand::{rngs::StdRng, SeedableRng,Rng};

use criterion::{
  criterion_group, criterion_main, Criterion
};
use rand::distributions::Uniform;

use pillar_model::{
  PillarModelFragment,
  restricted_recompression::DeltaFragment
};

fn sturmian_word<const EXPONENT: usize>(nr: usize) -> String {
  let mut a = String::from("b");
  let mut b = String::from("a");

  for _ in 0..nr-2 {
    a = b.repeat(EXPONENT)+&a;
    core::mem::swap(&mut a, &mut b);
  }
  b
}

const NR_OF_REPETITIONS : usize = 100; // bench.iter() already does some repetitions, so we don't need as many of our own
const SEED : u64 = 31415926;

fn bench_ipm<const STRM_NR: usize, const EXPONENT: usize>(c : &mut Criterion) {
  let mut rngo = StdRng::seed_from_u64(SEED);

  let fragment = DeltaFragment::from_randomized(sturmian_word::<EXPONENT>(STRM_NR));
  //println!("{}",fragment.len());
  let xlen_dist  = Uniform::new(1,fragment.len()/2);

  let parts : Vec<(DeltaFragment, DeltaFragment)> = (0..NR_OF_REPETITIONS).map(
    |_| {
      let xlen : u64 = rngo.sample(xlen_dist);
      let ylen = xlen*2-1;

      let ystart = rngo.sample(Uniform::new(0,fragment.len()-ylen));
      let xstart = rngo.sample(Uniform::new(0,fragment.len()-xlen));

      let y = fragment.extract(ystart, ystart+ylen);
      let x = fragment.extract(xstart,xstart+xlen);
      (x,y)
    }
  ).collect::<Vec<_>>();

  let name =
  NR_OF_REPETITIONS.to_string()
  + " ipm queries for string of length "
  + &fragment.len().to_string()
  + " (exp: " + &EXPONENT.to_string()  + ", layers: " + &fragment.layers().to_string() + ")";

  c.bench_function(
    &name,
    |b| {
      b.iter(|| {
        for i in 0..NR_OF_REPETITIONS {
          let x = &parts[i].0;
          let y = &parts[i].1;
          black_box(black_box(x).ipm(black_box(&y)));
        }
      }
      )
    });
}


// const TEXT_STRM_NR : &[usize] = &[5,7,10,12]; // 24 runs out of 32GB RAM + 32GB swap
criterion_group!(benches,
                 bench_ipm::<10 , 1>,
                 bench_ipm::<12 , 1>,
                 bench_ipm::<20 , 1>,
                 bench_ipm::<22 , 1>,

                 bench_ipm::<5  , 5>,
                 bench_ipm::<7  , 5>,
                 bench_ipm::<10 , 5>,
                 bench_ipm::<12 , 5>,
);
criterion_main!(benches);
