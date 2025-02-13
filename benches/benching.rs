use std::hint::black_box;
use rand::{rngs::StdRng, SeedableRng,Rng};

use bencher::{
  Bencher,
  benchmark_group,
  benchmark_main,
};
use rand::distributions::Uniform;

use pillar_model::{
  PillarModelFragment,
  restricted_recompression::DeltaFragment
};

const FIB_EXPONENT : usize = 5;

fn fibonacci_string(nr: usize) -> String {
  let mut a = String::from(".");
  let mut b = String::from("X");

  for _ in 0..nr-2 {
    a = a.repeat(FIB_EXPONENT)+&b;
    core::mem::swap(&mut a, &mut b);
  }
  b
}

const NR_OF_REPETITIONS : usize = 100; // bench.iter() already does some repetitions, so we don't need as many of our own
const SEED : u64 = 31415926;

fn bench_ipm<const FIB_NR: usize>(b : &mut Bencher) {
  let mut rngo = StdRng::seed_from_u64(SEED);

  let fragment = DeltaFragment::from_randomized(fibonacci_string(FIB_NR));
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

  b.iter(|| {
    for i in 0..NR_OF_REPETITIONS {
      let x = &parts[i].0;
      let y = &parts[i].1;
      black_box(black_box(x).ipm(black_box(&y)));
    }
  });
}


benchmark_group!(benches, bench_ipm::<6>, bench_ipm::<8>, bench_ipm::<10>, bench_ipm::<12>, bench_ipm::<14>, bench_ipm::<16>, bench_ipm::<18>, bench_ipm::<20>);
benchmark_main!(benches);
