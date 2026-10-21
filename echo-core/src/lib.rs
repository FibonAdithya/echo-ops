//! ECHO core: everything that does not need a GPU.
//! Modules: PTX normalizer, exact arithmetic, maths library, property checks, CPU oracles.

pub mod exact; // Task 5
pub mod math; // Task 4
pub mod oracle; // Task 7
pub mod prop; // Task 6
pub mod ptx; // Task 3
