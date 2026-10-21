//! Usage: normalize_file <in.ptx> <lib.ptx> <out.ptx>
//!
//! Normalizes one PTX file with the ECHO normalizer. `lib.ptx` is the ECHO library
//! (or an extern-declaration stub); its `.version`, `.target` and `.address_size`
//! lines are dropped before insertion. Pass `-` for no library.
//! Prints one summary line; exits 1 with the normalizer's error on failure.

use echo_core::ptx::{hazards, normalize};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: {} <in.ptx> <lib.ptx|-> <out.ptx>", args[0]);
        std::process::exit(2);
    }
    let input = std::fs::read_to_string(&args[1]).expect("read input");
    let lib = if args[2] == "-" {
        String::new()
    } else {
        let raw = std::fs::read_to_string(&args[2]).expect("read lib");
        raw.lines()
            .filter(|l| {
                let t = l.trim_start();
                !(t.starts_with(".version")
                    || t.starts_with(".target")
                    || t.starts_with(".address_size"))
            })
            .map(|l| format!("{l}\n"))
            .collect()
    };
    match normalize(&input, &lib) {
        Ok(n) => {
            std::fs::write(&args[3], &n.ptx).expect("write output");
            let left = hazards(&n.ptx);
            println!(
                "OK {} hazards_after={} raw_scan_whole_output={} finalize_kernels={:?} counts={:?}",
                args[1],
                left.len(),
                echo_core::ptx::forbidden_tokens(&n.ptx).len(),
                n.finalize_kernels,
                n.counts
            );
            for h in left.iter().take(5) {
                println!("  left: {}:{}: {}", h.kernel, h.line, h.instruction);
            }
        }
        Err(e) => {
            println!("ERR {} {e}", args[1]);
            std::process::exit(1);
        }
    }
}
