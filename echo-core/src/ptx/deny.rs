//! Raw-text deny scan (controller ruling R22).
//!
//! An independent backstop behind the lexer: after normalization, the user part
//! of the output is scanned as plain text, ignoring statement structure, so
//! comments, strings and directives are scanned as well. Any match is an error.
//! A hole in the lexer therefore fails closed for these hazard classes.
//!
//! Patterns (regex notation, `\b` = boundary between `[A-Za-z0-9_]` and anything else):
//! - `.approx` anywhere; `div.full` anywhere
//! - `\b(ex2|lg2|sin|cos|tanh|rsqrt)\.`
//! - `\b(mul|add|sub|mad|fma)(\.(ftz|sat))*\.(f16|f16x2|bf16|bf16x2|f32|f32x2|f64)\b`
//!   (float arithmetic without an explicit rounding mode)
//! - `\b(atom|red)\.[^;\s]*\.(f16|f16x2|bf16|bf16x2|f32|f32x2|f64)\b`
//! - `%(clock|clock64|globaltimer\w*|smid|nsmid|warpid|nwarpid|gridid)\b`
//! - `\bactivemask\b`

const MATHS: &[&str] = &["ex2", "lg2", "sin", "cos", "tanh", "rsqrt"];
const ARITH: &[&str] = &["mul", "add", "sub", "mad", "fma"];
// Longer names first only matters for readability: every candidate is tried.
const FLOATS: &[&str] = &["f16x2", "bf16x2", "f32x2", "f16", "bf16", "f32", "f64"];
const SPECIAL: &[&str] = &[
    "clock", "clock64", "smid", "nsmid", "warpid", "nwarpid", "gridid",
];

fn is_word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// True if `b[at..]` starts with `.` + a float type followed by a word boundary.
fn float_suffix_at(b: &[u8], at: usize) -> bool {
    if b.get(at) != Some(&b'.') {
        return false;
    }
    FLOATS.iter().any(|t| {
        let end = at + 1 + t.len();
        b.len() >= end && &b[at + 1..end] == t.as_bytes() && b.get(end).is_none_or(|&c| !is_word(c))
    })
}

/// Forbidden tokens on one line (byte-oriented; PTX is ASCII).
pub fn scan_line(line: &str) -> Vec<String> {
    let b = line.as_bytes();
    let mut found = Vec::new();
    for pat in [".approx", "div.full"] {
        if line.contains(pat) {
            found.push(pat.to_string());
        }
    }
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let mut j = i + 1;
            while j < b.len() && is_word(b[j]) {
                j += 1;
            }
            let w = &line[i + 1..j];
            if SPECIAL.contains(&w) || w.starts_with("globaltimer") {
                found.push(format!("%{w}"));
            }
            // '%' is not a word character, so the word after it is scanned next
            i += 1;
            continue;
        }
        if !is_word(b[i]) || (i > 0 && is_word(b[i - 1])) {
            i += 1;
            continue;
        }
        // a maximal word starting at a boundary
        let mut j = i;
        while j < b.len() && is_word(b[j]) {
            j += 1;
        }
        let w = &line[i..j];
        if MATHS.contains(&w) && b.get(j) == Some(&b'.') {
            found.push(format!("{w}."));
        } else if ARITH.contains(&w) {
            let mut k = j;
            loop {
                if float_suffix_at(b, k) {
                    found
                        .push(format!("{w}{}", &line[j..k]) + "<float type without rounding mode>");
                    break;
                }
                let next = &b[k..];
                if next.starts_with(b".ftz") || next.starts_with(b".sat") {
                    k += 4;
                } else {
                    break;
                }
            }
        } else if (w == "atom" || w == "red") && b.get(j) == Some(&b'.') {
            let mut end = j;
            while end < b.len() && b[end] != b';' && !b[end].is_ascii_whitespace() {
                end += 1;
            }
            if (j..end).any(|k| float_suffix_at(b, k)) {
                found.push(line[i..end].to_string());
            }
        } else if w == "activemask" {
            found.push(w.to_string());
        }
        i = j;
    }
    found
}

/// Every forbidden token in `text`, as (0-based line index, token).
pub fn forbidden_tokens(text: &str) -> Vec<(usize, String)> {
    text.split('\n')
        .enumerate()
        .flat_map(|(n, l)| scan_line(l).into_iter().map(move |t| (n, t)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::scan_line;

    #[test]
    fn patterns() {
        for bad in [
            "ex2.approx.f32 %f1, %f2;",
            "// uses ex2.approx",
            "\"sin.\"",
            "mul.f32 %f1",
            "add.ftz.sat.f32",
            "fma.f64",
            "sub.f16x2",
            "add.f32x2 %rd1, %rd2, %rd3;",
            "mul.ftz.f32x2",
            "red.global.add.f32x2",
            "atom.global.add.f32",
            "red.shared.add.noftz.bf16x2",
            "mov.u64 %rd1, %clock64;",
            "%globaltimer_lo",
            "%smid",
            "activemask.b32 %r1;",
            "div.full.f32",
            "rsqrt.rn.f64",
        ] {
            assert!(!scan_line(bad).is_empty(), "{bad}");
        }
        for ok in [
            "mul.rn.ftz.f32 %f3, %f1, %f2;",
            "add.u64 %echo_a, %rd5, 4;",
            "fma.rn.f32",
            "call (echo_r), __echo_ex2f, (echo_p);",
            "atom.global.add.u32 %r1, [%rd1], 1;",
            "atom.global.cas.b64",
            "mov.u32 %r1, %clock_hi;",
            "// .globl cosine_kernel",
            "mul.wide.s32 %rd5, %r1, 4;",
            "sqrt.rn.f32 %f1, %f2;",
            "my_sin.x",
            "cvt.rn.f32.f64",
        ] {
            assert!(scan_line(ok).is_empty(), "{ok}: {:?}", scan_line(ok));
        }
    }
}
