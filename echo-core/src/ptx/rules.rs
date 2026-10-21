//! Rewrite rules and the opcode allowlist (Task 3).
//!
//! `classify` decides, for one instruction, whether it is kept, rewritten into a
//! deterministic equivalent, or rejected. `normalize` applies the rewrites and
//! `hazards` reports every instruction that is not `Keep`.

use super::lex::Instr;

/// Opcode bases that may appear in normalized PTX.
///
/// Source: every opcode base in the 39 algorithm PTX files plus TIG's
/// instrumented challenge PTX (`build_ptx nn_iza`), CUDA 12.8, TIG flags
/// (`tools/ptx_inventory.py --bases`), reviewed by hand. Removed as hazards:
/// `ex2 lg2 sin cos rsqrt` (only `.approx` forms exist; they are rewritten into
/// calls / sqrt+rcp and are never kept). Added: `call`, which the rewrites emit.
/// `atom` is further restricted to `ALLOWED_ATOMS`; `div rcp sqrt` stay because
/// their `.rn` forms are IEEE-exact (`.approx`/`.full` are rewritten first).
pub const ALLOWLIST: &[&str] = &[
    "abs", "add", "and", "atom", "bar", "bra", "call", "cvt", "cvta", "div", "fma", "ld", "mad",
    "max", "membar", "min", "mov", "mul", "neg", "not", "or", "rcp", "rem", "ret", "selp", "setp",
    "shf", "shfl", "shl", "shr", "sqrt", "st", "sub", "xor",
];

/// Integer atomics that are allowed as they are: `add.u32` / `max.u32` (the
/// observed corpus) and `add.u64` / `xor.b64` (TIG's fuel/signature injection).
/// Every other `atom` is rejected, including `atom.global.cas.b64` in user code
/// (ruling R39: no algorithm in the corpus uses it; the ECHO library, which does,
/// is spliced in un-normalized under R14).
pub const ALLOWED_ATOMS: &[&str] = &[
    "atom.global.add.u32",
    "atom.global.add.u64",
    "atom.global.max.u32",
    "atom.global.xor.b64",
];

/// Symbol prefix reserved for the trusted ECHO library (controller ruling R14).
pub const RESERVED_PREFIX: &str = "__echo_";

/// ECHO library functions that user code may call (the rewrites emit these calls).
pub const PUBLIC_CALLS: &[&str] = &[
    "__echo_ex2f",
    "__echo_lg2f",
    "__echo_sinf",
    "__echo_cosf",
    "__echo_atomic_add_f32",
];

/// The callee of a `call` instruction: the first operand that is not a
/// parenthesised parameter list.
pub fn callee(ins: &Instr) -> Option<&str> {
    ins.operands
        .iter()
        .map(String::as_str)
        .find(|o| !o.starts_with('('))
}

/// Special registers whose value does not depend on the GPU model or on scheduling.
const ALLOWED_SPECIAL: &[&str] = &["%tid", "%ntid", "%ctaid", "%nctaid", "%laneid"];

/// PTX ISA special registers (prefix match handles `%pm0`, `%envreg3`,
/// `%globaltimer_lo`, `%cluster_ctaid`, `%reserved_smem_offset_0`, ...).
const SPECIAL_PREFIXES: &[&str] = &[
    "%tid",
    "%ntid",
    "%laneid",
    "%warpid",
    "%nwarpid",
    "%ctaid",
    "%nctaid",
    "%smid",
    "%nsmid",
    "%gridid",
    "%is_explicit_cluster",
    "%clusterid",
    "%nclusterid",
    "%cluster_",
    "%lanemask_",
    "%clock",
    "%pm",
    "%envreg",
    "%globaltimer",
    "%reserved_smem",
    "%total_smem_size",
    "%aggr_smem_size",
    "%dynamic_smem_size",
    "%current_graph_exec",
];

const FLOAT_TYPES: &[&str] = &[
    "f16", "f16x2", "bf16", "bf16x2", "f32", "f32x2", "f64", "e4m3x2", "e5m2x2",
];
const ROUNDING: &[&str] = &["rn", "rz", "rm", "rp", "rna", "rs"];

pub fn is_float_type(part: &str) -> bool {
    FLOAT_TYPES.contains(&part)
}

pub fn is_special_register(name: &str) -> bool {
    SPECIAL_PREFIXES.iter().any(|p| name.starts_with(p))
}

pub fn is_allowed_special(name: &str) -> bool {
    ALLOWED_SPECIAL.contains(&name) || name.starts_with("%lanemask_")
}

/// What the normalizer does with one instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Keep,
    /// Replace by one instruction; the predicate stays in place.
    Single {
        instr: Instr,
        kind: String,
    },
    /// Replace by several lines; a predicate becomes a branch around them.
    Multi {
        body: Vec<String>,
        kind: String,
        atomic: bool,
    },
    Reject(String),
}

/// Function-level facts that `classify` needs.
pub trait FnCtx {
    /// True if `reg` (a `%name`) is declared by a `.reg` directive in the function.
    fn declared(&self, reg: &str) -> bool;
    /// True if `token` appears in any statement of the function other than the
    /// current one and other than `.reg` declarations.
    fn used_elsewhere(&self, token: &str) -> bool;
}

fn f32_approx_form(parts: &[&str], base: &str) -> Option<bool> {
    // [base, "approx", ("ftz")?, "f32"]  -> Some(ftz)
    match parts {
        [b, "approx", "f32"] if *b == base => Some(false),
        [b, "approx", "ftz", "f32"] if *b == base => Some(true),
        _ => None,
    }
}

fn ftz(f: bool) -> &'static str {
    if f {
        ".ftz"
    } else {
        ""
    }
}

pub fn classify(ins: &Instr, ctx: &dyn FnCtx) -> Action {
    let parts = ins.parts();
    let base = ins.base();
    let ops = &ins.operands;

    // The __echo_ prefix belongs to the trusted library. User code may only call
    // its public entry points, exactly as the rewrites below do.
    for (k, op) in ops.iter().enumerate() {
        if let Some(id) = super::lex::find_prefixed_ident(op, RESERVED_PREFIX) {
            let public_call = base == "call"
                && callee(ins) == Some(op.as_str())
                && PUBLIC_CALLS.contains(&op.as_str())
                && ops.iter().enumerate().all(|(j, o)| {
                    j == k || super::lex::find_prefixed_ident(o, RESERVED_PREFIX).is_none()
                });
            if !public_call {
                return Action::Reject(format!(
                    "symbol {id} uses the reserved ECHO prefix __echo_"
                ));
            }
        }
    }

    // clock()/clock64(): only a plain move is rewritten, any other use is rejected below.
    if base == "mov" && ops.len() == 2 {
        let ty = parts.last().copied().unwrap_or("");
        let clk64 = ops[1] == "%clock64" && matches!(ty, "u64" | "b64" | "s64");
        let clk32 = ops[1] == "%clock" && matches!(ty, "u32" | "b32" | "s32");
        if parts.len() == 2 && (clk64 || clk32) {
            let mut i = ins.clone();
            i.operands[1] = "0".to_string();
            return Action::Single {
                instr: i,
                kind: "clock".into(),
            };
        }
    }

    for op in ops {
        for reg in super::lex::percent_idents(op) {
            if is_allowed_special(&reg) {
                continue;
            }
            if is_special_register(&reg) {
                return Action::Reject(format!(
                    "special register {reg} differs between GPUs or runs"
                ));
            }
            if !ctx.declared(&reg) {
                return Action::Reject(format!(
                    "undeclared register or unknown special register {reg}"
                ));
            }
        }
    }
    if let Some(p) = &ins.pred {
        let reg = p.trim_start_matches('!');
        if reg.starts_with('%') && !ctx.declared(reg) {
            return Action::Reject(format!("undeclared predicate register {reg}"));
        }
    }

    if base == "activemask" {
        return Action::Reject("activemask depends on warp scheduling".into());
    }

    if base == "atom" || base == "red" {
        let is_float = parts.iter().any(|p| is_float_type(p));
        if ins.opcode == "atom.global.add.f32" && ops.len() == 3 {
            let x = &ops[0];
            if x != "_" && ctx.used_elsewhere(x) {
                return Action::Reject("float atomicAdd return value is used".into());
            }
            return match atomic_body(&ops[1], &ops[2]) {
                Some(body) => Action::Multi {
                    body,
                    kind: ins.opcode.clone(),
                    atomic: true,
                },
                None => Action::Reject(format!("unsupported address operand {}", ops[1])),
            };
        }
        if is_float {
            return Action::Reject(
                "floating-point atomic other than atom.global.add.f32 (result depends on thread order)"
                    .into(),
            );
        }
        if !ALLOWED_ATOMS.contains(&ins.opcode.as_str()) {
            return Action::Reject(format!(
                "atomic {} is not in the ECHO allowlist",
                ins.opcode
            ));
        }
        return Action::Keep;
    }

    let approx = parts.contains(&"approx") || (base == "div" && parts.contains(&"full"));
    if approx {
        let simple = |new_base: &str, f: bool| {
            let mut i = ins.clone();
            i.opcode = format!("{new_base}.rn{}.f32", ftz(f));
            Action::Single {
                instr: i,
                kind: ins.opcode.clone(),
            }
        };
        match base {
            "div" => {
                let f = match parts.as_slice() {
                    ["div", "approx" | "full", "f32"] => Some(false),
                    ["div", "approx" | "full", "ftz", "f32"] => Some(true),
                    _ => None,
                };
                if let (Some(f), 3) = (f, ops.len()) {
                    return simple("div", f);
                }
            }
            "rcp" | "sqrt" => {
                if let (Some(f), 2) = (f32_approx_form(&parts, base), ops.len()) {
                    return simple(base, f);
                }
            }
            "rsqrt" => {
                if let (Some(f), 2) = (f32_approx_form(&parts, base), ops.len()) {
                    let z = ftz(f);
                    let body = vec![
                        "{".to_string(),
                        ".reg .f32 %echo_t;".to_string(),
                        format!("sqrt.rn{z}.f32 %echo_t, {};", ops[1]),
                        format!("rcp.rn{z}.f32 {}, %echo_t;", ops[0]),
                        "}".to_string(),
                    ];
                    return Action::Multi {
                        body,
                        kind: ins.opcode.clone(),
                        atomic: false,
                    };
                }
            }
            "ex2" | "lg2" | "sin" | "cos" => {
                if let (Some(_), 2) = (f32_approx_form(&parts, base), ops.len()) {
                    let body = vec![
                        "{".to_string(),
                        ".param .b32 echo_p;".to_string(),
                        ".param .b32 echo_r;".to_string(),
                        format!("st.param.b32 [echo_p], {};", ops[1]),
                        format!("call (echo_r), __echo_{base}f, (echo_p);"),
                        format!("ld.param.b32 {}, [echo_r];", ops[0]),
                        "}".to_string(),
                    ];
                    return Action::Multi {
                        body,
                        kind: ins.opcode.clone(),
                        atomic: false,
                    };
                }
            }
            _ => {}
        }
        return Action::Reject(format!(
            "approximate instruction {} has no deterministic ECHO rewrite",
            ins.opcode
        ));
    }

    // Contractible float add/sub/mul: without a rounding modifier ptxas may fuse
    // them into an FMA, and whether it does differs between targets.
    if matches!(base, "add" | "sub" | "mul")
        && parts.last().is_some_and(|t| is_float_type(t))
        && !parts.iter().any(|p| ROUNDING.contains(p))
    {
        let mut i = ins.clone();
        i.opcode = format!("{base}.rn.{}", parts[1..].join("."));
        return Action::Single {
            instr: i,
            kind: format!("{base}.rn"),
        };
    }

    if !ALLOWLIST.contains(&base) {
        return Action::Reject(format!("opcode {base} is not in the ECHO allowlist"));
    }
    Action::Keep
}

/// Body of the `atom.global.add.f32` rewrite; `None` if the address form is unknown.
fn atomic_body(addr: &str, b: &str) -> Option<Vec<String>> {
    let inner = addr.strip_prefix('[')?.strip_suffix(']')?.trim();
    let (base, off) = match inner.find(['+', '-']) {
        Some(0) | None => (inner, "0".to_string()),
        Some(k) => {
            let sign = &inner[k..k + 1];
            let rest = inner[k + 1..].trim();
            let off = if sign == "-" {
                format!("-{rest}")
            } else {
                rest.to_string()
            };
            (inner[..k].trim(), off)
        }
    };
    if base.is_empty() || off.is_empty() {
        return None;
    }
    // The offset must be an integer literal.
    let digits = off.trim_start_matches('-');
    let is_int = !digits.is_empty()
        && (digits.chars().all(|c| c.is_ascii_digit())
            || (digits.starts_with("0x") && digits[2..].chars().all(|c| c.is_ascii_hexdigit())));
    if !is_int || off.matches('-').count() > 1 {
        return None;
    }
    let add = if off.starts_with('-') {
        "add.s64"
    } else {
        "add.u64"
    };
    let mut body = vec!["{".to_string(), ".reg .u64 %echo_a;".to_string()];
    if base.starts_with('%') {
        body.push(format!("{add} %echo_a, {base}, {off};"));
    } else {
        if !base.chars().all(super::lex::is_ident_char) {
            return None;
        }
        body.push(format!("mov.u64 %echo_a, {base};"));
        body.push(format!("{add} %echo_a, %echo_a, {off};"));
    }
    body.extend([
        "cvta.global.u64 %echo_a, %echo_a;".to_string(),
        ".param .b64 echo_pa;".to_string(),
        ".param .b32 echo_pb;".to_string(),
        "st.param.b64 [echo_pa], %echo_a;".to_string(),
        format!("st.param.b32 [echo_pb], {b};"),
        "call __echo_atomic_add_f32, (echo_pa, echo_pb);".to_string(),
        "}".to_string(),
    ]);
    Some(body)
}
