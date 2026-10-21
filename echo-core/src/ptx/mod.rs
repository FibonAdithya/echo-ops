//! PTX normalizer (Task 3).
//!
//! `normalize` rewrites algorithm PTX at load time so that every instruction
//! gives the same bits on every NVIDIA GPU, and fails loudly (kernel, line,
//! instruction, reason) on anything it cannot make deterministic. The rewrite
//! table lives in `rules.rs`; this file tracks functions and applies it.
//!
//! The ECHO library text (`lib`) is trusted: it is spliced in after
//! `.address_size` without being normalized (it is hazard-checked when it is
//! built, by echo-kernels/check_hazards.py). The `__echo_` prefix is reserved
//! for it (controller ruling R14).

pub mod deny;
pub mod lex;
pub mod rules;

pub use deny::forbidden_tokens;

use lex::Stmt;
use rules::Action;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Normalized {
    pub ptx: String,
    /// Kernels that issue `__echo_atomic_add_f32` (directly, or every `.entry`
    /// when the atomic sits in a `.func`); their launches need the exact
    /// accumulator finalize step.
    pub finalize_kernels: BTreeSet<String>,
    /// Number of rewrites, keyed by the original opcode (or `add.rn` etc.).
    pub counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtxError {
    pub kernel: String,
    pub line: usize,
    pub instruction: String,
    pub reason: String,
}

impl fmt::Display for PtxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ECHO: {}:{}: {}: {}",
            self.kernel, self.line, self.instruction, self.reason
        )
    }
}

impl std::error::Error for PtxError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hazard {
    pub kernel: String,
    pub line: usize,
    pub instruction: String,
}

/// One source line with its 1-based line number (in the input, or in `lib`).
struct SrcLine {
    text: String,
    no: usize,
}

enum LineKind {
    /// Outside any function body (module directives, function headers).
    Module,
    /// Outside any function, but not a directive: rejected by `normalize`.
    Stray,
    /// A (logical) line inside the body of function `fi`. `head` is the text
    /// before the body's opening `{` when the header shares the line.
    Body {
        fi: usize,
        stmts: Vec<Stmt>,
        head: String,
    },
}

/// A statement may span at most this many lines before it is cut off (and rejected).
const MAX_STMT_LINES: usize = 4096;

struct Func {
    name: String,
    is_entry: bool,
    /// Index of the (logical) line holding the `.entry`/`.func` keyword.
    header_line: usize,
    /// Exact `.reg` names, and `%name<N>` ranges as prefix -> N.
    regs: HashSet<String>,
    ranges: HashMap<String, u64>,
    /// Every non-`.reg` statement of the body: (line index, statement index, text).
    stmts: Vec<(usize, usize, String)>,
    /// Smallest n such that no `$ECHO_SKIP_m` with m >= n exists in the body.
    next_skip: usize,
}

struct Ctx<'a> {
    f: &'a Func,
    current: (usize, usize),
}

impl rules::FnCtx for Ctx<'_> {
    fn declared(&self, reg: &str) -> bool {
        if self.f.regs.contains(reg) {
            return true;
        }
        let digits = reg.trim_end_matches(|c: char| c.is_ascii_digit());
        if digits.len() == reg.len() {
            return false;
        }
        let n: u64 = match reg[digits.len()..].parse() {
            Ok(n) => n,
            Err(_) => return false,
        };
        self.f.ranges.get(digits).is_some_and(|&max| n < max)
    }
    fn used_elsewhere(&self, token: &str) -> bool {
        self.f
            .stmts
            .iter()
            .any(|(li, si, t)| (*li, *si) != self.current && lex::contains_token(t, token))
    }
}

/// Name and kind of a function header line (`.entry` / `.func`), if it is one.
fn header(code: &str) -> Option<(String, bool)> {
    let toks: Vec<&str> = code.split_whitespace().collect();
    let kw = toks
        .iter()
        .position(|t| *t == ".entry" || *t == ".func" || t.starts_with(".func("))?;
    let is_entry = toks[kw] == ".entry";
    let kw_str = if is_entry { ".entry" } else { ".func" };
    let at = code.find(kw_str)? + kw_str.len();
    let mut rest = code[at..].trim_start();
    if rest.starts_with('(') {
        // return parameter list of a .func
        let mut depth = 0;
        let mut end = rest.len();
        for (i, c) in rest.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        rest = rest[end..].trim_start();
    }
    let name: String = rest
        .chars()
        .take_while(|&c| lex::is_ident_char(c))
        .collect();
    Some((name, is_entry))
}

struct Analysis {
    lines: Vec<SrcLine>,
    kinds: Vec<LineKind>,
    funcs: Vec<Func>,
}

/// Name at the start of a header continuation line (`foo(` or `(.param ...) foo(`).
fn name_on_continuation(code: &str) -> String {
    let mut rest = code.trim_start();
    if rest.starts_with('(') {
        let mut depth = 0;
        let mut end = None;
        for (i, c) in rest.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        match end {
            Some(e) => rest = rest[e..].trim_start(),
            None => return String::new(),
        }
    }
    rest.chars()
        .take_while(|&c| lex::is_ident_char(c))
        .collect()
}

/// Net `{` minus `}` outside string literals.
fn brace_balance(code: &str) -> i64 {
    let mut n = 0;
    let mut in_str = false;
    let mut prev = ' ';
    for c in code.chars() {
        if in_str {
            if c == '"' && prev != '\\' {
                in_str = false;
            }
        } else {
            match c {
                '"' => in_str = true,
                '{' => n += 1,
                '}' => n -= 1,
                _ => {}
            }
        }
        prev = c;
    }
    n
}

fn analyze(phys: Vec<SrcLine>) -> Analysis {
    enum St {
        Out,
        Header(usize),
        Body(usize, i64),
    }
    let mut strip = lex::CommentStripper::default();
    let codes: Vec<String> = phys.iter().map(|l| strip.line(&l.text)).collect();
    let mut st = St::Out;
    let mut init_depth: i64 = 0; // open braces of a multi-line module-level initializer
    let mut funcs: Vec<Func> = Vec::new();
    let mut lines: Vec<SrcLine> = Vec::with_capacity(phys.len());
    let mut kinds = Vec::with_capacity(phys.len());
    let mut lexer = lex::StmtLexer::default();
    let mut p = 0;
    while p < phys.len() {
        let code = &codes[p];
        let li = lines.len();
        let one = |p: usize| SrcLine {
            text: phys[p].text.clone(),
            no: phys[p].no,
        };
        if let St::Out = st {
            if init_depth > 0 {
                init_depth += brace_balance(code);
                lines.push(one(p));
                kinds.push(LineKind::Module);
                p += 1;
                continue;
            }
            if let Some((name, is_entry)) = header(code) {
                funcs.push(Func {
                    name,
                    is_entry,
                    header_line: li,
                    regs: HashSet::new(),
                    ranges: HashMap::new(),
                    stmts: Vec::new(),
                    next_skip: 0,
                });
                st = St::Header(funcs.len() - 1);
            } else {
                let t = code.trim();
                let kind = if t.is_empty() || t.starts_with('.') {
                    init_depth = brace_balance(t).max(0);
                    LineKind::Module
                } else {
                    LineKind::Stray
                };
                lines.push(one(p));
                kinds.push(kind);
                p += 1;
                continue;
            }
        }
        // Header or body line. A body (logical) line takes the following
        // physical lines while a statement is unfinished.
        let (fi, mut depth, body_part, head) = match st {
            St::Header(fi) => {
                if funcs[fi].name.is_empty() && funcs[fi].header_line != li {
                    funcs[fi].name = name_on_continuation(code);
                }
                match code.find('{') {
                    Some(pos) => {
                        lexer = lex::StmtLexer::default();
                        (fi, 0i64, code[pos..].to_string(), code[..pos].to_string())
                    }
                    None => {
                        if code.trim_end().ends_with(';') {
                            // a prototype (.extern .func ...;), not a definition
                            funcs.pop();
                            st = St::Out;
                        }
                        lines.push(one(p));
                        kinds.push(LineKind::Module);
                        p += 1;
                        continue;
                    }
                }
            }
            St::Body(fi, depth) => (fi, depth, code.clone(), String::new()),
            St::Out => unreachable!(),
        };
        let mut text = phys[p].text.clone();
        let no = phys[p].no;
        let mut stmts = lexer.feed(&body_part);
        while lexer.pending() && p + 1 < phys.len() && lexer.pending_lines() < MAX_STMT_LINES {
            p += 1;
            text.push('\n');
            text.push_str(&phys[p].text);
            stmts.extend(lexer.feed(&codes[p]));
        }
        stmts.extend(lexer.flush());
        for s in &stmts {
            match s {
                Stmt::Open => depth += 1,
                Stmt::Close => depth -= 1,
                _ => {}
            }
        }
        lines.push(SrcLine { text, no });
        kinds.push(LineKind::Body { fi, stmts, head });
        st = if depth <= 0 {
            St::Out
        } else {
            St::Body(fi, depth)
        };
        p += 1;
    }
    // Per-function facts.
    for (li, k) in kinds.iter().enumerate() {
        let LineKind::Body { fi, stmts, .. } = k else {
            continue;
        };
        let f = &mut funcs[*fi];
        for (si, s) in stmts.iter().enumerate() {
            match s {
                Stmt::Directive(t) if t.starts_with(".reg") => {
                    for (name, n) in lex::reg_decl_names(t) {
                        match n {
                            Some(n) => {
                                f.ranges.insert(name, n);
                            }
                            None => {
                                f.regs.insert(name);
                            }
                        }
                    }
                }
                Stmt::Directive(t) | Stmt::Instr(t) | Stmt::Unterminated(t) => {
                    f.stmts.push((li, si, t.clone()))
                }
                Stmt::Label(l) => {
                    if let Some(n) = l
                        .strip_prefix("$ECHO_SKIP_")
                        .and_then(|n| n.parse::<usize>().ok())
                    {
                        f.next_skip = f.next_skip.max(n + 1);
                    }
                }
                Stmt::Open | Stmt::Close => {}
            }
        }
    }
    Analysis {
        lines,
        kinds,
        funcs,
    }
}

/// Every statement that is not kept as is, with its action.
/// With `skip_lib`, bodies of `__echo_*` functions (the trusted ECHO library)
/// are not inspected.
fn actions(a: &Analysis, skip_lib: bool) -> Vec<(usize, usize, Action, String)> {
    let mut out = Vec::new();
    for (li, k) in a.kinds.iter().enumerate() {
        let LineKind::Body { fi, stmts, .. } = k else {
            continue;
        };
        let f = &a.funcs[*fi];
        if skip_lib && f.name.starts_with(rules::RESERVED_PREFIX) {
            continue;
        }
        for (si, s) in stmts.iter().enumerate() {
            let (action, text) = match s {
                Stmt::Directive(t) | Stmt::Label(t) | Stmt::Unterminated(t)
                    if reserved_ident(t).is_some() =>
                {
                    (
                        Action::Reject(format!(
                            "symbol {} uses the reserved ECHO prefix __echo_",
                            reserved_ident(t).unwrap()
                        )),
                        format!("{t};"),
                    )
                }
                Stmt::Directive(t) if t.contains('\n') => (
                    Action::Reject("directive spans several lines (missing ';'?)".into()),
                    format!("{t};"),
                ),
                Stmt::Directive(t) => match body_directive_problem(t) {
                    Some(reason) => (Action::Reject(reason), format!("{t};")),
                    None => continue,
                },
                Stmt::Unterminated(t) if lex::is_unterminated_directive(t) => {
                    match body_directive_problem(t) {
                        Some(reason) => (Action::Reject(reason), t.clone()),
                        None => continue,
                    }
                }
                Stmt::Instr(t) => {
                    let ctx = Ctx {
                        f,
                        current: (li, si),
                    };
                    let action = match lex::parse_instr(t) {
                        Some(ins) => rules::classify(&ins, &ctx),
                        None => Action::Reject("cannot parse instruction".into()),
                    };
                    (action, format!("{t};"))
                }
                Stmt::Unterminated(t) => (
                    Action::Reject("statement is not terminated by ';'".into()),
                    t.clone(),
                ),
                Stmt::Label(_) | Stmt::Open | Stmt::Close => continue,
            };
            if action != Action::Keep {
                out.push((li, si, action, text));
            }
        }
    }
    out
}

/// Ruling R22: raw-text deny scan over the normalized user output. Errors name
/// the input line the offending output line came from.
fn deny_scan(
    a: &Analysis,
    out: &[String],
    first_out: &[usize],
    by_line: &HashMap<usize, HashMap<usize, &Action>>,
) -> Result<(), PtxError> {
    for li in 0..a.lines.len() {
        let verbatim = !by_line.contains_key(&li);
        for entry in &out[first_out[li]..first_out[li + 1]] {
            for (sub, text) in entry.split('\n').enumerate() {
                let found = deny::scan_line(text);
                if let Some(tok) = found.first() {
                    let kernel = match &a.kinds[li] {
                        LineKind::Body { fi, .. } => a.funcs[*fi].name.clone(),
                        _ => String::new(),
                    };
                    return Err(PtxError {
                        kernel,
                        line: a.lines[li].no + if verbatim { sub } else { 0 },
                        instruction: text.trim().to_string(),
                        reason: format!(
                            "forbidden token `{tok}` in the normalized output (raw-text deny scan)"
                        ),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Directives allowed inside a function body (the ones nvcc emits there), with
/// their operand shape checked where a malformed operand could hide text.
fn body_directive_problem(text: &str) -> Option<String> {
    let kw = text.split_whitespace().next().unwrap_or("");
    match kw {
        ".reg" | ".param" | ".local" | ".shared" => None,
        ".pragma" if lex::valid_pragma(text) => None,
        ".loc" if lex::valid_loc(text) => None,
        ".pragma" | ".loc" => Some(format!("malformed {kw} directive")),
        _ => Some(format!("directive {kw} is not allowed in a function body")),
    }
}

fn src_lines(ptx: &str) -> Vec<SrcLine> {
    ptx.lines()
        .enumerate()
        .map(|(i, l)| SrcLine {
            text: l.to_string(),
            no: i + 1,
        })
        .collect()
}

/// Lists every instruction that `normalize` would rewrite or reject.
/// An empty list means nothing nondeterministic is left. Bodies of `__echo_*`
/// functions (the trusted ECHO library, spliced in by `normalize`) are skipped;
/// `normalize` rejects user code that uses that prefix.
pub fn hazards(ptx: &str) -> Vec<Hazard> {
    let a = analyze(src_lines(ptx));
    let mut out: Vec<Hazard> = actions(&a, true)
        .into_iter()
        .map(|(li, _, _, text)| Hazard {
            kernel: a.funcs[kernel_of(&a, li)].name.clone(),
            line: a.lines[li].no,
            instruction: text,
        })
        .collect();
    // Statements outside any function (normalize rejects them).
    for (li, k) in a.kinds.iter().enumerate() {
        if let LineKind::Stray = k {
            out.push(Hazard {
                kernel: String::new(),
                line: a.lines[li].no,
                instruction: lex::strip_comments(&a.lines[li].text).trim().to_string(),
            });
        }
    }
    out.sort_by_key(|h| h.line);
    out
}

fn kernel_of(a: &Analysis, li: usize) -> usize {
    match &a.kinds[li] {
        LineKind::Body { fi, .. } => *fi,
        LineKind::Module | LineKind::Stray => unreachable!("actions only come from body lines"),
    }
}

fn render_stmt(s: &Stmt) -> String {
    match s {
        Stmt::Open => "\t{".into(),
        Stmt::Close => "\t}".into(),
        Stmt::Label(l) => format!("{l}:"),
        Stmt::Directive(t) | Stmt::Instr(t) => format!("\t{t};"),
        Stmt::Unterminated(t) => format!("\t{t}"),
    }
}

/// Rewrites the user PTX in `ptx` into deterministic PTX, then splices the
/// trusted ECHO library `lib` (not normalized) after the `.address_size` line.
///
/// If `ptx` already contains `lib` verbatim (an already linked module), that text
/// is cut out first, so `normalize(normalize(x, lib).ptx, lib) == normalize(x, lib)`.
/// User code may not declare, define or reference any `__echo_` symbol, except
/// `call`s to the public library functions that the rewrites themselves emit.
pub fn normalize(ptx: &str, lib: &str) -> Result<Normalized, PtxError> {
    let lib: String = if lib.is_empty() || lib.ends_with('\n') {
        lib.to_string()
    } else {
        format!("{lib}\n")
    };
    let mut lines = match (!lib.is_empty()).then(|| ptx.find(lib.as_str())).flatten() {
        Some(pos) => {
            // Already linked: cut the library out, keep input line numbers.
            let cut_line = ptx[..pos].matches('\n').count();
            let lib_lines = lib.matches('\n').count();
            let user = format!("{}{}", &ptx[..pos], &ptx[pos + lib.len()..]);
            let mut ls = src_lines(&user);
            for l in ls.iter_mut().skip(cut_line) {
                l.no += lib_lines;
            }
            ls
        }
        None => src_lines(ptx),
    };
    let had_newline = ptx.ends_with('\n');
    let mut n = normalize_user(std::mem::take(&mut lines), had_newline)?;
    if !lib.is_empty() {
        let mut out = String::with_capacity(n.ptx.len() + lib.len());
        let mut inserted = false;
        for piece in n.ptx.split_inclusive('\n') {
            out.push_str(piece);
            if !inserted
                && lex::strip_comments(piece)
                    .trim_start()
                    .starts_with(".address_size")
            {
                if !piece.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str(&lib);
                inserted = true;
            }
        }
        if !inserted {
            return Err(PtxError {
                kernel: String::new(),
                line: 0,
                instruction: String::new(),
                reason: "no .address_size directive to insert the ECHO library after".into(),
            });
        }
        n.ptx = out;
    }
    Ok(n)
}

/// First identifier in `code` that starts with the reserved prefix `__echo_`.
fn reserved_ident(code: &str) -> Option<String> {
    lex::find_prefixed_ident(code, rules::RESERVED_PREFIX)
}

fn normalize_user(lines: Vec<SrcLine>, had_newline: bool) -> Result<Normalized, PtxError> {
    let a = analyze(lines);
    // Outside function bodies: stray statements, and the reserved prefix in
    // definitions/declarations of .func/.entry/.global/.shared/.const symbols
    // (also in a header that shares its line with the body's `{`).
    let mut strip = lex::CommentStripper::default();
    for (li, k) in a.kinds.iter().enumerate() {
        let code: Vec<String> = a.lines[li]
            .text
            .split('\n')
            .map(|l| strip.line(l))
            .collect();
        let code = code.join("\n");
        let err = |kernel: String, text: &str, reason: String| PtxError {
            kernel,
            line: a.lines[li].no,
            instruction: text.trim().to_string(),
            reason,
        };
        let reserved = |id: String| format!("symbol {id} uses the reserved ECHO prefix __echo_");
        match k {
            LineKind::Module => {
                if let Some(id) = reserved_ident(&code) {
                    let kernel = header(&code).map(|(n, _)| n).unwrap_or_default();
                    return Err(err(kernel, &code, reserved(id)));
                }
            }
            LineKind::Stray => {
                return Err(err(
                    String::new(),
                    &code,
                    "statement outside any function (not a directive)".into(),
                ))
            }
            LineKind::Body { fi, head, .. } => {
                if let Some(id) = reserved_ident(head) {
                    return Err(err(a.funcs[*fi].name.clone(), head, reserved(id)));
                }
            }
        }
    }
    for f in &a.funcs {
        let reason = if f.name.is_empty() {
            "cannot find the function name".to_string()
        } else if f.name.starts_with(rules::RESERVED_PREFIX) {
            format!("symbol {} uses the reserved ECHO prefix __echo_", f.name)
        } else {
            continue;
        };
        let text = lex::strip_comments(&a.lines[f.header_line].text);
        return Err(PtxError {
            kernel: f.name.clone(),
            line: a.lines[f.header_line].no,
            instruction: text.trim().to_string(),
            reason,
        });
    }
    let acts = actions(&a, false);
    if let Some((li, _, Action::Reject(reason), text)) = acts
        .iter()
        .find(|(_, _, act, _)| matches!(act, Action::Reject(_)))
    {
        return Err(PtxError {
            kernel: a.funcs[kernel_of(&a, *li)].name.clone(),
            line: a.lines[*li].no,
            instruction: text.clone(),
            reason: reason.clone(),
        });
    }

    let entries: Vec<String> = a
        .funcs
        .iter()
        .filter(|f| f.is_entry)
        .map(|f| f.name.clone())
        .collect();
    let mut by_line: HashMap<usize, HashMap<usize, &Action>> = HashMap::new();
    for (li, si, act, _) in &acts {
        by_line.entry(*li).or_default().insert(*si, act);
    }
    let mut next_skip: Vec<usize> = a.funcs.iter().map(|f| f.next_skip).collect();
    let mut finalize = BTreeSet::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut out: Vec<String> = Vec::with_capacity(a.lines.len());
    // out[first_out[li]..first_out[li + 1]] came from logical line li
    let mut first_out: Vec<usize> = Vec::with_capacity(a.lines.len() + 1);

    for (li, line) in a.lines.iter().enumerate() {
        first_out.push(out.len());
        let Some(line_acts) = by_line.get(&li) else {
            out.push(line.text.clone());
            continue;
        };
        let LineKind::Body { fi, stmts, .. } = &a.kinds[li] else {
            unreachable!()
        };
        for (si, s) in stmts.iter().enumerate() {
            match line_acts.get(&si) {
                None => out.push(render_stmt(s)),
                Some(Action::Single { instr, kind }) => {
                    out.push(format!("\t{}", instr.render()));
                    *counts.entry(kind.clone()).or_default() += 1;
                }
                Some(Action::Multi { body, kind, atomic }) => {
                    let pred = match s {
                        Stmt::Instr(t) => lex::parse_instr(t).and_then(|i| i.pred),
                        _ => None,
                    };
                    let skip = pred.map(|p| {
                        let n = next_skip[*fi];
                        next_skip[*fi] += 1;
                        let inverse = match p.strip_prefix('!') {
                            Some(pos) => pos.to_string(),
                            None => format!("!{p}"),
                        };
                        out.push(format!("\t@{inverse} bra $ECHO_SKIP_{n};"));
                        n
                    });
                    out.extend(body.iter().map(|b| format!("\t{b}")));
                    if let Some(n) = skip {
                        out.push(format!("$ECHO_SKIP_{n}:"));
                    }
                    *counts.entry(kind.clone()).or_default() += 1;
                    if *atomic {
                        if a.funcs[*fi].is_entry {
                            finalize.insert(a.funcs[*fi].name.clone());
                        } else {
                            finalize.extend(entries.iter().cloned());
                        }
                    }
                }
                Some(Action::Keep) | Some(Action::Reject(_)) => unreachable!(),
            }
        }
    }
    // Calls to __echo_atomic_add_f32 that are already in the input (a module
    // normalized before) need the finalize step too.
    for k in &a.kinds {
        let LineKind::Body { fi, stmts, .. } = k else {
            continue;
        };
        let calls_atomic = stmts.iter().any(|s| match s {
            Stmt::Instr(t) => lex::parse_instr(t).is_some_and(|i| {
                i.base() == "call" && rules::callee(&i) == Some("__echo_atomic_add_f32")
            }),
            _ => false,
        });
        if calls_atomic {
            if a.funcs[*fi].is_entry {
                finalize.insert(a.funcs[*fi].name.clone());
            } else {
                finalize.extend(entries.iter().cloned());
            }
        }
    }
    first_out.push(out.len());
    // Unconditional: no flag, env var or build mode disables the deny scan.
    deny_scan(&a, &out, &first_out, &by_line)?;
    let mut ptx_out = out.join("\n");
    if had_newline {
        ptx_out.push('\n');
    }
    Ok(Normalized {
        ptx: ptx_out,
        finalize_kernels: finalize,
        counts,
    })
}
