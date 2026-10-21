//! Task 3: PTX normalizer tests. Each test uses a small hand-written fixture in tests/fixtures.

use echo_core::ptx::{hazards, normalize, PtxError};
use std::collections::BTreeSet;

fn fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// Stub of the ECHO library: extern declarations with the real call signatures.
const STUB_LIB: &str = "\
.extern .func (.param .b32 r) __echo_ex2f (.param .b32 p);
.extern .func (.param .b32 r) __echo_lg2f (.param .b32 p);
.extern .func (.param .b32 r) __echo_sinf (.param .b32 p);
.extern .func (.param .b32 r) __echo_cosf (.param .b32 p);
.extern .func __echo_atomic_add_f32 (.param .b64 a, .param .b32 v);
";

fn norm(name: &str) -> String {
    match normalize(&fixture(name), "") {
        Ok(n) => n.ptx,
        Err(e) => panic!("{name}: unexpected error: {e}"),
    }
}

fn err(ptx: &str) -> PtxError {
    match normalize(ptx, "") {
        Ok(n) => panic!("expected an error, got:\n{}", n.ptx),
        Err(e) => e,
    }
}

/// 1-based line numbers of the lines marked `HAZARD` in a fixture.
fn marked_lines(ptx: &str) -> Vec<usize> {
    ptx.lines()
        .enumerate()
        .filter(|(_, l)| l.contains("// HAZARD"))
        .map(|(i, _)| i + 1)
        .collect()
}

#[test]
fn rewrites_contractible_mul_add_sub_to_rn() {
    let out = norm("contractible.ptx");
    assert!(out.contains("mul.rn.ftz.f32 %f3, %f1, %f2;"), "{out}");
    assert!(out.contains("add.rn.ftz.f32 %f4, %f3, %f1;"), "{out}");
    assert!(out.contains("sub.rn.f32 %f5, %f4, %f2;"), "{out}");
    assert!(out.contains("mul.rn.f64 %fd2, %fd1, %fd1;"), "{out}");
    for bad in ["mul.ftz.f32", "add.ftz.f32", "sub.f32", "mul.f64"] {
        assert!(!out.contains(bad), "{bad} left in:\n{out}");
    }
    // integer arithmetic is not touched
    assert!(out.contains("mul.lo.s32 \t%r1, %r2, %r3;"), "{out}");
    assert!(out.contains("add.s32 \t%r1, %r1, 1;"), "{out}");
    assert!(hazards(&out).is_empty());
}

#[test]
fn keeps_fma_rn_unchanged() {
    let out = norm("contractible.ptx");
    assert!(
        out.contains("\tfma.rn.ftz.f32 \t%f6, %f5, %f1, %f2;\n"),
        "{out}"
    );
    assert!(out.contains("\tadd.rn.f32 \t%f7, %f6, %f1;\n"), "{out}");
    assert!(!out.contains("rn.rn"), "{out}");
}

#[test]
fn rewrites_div_rcp_sqrt_approx_to_rn() {
    let out = norm("approx.ptx");
    assert!(out.contains("div.rn.ftz.f32 %f3, %f1, %f2;"), "{out}");
    assert!(out.contains("div.rn.f32 %f4, %f1, %f2;"), "{out}");
    assert!(out.contains("rcp.rn.ftz.f32 %f5, %f1;"), "{out}");
    assert!(out.contains("sqrt.rn.ftz.f32 %f6, %f1;"), "{out}");
    assert!(out.contains("div.rn.f32 %f8, %f1, %f2;"), "{out}");
    assert!(!out.contains(".approx"), "{out}");
    assert!(!out.contains("div.full"), "{out}");
}

#[test]
fn rewrites_rsqrt_to_sqrt_then_rcp() {
    let out = norm("approx.ptx");
    let want = "\t{\n\t.reg .f32 %echo_t;\n\tsqrt.rn.ftz.f32 %echo_t, %f1;\n\trcp.rn.ftz.f32 %f7, %echo_t;\n\t}\n";
    assert!(out.contains(want), "{out}");
    assert!(!out.contains("rsqrt"), "{out}");
}

#[test]
fn rewrites_ex2_to_call_preserving_predicate() {
    let out = norm("ex2_pred.ptx");
    let bra = out.find("@!%p1 bra $ECHO_SKIP_0;").expect(&out);
    let st = out.find("st.param.b32 [echo_p], %f1;").expect(&out);
    let call = out
        .find("call (echo_r), __echo_ex2f, (echo_p);")
        .expect(&out);
    let ld = out.find("ld.param.b32 %f2, [echo_r];").expect(&out);
    let label = out.find("$ECHO_SKIP_0:").expect(&out);
    assert!(bra < st && st < call && call < ld && ld < label, "{out}");
    assert!(!out.contains("call.uni"), "{out}");
    assert!(!out.contains("ex2.approx"), "{out}");
    assert!(hazards(&out).is_empty(), "{:?}", hazards(&out));
}

#[test]
fn negated_predicate_branches_on_the_positive_predicate() {
    let out = normalize(&fixture("all_rows.ptx"), "").unwrap().ptx;
    // `@!%p2 rsqrt...` and `@!%p2 atom...` must be skipped when %p2 is TRUE.
    assert_eq!(out.matches("@%p2 bra $ECHO_SKIP_").count(), 2, "{out}");
    assert!(!out.contains("@!!"), "{out}");
    // single-instruction rewrites keep the predicate in place
    assert!(out.contains("@%p1 sqrt.rn.ftz.f32 %f9, %f1;"), "{out}");
    // labels are unique within the function
    let labels: Vec<&str> = out
        .lines()
        .filter(|l| l.starts_with("$ECHO_SKIP_"))
        .collect();
    let uniq: BTreeSet<&&str> = labels.iter().collect();
    assert_eq!(labels.len(), 3, "{out}"); // @!%p2 rsqrt, @%p1 cos, @!%p2 atom
    assert_eq!(uniq.len(), labels.len(), "{out}");
}

#[test]
fn atomic_add_unused_result_becomes_call_and_marks_kernel() {
    let n = normalize(&fixture("atomic_unused.ptx"), "").unwrap();
    let want: BTreeSet<String> = ["k_atomic".to_string()].into();
    assert_eq!(n.finalize_kernels, want);
    let out = &n.ptx;
    assert!(out.contains("add.u64 %echo_a, %rd4, 0;"), "{out}");
    assert!(out.contains("cvta.global.u64 %echo_a, %echo_a;"), "{out}");
    assert!(out.contains("st.param.b64 [echo_pa], %echo_a;"), "{out}");
    assert!(out.contains("st.param.b32 [echo_pb], %f2;"), "{out}");
    assert!(
        out.contains("call __echo_atomic_add_f32, (echo_pa, echo_pb);"),
        "{out}"
    );
    assert!(!out.contains("atom.global.add.f32"), "{out}");
    assert_eq!(n.counts.get("atom.global.add.f32"), Some(&1));
}

#[test]
fn atomic_in_func_marks_every_entry() {
    let n = normalize(&fixture("all_rows.ptx"), "").unwrap();
    let want: BTreeSet<String> = ["k_plain".to_string(), "k_rows".to_string()].into();
    assert_eq!(n.finalize_kernels, want);
    // offset -8 and a symbol base
    assert!(n.ptx.contains("add.s64 %echo_a, %rd1, -8;"), "{}", n.ptx);
    assert!(
        n.ptx
            .contains("mov.u64 %echo_a, g_acc;\n\tadd.u64 %echo_a, %echo_a, 4;"),
        "{}",
        n.ptx
    );
}

#[test]
fn atomic_add_with_offset_operand() {
    let out = norm("atomic_offset.ptx");
    assert!(
        out.contains("add.u64 %echo_a, %rd5, 4;\n\tcvta.global.u64 %echo_a, %echo_a;"),
        "{out}"
    );
}

#[test]
fn atomic_add_used_result_is_rejected() {
    let e = err(&fixture("atomic_used.ptx"));
    assert!(e.reason.contains("return value"), "{e}");
    assert_eq!(e.kernel, "k_used");
    assert_eq!(e.line, 17);
    assert_eq!(e.instruction, "atom.global.add.f32 \t%f3, [%rd4], %f2;");
    assert_eq!(
        e.to_string(),
        "ECHO: k_used:17: atom.global.add.f32 \t%f3, [%rd4], %f2;: float atomicAdd return value is used"
    );
}

#[test]
fn atomic_add_result_read_earlier_in_loop_is_rejected() {
    let e = err(&fixture("atomic_loop.ptx"));
    assert_eq!(e.kernel, "k_loop");
    assert_eq!(e.line, 22);
    assert!(e.reason.contains("return value"), "{e}");
}

#[test]
fn normalize_does_not_duplicate_lib() {
    // has_lib.ptx is an already linked module: it contains the exact library text.
    let input = fixture("has_lib.ptx");
    let header = ".visible .func  (.param .b32 func_retval0) __echo_ex2f(";
    let lib = &input[input.find(header).unwrap()..input.find(".visible .entry").unwrap()];
    let out = normalize(&input, lib).unwrap().ptx;
    assert_eq!(out.matches(header).count(), 1, "{out}");
    assert!(
        out.contains("call (echo_r), __echo_ex2f, (echo_p);"),
        "{out}"
    );
    // the library sits right after .address_size
    assert!(out.contains(&format!(".address_size 64\n{lib}")), "{out}");
}

#[test]
fn lib_is_inserted_after_address_size() {
    let out = normalize(&fixture("ex2_pred.ptx"), STUB_LIB).unwrap().ptx;
    let want = format!(".address_size 64\n{STUB_LIB}");
    assert!(out.contains(&want), "{out}");
    assert_eq!(
        out.matches(".extern .func (.param .b32 r) __echo_ex2f")
            .count(),
        1
    );
    // normalizing the linked output with the same lib does not insert it again
    assert_eq!(normalize(&out, STUB_LIB).unwrap().ptx, out);
    let empty = normalize(&fixture("ex2_pred.ptx"), "").unwrap().ptx;
    assert!(!empty.contains("__echo_ex2f ("), "{empty}");
}

#[test]
fn lib_is_not_normalized() {
    let lib = fixture("lib_trusted.ptx");
    let n = normalize(&fixture("ex2_pred.ptx"), &lib).unwrap();
    assert!(
        n.ptx.contains(&format!(".address_size 64\n{lib}")),
        "{}",
        n.ptx
    );
    for line in [
        "\tatom.global.exch.b32 \t%r1, [%rd1], 1;\n",
        "\tatom.global.cas.b64 \t%rd3, [%rd1], 0, 1;\n",
        "\tmul.rn.f32 \t%f2, %f1, %f1;\n",
    ] {
        assert_eq!(n.ptx.matches(line).count(), 1, "{line:?} in:\n{}", n.ptx);
    }
    // the same exch in user code is rejected
    let user = fixture("ex2_pred.ptx")
        .replace(
            "\tmov.f32 \t%f2, 0f00000000;",
            "\tatom.global.exch.b32 \t%r1, [%rd2], 1;",
        )
        .replace(
            ".reg .f32 \t%f<3>;",
            ".reg .f32 \t%f<3>;\n\t.reg .b32 \t%r<2>;",
        );
    assert!(err(&user).reason.contains("allowlist"), "{}", err(&user));
    // hazards() skips the trusted library's functions
    assert!(hazards(&n.ptx).is_empty(), "{:?}", hazards(&n.ptx));
}

#[test]
fn user_code_cannot_define_reserved_symbols() {
    let base = fixture("clock.ptx");
    let evil_func = base.replace(
        ".visible .entry k_clock(",
        ".visible .func __echo_evil()\n{\n\tret;\n}\n\n.visible .entry k_clock(",
    );
    let e = err(&evil_func);
    assert!(e.reason.contains("reserved"), "{e}");
    assert_eq!(e.kernel, "__echo_evil");
    assert_eq!(e.line, 6);
    // header and `{` on one line, and an ex2.approx inside that hazards() would skip
    let one_line = base.replace(
        ".visible .entry k_clock(",
        ".visible .entry __echo_atomic_reset() {\n\t.reg .f32 %f<3>;\n\tex2.approx.f32 %f1, %f2;\n\tret;\n}\n\n.visible .entry k_clock(",
    );
    let e = err(&one_line);
    assert!(e.reason.contains("reserved"), "{e}");
    assert_eq!((e.kernel.as_str(), e.line), ("__echo_atomic_reset", 6));
    // name on the line after the keyword
    let next_line = base.replace(
        ".visible .entry k_clock(",
        ".visible .func\n__echo_evil2()\n{\n\tret;\n}\n\n.visible .entry k_clock(",
    );
    assert!(err(&next_line).reason.contains("reserved"));
    // a global variable, a reference to a library symbol, and a call to a
    // non-public library function are rejected too
    let evil_global = base.replace(
        ".address_size 64",
        ".address_size 64\n.global .align 4 .u32 __echo_acc_used;",
    );
    assert!(err(&evil_global).reason.contains("reserved"));
    let evil_ref = base.replace(
        "\tmov.u64 \t%rd3, %clock64;",
        "\tmov.u64 \t%rd3, __echo_acc_keys;",
    );
    let e = err(&evil_ref);
    assert!(e.reason.contains("reserved"), "{e}");
    assert_eq!((e.kernel.as_str(), e.line), ("k_clock", 17));
    let evil_call = base.replace(
        "\tmov.u64 \t%rd3, %clock64;",
        "\tcall __echo_atomic_reset, ();",
    );
    assert!(err(&evil_call).reason.contains("reserved"));
    // a direct call to a public library function is allowed and marks the kernel
    let direct = base.replace(
        "\tmov.u64 \t%rd3, %clock64;",
        "\t{\n\t.param .b64 echo_pa;\n\t.param .b32 echo_pb;\n\tcall __echo_atomic_add_f32, (echo_pa, echo_pb);\n\t}",
    );
    let n = normalize(&direct, "").unwrap();
    let want: BTreeSet<String> = ["k_clock".to_string()].into();
    assert_eq!(n.finalize_kernels, want);
}

#[test]
fn linked_input_equals_unlinked() {
    let lib = fixture("lib_trusted.ptx");
    for name in ["ex2_pred.ptx", "all_rows.ptx", "atomic_unused.ptx"] {
        let unlinked = fixture(name);
        let linked =
            unlinked.replacen(".address_size 64\n", &format!(".address_size 64\n{lib}"), 1);
        assert_ne!(linked, unlinked);
        let a = normalize(&unlinked, &lib).unwrap();
        let b = normalize(&linked, &lib).unwrap();
        assert_eq!(a, b, "{name}");
        // idempotent; finalize_kernels survive (calls already in the input count),
        // counts do not (nothing is rewritten the second time)
        let again = normalize(&a.ptx, &lib).unwrap();
        assert_eq!(again.ptx, a.ptx, "{name}: idempotent");
        assert_eq!(again.finalize_kernels, a.finalize_kernels, "{name}");
    }
    // error line numbers refer to the linked input
    let used = fixture("atomic_used.ptx");
    let linked = used.replacen(".address_size 64\n", &format!(".address_size 64\n{lib}"), 1);
    let e = match normalize(&linked, &lib) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert_eq!(e.line, 17 + lib.matches('\n').count(), "{e}");
}

#[test]
fn clock64_becomes_zero() {
    let out = norm("clock.ptx");
    assert!(out.contains("mov.u64 %rd3, 0;"), "{out}");
    assert!(out.contains("mov.u32 %r1, 0;"), "{out}");
    assert!(!out.contains("%clock"), "{out}");
}

#[test]
fn rejects_unseen_hazards() {
    let template = fixture("clock.ptx");
    let target = "\tmov.u64 \t%rd3, %clock64;";
    let line = template.lines().position(|l| l == target).unwrap() + 1;
    let cases = [
        ("\tmov.u32 \t%r1, %smid;", "%smid"),
        ("\ttanh.approx.f32 \t%f1, %f2;", "approx"),
        ("\tatom.shared.add.f32 \t%f1, [%rd2], %f2;", "atom"),
        ("\tex2.approx.f16 \t%rs1, %rs2;", "approx"),
        ("\tactivemask.b32 \t%r1;", "activemask"),
        ("\tmov.u64 \t%rd3, %globaltimer;", "%globaltimer"),
    ];
    for (instr, reason_part) in cases {
        let ptx = template.replace(target, instr);
        let e = err(&ptx);
        assert_eq!(e.kernel, "k_clock", "{instr}: {e}");
        assert_eq!(e.line, line, "{instr}: {e}");
        assert_eq!(e.instruction, instr.trim(), "{instr}: {e}");
        assert!(e.reason.contains(reason_part), "{instr}: {e}");
        assert_eq!(
            hazards(&ptx).iter().filter(|h| h.line == line).count(),
            1,
            "{instr}"
        );
    }
}

#[test]
fn rejects_every_reject_row() {
    // Remove the marked lines one by one from the top: each time, normalize must
    // fail on exactly the first remaining marked line.
    let src = fixture("rejects.ptx");
    let mut lines: Vec<String> = src.lines().map(String::from).collect();
    let mut seen = 0;
    while let Some(idx) = lines.iter().position(|l| l.contains("// HAZARD")) {
        let ptx = lines.join("\n");
        let e = err(&ptx);
        assert_eq!(e.line, idx + 1, "{e}");
        assert_eq!(e.kernel, "k_bad", "{e}");
        lines[idx] = String::new();
        seen += 1;
    }
    assert_eq!(seen, marked_lines(&src).len());
    assert!(normalize(&lines.join("\n"), "").is_ok());
}

#[test]
fn normalize_is_idempotent() {
    for name in [
        "all_rows.ptx",
        "contractible.ptx",
        "approx.ptx",
        "ex2_pred.ptx",
        "clock.ptx",
    ] {
        let once = normalize(&fixture(name), STUB_LIB).unwrap().ptx;
        let twice = normalize(&once, STUB_LIB).unwrap().ptx;
        assert_eq!(once, twice, "{name}");
        assert!(hazards(&once).is_empty(), "{name}: {:?}", hazards(&once));
    }
    // with a library in the module, normalizing again with the same lib is a no-op too
    let input = fixture("has_lib.ptx");
    let lib = &input[input.find(".visible .func").unwrap()..input.find(".visible .entry").unwrap()];
    let once = normalize(&input, lib).unwrap().ptx;
    assert_eq!(normalize(&once, lib).unwrap().ptx, once);
    let lib = fixture("lib_trusted.ptx");
    let once = normalize(&fixture("all_rows.ptx"), &lib).unwrap().ptx;
    assert_eq!(normalize(&once, &lib).unwrap().ptx, once);
}

#[test]
fn hazards_reports_every_table_row() {
    for name in ["all_rows.ptx", "rejects.ptx"] {
        let src = fixture(name);
        let got: Vec<usize> = hazards(&src).iter().map(|h| h.line).collect();
        assert_eq!(got, marked_lines(&src), "{name}: {:?}", hazards(&src));
    }
    let h = hazards(&fixture("rejects.ptx"));
    assert!(h.iter().all(|h| h.kernel == "k_bad"));
    assert_eq!(h[0].instruction, "atom.global.add.f32 \t%f3, [%rd2], %f1;");
    // on a line with several statements, the hazardous statement is named
    let vote = h
        .iter()
        .find(|h| h.instruction.starts_with("vote"))
        .unwrap();
    assert_eq!(
        vote.instruction,
        "vote.sync.ballot.b32 %r9, %p1, 0xffffffff;"
    );
}

// ---- robustness (fix round 1) ----

/// Runs `f` on another thread and fails if it does not return within `secs`
/// (a hang) or if it panics.
fn within<T: Send + 'static>(secs: u64, what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(std::time::Duration::from_secs(secs)) {
        Ok(v) => v,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("{what}: no result after {secs} s (hang)")
        }
        Err(e) => panic!("{what}: worker thread died (panic): {e}"),
    }
}

#[test]
fn stray_closers_terminate_with_an_error() {
    let template = fixture("clock.ptx");
    let target = "\tmov.u64 \t%rd3, %clock64;";
    let line = template.lines().position(|l| l == target).unwrap() + 1;
    for stray in ["\t)", "\t] x;", "\t)\n\t]", "\t) ] ) ]"] {
        let ptx = template.replace(target, stray);
        let r = within(10, stray, move || normalize(&ptx, ""));
        let e = r.expect_err(stray);
        assert_eq!(
            (e.kernel.as_str(), e.line),
            ("k_clock", line),
            "{stray:?}: {e}"
        );
    }
}

#[test]
fn nvcc_multiline_call_is_one_statement() {
    // Real nvcc 12.8 output (TIG flags) with a __noinline__ call and printf:
    // `call.uni (retval0), \n vprintf, \n ( \n param0, \n param1 \n );`
    let src = fixture("nvcc_multiline_call.ptx");
    let s2 = src.clone();
    let n = within(10, "multi-line call", move || normalize(&s2, STUB_LIB)).unwrap();
    // the two calls are kept byte for byte, the ex2 and the mul are rewritten
    for call in [
        "\tcall.uni (retval0), \n\thelper_noinline, \n\t(\n\tparam0, \n\tparam1\n\t);\n",
        "\tcall.uni (retval0), \n\tvprintf, \n\t(\n\tparam0, \n\tparam1\n\t);\n",
    ] {
        assert_eq!(n.ptx.matches(call).count(), 1, "{}", n.ptx);
    }
    assert!(
        n.ptx.contains("call (echo_r), __echo_ex2f, (echo_p);"),
        "{}",
        n.ptx
    );
    assert!(
        n.ptx.contains("mul.rn.ftz.f32 %f3, %f1, 0f3FB8AA3B;"),
        "{}",
        n.ptx
    );
    assert!(hazards(&n.ptx).is_empty(), "{:?}", hazards(&n.ptx));
    let lines: Vec<usize> = hazards(&src).iter().map(|h| h.line).collect();
    let mul = src.lines().position(|l| l.contains("mul.ftz.f32")).unwrap() + 1;
    assert_eq!(lines, vec![mul, mul + 1]);
    // dropping the final ';' of the call is an error naming the call's first line
    let call_line = src
        .lines()
        .position(|l| l.contains("call.uni (retval0), "))
        .unwrap()
        + 1;
    let broken = src.replacen(
        "\tparam1\n\t);\n\tld.param.f32",
        "\tparam1\n\t)\n\tld.param.f32",
        1,
    );
    let e = within(10, "broken call", move || normalize(&broken, "")).unwrap_err();
    assert_eq!((e.kernel.as_str(), e.line), ("k_print", call_line), "{e}");
}

#[test]
fn strings_and_comments_do_not_hide_statements() {
    let template = fixture("clock.ptx");
    let target = "\tmov.u64 \t%rd3, %clock64;";
    let line = template.lines().position(|l| l == target).unwrap() + 1;
    // a brace or a comment marker inside a string literal
    for prefix in [
        ".pragma \"{\";",
        ".pragma \"//\";",
        ".pragma \"/*\";",
        "/* c */",
    ] {
        let ptx = template.replace(target, &format!("\t{prefix} ex2.approx.f32 %f1, %f2;"));
        let h: Vec<_> = hazards(&ptx)
            .into_iter()
            .filter(|h| h.instruction.starts_with("ex2"))
            .collect();
        assert_eq!(h.len(), 1, "{prefix}: {h:?}");
        assert_eq!(
            (h[0].line, h[0].instruction.as_str()),
            (line, "ex2.approx.f32 %f1, %f2;")
        );
        let out = normalize(&ptx, "").unwrap().ptx;
        assert!(!out.contains("ex2.approx"), "{prefix}: {out}");
    }
    // a block comment over several lines hides only what is inside it
    let ptx = template.replace(
        target,
        "\t/* start\n\tex2.approx.f32 %f1, %f2;\n\tend */ ex2.approx.f32 %f2, %f1;",
    );
    let h: Vec<_> = hazards(&ptx)
        .into_iter()
        .filter(|h| h.instruction.starts_with("ex2"))
        .collect();
    assert_eq!(h.len(), 1, "{h:?}");
    assert_eq!(
        (h[0].line, h[0].instruction.as_str()),
        (line + 2, "ex2.approx.f32 %f2, %f1;")
    );
    // a directive missing its ';' cannot swallow the next instruction
    let ptx = template.replace(target, "\t.pragma \"nounroll\"\n\tex2.approx.f32 %f1, %f2;");
    let e = err(&ptx);
    assert_eq!((e.kernel.as_str(), e.line), ("k_clock", line), "{e}");
    assert!(hazards(&ptx).iter().any(|h| h.line == line));
    // nor can an instruction missing its ';'
    let ptx = template.replace(target, "\tmov.u32 %r1, %r1\n\tex2.approx.f32 %f1, %f2;");
    let e = err(&ptx);
    assert_eq!(
        (e.line, e.reason.as_str()),
        (line, "cannot parse instruction"),
        "{e}"
    );
}

#[test]
fn kernel_name_on_the_next_line() {
    let ptx = fixture("clock.ptx")
        .replace(".visible .entry k_clock(", ".visible .entry\nk_next(")
        .replace("\tmov.u64 \t%rd3, %clock64;", "\tmov.u32 \t%r1, %smid;");
    assert_eq!(err(&ptx).kernel, "k_next");
}

#[test]
fn rejects_statements_outside_functions() {
    let base = fixture("clock.ptx");
    let stray = base.replace(
        ".address_size 64",
        ".address_size 64\nex2.approx.f32 %f1, %f2;",
    );
    let e = err(&stray);
    assert_eq!((e.kernel.as_str(), e.line), ("", 5), "{e}");
    // an unrecognised header keyword leaves its body outside any function
    let typo = base.replace(".visible .entry k_clock(", ".visible .entri k_clock(");
    let e = err(&typo);
    assert!(e.reason.contains("outside any function"), "{e}");
    // a multi-line module-level initializer is fine
    let init = base.replace(
        ".address_size 64",
        ".address_size 64\n.global .align 4 .b8 tbl[4] = {1, 2,\n3, 4};",
    );
    assert!(normalize(&init, "").is_ok());
    // hazards() reports stray lines too
    let h = hazards(&stray);
    assert!(
        h.iter().any(|h| h.line == 5 && h.kernel.is_empty()),
        "{h:?}"
    );
}

#[test]
fn normalize_never_hangs_or_panics_on_mangled_input() {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let names = [
        "all_rows.ptx",
        "rejects.ptx",
        "nvcc_multiline_call.ptx",
        "has_lib.ptx",
        "atomic_loop.ptx",
        "ex2_pred.ptx",
    ];
    let junk = [
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "\"",
        "/*",
        "*/",
        ";",
        "@",
        "%",
        ":",
        ".",
        "\\",
        "//",
        ".loc 1 2 3 ",
        ".pragma \"",
        " ex2.approx.f32 %f1, %f2;",
        "%clock64",
        " mul.f32 ",
        " atom.global.add.f32 ",
        "$L9: ",
    ];
    for (k, name) in names.iter().enumerate() {
        let src = fixture(name);
        let lib = fixture("lib_trusted.ptx");
        within(60, name, move || {
            let mut rng = StdRng::seed_from_u64(0xEC40 + k as u64);
            for _ in 0..300 {
                let mut lines: Vec<String> = src.lines().map(String::from).collect();
                for _ in 0..rng.gen_range(1..6) {
                    let i = rng.gen_range(0..lines.len());
                    match rng.gen_range(0..4) {
                        0 => {
                            let cut: Vec<char> = lines[i].chars().collect();
                            let at = rng.gen_range(0..=cut.len());
                            lines[i] = cut[..at].iter().collect();
                        }
                        1 => {
                            let l = lines[i].clone();
                            lines.insert(i, l);
                        }
                        2 => {
                            lines.remove(i);
                            if lines.is_empty() {
                                lines.push(String::new());
                            }
                        }
                        _ => {
                            let j = junk[rng.gen_range(0..junk.len())];
                            let mut cut: Vec<char> = lines[i].chars().collect();
                            let at = rng.gen_range(0..=cut.len());
                            cut.splice(at..at, j.chars());
                            lines[i] = cut.into_iter().collect();
                        }
                    }
                }
                let m = lines.join("\n");
                let _ = hazards(&m);
                if let Ok(n) = normalize(&m, &lib) {
                    // whatever is accepted is a fixed point, and the raw-text
                    // deny scan (R22) finds nothing in it (lib_trusted is clean)
                    assert_eq!(normalize(&n.ptx, &lib).unwrap().ptx, n.ptx);
                    let bad = echo_core::ptx::forbidden_tokens(&n.ptx);
                    assert!(bad.is_empty(), "{bad:?} in accepted output:\n{}", n.ptx);
                }
            }
        });
    }
}

#[test]
fn skip_labels_restart_in_each_function() {
    let n = normalize(&fixture("two_skips.ptx"), STUB_LIB).unwrap();
    assert_eq!(n.ptx.matches("\n$ECHO_SKIP_0:\n").count(), 2, "{}", n.ptx);
    assert!(hazards(&n.ptx).is_empty());
}

// ---- fix round 2: directive shapes, string escapes, raw-text deny scan (R22) ----

/// The reviewer's probe module: kernel `k`, the probe text goes on line 15.
fn probe_module(probe: &str) -> String {
    format!(
        ".version 8.7\n.target sm_70\n.address_size 64\n\n.visible .entry k(\n\t.param .u64 p\n)\n{{\n\t.reg .f32 %f<3>;\n\t.reg .b32 %r<3>;\n\t.reg .b64 %rd<3>;\n\tld.param.u64 %rd1, [p];\n\tcvta.to.global.u64 %rd2, %rd1;\n\tld.global.f32 %f2, [%rd2];\n{probe}\n\tst.global.f32 [%rd2], %f1;\n\tret;\n}}\n"
    )
}

#[test]
fn reviewer_round2_probes_are_rejected() {
    // control: the module itself is fine
    assert!(normalize(&probe_module("\tmov.f32 %f1, %f2;"), "").is_ok());
    let probes = [
        ("B", "\t.local .b32 __echo_acc_keys"),
        ("B2", "\t.local .b32 x\n\tret;"),
        ("L1", "\t.loc 1 2 3 ex2.approx.f32 %f1, %f2;"),
        ("L2", "\t.loc 1 2 3 ex2.approx.f32 %f1, %f2\n\t;"),
        ("L4", "\t.file 1 \"a.cu\" ex2.approx.f32 %f1, %f2\n\t;"),
        ("L5", "\t.loc 1 2 3 __echo_acc_keys"),
        ("M4", "$L1: .loc 1 2 3 ex2.approx.f32 %f1, %f2;"),
        ("N1c", "\t.pragma \"a\\\"; ex2.approx.f32 %f1, %f2; //\";"),
        ("N1c_alone", "\t.pragma \"a\\\";"),
        ("pragma_junk", "\t.pragma \"nounroll\" ex2;"),
        ("unknown_directive", "\t.maxnreg 32;"),
    ];
    for (name, probe) in probes {
        let ptx = probe_module(probe);
        let e = match within(10, name, move || normalize(&ptx, "")) {
            Ok(n) => panic!("{name}: accepted:\n{}", n.ptx),
            Err(e) => e,
        };
        assert_eq!((e.kernel.as_str(), e.line), ("k", 15), "{name}: {e}");
        assert!(!hazards(&probe_module(probe)).is_empty(), "{name}");
    }
    // valid .loc forms and pragmas are kept
    for ok in [
        "\t.loc 1 2 3",
        "\t.loc 1 2 3;",
        "\t.loc 1 2 3, function_name $L__info_string0, inlined_at 1 5 7",
        "\t.pragma \"nounroll\";",
        "\t.pragma \"used_bytes_mask 4095\";",
    ] {
        let n = normalize(&probe_module(ok), "").unwrap_or_else(|e| panic!("{ok:?}: {e}"));
        assert!(n.ptx.contains(ok.trim()), "{ok}");
    }
}

const FORBIDDEN: &[&str] = &[
    ".approx",
    "ex2.x",
    "lg2.x",
    "sin.x",
    "cos.x",
    "tanh.x",
    "rsqrt.x",
    "mul.f32",
    "add.ftz.f32",
    "sub.sat.f64",
    "mad.f16",
    "fma.bf16x2",
    "add.f16x2",
    "mul.bf16",
    "atom.global.add.f32",
    "red.global.add.f64",
    "atom.shared.cas.f16",
    "%clock",
    "%clock64",
    "%globaltimer",
    "%globaltimer_hi",
    "%smid",
    "%nsmid",
    "%warpid",
    "%nwarpid",
    "%gridid",
    "activemask",
    "div.full",
];

#[test]
fn deny_scan_rejects_forbidden_tokens_in_comments_and_strings() {
    let base = fixture("clock.ptx");
    // a line that normalize keeps verbatim (a rewritten line loses its comment)
    let target = "\tcvta.to.global.u64 \t%rd2, %rd1;";
    let line = base.lines().position(|l| l == target).unwrap() + 1;
    for tok in FORBIDDEN {
        for wrapped in [
            format!("{target} // note: {tok}"),
            format!("{target} /* {tok} */"),
            format!("\t.pragma \"{tok}\";\n{target}"),
        ] {
            let ptx = base.replace(target, &wrapped);
            let e = err(&ptx);
            assert!(
                e.reason.contains("forbidden token"),
                "{tok} in {wrapped:?}: {e}"
            );
            assert_eq!((e.kernel.as_str(), e.line), ("k_clock", line), "{tok}: {e}");
        }
        // and in a module-level comment
        let ptx = base.replace(".address_size 64", &format!(".address_size 64\n// {tok}"));
        let e = err(&ptx);
        assert!(e.reason.contains("forbidden token"), "{tok}: {e}");
        assert_eq!((e.kernel.as_str(), e.line), ("", 5), "{tok}: {e}");
    }
    // the trusted library is not scanned (it is checked when it is built)
    let lib = "// a comment that mentions ex2.approx in the library\n";
    assert!(normalize(&base, lib).is_ok());
}

#[test]
fn f32x2_arithmetic_gets_rounding_mode_and_atomics_are_rejected() {
    // Packed f32x2 add without a rounding mode is contractible like add.f32:
    // it must be rewritten to .rn (or rejected), never kept as is.
    let template = fixture("clock.ptx");
    let target = "\tmov.u64 \t%rd3, %clock64;";
    let ptx = template.replace(target, "\tadd.f32x2 \t%rd3, %rd1, %rd2;");
    let out = normalize(&ptx, "").unwrap().ptx;
    assert!(out.contains("add.rn.f32x2 %rd3, %rd1, %rd2;"), "{out}");
    assert!(!out.contains("add.f32x2"), "{out}");
    assert!(
        hazards(&ptx).iter().any(|h| h.instruction.starts_with("add.f32x2")),
        "{:?}",
        hazards(&ptx)
    );
    // a float atomic on f32x2 is rejected
    let ptx = template.replace(target, "\tred.global.add.f32x2 \t[%rd2], %rd1;");
    assert!(err(&ptx).reason.contains("floating-point atomic"), "{}", err(&ptx));
    // the raw-text deny scan catches add.f32x2 where the lexer does not look (a
    // comment on a line that normalize keeps verbatim)
    let kept = "\tcvta.to.global.u64 \t%rd2, %rd1;";
    let ptx = template.replace(kept, &format!("{kept} // add.f32x2"));
    let e = err(&ptx);
    assert!(e.reason.contains("deny scan") && e.instruction.contains("add.f32x2"), "{e}");
}

#[test]
fn cas_in_user_code_is_rejected() {
    let template = fixture("clock.ptx");
    let target = "\tmov.u64 \t%rd3, %clock64;";
    let ptx = template.replace(target, "\tatom.global.cas.b64 \t%rd3, [%rd2], %rd1, %rd2;");
    let e = err(&ptx);
    assert!(e.reason.contains("not in the ECHO allowlist"), "{e}");
    assert_eq!(e.instruction, "atom.global.cas.b64 \t%rd3, [%rd2], %rd1, %rd2;");
}
