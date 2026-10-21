//! Comment stripping and statement lexing for PTX.
//!
//! One line can hold several statements (inline asm such as
//! `{  cvt.f32.f16 %f7, %rs1;}` puts a scope brace, an instruction and a
//! closing brace on one line), and one statement can span several lines (nvcc
//! writes `call.uni (retval0), \n vprintf, \n ( \n param0, \n param1 \n );`).
//! `StmtLexer` is therefore a streaming lexer: it is fed one comment-free line
//! at a time and keeps an unfinished statement across lines. Every statement is
//! lexed, so no instruction can hide behind another one. Every step of the
//! lexer consumes at least one character, so it always terminates.

/// Strips `//` and `/* ... */` comments line by line; a block comment may span
/// lines. Comment markers inside string literals are not comments.
#[derive(Debug, Default)]
pub struct CommentStripper {
    in_block: bool,
}

impl CommentStripper {
    pub fn line(&mut self, line: &str) -> String {
        let chars: Vec<char> = line.chars().collect();
        let mut out = String::with_capacity(line.len());
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            let next = chars.get(i + 1).copied();
            if self.in_block {
                if c == '*' && next == Some('/') {
                    self.in_block = false;
                    out.push(' ');
                    i += 2;
                } else {
                    i += 1;
                }
                continue;
            }
            match (c, next) {
                ('/', Some('/')) => break,
                ('/', Some('*')) => {
                    self.in_block = true;
                    i += 2;
                }
                ('"', _) => {
                    let end = string_end(&chars, i);
                    out.extend(&chars[i..end]);
                    i = end;
                }
                _ => {
                    out.push(c);
                    i += 1;
                }
            }
        }
        out
    }
}

/// Removes comments from a single line (a fresh `CommentStripper`).
pub fn strip_comments(line: &str) -> String {
    CommentStripper::default().line(line)
}

/// Index one past the end of the string literal starting at `chars[start] == '"'`
/// (or the end of the line if it is not closed). Always `> start`.
///
/// There are no escapes: ptxas ends a string at the first `"` (`"a\"` is the
/// string `a\`), so this does too.
fn string_end(chars: &[char], start: usize) -> usize {
    match chars[start + 1..].iter().position(|&c| c == '"') {
        Some(k) => start + 1 + k + 1,
        None => chars.len(),
    }
}

/// One statement inside a function body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    /// A scope-opening `{` at statement position.
    Open,
    /// A scope-closing `}` at statement position.
    Close,
    /// `name:`
    Label(String),
    /// A directive such as `.reg .f32 %f<4>;` (text without the trailing `;`).
    Directive(String),
    /// An instruction (text without the trailing `;`; may contain newlines).
    Instr(String),
    /// Text that ended without `;`: `.loc`/`.file` (which take none), or a
    /// statement cut off by a closing `}` or by the end of the input.
    Unterminated(String),
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_' || c == '$' || c == '%'
}

pub fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '$'
}

/// Directives that are not terminated by `;`.
pub fn is_unterminated_directive(text: &str) -> bool {
    matches!(text.split_whitespace().next(), Some(".loc" | ".file")) && !text.trim().contains('\n')
}

fn is_uint(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

/// Checks the operand shape of a `.loc` directive (text without `;`):
/// `.loc file line col` optionally followed by
/// `, function_name label[+off], inlined_at file line col`.
pub fn valid_loc(text: &str) -> bool {
    let parts: Vec<&str> = text.trim().split(',').map(str::trim).collect();
    let head: Vec<&str> = parts[0].split_whitespace().collect();
    if head.len() != 4 || head[0] != ".loc" || !head[1..].iter().all(|t| is_uint(t)) {
        return false;
    }
    match parts.len() {
        1 => true,
        3 => {
            let f: Vec<&str> = parts[1].split_whitespace().collect();
            let label_ok = f.len() == 2
                && f[0] == "function_name"
                && f[1].split('+').enumerate().all(|(i, p)| {
                    if i == 0 {
                        !p.is_empty() && p.chars().all(is_ident_char)
                    } else {
                        is_uint(p)
                    }
                })
                && f[1].matches('+').count() <= 1;
            let a: Vec<&str> = parts[2].split_whitespace().collect();
            label_ok && a.len() == 4 && a[0] == "inlined_at" && a[1..].iter().all(|t| is_uint(t))
        }
        _ => false,
    }
}

/// Checks a `.pragma` directive: one or more string literals separated by commas.
pub fn valid_pragma(text: &str) -> bool {
    let Some(rest) = text.trim().strip_prefix(".pragma") else {
        return false;
    };
    if !rest.starts_with(char::is_whitespace) {
        return false;
    }
    let chars: Vec<char> = rest.chars().collect();
    let mut i = 0;
    let mut want_string = true;
    let mut strings = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if want_string && c == '"' {
            let end = string_end(&chars, i);
            if end > chars.len() || chars[end - 1] != '"' || end == i + 1 {
                return false;
            }
            // A backslash in a string means different things to different
            // tools (ptxas: none, C-like lexers: an escape); nvcc never emits one.
            if chars[i..end].contains(&'\\') {
                return false;
            }
            strings += 1;
            want_string = false;
            i = end;
        } else if !want_string && c == ',' {
            want_string = true;
            i += 1;
        } else {
            return false;
        }
    }
    strings > 0 && !want_string
}

/// Streaming statement lexer for function bodies.
#[derive(Debug, Default)]
pub struct StmtLexer {
    buf: String,
    depth: i32,
    active: bool,
    /// Physical lines the current statement spans.
    lines: usize,
}

impl StmtLexer {
    /// True while a statement is unfinished (it continues on the next line).
    pub fn pending(&self) -> bool {
        self.active
    }

    /// Number of lines the unfinished statement spans so far.
    pub fn pending_lines(&self) -> usize {
        if self.active {
            self.lines
        } else {
            0
        }
    }

    fn emit(&mut self, out: &mut Vec<Stmt>, terminated: bool) {
        let text = self.buf.trim().to_string();
        self.active = false;
        self.buf.clear();
        out.push(if !terminated {
            Stmt::Unterminated(text)
        } else if text.starts_with('.') {
            Stmt::Directive(text)
        } else {
            Stmt::Instr(text)
        });
    }

    /// Lexes one comment-free line; returns the statements completed on it.
    pub fn feed(&mut self, code: &str) -> Vec<Stmt> {
        let chars: Vec<char> = code.chars().collect();
        let mut out = Vec::new();
        if self.active {
            self.buf.push('\n');
            self.lines += 1;
        }
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if !self.active {
                if c.is_whitespace() || c == ';' {
                    i += 1;
                    continue;
                }
                if c == '{' {
                    out.push(Stmt::Open);
                    i += 1;
                    continue;
                }
                if c == '}' {
                    out.push(Stmt::Close);
                    i += 1;
                    continue;
                }
                if is_ident_start(c) {
                    let mut j = i + 1;
                    while j < chars.len() && is_ident_char(chars[j]) {
                        j += 1;
                    }
                    if j < chars.len() && chars[j] == ':' {
                        out.push(Stmt::Label(chars[i..j].iter().collect()));
                        i = j + 1;
                        continue;
                    }
                }
                self.active = true;
                self.buf.clear();
                self.depth = 0;
                self.lines = 1;
            }
            // Inside a statement.
            match c {
                '"' => {
                    let end = string_end(&chars, i);
                    self.buf.extend(&chars[i..end]);
                    i = end;
                }
                '{' | '[' | '(' => {
                    self.depth += 1;
                    self.buf.push(c);
                    i += 1;
                }
                '}' if self.depth == 0 => {
                    // A scope brace closes before this statement's ';'. Emit the
                    // statement as unterminated; the '}' is lexed next as Close.
                    self.emit(&mut out, false);
                }
                '}' | ']' | ')' => {
                    // A stray ')' or ']' at depth 0 stays in the text (it will not
                    // parse) and is consumed like any other character.
                    if self.depth > 0 {
                        self.depth -= 1;
                    }
                    self.buf.push(c);
                    i += 1;
                }
                ';' if self.depth == 0 => {
                    self.emit(&mut out, true);
                    i += 1;
                }
                _ => {
                    self.buf.push(c);
                    i += 1;
                }
            }
        }
        if self.active && self.depth == 0 && is_unterminated_directive(&self.buf) {
            self.emit(&mut out, false);
        }
        out
    }

    /// Ends the input: an unfinished statement becomes `Unterminated`.
    pub fn flush(&mut self) -> Option<Stmt> {
        if self.active {
            let mut out = Vec::new();
            self.emit(&mut out, false);
            out.pop()
        } else {
            None
        }
    }
}

/// Splits one comment-free piece of text into statements (an unfinished
/// statement at the end becomes `Unterminated`).
pub fn split_statements(code: &str) -> Vec<Stmt> {
    let mut lx = StmtLexer::default();
    let mut out = lx.feed(code);
    out.extend(lx.flush());
    out
}

/// A parsed instruction: `[@[!]pred] opcode operand, operand, ...`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instr {
    /// Predicate guard without the `@`, e.g. `%p1` or `!%p1`.
    pub pred: Option<String>,
    pub opcode: String,
    pub operands: Vec<String>,
}

impl Instr {
    pub fn base(&self) -> &str {
        self.opcode.split('.').next().unwrap_or("")
    }
    pub fn parts(&self) -> Vec<&str> {
        self.opcode.split('.').collect()
    }
    /// Canonical single-line rendering with a trailing `;`.
    pub fn render(&self) -> String {
        let mut s = String::new();
        if let Some(p) = &self.pred {
            s.push('@');
            s.push_str(p);
            s.push(' ');
        }
        s.push_str(&self.opcode);
        if !self.operands.is_empty() {
            s.push(' ');
            s.push_str(&self.operands.join(", "));
        }
        s.push(';');
        s
    }
}

/// Parses instruction text (without `;`). Returns `None` if there is no opcode.
pub fn parse_instr(text: &str) -> Option<Instr> {
    let mut rest = text.trim();
    let mut pred = None;
    if let Some(r) = rest.strip_prefix('@') {
        let end = r.find(char::is_whitespace)?;
        pred = Some(r[..end].to_string());
        rest = r[end..].trim_start();
    }
    if let Some(p) = &pred {
        let reg = p.strip_prefix('!').unwrap_or(p);
        if reg.is_empty() || !reg.chars().all(|c| is_ident_char(c) || c == '%') {
            return None;
        }
    }
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let opcode = rest[..end].to_string();
    if opcode.is_empty()
        || !opcode.chars().next().unwrap().is_ascii_alphabetic()
        || !opcode
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '_')
    {
        return None;
    }
    let operands = split_operands(&rest[end..]);
    // Each operand is one token once bracketed groups are collapsed. This keeps
    // a statement that swallowed the next line (a missing ';') from passing as
    // a harmless instruction with odd operands.
    if !operands.iter().all(|o| operand_is_single_token(o)) {
        return None;
    }
    Some(Instr {
        pred,
        opcode,
        operands,
    })
}

fn operand_is_single_token(op: &str) -> bool {
    let mut collapsed = String::new();
    let mut depth = 0i32;
    for c in op.chars() {
        match c {
            '{' | '[' | '(' => {
                if depth == 0 {
                    collapsed.push('X');
                }
                depth += 1;
            }
            '}' | ']' | ')' => depth -= 1,
            _ if depth == 0 => collapsed.push(c),
            _ => {}
        }
        if depth < 0 {
            return false;
        }
    }
    depth == 0 && collapsed.split_whitespace().count() == 1
}

/// Splits an operand list at top-level commas.
pub fn split_operands(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '{' | '[' | '(' => {
                depth += 1;
                cur.push(c)
            }
            '}' | ']' | ')' => {
                depth -= 1;
                cur.push(c)
            }
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    let t = cur.trim();
    if !t.is_empty() || !out.is_empty() {
        out.push(t.to_string());
    }
    out
}

/// Every `%name` token in `text`, with a `.x`/`.y`/`.z` suffix stripped.
pub fn percent_idents(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '%' && (i == 0 || !is_ident_char(chars[i - 1])) {
            let mut j = i + 1;
            while j < chars.len() && is_ident_char(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// True if `token` occurs in `text` as a whole identifier (`%f9` does not match `%f90`).
pub fn contains_token(text: &str, token: &str) -> bool {
    let mut start = 0;
    while let Some(pos) = text[start..].find(token) {
        let at = start + pos;
        let before = text[..at].chars().next_back();
        let after = text[at + token.len()..].chars().next();
        let tok_starts_ident = token.chars().next().is_some_and(is_ident_char);
        let before_ok = match before {
            None => true,
            Some(c) => !(is_ident_char(c) || (tok_starts_ident && c == '%')),
        };
        let after_ok = after.is_none_or(|c| !is_ident_char(c));
        if before_ok && after_ok {
            return true;
        }
        start = at + token.len().max(1);
    }
    false
}

/// First identifier in `text` that starts with `prefix` (at an identifier boundary).
pub fn find_prefixed_ident(text: &str, prefix: &str) -> Option<String> {
    let mut start = 0;
    while let Some(pos) = text[start..].find(prefix) {
        let at = start + pos;
        let before_ok = text[..at]
            .chars()
            .next_back()
            .is_none_or(|c| !(is_ident_char(c) || c == '%'));
        if before_ok {
            let id: String = text[at..]
                .chars()
                .take_while(|&c| is_ident_char(c))
                .collect();
            return Some(id);
        }
        start = at + prefix.len();
    }
    None
}

/// Register names declared by a `.reg` directive. `%f<4>` is returned as
/// `("%f", Some(4))`, a plain name as `(name, None)`.
pub fn reg_decl_names(directive: &str) -> Vec<(String, Option<u64>)> {
    let Some(rest) = directive.trim().strip_prefix(".reg") else {
        return Vec::new();
    };
    // Skip state-space/type/vector/alignment tokens (all start with '.', plus
    // the numeric argument of `.align`).
    let mut s = rest.trim_start();
    loop {
        if s.starts_with('.') {
            let end = s.find(char::is_whitespace).unwrap_or(s.len());
            let tok = &s[..end];
            s = s[end..].trim_start();
            if tok == ".align" {
                let end = s.find(char::is_whitespace).unwrap_or(s.len());
                s = s[end..].trim_start();
            }
        } else {
            break;
        }
    }
    let mut out = Vec::new();
    for name in s.split(',') {
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        if let Some(lt) = name.find('<') {
            let n = name[lt + 1..]
                .trim_end_matches('>')
                .trim()
                .parse::<u64>()
                .ok();
            out.push((name[..lt].trim().to_string(), n));
        } else {
            out.push((name.to_string(), None));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_inline_asm_line() {
        let s = split_statements("{  cvt.f32.f16 %f7, %rs1;}");
        assert_eq!(
            s,
            vec![
                Stmt::Open,
                Stmt::Instr("cvt.f32.f16 %f7, %rs1".into()),
                Stmt::Close
            ]
        );
    }

    #[test]
    fn vector_braces_are_operands() {
        let s = split_statements("ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd1];");
        assert_eq!(
            s,
            vec![Stmt::Instr(
                "ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd1]".into()
            )]
        );
    }

    #[test]
    fn two_instructions_on_one_line_are_both_seen() {
        let s = split_statements("mov.u32 %r1, %r2; ex2.approx.f32 %f1, %f2;");
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn stray_closers_and_strings() {
        assert_eq!(split_statements(")"), vec![Stmt::Unterminated(")".into())]);
        assert_eq!(split_statements("] x;"), vec![Stmt::Instr("] x".into())]);
        assert_eq!(
            split_statements(".pragma \"{\"; ex2.approx.f32 %f1, %f2;"),
            vec![
                Stmt::Directive(".pragma \"{\"".into()),
                Stmt::Instr("ex2.approx.f32 %f1, %f2".into())
            ]
        );
        assert_eq!(
            strip_comments(".pragma \"//\"; x // c"),
            ".pragma \"//\"; x "
        );
        let mut cs = CommentStripper::default();
        assert_eq!(cs.line("a /* b"), "a ");
        assert_eq!(cs.line("c */ d"), "  d");
        // no escapes inside strings (ptxas: "a\" is the string a\)
        assert_eq!(
            split_statements(".pragma \"a\\\"; ex2.approx.f32 %f1, %f2; //\";"),
            vec![
                Stmt::Directive(".pragma \"a\\\"".into()),
                Stmt::Instr("ex2.approx.f32 %f1, %f2".into()),
                // (comments are stripped before lexing; here the trailing `"`
                // opens a string that runs to the end of the line)
                Stmt::Unterminated("//\";".into()),
            ]
        );
        assert_eq!(
            strip_comments(".pragma \"a\\\"; x // \";"),
            ".pragma \"a\\\"; x "
        );
    }

    #[test]
    fn token_boundaries() {
        assert!(contains_token("st.global.f32 [%rd1], %f9", "%f9"));
        assert!(!contains_token("st.global.f32 [%rd1], %f90", "%f9"));
        assert!(!contains_token("mov.f32 %xf9, 0", "%f9"));
        assert!(contains_token("add r_sig, r_sig, 1", "r_sig"));
        assert!(!contains_token("add r_sig2, x", "r_sig"));
    }

    #[test]
    fn reg_decls() {
        assert_eq!(
            reg_decl_names(".reg .f32 \t%f<37>"),
            vec![("%f".into(), Some(37))]
        );
        assert_eq!(
            reg_decl_names(".reg .u64 r_signature, %x"),
            vec![("r_signature".into(), None), ("%x".into(), None)]
        );
    }
}
