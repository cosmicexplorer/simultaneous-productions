#![recursion_limit = "128"]
#![feature(proc_macro_diagnostic)]
#![feature(fn_traits)]
#![feature(trace_macros)]
/* These clippy lint descriptions are purely non-functional and do not affect the functionality
 * or correctness of the code.
 * TODO: rustfmt breaks multiline comments when used one on top of another! (each with its own
 * pair of delimiters)
 * Note: run clippy with: rustup run nightly cargo-clippy! */
#![deny(warnings)]
// Enable all clippy lints except for many of the pedantic ones. It's a shame this needs to be
// copied and pasted across crates, but there doesn't appear to be a way to include inner attributes
// from a common source.
#![deny(
  clippy::all,
  clippy::default_trait_access,
  clippy::expl_impl_clone_on_copy,
  clippy::if_not_else,
  clippy::needless_continue,
  clippy::single_match_else,
  clippy::unseparated_literal_suffix,
  clippy::used_underscore_binding
)]
// It is often more clear to show that nothing is being moved.
#![allow(clippy::match_ref_pats)]
// Subjective style.
#![allow(
  clippy::derive_hash_xor_eq,
  clippy::len_without_is_empty,
  clippy::redundant_field_names,
  clippy::too_many_arguments
)]
// Default isn't as big a deal as people seem to think it is.
#![allow(clippy::new_without_default, clippy::new_ret_no_self)]
// Arc<Mutex> can be more clear than needing to grok Orderings:
#![allow(clippy::mutex_atomic)]

extern crate proc_macro;
use self::proc_macro::TokenStream;

use quote::{quote, quote_spanned};
use syn::{
  parse::{Parse, ParseStream, Result},
  parse_macro_input,
  spanned::Spanned,
  Expr, Ident, Token, Type, Visibility,
};

struct List<T>(Vec<T>);

struct Case {
  
}

struct Production {
  ty: Type,
  cases: List<Case>,
}

#[proc_macro]
fn typed_productions(input: TokenStream) -> TokenStream {
  /* typed_productions![ */
  /* A => [ */
  /* (case (["a"], (x: B)) => { A(x) }) */
  /* ] */
  /* ] */
  let productions: Vec<P<Expr>> = args.iter().map(|prod| match prod {
    TokenTree::Delimited(_span, DelimToken::Brace, production) =>
      match &production.into_trees().collect::<Vec<_>>()[..] {
        [
          TokenTree::Token(Token {
            kind: Ident(production_name, _is_raw),
            ..
          }),
          TokenTree::Token(Token { kind: FatArrow, .. }),
          TokenTree::Delimited(_span, DelimToken::Bracket, cases),
        ] => {
          let typed_cases = cases.into_trees().map(|case| match case {
            TokenTree::Delimited(_span, DelimToken::Paren, case_declaration) => {
              match &case_declaration.into_trees().collect::<Vec<_>>()[..] {
                [
                  TokenTree::Token(Token {
                    kind: Ident(case_ident_name, _is_raw),
                    ..
                  }),
                  TokenTree::Delimited(_span, DelimToken::Paren, arg_decls),
                  TokenTree::Token(Token { kind: FatArrow, .. }),
                  TokenTree::Delimited(_span2, DelimToken::Brace, case_body),
                ] => {
                  let typed_args: Vec<(AstIdent, AstIdent)> = arg_decls
                    .into_trees()
                    .flat_map(|decl| match decl {
                      /* This is a literal -- no args to add to the method. */
                      TokenTree::Delimited(_span, DelimToken::Bracket, _body) => None,
                      /* This is a subproduction, so we add the arg. */
                      TokenTree::Delimited(_span, DelimToken::Paren, typed_subprod) =>
                        match typed_subprod.into_trees().collect::<Vec<_>>()[..] {
                          [
                            TokenTree::Token(Token {
                              kind: Ident(arg_name, _is_raw),
                              ..
                            }),
                            TokenTree::Token(Token {
                              kind: Colon,
                              ..
                            }),
                            TokenTree::Token(Token {
                              kind: Ident(arg_type, _is_raw2),
                              ..
                            }),
                          ] => Some(
                            (
                              AstIdent::from_str(&arg_name.as_str()),
                              AstIdent::from_str(&arg_type.as_str()),
                            )),
                          _ => panic!("???"),
                        },
                      _ => None,
                    })
                    .collect();
                  let constituent_case_elements: Vec<P<Expr>> =
                    arg_decls.into_trees().map(|decl| match decl {
                      TokenTree::Delimited(group_span, DelimToken::Bracket, literal) => {
                        let entire_group_span = group_span.entire();
                        match literal.trees().collect::<Vec<_>>()[..] {
                          [TokenTree::Token(Token { kind: Literal(Lit { kind: Str, .. }), .. })] =>
                            (),
                          _ => panic!("???"),
                        }
                        let mut p = cx.new_parser_from_tts(&literal.trees().collect::<Vec<_>>());
                        let literal_expr = p
                          .parse_expr()
                          .expect("expected expression to parse as a literal string");
                        cx.expr_call_global(
                          entire_group_span,
                          vec![AstIdent::from_str("CaseElement"), AstIdent::from_str("Lit")],
                          vec![cx.expr_call_global(
                            entire_group_span,
                            vec![AstIdent::from_str("Literal"), AstIdent::from_str("from")],
                            vec![literal_expr],
                          )],
                        )
                      },
                      TokenTree::Delimited(group_span, DelimToken::Paren, arg_binding) => {
                        let entire_group_span = group_span.entire();
                        match arg_binding.trees().collect::<Vec<_>>()[..] {
                          [
                            TokenTree::Token(Token {
                              kind: Ident(arg_name, _is_raw),
                              ..
                            }),
                            TokenTree::Token(Token {
                              kind: Colon,
                              ..
                            }),
                            TokenTree::Token(Token {
                              kind: Ident(arg_type, _is_raw2),
                              ..
                            }),
                          ] => {
                            cx.expr_call_global(
                              entire_group_span,
                              vec![AstIdent::from_str("CaseElement"), AstIdent::from_str("Prod")],
                              vec![cx.expr_call_global(
                                entire_group_span,
                                vec![AstIdent::from_str("ProductionReference"), AstIdent::from_str("new")],
                                vec![]
                              )]
                            )
                          }
                        }
                      },
                      _ => panic!("???"),
                    }).collect();
                },
              }
            },
          });
        },
        _ => panic!("???"),
      },
    _ => panic!("???"),
  }).collect();

  MacEager::expr(cx.expr_call_global(
    sp,
    vec![
      AstIdent::from_str("TypedSimultaneousProductions"),
      AstIdent::from_str("new"),
    ],
    vec![],
  ))
}
