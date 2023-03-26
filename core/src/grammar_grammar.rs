/*
 * Description: Convert an EBNF-like syntax into an executable SimultaneousProductions instance.
 *
 * Copyright (C) 2023 Danny McClanahan <dmcC2@hypnicjerk.ai>
 * SPDX-License-Identifier: AGPL-3.0
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/* FIXME: update the "Line Format" docs!! */
//! Convert an EBNF-like syntax into an executable
//! [`SimultaneousProductions`](gs::synthesis::SimultaneousProductions) instance!
//!
//! # Line Format
//! The format expects any number of lines of text formatted like:
//! 1. `Line` = `ProductionName: CaseDefinition`
//! 1. `ProductionName` = `/\$([^\$]|\$\$)*\$/`
//! 1. `CaseDefinition` = `CaseHead( -> CaseHead)*`
//! 1. `CaseHead` = `ProductionRef|Literal`
//! 1. `ProductionRef` = `ProductionName`
//! 1. `Literal` = `/<([^>]|>>)*>/`
//!
//! Each `Line` appends a new `CaseDefinition` to the list of [`Case`](Case)s registered for the
//! production named `ProductionName`! Each case is a sequence of `CaseHead`s demarcated by the
//! ` -> ` literal, and each `CaseHead` may either be a `ProductionRef` (which is just
//! a `ProductionName`) or a `Literal` (which is a sequence of
//! unicode characters).
//!
//! **Note that leading and trailing whitespace is trimmed from each line, and lines that are
//! empty or contain only whitespace are skipped.**
//!
//! Here's an example of a parser that accepts the input `/abab?/` for the production `$B`:
//!```
//! use sp_core::grammar_specification::constraints::SerializableGrammar;
//! use sp_core::grammar_grammar::*;
//! use sp_core::text_backend::*;
//!
//! let sp = SP::parse(&SPTextFormat::from(
//!   "\
//! $A$: <ab>
//! $B$: <ab> -> $A$
//! $B$: $A$ -> <a>
//! ".to_string()
//! )).unwrap();
//!
//! assert_eq!(
//!   sp,
//!   SP::from(
//!     [
//!       (
//!         ProductionReference::from("A"),
//!         Production::from([Case::from([CE::Lit(Lit::from("ab"))].as_ref())].as_ref()),
//!       ),
//!       (
//!         ProductionReference::from("B"),
//!         Production::from(
//!           [
//!             Case::from(
//!               [
//!                 CE::Lit(Lit::from("ab")),
//!                 CE::Prod(ProductionReference::from("A")),
//!               ]
//!               .as_ref(),
//!             ),
//!             Case::from(
//!               [
//!                 CE::Prod(ProductionReference::from("A")),
//!                 CE::Lit(Lit::from("a")),
//!               ]
//!               .as_ref(),
//!             ),
//!           ]
//!           .as_ref(),
//!         ),
//!       ),
//!     ]
//!     .as_ref(),
//!   )
//! );
//!```
//!
//! ## Literal `>` and `$`
//! To form a single literal `>` character, provide `>>` within a `Literal`. To form a single
//! literal `$` character for a production name, provide `$$` within a `ProductionName`:
//!```
//! use sp_core::grammar_specification::constraints::SerializableGrammar;
//! use sp_core::grammar_grammar::*;
//! use sp_core::text_backend::*;
//!
//! let sp = SP::parse(&SPTextFormat::from(
//!   "$A$$B$: <a>>b>".to_string()
//! )).unwrap();
//!
//! assert_eq!(
//!   sp,
//!   SP::from(
//!     [(
//!       ProductionReference::from("A$B"),
//!       Production::from([Case::from([CE::Lit(Lit::from("a>b"))].as_ref())].as_ref())
//!     )]
//!     .as_ref()
//!   )
//! );
//!```
//!
//! ## `->` are ignored
//!```
//! use sp_core::grammar_specification::constraints::SerializableGrammar;
//! use sp_core::grammar_grammar::*;
//! use sp_core::text_backend::*;
//!
//! let sp1 = SP::parse(&SPTextFormat::from(
//!   "$A$: (<a> <b>) $C$".to_string()
//! )).unwrap();
//! let sp2 = SP::parse(&SPTextFormat::from(
//!   "$A$: (<a> -> <b>) -> $C$".to_string()
//! )).unwrap();
//!
//! assert_eq!(sp1, sp2);
//!```
//!
//! ## Grouped Elements
//!```
//! use sp_core::grammar_specification::constraints::SerializableGrammar;
//! use sp_core::grammar_grammar::*;
//! use sp_core::text_backend::*;
//!
//! let sp = SP::parse(&SPTextFormat::from(
//!   "\
//! $A$: (<a>)
//! $A$: ($A$ -> (<b>))".to_string()
//! )).unwrap();
//!
//! assert_eq!(
//!   sp,
//!   SP::from(
//!     [(
//!       ProductionReference::from("A"),
//!       Production::from([
//!         Case::from([CE::Group(Group {
//!           elements: vec![CE::Lit(Lit::from("a"))],
//!           ..Default::default()
//!         })].as_ref()),
//!         Case::from([CE::Group(Group {
//!           elements: vec![
//!             CE::Prod(ProductionReference::from("A")),
//!             CE::Group(Group {
//!               elements: vec![CE::Lit(Lit::from("b"))],
//!               ..Default::default()
//!             }),
//!           ],
//!           ..Default::default()
//!         })].as_ref()),
//!       ].as_ref())
//!     )]
//!     .as_ref()
//!   )
//! );
//!```
//!
//! ### `?` operator
//!```
//! use sp_core::grammar_specification::constraints::SerializableGrammar;
//! use sp_core::grammar_specification::synthesis::GroupOperator;
//! use sp_core::grammar_grammar::*;
//! use sp_core::text_backend::*;
//!
//! let sp = SP::parse(&SPTextFormat::from(
//!   "\
//! $A$: <a>?
//! $A$: <a>??
//! $A$: (<a>)?
//! $A$: (<a>?)
//! $A$: (<a>?)?
//! $A$: (<a> -> $B$?) -> <c>?".to_string()
//! )).unwrap();
//!
//! assert_eq!(
//!   sp,
//!   SP::from(
//!     [(
//!       ProductionReference::from("A"),
//!       Production::from([
//!         Case::from([CE::Group(Group {
//!           elements: vec![CE::Lit(Lit::from("a"))],
//!           op: GroupOperator::Optional,
//!         })].as_ref()),
//!         Case::from([CE::Group(Group {
//!           elements: vec![CE::Lit(Lit::from("a"))],
//!           op: GroupOperator::Optional,
//!         })].as_ref()),
//!         Case::from([CE::Group(Group {
//!           elements: vec![CE::Lit(Lit::from("a"))],
//!           op: GroupOperator::Optional,
//!         })].as_ref()),
//!         Case::from([CE::Group(Group {
//!           elements: vec![CE::Group(Group {
//!             elements: vec![CE::Lit(Lit::from("a"))],
//!             op: GroupOperator::Optional,
//!           })],
//!           op: GroupOperator::NoOp,
//!         })].as_ref()),
//!         Case::from([CE::Group(Group {
//!           elements: vec![CE::Group(Group {
//!             elements: vec![CE::Lit(Lit::from("a"))],
//!             op: GroupOperator::Optional,
//!           })],
//!           op: GroupOperator::Optional,
//!         })].as_ref()),
//!         Case::from([
//!           CE::Group(Group {
//!             elements: vec![
//!               CE::Lit(Lit::from("a")),
//!               CE::Group(Group {
//!                 elements: vec![CE::Prod(ProductionReference::from("B"))],
//!                 op: GroupOperator::Optional,
//!               }),
//!             ],
//!             op: GroupOperator::NoOp,
//!           }),
//!           CE::Group(Group {
//!             elements: vec![CE::Lit(Lit::from("c"))],
//!             op: GroupOperator::Optional,
//!           }),
//!         ].as_ref()),
//!       ].as_ref())
//!     )]
//!     .as_ref()
//!   )
//! );
//!```

use crate::{
  grammar_specification::{self as gs, constraints::SerializableGrammar},
  text_backend::{Case, Group, Lit, Production, ProductionReference, CE, SP},
};

use displaydoc::Display;
use regex::Regex;
use thiserror::Error;

use core::{
  iter::{IntoIterator, Iterator},
  str,
};

#[derive(Debug, Display, Error, Clone)]
pub enum GrammarGrammarParsingError {
  /// line {0} didn't match LINE: '{1}'
  LineMatchFailed(String, &'static Regex),
  /// case {0} didn't match CASE: '{1}'
  CaseMatchFailed(String, &'static Regex),
  /// unmatched close paren
  UnmatchedCloseParen,
  /// unmatched open paren
  UnmatchedOpenParen,
  /// unmatched prefix operator
  UnmatchedPrefixOperator,
}

/// grammar definition: "{0}"
#[derive(Debug, Clone, Display, Eq, PartialEq, Hash)]
pub struct SPTextFormat(String);

impl From<String> for SPTextFormat {
  fn from(s: String) -> Self {
    Self(s)
  }
}

impl AsRef<str> for SPTextFormat {
  fn as_ref(&self) -> &str {
    self.0.as_str()
  }
}

impl SerializableGrammar for SP {
  type Out = SPTextFormat;

  type ParseError = GrammarGrammarParsingError;
  fn parse(out: &Self::Out) -> Result<Self, Self::ParseError>
  where
    Self: Sized,
  {
    use indexmap::IndexMap;
    use lazy_static::lazy_static;

    use std::cell::RefCell;
    use std::rc::Rc;

    let grammar: &str = out.as_ref();

    const CASE_EL_PATTERN: &str =
      "\\$(?:[^\\$]|\\$\\$)*\\$|<(?:[^>]|>>)*>|[\\(\\)]|\\?|->|[[:space:]]+";

    lazy_static! {
      static ref EMPTY_SPACE: Regex = Regex::new("^[[:space:]]*$").unwrap();
      static ref LINE: Regex = Regex::new(
        "^[[:space:]]*(?P<prod>\\$(?:[^\\$]|\\$\\$)*\\$):[[:space:]]*(?P<rest>.+)[[:space:]]*$"
      )
      .unwrap();
      static ref CASE_EL: Regex = Regex::new(CASE_EL_PATTERN).unwrap();
      static ref CASE: Regex = Regex::new(format!("^(?:{0})+$", CASE_EL_PATTERN).as_str()).unwrap();
    }

    fn parse_doubled_escape(escape_char: char, s: &str) -> String {
      /* (1) Strip the end marker. */
      assert!(s.ends_with(escape_char));
      let s = &s[..(s.len() - 1)];
      dbg!(s);
      /* (2) Un-escape any doubled `escape_char`. */
      let mut prior_escape_char: bool = false;
      let mut chars: Vec<char> = Vec::new();
      for c in s.chars() {
        if c == escape_char {
          if prior_escape_char {
            chars.push(c);
            prior_escape_char = false;
          } else {
            prior_escape_char = true;
          }
        } else {
          assert!(
            !prior_escape_char,
            "no undoubled escape char should be here!"
          );
          chars.push(c);
        }
      }
      chars.into_iter().collect()
    }

    let mut cases: IndexMap<String, Vec<Vec<CE>>> = IndexMap::new();

    for line in grammar.lines() {
      /* LINE trims off any leading or trailing whitespace. */
      let caps = match LINE.captures(line) {
        Some(caps) => caps,
        None => {
          /* Ignore lines that are empty or contain only spaces. */
          if EMPTY_SPACE.is_match(line) {
            continue;
          } else {
            return Err(GrammarGrammarParsingError::LineMatchFailed(
              line.to_string(),
              &LINE,
            ));
          }
        },
      };
      let prod = caps.name("prod").unwrap().as_str();
      dbg!(prod);
      assert!(prod.starts_with('$'));
      let prod = parse_doubled_escape('$', &prod[1..]);
      dbg!(&prod);
      let rest = caps.name("rest").unwrap().as_str();

      dbg!(rest);
      /* CASE trims off any trailing whitespace that wasn't caught by the LINE pattern. */
      /* (This is likely due to longest-first matching.) */
      if !CASE.is_match(rest) {
        return Err(GrammarGrammarParsingError::CaseMatchFailed(
          rest.to_string(),
          &CASE,
        ));
      }

      #[derive(Debug, Clone)]
      struct Context {
        pub case_els: Vec<CE>,
        pub parent: Option<Rc<RefCell<Context>>>,
      }

      impl Context {
        pub fn new() -> Self {
          Self {
            case_els: Vec::new(),
            parent: None,
          }
        }
      }

      fn process_case_el(
        ctx: &mut Rc<RefCell<Context>>,
        case_el: &str,
      ) -> Result<(), GrammarGrammarParsingError> {
        match case_el {
          "(" => {
            *ctx = Rc::new(RefCell::new(Context {
              case_els: Vec::new(),
              parent: Some(Rc::clone(ctx)),
            }));
          },
          ")" => {
            let inner_case_els = ctx.borrow().case_els.clone();
            let parent: Rc<RefCell<Context>> = ctx
              .borrow_mut()
              .parent
              .as_mut()
              .cloned()
              .ok_or(GrammarGrammarParsingError::UnmatchedCloseParen)?;
            parent.borrow_mut().case_els.push(CE::Group(Group {
              elements: inner_case_els,
              op: gs::synthesis::GroupOperator::NoOp,
            }));
            *ctx = parent;
          },
          "?" => {
            let mut ctx = ctx.borrow_mut();
            let prior_element: &mut CE = ctx
              .case_els
              .last_mut()
              .ok_or(GrammarGrammarParsingError::UnmatchedPrefixOperator)?;
            *prior_element = match prior_element {
              CE::Lit(lit) => CE::Group(Group {
                elements: vec![CE::Lit(lit.clone())],
                op: gs::synthesis::GroupOperator::Optional,
              }),
              CE::Prod(prod_ref) => CE::Group(Group {
                elements: vec![CE::Prod(prod_ref.clone())],
                op: gs::synthesis::GroupOperator::Optional,
              }),
              CE::Group(ref mut group) => CE::Group(match group.op.clone() {
                /* Discard repeated ??. */
                gs::synthesis::GroupOperator::Optional => group.clone(),
                /* Convert a no-op to a ?. */
                gs::synthesis::GroupOperator::NoOp => {
                  group.op = gs::synthesis::GroupOperator::Optional;
                  group.clone()
                },
                /* Otherwise, wrap the group within an outer group with the ?. */
                _ => {
                  group.elements = vec![CE::Group(group.clone())];
                  group.op = gs::synthesis::GroupOperator::Optional;
                  group.clone()
                },
              }),
            };
          },
          "->" => { /* NB: these arrows are optional and do nothing! */ },
          case_el if case_el.starts_with('$') => {
            let prod_ref = parse_doubled_escape('$', &case_el[1..]);
            ctx
              .borrow_mut()
              .case_els
              .push(CE::Prod(ProductionReference::from(prod_ref.as_str())));
          },
          case_el if case_el.starts_with('<') => {
            let lit = parse_doubled_escape('>', &case_el[1..]);
            ctx
              .borrow_mut()
              .case_els
              .push(CE::Lit(Lit::from(lit.as_str())));
          },
          case_el => {
            assert!(EMPTY_SPACE.is_match(case_el));
          },
        }
        Ok(())
      }

      let mut ctx = Rc::new(RefCell::new(Context::new()));
      for case_el in CASE_EL.find_iter(rest).map(|m| m.as_str()) {
        process_case_el(&mut ctx, case_el)?;
      }
      if ctx.borrow().parent.is_some() {
        return Err(GrammarGrammarParsingError::UnmatchedOpenParen);
      }

      cases
        .entry(prod.to_string())
        .or_insert_with(Vec::new)
        .push(ctx.borrow().case_els.clone());
    }

    let cases: Vec<(ProductionReference, Production)> = cases
      .into_iter()
      .map(|(pr, prod)| {
        let cases: Vec<Case> = prod
          .into_iter()
          .map(|case_els| Case::from(&case_els[..]))
          .collect();
        (
          ProductionReference::from(pr.as_str()),
          Production::from(&cases[..]),
        )
      })
      .collect();
    Ok(SP::from(&cases[..]))
  }

  type SerializeError = ();
  fn serialize(&self) -> Result<Self::Out, Self::SerializeError> {
    use itertools::Itertools; /* for intersperse() */

    let mut lines: Vec<String> = Vec::new();
    for (cur_prod_ref, prod) in self.clone() {
      let cur_prod_ref = cur_prod_ref.into_string().replace('$', "$$");
      for case in prod {
        let mut case_elements: Vec<String> = Vec::new();
        for case_el in case {
          fn format_lit(lit: Lit) -> String {
            format!("<{0}>", lit.into_string().replace('>', ">>"))
          }
          fn format_prod_ref(prod_ref: ProductionReference) -> String {
            format!("${0}$", prod_ref.into_string().replace('$', "$$"))
          }
          fn format_group(group: Group) -> String {
            let Group { elements, op } = group;
            let joined_elements: String = elements
              .into_iter()
              .map(|group_el| match group_el {
                CE::Lit(lit) => format_lit(lit),
                CE::Prod(prod_ref) => format_prod_ref(prod_ref),
                CE::Group(group) => format_group(group),
              })
              .intersperse(" -> ".to_string())
              .collect();
            format!("({0}){1}", joined_elements, op)
          }

          let formatted_case_el = match case_el {
            CE::Lit(lit) => format_lit(lit),
            CE::Prod(prod_ref) => format_prod_ref(prod_ref),
            CE::Group(group) => format_group(group),
          };
          case_elements.push(formatted_case_el);
        }
        let case_elements: String = case_elements
          .into_iter()
          .intersperse(" -> ".to_string())
          .collect();
        let case_line = format!("${0}$: {1}", cur_prod_ref, case_elements);
        lines.push(case_line);
      }
    }

    let lines: String = lines.into_iter().intersperse("\n".to_string()).collect();
    Ok(SPTextFormat::from(lines))
  }
}

#[cfg(test)]
pub mod proptest_strategies {
  use super::*;
  use crate::grammar_specification as gs;

  use proptest::{prelude::*, strategy::Strategy};

  prop_compose! {
    pub fn production_name(ensure_cash: bool)
      (prod_name in prop::string::string_regex(".*").unwrap(),
       selector in any::<prop::sample::Selector>()) -> ProductionReference {
        if ensure_cash {
          let cash_index = selector.select(
            prod_name.as_str().char_indices().map(|(i, _)| i)
              .chain([prod_name.len()].iter().cloned())
          );
          let (left, right) = prod_name.as_str().split_at(cash_index);
          let new_prod_name = format!("{0}${1}", left, right);
          ProductionReference::from(new_prod_name.as_str())
        } else {
          ProductionReference::from(prod_name.as_str())
        }
      }
  }
  prop_compose! {
    pub fn prod_names(ensure_cash: bool, min_size: usize, max_size: usize)
      (all_names in prop::collection::vec(production_name(ensure_cash), min_size..=max_size)
       .prop_filter("no duplicate prod names allowed",
                    |v| {
                      let mut v_unique = v.clone();
                      v_unique.sort_unstable();
                      v_unique.dedup();
                      v_unique.len() == v.len()
                    })
      ) -> Vec<ProductionReference> {
      all_names
    }
  }
  prop_compose! {
    pub fn literal(ensure_arrow: bool)
      (lit in prop::string::string_regex(".*").unwrap(),
       selector in any::<prop::sample::Selector>()) -> Lit {
        if ensure_arrow {
          let arrow_index = selector.select(
            lit.as_str().char_indices().map(|(i, _)| i)
              .chain([lit.len()].iter().cloned())
          );
          let (left, right) = lit.as_str().split_at(arrow_index);
          let new_lit = format!("{0}>{1}", left, right);
          Lit::from(new_lit.as_str())
        } else {
          Lit::from(lit.as_str())
        }
      }
  }
  prop_compose! {
    pub fn valid_prod_ref(refs: Vec<ProductionReference>)
      (ref_index in 0..refs.len()) -> ProductionReference {
        refs[ref_index].clone()
      }
  }
  pub fn group_op() -> impl Strategy<Value = gs::synthesis::GroupOperator> {
    prop_oneof![
      Just(gs::synthesis::GroupOperator::NoOp),
      Just(gs::synthesis::GroupOperator::Optional),
    ]
  }
  pub fn case_element(
    refs: Vec<ProductionReference>,
    ensure_arrow: bool,
    group_min_length: usize,
    group_max_length: usize,
    remaining_depth: usize,
  ) -> impl Strategy<Value = CE> {
    if remaining_depth == 0 {
      prop_oneof![
        literal(ensure_arrow).prop_map(|lit| CE::Lit(lit)),
        valid_prod_ref(refs.clone()).prop_map(|prod_ref| CE::Prod(prod_ref)),
      ]
      .boxed()
    } else {
      prop_oneof![
        literal(ensure_arrow).prop_map(|lit| CE::Lit(lit)),
        valid_prod_ref(refs.clone()).prop_map(|prod_ref| CE::Prod(prod_ref)),
        group(
          refs.clone(),
          ensure_arrow,
          group_min_length,
          group_max_length,
          remaining_depth,
        )
        .prop_map(|group| CE::Group(group)),
      ]
      .boxed()
    }
  }
  pub fn case(
    refs: Vec<ProductionReference>,
    ensure_arrow: bool,
    min_length: usize,
    max_length: usize,
    remaining_depth: usize,
  ) -> impl Strategy<Value = Case> {
    prop::collection::vec(
      /* NB: we reuse {min,max}_length for case length and group length for simplicity! */
      case_element(refs, ensure_arrow, min_length, max_length, remaining_depth),
      min_length..=max_length,
    )
    .prop_map(|elements| Case::via_into_iter(elements.into_iter()))
  }
  prop_compose! {
    pub fn group(
      refs: Vec<ProductionReference>,
      ensure_arrow: bool,
      min_length: usize,
      max_length: usize,
      remaining_depth: usize,
      /* NB: we reduce remaining_depth by 1 here! */
    )(
      c in case(refs, ensure_arrow, min_length, max_length, remaining_depth - 1).boxed(),
      op in group_op(),
    ) -> Group {
      let c: Case = c;
      let elements: Vec<CE> = c.into_iter().collect();
      Group {
        elements,
        op,
      }
    }
  }
  pub fn production(
    refs: Vec<ProductionReference>,
    ensure_arrow: bool,
    min_case_length: usize,
    max_case_length: usize,
    min_size: usize,
    max_size: usize,
    max_group_depth: usize,
  ) -> impl Strategy<Value = Production> {
    prop::collection::vec(
      case(
        refs,
        ensure_arrow,
        min_case_length,
        max_case_length,
        max_group_depth,
      ),
      min_size..=max_size,
    )
    .prop_map(|cases| Production::via_into_iter(cases.into_iter()))
  }
  pub fn sp(
    ensure_cash: bool,
    ensure_arrow: bool,
    min_prods: usize,
    max_prods: usize,
    min_cases: usize,
    max_cases: usize,
    min_case_els: usize,
    max_case_els: usize,
    max_group_depth: usize,
  ) -> impl Strategy<Value = SP> {
    prod_names(ensure_cash, min_prods, max_prods)
      .prop_flat_map(move |names: Vec<ProductionReference>| {
        let n = names.len();
        (
          Just(names.clone()),
          prop::collection::vec(
            production(
              names,
              ensure_arrow.clone(),
              min_case_els,
              max_case_els,
              min_cases,
              max_cases,
              max_group_depth,
            ),
            n,
          ),
        )
      })
      .prop_map(
        |(names, prods): (Vec<ProductionReference>, Vec<Production>)| {
          let joined: Vec<(ProductionReference, Production)> =
            names.into_iter().zip(prods.into_iter()).collect();
          SP::via_into_iter(joined.into_iter())
        },
      )
      .prop_filter("must have at least one prod ref", move |sp: &SP| {
        !ensure_cash || {
          sp.as_new_vec().into_iter().any(|(_, prod)| {
            prod.as_new_vec().into_iter().any(|case| {
              case
                .as_new_vec()
                .into_iter()
                .any(|case_el| matches![case_el, CE::Prod(_)])
            })
          })
        }
      })
      .prop_filter("must have at least one nonempty literal", move |sp: &SP| {
        !ensure_arrow || {
          sp.as_new_vec().into_iter().any(|(_, prod)| {
            prod.as_new_vec().into_iter().any(|case| {
              case.as_new_vec().into_iter().any(|case_el| match case_el {
                CE::Lit(lit) => {
                  assert!(!lit.into_string().is_empty());
                  true
                },
                _ => false,
              })
            })
          })
        }
      })
  }

  #[test]
  fn test_prod_vec() {
    use proptest::{strategy::ValueTree, test_runner::TestRunner};

    let mut runner = TestRunner::deterministic();
    let prod_names: Vec<ProductionReference> = prod_names(false, 2, 5)
      .new_tree(&mut runner)
      .unwrap()
      .current();
    assert!(prod_names.len() >= 2);
    assert!(prod_names.len() <= 5);
    assert_ne!(prod_names[0], prod_names[1]);
  }
}

#[cfg(test)]
mod test {
  use super::{proptest_strategies::*, *};

  use proptest::prelude::*;

  proptest! {
    #[test]
    fn test_serde(sp in sp(false, false, 1, 20, 1, 5, 1, 5, 3)) {
      let text_grammar = sp.serialize().unwrap();
      prop_assert_eq!(sp, SP::parse(&text_grammar).unwrap());
    }
  }
  proptest! {
    #[test]
    fn test_serde_cash(sp in sp(true, false, 1, 20, 1, 5, 1, 5, 3)) {
      let text_grammar = sp.serialize().unwrap();
      prop_assert_eq!(sp, SP::parse(&text_grammar).unwrap());
    }
  }
  proptest! {
    #[test]
    fn test_serde_arrow(sp in sp(false, true, 1, 20, 1, 5, 1, 5, 3)) {
      let text_grammar = sp.serialize().unwrap();
      prop_assert_eq!(sp, SP::parse(&text_grammar).unwrap());
    }
  }
  proptest! {
    #[test]
    fn test_serde_cash_arrow(sp in sp(true, true, 1, 20, 1, 5, 1, 5, 3)) {
      let text_grammar = sp.serialize().unwrap();
      prop_assert_eq!(sp, SP::parse(&text_grammar).unwrap());
    }
  }

  #[test]
  fn test_serialize() {
    let sp = SP::from(
      [(
        ProductionReference::from("A"),
        Production::from(
          [
            Case::from([CE::Lit(Lit::from("a"))].as_ref()),
            Case::from(
              [
                CE::Prod(ProductionReference::from("A")),
                CE::Lit(Lit::from("b")),
              ]
              .as_ref(),
            ),
          ]
          .as_ref(),
        ),
      )]
      .as_ref(),
    );

    assert_eq!(
      sp.serialize().unwrap(),
      SPTextFormat::from(
        "\
$A$: <a>
$A$: $A$ -> <b>"
          .to_string()
      )
    );
  }

  #[test]
  fn test_serialize_escapes_double_cash() {
    let sp = SP::from(
      [(
        ProductionReference::from("A$A"),
        Production::from(
          [Case::from(
            [
              CE::Prod(ProductionReference::from("A$A")),
              CE::Lit(Lit::from("b")),
            ]
            .as_ref(),
          )]
          .as_ref(),
        ),
      )]
      .as_ref(),
    );

    let text_grammar = sp.serialize().unwrap();

    assert_eq!(
      &text_grammar,
      &SPTextFormat::from("$A$$A$: $A$$A$ -> <b>".to_string())
    );

    assert_eq!(SP::parse(&text_grammar).unwrap(), sp);
  }

  #[test]
  fn test_empty() {
    let sp = SP::parse(&SPTextFormat::from("".to_string())).unwrap();

    assert_eq!(sp, SP::from([].as_ref()));
  }

  #[test]
  fn test_strips_whitespace() {
    let sp = SP::parse(&SPTextFormat::from(" $A$: <a> ".to_string())).unwrap();

    assert_eq!(
      sp,
      SP::from(
        [(
          ProductionReference::from("A"),
          Production::from([Case::from([CE::Lit(Lit::from("a"))].as_ref())].as_ref())
        )]
        .as_ref()
      )
    );
  }

  #[test]
  fn test_escapes_double_right_arrow() {
    let sp = SP::parse(&SPTextFormat::from("$A$: <a>>b>".to_string())).unwrap();

    assert_eq!(
      sp,
      SP::from(
        [(
          ProductionReference::from("A"),
          Production::from([Case::from([CE::Lit(Lit::from("a>b"))].as_ref())].as_ref())
        )]
        .as_ref()
      )
    );

    assert_eq!(sp.serialize().unwrap().as_ref(), "$A$: <a>>b>");
  }

  #[test]
  fn test_production_ref() {
    let sp = SP::parse(&SPTextFormat::from("$A$: $B$".to_string())).unwrap();

    assert_eq!(
      sp,
      SP::from(
        [(
          ProductionReference::from("A"),
          Production::from(
            [Case::from(
              [CE::Prod(ProductionReference::from("B"))].as_ref()
            )]
            .as_ref()
          )
        )]
        .as_ref()
      ),
    );
  }

  #[test]
  fn test_line_match_fail() {
    match SP::parse(&SPTextFormat::from("A = B".to_string())) {
      Err(GrammarGrammarParsingError::LineMatchFailed(line, _)) => {
        assert_eq!(&line, "A = B");
      },
      _ => unreachable!(),
    };
  }

  #[test]
  fn test_case_match_fail() {
    match SP::parse(&SPTextFormat::from("$A$: asdf".to_string())) {
      Err(GrammarGrammarParsingError::CaseMatchFailed(case, _)) => {
        assert_eq!(&case, "asdf");
      },
      _ => unreachable!(),
    }
  }

  /* FIXME: make proptest versions to generate strings that fail here too! */
  #[test]
  fn test_unmatched_open_paren() {
    let sp = SP::parse(&SPTextFormat::from("$A$: (()".to_string()));
    assert!(matches![sp, Err(GrammarGrammarParsingError::UnmatchedOpenParen)]);
  }

  #[test]
  fn test_unmatched_close_paren() {
    let sp = SP::parse(&SPTextFormat::from("$A$: ())".to_string()));
    assert!(matches![sp, Err(GrammarGrammarParsingError::UnmatchedCloseParen)]);
  }

  #[test]
  fn test_unmatched_prefix_operator() {
    let sp = SP::parse(&SPTextFormat::from("$A$: ?".to_string()));
    assert!(matches![sp, Err(GrammarGrammarParsingError::UnmatchedPrefixOperator)]);
  }
}
