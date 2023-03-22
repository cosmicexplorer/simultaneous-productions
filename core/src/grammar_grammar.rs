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

use crate::{
  grammar_specification::constraints::SerializableGrammar,
  text_backend::{Case, Lit, Production, ProductionReference, CE, SP},
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

    let grammar: &str = out.as_ref();

    lazy_static! {
      static ref MAYBE_SPACE: Regex = Regex::new("^[[:space:]]*$").unwrap();
      static ref LINE: Regex = Regex::new(
        "^[[:space:]]*(?P<prod>\\$(?:[^\\$]|\\$\\$)*\\$):[[:space:]]*(?P<rest>.+)[[:space:]]*$"
      ).unwrap();
      static ref CASE: Regex = Regex::new(
        "^(?P<head>\\$(?:[^\\$]|\\$\\$)*\\$|<(?:[^>]|>>)*>)(?:[[:space:]]*->[[:space:]]*(?P<tail>.+))?[[:space:]]*$"
      )
      .unwrap();
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
          if MAYBE_SPACE.is_match(line) {
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

      let mut case_els: Vec<CE> = Vec::new();
      dbg!(rest);
      /* CASE trims off any trailing whitespace that wasn't caught by the LINE pattern. */
      /* (This is likely due to longest-first matching.) */
      let caps = CASE
        .captures(rest)
        .ok_or_else(|| GrammarGrammarParsingError::CaseMatchFailed(rest.to_string(), &CASE))?;
      let head = caps.name("head").unwrap().as_str();
      let mut tail = caps.name("tail").map(|c| c.as_str());

      fn parse_case_element(case_el: &str) -> CE {
        if case_el.starts_with('$') {
          let prod_ref = parse_doubled_escape('$', &case_el[1..]);
          CE::Prod(ProductionReference::from(prod_ref.as_str()))
        } else {
          assert!(case_el.starts_with('<'));
          let lit = parse_doubled_escape('>', &case_el[1..]);
          CE::Lit(Lit::from(lit.as_str()))
        }
      }

      let cur_ce = parse_case_element(head);
      case_els.push(cur_ce);

      while let Some(cur_tail_nonempty) = tail {
        dbg!(cur_tail_nonempty);
        let caps = CASE.captures(cur_tail_nonempty).ok_or_else(|| {
          GrammarGrammarParsingError::CaseMatchFailed(cur_tail_nonempty.to_string(), &CASE)
        })?;
        let head = caps.name("head").unwrap().as_str();
        /* Mutate `tail` here, which will affect the `while` condition. */
        tail = caps.name("tail").map(|c| c.as_str());

        let cur_ce = parse_case_element(head);
        case_els.push(cur_ce);
      }

      cases
        .entry(prod.to_string())
        .or_insert_with(Vec::new)
        .push(case_els);
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
          match case_el {
            CE::Lit(lit) => {
              case_elements.push(format!("<{0}>", lit.into_string().replace('>', ">>")));
            },
            CE::Prod(prod_ref) => {
              case_elements.push(format!("${0}$", prod_ref.into_string().replace('$', "$$")));
            },
          }
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
  pub fn case_element(
    refs: Vec<ProductionReference>,
    ensure_arrow: bool,
  ) -> impl Strategy<Value = CE> {
    prop_oneof![
      literal(ensure_arrow).prop_map(|lit| CE::Lit(lit)),
      valid_prod_ref(refs).prop_map(|prod_ref| CE::Prod(prod_ref)),
    ]
  }
  pub fn case(
    refs: Vec<ProductionReference>,
    ensure_arrow: bool,
    min_length: usize,
    max_length: usize,
  ) -> impl Strategy<Value = Case> {
    prop::collection::vec(case_element(refs, ensure_arrow), min_length..=max_length)
      .prop_map(|elements| Case::via_into_iter(elements.into_iter()))
  }
  pub fn production(
    refs: Vec<ProductionReference>,
    ensure_arrow: bool,
    min_case_length: usize,
    max_case_length: usize,
    min_size: usize,
    max_size: usize,
  ) -> impl Strategy<Value = Production> {
    prop::collection::vec(
      case(refs, ensure_arrow, min_case_length, max_case_length),
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
    fn test_serde(sp in sp(false, false, 1, 20, 1, 5, 1, 5)) {
      let text_grammar = sp.serialize().unwrap();
      assert_eq!(sp, SP::parse(&text_grammar).unwrap());
    }
  }
  proptest! {
    #[test]
    fn test_serde_cash(sp in sp(true, false, 1, 20, 1, 5, 1, 5)) {
      let text_grammar = sp.serialize().unwrap();
      assert_eq!(sp, SP::parse(&text_grammar).unwrap());
    }
  }
  proptest! {
    #[test]
    fn test_serde_arrow(sp in sp(false, true, 1, 20, 1, 5, 1, 5)) {
      let text_grammar = sp.serialize().unwrap();
      assert_eq!(sp, SP::parse(&text_grammar).unwrap());
    }
  }
  proptest! {
    #[test]
    fn test_serde_cash_arrwo(sp in sp(true, true, 1, 20, 1, 5, 1, 5)) {
      let text_grammar = sp.serialize().unwrap();
      assert_eq!(sp, SP::parse(&text_grammar).unwrap());
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
}
