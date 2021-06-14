/*
 * Description: ???
 *
 * Copyright (C) 2019-2021 Danny McClanahan <dmcC2@hypnicjerk.ai>
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
#![feature(fn_traits)]
#![feature(trace_macros)]
#![feature(trait_alias)]
/* These clippy lint descriptions are purely non-functional and do not affect the functionality
 * or correctness of the code.
 * TODO: rustfmt breaks multiline comments when used one on top of another! (each with its own
 * pair of delimiters)
 * Note: run clippy with: rustup run nightly cargo-clippy! */
#![allow(missing_docs)]
/* Ensure any doctest warnings fails the doctest! */
#![doc(test(attr(deny(warnings))))]
/* Enable all clippy lints except for many of the pedantic ones. It's a shame this needs to be
 * copied and pasted across crates, but there doesn't appear to be a way to include inner
 * attributes from a common source. */
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
/* It is often more clear to show that nothing is being moved. */
#![allow(clippy::match_ref_pats)]
/* Subjective style. */
#![allow(
  clippy::derive_hash_xor_eq,
  clippy::len_without_is_empty,
  clippy::redundant_field_names,
  clippy::too_many_arguments
)]
/* Default isn't as big a deal as people seem to think it is. */
#![allow(clippy::new_without_default, clippy::new_ret_no_self)]
/* Arc<Mutex> can be more clear than needing to grok Orderings. */
#![allow(clippy::mutex_atomic)]

#[macro_use]
pub mod binding;

/// The basic structs which define an input grammar.
///
/// While macros may be able to streamline the process of declaring a grammar,
/// their stability guarantees can be much lower than the definitions in this
/// module.
pub mod api {
  use sp_core::{
    grammar_specification as core_spec,
    graph_coordinates::{CaseElRef, CaseRef, Counter, ProdRef},
    token::Token,
  };

  use indexmap::IndexMap;

  use std::marker::PhantomData;

  /// A contiguous sequence of tokens.
  #[derive(Debug, Clone)]
  pub struct Literal<Tok: Token>(pub Vec<Tok>);

  impl From<&str> for Literal<char> {
    fn from(s: &str) -> Self { Self(s.chars().collect()) }
  }

  impl<Tok: Token> From<&[Tok]> for Literal<Tok> {
    fn from(s: &[Tok]) -> Self { Self(s.to_vec()) }
  }

  impl<'a, Tok: Token> From<Literal<Tok>> for core_spec::Literal<'a, Tok> {
    fn from(value: Literal<Tok>) -> Self { Self(&value) }
  }

  /// A reference to another production within the same set.
  ///
  /// The string must match the assigned name of a [Production] in a set of
  /// [SimultaneousProductions].
  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct ProductionReference<'a, Tok: Token>(String, PhantomData<&'a Tok>);

  impl<'a, Tok: Token> ProductionReference<'a, Tok> {
    pub fn new(s: &str) -> Self { ProductionReference(s.to_string(), PhantomData) }
  }

  #[derive(Debug)]
  pub enum LoweringError {
    InvalidProduction(String),
  }

  impl<'a, Tok: Token> Counter for ProductionReference<'a, Tok> {
    type Arena = SimultaneousProductions<'a, Tok>;
    type Value = Result<ProdRef<'a, Tok>, LoweringError>;

    fn dereference(&self, arena: Self::Arena) -> Self::Value {
      match arena.0.get_full(&self.0) {
        None => Err(LoweringError::InvalidProduction(format!(
          "target of production reference {:?} should exist in S.P. {:?}!",
          &self, arena,
        ))),
        Some((index, _, _)) => Ok(ProdRef(index)),
      }
    }
  }

  /// Each individual element that can be matched against some input in a case.
  #[derive(Debug, Clone)]
  pub enum CaseElement<'a, Tok: Token> {
    Lit(Literal<Tok>),
    Prod(ProductionReference<'a, Tok>),
  }

  /// A sequence of *elements* which, if successfully matched against some
  /// *input*, represents some *production*.
  #[derive(Debug, Clone)]
  pub struct Case<'a, Tok: Token>(pub Vec<CaseElement<'a, Tok>>);

  /// A disjunction of cases.
  #[derive(Debug, Clone)]
  pub struct Production<'a, Tok: Token>(pub Vec<Case<'a, Tok>>);

  /// A conjunction of productions.
  #[derive(Debug, Clone)]
  pub struct SimultaneousProductions<'a, Tok: Token>(
    pub IndexMap<ProductionReference<'a, Tok>, Production<'a, Tok>>,
  );
}

///
/// Syntax sugar for inline modifications to productions.
pub mod operators {
  use super::lowering_to_indices::graph_representation::*;

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct OperatorResult {
    pub result: Vec<CaseEl>,
  }

  pub trait UnaryOperator {
    fn operate(&self, prods: &mut DetokenizedProductions) -> OperatorResult;
  }

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct KleeneStar {
    pub group: Vec<CaseEl>,
  }

  impl UnaryOperator for KleeneStar {
    fn operate(&self, prods: &mut DetokenizedProductions) -> OperatorResult {
      let (new_prod_ref, &mut ProductionImpl(ref mut new_prod)) = prods.new_production();
      /* Add an empty case. */
      new_prod.push(CaseImpl(vec![]));
      /* Add a case traversing the initial group! */
      new_prod.push(CaseImpl(
        self
          .group
          .iter()
          .cloned()
          /* Allow circling back at the end! */
          .chain(vec![CaseEl::Prod(new_prod_ref)])
          .collect(),
      ));
      /* The result is just a single reference to the new production! */
      OperatorResult {
        result: vec![CaseEl::Prod(new_prod_ref)],
      }
    }
  }

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct Repeated {
    pub lower_bound: Option<usize>,
    pub upper_bound: Option<usize>,
    pub group: Vec<CaseEl>,
  }

  impl UnaryOperator for Repeated {
    fn operate(&self, prods: &mut DetokenizedProductions) -> OperatorResult {
      let prologue_length = self
        .lower_bound
        .map(|i| if i > 0 { i - 1 } else { i })
        .unwrap_or(0);
      let prologue = (0..prologue_length).flat_map(|_| self.group.clone());

      let epilogue: Box<dyn std::iter::Iterator<Item=CaseEl>> =
        if let Some(upper_bound) = self.upper_bound {
          /* If we have a definite upper bound, make up the difference in length from
           * the initial left side. */
          Box::new((0..(upper_bound - prologue_length)).flat_map(|_| self.group.clone()))
        } else {
          /* If not, we can go forever, or not at all, so we can just apply a Kleene
           * star to this! */
          let starred = KleeneStar {
            group: self.group.clone(),
          };
          let OperatorResult { result } = starred.operate(prods);
          Box::new(result.into_iter())
        };

      let result = prologue.chain(epilogue).collect();
      OperatorResult { result }
    }
  }

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct Optional {
    pub group: Vec<CaseEl>,
  }

  impl UnaryOperator for Optional {
    fn operate(&self, prods: &mut DetokenizedProductions) -> OperatorResult {
      let (new_prod_ref, &mut ProductionImpl(ref mut new_prod)) = prods.new_production();
      /* Add an empty case. */
      new_prod.push(CaseImpl(vec![]));
      /* Add a non-empty case. */
      new_prod.push(CaseImpl(self.group.clone()));
      OperatorResult {
        result: vec![CaseEl::Prod(new_prod_ref)],
      }
    }
  }
}
