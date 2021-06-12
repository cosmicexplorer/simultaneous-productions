/*
 * Description: ???
 *
 * Copyright (C) 2021 Danny McClanahan <dmcC2@hypnicjerk.ai>
 * SPDX-License-Identifier: GPL-3.0
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#![no_std]
#![warn(stable_features)]
#![feature(trait_alias)]
#![feature(generic_associated_types)]
#![feature(allocator_api)]
#![feature(alloc)]
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

pub(crate) mod grammar_indexing;
pub(crate) mod interns;
pub(crate) mod lowering_to_indices;

pub use indexmap::vec;

/// Definition of the trait used to parameterize an atomic input component.
pub mod token {
  use core::{
    fmt::{Debug, Display},
    hash::Hash,
  };

  /// The constraints required for any token stream parsed by this crate.
  pub trait Token = Debug+Display+PartialEq+Eq+Hash+Copy+Clone;
}

/// The basic structs which define an input grammar.
///
/// While macros may be able to streamline the process of declaring a grammar,
/// their stability guarantees can be much lower than the definitions in this
/// module.
pub mod grammar_specification {
  use core::{convert::Into, iter::IntoIterator};

  /// A contiguous sequence of tokens.
  pub trait Literal<Tok>: IntoIterator<Item=Tok> {}

  pub trait ProductionReference<ID>: Into<ID> {}

  /// Each individual element that can be matched against some input in a case.
  pub enum CaseElement<ID, Tok, Lit, PR>
  where
    Lit: Literal<Tok>,
    PR: ProductionReference<ID>,
  {
    Lit(Lit),
    Prod(PR),
  }

  /// A sequence of *elements* which, if successfully matched against some
  /// *input*, represents some *production*.
  pub trait Case<ID, Tok, Lit, PR>: IntoIterator<Item=CaseElement<ID, Tok, Lit, PR>>
  where
    Lit: Literal<Tok>,
    PR: ProductionReference<ID>,
  {
  }

  /// A disjunction of cases.
  pub trait Production<ID, Tok, Lit, PR, C>: IntoIterator<Item=C>
  where
    Lit: Literal<Tok>,
    PR: ProductionReference<ID>,
    C: Case<ID, Tok, Lit, PR>,
  {
  }

  /// A conjunction of productions.
  pub trait SimultaneousProductions<ID, Tok, Lit, PR, C, P>: IntoIterator<Item=(PR, P)>
  where
    Lit: Literal<Tok>,
    PR: ProductionReference<ID>,
    C: Case<ID, Tok, Lit, PR>,
    P: Production<ID, Tok, Lit, PR, C>,
  {
  }
}

#[cfg(test)]
pub mod test_framework {
  use super::grammar_specification::*;
  use crate::lowering_to_indices::graph_coordinates::*;

  pub fn new_token_position(prod_ind: usize, case_ind: usize, case_el_ind: usize) -> TokenPosition {
    TokenPosition {
      prod: ProdRef::new(prod_ind),
      case: CaseRef::new(case_ind),
      case_el: CaseElRef::new(case_el_ind),
    }
  }

  pub fn non_cyclic_productions() -> SimultaneousProductions<char> {
    SimultaneousProductions(
      [
        (
          ProductionReference::new("a"),
          Production(vec![Case(vec![CaseElement::Lit(Literal::from("ab"))])]),
        ),
        (
          ProductionReference::new("b"),
          Production(vec![
            Case(vec![
              CaseElement::Lit(Literal::from("ab")),
              CaseElement::Prod(ProductionReference::new("a")),
            ]),
            Case(vec![
              CaseElement::Prod(ProductionReference::new("a")),
              CaseElement::Lit(Literal::from("a")),
            ]),
          ]),
        ),
      ]
      .iter()
      .cloned()
      .collect(),
    )
  }

  pub fn basic_productions() -> SimultaneousProductions<char> {
    SimultaneousProductions(
      [
        (
          ProductionReference::new("P_1"),
          Production(vec![
            Case(vec![CaseElement::Lit(Literal::from("abc"))]),
            Case(vec![
              CaseElement::Lit(Literal::from("a")),
              CaseElement::Prod(ProductionReference::new("P_1")),
              CaseElement::Lit(Literal::from("c")),
            ]),
            Case(vec![
              CaseElement::Lit(Literal::from("bc")),
              CaseElement::Prod(ProductionReference::new("P_2")),
            ]),
          ]),
        ),
        (
          ProductionReference::new("P_2"),
          Production(vec![
            Case(vec![CaseElement::Prod(ProductionReference::new("P_1"))]),
            Case(vec![CaseElement::Prod(ProductionReference::new("P_2"))]),
            Case(vec![
              CaseElement::Prod(ProductionReference::new("P_1")),
              CaseElement::Lit(Literal::from("bc")),
            ]),
          ]),
        ),
      ]
      .iter()
      .cloned()
      .collect(),
    )
  }
}
