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
#![feature(trait_alias)]
#![feature(allocator_api)]
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

pub use indexmap::{collections, vec};

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
  use core::{convert::Into, iter::IntoIterator, marker::PhantomData};

  /// A contiguous sequence of tokens.
  pub trait Literal<Tok>: IntoIterator<Item=Tok> {}

  pub trait ProductionReference<ID>: Into<ID> {}

  /// Each individual element that can be matched against some input in a case.
  #[derive(Clone)]
  pub enum CaseElement<ID, Tok, Lit, PR>
  where
    Lit: Literal<Tok>,
    PR: ProductionReference<ID>,
  {
    Lit(Lit, PhantomData<Tok>),
    Prod(PR, PhantomData<ID>),
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
  use super::grammar_specification as gs;
  use crate::{lowering_to_indices::graph_coordinates as gc, vec::Vec};

  use core::{fmt, iter::IntoIterator, marker::PhantomData, str};

  pub fn new_token_position(
    prod_ind: usize,
    case_ind: usize,
    case_el_ind: usize,
  ) -> gc::TokenPosition {
    gc::TokenPosition {
      prod: gc::ProdRef(prod_ind),
      case: gc::CaseRef(case_ind),
      el: gc::CaseElRef(case_el_ind),
    }
  }

  #[derive(Clone)]
  pub struct Lit(Vec<u8>);

  impl From<&str> for Lit {
    fn from(value: &str) -> Self { Self(value.as_bytes().to_vec()) }
  }

  impl fmt::Debug for Lit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
      write!(f, "Lit({:?})", str::from_utf8(&self.0).unwrap())
    }
  }

  impl IntoIterator for Lit {
    type IntoIter = <Vec<char> as IntoIterator>::IntoIter;
    type Item = char;

    fn into_iter(self) -> Self::IntoIter {
      let char_vec: Vec<char> = str::from_utf8(&self.0).unwrap().chars().collect();
      char_vec.into_iter()
    }
  }

  impl gs::Literal<char> for Lit {}

  #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub struct ProductionReference(Vec<u8>);

  impl From<&str> for ProductionReference {
    fn from(value: &str) -> Self { Self(value.as_bytes().to_vec()) }
  }

  impl fmt::Debug for ProductionReference {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
      write!(
        f,
        "ProductionReference({:?})",
        str::from_utf8(&self.0).unwrap()
      )
    }
  }

  impl gs::ProductionReference<ProductionReference> for ProductionReference {}

  pub type CE = gs::CaseElement<ProductionReference, char, Lit, ProductionReference>;

  #[derive(Clone)]
  pub struct Case(Vec<CE>);

  impl From<&[CE]> for Case {
    fn from(value: &[CE]) -> Self { Self(value.to_vec()) }
  }

  impl IntoIterator for Case {
    type IntoIter = <Vec<CE> as IntoIterator>::IntoIter;
    type Item = CE;

    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
  }

  impl gs::Case<ProductionReference, char, Lit, ProductionReference> for Case {}

  #[derive(Clone)]
  pub struct Production(Vec<Case>);

  impl From<&[Case]> for Production {
    fn from(value: &[Case]) -> Self { Self(value.to_vec()) }
  }

  impl IntoIterator for Production {
    type IntoIter = <Vec<Case> as IntoIterator>::IntoIter;
    type Item = Case;

    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
  }

  impl gs::Production<ProductionReference, char, Lit, ProductionReference, Case> for Production {}

  pub struct SP(Vec<(ProductionReference, Production)>);

  impl From<&[(ProductionReference, Production)]> for SP {
    fn from(value: &[(ProductionReference, Production)]) -> Self { Self(value.to_vec()) }
  }

  impl IntoIterator for SP {
    type IntoIter = <Vec<(ProductionReference, Production)> as IntoIterator>::IntoIter;
    type Item = (ProductionReference, Production);

    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
  }

  impl
    gs::SimultaneousProductions<
      ProductionReference,
      char,
      Lit,
      ProductionReference,
      Case,
      Production,
    > for SP
  {
  }

  pub fn non_cyclic_productions() -> SP {
    SP::from(
      [
        (
          ProductionReference::from("a"),
          Production::from([Case::from([CE::Lit(Lit::from("ab"), PhantomData)].as_ref())].as_ref()),
        ),
        (
          ProductionReference::from("b"),
          Production::from(
            [
              Case::from(
                [
                  CE::Lit(Lit::from("ab"), PhantomData),
                  CE::Prod(ProductionReference::from("a"), PhantomData),
                ]
                .as_ref(),
              ),
              Case::from(
                [
                  CE::Prod(ProductionReference::from("a"), PhantomData),
                  CE::Lit(Lit::from("a"), PhantomData),
                ]
                .as_ref(),
              ),
            ]
            .as_ref(),
          ),
        ),
      ]
      .as_ref(),
    )
  }

  pub fn basic_productions() -> SP {
    SP::from(
      [
        (
          ProductionReference::from("P_1"),
          Production::from(
            [
              Case::from([CE::Lit(Lit::from("abc"), PhantomData)].as_ref()),
              Case::from(
                [
                  CE::Lit(Lit::from("a"), PhantomData),
                  CE::Prod(ProductionReference::from("P_1"), PhantomData),
                  CE::Lit(Lit::from("c"), PhantomData),
                ]
                .as_ref(),
              ),
              Case::from(
                [
                  CE::Lit(Lit::from("bc"), PhantomData),
                  CE::Prod(ProductionReference::from("P_2"), PhantomData),
                ]
                .as_ref(),
              ),
            ]
            .as_ref(),
          ),
        ),
        (
          ProductionReference::from("P_2"),
          Production::from(
            [
              Case::from([CE::Prod(ProductionReference::from("P_1"), PhantomData)].as_ref()),
              Case::from([CE::Prod(ProductionReference::from("P_2"), PhantomData)].as_ref()),
              Case::from(
                [
                  CE::Prod(ProductionReference::from("P_1"), PhantomData),
                  CE::Lit(Lit::from("bc"), PhantomData),
                ]
                .as_ref(),
              ),
            ]
            .as_ref(),
          ),
        ),
      ]
      .as_ref(),
    )
  }
}
