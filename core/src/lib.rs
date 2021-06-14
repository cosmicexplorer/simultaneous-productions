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
#![feature(associated_type_defaults)]
#![feature(generators, generator_trait)]
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
pub(crate) mod parsing;
pub(crate) mod reconstruction;

pub(crate) mod types {
  pub use indexmap::{vec::Vec, Global};
  use twox_hash::XxHash64;

  use core::hash::BuildHasherDefault;

  pub type DefaultHasher = BuildHasherDefault<XxHash64>;
}

/// Definition of the trait used to parameterize an atomic input component.
pub mod token {
  use core::{
    fmt::{Debug, Display},
    hash::Hash,
  };

  /// The constraints required for any token stream parsed by this crate.
  pub trait Token = Debug+Display+PartialEq+Eq+Hash+Copy+Clone;
}

pub mod allocation {
  use core::alloc::Allocator;

  pub trait HandoffAllocable {
    type Arena: Allocator;
    fn allocator_handoff(&self) -> Self::Arena;
  }
}

/// The basic traits which define an input *grammar* (TODO: link to paper!).
///
/// *Implementation Note: While macros may be able to streamline the process of
/// declaring a grammar, their stability guarantees can be much lower than the
/// definitions in this module.*
pub mod grammar_specification {
  #[cfg(doc)]
  use super::input_stream::Input;

  use core::iter::IntoIterator;

  /// A contiguous sequence of tokens.
  pub trait Literal: IntoIterator {
    /// Specifies the type of "token" to iterate over when constructing a
    /// grammar.
    ///
    /// This parameter is *separate from, but may be the same as* the tokens we
    /// can actually parse with in [Input::Tok].
    type Tok;
    /// Override [IntoIterator::Item] with this trait's parameter.
    ///
    /// *Implementation Note: We could just leave this trait empty, but that
    /// would make it unclear there is an `Item` type that needs to be
    /// set elsewhere.*
    type Item = Self::Tok;
  }

  pub trait ProductionReference: Into<Self::ID> {
    type ID;
  }

  /// Each individual element that can be matched against some input in a case.
  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub enum CaseElement<Lit, PR> {
    Lit(Lit),
    Prod(PR),
  }

  /// A sequence of *elements* which, if successfully matched against some
  /// *input*, represents some *production*.
  pub trait Case: IntoIterator {
    type Lit: Literal;
    type PR: ProductionReference;
    type Item = CaseElement<Self::Lit, Self::PR>;
  }

  /// A disjunction of cases.
  pub trait Production: IntoIterator {
    type C: Case;
    type Item = Self::C;
  }

  /// A conjunction of productions.
  pub trait SimultaneousProductions: IntoIterator {
    type P: Production;
    type Item = (
      <<<Self as SimultaneousProductions>::P as Production>::C as Case>::PR,
      Self::P,
    );
  }
}

pub(crate) mod impls {
  use super::grammar_specification as gs;

  use core::fmt;

  impl<Lit, PR> gs::CaseElement<Lit, PR>
  where
    Lit: fmt::Display,
    PR: fmt::Display,
  {
    fn descriptor(&self, f: &mut fmt::Formatter) -> fmt::Result {
      match self {
        Self::Lit(lit) => write!(f, "literal: {}", lit),
        Self::Prod(pr) => write!(f, "production reference: {}", pr),
      }
    }
  }

  impl<Lit, PR> fmt::Display for gs::CaseElement<Lit, PR>
  where
    Lit: fmt::Display,
    PR: fmt::Display,
  {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
      /* FIXME: writing nested expressions like parentheses should be factored out
       * elsewhere! */
      write!(f, "(CaseElement: ")
        .and_then(|()| self.descriptor(f))
        .and_then(|()| write!(f, ")"))
    }
  }
}

/// The basic traits which define the *input*, *actions*, and *output* of a
/// parse.
pub mod execution {
  use core::iter::{IntoIterator, Iterator};

  pub trait Input: Iterator {
    type Tok;
    type Item = Self::Tok;
  }

  pub trait Output: IntoIterator {
    type Out;
    type Item = Self::Out;
  }
}

/// Helper methods to improve the ergonomics of testing in a [`no_std`]
/// environment.
///
/// [`no_std`]: https://docs.rust-embedded.org/book/intro/no-std.html
#[cfg(test)]
pub mod test_framework {
  use super::grammar_specification as gs;
  use crate::{lowering_to_indices::graph_coordinates as gc, types::Vec};

  use core::{
    fmt,
    hash::{Hash, Hasher},
    iter::IntoIterator,
    str,
  };

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

  impl gs::Literal for Lit {
    type Tok = char;
  }

  impl Hash for Lit {
    fn hash<H: Hasher>(&self, state: &mut H) { self.0.hash(state); }
  }

  impl PartialEq for Lit {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
  }

  impl Eq for Lit {}

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

  impl gs::ProductionReference for ProductionReference {
    type ID = Self;
  }

  pub type CE = gs::CaseElement<Lit, ProductionReference>;

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

  impl gs::Case for Case {
    type Lit = Lit;
    type PR = ProductionReference;
  }

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

  impl gs::Production for Production {
    type C = Case;
  }

  pub struct SP(Vec<(ProductionReference, Production)>);

  impl From<&[(ProductionReference, Production)]> for SP {
    fn from(value: &[(ProductionReference, Production)]) -> Self { Self(value.to_vec()) }
  }

  impl IntoIterator for SP {
    type IntoIter = <Vec<(ProductionReference, Production)> as IntoIterator>::IntoIter;
    type Item = (ProductionReference, Production);

    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
  }

  impl gs::SimultaneousProductions for SP {
    /* type Item = Box<(ProductionReference, Self::P), Global>; */
    type P = Production;
  }

  pub fn non_cyclic_productions() -> SP {
    SP::from(
      [
        (
          ProductionReference::from("a"),
          Production::from([Case::from([CE::Lit(Lit::from("ab"))].as_ref())].as_ref()),
        ),
        (
          ProductionReference::from("b"),
          Production::from(
            [
              Case::from(
                [
                  CE::Lit(Lit::from("ab")),
                  CE::Prod(ProductionReference::from("a")),
                ]
                .as_ref(),
              ),
              Case::from(
                [
                  CE::Prod(ProductionReference::from("a")),
                  CE::Lit(Lit::from("a")),
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
              Case::from([CE::Lit(Lit::from("abc"))].as_ref()),
              Case::from(
                [
                  CE::Lit(Lit::from("a")),
                  CE::Prod(ProductionReference::from("P_1")),
                  CE::Lit(Lit::from("c")),
                ]
                .as_ref(),
              ),
              Case::from(
                [
                  CE::Lit(Lit::from("bc")),
                  CE::Prod(ProductionReference::from("P_2")),
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
              Case::from([CE::Prod(ProductionReference::from("P_1"))].as_ref()),
              Case::from([CE::Prod(ProductionReference::from("P_2"))].as_ref()),
              Case::from(
                [
                  CE::Prod(ProductionReference::from("P_1")),
                  CE::Lit(Lit::from("bc")),
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
