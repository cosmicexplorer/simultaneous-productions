/*
 * Description: Implement the Simultaneous Productions general parsing
 * method.
 *
 * Copyright (C) 2019-2023 Danny McClanahan <dmcC2@hypnicjerk.ai>
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

//! Implement the Simultaneous Productions general parsing method.

/* These clippy lint descriptions are purely non-functional and do not affect the functionality
 * or correctness of the code.
 * TODO: rustfmt breaks multiline comments when used one on top of another! (each with its own
 * pair of delimiters)
 * Note: run clippy with: rustup run nightly cargo-clippy! */
#![warn(missing_docs)]
/* There should be no need to use unsafe code here! */
#![deny(unsafe_code)]
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

mod grammar_indexing;
mod interns;
mod lowering_to_indices;
mod parsing;
mod reconstruction;

/// The basic traits which define an input *grammar* (TODO: link to paper!).
///
/// *Implementation Note: While macros may be able to streamline the process of
/// declaring a grammar, their stability guarantees can be much lower than the
/// definitions in this module.*
pub mod grammar_specification {
  /// Aliases used in the grammar specification.
  pub mod types {
    use core::hash::Hash;

    /// Necessary requirement to hash an object, but not e.g. to
    /// lexicographically sort it.
    pub trait Hashable: Hash+Eq {}
  }

  /// Grammar components which expand into exactly one specific token.
  pub mod direct {
    use core::iter::IntoIterator;

    /// A contiguous sequence of tokens.
    pub trait Literal: IntoIterator {
      /// Specifies the type of "token" to iterate over when constructing a
      /// grammar.
      ///
      /// This parameter is *separate from, but may be the same as* the tokens
      /// we can actually parse with
      /// [Input::InChunk][super::execution::Input].
      type Tok: super::types::Hashable;
      /// Override [IntoIterator::Item] with this trait's parameter.
      type Item: Into<Self::Tok>;
    }
  }

  /// Grammar components which expand into the content of another production
  /// within the grammar.
  pub mod indirect {
    /// A type representing a [Production] that the grammar should satisfy at
    /// that position.
    pub trait ProductionReference: Into<Self::ID> {
      /// Parameterized type to reference the identity of some particular
      /// [Production].
      type ID: super::types::Hashable;
    }
  }

  pub mod context {
    pub trait ContextName: Into<Self::N> {
      type N: super::types::Hashable;
    }

    pub struct ContextDeclaration<Name: ContextName, PR: super::indirect::ProductionReference> {
      pub name: Name,
      pub prod_ref: PR,
    }
  }

  /// Grammar components which synthesize the lower-level elements from
  /// [direct], [indirect], [explicit], and [undecidable].
  pub mod synthesis {
    use super::{direct::Literal, indirect::ProductionReference};

    use core::iter::IntoIterator;

    use displaydoc::Display;

    /// Each individual element that can be matched against some input in a
    /// case.
    #[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum CaseElement<Lit, PR> {
      /// literal value {0}
      Lit(Lit),
      /// production reference {0}
      Prod(PR),
    }

    /// A sequence of *elements* which, if successfully matched against some
    /// *input*, represents some *production*.
    pub trait Case: IntoIterator {
      /// Literal tokens used. in this case.
      type Lit: Literal;
      /// References to productions used in this case.
      type PR: ProductionReference;
      /// Override of [Iterator::Item].
      type Item: Into<CaseElement<Self::Lit, Self::PR>>;
    }

    /// A disjunction of cases.
    pub trait Production: IntoIterator {
      /// Cases used in this production.
      type C: Case;
      /// Override of [Iterator::Item].
      type Item: Into<Self::C>;
    }

    /// A conjunction of productions (a grammar!).
    pub trait SimultaneousProductions: IntoIterator {
      /// Productions used in this grammar.
      type P: Production;
      /// Override of [Iterator::Item].
      type Item: Into<(<<Self::P as Production>::C as Case>::PR, Self::P)>;
    }
  }
}

/// The basic traits which define the *input*, *actions*, and *output* of a
/// parse.
///
/// The basic trait [`execution::Transformer`] allows constructing pipelines of
/// multiple separate monadic interfaces:
/// 1. **Iterators:** see [`execution::iterator_api`].
/// 2. **Generators:** see the `sp_generator_api` crate.
/// 3. **Streams:** see the `sp_stream_api` crate..
pub mod execution {
  /// A "stream-like" type.
  ///
  /// A "stream-like" type has a method that returns one instance of
  /// [Self::InChunk] at a time, possibly in a blocking fashion.
  pub trait Input {
    /// Type of object to iterate over.
    type InChunk;
  }

  /// Another stream-like type.
  pub trait Output {
    /// Type of object to iterate over.
    type OutChunk;
  }

  /// A stream-like type which transforms [Self::I] into [Self::O].
  ///
  /// See the node.js [transform stream API docs] as inspiration!
  ///
  /// [transform stream API docs]: https://nodejs.org/api/stream.html#stream_implementing_a_transform_stream
  pub trait Transformer {
    /// Input stream for this transformer to consume.
    type I: Input;
    /// Output stream for this transformer to produce.
    type O: Output;
    /// The return value of [Self::transform].
    ///
    /// This type is intentionally not constrained at all in order to conform to
    /// multiple monadic APIs in a [prototypal] way. *See [iterator_api].*
    ///
    /// [prototypal]: https://en.wikipedia.org/wiki/Prototype_pattern
    type R;
    /// Consume a single block of `input`, modify any internal state, and
    /// produce a result.
    fn transform(&mut self, input: <Self::I as Input>::InChunk) -> Self::R;
  }

  /// An [`Iterator`][core::iter::Iterator]-based API to a [`Transformer`].
  pub mod iterator_api {
    use super::*;

    /// A wrapper struct which consumes a transformer `ST` and an input iterable
    /// `I`.
    ///
    /// Implements [`Iterator`] such that [`Iterator::Item`] is equal to
    /// [`Transformer::O`] when `ST` implements [`Transformer`].
    #[derive(Debug, Default, Copy, Clone)]
    pub struct STIterator<ST, I> {
      state: ST,
      iter: I,
    }

    impl<ST, I> STIterator<ST, I> {
      /// Create a new instance from a [`Transformer`] `ST` and an [`Iterator`]
      /// `I`.
      pub fn new(state: ST, iter: I) -> Self { Self { state, iter } }
    }

    impl<ST, I> From<I> for STIterator<ST, I>
    where ST: Default
    {
      fn from(value: I) -> Self { Self::new(ST::default(), value) }
    }

    impl<ST, I, II, O, OO, R> Iterator for STIterator<ST, I>
    where
      I: Input<InChunk=II>+Iterator<Item=II>,
      O: Output<OutChunk=OO>+Iterator<Item=OO>,
      R: Into<Option<OO>>,
      ST: Transformer<I=I, O=O, R=R>,
    {
      type Item = OO;

      fn next(&mut self) -> Option<Self::Item> {
        self
          .iter
          .next()
          .and_then(|input| self.state.transform(input).into())
      }
    }
  }
}

/// The various phases that a grammar (in [preprocessing][state::preprocessing])
/// and then a parse (in [active][state::active]) goes through.
pub mod state {
  #[cfg(doc)]
  use crate::execution::Input;
  #[cfg(doc)]
  use active::{InProgress, Ready};
  #[cfg(doc)]
  use preprocessing::{Detokenized, Indexed, Init};

  /// Phases of interpreting an S.P. grammar into an executable specification.
  ///
  /// `[Init] -> [Detokenized] -> [Indexed] (-> [Ready])`
  pub mod preprocessing {
    use crate::{
      grammar_indexing as gi, grammar_specification as gs,
      lowering_to_indices::grammar_building as gb, parsing as p,
    };

    use core::{fmt, iter::IntoIterator};

    /// Container for an implementor of
    /// [gs::synthesis::SimultaneousProductions].
    #[derive(Debug, Copy, Clone)]
    pub struct Init<SP>(pub SP);

    impl<Tok, Lit, ID, PR, C, P, SP> Init<SP>
    where
      Tok: gs::types::Hashable,
      Lit: gs::direct::Literal<Tok=Tok>+IntoIterator<Item=Tok>,
      ID: gs::types::Hashable+Clone,
      PR: gs::indirect::ProductionReference<ID=ID>,
      C: gs::synthesis::Case<PR=PR>+IntoIterator<Item=gs::synthesis::CaseElement<Lit, PR>>,
      P: gs::synthesis::Production<C=C>+IntoIterator<Item=C>,
      SP: gs::synthesis::SimultaneousProductions<P=P>+IntoIterator<Item=(PR, P)>,
    {
      /// Create a [`gb::TokenGrammar`] and convert it to [`Detokenized`] for
      /// further preprocessing.
      pub fn try_index(self) -> Result<Detokenized<Tok>, gb::GrammarConstructionError<ID>> {
        Ok(Detokenized(gb::TokenGrammar::new(self.0)?))
      }
    }

    /// Container after converting the tokens into [gc::TokenPosition]s.
    #[derive(Debug, Clone)]
    pub struct Detokenized<Tok>(pub gb::TokenGrammar<Tok>);

    impl<Tok> Detokenized<Tok> {
      /// Create a [`gi::PreprocessedGrammar`] and convert it to [`Indexed`] for
      /// further preprocessing.
      pub fn index(self) -> Indexed<Tok> { Indexed(gi::PreprocessedGrammar::new(self.0)) }
    }

    /// Container for an immediately executable grammar.
    #[derive(Debug, Clone)]
    pub struct Indexed<Tok>(pub gi::PreprocessedGrammar<Tok>);

    impl<Tok> Indexed<Tok>
    where Tok: gs::types::Hashable+fmt::Debug+Clone
    {
      /// Create a [`p::ParseableGrammar`] and convert to a parseable state.
      ///
      /// **FIXME: `input` should be a [crate::execution::Input]!!**
      pub fn attach_input(
        &self,
        input: &p::Input<Tok>,
      ) -> Result<super::active::Ready<'_>, p::ParsingInputFailure<Tok>> {
        Ok(super::active::Ready::new(p::ParseableGrammar::new(
          self.0.clone(),
          input,
        )?))
      }
    }
  }

  /// Phases of receiving an [Input] and parsing something useful out of it.
  ///
  /// `([Indexed] ->) [Ready] -> [InProgress]`
  pub mod active {
    use crate::parsing as p;

    use core::marker::PhantomData;

    /// Container for a parseable grammar that propagates the lifetime of an
    /// input.
    #[derive(Debug, Clone)]
    pub struct Ready<'a>(pub p::ParseableGrammar, PhantomData<&'a u8>);

    impl<'a> Ready<'a> {
      #[allow(missing_docs)]
      pub fn new(grammar: p::ParseableGrammar) -> Self { Self(grammar, PhantomData) }

      /// "Detokenize" *(TODO: cite!)* the input and produce a [`p::Parse`]
      /// instance!
      pub fn initialize_parse(self) -> InProgress<'a> {
        InProgress::new(p::Parse::initialize_with_trees_for_adjacent_pairs(self.0))
      }
    }

    /// The final form of an initialized parse, ready to iterate over the input!
    #[derive(Debug, Clone)]
    pub struct InProgress<'a>(pub p::Parse, PhantomData<&'a u8>);

    impl<'a> InProgress<'a> {
      #[allow(missing_docs)]
      pub fn new(parse: p::Parse) -> Self { Self(parse, PhantomData) }
    }
  }
}

/// Helper methods to improve the ergonomics of testing in a [`no_std`]
/// environment.
///
/// [`no_std`]: https://docs.rust-embedded.org/book/intro/no-std.html
#[cfg(test)]
pub mod test_framework {
  use super::grammar_specification as gs;
  use crate::lowering_to_indices::graph_coordinates as gc;

  use core::{
    hash::{Hash, Hasher},
    iter::{IntoIterator, Iterator},
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

  /// Declare a type backed by [Vec::IntoIter] which forwards trait
  /// implementations to a newly constructed vector type.
  ///
  /// This allows us to implement [Iterator] without having to create a name for
  /// an intermediate `IntoIter` type.
  macro_rules! into_iter {
    ($type_name:ident, $item:ty) => {
      #[derive(Debug, Clone)]
      pub struct $type_name(<Vec<$item> as IntoIterator>::IntoIter);

      impl $type_name {
        fn as_new_vec(&self) -> Vec<$item> { self.0.clone().collect() }
      }

      impl From<&[$item]> for $type_name {
        fn from(value: &[$item]) -> Self {
          Self(value.iter().cloned().collect::<Vec<_>>().into_iter())
        }
      }

      impl Iterator for $type_name {
        type Item = $item;

        fn next(&mut self) -> Option<Self::Item> { self.0.next() }
      }

      impl Hash for $type_name {
        fn hash<H: Hasher>(&self, state: &mut H) { self.as_new_vec().hash(state); }
      }

      impl PartialEq for $type_name {
        fn eq(&self, other: &Self) -> bool { self.as_new_vec() == other.as_new_vec() }
      }

      impl Eq for $type_name {}

      impl Default for $type_name {
        fn default() -> Self { Self(Vec::new().into_iter()) }
      }
    };
  }

  /// A specialization of [into_iter] for strings.
  macro_rules! string_iter {
    ($type_name:ident) => {
      into_iter![$type_name, char];

      impl From<&str> for $type_name {
        fn from(value: &str) -> Self {
          let chars = value.chars().collect::<Vec<_>>();
          Self::from(&chars[..])
        }
      }
    };
  }

  string_iter![Lit];

  impl gs::types::Hashable for char {}

  impl gs::direct::Literal for Lit {
    type Item = char;
    type Tok = char;
  }

  string_iter![ProductionReference];

  impl gs::types::Hashable for ProductionReference {}

  impl gs::indirect::ProductionReference for ProductionReference {
    type ID = Self;
  }

  pub type CE = gs::synthesis::CaseElement<Lit, ProductionReference>;

  into_iter![Case, CE];

  impl gs::synthesis::Case for Case {
    type Item = CE;
    type Lit = Lit;
    type PR = ProductionReference;
  }

  into_iter![Production, Case];

  impl gs::synthesis::Production for Production {
    type C = Case;
    type Item = Case;
  }

  into_iter![SP, (ProductionReference, Production)];

  impl gs::synthesis::SimultaneousProductions for SP {
    type Item = (ProductionReference, Self::P);
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
