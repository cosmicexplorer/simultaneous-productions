/*
 * Description: Implement the Simultaneous Productions general parsing
 * method.
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

//! Implement the Simultaneous Productions general parsing method.
//!
//! ### Unstable Features
//! Currently, this crate does not support being built on stable rust, due to
//! the use of the following unstable features:
//! 1. [ ] `#![feature(allocator_api)]`: see [`allocation`].
//! 2. [ ] `#![feature(generators, generator_trait)]`: see
//!    [`execution::generator_api`].
//! 3. [ ] `#![feature(async_stream)]`: see [`execution::stream_api`].

#![no_std]
#![allow(incomplete_features)]
#![feature(allocator_api)]
#![feature(generators, generator_trait)]
#![feature(async_stream)]
/* These clippy lint descriptions are purely non-functional and do not affect the functionality
 * or correctness of the code.
 * TODO: rustfmt breaks multiline comments when used one on top of another! (each with its own
 * pair of delimiters)
 * Note: run clippy with: rustup run nightly cargo-clippy! */
#![warn(missing_docs)]
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

mod types {
  pub use indexmap::alloc_inner::{Allocator, Global, Vec};
  use twox_hash::XxHash64;

  use core::hash::BuildHasherDefault;

  pub type DefaultHasher = BuildHasherDefault<XxHash64>;
}

/// Bridge to the unstable [`allocator_api` module].
///
/// ### TODO
/// Currently, this crate does not support being built on stable rust, due to
/// the use of the unstable `#![feature(allocator_api)]`.
/// - [ ] Use the (old, but maybe ok?) [`allocator_api` crate] to enable this?
///
/// [`allocator_api` module]: https://doc.rust-lang.org/unstable-book/library-features/allocator-api.html
/// [`allocator_api` crate]: https://docs.rs/allocator_api/0.6.0/allocator_api/
pub mod allocation {
  use super::types::Allocator;

  /// Allows an [Allocator]-based collection to generate a reference to its
  /// allocator.
  ///
  /// Implementing this trait allows a [prototypal] mechanism of copying
  /// references to the allocator used for some some originally-allocated
  /// collection. It is used in this crate to re-use a parameterized allocator
  /// for the whole of the `preprocessing` regime **(TODO: cite!)**.
  ///
  /// [prototypal]: https://en.wikipedia.org/wiki/Prototype_pattern
  pub trait HandoffAllocable {
    /// The type of allocator to use, for both the original collection and the
    /// one being "handed off" to.
    type Arena: Allocator;
    /// Produce a new instance of an [Allocator].
    ///
    /// **Note:** Implementing this trait typically requires using the
    /// `impl<..., Arena> ... where Arena: Allocator+[Clone]` trait bound in
    /// order to produce a new instance from a reference `&self`.
    fn allocator_handoff(&self) -> Self::Arena;
  }
}

/// The basic traits which define an input *grammar* (TODO: link to paper!).
///
/// *Implementation Note: While macros may be able to streamline the process of
/// declaring a grammar, their stability guarantees can be much lower than the
/// definitions in this module.*
pub mod grammar_specification {
  use displaydoc::Display;

  use core::iter::IntoIterator;

  /// A contiguous sequence of tokens.
  pub trait Literal: IntoIterator {
    /// Specifies the type of "token" to iterate over when constructing a
    /// grammar.
    ///
    /// This parameter is *separate from, but may be the same as* the tokens we
    /// can actually parse with [Input::InChunk][super::execution::Input].
    type Tok;
    /// Override [IntoIterator::Item] with this trait's parameter.
    ///
    /// *Implementation Note: We could just leave this trait empty, but that
    /// would make it unclear there is an `Item` type that needs to be
    /// set elsewhere.*
    type Item: Into<Self::Tok>;
  }

  /// A type representing a [Production] that the grammar should satisfy at that
  /// position.
  pub trait ProductionReference: Into<Self::ID> {
    /// Parameterized type to reference the identity of some particular
    /// [Production].
    type ID;
  }

  /// Each individual element that can be matched against some input in a case.
  #[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub enum CaseElement<Lit, PR> {
    /// literal value {0}
    Lit(Lit),
    /// production reference {0}
    Prod(PR),
  }

  /// A sequence of *elements* which, if successfully matched against some
  /// *input*, represents some *production*.
  pub trait Case: Iterator {
    /// Literal tokens used. in this case.
    type Lit: Literal;
    /// References to productions used in this case.
    type PR: ProductionReference;
    /// Override of [Iterator::Item].
    type Item: Into<CaseElement<Self::Lit, Self::PR>>;
  }

  /// A disjunction of cases.
  pub trait Production: Iterator {
    /// Cases used in this production.
    type C: Case;
    /// Override of [Iterator::Item].
    type Item: Into<Self::C>;
  }

  /// A conjunction of productions (a grammar!).
  pub trait SimultaneousProductions: Iterator {
    /// Productions used in this grammar.
    type P: Production;
    /// Override of [Iterator::Item].
    type Item: Into<(<<Self::P as Production>::C as Case>::PR, Self::P)>;
  }
}

/// The basic traits which define the *input*, *actions*, and *output* of a
/// parse.
///
/// The basic trait [`execution::Transformer`] allows constructing pipelines of
/// multiple separate monadic interfaces:
/// 1. **Iterators:** see [`execution::iterator_api`].
/// 2. **Generators:** see [`execution::generator_api`] (requires the
///    `"generator-api"` crate feature).
/// 3. **Streams:** see [`execution::stream_api`] (requires the `"stream-api"`
///    crate feature).
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

  /// A [`Generator`][core::ops::Generator]-based API to a [`Transformer`].
  ///
  /// Requires the crate feature `"generator-api"`.
  #[cfg(feature = "generator-api")]
  pub mod generator_api {
    use super::*;

    use core::{
      marker::{PhantomData, Unpin},
      ops::{Generator, GeneratorState},
      pin::Pin,
    };

    /// A wrapper struct which consumes a transformer `ST`.
    ///
    /// Implements [`Generator`] such that [`Generator::Yield`] is equal to
    /// [`Transformer::O`] when `ST` implements [`Transformer`].
    ///
    /// *Note that the generator api here requires injecting no external state,
    /// unlike for [`iterator_api`][super::iterator_api] and
    /// [`stream_api`][super::stream_api]!*
    #[derive(Debug, Default, Copy, Clone)]
    pub struct STGenerator<ST, Ret>(ST, PhantomData<Ret>);

    impl<ST, Ret> From<ST> for STGenerator<ST, Ret> {
      fn from(value: ST) -> Self { Self(value, PhantomData) }
    }

    impl<ST, R, Ret> Generator<<ST::I as Input>::InChunk> for STGenerator<ST, Ret>
    where
      Ret: Unpin,
      R: Into<GeneratorState<<ST::O as Output>::OutChunk, Ret>>,
      ST: Transformer<R=R>+Unpin,
    {
      type Return = Ret;
      type Yield = <ST::O as Output>::OutChunk;

      fn resume(
        self: Pin<&mut Self>,
        arg: <ST::I as Input>::InChunk,
      ) -> GeneratorState<Self::Yield, Self::Return> {
        let Self(state, _) = self.get_mut();
        state.transform(arg).into()
      }
    }
  }

  /// A [`Stream`][core::stream::Stream]-based API to a [`Transformer`].
  ///
  /// Requires the crate feature `"stream-api"`.
  #[cfg(feature = "stream-api")]
  pub mod stream_api {
    use super::*;

    use core::{
      marker::Unpin,
      pin::Pin,
      stream::Stream,
      task::{Context, Poll},
    };

    /// A wrapper struct which consumes a transformer `ST` and a stream `I`.
    ///
    /// Implements [`Stream`] such that [`Stream::Item`] is equal to
    /// [`Transformer::O`] when `ST` implements [`Transformer`].
    #[derive(Debug, Default, Copy, Clone)]
    pub struct STStream<ST, I> {
      state: ST,
      stream: I,
    }

    impl<ST, I> STStream<ST, I> {
      /// Create a new instance given a [Transformer] instance and an input
      /// stream.
      pub fn new(state: ST, stream: I) -> Self { Self { state, stream } }
    }

    impl<ST, I> From<I> for STStream<ST, I>
    where ST: Default
    {
      fn from(value: I) -> Self {
        Self {
          state: ST::default(),
          stream: value,
        }
      }
    }

    impl<ST, I, II, O, OO, R> Stream for STStream<ST, I>
    where
      I: Input<InChunk=II>+Stream<Item=II>+Unpin,
      O: Output<OutChunk=OO>+Stream<Item=OO>,
      R: Into<Poll<Option<OO>>>,
      ST: Transformer<I=I, O=O, R=R>+Unpin,
    {
      type Item = OO;

      fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let Self { state, stream } = self.get_mut();
        match Pin::new(stream).poll_next(cx) {
          Poll::Pending => Poll::Pending,
          Poll::Ready(None) => Poll::Ready(None),
          Poll::Ready(Some(x)) => state.transform(x).into(),
        }
      }
    }
  }
}

/// The various phases that a grammar and/or a parse goes through.
mod state {
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

    use core::{alloc::Allocator, fmt, hash::Hash, iter::IntoIterator};

    #[derive(Debug, Copy, Clone)]
    pub struct Init<SP>(pub SP);

    impl<Tok, Lit, ID, PR, C, P, SP> Init<SP>
    where
      Tok: Hash+Eq,
      Lit: gs::Literal<Tok=Tok>+IntoIterator<Item=Tok>,
      ID: Hash+Eq+Clone,
      PR: gs::ProductionReference<ID=ID>,
      C: gs::Case<PR=PR>+IntoIterator<Item=gs::CaseElement<Lit, PR>>,
      P: gs::Production<C=C>+IntoIterator<Item=C>,
      SP: gs::SimultaneousProductions<P=P>+IntoIterator<Item=(PR, P)>,
    {
      /// Create a [`gb::TokenGrammar`] and convert it to [`Detokenized`] for
      /// further preprocessing.
      #[allow(dead_code)]
      pub fn try_index_with_allocator<Arena>(
        self,
        arena: Arena,
      ) -> Result<Detokenized<Tok, Arena>, gb::GrammarConstructionError<ID>>
      where
        Arena: Allocator+Clone,
      {
        Ok(Detokenized(gb::TokenGrammar::new(self.0, arena)?))
      }
    }

    #[derive(Debug, Clone)]
    pub struct Detokenized<Tok, Arena>(pub gb::TokenGrammar<Tok, Arena>)
    where Arena: Allocator+Clone;

    impl<Tok, Arena> Detokenized<Tok, Arena>
    where Arena: Allocator+Clone
    {
      /// Create a [`gi::PreprocessedGrammar`] and convert it to [`Indexed`] for
      /// further preprocessing.
      #[allow(dead_code)]
      pub fn index(self) -> Indexed<Tok, Arena> { Indexed(gi::PreprocessedGrammar::new(self.0)) }
    }

    #[derive(Debug, Clone)]
    pub struct Indexed<Tok, Arena>(pub gi::PreprocessedGrammar<Tok, Arena>)
    where Arena: Allocator+Clone;

    impl<Tok, Arena> Indexed<Tok, Arena>
    where
      Tok: Hash+Eq+fmt::Debug+Clone,
      Arena: Allocator+Clone,
    {
      /// Create a [`p::ParseableGrammar`] and convert to a parseable state.
      ///
      /// **FIXME: `input` should be a [crate::execution::Input]!!**
      #[allow(dead_code)]
      pub fn attach_input(
        &self,
        input: &p::Input<Tok, Arena>,
      ) -> Result<super::active::Ready<'_, Arena>, p::ParsingInputFailure<Tok>> {
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

    use core::{alloc::Allocator, marker::PhantomData};

    #[derive(Debug, Clone)]
    pub struct Ready<'a, Arena>(pub p::ParseableGrammar<Arena>, PhantomData<&'a u8>)
    where Arena: Allocator+Clone;

    impl<'a, Arena> Ready<'a, Arena>
    where Arena: Allocator+Clone
    {
      pub fn new(grammar: p::ParseableGrammar<Arena>) -> Self { Self(grammar, PhantomData) }

      /// "Detokenize" *(TODO: cite!)* the input and produce a [`p::Parse`]
      /// instance!
      #[allow(dead_code)]
      pub fn initialize_parse(self) -> InProgress<'a, Arena> {
        InProgress::new(p::Parse::initialize_with_trees_for_adjacent_pairs(self.0))
      }
    }

    /// The final form of an initialized parse, ready to iterate over the input!
    #[derive(Debug, Clone)]
    pub struct InProgress<'a, Arena>(pub p::Parse<Arena>, PhantomData<&'a u8>)
    where Arena: Allocator+Clone;

    impl<'a, Arena> InProgress<'a, Arena>
    where Arena: Allocator+Clone
    {
      pub fn new(parse: p::Parse<Arena>) -> Self { Self(parse, PhantomData) }
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
  use crate::{lowering_to_indices::graph_coordinates as gc, types::Vec};

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

  /* #[derive(Delegate)] */
  #[derive(Debug, Clone)]
  pub struct Lit(
    /* #[trait(Iterator)] */
    <Vec<char> as IntoIterator>::IntoIter,
  );

  /* derive_from_method![Lit::as_new_vec, Hash, PartialEq, Eq] */
  impl Lit {
    fn as_new_vec(&self) -> Vec<char> { self.0.clone().collect() }
  }

  impl From<&str> for Lit {
    fn from(value: &str) -> Self { Self(value.chars().collect::<Vec<_>>().into_iter()) }
  }

  impl Iterator for Lit {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> { self.0.next() }
  }

  impl gs::Literal for Lit {
    type Item = char;
    type Tok = char;
  }

  impl Hash for Lit {
    fn hash<H: Hasher>(&self, state: &mut H) { self.as_new_vec().hash(state); }
  }

  impl PartialEq for Lit {
    fn eq(&self, other: &Self) -> bool { self.as_new_vec() == other.as_new_vec() }
  }

  impl Eq for Lit {}

  #[derive(Debug, Clone)]
  pub struct ProductionReference(<Vec<char> as IntoIterator>::IntoIter);

  impl ProductionReference {
    fn as_new_vec(&self) -> Vec<char> { self.0.clone().collect() }
  }

  impl From<&str> for ProductionReference {
    fn from(value: &str) -> Self { Self(value.chars().collect::<Vec<_>>().into_iter()) }
  }

  impl Iterator for ProductionReference {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> { self.0.next() }
  }

  impl Hash for ProductionReference {
    fn hash<H: Hasher>(&self, state: &mut H) { self.as_new_vec().hash(state); }
  }

  impl PartialEq for ProductionReference {
    fn eq(&self, other: &Self) -> bool { self.as_new_vec() == other.as_new_vec() }
  }

  impl Eq for ProductionReference {}

  impl gs::ProductionReference for ProductionReference {
    type ID = Self;
  }

  pub type CE = gs::CaseElement<Lit, ProductionReference>;

  #[derive(Debug, Clone)]
  pub struct Case(<Vec<CE> as IntoIterator>::IntoIter);

  impl From<&[CE]> for Case {
    fn from(value: &[CE]) -> Self { Self(value.to_vec().into_iter()) }
  }

  impl Iterator for Case {
    type Item = CE;

    fn next(&mut self) -> Option<Self::Item> { self.0.next() }
  }

  impl gs::Case for Case {
    type Item = CE;
    type Lit = Lit;
    type PR = ProductionReference;
  }

  #[derive(Debug, Clone)]
  pub struct Production(<Vec<Case> as IntoIterator>::IntoIter);

  impl From<&[Case]> for Production {
    fn from(value: &[Case]) -> Self { Self(value.to_vec().into_iter()) }
  }

  impl Iterator for Production {
    type Item = Case;

    fn next(&mut self) -> Option<Self::Item> { self.0.next() }
  }

  impl gs::Production for Production {
    type C = Case;
    type Item = Case;
  }

  #[derive(Debug, Clone)]
  pub struct SP(<Vec<(ProductionReference, Production)> as IntoIterator>::IntoIter);

  impl From<&[(ProductionReference, Production)]> for SP {
    fn from(value: &[(ProductionReference, Production)]) -> Self {
      Self(value.to_vec().into_iter())
    }
  }

  impl Iterator for SP {
    type Item = (ProductionReference, Production);

    fn next(&mut self) -> Option<Self::Item> { self.0.next() }
  }

  impl gs::SimultaneousProductions for SP {
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
