/*
 * Description: Adapter for the S.P. parsing method to the experimental rust
 *              generator API.
 *
 * Copyright (C) 2021 Danny McClanahan <dmcC2@hypnicjerk.ai>
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

//! A [`Generator`][core::ops::Generator]-based API to an
//! [`sp_core::execution::Transformer`].
//!
//! Requires nightly rust with the crate feature `#![feature(generators,
//! generator_trait)]`.

#![no_std]
#![allow(incomplete_features)]
#![feature(generators, generator_trait)]
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

use sp_core::execution::{Input, Output, Transformer};

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
/// unlike for the [`iterator_api`][sp_core::execution::iterator_api]!*
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
