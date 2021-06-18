/*
 * Description: Adapter for the S.P. parsing method to the experimental rust
 *              async stream API.
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

//! A [`Stream`][core::stream::Stream]-based API to an
//! [`sp_core::execution::Transformer`].
//!
//! Requires nightly rust with the crate feature `#![feature(async_stream)]`.

#![no_std]
#![allow(incomplete_features)]
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

use sp_core::execution::{Input, Output, Transformer};

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
