/*
 * Description: Allocate objects within an arena and give them names.
 *
 * Copyright (C) 2021-2022 Danny McClanahan <dmcC2@hypnicjerk.ai>
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

use crate::{allocation::HandoffAllocable, types::Vec};

use core::{alloc::Allocator, fmt, marker::PhantomData};

pub struct InternArena<T, R, Arena>
where Arena: Allocator
{
  obarray: Vec<T, Arena>,
  #[doc(hidden)]
  _x: PhantomData<R>,
}

impl<T, R, Arena> fmt::Debug for InternArena<T, R, Arena>
where
  T: fmt::Debug,
  Arena: Allocator,
{
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "InternArena({:?})", &self.obarray)
  }
}

impl<T, R, Arena> InternArena<T, R, Arena>
where Arena: Allocator
{
  pub fn new(arena: Arena) -> Self {
    Self {
      obarray: Vec::new_in(arena),
      _x: PhantomData,
    }
  }
}

impl<T, R, Arena> HandoffAllocable for InternArena<T, R, Arena>
where Arena: Allocator+Clone
{
  type Arena = Arena;

  fn allocator_handoff(&self) -> Arena { self.obarray.allocator().clone() }
}


impl<T, R, Arena> Clone for InternArena<T, R, Arena>
where
  T: Clone,
  Arena: Allocator+Clone,
{
  fn clone(&self) -> Self {
    Self {
      obarray: self.obarray.clone(),
      _x: PhantomData,
    }
  }
}

impl<T, R, Arena> InternArena<T, R, Arena>
where
  R: From<usize>,
  Arena: Allocator,
{
  pub fn intern_always_new_increasing(&mut self, x: T) -> R {
    self.obarray.push(x);
    let new_element_index: usize = self.obarray.len() - 1;
    new_element_index.into()
  }
}

impl<T, R, Arena> InternArena<T, R, Arena>
where
  R: From<usize>,
  Arena: Allocator+Clone,
{
  pub fn into_vec(self) -> Vec<(R, T), Arena> {
    let mut ret: Vec<(R, T), Arena> = Vec::new_in(self.allocator_handoff());
    let pairs = self
      .obarray
      .into_iter()
      .enumerate()
      .map(|(index, value)| (R::from(index), value));
    ret.extend(pairs);
    ret
  }
}

impl<T, R, Arena> From<Vec<T, Arena>> for InternArena<T, R, Arena>
where
  R: From<usize>,
  Arena: Allocator,
{
  fn from(value: Vec<T, Arena>) -> Self {
    Self {
      obarray: value,
      _x: PhantomData,
    }
  }
}

impl<T, R, Arena> InternArena<T, R, Arena>
where
  T: Eq,
  R: From<usize>,
  Arena: Allocator,
{
  fn key_for(&self, x: &T) -> Option<R> { self.obarray.iter().position(|y| y == x).map(R::from) }

  pub fn retrieve_intern(&self, x: &T) -> Option<R> { self.key_for(x) }

  pub fn intern_exclusive(&mut self, x: T) -> R {
    self
      .key_for(&x)
      .unwrap_or_else(|| self.intern_always_new_increasing(x))
  }
}

impl<T, R, Arena> PartialEq for InternArena<T, R, Arena>
where
  T: Eq,
  Arena: Allocator,
{
  fn eq(&self, other: &Self) -> bool { self.obarray == other.obarray }
}

impl<T, R, Arena> Eq for InternArena<T, R, Arena>
where
  T: Eq,
  Arena: Allocator,
{
}
