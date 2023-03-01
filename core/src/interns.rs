/*
 * Description: Allocate objects within an arena and give them names.
 *
 * Copyright (C) 2021-2023 Danny McClanahan <dmcC2@hypnicjerk.ai>
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

use core::{fmt, marker::PhantomData};

pub struct InternArena<T, R> {
  obarray: Vec<T>,
  #[doc(hidden)]
  _x: PhantomData<R>,
}

impl<T, R> fmt::Debug for InternArena<T, R>
where T: fmt::Debug
{
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "InternArena({:?})", &self.obarray)
  }
}

impl<T, R> InternArena<T, R> {
  pub fn new() -> Self {
    Self {
      obarray: Vec::new(),
      _x: PhantomData,
    }
  }
}

impl<T, R> Clone for InternArena<T, R>
where T: Clone
{
  fn clone(&self) -> Self {
    Self {
      obarray: self.obarray.clone(),
      _x: PhantomData,
    }
  }
}

impl<T, R> InternArena<T, R>
where R: From<usize>
{
  pub fn intern_always_new_increasing(&mut self, x: T) -> R {
    self.obarray.push(x);
    let new_element_index: usize = self.obarray.len() - 1;
    new_element_index.into()
  }
}

impl<T, R> InternArena<T, R>
where R: From<usize>
{
  pub fn into_vec(self) -> Vec<(R, T)> {
    let mut ret: Vec<(R, T)> = Vec::new();
    let pairs = self
      .obarray
      .into_iter()
      .enumerate()
      .map(|(index, value)| (R::from(index), value));
    ret.extend(pairs);
    ret
  }
}

impl<T, R> From<Vec<T>> for InternArena<T, R>
where R: From<usize>
{
  fn from(value: Vec<T>) -> Self {
    Self {
      obarray: value,
      _x: PhantomData,
    }
  }
}

impl<T, R> InternArena<T, R>
where
  T: Eq,
  R: From<usize>,
{
  fn key_for(&self, x: &T) -> Option<R> { self.obarray.iter().position(|y| y == x).map(R::from) }

  pub fn retrieve_intern(&self, x: &T) -> Option<R> { self.key_for(x) }

  pub fn intern_exclusive(&mut self, x: T) -> R {
    self
      .key_for(&x)
      .unwrap_or_else(|| self.intern_always_new_increasing(x))
  }
}

impl<T, R> PartialEq for InternArena<T, R>
where T: Eq
{
  fn eq(&self, other: &Self) -> bool { self.obarray == other.obarray }
}

impl<T, R> Eq for InternArena<T, R> where T: Eq {}
