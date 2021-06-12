/* Copyright (C) 2021 Danny McClanahan <dmcC2@hypnicjerk.ai> */
/* SPDX-License-Identifier: GPL-3.0 */

use crate::vec::Vec;

use core::{alloc::Allocator, fmt, marker::PhantomData};

pub struct InternArena<T, R, Arena>
where Arena: Allocator
{
  obarray: Vec<T, Arena>,
  arena: Arena,
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
where Arena: Allocator+Clone
{
  pub fn new(arena: Arena) -> Self {
    Self {
      obarray: Vec::new_in(arena.clone()),
      arena,
      _x: PhantomData,
    }
  }

  pub fn arena(&self) -> Arena { self.arena.clone() }
}

impl<T, R, Arena> Clone for InternArena<T, R, Arena>
where
  T: Clone,
  Arena: Allocator+Clone,
{
  fn clone(&self) -> Self {
    Self {
      obarray: self.obarray.clone(),
      arena: self.arena.clone(),
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
    let mut ret: Vec<(R, T), Arena> = Vec::new_in(self.arena.clone());
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
  Arena: Allocator+Clone,
{
  fn from(value: Vec<T, Arena>) -> Self {
    let arena = value.allocator().clone();
    Self {
      obarray: value,
      arena,
      _x: PhantomData,
    }
  }
}

impl<T, R, Arena> InternArena<T, R, Arena>
where
  R: Into<usize>,
  Arena: Allocator,
{
  pub fn retrieve(&self, key: R) -> &T {
    self
      .obarray
      .get(key.into())
      .expect("the type safety with the R parameter was supposed to stop this")
  }
}

impl<T, R, Arena> InternArena<T, R, Arena>
where
  T: Eq,
  R: From<usize>,
  Arena: Allocator,
{
  fn key_for(&self, x: &T) -> Option<R> { self.obarray.iter().position(|y| y == x).map(R::from) }

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
