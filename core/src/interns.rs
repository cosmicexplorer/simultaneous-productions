/* Copyright (C) 2021 Danny McClanahan <dmcC2@hypnicjerk.ai> */
/* SPDX-License-Identifier: AGPL-3.0 */

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
  Arena: Allocator
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
  R: Into<usize>,
  Arena: Allocator,
{
  #[allow(dead_code)]
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
