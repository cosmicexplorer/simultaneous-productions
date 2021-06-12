/* Copyright (C) 2021 Danny McClanahan <dmcC2@hypnicjerk.ai> */
/* SPDX-License-Identifier: GPL-3.0 */

use crate::vec::Vec;

use core::{alloc::Allocator, marker::PhantomData};

/* use indexmap::IndexSet; */
/* use twox_hash::XxHash64; */
/* #[derive(Debug, Clone)] */
/* pub struct InternArena<T, R, Arena: Allocator+Clone> { */
/* obarray: IndexSet<T, Arena, BuildHasherDefault<XxHash64>>, */
/* _x: PhantomData<R>, */
/* } */

pub struct InternArena<T, R, Arena>
where Arena: Allocator
{
  obarray: Vec<T, Arena>,
  arena: Arena,
  _x: PhantomData<R>,
}

impl<T, R, Arena> InternArena<T, R, Arena>
where Arena: Allocator
{
  pub fn new(arena: Arena) -> Self {
    Self {
      obarray: Vec::new_in(arena.clone()),
      arena,
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

  pub fn into_vec(self) -> Vec<(R, T), Arena> {
    let mut ret: Vec<(R, T), Arena> = Vec::new_in(self.arena.clone());
    let pairs = self
      .obarray
      .into_iter()
      .enumerate()
      .map(|(index, value)| (R::into(index), value));
    ret.extend(pairs);
    ret
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
  fn key_for(&self, x: &T) -> Option<R> { self.obarray.iter().position(|y| y == *x).map(R::into) }

  pub fn intern_exclusive(&mut self, x: T) -> R {
    self
      .key_for(&x)
      .unwrap_or_else(|| self.intern_always_new_increasing(x))
  }
}
