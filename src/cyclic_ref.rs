/*
   Description: A wrapper type allowing cyclic ref-counted pointers.
   Copyright (C) 2019  Danny McClanahan (https://twitter.com/hipsterelectron)

   TODO: Get Twitter to sign a copyright disclaimer!

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

use std::{
  borrow::{Borrow, BorrowMut},
  cell::RefCell,
  hash::{Hash, Hasher},
  rc::{Rc, Weak},
};

///
/// A way to wrap either a strong or weak pointer, assuming you've set up the
/// reference cycles so that this doesn't cause a memory leak. Useful for
/// intentional cycles.
#[derive(Debug, Clone)]
pub enum StrongOrWeakRef<T> {
  Strong(Rc<RefCell<T>>),
  Weak(Weak<RefCell<T>>),
}

impl<T> StrongOrWeakRef<T> {
  /* NB: Should be called as an associated method! */
  pub fn clone_strong(&self) -> Self {
    match self {
      StrongOrWeakRef::Strong(strong_ptr) => StrongOrWeakRef::Strong(Rc::clone(strong_ptr)),
      StrongOrWeakRef::Weak(weak_ptr) => {
        let strong_version = weak_ptr
          .upgrade()
          .expect("weak StrongOrWeakRef upgrade failed for clone_strong!");
        StrongOrWeakRef::Strong(Rc::clone(&strong_version))
      },
    }
  }

  /* NB: Should be called as an associated method! */
  pub fn clone_weak(&self) -> Self {
    match self {
      StrongOrWeakRef::Strong(strong_ptr) => {
        StrongOrWeakRef::Weak(Rc::downgrade(&Rc::clone(strong_ptr)))
      },
      StrongOrWeakRef::Weak(weak_ptr) => StrongOrWeakRef::Weak(Weak::clone(weak_ptr)),
    }
  }

  pub fn make_strong(item: T) -> Self {
    StrongOrWeakRef::Strong(Rc::new(RefCell::new(item)))
  }
}

/* TODO: make this less hacky somehow, this is ridiculous but I cannot figure
 * out why .borrow() and .borrow_mut() don't just work to forward borrowing! */
impl<T> Borrow<T> for StrongOrWeakRef<T> {
  fn borrow(&self) -> &T {
    match self {
      StrongOrWeakRef::Strong(strong_ptr) => {
        let x: *mut T = strong_ptr.as_ptr();
        unsafe {
          &*x
        }
      },
      StrongOrWeakRef::Weak(weak_ptr) => {
        let x: *mut T = weak_ptr
          .upgrade()
          .expect("weak StrongOrWeakRef upgrade failed for borrow!")
          .as_ptr();
        unsafe {
          &*x
        }
      },
    }
  }
}

impl<T> BorrowMut<T> for StrongOrWeakRef<T> {
  fn borrow_mut(&mut self) -> &mut T {
    match self {
      StrongOrWeakRef::Strong(strong_ptr) => {
        let x: *mut T = strong_ptr.as_ptr();
        unsafe {
          &mut *x
        }
      },
      StrongOrWeakRef::Weak(weak_ptr) => {
        let x: *mut T = weak_ptr
          .upgrade()
          .expect("weak StrongOrWeakRef upgrade failed for borrow_mut!")
          .as_ptr();
        unsafe {
          &mut *x
        }
      },
    }
  }
}

impl<T> PartialEq for StrongOrWeakRef<T> {
  /* Check equality by pointer value. */
  fn eq(&self, other: &Self) -> bool {
    let strong_self = match self {
      StrongOrWeakRef::Strong(x) => x.clone(),
      StrongOrWeakRef::Weak(x) => x.upgrade().expect("weak self during eq"),
    };
    let strong_other = match other {
      StrongOrWeakRef::Strong(x) => x.clone(),
      StrongOrWeakRef::Weak(x) => x.upgrade().expect("weak other during eq"),
    };
    Rc::ptr_eq(&strong_self, &strong_other)
  }
}

impl<T> Eq for StrongOrWeakRef<T> {}

impl<T> Hash for StrongOrWeakRef<T> {
  fn hash<H: Hasher>(&self, state: &mut H) {
    /* Hash by the pointer value. */
    let contained_ptr: *mut T = match self {
      StrongOrWeakRef::Strong(strong_ptr) => strong_ptr.as_ptr(),
      StrongOrWeakRef::Weak(weak_ptr) => weak_ptr
        .upgrade()
        .expect("weak StrongOrWeakRef upgrade failed for hash!")
        .as_ptr(),
    };
    state.write_usize(contained_ptr as usize);
  }
}

///
/// A tiny trait to indicate that some form of equality is defined which reaches into the contents
/// of each node instead of just counting references.
///
pub trait DeeplyEqual {
  fn deeply_equal(&self, other: &Self) -> bool;
}
