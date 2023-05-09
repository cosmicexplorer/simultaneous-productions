/*
 * Description: Implementation of parsing.
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

//! Implementation of parsing. Performance does *(eventually)* matter here.

use crate::{
  grammar_indexing as gi, grammar_specification as gs,
  lowering_to_indices::{grammar_building as gb, graph_coordinates as gc},
};

use displaydoc::Display;
use indexmap::{IndexMap, IndexSet};
use priority_queue::PriorityQueue;
use thiserror::Error;

use core::{cmp, fmt, hash::Hash};

#[derive(Debug, Display, Error, Clone)]
pub enum ParseReconciliationError {}

/* pub enum ParseReconciliationResult { */
/*   Empty, */
/*   Some(Vec<Subparse>), */
/* } */

pub enum Direction {
  Left,
  Right,
}

pub trait Subparse {
  fn try_stack_step(self, direction: Direction, step: gi::NamedOrAnonStep) -> Option<Self>;
  fn get_neighboring_stack_steps(self, direction: Direction) -> Vec<gi::NamedOrAnonStep>;
}

pub enum SubparseEntryPoint {
  Literal(A),
  Graph(B),
}

pub enum Subparse {
  Literal(A),
  Graph(B),
}

impl Subparse {
  pub fn intersect_with_neighboring(self, from_direction: Direction) -> Vec<Self> {}
}

/* Trying to find:

1. a sequence of states S = s_i that correspond to the input token sequence (with a starting Start
   and ending End state),
2. the stack steps Z(i,i+1) in between each consecutive state s_i in that sequence S,

such that Z(i,i+i+1) is a valid path from s_i -> s_{i + 1} in the original input grammar.


We do this by:
1. Assigning all possible states to each element of the input token sequence (as in, each token maps
   to at least one state, so we start by assuming all states are valid).
2. For each i, look at the grammar definition and identify the set of all valid paths P(i)
   from s_i -> s_{i + 1} (which may include cyclical paths and therefore may have infinite
   cardinality).
3. For all i, attempt to "zip" each pair of adjacent paths from P(i) and P(i+1), and discard
   adjacent paths which are unsound (contain contradictory stack steps such as a pop without
   a corresponding push).
   - Note that the order in which adjacent paths are selected and evaluated is not yet specified.
4. A parse succeeds when a path zips across the entire input, and fails if every sub-path is
   discarded.


As it stands, a parse may also continue forever, regardless of whether the answer to "does this
string belong to my language's grammar" is "yes" or "no". However, we will also introduce some
additional constraints to the search process which ensure termination in the "yes" as well as "no"
case:
1. Ordering: evaluate all (pairs of?) smaller adjacent paths before (pairs of?) larger
   adjacent paths.
   - This is needed in a basic sense, so that `A = ""; A = Aa` sees whether popping the A stack is
     needed on the right before pushing more and more on the left infinitely.
     - However, this allows users to provide *inherently infinite constructions*
       like `A = ""; A = AaA` in their grammar (this one is also ambiguous, which may also be
       desired).
   - This should ensure that if there is a finite "yes" solution, that the parse will always find
     that solution in finite time (THAT IS THE INTENTION: NEED TO PROVE THIS).
   - TODO: "larger" in terms of spanned input tokens, or in terms of spanned stack transitions?
     Likely both, in order to cover `A = AaA`?
2. Bounding, which "cuts off" any further steps into a stack cycle according to some criteria:
   a. "lookaround": bound how many input tokens the current path is allowed to span before cutoff.
   b. "recursion depth": bound how many times the path between two adjacent states is allowed to
      cycle (to intersect itself) before cutoff.
   - Bounds should be applicable to individual productions or cases as well as the entire graph,
     **since the end user will be best able to determine which bounds apply to their language**.
     - Any bound on the "stack cycle" mechanism will ensure termination, but most stack cycles are
       intended to remain unbounded (e.g. I want "a+" to count any number of consecutive "a"s). The
       end user will be best able to determine which constraint is appropriate to their use case.
   - TODO: what is "cutoff"? It appears to refer to the intermediate paths spawned between
     some interval.


Each "stack cycle" is a cyclic path of stack steps between (not including!!!) state vertices in the
`EpsilonIntervalGraph` (which may or may not begin or end with the cycle!). With a finite
breadth-first search, we can identify all cycles in all paths between a given consecutive state pair
and mark all nodes in the cycle as cyclic........?


!

I'm pretty sure undecidability only occurs with multiple separate stacks, which isn't
implemented yet. So we don't need to consider "bounding" yet, and we can implement "ordering" for
now by simply not considering longer stack transitions before exhausting the shorter ones (which
requires us to do a breadth-first search when connecting up adjacent sub-parses!).
 */
