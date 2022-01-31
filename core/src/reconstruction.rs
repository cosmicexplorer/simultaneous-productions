/*
 * Description: Reconstruct the segments of the input corresponding to
 * successfully-matched parse trees.
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

use crate::{
  allocation::HandoffAllocable, grammar_indexing as gi,
  lowering_to_indices::graph_coordinates as gc, parsing as p, types::Vec,
};

use core::alloc::Allocator;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReconstructionError {}

///
/// TODO: why is this the appropriate representation for an intermediate
/// reconstruction?
#[derive(Clone)]
pub struct IntermediateReconstruction<Arena>
where Arena: Allocator
{
  pub prod_case: gc::ProdCaseRef,
  pub args: Vec<CompleteSubReconstruction<Arena>, Arena>,
}

impl<Arena> IntermediateReconstruction<Arena>
where Arena: Allocator
{
  pub fn empty_for_case(prod_case: gc::ProdCaseRef, arena: Arena) -> Self {
    IntermediateReconstruction {
      prod_case,
      args: Vec::new_in(arena),
    }
  }
}

impl<Arena> HandoffAllocable for IntermediateReconstruction<Arena>
where Arena: Allocator+Clone
{
  type Arena = Arena;

  fn allocator_handoff(&self) -> Arena { self.args.allocator().clone() }
}

#[derive(Clone)]
pub enum DirectionalIntermediateReconstruction<Arena>
where Arena: Allocator
{
  Rightwards(IntermediateReconstruction<Arena>),
  Leftwards(IntermediateReconstruction<Arena>),
}

impl<Arena> HandoffAllocable for DirectionalIntermediateReconstruction<Arena>
where Arena: Allocator+Clone
{
  type Arena = Arena;

  fn allocator_handoff(&self) -> Arena {
    match self {
      Self::Rightwards(x) => x.allocator_handoff(),
      Self::Leftwards(x) => x.allocator_handoff(),
    }
  }
}

impl<Arena> DirectionalIntermediateReconstruction<Arena>
where Arena: Allocator+Clone
{
  pub fn add_completed(self, sub: CompleteSubReconstruction<Arena>) -> Self {
    let mut ret: Vec<CompleteSubReconstruction<Arena>, Arena> =
      Vec::new_in(self.allocator_handoff());
    match self {
      Self::Rightwards(IntermediateReconstruction { prod_case, args }) => {
        ret.extend(args.into_iter());
        ret.push(sub);
        Self::Rightwards(IntermediateReconstruction {
          prod_case,
          args: ret,
        })
      },
      Self::Leftwards(IntermediateReconstruction { prod_case, args }) => {
        ret.push(sub);
        ret.extend(args.into_iter());
        Self::Leftwards(IntermediateReconstruction {
          prod_case,
          args: ret,
        })
      },
    }
  }
}

#[derive(Clone)]
pub enum ReconstructionElement<Arena>
where Arena: Allocator
{
  Intermediate(DirectionalIntermediateReconstruction<Arena>),
  CompletedSub(CompleteSubReconstruction<Arena>),
}

#[derive(Clone)]
pub struct InProgressReconstruction<Arena>
where Arena: Allocator
{
  pub elements: Vec<ReconstructionElement<Arena>, Arena>,
}

impl<Arena> HandoffAllocable for InProgressReconstruction<Arena>
where Arena: Allocator+Clone
{
  type Arena = Arena;

  fn allocator_handoff(&self) -> Arena { self.elements.allocator().clone() }
}

impl<Arena> InProgressReconstruction<Arena>
where Arena: Allocator+Clone
{
  pub fn joined(sub_reconstructions: Vec<Self, Arena>) -> Self {
    let arena = sub_reconstructions.allocator().clone();
    sub_reconstructions.into_iter().fold(
      InProgressReconstruction {
        elements: Vec::new_in(arena),
      },
      |acc, next| acc.join(next),
    )
  }

  pub fn join(self, other: Self) -> Self {
    let arena = self.allocator_handoff();
    /* dbg!(&self); */
    /* dbg!(&other); */
    let InProgressReconstruction {
      elements: left_initial_elements,
    } = self;
    let InProgressReconstruction {
      elements: right_initial_elements,
    } = other;

    /* Initialize two queues, with the left empty, and the right containing the
     * concatenation of both objects. */
    let mut right_side: Vec<ReconstructionElement<Arena>, Arena> = Vec::with_capacity_in(
      left_initial_elements.len() + right_initial_elements.len(),
      arena.clone(),
    );
    right_side.extend(left_initial_elements.into_iter());
    right_side.extend(right_initial_elements.into_iter());

    let mut left_side: Vec<ReconstructionElement<Arena>, Arena> = Vec::new_in(arena.clone());
    /* TODO: document how this zippering works with two queues! */
    while !right_side.is_empty() {
      if left_side.is_empty() {
        left_side.push(right_side.remove(0));
        continue;
      }
      let left_intermediate = left_side.pop().unwrap();
      let right_intermediate = right_side.remove(0);
      /* dbg!(&left_intermediate); */
      /* dbg!(&right_intermediate); */
      match (left_intermediate, right_intermediate) {
        (
          ReconstructionElement::Intermediate(DirectionalIntermediateReconstruction::Rightwards(
            left,
          )),
          ReconstructionElement::CompletedSub(complete_right),
        ) => {
          let inner_element = ReconstructionElement::Intermediate(
            DirectionalIntermediateReconstruction::Rightwards(left).add_completed(complete_right),
          );
          left_side.push(inner_element);
        },
        (
          ReconstructionElement::CompletedSub(complete_left),
          ReconstructionElement::Intermediate(DirectionalIntermediateReconstruction::Leftwards(
            right,
          )),
        ) => {
          let inner_element = ReconstructionElement::Intermediate(
            DirectionalIntermediateReconstruction::Leftwards(right).add_completed(complete_left),
          );
          right_side.insert(0, inner_element);
        },
        (
          ReconstructionElement::Intermediate(DirectionalIntermediateReconstruction::Rightwards(
            IntermediateReconstruction {
              prod_case: left_prod_case,
              args: left_args,
            },
          )),
          ReconstructionElement::Intermediate(DirectionalIntermediateReconstruction::Leftwards(
            IntermediateReconstruction {
              prod_case: right_prod_case,
              args: right_args,
            },
          )),
        ) => {
          if left_prod_case == right_prod_case {
            /* Complete the paired group, and push it back onto the left stack. Left was
             * chosen arbitrarily here. */
            let mut all_args: Vec<_, Arena> =
              Vec::with_capacity_in(left_args.len() + right_args.len(), arena.clone());
            all_args.extend(left_args.into_iter());
            all_args.extend(right_args.into_iter());
            let inner_element = ReconstructionElement::CompletedSub(
              CompleteSubReconstruction::Completed(CompletedCaseReconstruction {
                prod_case: left_prod_case,
                args: all_args,
              }),
            );
            left_side.push(inner_element);
          } else {
            /* TODO: support non-hierarchical input! */
            todo!("non-hierarchical input recovery is not yet supported!");
          }
        },
        /* Shuffle everything down one! */
        (
          ReconstructionElement::Intermediate(DirectionalIntermediateReconstruction::Leftwards(
            pointing_left,
          )),
          x_right,
        ) => {
          left_side.push(ReconstructionElement::Intermediate(
            DirectionalIntermediateReconstruction::Leftwards(pointing_left),
          ));
          left_side.push(x_right);
        },
        (
          x_left,
          ReconstructionElement::Intermediate(DirectionalIntermediateReconstruction::Rightwards(
            pointing_right,
          )),
        ) => {
          left_side.push(x_left);
          left_side.push(ReconstructionElement::Intermediate(
            DirectionalIntermediateReconstruction::Rightwards(pointing_right),
          ));
        },
        (
          ReconstructionElement::CompletedSub(complete_left),
          ReconstructionElement::CompletedSub(complete_right),
        ) => {
          left_side.push(ReconstructionElement::CompletedSub(complete_left));
          left_side.push(ReconstructionElement::CompletedSub(complete_right));
        },
      }
    }
    /* dbg!(&left_side); */
    InProgressReconstruction {
      elements: left_side,
    }
  }
}

impl<Arena> InProgressReconstruction<Arena>
where Arena: Allocator+Clone
{
  #[allow(dead_code)]
  pub fn new(tree: p::SpanningSubtreeRef, parse: &p::Parse<Arena>) -> Self {
    let arena = parse.allocator_handoff();
    let &p::Parse {
      grammar: p::ParseableGrammar {
        ref anon_step_mapping,
        ..
      },
      ..
    } = parse;
    let p::SpanningSubtree {
      input_span:
        p::FlattenedSpanInfo {
          state_pair: gi::StatePair { left, right },
          stack_diff: gi::StackDiffSegment(stack_diff),
          ..
        },
      parents,
      ..
    } = parse
      .get_spanning_subtree(tree)
      .expect("tree ref should have been in parse");

    let (prologue, epilogue) = match parents {
      None => {
        let mut left_ret: Vec<_, Arena> = Vec::with_capacity_in(1, arena.clone());
        left_ret.push(ReconstructionElement::CompletedSub(
          CompleteSubReconstruction::State(*left),
        ));
        let mut right_ret: Vec<_, Arena> = Vec::with_capacity_in(1, arena.clone());
        right_ret.push(ReconstructionElement::CompletedSub(
          CompleteSubReconstruction::State(*right),
        ));
        (
          InProgressReconstruction { elements: left_ret },
          InProgressReconstruction {
            elements: right_ret,
          },
        )
      },
      Some(p::ParentInfo {
        left_parent,
        right_parent,
      }) => (
        Self::new(*left_parent, parse),
        Self::new(*right_parent, parse),
      ),
    };

    /* dbg!(&prologue); */
    /* dbg!(&epilogue); */
    /* dbg!(&stack_diff); */
    let mut middle_elements: Vec<InProgressReconstruction<Arena>, Arena> =
      Vec::new_in(arena.clone());
    /* The `stack_diff` is just a flattened version of the parents' diffs -- we
     * don't add it twice! */
    if parents.is_none() {
      /* Convert the `stack_diff` into its own set of possibly-incomplete
       * sub-reconstructions! */
      middle_elements.extend(stack_diff.iter().flat_map(|step| match step {
        /* NB: "named" steps are only relevant for constructing the interval graph with
         * anonymous steps, which denote the correct `ProdCaseRef` to use, so we
         * discard them here. */
        gi::NamedOrAnonStep::Named(_) => None,
        gi::NamedOrAnonStep::Anon(anon_step) => match anon_step {
          gi::AnonStep::Positive(anon_sym) => {
            let maybe_ref: &gi::UnflattenedProdCaseRef = anon_step_mapping
              .get(anon_sym)
              .unwrap_or_else(|| unreachable!("no state found for anon sym {:?}", anon_sym));
            match maybe_ref {
              &gi::UnflattenedProdCaseRef::PassThrough => None,
              &gi::UnflattenedProdCaseRef::Case(ref x) => {
                let mut elements: Vec<_, Arena> = Vec::with_capacity_in(1, arena.clone());
                elements.push(ReconstructionElement::Intermediate(
                  DirectionalIntermediateReconstruction::Rightwards(
                    IntermediateReconstruction::empty_for_case(*x, arena.clone()),
                  ),
                ));
                Some(InProgressReconstruction { elements })
              },
            }
          },
          gi::AnonStep::Negative(anon_sym) => {
            let maybe_ref: &gi::UnflattenedProdCaseRef = anon_step_mapping
              .get(anon_sym)
              .unwrap_or_else(|| unreachable!("no state found for anon sym {:?}", anon_sym));
            match maybe_ref {
              &gi::UnflattenedProdCaseRef::PassThrough => None,
              &gi::UnflattenedProdCaseRef::Case(ref x) => {
                let mut elements: Vec<_, Arena> = Vec::with_capacity_in(1, arena.clone());
                elements.push(ReconstructionElement::Intermediate(
                  DirectionalIntermediateReconstruction::Leftwards(
                    IntermediateReconstruction::empty_for_case(*x, arena.clone()),
                  ),
                ));
                Some(InProgressReconstruction { elements })
              },
            }
          },
        },
      }));
    };
    /* eprintln!("middle_elements: {:?}", middle_elements); */

    let mut ret: Vec<_, Arena> = Vec::with_capacity_in(middle_elements.len() + 2, arena);
    ret.push(prologue);
    ret.extend(middle_elements.into_iter());
    ret.push(epilogue);
    InProgressReconstruction::joined(ret)
  }
}

#[derive(Debug, Clone)]
pub struct CompletedCaseReconstruction<Arena>
where Arena: Allocator
{
  pub prod_case: gc::ProdCaseRef,
  pub args: Vec<CompleteSubReconstruction<Arena>, Arena>,
}

impl<Arena> PartialEq for CompletedCaseReconstruction<Arena>
where Arena: Allocator
{
  fn eq(&self, other: &Self) -> bool {
    self.prod_case == other.prod_case && self.args == other.args
  }
}

impl<Arena> Eq for CompletedCaseReconstruction<Arena> where Arena: Allocator {}

#[derive(Debug, Clone)]
pub enum CompleteSubReconstruction<Arena>
where Arena: Allocator
{
  State(gi::LoweredState),
  Completed(CompletedCaseReconstruction<Arena>),
}

impl<Arena> PartialEq for CompleteSubReconstruction<Arena>
where Arena: Allocator
{
  fn eq(&self, other: &Self) -> bool {
    match (self, other) {
      (Self::State(x), Self::State(y)) if x == y => true,
      (Self::Completed(x), Self::Completed(y)) if x == y => true,
      _ => false,
    }
  }
}

impl<Arena> Eq for CompleteSubReconstruction<Arena> where Arena: Allocator {}

#[derive(Debug, Clone)]
pub struct CompletedWholeReconstruction<Arena>(pub Vec<CompleteSubReconstruction<Arena>, Arena>)
where Arena: Allocator;

impl<Arena> CompletedWholeReconstruction<Arena>
where Arena: Allocator+Clone
{
  #[allow(dead_code)]
  pub fn new(maybe_completed_constructions: InProgressReconstruction<Arena>) -> Self {
    let arena = maybe_completed_constructions.allocator_handoff();
    let mut sub_constructions: Vec<_, Arena> =
      Vec::with_capacity_in(maybe_completed_constructions.elements.len(), arena);
    sub_constructions.extend(maybe_completed_constructions.elements.into_iter().map(
      |el| match el {
        ReconstructionElement::Intermediate(_) => {
          unreachable!("expected all sub constructions to be completed!");
        },
        ReconstructionElement::CompletedSub(x) => x,
      },
    ));
    CompletedWholeReconstruction(sub_constructions)
  }
}

impl<Arena> PartialEq for CompletedWholeReconstruction<Arena>
where Arena: Allocator
{
  fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<Arena> Eq for CompletedWholeReconstruction<Arena> where Arena: Allocator {}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    grammar_indexing as gi,
    lowering_to_indices::graph_coordinates as gc,
    parsing as p, state,
    test_framework::{new_token_position, non_cyclic_productions},
    types::Global,
  };

  #[test]
  fn reconstructs_from_parse() {
    let prods = non_cyclic_productions();
    let detokenized = state::preprocessing::Init(prods)
      .try_index_with_allocator(Global)
      .unwrap();
    let indexed = detokenized.index();
    let string_input = "ab";
    let input = p::Input(string_input.chars().collect());
    let i2 = indexed.clone();
    let ready = i2.attach_input(&input).unwrap();

    let mut parse = ready.initialize_parse().0;

    let spanning_subtree_ref = parse.get_next_parse();
    let reconstructed = InProgressReconstruction::new(spanning_subtree_ref, &parse);
    let completely_reconstructed = CompletedWholeReconstruction::new(reconstructed);
    assert_eq!(
      completely_reconstructed,
      CompletedWholeReconstruction(
        [
          CompleteSubReconstruction::State(gi::LoweredState::Start),
          CompleteSubReconstruction::Completed(CompletedCaseReconstruction {
            prod_case: gc::ProdCaseRef {
              prod: gc::ProdRef(0),
              case: gc::CaseRef(0)
            },
            args: [
              CompleteSubReconstruction::State(gi::LoweredState::Within(new_token_position(
                0, 0, 0
              ))),
              CompleteSubReconstruction::State(gi::LoweredState::Within(new_token_position(
                0, 0, 1
              ))),
            ]
            .as_ref()
            .to_vec()
          }),
          CompleteSubReconstruction::State(gi::LoweredState::End),
        ]
        .as_ref()
        .to_vec()
      )
    );

    /* Try it again, crossing productions this time. */
    let longer_string_input = "abab";
    let longer_input = p::Input(longer_string_input.chars().collect());
    let longer_ready = indexed.attach_input(&longer_input).unwrap();
    let mut longer_parse = longer_ready.initialize_parse().0;
    let first_parsed_longer_string = longer_parse.get_next_parse();
    let longer_reconstructed =
      InProgressReconstruction::new(first_parsed_longer_string, &longer_parse);
    let longer_completely_reconstructed = CompletedWholeReconstruction::new(longer_reconstructed);
    assert_eq!(
      longer_completely_reconstructed,
      CompletedWholeReconstruction(
        [
          CompleteSubReconstruction::State(gi::LoweredState::Start),
          CompleteSubReconstruction::Completed(CompletedCaseReconstruction {
            prod_case: gc::ProdCaseRef {
              prod: gc::ProdRef(1),
              case: gc::CaseRef(0),
            },
            args: [
              CompleteSubReconstruction::State(gi::LoweredState::Within(new_token_position(
                1, 0, 0
              ))),
              CompleteSubReconstruction::State(gi::LoweredState::Within(new_token_position(
                1, 0, 1
              ))),
              CompleteSubReconstruction::Completed(CompletedCaseReconstruction {
                prod_case: gc::ProdCaseRef {
                  prod: gc::ProdRef(0),
                  case: gc::CaseRef(0),
                },
                args: [
                  CompleteSubReconstruction::State(gi::LoweredState::Within(new_token_position(
                    0, 0, 0
                  ))),
                  CompleteSubReconstruction::State(gi::LoweredState::Within(new_token_position(
                    0, 0, 1
                  ))),
                ]
                .as_ref()
                .to_vec(),
              })
            ]
            .as_ref()
            .to_vec(),
          }),
          CompleteSubReconstruction::State(gi::LoweredState::End),
        ]
        .as_ref()
        .to_vec()
      )
    );
  }
}
