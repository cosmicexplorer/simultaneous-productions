/* Copyright (C) 2021 Danny McClanahan <dmcC2@hypnicjerk.ai> */
/* SPDX-License-Identifier: GPL-3.0 */

use crate::{grammar_indexing::*, parsing::*};

use typename::TypeName;

use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
pub struct ReconstructionError(String);

///
/// TODO: why is this the appropriate representation for an intermediate
/// reconstruction?
#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
pub struct IntermediateReconstruction {
  pub prod_case: ProdCaseRef,
  pub args: Vec<CompleteSubReconstruction>,
}

impl IntermediateReconstruction {
  pub fn empty_for_case(prod_case: ProdCaseRef) -> Self {
    IntermediateReconstruction {
      prod_case,
      args: vec![],
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
pub enum DirectionalIntermediateReconstruction {
  Rightwards(IntermediateReconstruction),
  Leftwards(IntermediateReconstruction),
}

impl DirectionalIntermediateReconstruction {
  pub fn add_completed(self, sub: CompleteSubReconstruction) -> Self {
    match self {
      Self::Rightwards(IntermediateReconstruction { prod_case, args }) => {
        Self::Rightwards(IntermediateReconstruction {
          prod_case,
          args: args.into_iter().chain(vec![sub]).collect(),
        })
      },
      Self::Leftwards(IntermediateReconstruction { prod_case, args }) => {
        Self::Leftwards(IntermediateReconstruction {
          prod_case,
          args: vec![sub].into_iter().chain(args).collect(),
        })
      },
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
pub enum ReconstructionElement {
  Intermediate(DirectionalIntermediateReconstruction),
  CompletedSub(CompleteSubReconstruction),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
pub struct InProgressReconstruction {
  pub elements: Vec<ReconstructionElement>,
}

impl InProgressReconstruction {
  pub fn empty() -> Self { InProgressReconstruction { elements: vec![] } }

  pub fn with_elements(elements: Vec<ReconstructionElement>) -> Self {
    InProgressReconstruction {
      elements: elements.into_iter().collect(),
    }
  }

  pub fn join(self, other: Self) -> Self {
    dbg!(&self);
    dbg!(&other);
    let InProgressReconstruction {
      elements: left_initial_elements,
    } = self;
    let InProgressReconstruction {
      elements: right_initial_elements,
    } = other;

    /* Initialize two queues, with the left empty, and the right containing the
     * concatenation of both objects. */
    let mut right_side: VecDeque<_> = left_initial_elements
      .into_iter()
      .chain(right_initial_elements.into_iter())
      .collect();
    let mut left_side: VecDeque<ReconstructionElement> = VecDeque::new();
    /* TODO: document how this zippering works with two queues! */
    while !right_side.is_empty() {
      if left_side.is_empty() {
        left_side.push_back(right_side.pop_front().unwrap());
        continue;
      }
      let left_intermediate = left_side.pop_back().unwrap();
      let right_intermediate = right_side.pop_front().unwrap();
      dbg!(&left_intermediate);
      dbg!(&right_intermediate);
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
          left_side.push_back(inner_element);
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
          right_side.push_front(inner_element);
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
            let inner_element = ReconstructionElement::CompletedSub(
              CompleteSubReconstruction::Completed(CompletedCaseReconstruction {
                prod_case: left_prod_case,
                args: left_args
                  .into_iter()
                  .chain(right_args.into_iter())
                  .collect(),
              }),
            );
            left_side.push_back(inner_element);
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
          left_side.push_back(ReconstructionElement::Intermediate(
            DirectionalIntermediateReconstruction::Leftwards(pointing_left),
          ));
          left_side.push_back(x_right);
        },
        (
          x_left,
          ReconstructionElement::Intermediate(DirectionalIntermediateReconstruction::Rightwards(
            pointing_right,
          )),
        ) => {
          left_side.push_back(x_left);
          left_side.push_back(ReconstructionElement::Intermediate(
            DirectionalIntermediateReconstruction::Rightwards(pointing_right),
          ));
        },
        (
          ReconstructionElement::CompletedSub(complete_left),
          ReconstructionElement::CompletedSub(complete_right),
        ) => {
          left_side.push_back(ReconstructionElement::CompletedSub(complete_left));
          left_side.push_back(ReconstructionElement::CompletedSub(complete_right));
        },
      }
    }
    dbg!(&left_side);
    InProgressReconstruction::with_elements(left_side.into_iter().collect())
  }

  pub fn joined(sub_reconstructions: Vec<Self>) -> Self {
    sub_reconstructions
      .into_iter()
      .fold(InProgressReconstruction::empty(), |acc, next| {
        acc.join(next)
      })
  }

  pub fn new(tree: SpanningSubtreeRef, parse: &Parse) -> Self {
    let &Parse {
      grammar: ParseableGrammar {
        ref anon_step_mapping,
        ..
      },
      ..
    } = parse;
    let SpanningSubtree {
      input_span:
        FlattenedSpanInfo {
          state_pair: StatePair { left, right },
          stack_diff: StackDiffSegment(stack_diff),
          ..
        },
      parents,
      ..
    } = parse
      .get_spanning_subtree(tree)
      .map(|x| {
        eprintln!("tree: {:?}", x);
        x
      })
      .expect("tree ref should have been in parse");

    let (prologue, epilogue) = match parents {
      None => (
        InProgressReconstruction::with_elements(vec![ReconstructionElement::CompletedSub(
          CompleteSubReconstruction::State(*left),
        )]),
        InProgressReconstruction::with_elements(vec![ReconstructionElement::CompletedSub(
          CompleteSubReconstruction::State(*right),
        )]),
      ),
      Some(ParentInfo {
        left_parent,
        right_parent,
      }) => (
        Self::new(*left_parent, parse),
        Self::new(*right_parent, parse),
      ),
    };

    dbg!(&prologue);
    dbg!(&epilogue);
    dbg!(&stack_diff);
    let middle_elements: Vec<InProgressReconstruction> = match parents {
      /* The `stack_diff` is just a flattened version of the parents' diffs -- we don't add it
       * twice! */
      Some(_) => vec![],
      /* Convert the `stack_diff` into its own set of possibly-incomplete
       * sub-reconstructions! */
      None => stack_diff
        .iter()
        .flat_map(|step| match step {
          /* NB: "named" steps are only relevant for constructing the interval graph with
           * anonymous steps, which denote the correct `ProdCaseRef` to use, so we
           * discard them here. */
          NamedOrAnonStep::Named(_) => None,
          NamedOrAnonStep::Anon(anon_step) => match anon_step {
            AnonStep::Positive(anon_sym) => {
              let maybe_ref: &UnflattenedProdCaseRef = anon_step_mapping
                .get(anon_sym)
                .expect(format!("no state found for anon sym {:?}", anon_sym).as_str());
              match maybe_ref {
                &UnflattenedProdCaseRef::PassThrough => None,
                &UnflattenedProdCaseRef::Case(ref x) => {
                  Some(InProgressReconstruction::with_elements(vec![
                    ReconstructionElement::Intermediate(
                      DirectionalIntermediateReconstruction::Rightwards(
                        IntermediateReconstruction::empty_for_case(*x),
                      ),
                    ),
                  ]))
                },
              }
            },
            AnonStep::Negative(anon_sym) => {
              let maybe_ref: &UnflattenedProdCaseRef = anon_step_mapping
                .get(anon_sym)
                .expect(format!("no state found for anon sym {:?}", anon_sym).as_str());
              match maybe_ref {
                &UnflattenedProdCaseRef::PassThrough => None,
                &UnflattenedProdCaseRef::Case(ref x) => {
                  Some(InProgressReconstruction::with_elements(vec![
                    ReconstructionElement::Intermediate(
                      DirectionalIntermediateReconstruction::Leftwards(
                        IntermediateReconstruction::empty_for_case(*x),
                      ),
                    ),
                  ]))
                },
              }
            },
          },
        })
        .collect(),
    };
    eprintln!("middle_elements: {:?}", middle_elements);

    InProgressReconstruction::joined(
      vec![prologue]
        .into_iter()
        .chain(middle_elements.into_iter())
        .chain(vec![epilogue])
        .collect(),
    )
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
pub struct CompletedCaseReconstruction {
  pub prod_case: ProdCaseRef,
  pub args: Vec<CompleteSubReconstruction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
pub enum CompleteSubReconstruction {
  State(LoweredState),
  Completed(CompletedCaseReconstruction),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
pub struct CompletedWholeReconstruction(pub Vec<CompleteSubReconstruction>);

impl CompletedWholeReconstruction {
  pub fn new(maybe_completed_constructions: InProgressReconstruction) -> Self {
    let sub_constructions: Vec<_> = maybe_completed_constructions
      .elements
      .into_iter()
      .map(|el| match el {
        ReconstructionElement::Intermediate(_) => {
          unreachable!("expected all sub constructions to be completed!");
        },
        ReconstructionElement::CompletedSub(x) => x,
      })
      .collect();
    CompletedWholeReconstruction(sub_constructions)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    lowering_to_indices::{graph_coordinates::*, mapping_to_tokens::*},
    tests::non_cyclic_productions,
  };

  #[test]
  fn reconstructs_from_parse() {
    let prods = non_cyclic_productions();
    let token_grammar = TokenGrammar::new(&prods);
    let preprocessed_grammar = PreprocessedGrammar::new(&token_grammar);
    let string_input = "ab";
    let input = Input(string_input.chars().collect());
    let parseable_grammar = ParseableGrammar::new::<char>(preprocessed_grammar.clone(), &input);

    let mut parse = Parse::initialize_with_trees_for_adjacent_pairs(&parseable_grammar);

    let spanning_subtree_ref = parse.get_next_parse();
    let reconstructed = InProgressReconstruction::new(spanning_subtree_ref, &parse);
    let completely_reconstructed = CompletedWholeReconstruction::new(reconstructed);
    assert_eq!(
      completely_reconstructed,
      CompletedWholeReconstruction(vec![
        CompleteSubReconstruction::State(LoweredState::Start),
        CompleteSubReconstruction::Completed(CompletedCaseReconstruction {
          prod_case: ProdCaseRef {
            prod: ProdRef(0),
            case: CaseRef(0)
          },
          args: vec![
            CompleteSubReconstruction::State(LoweredState::Within(TokenPosition {
              prod: ProdRef(0),
              case: CaseRef(0),
              case_el: CaseElRef(0)
            })),
            CompleteSubReconstruction::State(LoweredState::Within(TokenPosition {
              prod: ProdRef(0),
              case: CaseRef(0),
              case_el: CaseElRef(1)
            })),
          ]
        }),
        CompleteSubReconstruction::State(LoweredState::End),
      ])
    );

    /* Try it again, crossing productions this time. */
    let longer_string_input = "abab";
    let longer_input = Input(longer_string_input.chars().collect());
    let longer_parseable_grammar =
      ParseableGrammar::new::<char>(preprocessed_grammar, &longer_input);
    let mut longer_parse =
      Parse::initialize_with_trees_for_adjacent_pairs(&longer_parseable_grammar);
    let first_parsed_longer_string = longer_parse.get_next_parse();
    let longer_reconstructed =
      InProgressReconstruction::new(first_parsed_longer_string, &longer_parse);
    let longer_completely_reconstructed = CompletedWholeReconstruction::new(longer_reconstructed);
    assert_eq!(
      longer_completely_reconstructed,
      CompletedWholeReconstruction(vec![
        CompleteSubReconstruction::State(LoweredState::Start),
        CompleteSubReconstruction::Completed(CompletedCaseReconstruction {
          prod_case: ProdCaseRef {
            prod: ProdRef(1),
            case: CaseRef(0),
          },
          args: vec![
            CompleteSubReconstruction::State(LoweredState::Within(TokenPosition::new(1, 0, 0))),
            CompleteSubReconstruction::State(LoweredState::Within(TokenPosition::new(1, 0, 1))),
            CompleteSubReconstruction::Completed(CompletedCaseReconstruction {
              prod_case: ProdCaseRef {
                prod: ProdRef(0),
                case: CaseRef(0),
              },
              args: vec![
                CompleteSubReconstruction::State(LoweredState::Within(TokenPosition::new(0, 0, 0))),
                CompleteSubReconstruction::State(LoweredState::Within(TokenPosition::new(0, 0, 1))),
              ],
            })
          ],
        }),
        CompleteSubReconstruction::State(LoweredState::End),
      ])
    );
  }
}
