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
  grammar_indexing as gi,
  grammar_specification::{self as gs, graphviz as gv},
  lowering_to_indices::{grammar_building as gb, graph_coordinates as gc},
};

use indexmap::{IndexMap, IndexSet};
use priority_queue::PriorityQueue;

use core::{
  cmp, fmt,
  hash::{Hash, Hasher},
};

#[derive(Debug, Clone)]
pub struct Input<Tok>(pub Vec<Tok>);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InputTokenIndex(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InputRange {
  left_index: InputTokenIndex,
  right_index: InputTokenIndex,
}

impl InputRange {
  /// Ensure the `left_index` is less than or equal to the `right_index`.
  ///
  /// **TODO: is this correct? Or strictly less? Why?**
  pub fn new(left_index: InputTokenIndex, right_index: InputTokenIndex) -> Self {
    assert!(left_index.0 <= right_index.0);
    InputRange {
      left_index,
      right_index,
    }
  }

  fn width(&self) -> usize { self.right_index.0 - self.left_index.0 }
}

trait SpansRange {
  fn range(&self) -> InputRange;
}

/// A flattened version of the information in a [SpanningSubtree].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FlattenedSpanInfo {
  pub state_pair: gi::StatePair,
  pub input_range: InputRange,
  pub stack_diff: gi::StackDiffSegment,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpanningSubtreeRef(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParentInfo {
  pub left_parent: SpanningSubtreeRef,
  pub right_parent: SpanningSubtreeRef,
}

/// We want to have a consistent `id` within each [SpanningSubtree], so we add
/// new trees via a specific method which assigns them an id.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpanningSubtreeToCreate {
  pub input_span: FlattenedSpanInfo,
  pub parents: Option<ParentInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletelyFlattenedSubtree {
  pub states: Vec<gi::LoweredState>,
  pub input_range: InputRange,
}

pub trait FlattenableToStates {
  fn flatten_to_states(&self, parse: &Parse) -> CompletelyFlattenedSubtree;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpanningSubtree {
  pub input_span: FlattenedSpanInfo,
  pub parents: Option<ParentInfo>,
  pub id: SpanningSubtreeRef,
}

impl FlattenableToStates for SpanningSubtree {
  fn flatten_to_states(&self, parse: &Parse) -> CompletelyFlattenedSubtree {
    match self.parents {
      None => {
        let mut states: Vec<_> = Vec::with_capacity(2);
        states.extend_from_slice(&[
          self.input_span.state_pair.left,
          self.input_span.state_pair.right,
        ]);
        CompletelyFlattenedSubtree {
          states,
          input_range: self.input_span.input_range,
        }
      },
      Some(ParentInfo {
        left_parent,
        right_parent,
      }) => {
        let CompletelyFlattenedSubtree {
          states: left_states,
          input_range: left_range,
        } = parse
          .get_spanning_subtree(left_parent)
          .unwrap()
          .flatten_to_states(parse);
        let CompletelyFlattenedSubtree {
          states: right_states,
          input_range: right_range,
        } = parse
          .get_spanning_subtree(right_parent)
          .unwrap()
          .flatten_to_states(parse);
        /* dbg!(&left_states); */
        /* dbg!(&left_range); */
        /* dbg!(&right_states); */
        /* dbg!(&right_range); */
        /* dbg!(&self.input_span); */
        /* If the left range *ends* with the same state the right range *starts*
         * with, then we can merge the left and right paths to get a new
         * valid path through the state space. */
        assert_eq!(left_range.right_index.0, right_range.left_index.0);
        assert_eq!(left_states.last(), right_states.first());
        let right = &right_states[1..];
        let mut linked_states: Vec<gi::LoweredState> =
          Vec::with_capacity(left_states.len() + right.len());
        linked_states.extend_from_slice(&left_states);
        linked_states.extend_from_slice(right);
        CompletelyFlattenedSubtree {
          states: linked_states,
          input_range: InputRange::new(left_range.left_index, right_range.right_index),
        }
      },
    }
  }
}

impl SpansRange for SpanningSubtree {
  fn range(&self) -> InputRange {
    let SpanningSubtree {
      input_span: FlattenedSpanInfo { input_range, .. },
      ..
    } = self;
    *input_range
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PossibleStates(pub Vec<gi::LoweredState>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseableGrammar {
  pub input_as_states: Vec<PossibleStates>,
  pub pairwise_state_transition_table: IndexMap<gi::StatePair, Vec<gi::StackDiffSegment>>,
  /// Provide available stack cycles to the parse engine.
  pub cyclic_subgraph: gi::EpsilonNodeStateSubgraph,
  /* TODO: remove this; it appears to only be used in the (broken) parse reconstruction code. */
  pub anon_step_mapping: IndexMap<gi::AnonSym, gi::UnflattenedProdCaseRef>,
}

impl ParseableGrammar {
  pub fn build_dot_graph(self) -> gv::GraphBuilder {
    let mut gb = gv::GraphBuilder::new();

    let Self {
      input_as_states,
      pairwise_state_transition_table,
      ..
    } = self;

    let mut state_vertices: IndexMap<gi::LoweredState, gv::Vertex> = IndexMap::new();

    /* (A) Draw out all the states and transitions between them. */
    {
      todo!("parsing clearly has never worked");
    }

    gb
  }

  /* TODO: get the transitive closure of this to get all the consecutive series
   * of states *over* length 2 and their corresponding stack diffs -- this
   * enables e.g. the use of SIMD instructions to find those series of
   * states! */
  fn connect_stack_diffs(
    transitions: &[gi::CompletedStatePairWithVertices],
  ) -> IndexMap<gi::StatePair, Vec<gi::StackDiffSegment>> {
    let mut paired_segments: IndexMap<gi::StatePair, Vec<gi::StackDiffSegment>> = IndexMap::new();

    for single_transition in transitions.iter() {
      let gi::CompletedStatePairWithVertices {
        state_pair,
        interval: gi::ContiguousNonterminalInterval { interval },
      } = single_transition;

      let mut diff: Vec<_> = Vec::new();
      diff.extend(interval.iter().flat_map(|vtx| vtx.get_step()));

      let cur_entry = paired_segments
        .entry(*state_pair)
        .or_insert_with(|| Vec::new());
      (*cur_entry).push(gi::StackDiffSegment(diff));
    }

    paired_segments
  }

  fn get_possible_states_for_input<Tok>(
    tokens: &gb::InternedLookupTable<Tok, gc::TokRef>,
    input: &Input<Tok>,
  ) -> Result<Vec<PossibleStates>, ParsingInputFailure<Tok>>
  where
    Tok: crate::grammar_specification::constraints::Hashable+fmt::Debug+Clone,
  {
    /* NB: Bookend the internal states with Start and End states (creating a
     * vector with 2 more entries than `input`)! */
    let mut st: Vec<_> = Vec::with_capacity(1);
    st.push(gi::LoweredState::Start);

    let mut ps: Vec<PossibleStates> = Vec::new();
    ps.push(PossibleStates(st));

    for tok in input.0.iter() {
      let tok_ref = tokens
        .retrieve_intern(tok)
        .ok_or_else(|| ParsingInputFailure::UnknownToken(tok.clone()))?;
      let tok_positions = tokens
        .get(tok_ref)
        .ok_or(ParsingInputFailure::UnknownTokRef(tok_ref))?;
      let mut states: Vec<_> = Vec::with_capacity(tok_positions.len());
      states.extend(
        tok_positions
          .iter()
          .map(|pos| gi::LoweredState::Within(*pos)),
      );
      ps.push(PossibleStates(states));
    }

    let mut end: Vec<_> = Vec::with_capacity(1);
    end.push(gi::LoweredState::End);
    ps.push(PossibleStates(end));

    Ok(ps)
  }

  pub fn new<Tok>(
    grammar: gi::PreprocessedGrammar<Tok>,
    input: &Input<Tok>,
  ) -> Result<Self, ParsingInputFailure<Tok>>
  where
    Tok: gs::constraints::Hashable+fmt::Debug+Clone,
  {
    let gi::PreprocessedGrammar {
      cyclic_graph_decomposition:
        gi::CyclicGraphDecomposition {
          cyclic_subgraph,
          pairwise_state_transitions,
          anon_step_mapping,
        },
      token_states_mapping,
    } = grammar;
    Ok(ParseableGrammar {
      input_as_states: Self::get_possible_states_for_input(&token_states_mapping, input)?,
      pairwise_state_transition_table: Self::connect_stack_diffs(&pairwise_state_transitions),
      cyclic_subgraph,
      anon_step_mapping,
    })
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ParseResult {
  Incomplete,
  Complete(SpanningSubtreeRef),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ParsingInputFailure<Tok> {
  UnknownToken(Tok),
  UnknownTokRef(gc::TokRef),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ParsingFailure {
  NoMoreSpans,
}

#[derive(Debug, Clone)]
pub struct Parse {
  pub spans: PriorityQueue<SpanningSubtree, usize>,
  pub grammar: ParseableGrammar,
  /* TODO: lexicographically sort these! */
  pub finishes_at_left: IndexMap<InputTokenIndex, IndexSet<SpanningSubtree>>,
  pub finishes_at_right: IndexMap<InputTokenIndex, IndexSet<SpanningSubtree>>,
  pub spanning_subtree_table: Vec<SpanningSubtree>,
}

impl Parse {
  #[cfg(test)]
  pub fn get_next_parse(&mut self) -> SpanningSubtreeRef {
    loop {
      match self.advance() {
        Ok(ParseResult::Incomplete) => (),
        Ok(ParseResult::Complete(tree_ref)) => {
          return tree_ref;
        },
        Err(e) => panic!("{:?}", e),
      }
    }
  }

  /* NB: Intentionally private! */
  fn new(grammar: ParseableGrammar) -> Self {
    Parse {
      spans: PriorityQueue::new(),
      grammar,
      finishes_at_left: IndexMap::new(),
      finishes_at_right: IndexMap::new(),
      spanning_subtree_table: Vec::new(),
    }
  }

  fn add_spanning_subtree(&mut self, span: &SpanningSubtreeToCreate) {
    let SpanningSubtreeToCreate {
      input_span:
        FlattenedSpanInfo {
          input_range: InputRange {
            left_index,
            right_index,
          },
          ..
        },
      ..
    } = span.clone();

    let new_ref_id = SpanningSubtreeRef(self.spanning_subtree_table.len());
    let new_span = SpanningSubtree {
      input_span: span.input_span.clone(),
      parents: span.parents,
      id: new_ref_id,
    };
    self.spanning_subtree_table.push(new_span.clone());

    let left_entry = self
      .finishes_at_left
      .entry(left_index)
      .or_insert_with(|| IndexSet::new());
    (*left_entry).insert(new_span.clone());
    let right_entry = self
      .finishes_at_right
      .entry(right_index)
      .or_insert_with(|| IndexSet::new());
    (*right_entry).insert(new_span.clone());

    self.spans.push(new_span.clone(), new_span.range().width());
  }

  fn generate_subtrees_for_pair(
    pair: &gi::StatePair,
    left_index: InputTokenIndex,
    right_index: InputTokenIndex,
    diffs: Vec<gi::StackDiffSegment>,
  ) -> IndexSet<SpanningSubtreeToCreate> {
    let gi::StatePair { left, right } = pair;
    let mut ret: IndexSet<SpanningSubtreeToCreate> = IndexSet::with_capacity(diffs.len());
    ret.extend(diffs.into_iter().map(|stack_diff| SpanningSubtreeToCreate {
      input_span: FlattenedSpanInfo {
        state_pair: gi::StatePair {
          left: *left,
          right: *right,
        },
        input_range: InputRange::new(left_index, right_index),
        /* TODO: lexicographically sort these??? */
        stack_diff,
      },
      parents: None,
    }));
    ret
  }

  pub fn initialize_with_trees_for_adjacent_pairs(grammar: ParseableGrammar) -> Self {
    let mut parse = Self::new(grammar.clone());

    let ParseableGrammar {
      input_as_states,
      pairwise_state_transition_table,
      ..
    } = grammar;


    let states_to_take_len: usize = input_as_states.len() - 1;
    for (i, left_states) in input_as_states.iter().take(states_to_take_len).enumerate() {
      let right_states = input_as_states.get(i + 1).unwrap();
      for left in left_states.0.iter() {
        for right in right_states.0.iter() {
          let pair = gi::StatePair {
            left: *left,
            right: *right,
          };
          let stack_diffs = pairwise_state_transition_table
            .get(&pair)
            .cloned()
            .unwrap_or_else(|| Vec::new());

          for new_tree in Self::generate_subtrees_for_pair(
            &pair,
            InputTokenIndex(i),
            InputTokenIndex(i + 1),
            stack_diffs,
          )
          .into_iter()
          {
            parse.add_spanning_subtree(&new_tree);
          }
        }
      }
    }

    parse
  }

  /* Given two adjacent stack diffs, check whether they are compatible, and if
   * so, return the resulting stack diff from joining them. */
  fn stack_diff_pair_zipper(
    left_diff: gi::StackDiffSegment,
    right_diff: gi::StackDiffSegment,
  ) -> Option<gi::StackDiffSegment> {
    let gi::StackDiffSegment(left_steps) = left_diff;
    let gi::StackDiffSegment(right_steps) = right_diff;

    /* "Compatibility" is checked by seeing whether the stack steps up to the
     * minimum length of both either cancel each other out, or are the same
     * polarity. */
    let min_length: usize = cmp::min(left_steps.len(), right_steps.len());

    /* To get the same number of elements in both left and right, we reverse the
     * left, take off some elements, then reverse it back. */
    let mut rev_left: Vec<gi::NamedOrAnonStep> = Vec::with_capacity(left_steps.len());
    rev_left.extend(left_steps.into_iter().rev());

    /* NB: We keep the left zippered elements reversed so that we compare stack
     * elements outward from the center along both the left and right
     * sides. */
    let mut cmp_left: Vec<gi::NamedOrAnonStep> = Vec::with_capacity(min_length);
    cmp_left.extend(rev_left.iter().cloned().take(min_length));

    let mut cmp_right: Vec<gi::NamedOrAnonStep> = Vec::with_capacity(min_length);
    cmp_right.extend(right_steps.iter().cloned().take(min_length));

    let mut leftover_left: Vec<gi::NamedOrAnonStep> = Vec::new();
    leftover_left.extend(rev_left.iter().cloned().skip(min_length).rev());

    let mut leftover_right: Vec<gi::NamedOrAnonStep> = Vec::new();
    leftover_right.extend(right_steps.iter().cloned().skip(min_length));
    assert!(leftover_left.is_empty() || leftover_right.is_empty());

    let mut connected: Vec<gi::NamedOrAnonStep> = Vec::new();
    for i in 0..min_length {
      match cmp_left[i].sequence(cmp_right[i]) {
        Ok(None) => (),
        Ok(Some((left_step, right_step))) => {
          connected.extend(cmp_left[(i + 1)..min_length].iter().cloned().rev());
          connected.push(left_step);
          connected.push(right_step);
          connected.extend(cmp_right[(i + 1)..min_length].iter().cloned());
          /* FIXME: why just break here? */
          break;
        },
        Err(_) => {
          /* unreachable!("when does this happen? {:?}", e); */
          return None;
        },
      }
    }

    /* Put the leftover left and right on the left and right of the resulting
     * stack steps! */
    let mut all_steps: Vec<gi::NamedOrAnonStep> =
      Vec::with_capacity(leftover_left.len() + connected.len() + leftover_right.len());
    all_steps.extend(
      leftover_left
        .into_iter()
        .chain(connected.into_iter())
        .chain(leftover_right.into_iter()),
    );

    Some(gi::StackDiffSegment(all_steps))
  }

  pub fn get_spanning_subtree(&self, span_ref: SpanningSubtreeRef) -> Option<&SpanningSubtree> {
    self.spanning_subtree_table.get(span_ref.0)
  }

  pub fn advance(&mut self) -> Result<ParseResult, ParsingFailure> {
    /* dbg!(&self.spans); */
    /* dbg!(&self.finishes_at_left); */
    /* dbg!(&self.finishes_at_right); */
    let maybe_front = self.spans.pop();
    if let Some((cur_span, _priority)) = maybe_front {
      let SpanningSubtree {
        input_span:
          FlattenedSpanInfo {
            state_pair:
              gi::StatePair {
                left: cur_left,
                right: cur_right,
              },
            input_range:
              InputRange {
                left_index: InputTokenIndex(cur_left_index),
                right_index: InputTokenIndex(cur_right_index),
              },
            stack_diff: cur_stack_diff,
          },
        ..
      } = cur_span.clone();

      /* dbg!(&cur_span); */

      /* TODO: ensure all entries of `.finishes_at_left` and `.finishes_at_right`
       * are lexicographically sorted! */
      /* Check all right-neighbors for compatible stack diffs. */
      for right_neighbor in self
        .finishes_at_left
        .get(&InputTokenIndex(cur_right_index))
        .cloned()
        .unwrap_or_else(|| IndexSet::new())
        .iter()
      {
        /* dbg!(&right_neighbor); */
        let SpanningSubtree {
          input_span:
            FlattenedSpanInfo {
              state_pair:
                gi::StatePair {
                  left: _right_left,
                  right: right_right,
                },
              input_range:
                InputRange {
                  left_index: InputTokenIndex(right_left_index),
                  right_index: InputTokenIndex(right_right_index),
                },
              stack_diff: right_stack_diff,
            },
          ..
        } = right_neighbor.clone();
        assert_eq!(right_left_index, cur_right_index);

        if let Some(merged_diff) =
          Self::stack_diff_pair_zipper(cur_stack_diff.clone(), right_stack_diff)
        {
          let new_tree = SpanningSubtreeToCreate {
            input_span: FlattenedSpanInfo {
              state_pair: gi::StatePair {
                left: cur_left,
                right: right_right,
              },
              input_range: InputRange {
                left_index: InputTokenIndex(cur_left_index),
                right_index: InputTokenIndex(right_right_index),
              },
              stack_diff: merged_diff,
            },
            parents: Some(ParentInfo {
              left_parent: cur_span.id,
              right_parent: right_neighbor.id,
            }),
          };
          self.add_spanning_subtree(&new_tree);
        }
      }

      /* dbg!(cur_left_index); */
      /* Check all left-neighbors for compatible stack diffs. */
      let maybe_set = if cur_left_index == 0 {
        None
      } else {
        self.finishes_at_right.get(&InputTokenIndex(cur_left_index))
      };
      for left_neighbor in maybe_set.cloned().unwrap_or_else(|| IndexSet::new()).iter() {
        /* dbg!(&left_neighbor); */
        let SpanningSubtree {
          input_span:
            FlattenedSpanInfo {
              state_pair:
                gi::StatePair {
                  left: left_left,
                  right: _left_right,
                },
              input_range:
                InputRange {
                  left_index: InputTokenIndex(left_left_index),
                  right_index: InputTokenIndex(left_right_index),
                },
              stack_diff: left_stack_diff,
            },
          ..
        } = left_neighbor.clone();
        assert_eq!(left_right_index, cur_left_index);

        if let Some(merged_diff) =
          Self::stack_diff_pair_zipper(left_stack_diff, cur_stack_diff.clone())
        {
          let new_tree = SpanningSubtreeToCreate {
            input_span: FlattenedSpanInfo {
              state_pair: gi::StatePair {
                left: left_left,
                right: cur_right,
              },
              input_range: InputRange {
                left_index: InputTokenIndex(left_left_index),
                right_index: InputTokenIndex(cur_right_index),
              },
              stack_diff: merged_diff,
            },
            parents: Some(ParentInfo {
              left_parent: left_neighbor.id,
              right_parent: cur_span.id,
            }),
          };
          self.add_spanning_subtree(&new_tree);
        }
      }

      /* dbg!((&cur_left, &cur_right, &cur_stack_diff)); */

      /* Check if we now span across the whole input! */
      /* NB: It's RIDICULOUS how simple this check is!!! */
      match (cur_left, cur_right, &cur_stack_diff) {
        (gi::LoweredState::Start, gi::LoweredState::End, &gi::StackDiffSegment(ref stack_diff))
          if stack_diff.is_empty() =>
        {
          Ok(ParseResult::Complete(cur_span.id))
        },
        _ => Ok(ParseResult::Incomplete),
      }
    } else {
      Err(ParsingFailure::NoMoreSpans)
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    grammar_indexing as gi, state,
    test_framework::{basic_productions, new_token_position, non_cyclic_productions},
  };

  #[test]
  fn dynamic_parse_state() {
    let prods = non_cyclic_productions();

    let detokenized = state::preprocessing::Init(prods).try_index().unwrap();
    let indexed = detokenized.index();
    let string_input = "ab";
    let input = Input(string_input.chars().collect());
    let parseable_grammar: ParseableGrammar = indexed.attach_input(&input).unwrap().0;

    assert_eq!(
      parseable_grammar.input_as_states.clone(),
      [
        PossibleStates([gi::LoweredState::Start].as_ref().to_vec()),
        PossibleStates(
          [
            gi::LoweredState::Within(new_token_position(0, 0, 0)),
            gi::LoweredState::Within(new_token_position(1, 0, 0)),
            gi::LoweredState::Within(new_token_position(1, 1, 1)),
          ]
          .as_ref()
          .to_vec()
        ),
        PossibleStates(
          [
            gi::LoweredState::Within(new_token_position(0, 0, 1)),
            gi::LoweredState::Within(new_token_position(1, 0, 1)),
          ]
          .as_ref()
          .to_vec()
        ),
        PossibleStates([gi::LoweredState::End].as_ref().to_vec()),
      ]
      .as_ref()
      .to_vec()
    );

    assert_eq!(
      parseable_grammar.pairwise_state_transition_table.clone(),
      [
        (
          gi::StatePair {
            left: gi::LoweredState::Start,
            right: gi::LoweredState::Within(new_token_position(0, 0, 0)),
          },
          [
            gi::StackDiffSegment(
              [
                gi::NamedOrAnonStep::Named(gi::StackStep::Positive(gi::StackSym(gc::ProdRef(0)))),
                gi::NamedOrAnonStep::Anon(gi::AnonStep::Positive(gi::AnonSym(0))),
              ]
              .as_ref()
              .to_vec()
            ),
            gi::StackDiffSegment(
              [
                gi::NamedOrAnonStep::Named(gi::StackStep::Positive(gi::StackSym(gc::ProdRef(1)))),
                gi::NamedOrAnonStep::Anon(gi::AnonStep::Positive(gi::AnonSym(3))),
                gi::NamedOrAnonStep::Anon(gi::AnonStep::Positive(gi::AnonSym(4))),
                gi::NamedOrAnonStep::Named(gi::StackStep::Positive(gi::StackSym(gc::ProdRef(0)))),
                gi::NamedOrAnonStep::Anon(gi::AnonStep::Positive(gi::AnonSym(0))),
              ]
              .as_ref()
              .to_vec()
            ),
          ]
          .as_ref()
          .to_vec(),
        ),
        (
          gi::StatePair {
            left: gi::LoweredState::Start,
            right: gi::LoweredState::Within(new_token_position(1, 0, 0)),
          },
          [gi::StackDiffSegment(
            [
              gi::NamedOrAnonStep::Named(gi::StackStep::Positive(gi::StackSym(gc::ProdRef(1)))),
              gi::NamedOrAnonStep::Anon(gi::AnonStep::Positive(gi::AnonSym(1))),
            ]
            .as_ref()
            .to_vec()
          )]
          .as_ref()
          .to_vec(),
        ),
        (
          gi::StatePair {
            left: gi::LoweredState::Within(new_token_position(0, 0, 0)),
            right: gi::LoweredState::Within(new_token_position(0, 0, 1)),
          },
          [gi::StackDiffSegment([].as_ref().to_vec()),]
            .as_ref()
            .to_vec(),
        ),
        (
          gi::StatePair {
            left: gi::LoweredState::Within(new_token_position(1, 0, 0)),
            right: gi::LoweredState::Within(new_token_position(1, 0, 1)),
          },
          [gi::StackDiffSegment([].as_ref().to_vec()),]
            .as_ref()
            .to_vec(),
        ),
        (
          gi::StatePair {
            left: gi::LoweredState::Within(new_token_position(1, 1, 1)),
            right: gi::LoweredState::End,
          },
          [gi::StackDiffSegment(
            [
              gi::NamedOrAnonStep::Anon(gi::AnonStep::Negative(gi::AnonSym(3))),
              gi::NamedOrAnonStep::Named(gi::StackStep::Negative(gi::StackSym(gc::ProdRef(1)))),
            ]
            .as_ref()
            .to_vec()
          )]
          .as_ref()
          .to_vec(),
        ),
        (
          gi::StatePair {
            left: gi::LoweredState::Within(new_token_position(0, 0, 1)),
            right: gi::LoweredState::End,
          },
          [
            gi::StackDiffSegment(
              [
                gi::NamedOrAnonStep::Anon(gi::AnonStep::Negative(gi::AnonSym(0))),
                gi::NamedOrAnonStep::Named(gi::StackStep::Negative(gi::StackSym(gc::ProdRef(0)))),
              ]
              .as_ref()
              .to_vec()
            ),
            gi::StackDiffSegment(
              [
                gi::NamedOrAnonStep::Anon(gi::AnonStep::Negative(gi::AnonSym(0))),
                gi::NamedOrAnonStep::Named(gi::StackStep::Negative(gi::StackSym(gc::ProdRef(0)))),
                gi::NamedOrAnonStep::Anon(gi::AnonStep::Negative(gi::AnonSym(2))),
                gi::NamedOrAnonStep::Anon(gi::AnonStep::Negative(gi::AnonSym(1))),
                gi::NamedOrAnonStep::Named(gi::StackStep::Negative(gi::StackSym(gc::ProdRef(1)))),
              ]
              .as_ref()
              .to_vec()
            ),
          ]
          .as_ref()
          .to_vec()
        ),
        (
          gi::StatePair {
            left: gi::LoweredState::Within(new_token_position(0, 0, 1)),
            right: gi::LoweredState::Within(new_token_position(1, 1, 1)),
          },
          [gi::StackDiffSegment(
            [
              gi::NamedOrAnonStep::Anon(gi::AnonStep::Negative(gi::AnonSym(0))),
              gi::NamedOrAnonStep::Named(gi::StackStep::Negative(gi::StackSym(gc::ProdRef(0)))),
              gi::NamedOrAnonStep::Anon(gi::AnonStep::Negative(gi::AnonSym(4))),
            ]
            .as_ref()
            .to_vec()
          )]
          .as_ref()
          .to_vec()
        ),
        (
          gi::StatePair {
            left: gi::LoweredState::Within(new_token_position(1, 0, 1)),
            right: gi::LoweredState::Within(new_token_position(0, 0, 0)),
          },
          [gi::StackDiffSegment(
            [
              gi::NamedOrAnonStep::Anon(gi::AnonStep::Positive(gi::AnonSym(2))),
              gi::NamedOrAnonStep::Named(gi::StackStep::Positive(gi::StackSym(gc::ProdRef(0)))),
              gi::NamedOrAnonStep::Anon(gi::AnonStep::Positive(gi::AnonSym(0))),
            ]
            .as_ref()
            .to_vec()
          )]
          .as_ref()
          .to_vec()
        ),
      ]
      .as_ref()
      .to_vec()
      .into_iter()
      .collect::<IndexMap<gi::StatePair, Vec<gi::StackDiffSegment>>>()
    );

    let mut parse = state::active::Ready::new(parseable_grammar.clone())
      .initialize_parse()
      .0;
    let Parse {
      spans,
      grammar: new_parseable_grammar,
      finishes_at_left,
      finishes_at_right,
      spanning_subtree_table,
    } = parse.clone();
    assert_eq!(new_parseable_grammar, parseable_grammar);

    assert_eq!(
      spans
        .iter()
        .map(|(x, y)| (x.clone(), y.clone()))
        .collect::<Vec<_>>(),
      [
        (
          SpanningSubtree {
            input_span: FlattenedSpanInfo {
              state_pair: gi::StatePair {
                left: gi::LoweredState::Start,
                right: gi::LoweredState::Within(new_token_position(0, 0, 0))
              },
              input_range: InputRange {
                left_index: InputTokenIndex(0),
                right_index: InputTokenIndex(1)
              },
              stack_diff: gi::StackDiffSegment(
                [
                  gi::NamedOrAnonStep::Named(gi::StackStep::Positive(gi::StackSym(gc::ProdRef(0)))),
                  gi::NamedOrAnonStep::Anon(gi::AnonStep::Positive(gi::AnonSym(0))),
                ]
                .as_ref()
                .to_vec()
              ),
            },
            parents: None,
            id: SpanningSubtreeRef(0)
          },
          1
        ),
        (
          SpanningSubtree {
            input_span: FlattenedSpanInfo {
              state_pair: gi::StatePair {
                left: gi::LoweredState::Start,
                right: gi::LoweredState::Within(new_token_position(0, 0, 0))
              },
              input_range: InputRange {
                left_index: InputTokenIndex(0),
                right_index: InputTokenIndex(1)
              },
              stack_diff: gi::StackDiffSegment(
                [
                  gi::NamedOrAnonStep::Named(gi::StackStep::Positive(gi::StackSym(gc::ProdRef(1)))),
                  gi::NamedOrAnonStep::Anon(gi::AnonStep::Positive(gi::AnonSym(3))),
                  gi::NamedOrAnonStep::Anon(gi::AnonStep::Positive(gi::AnonSym(4))),
                  gi::NamedOrAnonStep::Named(gi::StackStep::Positive(gi::StackSym(gc::ProdRef(0)))),
                  gi::NamedOrAnonStep::Anon(gi::AnonStep::Positive(gi::AnonSym(0))),
                ]
                .as_ref()
                .to_vec()
              )
            },
            parents: None,
            id: SpanningSubtreeRef(1)
          },
          1
        ),
        (
          SpanningSubtree {
            input_span: FlattenedSpanInfo {
              state_pair: gi::StatePair {
                left: gi::LoweredState::Start,
                right: gi::LoweredState::Within(new_token_position(1, 0, 0))
              },
              input_range: InputRange {
                left_index: InputTokenIndex(0),
                right_index: InputTokenIndex(1)
              },
              stack_diff: gi::StackDiffSegment(
                [
                  gi::NamedOrAnonStep::Named(gi::StackStep::Positive(gi::StackSym(gc::ProdRef(1)))),
                  gi::NamedOrAnonStep::Anon(gi::AnonStep::Positive(gi::AnonSym(1))),
                ]
                .as_ref()
                .to_vec()
              ),
            },
            parents: None,
            id: SpanningSubtreeRef(2)
          },
          1
        ),
        (
          SpanningSubtree {
            input_span: FlattenedSpanInfo {
              state_pair: gi::StatePair {
                left: gi::LoweredState::Within(new_token_position(0, 0, 0)),
                right: gi::LoweredState::Within(new_token_position(0, 0, 1))
              },
              input_range: InputRange {
                left_index: InputTokenIndex(1),
                right_index: InputTokenIndex(2)
              },
              stack_diff: gi::StackDiffSegment([].as_ref().to_vec())
            },
            parents: None,
            id: SpanningSubtreeRef(3)
          },
          1
        ),
        (
          SpanningSubtree {
            input_span: FlattenedSpanInfo {
              state_pair: gi::StatePair {
                left: gi::LoweredState::Within(new_token_position(1, 0, 0)),
                right: gi::LoweredState::Within(new_token_position(1, 0, 1))
              },
              input_range: InputRange {
                left_index: InputTokenIndex(1),
                right_index: InputTokenIndex(2)
              },
              stack_diff: gi::StackDiffSegment([].as_ref().to_vec())
            },
            parents: None,
            id: SpanningSubtreeRef(4)
          },
          1
        ),
        (
          SpanningSubtree {
            input_span: FlattenedSpanInfo {
              state_pair: gi::StatePair {
                left: gi::LoweredState::Within(new_token_position(0, 0, 1)),
                right: gi::LoweredState::End
              },
              input_range: InputRange {
                left_index: InputTokenIndex(2),
                right_index: InputTokenIndex(3)
              },
              stack_diff: gi::StackDiffSegment(
                [
                  gi::NamedOrAnonStep::Anon(gi::AnonStep::Negative(gi::AnonSym(0))),
                  gi::NamedOrAnonStep::Named(gi::StackStep::Negative(gi::StackSym(gc::ProdRef(0)))),
                ]
                .as_ref()
                .to_vec()
              )
            },
            parents: None,
            id: SpanningSubtreeRef(5)
          },
          1
        ),
        (
          SpanningSubtree {
            input_span: FlattenedSpanInfo {
              state_pair: gi::StatePair {
                left: gi::LoweredState::Within(new_token_position(0, 0, 1)),
                right: gi::LoweredState::End
              },
              input_range: InputRange {
                left_index: InputTokenIndex(2),
                right_index: InputTokenIndex(3)
              },
              stack_diff: gi::StackDiffSegment(
                [
                  gi::NamedOrAnonStep::Anon(gi::AnonStep::Negative(gi::AnonSym(0))),
                  gi::NamedOrAnonStep::Named(gi::StackStep::Negative(gi::StackSym(gc::ProdRef(0)))),
                  gi::NamedOrAnonStep::Anon(gi::AnonStep::Negative(gi::AnonSym(2))),
                  gi::NamedOrAnonStep::Anon(gi::AnonStep::Negative(gi::AnonSym(1))),
                  gi::NamedOrAnonStep::Named(gi::StackStep::Negative(gi::StackSym(gc::ProdRef(1))))
                ]
                .as_ref()
                .to_vec()
              ),
            },
            parents: None,
            id: SpanningSubtreeRef(6)
          },
          1
        )
      ]
      .as_ref()
      .to_vec()
    );
    let all_spans: Vec<SpanningSubtree> = spans.into_iter().map(|(x, _)| x.clone()).collect();

    fn get_span(all_spans: &Vec<SpanningSubtree>, index: usize) -> SpanningSubtree {
      all_spans.get(index).unwrap().clone()
    }

    fn collect_spans(
      all_spans: &Vec<SpanningSubtree>,
      indices: Vec<usize>,
    ) -> IndexSet<SpanningSubtree> {
      indices
        .into_iter()
        .map(|x| get_span(all_spans, x))
        .collect()
    }

    /* NB: These explicit type ascriptions are necessary for some reason... */
    let expected_at_left: IndexMap<InputTokenIndex, IndexSet<SpanningSubtree>> = [
      (
        InputTokenIndex(0),
        collect_spans(&all_spans, [0, 1, 2].as_ref().to_vec()),
      ),
      (
        InputTokenIndex(1),
        collect_spans(&all_spans, [3, 4].as_ref().to_vec()),
      ),
      (
        InputTokenIndex(2),
        collect_spans(&all_spans, [5, 6].as_ref().to_vec()),
      ),
    ]
    .iter()
    .cloned()
    .collect();
    assert_eq!(finishes_at_left, expected_at_left);

    let expected_at_right: IndexMap<InputTokenIndex, IndexSet<SpanningSubtree>> = [
      (
        InputTokenIndex(1),
        collect_spans(&all_spans, [0, 1, 2].as_ref().to_vec()),
      ),
      (
        InputTokenIndex(2),
        collect_spans(&all_spans, [3, 4].as_ref().to_vec()),
      ),
      (
        InputTokenIndex(3),
        collect_spans(&all_spans, [5, 6].as_ref().to_vec()),
      ),
    ]
    .iter()
    .cloned()
    .collect();
    assert_eq!(finishes_at_right, expected_at_right);

    assert_eq!(spanning_subtree_table, all_spans.clone());

    let orig_num_subtrees = parse.spanning_subtree_table.len();
    assert_eq!(parse.advance(), Ok(ParseResult::Incomplete));
    assert_eq!(parse.spanning_subtree_table.len(), orig_num_subtrees + 2);
    assert_eq!(parse.advance(), Ok(ParseResult::Incomplete));
    assert_eq!(parse.spanning_subtree_table.len(), orig_num_subtrees + 4);

    let expected_first_new_subtree = SpanningSubtree {
      input_span: FlattenedSpanInfo {
        state_pair: gi::StatePair {
          left: gi::LoweredState::Start,
          right: gi::LoweredState::End,
        },
        input_range: InputRange::new(InputTokenIndex(0), InputTokenIndex(3)),
        stack_diff: gi::StackDiffSegment(Vec::new()),
      },
      parents: Some(ParentInfo {
        left_parent: SpanningSubtreeRef(7),
        right_parent: SpanningSubtreeRef(5),
      }),
      id: SpanningSubtreeRef(9),
    };

    let expected_subtree = SpanningSubtree {
      input_span: FlattenedSpanInfo {
        state_pair: gi::StatePair {
          left: gi::LoweredState::Start,
          right: gi::LoweredState::End,
        },
        input_range: InputRange::new(InputTokenIndex(0), InputTokenIndex(3)),
        stack_diff: gi::StackDiffSegment(
          [
            gi::NamedOrAnonStep::Anon(gi::AnonStep::Negative(gi::AnonSym(2))),
            gi::NamedOrAnonStep::Anon(gi::AnonStep::Negative(gi::AnonSym(1))),
            gi::NamedOrAnonStep::Named(gi::StackStep::Negative(gi::StackSym(gc::ProdRef(1)))),
          ]
          .as_ref()
          .to_vec(),
        ),
      },
      parents: Some(ParentInfo {
        left_parent: SpanningSubtreeRef(7),
        right_parent: SpanningSubtreeRef(6),
      }),
      id: SpanningSubtreeRef(10),
    };
    assert_eq!(parse.spanning_subtree_table.last(), Some(&expected_subtree));
    assert_eq!(
      parse.get_spanning_subtree(SpanningSubtreeRef(10)),
      Some(&expected_subtree)
    );

    assert_eq!(
      parse.advance(),
      Ok(ParseResult::Complete(SpanningSubtreeRef(9)))
    );
    assert_eq!(
      parse.get_spanning_subtree(SpanningSubtreeRef(9)),
      Some(&expected_first_new_subtree),
    );
    assert_eq!(
      expected_first_new_subtree.flatten_to_states(&parse),
      CompletelyFlattenedSubtree {
        states: [
          gi::LoweredState::Start,
          gi::LoweredState::Within(new_token_position(0, 0, 0)),
          gi::LoweredState::Within(new_token_position(0, 0, 1)),
          gi::LoweredState::End,
        ]
        .as_ref()
        .to_vec(),
        input_range: InputRange::new(InputTokenIndex(0), InputTokenIndex(3)),
      }
    );

    let mut hit_end: bool = false;
    while !hit_end {
      match parse.advance() {
        Ok(ParseResult::Incomplete) => (),
        /* NB: `expected_subtree` at SpanningSubtreeRef(10) has a non-empty stack diff, so it
         * shouldn't be counted as a complete parse! We verify that here. */
        Ok(ParseResult::Complete(SpanningSubtreeRef(i))) => assert!(i != 10),
        Err(_) => {
          hit_end = true;
          break;
        },
      }
    }
    assert!(hit_end);
  }

  #[test]
  fn non_cyclic_parse_graphvis() {
    let prods = non_cyclic_productions();

    let detokenized = state::preprocessing::Init(prods).try_index().unwrap();
    let indexed = detokenized.index();
    let string_input = "ab";
    let input = Input(string_input.chars().collect());
    let parseable_grammar: ParseableGrammar = indexed.attach_input(&input).unwrap().0;

    let gb = parseable_grammar.build_dot_graph();
    let gv::DotOutput(output) = gb.build(gv::Id("test_graph".to_string()));

    assert_eq!(output, "asdf");
  }

  #[test]
  fn basic_parse_graphvis() {
    let prods = basic_productions();

    let detokenized = state::preprocessing::Init(prods).try_index().unwrap();
    let indexed = detokenized.index();
    let string_input = "abc";
    let input = Input(string_input.chars().collect());
    let parseable_grammar: ParseableGrammar = indexed.attach_input(&input).unwrap().0;

    let gb = parseable_grammar.build_dot_graph();
    let gv::DotOutput(output) = gb.build(gv::Id("test_graph".to_string()));

    assert_eq!(output, "asdf");
  }
}
