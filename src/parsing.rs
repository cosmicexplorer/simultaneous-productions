/* Copyright (C) 2021 Danny McClanahan <dmcC2@hypnicjerk.ai> */
/* SPDX-License-Identifier: GPL-3.0 */

//! Implementation of parsing. Performance /does/ (eventually) matter here.

use crate::{grammar_indexing::*, lowering_to_indices::graph_coordinates::*, token::*};

use indexmap::{IndexMap, IndexSet};
use priority_queue::PriorityQueue;

#[derive(Debug, Clone)]
pub struct Input<Tok: Token>(pub Vec<Tok>);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct InputTokenIndex(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct InputRange {
  pub left_index: InputTokenIndex,
  pub right_index: InputTokenIndex,
}

impl InputRange {
  pub fn new(left_index: InputTokenIndex, right_index: InputTokenIndex) -> Self {
    assert!(left_index.0 < right_index.0);
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

/* A flattened version of the information in a `SpanningSubtree`. */
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FlattenedSpanInfo {
  pub state_pair: StatePair,
  pub input_range: InputRange,
  pub stack_diff: StackDiffSegment,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SpanningSubtreeRef(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ParentInfo {
  pub left_parent: SpanningSubtreeRef,
  pub right_parent: SpanningSubtreeRef,
}

/* We want to have a consistent `id` within each `SpanningSubtree`, so we add
 * new trees via a specific method which assigns them an id. */
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpanningSubtreeToCreate {
  pub input_span: FlattenedSpanInfo,
  pub parents: Option<ParentInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompletelyFlattenedSubtree {
  pub states: Vec<LoweredState>,
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
      None => CompletelyFlattenedSubtree {
        states: vec![
          self.input_span.state_pair.left,
          self.input_span.state_pair.right,
        ],
        input_range: self.input_span.input_range,
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
          .flatten_to_states(&parse);
        let CompletelyFlattenedSubtree {
          states: right_states,
          input_range: right_range,
        } = parse
          .get_spanning_subtree(right_parent)
          .unwrap()
          .flatten_to_states(&parse);
        dbg!(&left_states);
        dbg!(&left_range);
        dbg!(&right_states);
        dbg!(&right_range);
        dbg!(&self.input_span);
        /* If the left range *ends* with the same state the right range *starts*
         * with, then we can merge the left and right paths to get a new
         * valid path through the state space. */
        assert_eq!(left_range.right_index.0, right_range.left_index.0);
        assert_eq!(left_states.last(), right_states.first());
        let linked_states: Vec<LoweredState> = left_states
          .into_iter()
          .chain(right_states[1..].into_iter().cloned())
          .collect();
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
pub struct PossibleStates(pub Vec<LoweredState>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseableGrammar {
  pub input_as_states: Vec<PossibleStates>,
  pub pairwise_state_transition_table: IndexMap<StatePair, Vec<StackDiffSegment>>,
  pub anon_step_mapping: IndexMap<AnonSym, UnflattenedProdCaseRef>,
}

impl ParseableGrammar {
  /* TODO: get the transitive closure of this to get all the consecutive series
   * of states *over* length 2 and their corresponding stack diffs -- this
   * enables e.g. the use of SIMD instructions to find those series of
   * states! */
  fn connect_stack_diffs(
    transitions: &Vec<CompletedStatePairWithVertices>,
  ) -> IndexMap<StatePair, Vec<StackDiffSegment>> {
    let mut paired_segments: IndexMap<StatePair, Vec<StackDiffSegment>> = IndexMap::new();

    for single_transition in transitions.iter() {
      let CompletedStatePairWithVertices {
        state_pair,
        interval: ContiguousNonterminalInterval(interval),
      } = single_transition;

      let diff: Vec<_> = interval.iter().flat_map(|vtx| vtx.get_step()).collect();

      let cur_entry = paired_segments.entry(*state_pair).or_insert(vec![]);
      (*cur_entry).push(StackDiffSegment(diff));
    }

    paired_segments
  }

  fn get_possible_states_for_input<Tok: Token>(
    mapping: &IndexMap<Tok, Vec<TokenPosition>>,
    input: &Input<Tok>,
  ) -> Vec<PossibleStates> {
    /* NB: Bookend the internal states with Start and End states (creating a
     * vector with 2 more entries than `input`)! */
    vec![PossibleStates(vec![LoweredState::Start])]
      .into_iter()
      .chain(input.0.iter().map(|tok| {
        mapping
          .get(tok)
          .map(|positions| {
            let states: Vec<_> = positions
              .iter()
              .map(|pos| LoweredState::Within(*pos))
              .collect();
            PossibleStates(states)
          })
          .expect(format!("no tokens found for token {:?} in input {:?}", tok, input).as_str())
      }))
      .chain(vec![PossibleStates(vec![LoweredState::End])])
      .collect()
  }

  pub fn new<Tok: Token>(grammar: PreprocessedGrammar<Tok>, input: &Input<Tok>) -> Self {
    let PreprocessedGrammar {
      cyclic_graph_decomposition:
        CyclicGraphDecomposition {
          pairwise_state_transitions,
          anon_step_mapping,
          ..
        },
      token_states_mapping,
    } = grammar;
    ParseableGrammar {
      input_as_states: Self::get_possible_states_for_input(&token_states_mapping, input),
      pairwise_state_transition_table: Self::connect_stack_diffs(&pairwise_state_transitions),
      anon_step_mapping,
    }
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ParseResult {
  Incomplete,
  Complete(SpanningSubtreeRef),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParsingFailure(String);

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
  fn new(grammar: &ParseableGrammar) -> Self {
    Parse {
      spans: PriorityQueue::new(),
      grammar: grammar.clone(),
      finishes_at_left: IndexMap::new(),
      finishes_at_right: IndexMap::new(),
      spanning_subtree_table: vec![],
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
      parents: span.parents.clone(),
      id: new_ref_id,
    };
    self.spanning_subtree_table.push(new_span.clone());

    let left_entry = self
      .finishes_at_left
      .entry(left_index)
      .or_insert_with(IndexSet::new);
    (*left_entry).insert(new_span.clone());
    let right_entry = self
      .finishes_at_right
      .entry(right_index)
      .or_insert_with(IndexSet::new);
    (*right_entry).insert(new_span.clone());

    self
      .spans
      .push(new_span.clone(), new_span.clone().range().width());
  }

  fn generate_subtrees_for_pair(
    pair: &StatePair,
    left_index: InputTokenIndex,
    right_index: InputTokenIndex,
    diffs: Vec<StackDiffSegment>,
  ) -> IndexSet<SpanningSubtreeToCreate> {
    let StatePair { left, right } = pair;
    diffs
      .into_iter()
      .map(|stack_diff| SpanningSubtreeToCreate {
        input_span: FlattenedSpanInfo {
          state_pair: StatePair {
            left: *left,
            right: *right,
          },
          input_range: InputRange::new(left_index, right_index),
          /* TODO: lexicographically sort these??? */
          stack_diff: stack_diff.clone(),
        },
        parents: None,
      })
      .collect()
  }

  pub fn initialize_with_trees_for_adjacent_pairs(grammar: &ParseableGrammar) -> Self {
    let ParseableGrammar {
      input_as_states,
      pairwise_state_transition_table,
      ..
    } = grammar;

    let mut parse = Self::new(grammar);

    for (i, left_states) in input_as_states.iter().cloned().enumerate() {
      assert!(i <= input_as_states.len());
      if i >= input_as_states.len() - 1 {
        break;
      }
      let right_states = input_as_states.get(i + 1).unwrap();
      for left in left_states.0.iter() {
        for right in right_states.0.iter() {
          let pair = StatePair {
            left: *left,
            right: *right,
          };
          let stack_diffs = pairwise_state_transition_table
            .get(&pair)
            .cloned()
            .unwrap_or_else(Vec::new);

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
    left_diff: StackDiffSegment,
    right_diff: StackDiffSegment,
  ) -> Option<StackDiffSegment> {
    let StackDiffSegment(left_steps) = left_diff;
    let StackDiffSegment(right_steps) = right_diff;

    /* "Compatibility" is checked by seeing whether the stack steps up to the
     * minimum length of both either cancel each other out, or are the same
     * polarity. */
    let min_length = vec![left_steps.len(), right_steps.len()]
      .into_iter()
      .min()
      .unwrap();
    /* To get the same number of elements in both left and right, we reverse the
     * left, take off some elements, then reverse it back. */
    let rev_left: Vec<_> = left_steps.into_iter().rev().collect();

    /* NB: We keep the left zippered elements reversed so that we compare stack
     * elements outward from the center along both the left and right
     * sides. */
    let cmp_left: Vec<_> = rev_left.iter().cloned().take(min_length).collect();
    let cmp_right: Vec<_> = right_steps.iter().cloned().take(min_length).collect();

    let leftover_left: Vec<_> = rev_left.iter().cloned().skip(min_length).rev().collect();
    let leftover_right: Vec<_> = right_steps.iter().cloned().skip(min_length).collect();
    assert!(leftover_left.is_empty() || leftover_right.is_empty());

    let mut connected: Vec<NamedOrAnonStep> = vec![];
    for (i, left_step, right_step) in (0..min_length).map(|i| {
      (
        i,
        cmp_left.get(i).unwrap().clone(),
        cmp_right.get(i).unwrap().clone(),
      )
    }) {
      match left_step.sequence(right_step) {
        Ok(x) => {
          if x.len() == 2 {
            connected = cmp_left[(i + 1)..min_length]
              .iter()
              .cloned()
              .rev()
              .chain(x)
              .chain(cmp_right[(i + 1)..min_length].iter().cloned())
              .collect();
            break;
          } else if x.len() == 0 {
            ()
          } else {
            panic!("unidentified sequence of stack steps: {:?}", x)
          }
        },
        Err(_) => {
          return None;
        },
      }
    }

    /* Put the leftover left and right on the left and right of the resulting
     * stack steps! */
    let all_steps: Vec<_> = leftover_left
      .into_iter()
      .chain(connected)
      .chain(leftover_right.into_iter())
      .collect();
    Some(StackDiffSegment(all_steps))
  }

  pub fn get_spanning_subtree(&self, span_ref: SpanningSubtreeRef) -> Option<&SpanningSubtree> {
    self.spanning_subtree_table.get(span_ref.0)
  }

  pub fn advance(&mut self) -> Result<ParseResult, ParsingFailure> {
    dbg!(&self.spans);
    dbg!(&self.finishes_at_left);
    dbg!(&self.finishes_at_right);
    let maybe_front = self.spans.pop();
    if let Some((cur_span, _priority)) = maybe_front {
      let SpanningSubtree {
        input_span:
          FlattenedSpanInfo {
            state_pair:
              StatePair {
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

      dbg!(&cur_span);

      /* TODO: ensure all entries of `.finishes_at_left` and `.finishes_at_right`
       * are lexicographically sorted! */
      /* Check all right-neighbors for compatible stack diffs. */
      for right_neighbor in self
        .finishes_at_left
        .get(&InputTokenIndex(cur_right_index))
        .cloned()
        .unwrap_or_else(IndexSet::new)
        .iter()
      {
        dbg!(&right_neighbor);
        let SpanningSubtree {
          input_span:
            FlattenedSpanInfo {
              state_pair:
                StatePair {
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
              state_pair: StatePair {
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

      dbg!(cur_left_index);
      /* Check all left-neighbors for compatible stack diffs. */
      let maybe_set = if cur_left_index == 0 {
        None
      } else {
        self.finishes_at_right.get(&InputTokenIndex(cur_left_index))
      };
      for left_neighbor in maybe_set.cloned().unwrap_or_else(IndexSet::new).iter() {
        dbg!(&left_neighbor);
        let SpanningSubtree {
          input_span:
            FlattenedSpanInfo {
              state_pair:
                StatePair {
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
              state_pair: StatePair {
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

      dbg!((&cur_left, &cur_right, &cur_stack_diff));

      /* Check if we now span across the whole input! */
      /* NB: It's RIDICULOUS how simple this check is!!! */
      match (cur_left, cur_right, &cur_stack_diff) {
        (LoweredState::Start, LoweredState::End, &StackDiffSegment(ref stack_diff))
          if stack_diff.is_empty() =>
        {
          Ok(ParseResult::Complete(cur_span.id))
        },
        _ => Ok(ParseResult::Incomplete),
      }
    } else {
      Err(ParsingFailure("no more spans to iterate over!".to_string()))
    }
  }
}
