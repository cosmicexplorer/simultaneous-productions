/* Copyright (C) 2021 Danny McClanahan <dmcC2@hypnicjerk.ai> */
/* SPDX-License-Identifier: AGPL-3.0 */

//! Implementation of parsing. Performance does *(eventually)* matter here.

use crate::{
  allocation::HandoffAllocable,
  grammar_indexing as gi,
  lowering_to_indices::{grammar_building as gb, graph_coordinates as gc},
  types::{DefaultHasher, Vec},
};

use indexmap::{IndexMap, IndexSet};
use priority_queue::PriorityQueue;

use core::{
  alloc::Allocator,
  cmp, fmt,
  hash::{Hash, Hasher},
};

#[derive(Debug, Clone)]
pub struct Input<Tok, Arena>(pub Vec<Tok, Arena>)
where Arena: Allocator;

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
#[derive(Clone)]
pub struct FlattenedSpanInfo<Arena>
where Arena: Allocator
{
  pub state_pair: gi::StatePair,
  pub input_range: InputRange,
  pub stack_diff: gi::StackDiffSegment<Arena>,
}

impl<Arena> HandoffAllocable for FlattenedSpanInfo<Arena>
where Arena: Allocator+Clone
{
  type Arena = Arena;

  fn allocator_handoff(&self) -> Arena { self.stack_diff.allocator_handoff() }
}

impl<Arena> PartialEq for FlattenedSpanInfo<Arena>
where Arena: Allocator
{
  fn eq(&self, other: &Self) -> bool {
    self.state_pair == other.state_pair
      && self.input_range == other.input_range
      && self.stack_diff == other.stack_diff
  }
}

impl<Arena> Eq for FlattenedSpanInfo<Arena> where Arena: Allocator {}

impl<Arena> fmt::Debug for FlattenedSpanInfo<Arena>
where Arena: Allocator
{
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(
      f,
      "FlattenedSpanInfo {{ state_pair: {:?}, input_range: {:?}, stack_diff: {:?} }}",
      self.state_pair, self.input_range, self.stack_diff
    )
  }
}

impl<Arena> Hash for FlattenedSpanInfo<Arena>
where Arena: Allocator
{
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.state_pair.hash(state);
    self.input_range.hash(state);
    self.stack_diff.hash(state);
  }
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
#[derive(Clone)]
pub struct SpanningSubtreeToCreate<Arena>
where Arena: Allocator
{
  pub input_span: FlattenedSpanInfo<Arena>,
  pub parents: Option<ParentInfo>,
}

impl<Arena> HandoffAllocable for SpanningSubtreeToCreate<Arena>
where Arena: Allocator+Clone
{
  type Arena = Arena;

  fn allocator_handoff(&self) -> Arena { self.input_span.allocator_handoff() }
}

impl<Arena> PartialEq for SpanningSubtreeToCreate<Arena>
where Arena: Allocator
{
  fn eq(&self, other: &Self) -> bool {
    self.input_span == other.input_span && self.parents == other.parents
  }
}

impl<Arena> Eq for SpanningSubtreeToCreate<Arena> where Arena: Allocator {}

impl<Arena> fmt::Debug for SpanningSubtreeToCreate<Arena>
where Arena: Allocator
{
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(
      f,
      "SpanningSubtreeToCreate {{ input_span: {:?}, parents: {:?} }}",
      self.input_span, self.parents
    )
  }
}

impl<Arena> Hash for SpanningSubtreeToCreate<Arena>
where Arena: Allocator
{
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.input_span.hash(state);
    self.parents.hash(state);
  }
}

#[derive(Debug, Clone)]
pub struct CompletelyFlattenedSubtree<Arena>
where Arena: Allocator
{
  pub states: Vec<gi::LoweredState, Arena>,
  pub input_range: InputRange,
}

impl<Arena> PartialEq for CompletelyFlattenedSubtree<Arena>
where Arena: Allocator
{
  fn eq(&self, other: &Self) -> bool {
    self.states == other.states && self.input_range == other.input_range
  }
}

impl<Arena> Eq for CompletelyFlattenedSubtree<Arena> where Arena: Allocator {}

pub trait FlattenableToStates<Arena>
where Arena: Allocator+Clone
{
  fn flatten_to_states(&self, parse: &Parse<Arena>) -> CompletelyFlattenedSubtree<Arena>;
}

#[derive(Clone)]
pub struct SpanningSubtree<Arena>
where Arena: Allocator
{
  pub input_span: FlattenedSpanInfo<Arena>,
  pub parents: Option<ParentInfo>,
  pub id: SpanningSubtreeRef,
}

impl<Arena> HandoffAllocable for SpanningSubtree<Arena>
where Arena: Allocator+Clone
{
  type Arena = Arena;

  fn allocator_handoff(&self) -> Arena { self.input_span.allocator_handoff() }
}

impl<Arena> PartialEq for SpanningSubtree<Arena>
where Arena: Allocator
{
  fn eq(&self, other: &Self) -> bool {
    self.input_span == other.input_span && self.parents == other.parents && self.id == other.id
  }
}

impl<Arena> Eq for SpanningSubtree<Arena> where Arena: Allocator {}

impl<Arena> fmt::Debug for SpanningSubtree<Arena>
where Arena: Allocator
{
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(
      f,
      "SpanningSubtree {{ input_span: {:?}, parents: {:?}, id: {:?} }}",
      self.input_span, self.parents, self.id
    )
  }
}

impl<Arena> Hash for SpanningSubtree<Arena>
where Arena: Allocator
{
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.input_span.hash(state);
    self.parents.hash(state);
    self.id.hash(state);
  }
}

impl<Arena> FlattenableToStates<Arena> for SpanningSubtree<Arena>
where Arena: Allocator+Clone
{
  fn flatten_to_states(&self, parse: &Parse<Arena>) -> CompletelyFlattenedSubtree<Arena> {
    let arena = self.allocator_handoff();
    match self.parents {
      None => {
        let mut states: Vec<_, Arena> = Vec::with_capacity_in(2, arena);
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
        let mut linked_states: Vec<gi::LoweredState, Arena> =
          Vec::with_capacity_in(left_states.len() + right.len(), arena);
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

impl<Arena> SpansRange for SpanningSubtree<Arena>
where Arena: Allocator
{
  fn range(&self) -> InputRange {
    let SpanningSubtree {
      input_span: FlattenedSpanInfo { input_range, .. },
      ..
    } = self;
    *input_range
  }
}

#[derive(Debug, Clone)]
pub struct PossibleStates<Arena>(pub Vec<gi::LoweredState, Arena>)
where Arena: Allocator;

impl<Arena> PartialEq for PossibleStates<Arena>
where Arena: Allocator
{
  fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<Arena> Eq for PossibleStates<Arena> where Arena: Allocator {}

#[derive(Debug, Clone)]
pub struct ParseableGrammar<Arena>
where Arena: Allocator+Clone
{
  pub input_as_states: Vec<PossibleStates<Arena>, Arena>,
  pub pairwise_state_transition_table:
    IndexMap<gi::StatePair, Vec<gi::StackDiffSegment<Arena>, Arena>, Arena, DefaultHasher>,
  pub anon_step_mapping: IndexMap<gi::AnonSym, gi::UnflattenedProdCaseRef, Arena, DefaultHasher>,
}

impl<Arena> HandoffAllocable for ParseableGrammar<Arena>
where Arena: Allocator+Clone
{
  type Arena = Arena;

  fn allocator_handoff(&self) -> Arena { self.anon_step_mapping.arena() }
}

impl<Arena> PartialEq for ParseableGrammar<Arena>
where Arena: Allocator+Clone
{
  fn eq(&self, other: &Self) -> bool {
    self.input_as_states == other.input_as_states
      && self.pairwise_state_transition_table == other.pairwise_state_transition_table
      && self.anon_step_mapping == other.anon_step_mapping
  }
}

impl<Arena> Eq for ParseableGrammar<Arena> where Arena: Allocator+Clone {}

impl<Arena> ParseableGrammar<Arena>
where Arena: Allocator+Clone
{
  /* TODO: get the transitive closure of this to get all the consecutive series
   * of states *over* length 2 and their corresponding stack diffs -- this
   * enables e.g. the use of SIMD instructions to find those series of
   * states! */
  fn connect_stack_diffs(
    transitions: &[gi::CompletedStatePairWithVertices<Arena>],
    arena: Arena,
  ) -> IndexMap<gi::StatePair, Vec<gi::StackDiffSegment<Arena>, Arena>, Arena, DefaultHasher> {
    let mut paired_segments: IndexMap<
      gi::StatePair,
      Vec<gi::StackDiffSegment<Arena>, Arena>,
      Arena,
      DefaultHasher,
    > = IndexMap::new_in(arena);

    for single_transition in transitions.iter() {
      let gi::CompletedStatePairWithVertices {
        state_pair,
        /* FIXME: why do we care about this separate (?) arena? */
        interval: gi::ContiguousNonterminalInterval { interval, arena },
      } = single_transition;

      let mut diff: Vec<_, Arena> = Vec::new_in(arena.clone());
      diff.extend(interval.iter().flat_map(|vtx| vtx.get_step()));

      let cur_entry = paired_segments
        .entry(*state_pair)
        .or_insert_with(|| Vec::new_in(arena.clone()));
      (*cur_entry).push(gi::StackDiffSegment(diff));
    }

    paired_segments
  }

  fn get_possible_states_for_input<Tok>(
    alphabet: &gb::Alphabet<Tok, Arena>,
    mapping: &gb::AlphabetMapping<Arena>,
    input: &Input<Tok, Arena>,
  ) -> Result<Vec<PossibleStates<Arena>, Arena>, ParsingInputFailure<Tok>>
  where
    Tok: Hash+Eq+fmt::Debug+Clone,
  {
    let arena = mapping.allocator_handoff();

    /* NB: Bookend the internal states with Start and End states (creating a
     * vector with 2 more entries than `input`)! */
    let mut st: Vec<_, Arena> = Vec::with_capacity_in(1, arena.clone());
    st.push(gi::LoweredState::Start);

    let mut ps: Vec<PossibleStates<Arena>, Arena> = Vec::new_in(arena.clone());
    ps.push(PossibleStates(st));

    for tok in input.0.iter() {
      let tok_ref = alphabet
        .0
        .retrieve_intern(tok)
        .ok_or_else(|| ParsingInputFailure::UnknownToken(tok.clone()))?;
      let tok_positions = mapping
        .get(tok_ref)
        .ok_or(ParsingInputFailure::UnknownTokRef(tok_ref))?;
      let mut states: Vec<_, Arena> = Vec::with_capacity_in(tok_positions.len(), arena.clone());
      states.extend(
        tok_positions
          .iter()
          .map(|pos| gi::LoweredState::Within(*pos)),
      );
      ps.push(PossibleStates(states));
    }

    let mut end: Vec<_, Arena> = Vec::with_capacity_in(1, arena);
    end.push(gi::LoweredState::End);
    ps.push(PossibleStates(end));

    Ok(ps)
  }

  #[allow(dead_code)]
  pub fn new<Tok>(
    grammar: gi::PreprocessedGrammar<Tok, Arena>,
    input: &Input<Tok, Arena>,
  ) -> Result<Self, ParsingInputFailure<Tok>>
  where
    Tok: Hash+Eq+fmt::Debug+Clone,
  {
    let arena = grammar.allocator_handoff();
    let gi::PreprocessedGrammar {
      cyclic_graph_decomposition:
        gi::CyclicGraphDecomposition {
          pairwise_state_transitions,
          anon_step_mapping,
          ..
        },
      token_states_mapping,
      alphabet,
    } = grammar;
    Ok(ParseableGrammar {
      input_as_states: Self::get_possible_states_for_input(
        &alphabet,
        &token_states_mapping,
        input,
      )?,
      pairwise_state_transition_table: Self::connect_stack_diffs(
        &pairwise_state_transitions,
        arena,
      ),
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
  #[allow(dead_code)]
  UnknownToken(Tok),
  #[allow(dead_code)]
  UnknownTokRef(gc::TokRef),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ParsingFailure {
  NoMoreSpans,
}

#[derive(Debug, Clone)]
pub struct Parse<Arena>
where Arena: Allocator+Clone
{
  pub spans: PriorityQueue<SpanningSubtree<Arena>, usize, Arena, DefaultHasher>,
  pub grammar: ParseableGrammar<Arena>,
  /* TODO: lexicographically sort these! */
  pub finishes_at_left: IndexMap<
    InputTokenIndex,
    IndexSet<SpanningSubtree<Arena>, Arena, DefaultHasher>,
    Arena,
    DefaultHasher,
  >,
  pub finishes_at_right: IndexMap<
    InputTokenIndex,
    IndexSet<SpanningSubtree<Arena>, Arena, DefaultHasher>,
    Arena,
    DefaultHasher,
  >,
  pub spanning_subtree_table: Vec<SpanningSubtree<Arena>, Arena>,
}

impl<Arena> HandoffAllocable for Parse<Arena>
where Arena: Allocator+Clone
{
  type Arena = Arena;

  fn allocator_handoff(&self) -> Arena { self.grammar.allocator_handoff() }
}

impl<Arena> Parse<Arena>
where Arena: Allocator+Clone
{
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
  fn new(grammar: ParseableGrammar<Arena>) -> Self {
    let arena = grammar.allocator_handoff();
    Parse {
      spans: PriorityQueue::new_in(arena.clone()),
      grammar,
      finishes_at_left: IndexMap::new_in(arena.clone()),
      finishes_at_right: IndexMap::new_in(arena.clone()),
      spanning_subtree_table: Vec::new_in(arena),
    }
  }

  fn add_spanning_subtree(&mut self, span: &SpanningSubtreeToCreate<Arena>) {
    let arena = span.allocator_handoff();
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
      .or_insert_with(|| IndexSet::new_in(arena.clone()));
    (*left_entry).insert(new_span.clone());
    let right_entry = self
      .finishes_at_right
      .entry(right_index)
      .or_insert_with(|| IndexSet::new_in(arena.clone()));
    (*right_entry).insert(new_span.clone());

    self.spans.push(new_span.clone(), new_span.range().width());
  }

  fn generate_subtrees_for_pair(
    pair: &gi::StatePair,
    left_index: InputTokenIndex,
    right_index: InputTokenIndex,
    diffs: Vec<gi::StackDiffSegment<Arena>, Arena>,
  ) -> IndexSet<SpanningSubtreeToCreate<Arena>, Arena, DefaultHasher> {
    let arena = diffs.allocator().clone();
    let gi::StatePair { left, right } = pair;
    let mut ret: IndexSet<SpanningSubtreeToCreate<Arena>, Arena, DefaultHasher> =
      IndexSet::with_capacity_in(diffs.len(), arena);
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

  #[allow(dead_code)]
  pub fn initialize_with_trees_for_adjacent_pairs(grammar: ParseableGrammar<Arena>) -> Self {
    let arena = grammar.allocator_handoff();

    let mut parse = Self::new(grammar.clone());

    let ParseableGrammar {
      input_as_states,
      pairwise_state_transition_table,
      ..
    } = grammar;

    for (i, left_states) in input_as_states.iter().cloned().enumerate() {
      assert!(i <= input_as_states.len());
      if i >= input_as_states.len() - 1 {
        break;
      }
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
            .unwrap_or_else(|| Vec::new_in(arena.clone()));

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
    left_diff: gi::StackDiffSegment<Arena>,
    right_diff: gi::StackDiffSegment<Arena>,
  ) -> Option<gi::StackDiffSegment<Arena>> {
    let arena = left_diff.allocator_handoff();

    let gi::StackDiffSegment(left_steps) = left_diff;
    let gi::StackDiffSegment(right_steps) = right_diff;

    /* "Compatibility" is checked by seeing whether the stack steps up to the
     * minimum length of both either cancel each other out, or are the same
     * polarity. */
    let min_length: usize = cmp::min(left_steps.len(), right_steps.len());

    /* To get the same number of elements in both left and right, we reverse the
     * left, take off some elements, then reverse it back. */
    let mut rev_left: Vec<gi::NamedOrAnonStep, Arena> =
      Vec::with_capacity_in(left_steps.len(), arena.clone());
    rev_left.extend(left_steps.into_iter().rev());

    /* NB: We keep the left zippered elements reversed so that we compare stack
     * elements outward from the center along both the left and right
     * sides. */
    let mut cmp_left: Vec<gi::NamedOrAnonStep, Arena> =
      Vec::with_capacity_in(min_length, arena.clone());
    cmp_left.extend(rev_left.iter().cloned().take(min_length));

    let mut cmp_right: Vec<gi::NamedOrAnonStep, Arena> =
      Vec::with_capacity_in(min_length, arena.clone());
    cmp_right.extend(right_steps.iter().cloned().take(min_length));

    let mut leftover_left: Vec<gi::NamedOrAnonStep, Arena> = Vec::new_in(arena.clone());
    leftover_left.extend(rev_left.iter().cloned().skip(min_length).rev());

    let mut leftover_right: Vec<gi::NamedOrAnonStep, Arena> = Vec::new_in(arena.clone());
    leftover_right.extend(right_steps.iter().cloned().skip(min_length));
    assert!(leftover_left.is_empty() || leftover_right.is_empty());

    let mut connected: Vec<gi::NamedOrAnonStep, Arena> = Vec::new_in(arena.clone());
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
    let mut all_steps: Vec<gi::NamedOrAnonStep, Arena> = Vec::with_capacity_in(
      leftover_left.len() + connected.len() + leftover_right.len(),
      arena,
    );
    all_steps.extend(
      leftover_left
        .into_iter()
        .chain(connected.into_iter())
        .chain(leftover_right.into_iter()),
    );

    Some(gi::StackDiffSegment(all_steps))
  }

  pub fn get_spanning_subtree(
    &self,
    span_ref: SpanningSubtreeRef,
  ) -> Option<&SpanningSubtree<Arena>> {
    self.spanning_subtree_table.get(span_ref.0)
  }

  #[allow(dead_code)]
  pub fn advance(&mut self) -> Result<ParseResult, ParsingFailure> {
    /* dbg!(&self.spans); */
    /* dbg!(&self.finishes_at_left); */
    /* dbg!(&self.finishes_at_right); */
    let arena = self.allocator_handoff();
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
        .unwrap_or_else(|| IndexSet::new_in(arena.clone()))
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
      for left_neighbor in maybe_set
        .cloned()
        .unwrap_or_else(|| IndexSet::new_in(arena.clone()))
        .iter()
      {
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
    grammar_indexing as gi,
    lowering_to_indices::grammar_building as gb,
    test_framework::{new_token_position, non_cyclic_productions},
    types::Global,
  };

  #[test]
  fn dynamic_parse_state() {
    let prods = non_cyclic_productions();

    let token_grammar = gb::TokenGrammar::new(prods, Global).unwrap();
    let preprocessed_grammar = gi::PreprocessedGrammar::new(token_grammar);
    let string_input = "ab";
    let input = Input(string_input.chars().collect());
    let parseable_grammar = ParseableGrammar::new::<char>(preprocessed_grammar, &input).unwrap();

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
      .collect::<IndexMap<gi::StatePair, Vec<gi::StackDiffSegment<Global>>>>()
    );

    let mut parse = Parse::initialize_with_trees_for_adjacent_pairs(parseable_grammar.clone());
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
    let all_spans: Vec<SpanningSubtree<Global>> =
      spans.into_iter().map(|(x, _)| x.clone()).collect();

    fn get_span(all_spans: &Vec<SpanningSubtree<Global>>, index: usize) -> SpanningSubtree<Global> {
      all_spans.get(index).unwrap().clone()
    }

    fn collect_spans(
      all_spans: &Vec<SpanningSubtree<Global>>,
      indices: Vec<usize>,
    ) -> IndexSet<SpanningSubtree<Global>, Global, DefaultHasher> {
      indices
        .into_iter()
        .map(|x| get_span(all_spans, x))
        .collect()
    }

    /* NB: These explicit type ascriptions are necessary for some reason... */
    let expected_at_left: IndexMap<
      InputTokenIndex,
      IndexSet<SpanningSubtree<Global>, Global, DefaultHasher>,
      Global,
      DefaultHasher,
    > = [
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

    let expected_at_right: IndexMap<
      InputTokenIndex,
      IndexSet<SpanningSubtree<Global>, Global, DefaultHasher>,
      Global,
      DefaultHasher,
    > = [
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
}
