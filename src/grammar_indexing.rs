/* Copyright (C) 2021 Danny McClanahan <dmcC2@hypnicjerk.ai> */
/* SPDX-License-Identifier: GPL-3.0 */

//! Implementation for getting a [super::grammar_indexing::PreprocessedGrammar].
//!
//! Performance doesn't matter here.

use crate::{
  lowering_to_indices::{graph_coordinates::*, graph_representation::*, mapping_to_tokens::*},
  token::*,
};

use indexmap::{IndexMap, IndexSet};
use typename::TypeName;

use std::{
  collections::{HashSet, VecDeque},
  hash::{Hash, Hasher},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, TypeName)]
pub struct ProdCaseRef {
  pub prod: ProdRef,
  pub case: CaseRef,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, TypeName)]
pub enum UnflattenedProdCaseRef {
  PassThrough,
  Case(ProdCaseRef),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct StackSym(pub ProdRef);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum UnsignedStep {
  Named(StackSym),
  Anon(AnonSym),
}

trait AsUnsignedStep {
  fn as_unsigned_step(&self) -> UnsignedStep;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Polarity {
  Positive,
  Negative,
}

trait Polar {
  fn polarity(&self) -> Polarity;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StackStep {
  Positive(StackSym),
  Negative(StackSym),
}

impl Polar for StackStep {
  fn polarity(&self) -> Polarity {
    match self {
      Self::Positive(_) => Polarity::Positive,
      Self::Negative(_) => Polarity::Negative,
    }
  }
}

impl AsUnsignedStep for StackStep {
  fn as_unsigned_step(&self) -> UnsignedStep {
    match self {
      Self::Positive(sym) => UnsignedStep::Named(*sym),
      Self::Negative(sym) => UnsignedStep::Named(*sym),
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StackStepError(String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NamedOrAnonStep {
  Named(StackStep),
  Anon(AnonStep),
}

impl Polar for NamedOrAnonStep {
  fn polarity(&self) -> Polarity {
    match self {
      Self::Named(step) => step.polarity(),
      Self::Anon(step) => step.polarity(),
    }
  }
}

impl AsUnsignedStep for NamedOrAnonStep {
  fn as_unsigned_step(&self) -> UnsignedStep {
    match self {
      Self::Named(step) => step.as_unsigned_step(),
      Self::Anon(step) => step.as_unsigned_step(),
    }
  }
}

impl NamedOrAnonStep {
  pub fn sequence(self, other: Self) -> Result<Vec<NamedOrAnonStep>, StackStepError> {
    if self.polarity() == other.polarity() {
      Ok(vec![self, other])
    } else if self.as_unsigned_step() == other.as_unsigned_step() {
      Ok(vec![])
    } else {
      Err(StackStepError(format!(
        "failed to sequence steps {:?} and {:?}",
        self, other
      )))
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StackDiffSegment(pub Vec<NamedOrAnonStep>);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TrieNodeRef(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackTrieNode {
  pub stack_diff: StackDiffSegment,
  /* During parsing, the top of the stack will be a named or anonymous symbol. We can negate
   * that (it should always be a positive step on the top of the stack, so a negative
   * step, I think) to get a NamedOrAnonStep which can index into the relevant segments.
   * This supports stack cycles, as well as using an Rc<StackTrieNode> to manage state
   * during the parse. TODO: make a "build" method that removes the RefCell, coalesces
   * stack diffs, and makes the next nodes an IndexMap (??? on the last part given lex
   * BFS?!)! */
  pub next_nodes: IndexSet<StackTrieNextEntry>,
  /* Doubly-linked so that they can be traversed from either direction -- this is (maybe) key
   * to parallelism in parsing! */
  pub prev_nodes: IndexSet<StackTrieNextEntry>,
}

impl Hash for StackTrieNode {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.stack_diff.hash(state);
    self.next_nodes.iter().collect::<Vec<_>>().hash(state);
    self.prev_nodes.iter().collect::<Vec<_>>().hash(state);
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StackTrieNextEntry {
  Completed(LoweredState),
  Incomplete(TrieNodeRef),
}

impl StackTrieNode {
  fn bare(vtx: &EpsilonGraphVertex) -> Self {
    StackTrieNode {
      stack_diff: StackDiffSegment(vtx.get_step().map_or(vec![], |s| vec![s])),
      next_nodes: IndexSet::new(),
      prev_nodes: IndexSet::new(),
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpsilonNodeStateSubgraph {
  pub vertex_mapping: IndexMap<EpsilonGraphVertex, TrieNodeRef>,
  pub trie_node_universe: Vec<StackTrieNode>,
}

impl EpsilonNodeStateSubgraph {
  fn new() -> Self {
    EpsilonNodeStateSubgraph {
      vertex_mapping: IndexMap::new(),
      trie_node_universe: vec![],
    }
  }

  fn get_trie(&mut self, node_ref: TrieNodeRef) -> &mut StackTrieNode {
    let TrieNodeRef(node_index) = node_ref;
    /* NB: This does a .get_mut().unwrap() under the hood, so it is still
     * bounds-checked! */
    &mut self.trie_node_universe[node_index]
  }

  fn trie_ref_for_vertex(&mut self, vtx: &EpsilonGraphVertex) -> TrieNodeRef {
    let basic_node = StackTrieNode::bare(vtx);
    let trie_node_ref_for_vertex = if let Some(x) = self.vertex_mapping.get(vtx) {
      *x
    } else {
      let next_ref = TrieNodeRef(self.trie_node_universe.len());
      self.trie_node_universe.push(basic_node.clone());
      self.vertex_mapping.insert(*vtx, next_ref);
      next_ref
    };
    /* All trie nodes corresponding to the same vertex should have the same stack
     * diff! */
    assert_eq!(
      self.get_trie(trie_node_ref_for_vertex).stack_diff,
      basic_node.stack_diff
    );
    trie_node_ref_for_vertex
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum LoweredState {
  Start,
  End,
  Within(TokenPosition),
}

impl LoweredState {
  fn from_vertex(vtx: EpsilonGraphVertex) -> Self {
    match vtx {
      EpsilonGraphVertex::Start(_) => LoweredState::Start,
      EpsilonGraphVertex::End(_) => LoweredState::End,
      EpsilonGraphVertex::State(pos) => LoweredState::Within(pos),
      EpsilonGraphVertex::Anon(_) => panic!("an anonymous vertex cannot start an interval!"),
    }
  }
}

// TODO: consider the relationship between populating token transitions in the
// lookbehind cache to some specific depth (e.g. strings of 3, 4, 5 tokens)
// and SIMD type 1 instructions (my notations: meaning recognizing a specific
// contiguous sequence of tokens (bytes)). SIMD type 2 (finding a
// specific token in a longer string of bytes) can already easily be used with
// just token pairs (and others).
// TODO: consider GPU parsing before the above!
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct StatePair {
  pub left: LoweredState,
  pub right: LoweredState,
}

#[cfg(test)]
impl StatePair {
  pub fn new(left: LoweredState, right: LoweredState) -> Self { StatePair { left, right } }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct AnonSym(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AnonStep {
  Positive(AnonSym),
  Negative(AnonSym),
}

impl Polar for AnonStep {
  fn polarity(&self) -> Polarity {
    match self {
      Self::Positive(_) => Polarity::Positive,
      Self::Negative(_) => Polarity::Negative,
    }
  }
}

impl AsUnsignedStep for AnonStep {
  fn as_unsigned_step(&self) -> UnsignedStep {
    match self {
      Self::Positive(anon_sym) => UnsignedStep::Anon(*anon_sym),
      Self::Negative(anon_sym) => UnsignedStep::Anon(*anon_sym),
    }
  }
}

/* Fun fact: I'm pretty sure this /is/ actually an interval graph,
 * describing the continuous strings of terminals in a TokenGrammar! */
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum EpsilonGraphVertex {
  Start(ProdRef),
  End(ProdRef),
  Anon(AnonStep),
  State(TokenPosition),
}

impl EpsilonGraphVertex {
  pub fn get_step(&self) -> Option<NamedOrAnonStep> {
    match self {
      EpsilonGraphVertex::Start(prod_ref) => Some(NamedOrAnonStep::Named(StackStep::Positive(
        StackSym(*prod_ref),
      ))),
      EpsilonGraphVertex::End(prod_ref) => Some(NamedOrAnonStep::Named(StackStep::Negative(
        StackSym(*prod_ref),
      ))),
      EpsilonGraphVertex::Anon(anon_step) => Some(NamedOrAnonStep::Anon(*anon_step)),
      /* NB: This should always be at the end of the "nonterminals"! */
      EpsilonGraphVertex::State(_) => None,
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContiguousNonterminalInterval(pub Vec<EpsilonGraphVertex>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CyclicGraphDecomposition {
  pub cyclic_subgraph: EpsilonNodeStateSubgraph,
  pub pairwise_state_transitions: Vec<CompletedStatePairWithVertices>,
  pub anon_step_mapping: IndexMap<AnonSym, UnflattenedProdCaseRef>,
}

/* Pointers to the appropriate "forests" of stack transitions
 * starting/completing at each state. "starting" and "completing" are
 * mirrored to allow working away at mapping states to input token indices
 * from either direction, which is intended to allow for parallelism. They're
 * not really "forests" because they *will* have cycles except in very simple
 * grammars (CFGs and below, I think? Unclear if the Chomsky hierarchy
 * still applies). */
// TODO: fix the above incorrect docstring!
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpsilonIntervalGraph {
  pub all_intervals: Vec<ContiguousNonterminalInterval>,
  pub anon_step_mapping: IndexMap<AnonSym, UnflattenedProdCaseRef>,
}

impl EpsilonIntervalGraph {
  pub fn find_start_end_indices(&self) -> IndexMap<ProdRef, StartEndEpsilonIntervals> {
    let mut epsilon_subscripts_index: IndexMap<ProdRef, StartEndEpsilonIntervals> = IndexMap::new();
    let EpsilonIntervalGraph { all_intervals, .. } = self;
    for interval in all_intervals.iter() {
      let ContiguousNonterminalInterval(vertices) = interval.clone();
      /* We should always have a start and end node. */
      assert!(vertices.len() >= 2);
      let first = vertices[0];
      match first {
          EpsilonGraphVertex::Start(start_prod_ref) => {
            let intervals_for_this_prod = epsilon_subscripts_index.entry(start_prod_ref)
              .or_insert(StartEndEpsilonIntervals::new());
            (*intervals_for_this_prod).start_epsilons.push(interval.clone());
          },
          EpsilonGraphVertex::End(end_prod_ref) => {
            let intervals_for_this_prod = epsilon_subscripts_index.entry(end_prod_ref)
              .or_insert(StartEndEpsilonIntervals::new());
            (*intervals_for_this_prod).end_epsilons.push(interval.clone());
          },
          _ => unreachable!("the beginning of an interval should always be a start (epsilon) or end (epsilon prime) vertex"),
        }
    }
    epsilon_subscripts_index
  }

  pub fn connect_all_vertices(self) -> CyclicGraphDecomposition {
    let intervals_indexed_by_start_and_end = self.find_start_end_indices();
    let EpsilonIntervalGraph {
      all_intervals,
      anon_step_mapping,
    } = self;

    let mut all_completed_pairs_with_vertices: Vec<CompletedStatePairWithVertices> = vec![];
    /* NB: When finding token transitions, we keep track of which intermediate
     * transition states we've already seen by using this Hash impl. If any
     * stack cycles are detected when performing a single iteration, the
     * `todo` is dropped, but as there may be multiple paths to
     * the same intermediate transition state, we additionally require checking
     * the identity of intermediate transition states to avoid looping
     * forever. */
    let mut seen_transitions: HashSet<IntermediateTokenTransition> = HashSet::new();
    let mut traversal_queue: VecDeque<IntermediateTokenTransition> = all_intervals
      .iter()
      .map(IntermediateTokenTransition::new)
      .collect();
    let mut all_stack_cycles: Vec<SingleStackCycle> = vec![];

    /* Find all the token transitions! */
    while !traversal_queue.is_empty() {
      let cur_transition = traversal_queue.pop_front().unwrap();
      if seen_transitions.contains(&cur_transition) {
        continue;
      }
      seen_transitions.insert(cur_transition.clone());
      let TransitionIterationResult {
        completed,
        todo,
        cycles,
      } = cur_transition.iterate_and_maybe_complete(&intervals_indexed_by_start_and_end);
      all_completed_pairs_with_vertices.extend(completed);
      traversal_queue.extend(todo);
      all_stack_cycles.extend(cycles);
    }

    let merged_stack_cycles: EpsilonNodeStateSubgraph = {
      let mut ret = EpsilonNodeStateSubgraph::new();
      for cycle in all_stack_cycles.into_iter() {
        let SingleStackCycle(vertices) = cycle;
        for (cur_vtx_index, vtx) in vertices.iter().enumerate() {
          let cur_trie_ref = ret.trie_ref_for_vertex(vtx);
          let next_trie_ref = {
            let next_vtx_index = (cur_vtx_index + 1) % vertices.len();
            let next_vertex = vertices[next_vtx_index];
            ret.trie_ref_for_vertex(&next_vertex)
          };
          {
            /* Add a forward link from the current to the next vertex's node. */
            let cur_trie = ret.get_trie(cur_trie_ref);
            cur_trie
              .next_nodes
              .insert(StackTrieNextEntry::Incomplete(next_trie_ref));
          }
          {
            /* Add a back edge from the next to the current. */
            let next_trie = ret.get_trie(next_trie_ref);
            next_trie
              .prev_nodes
              .insert(StackTrieNextEntry::Incomplete(cur_trie_ref));
          }
        }
      }
      ret
    };

    CyclicGraphDecomposition {
      cyclic_subgraph: merged_stack_cycles,
      pairwise_state_transitions: all_completed_pairs_with_vertices,
      anon_step_mapping,
    }
  }
}

/// The intervals of nonterminals which begin at epsilon (start) or epsilon
/// prime (end) for some ProdRef.
///
/// This is only a concept in the interval graph and is flattened to a single
/// epsilon/epsilon prime when the PreprocessedGrammar is finally
/// constructed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StartEndEpsilonIntervals {
  pub start_epsilons: Vec<ContiguousNonterminalInterval>,
  pub end_epsilons: Vec<ContiguousNonterminalInterval>,
}

impl StartEndEpsilonIntervals {
  fn new() -> Self {
    StartEndEpsilonIntervals {
      start_epsilons: vec![],
      end_epsilons: vec![],
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompletedStatePairWithVertices {
  pub state_pair: StatePair,
  pub interval: ContiguousNonterminalInterval,
}

#[cfg(test)]
impl CompletedStatePairWithVertices {
  pub fn new(state_pair: StatePair, interval: ContiguousNonterminalInterval) -> Self {
    CompletedStatePairWithVertices {
      state_pair,
      interval,
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SingleStackCycle(Vec<EpsilonGraphVertex>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct TransitionIterationResult {
  completed: Vec<CompletedStatePairWithVertices>,
  todo: Vec<IntermediateTokenTransition>,
  cycles: Vec<SingleStackCycle>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IntermediateTokenTransition {
  cur_traversal_intermediate_nonterminals: IndexSet<EpsilonGraphVertex>,
  rest_of_interval: Vec<EpsilonGraphVertex>,
}

/// This Hash implementation is stable because the collection types in this
/// struct have a specific ordering.
impl Hash for IntermediateTokenTransition {
  fn hash<H: Hasher>(&self, state: &mut H) {
    for intermediate_vertex in self.cur_traversal_intermediate_nonterminals.iter() {
      intermediate_vertex.hash(state);
    }
    for subsequent_vertex in self.rest_of_interval.iter() {
      subsequent_vertex.hash(state);
    }
  }
}

impl IntermediateTokenTransition {
  fn new(wrapped_interval: &ContiguousNonterminalInterval) -> Self {
    let ContiguousNonterminalInterval(interval) = wrapped_interval;
    /* All intervals have a start and end node. */
    assert!(interval.len() >= 2);
    let start = interval[0];
    let rest_of_interval = interval[1..].to_vec();
    IntermediateTokenTransition {
      cur_traversal_intermediate_nonterminals: vec![start].into_iter().collect(),
      rest_of_interval,
    }
  }

  /// Check for cycles given a vertex `next`.
  ///
  /// This method supports multiple paths to the same vertex, each of which
  /// are a cycle, by pulling out the constituent vertices from the
  /// current set of "intermediate" nonterminals at
  /// [Self::cur_traversal_intermediate_nonterminals].
  fn check_for_cycles(
    &self,
    next: EpsilonGraphVertex,
  ) -> (IndexSet<EpsilonGraphVertex>, Vec<SingleStackCycle>) {
    let mut prev_nonterminals = self.cur_traversal_intermediate_nonterminals.clone();
    let (cur_vtx_ind, was_new_insert) = prev_nonterminals.insert_full(next);
    if was_new_insert {
      (prev_nonterminals, vec![])
    } else {
      /* If we have already seen this vertex, then a cycle was detected! */
      /* The cycle contains the start vertex and all the ones after it. */
      let cycle_elements: Vec<EpsilonGraphVertex> = prev_nonterminals
        .iter()
        .skip(cur_vtx_ind)
        .cloned()
        .collect();
      let cur_cycle = SingleStackCycle(cycle_elements);
      /* Shuffle all the intermediate vertices off, but keep the cycle start
       * vertex. */
      let remaining_elements: IndexSet<EpsilonGraphVertex> = prev_nonterminals
        .into_iter()
        .take(cur_vtx_ind + 1)
        .collect();
      (remaining_elements, vec![cur_cycle])
    }
  }

  /// TODO: document this great method!!!
  fn process_next_vertex(
    &self,
    start: &EpsilonGraphVertex,
    next: EpsilonGraphVertex,
    indexed_intervals: &IndexMap<ProdRef, StartEndEpsilonIntervals>,
    intermediate_nonterminals_for_next_step: IndexSet<EpsilonGraphVertex>,
  ) -> (
    Vec<CompletedStatePairWithVertices>,
    Vec<IntermediateTokenTransition>,
  ) {
    match next {
      /* Complete a transition, but also add more continuing from the start vertex. */
      EpsilonGraphVertex::Start(start_prod_ref) => {
        /* We only have this single next node, since we always start or end at a
         * start or end. */
        assert_eq!(self.rest_of_interval.len(), 1);
        /* NB: In the model we use for state transitions `A`, we never start from an
         * End node or end on a Start node, so we can skip completed paths
         * entirely here. */
        let passthrough_intermediates: Vec<IntermediateTokenTransition> = indexed_intervals
            .get(&start_prod_ref)
            .expect("all `ProdRef`s should have been accounted for when grouping by start and end intervals")
            .start_epsilons
            .iter()
            .map(|ContiguousNonterminalInterval(next_vertices)| {
              IntermediateTokenTransition {
                cur_traversal_intermediate_nonterminals: intermediate_nonterminals_for_next_step.clone(),
                /* Get the rest of the interval without the epsilon node that it starts with. */
                rest_of_interval: next_vertices[1..].to_vec(),
              }
            })
            .collect();
        (vec![], passthrough_intermediates)
      },
      /* Similarly to ending on a Start vertex. */
      EpsilonGraphVertex::End(end_prod_ref) => {
        /* We only have this single next node, since we always start or end at a
         * start or end. */
        assert_eq!(self.rest_of_interval.len(), 1);
        let completed_path_makes_sense = match start {
          EpsilonGraphVertex::State(_) => true,
          EpsilonGraphVertex::Start(_) => true,
          EpsilonGraphVertex::End(_) => false,
          EpsilonGraphVertex::Anon(_) => {
            panic!("an anonymous vertex should not be at the start of an interval!")
          },
        };
        let completed = if completed_path_makes_sense {
          let completed_state_pair = StatePair {
            left: LoweredState::from_vertex(*start),
            right: LoweredState::End,
          };
          let relevant_interval_with_terminals: Vec<EpsilonGraphVertex> =
            intermediate_nonterminals_for_next_step
              .iter()
              .cloned()
              .collect();
          let single_completion = CompletedStatePairWithVertices {
            state_pair: completed_state_pair,
            interval: ContiguousNonterminalInterval(relevant_interval_with_terminals),
          };
          vec![single_completion]
        } else {
          vec![]
        };
        let passthrough_intermediates: Vec<IntermediateTokenTransition> = indexed_intervals
            .get(&end_prod_ref)
            .expect("all `ProdRef`s should have been accounted for when grouping by start and end intervals")
            .end_epsilons
            .iter()
            .map(|ContiguousNonterminalInterval(next_vertices)| {
              IntermediateTokenTransition {
                cur_traversal_intermediate_nonterminals: intermediate_nonterminals_for_next_step.clone(),
                /* Get the rest of the interval without the epsilon node that it starts with. */
                rest_of_interval: next_vertices[1..].to_vec(),
              }
            })
            .collect();
        (completed, passthrough_intermediates)
      },
      /* `next` is the anonymous vertex, which is all we need it for. */
      EpsilonGraphVertex::Anon(_) => (vec![], vec![IntermediateTokenTransition {
        cur_traversal_intermediate_nonterminals: intermediate_nonterminals_for_next_step,
        rest_of_interval: self.rest_of_interval[1..].to_vec(),
      }]),
      /* Similar to start and end, but the `todo` starts off at the state. */
      EpsilonGraphVertex::State(state_pos) => {
        let completed_state_pair = StatePair {
          left: LoweredState::from_vertex(*start),
          right: LoweredState::Within(state_pos),
        };
        let completed_path_makes_sense = match start {
          EpsilonGraphVertex::State(_) => true,
          EpsilonGraphVertex::Start(_) => true,
          EpsilonGraphVertex::End(_) => false,
          EpsilonGraphVertex::Anon(_) => {
            panic!("an anonymous vertex should not be at the start of an interval!")
          },
        };
        let completed = if completed_path_makes_sense {
          let relevant_interval_with_terminals: Vec<EpsilonGraphVertex> =
            intermediate_nonterminals_for_next_step
              .iter()
              .cloned()
              .collect();
          let single_completion = CompletedStatePairWithVertices {
            state_pair: completed_state_pair,
            interval: ContiguousNonterminalInterval(relevant_interval_with_terminals),
          };
          vec![single_completion]
        } else {
          vec![]
        };
        (completed, vec![IntermediateTokenTransition {
          /* NB: starting off /at/ the current state vertex! */
          cur_traversal_intermediate_nonterminals: vec![next].into_iter().collect(),
          rest_of_interval: self.rest_of_interval[1..].to_vec(),
        }])
      },
    }
  }

  fn iterate_and_maybe_complete(
    &self,
    indexed_intervals: &IndexMap<ProdRef, StartEndEpsilonIntervals>,
  ) -> TransitionIterationResult {
    assert!(!self.cur_traversal_intermediate_nonterminals.is_empty());
    let start = self
      .cur_traversal_intermediate_nonterminals
      .iter()
      .next()
      .unwrap();
    assert!(!self.rest_of_interval.is_empty());
    let next = self.rest_of_interval[0];

    let (intermediate_nonterminals_for_next_step, cycles) = self.check_for_cycles(next);
    let (completed, todo) = self.process_next_vertex(
      start,
      next,
      indexed_intervals,
      intermediate_nonterminals_for_next_step,
    );
    TransitionIterationResult {
      completed,
      /* NB: If cycles were detected, don't return any `todo` nodes, as we have already
       * traversed them! */
      todo: if cycles.is_empty() { todo } else { vec![] },
      cycles,
    }
  }
}

// NB: There is no reference to any `TokenGrammar` -- this is intentional, and
// I believe makes it easier to have the runtime we want just fall out of the
// code without too much work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreprocessedGrammar<Tok: Token> {
  // These don't need to be quick to access or otherwise optimized for the algorithm until we
  // create a `Parse` -- these are chosen to reduce redundancy.
  // `M: T -> {Q}`, where `{Q}` is sets of states!
  pub token_states_mapping: IndexMap<Tok, Vec<TokenPosition>>,
  // `A: T x T -> {S}^+_-`, where `{S}^+_-` (LaTeX formatting) is ordered sequences of signed
  // stack symbols!
  pub cyclic_graph_decomposition: CyclicGraphDecomposition,
}

impl<Tok: Token> PreprocessedGrammar<Tok> {
  /* Intended to reduce visual clutter in the implementation of interval
   * production. */
  fn make_pos_neg_anon_steps(
    cur_index: &mut usize,
    anon_step_mapping: &mut IndexMap<AnonSym, UnflattenedProdCaseRef>,
    cur_case: UnflattenedProdCaseRef,
  ) -> (EpsilonGraphVertex, EpsilonGraphVertex) {
    let the_sym = AnonSym(*cur_index);
    *cur_index += 1;
    anon_step_mapping.insert(the_sym, cur_case);
    (
      EpsilonGraphVertex::Anon(AnonStep::Positive(the_sym)),
      EpsilonGraphVertex::Anon(AnonStep::Negative(the_sym)),
    )
  }

  /// TODO: document this great method!!!
  pub fn produce_terminals_interval_graph(grammar: &TokenGrammar<Tok>) -> EpsilonIntervalGraph {
    /* We would like to just accept a LoweredProductions here, but we call this
     * method directly in testing, and without the whole grammar object
     * the type ascription is ambiguous. */
    // TODO: what is "type ascription" referring to here^ lol
    let TokenGrammar {
      graph: production_graph,
      ..
    } = grammar;
    let LoweredProductions(prods) = production_graph;
    /* We would really like to use .flat_map()s here, but it's not clear how to
     * do that while mutating the global `cur_anon_sym_index` value. When
     * `move` is used on the inner loop, the value of `cur_anon_sym_index`
     * mysteriously gets reset, even if `move` is also used on the
     * outer loop. */
    let mut cur_anon_sym_index: usize = 0;
    let mut really_all_intervals: Vec<ContiguousNonterminalInterval> = vec![];
    let mut anon_step_mapping: IndexMap<AnonSym, UnflattenedProdCaseRef> = IndexMap::new();
    for (prod_ind, the_prod) in prods.iter().enumerate() {
      let cur_prod_ref = ProdRef(prod_ind);
      let ProductionImpl(cases) = the_prod;
      for (case_ind, the_case) in cases.iter().enumerate() {
        let cur_case_ref = CaseRef(case_ind);
        let CaseImpl(elements_of_case) = the_case;
        let mut all_intervals_from_this_case: Vec<ContiguousNonterminalInterval> = vec![];
        /* NB: make an anon sym whenever stepping onto a case! */
        let cur_prod_case = ProdCaseRef {
          prod: cur_prod_ref,
          case: cur_case_ref,
        };
        let (pos_case_anon, neg_case_anon) = Self::make_pos_neg_anon_steps(
          &mut cur_anon_sym_index,
          &mut anon_step_mapping,
          UnflattenedProdCaseRef::Case(cur_prod_case),
        );
        let mut cur_elements: Vec<EpsilonGraphVertex> =
          vec![EpsilonGraphVertex::Start(cur_prod_ref), pos_case_anon];
        for (element_of_case_ind, el) in elements_of_case.iter().enumerate() {
          let cur_el_ref = CaseElRef(element_of_case_ind);
          let cur_pos = TokenPosition {
            prod: cur_prod_ref,
            case: cur_case_ref,
            case_el: cur_el_ref,
          };
          match el {
            /* Continue the current interval of nonterminals. */
            CaseEl::Tok(_) => cur_elements.push(EpsilonGraphVertex::State(cur_pos)),
            /* The current case invokes a subproduction, so is split into two intervals
             * here, using anonymous symbols to keep track of where in
             * this case we jumped off of and where we can jump back
             * onto to satisfy this case of this production. */
            CaseEl::Prod(target_subprod_ref) => {
              /* Generate anonymous steps for the current subprod split. */
              let (pos_anon, neg_anon) = Self::make_pos_neg_anon_steps(
                &mut cur_anon_sym_index,
                &mut anon_step_mapping,
                UnflattenedProdCaseRef::PassThrough,
              );

              /* Generate the interval terminating at the current subprod split. */
              let suffix_for_subprod_split = vec![
                pos_anon,
                /* We /end/ this interval with a "start" vertex because this is going
                 * /into/ a subproduction! */
                EpsilonGraphVertex::Start(*target_subprod_ref),
              ];
              /* NB: we empty `cur_elements` here! */
              let interval_upto_subprod = ContiguousNonterminalInterval(
                cur_elements
                  .drain(..)
                  .chain(suffix_for_subprod_split.into_iter())
                  .collect(),
              );
              /* NB: Mutate the loop state! */
              /* Start a new interval of nonterminals which must come after the current
               * subprod split. */
              cur_elements.extend(vec![
                /* We /start/ with an "end" vertex because this comes /out/ of a
                 * subproduction! */
                EpsilonGraphVertex::End(*target_subprod_ref),
                neg_anon,
              ]);
              /* Register the interval we just cut off in the results list. */
              all_intervals_from_this_case.push(interval_upto_subprod);
            },
          }
        }
        /* Construct the interval of all remaining nonterminals to the end of the
         * production. */
        let suffix_for_end_of_case = vec![neg_case_anon, EpsilonGraphVertex::End(cur_prod_ref)];
        let final_interval = ContiguousNonterminalInterval(
          cur_elements
            .into_iter()
            .chain(suffix_for_end_of_case.into_iter())
            .collect(),
        );
        /* Register the interval of all remaining nonterminals in the results list. */
        all_intervals_from_this_case.push(final_interval);
        /* Return all the intervals from this case. */
        really_all_intervals.extend(all_intervals_from_this_case);
      }
    }
    EpsilonIntervalGraph {
      all_intervals: really_all_intervals,
      anon_step_mapping,
    }
  }

  pub fn new(grammar: &TokenGrammar<Tok>) -> Self {
    let terminals_interval_graph: EpsilonIntervalGraph =
      Self::produce_terminals_interval_graph(grammar);
    let cyclic_graph_decomposition: CyclicGraphDecomposition =
      terminals_interval_graph.connect_all_vertices();
    PreprocessedGrammar {
      token_states_mapping: grammar.index_token_states(),
      cyclic_graph_decomposition,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::test_framework::{basic_productions, non_cyclic_productions};

  #[test]
  fn token_grammar_state_indexing() {
    let prods = non_cyclic_productions();
    let grammar = TokenGrammar::new(&prods);
    assert_eq!(
      grammar.index_token_states(),
      [
        ('a', vec![
          TokenPosition::new(0, 0, 0),
          TokenPosition::new(1, 0, 0),
          TokenPosition::new(1, 1, 1),
        ]),
        ('b', vec![
          TokenPosition::new(0, 0, 1),
          TokenPosition::new(1, 0, 1)
        ],),
      ]
      .iter()
      .cloned()
      .collect::<IndexMap<_, _>>(),
    )
  }


  #[test]
  fn terminals_interval_graph() {
    let noncyclic_prods = non_cyclic_productions();
    let noncyclic_grammar = TokenGrammar::new(&noncyclic_prods);
    let noncyclic_interval_graph =
      PreprocessedGrammar::produce_terminals_interval_graph(&noncyclic_grammar);

    let s_0 = TokenPosition::new(0, 0, 0);
    let s_1 = TokenPosition::new(0, 0, 1);
    let a_prod = ProdRef(0);

    let s_2 = TokenPosition::new(1, 0, 0);
    let s_3 = TokenPosition::new(1, 0, 1);
    let s_4 = TokenPosition::new(1, 1, 1);
    let b_prod = ProdRef(1);

    let a_start = EpsilonGraphVertex::Start(a_prod);
    let a_prod_anon_start = EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(0)));
    let a_0_0 = EpsilonGraphVertex::State(s_0);
    let a_0_1 = EpsilonGraphVertex::State(s_1);
    let a_prod_anon_end = EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(0)));
    let a_end = EpsilonGraphVertex::End(a_prod);

    let b_start = EpsilonGraphVertex::Start(b_prod);
    let b_prod_anon_start = EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(1)));
    let b_0_0 = EpsilonGraphVertex::State(s_2);
    let b_0_1 = EpsilonGraphVertex::State(s_3);
    let b_0_anon_0_start = EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(2)));
    let b_0_anon_0_end = EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(2)));
    let b_1_anon_0_start = EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(3)));
    let b_1_anon_0_start_2 = EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(4)));
    let b_1_anon_0_end = EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(3)));
    let b_1_anon_0_end_2 = EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(4)));
    let b_1_1 = EpsilonGraphVertex::State(s_4);
    let b_prod_anon_end = EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(1)));
    let b_end = EpsilonGraphVertex::End(b_prod);

    let a_0 = ContiguousNonterminalInterval(vec![
      a_start,
      a_prod_anon_start,
      a_0_0,
      a_0_1,
      a_prod_anon_end,
      a_end,
    ]);
    let b_start_to_a_start_0 = ContiguousNonterminalInterval(vec![
      b_start,
      b_prod_anon_start,
      b_0_0,
      b_0_1,
      b_0_anon_0_start,
      a_start,
    ]);
    let a_end_to_b_end_0 =
      ContiguousNonterminalInterval(vec![a_end, b_0_anon_0_end, b_prod_anon_end, b_end]);
    let b_start_to_a_start_1 =
      ContiguousNonterminalInterval(vec![b_start, b_1_anon_0_start, b_1_anon_0_start_2, a_start]);
    let a_end_to_b_end_1 =
      ContiguousNonterminalInterval(vec![a_end, b_1_anon_0_end_2, b_1_1, b_1_anon_0_end, b_end]);

    assert_eq!(noncyclic_interval_graph, EpsilonIntervalGraph {
      all_intervals: vec![
        a_0.clone(),
        b_start_to_a_start_0.clone(),
        a_end_to_b_end_0.clone(),
        b_start_to_a_start_1.clone(),
        a_end_to_b_end_1.clone(),
      ],
      anon_step_mapping: [
        (
          AnonSym(0),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: ProdRef(0),
            case: CaseRef(0)
          })
        ),
        (
          AnonSym(1),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: ProdRef(1),
            case: CaseRef(0)
          })
        ),
        (AnonSym(2), UnflattenedProdCaseRef::PassThrough),
        (
          AnonSym(3),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: ProdRef(1),
            case: CaseRef(1)
          })
        ),
        (AnonSym(4), UnflattenedProdCaseRef::PassThrough),
      ]
      .iter()
      .cloned()
      .collect(),
    });

    /* Now check for indices. */
    let intervals_by_start_and_end = noncyclic_interval_graph.find_start_end_indices();
    assert_eq!(
      intervals_by_start_and_end,
      vec![
        (a_prod, StartEndEpsilonIntervals {
          start_epsilons: vec![a_0.clone()],
          end_epsilons: vec![a_end_to_b_end_0.clone(), a_end_to_b_end_1.clone()],
        },),
        (b_prod, StartEndEpsilonIntervals {
          start_epsilons: vec![b_start_to_a_start_0.clone(), b_start_to_a_start_1.clone()],
          end_epsilons: vec![],
        },),
      ]
      .iter()
      .cloned()
      .collect::<IndexMap<ProdRef, StartEndEpsilonIntervals>>()
    );

    /* Now check that the transition graph is as we expect. */
    let CyclicGraphDecomposition {
      cyclic_subgraph: merged_stack_cycles,
      pairwise_state_transitions: all_completed_pairs_with_vertices,
      anon_step_mapping,
    } = noncyclic_interval_graph.connect_all_vertices();
    /* There are no stack cycles in the noncyclic graph. */
    assert_eq!(merged_stack_cycles, EpsilonNodeStateSubgraph {
      vertex_mapping: IndexMap::new(),
      trie_node_universe: vec![],
    });
    assert_eq!(
      anon_step_mapping,
      [
        (
          AnonSym(0),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: ProdRef(0),
            case: CaseRef(0)
          })
        ),
        (
          AnonSym(1),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: ProdRef(1),
            case: CaseRef(0)
          })
        ),
        (AnonSym(2), UnflattenedProdCaseRef::PassThrough),
        (
          AnonSym(3),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: ProdRef(1),
            case: CaseRef(1)
          })
        ),
        (AnonSym(4), UnflattenedProdCaseRef::PassThrough),
      ]
      .iter()
      .cloned()
      .collect::<IndexMap<_, _>>()
    );

    assert_eq!(all_completed_pairs_with_vertices, vec![
      /* 1 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Start, LoweredState::Within(s_0)),
        ContiguousNonterminalInterval(vec![a_start, a_prod_anon_start, a_0_0]),
      ),
      /* 2 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Start, LoweredState::Within(s_2)),
        ContiguousNonterminalInterval(vec![b_start, b_prod_anon_start, b_0_0]),
      ),
      /* 3 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Within(s_0), LoweredState::Within(s_1)),
        ContiguousNonterminalInterval(vec![a_0_0, a_0_1]),
      ),
      /* 4 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Within(s_2), LoweredState::Within(s_3)),
        ContiguousNonterminalInterval(vec![b_0_0, b_0_1]),
      ),
      /* 5 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Within(s_4), LoweredState::End),
        ContiguousNonterminalInterval(vec![b_1_1, b_1_anon_0_end, b_end]),
      ),
      /* 6 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Within(s_1), LoweredState::End),
        ContiguousNonterminalInterval(vec![a_0_1, a_prod_anon_end, a_end]),
      ),
      /* 7 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Start, LoweredState::Within(s_0)),
        ContiguousNonterminalInterval(vec![
          b_start,
          b_1_anon_0_start,
          b_1_anon_0_start_2,
          a_start,
          a_prod_anon_start,
          a_0_0
        ]),
      ),
      /* 8 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Within(s_1), LoweredState::Within(s_4)),
        ContiguousNonterminalInterval(vec![a_0_1, a_prod_anon_end, a_end, b_1_anon_0_end_2, b_1_1]),
      ),
      /* 9 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Within(s_3), LoweredState::Within(s_0)),
        ContiguousNonterminalInterval(vec![
          b_0_1,
          b_0_anon_0_start,
          a_start,
          a_prod_anon_start,
          a_0_0
        ]),
      ),
      /* 10 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Within(s_1), LoweredState::End),
        ContiguousNonterminalInterval(vec![
          a_0_1,
          a_prod_anon_end,
          a_end,
          b_0_anon_0_end,
          b_prod_anon_end,
          b_end
        ]),
      ),
    ]);

    /* Now do the same, but for `basic_productions()`. */
    /* TODO: test `.find_start_end_indices()` and `.connect_all_vertices()` here
     * too! */
    let prods = basic_productions();
    let grammar = TokenGrammar::new(&prods);
    let interval_graph = PreprocessedGrammar::produce_terminals_interval_graph(&grammar);
    assert_eq!(interval_graph.clone(), EpsilonIntervalGraph {
      all_intervals: vec![
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(0)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(0))),
          EpsilonGraphVertex::State(TokenPosition::new(0, 0, 0)),
          EpsilonGraphVertex::State(TokenPosition::new(0, 0, 1)),
          EpsilonGraphVertex::State(TokenPosition::new(0, 0, 2)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(0))),
          EpsilonGraphVertex::End(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(0)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(1))),
          EpsilonGraphVertex::State(TokenPosition::new(0, 1, 0)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(2))),
          EpsilonGraphVertex::Start(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(0)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(2))),
          EpsilonGraphVertex::State(TokenPosition::new(0, 1, 2)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(1))),
          EpsilonGraphVertex::End(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(0)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(3))),
          EpsilonGraphVertex::State(TokenPosition::new(0, 2, 0)),
          EpsilonGraphVertex::State(TokenPosition::new(0, 2, 1)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(4))),
          EpsilonGraphVertex::Start(ProdRef(1)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(4))),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(3))),
          EpsilonGraphVertex::End(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(5))),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(6))),
          EpsilonGraphVertex::Start(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(0)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(6))),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(5))),
          EpsilonGraphVertex::End(ProdRef(1)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(7))),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(8))),
          EpsilonGraphVertex::Start(ProdRef(1)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(8))),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(7))),
          EpsilonGraphVertex::End(ProdRef(1)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(9))),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(10))),
          EpsilonGraphVertex::Start(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(0)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(10))),
          EpsilonGraphVertex::State(TokenPosition::new(1, 2, 1)),
          EpsilonGraphVertex::State(TokenPosition::new(1, 2, 2)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(9))),
          EpsilonGraphVertex::End(ProdRef(1)),
        ]),
      ],
      anon_step_mapping: [
        (
          AnonSym(0),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: ProdRef(0),
            case: CaseRef(0)
          })
        ),
        (
          AnonSym(1),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: ProdRef(0),
            case: CaseRef(1)
          })
        ),
        (AnonSym(2), UnflattenedProdCaseRef::PassThrough),
        (
          AnonSym(3),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: ProdRef(0),
            case: CaseRef(2)
          })
        ),
        (AnonSym(4), UnflattenedProdCaseRef::PassThrough),
        (
          AnonSym(5),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: ProdRef(1),
            case: CaseRef(0)
          })
        ),
        (AnonSym(6), UnflattenedProdCaseRef::PassThrough),
        (
          AnonSym(7),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: ProdRef(1),
            case: CaseRef(1)
          })
        ),
        (AnonSym(8), UnflattenedProdCaseRef::PassThrough),
        (
          AnonSym(9),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: ProdRef(1),
            case: CaseRef(2)
          })
        ),
        (AnonSym(10), UnflattenedProdCaseRef::PassThrough),
      ]
      .iter()
      .cloned()
      .collect(),
    });
  }

  /* TODO: consider creating/using a generic tree diffing algorithm in case
   * that speeds up debugging (this might conflict with the benefits of using
   * totally ordered IndexMaps though, namely determinism, as well as knowing
   * exactly which order your subtrees are created in)! */
  #[test]
  fn noncyclic_transition_graph() {
    let prods = non_cyclic_productions();
    let grammar = TokenGrammar::new(&prods);
    let preprocessed_grammar = PreprocessedGrammar::new(&grammar);
    let first_a = TokenPosition::new(0, 0, 0);
    let first_b = TokenPosition::new(0, 0, 1);
    let second_a = TokenPosition::new(1, 0, 0);
    let second_b = TokenPosition::new(1, 0, 1);
    let third_a = TokenPosition::new(1, 1, 1);
    let a_prod = ProdRef(0);
    let b_prod = ProdRef(1);
    assert_eq!(
      preprocessed_grammar.token_states_mapping.clone(),
      vec![
        ('a', vec![first_a, second_a, third_a],),
        ('b', vec![first_b, second_b],),
      ]
      .iter()
      .cloned()
      .collect::<IndexMap<char, Vec<TokenPosition>>>(),
    );

    let other_cyclic_graph_decomposition = CyclicGraphDecomposition {
      cyclic_subgraph: EpsilonNodeStateSubgraph {
        vertex_mapping: IndexMap::new(),
        trie_node_universe: vec![],
      },
      pairwise_state_transitions: vec![
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Start,
            right: LoweredState::Within(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(0),
            }),
          },
          interval: ContiguousNonterminalInterval(vec![
            EpsilonGraphVertex::Start(a_prod),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(0))),
            EpsilonGraphVertex::State(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(0),
            }),
          ]),
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Start,
            right: LoweredState::Within(TokenPosition {
              prod: b_prod,
              case: CaseRef(0),
              case_el: CaseElRef(0),
            }),
          },
          interval: ContiguousNonterminalInterval(vec![
            EpsilonGraphVertex::Start(b_prod),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(1))),
            EpsilonGraphVertex::State(TokenPosition {
              prod: b_prod,
              case: CaseRef(0),
              case_el: CaseElRef(0),
            }),
          ]),
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(0),
            }),
            right: LoweredState::Within(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(1),
            }),
          },
          interval: ContiguousNonterminalInterval(vec![
            EpsilonGraphVertex::State(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(0),
            }),
            EpsilonGraphVertex::State(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(1),
            }),
          ]),
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(TokenPosition {
              prod: b_prod,
              case: CaseRef(0),
              case_el: CaseElRef(0),
            }),
            right: LoweredState::Within(TokenPosition {
              prod: b_prod,
              case: CaseRef(0),
              case_el: CaseElRef(1),
            }),
          },
          interval: ContiguousNonterminalInterval(vec![
            EpsilonGraphVertex::State(TokenPosition {
              prod: b_prod,
              case: CaseRef(0),
              case_el: CaseElRef(0),
            }),
            EpsilonGraphVertex::State(TokenPosition {
              prod: b_prod,
              case: CaseRef(0),
              case_el: CaseElRef(1),
            }),
          ]),
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(TokenPosition {
              prod: b_prod,
              case: CaseRef(1),
              case_el: CaseElRef(1),
            }),
            right: LoweredState::End,
          },
          interval: ContiguousNonterminalInterval(vec![
            EpsilonGraphVertex::State(TokenPosition {
              prod: b_prod,
              case: CaseRef(1),
              case_el: CaseElRef(1),
            }),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(3))),
            EpsilonGraphVertex::End(b_prod),
          ]),
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(1),
            }),
            right: LoweredState::End,
          },
          interval: ContiguousNonterminalInterval(vec![
            EpsilonGraphVertex::State(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(1),
            }),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(0))),
            EpsilonGraphVertex::End(a_prod),
          ]),
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Start,
            right: LoweredState::Within(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(0),
            }),
          },
          interval: ContiguousNonterminalInterval(vec![
            EpsilonGraphVertex::Start(b_prod),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(3))),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(4))),
            EpsilonGraphVertex::Start(a_prod),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(0))),
            EpsilonGraphVertex::State(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(0),
            }),
          ]),
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(1),
            }),
            right: LoweredState::Within(TokenPosition {
              prod: b_prod,
              case: CaseRef(1),
              case_el: CaseElRef(1),
            }),
          },
          interval: ContiguousNonterminalInterval(vec![
            EpsilonGraphVertex::State(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(1),
            }),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(0))),
            EpsilonGraphVertex::End(a_prod),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(4))),
            EpsilonGraphVertex::State(TokenPosition {
              prod: b_prod,
              case: CaseRef(1),
              case_el: CaseElRef(1),
            }),
          ]),
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(TokenPosition {
              prod: b_prod,
              case: CaseRef(0),
              case_el: CaseElRef(1),
            }),
            right: LoweredState::Within(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(0),
            }),
          },
          interval: ContiguousNonterminalInterval(vec![
            EpsilonGraphVertex::State(TokenPosition {
              prod: b_prod,
              case: CaseRef(0),
              case_el: CaseElRef(1),
            }),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(2))),
            EpsilonGraphVertex::Start(a_prod),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(0))),
            EpsilonGraphVertex::State(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(0),
            }),
          ]),
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(1),
            }),
            right: LoweredState::End,
          },
          interval: ContiguousNonterminalInterval(vec![
            EpsilonGraphVertex::State(TokenPosition {
              prod: a_prod,
              case: CaseRef(0),
              case_el: CaseElRef(1),
            }),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(0))),
            EpsilonGraphVertex::End(a_prod),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(2))),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(1))),
            EpsilonGraphVertex::End(b_prod),
          ]),
        },
      ],
      anon_step_mapping: [
        (
          AnonSym(0),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: a_prod,
            case: CaseRef(0),
          }),
        ),
        (
          AnonSym(1),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: b_prod,
            case: CaseRef(0),
          }),
        ),
        (AnonSym(2), UnflattenedProdCaseRef::PassThrough),
        (
          AnonSym(3),
          UnflattenedProdCaseRef::Case(ProdCaseRef {
            prod: b_prod,
            case: CaseRef(1),
          }),
        ),
        (AnonSym(4), UnflattenedProdCaseRef::PassThrough),
      ]
      .iter()
      .cloned()
      .collect::<IndexMap<_, _>>(),
    };

    assert_eq!(
      preprocessed_grammar.cyclic_graph_decomposition,
      other_cyclic_graph_decomposition,
    );
  }

  #[test]
  fn cyclic_transition_graph() {
    let prods = basic_productions();
    let grammar = TokenGrammar::new(&prods);
    let preprocessed_grammar = PreprocessedGrammar::new(&grammar);

    let first_a = TokenPosition::new(0, 0, 0);
    let second_a = TokenPosition::new(0, 1, 0);

    let first_b = TokenPosition::new(0, 0, 1);
    let second_b = TokenPosition::new(0, 2, 0);
    let third_b = TokenPosition::new(1, 2, 1);

    let first_c = TokenPosition::new(0, 0, 2);
    let second_c = TokenPosition::new(0, 1, 2);
    let third_c = TokenPosition::new(0, 2, 1);
    let fourth_c = TokenPosition::new(1, 2, 2);

    let a_prod = ProdRef(0);
    let b_prod = ProdRef(1);
    let _c_prod = ProdRef(2); /* unused */

    assert_eq!(
      preprocessed_grammar.token_states_mapping.clone(),
      vec![
        ('a', vec![first_a, second_a]),
        ('b', vec![first_b, second_b, third_b]),
        ('c', vec![first_c, second_c, third_c, fourth_c]),
      ]
      .into_iter()
      .collect::<IndexMap<_, _>>()
    );

    assert_eq!(
      preprocessed_grammar
        .cyclic_graph_decomposition
        .cyclic_subgraph
        .vertex_mapping
        .clone(),
      [
        /* 0 */
        (EpsilonGraphVertex::Start(b_prod), TrieNodeRef(0)),
        /* 1 */
        (
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(7))),
          TrieNodeRef(1)
        ),
        /* 2 */
        (
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(8))),
          TrieNodeRef(2)
        ),
        /* 3 */
        (EpsilonGraphVertex::End(b_prod), TrieNodeRef(3)),
        /* 4 */
        (
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(8))),
          TrieNodeRef(4)
        ),
        /* 5 */
        (
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(7))),
          TrieNodeRef(5)
        ),
        /* 6 */
        (
          EpsilonGraphVertex::State(TokenPosition {
            prod: a_prod,
            case: CaseRef(1),
            case_el: CaseElRef(0)
          }),
          TrieNodeRef(6)
        ),
        /* 7 */
        (
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(2))),
          TrieNodeRef(7)
        ),
        /* 8 */
        (EpsilonGraphVertex::Start(a_prod), TrieNodeRef(8)),
        /* 9 */
        (
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(1))),
          TrieNodeRef(9)
        ),
        /* 10 */
        (
          EpsilonGraphVertex::State(TokenPosition {
            prod: a_prod,
            case: CaseRef(1),
            case_el: CaseElRef(2)
          }),
          TrieNodeRef(10)
        ),
        /* 11 */
        (
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(1))),
          TrieNodeRef(11)
        ),
        /* 12 */
        (EpsilonGraphVertex::End(a_prod), TrieNodeRef(12)),
        /* 13 */
        (
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(2))),
          TrieNodeRef(13)
        ),
        /* 14 */
        (
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(4))),
          TrieNodeRef(14)
        ),
        /* 15 */
        (
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(3))),
          TrieNodeRef(15)
        ),
        /* 16 */
        (
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(6))),
          TrieNodeRef(16)
        ),
        /* 17 */
        (
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(5))),
          TrieNodeRef(17)
        )
      ]
      .iter()
      .cloned()
      .collect::<IndexMap<_, _>>()
    );

    assert_eq!(
      preprocessed_grammar
        .cyclic_graph_decomposition
        .cyclic_subgraph
        .trie_node_universe,
      vec![
        /* 0 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::Positive(
            StackSym(b_prod)
          ))]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(1))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(2))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 1 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(7)))]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(2))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(0))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 2 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(8)))]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(0))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(1))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 3 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::Negative(
            StackSym(b_prod)
          ))]),
          next_nodes: [
            StackTrieNextEntry::Incomplete(TrieNodeRef(4)),
            StackTrieNextEntry::Incomplete(TrieNodeRef(14))
          ]
          .iter()
          .cloned()
          .collect::<IndexSet<_>>(),
          prev_nodes: [
            StackTrieNextEntry::Incomplete(TrieNodeRef(5)),
            StackTrieNextEntry::Incomplete(TrieNodeRef(17))
          ]
          .iter()
          .cloned()
          .collect::<IndexSet<_>>()
        },
        /* 4 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(8)))]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(5))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(3))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 5 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(7)))]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(3))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(4))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 6 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(7))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(9))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 7 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(2)))]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(8))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(6))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 8 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::Positive(
            StackSym(a_prod)
          ))]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(9))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(7))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 9 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(1)))]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(6))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(8))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 10 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(11))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(13))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 11 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(1)))]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(12))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(10))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 12 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::Negative(
            StackSym(a_prod)
          ))]),
          next_nodes: [
            StackTrieNextEntry::Incomplete(TrieNodeRef(13)),
            StackTrieNextEntry::Incomplete(TrieNodeRef(16))
          ]
          .iter()
          .cloned()
          .collect::<IndexSet<_>>(),
          prev_nodes: [
            StackTrieNextEntry::Incomplete(TrieNodeRef(11)),
            StackTrieNextEntry::Incomplete(TrieNodeRef(15))
          ]
          .iter()
          .cloned()
          .collect::<IndexSet<_>>()
        },
        /* 13 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(2)))]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(10))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(12))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 14 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(4)))]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(15))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(3))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 15 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(3)))]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(12))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(14))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 16 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(6)))]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(17))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(12))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        },
        /* 17 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(5)))]),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(3))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(16))]
            .iter()
            .cloned()
            .collect::<IndexSet<_>>()
        }
      ]
    );
  }
}
