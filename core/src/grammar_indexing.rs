/* Copyright (C) 2021 Danny McClanahan <dmcC2@hypnicjerk.ai> */
/* SPDX-License-Identifier: AGPL-3.0 */

//! Implementation for getting a [PreprocessedGrammar].
//!
//! This phase is **extremely complex** and **may not work yet!** This module
//! detects all *stack cycles* and *contiguous state
//! transitions* so that later in [super::parsing] we can ensure the parsing
//! will always terminate by *bounding context-sensitivity* **(TODO: link to
//! paper!)**.
//!
//! *Implementation Note: Performance doesn't matter here.*

use crate::{
  allocation::HandoffAllocable,
  lowering_to_indices::{grammar_building as gb, graph_coordinates as gc},
  types::{DefaultHasher, Vec},
};

use indexmap::{IndexMap, IndexSet};

use core::{
  alloc::Allocator,
  cmp::{Eq, PartialEq},
  fmt,
  hash::{Hash, Hasher},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum UnflattenedProdCaseRef {
  PassThrough,
  Case(gc::ProdCaseRef),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct StackSym(pub gc::ProdRef);

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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum LoweredState {
  Start,
  End,
  Within(gc::TokenPosition),
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

/* Fun fact: I'm pretty sure this /is/ actually an interval graph,
 * describing the continuous strings of terminals in a TokenGrammar! */
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum EpsilonGraphVertex {
  Start(gc::ProdRef),
  End(gc::ProdRef),
  Anon(AnonStep),
  State(gc::TokenPosition),
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StackStepError {
  StepConcatenationError(NamedOrAnonStep, NamedOrAnonStep),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
  pub fn sequence(
    self,
    other: Self,
  ) -> Result<Option<(NamedOrAnonStep, NamedOrAnonStep)>, StackStepError> {
    if self.polarity() == other.polarity() {
      Ok(Some((self, other)))
    } else if self.as_unsigned_step() == other.as_unsigned_step() {
      Ok(None)
    } else {
      Err(StackStepError::StepConcatenationError(self, other))
    }
  }
}

#[derive(Clone)]
pub struct StackDiffSegment<Arena>(pub Vec<NamedOrAnonStep, Arena>)
where Arena: Allocator;

impl<Arena> HandoffAllocable for StackDiffSegment<Arena>
where Arena: Allocator+Clone
{
  type Arena = Arena;

  fn allocator_handoff(&self) -> Arena { self.0.allocator().clone() }
}

impl<Arena> PartialEq for StackDiffSegment<Arena>
where Arena: Allocator
{
  fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<Arena> Eq for StackDiffSegment<Arena> where Arena: Allocator {}

impl<Arena> fmt::Debug for StackDiffSegment<Arena>
where Arena: Allocator
{
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "StackDiffSegment({:?})", &self.0)
  }
}

impl<Arena> Hash for StackDiffSegment<Arena>
where Arena: Allocator
{
  fn hash<H: Hasher>(&self, state: &mut H) { self.0.hash(state) }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TrieNodeRef(pub usize);

#[derive(Debug, Clone)]
pub struct StackTrieNode<Arena>
where Arena: Allocator+Clone
{
  pub stack_diff: StackDiffSegment<Arena>,
  /* During parsing, the top of the stack will be a named or anonymous symbol. We can negate
   * that (it should always be a positive step on the top of the stack, so a negative
   * step, I think) to get a NamedOrAnonStep which can index into the relevant segments.
   * This supports stack cycles, as well as using an Rc<StackTrieNode> to manage state
   * during the parse. TODO: make a "build" method that removes the RefCell, coalesces
   * stack diffs, and makes the next nodes an IndexMap (??? on the last part given lex
   * BFS?!)! */
  pub next_nodes: IndexSet<StackTrieNextEntry, Arena, DefaultHasher>,
  /* Doubly-linked so that they can be traversed from either direction -- this is (maybe) key
   * to parallelism in parsing! */
  pub prev_nodes: IndexSet<StackTrieNextEntry, Arena, DefaultHasher>,
}


impl<Arena> PartialEq for StackTrieNode<Arena>
where Arena: Allocator+Clone
{
  fn eq(&self, other: &Self) -> bool {
    self.stack_diff == other.stack_diff
      && self.next_nodes == other.next_nodes
      && self.prev_nodes == other.prev_nodes
  }
}

impl<Arena> Eq for StackTrieNode<Arena> where Arena: Allocator+Clone {}

impl<Arena> StackTrieNode<Arena>
where Arena: Allocator+Clone
{
  fn bare(vtx: EpsilonGraphVertex, arena: Arena) -> Self {
    let mut diff = Vec::new_in(arena.clone());
    match vtx.get_step() {
      None => (),
      Some(step) => {
        diff.push(step);
      },
    }
    StackTrieNode {
      stack_diff: StackDiffSegment(diff),
      next_nodes: IndexSet::new_in(arena.clone()),
      prev_nodes: IndexSet::new_in(arena),
    }
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StackTrieNextEntry {
  #[allow(dead_code)]
  Completed(LoweredState),
  Incomplete(TrieNodeRef),
}

#[derive(Debug, Clone)]
pub struct EpsilonNodeStateSubgraph<Arena>
where Arena: Allocator+Clone
{
  pub vertex_mapping: IndexMap<EpsilonGraphVertex, TrieNodeRef, Arena, DefaultHasher>,
  pub trie_node_universe: Vec<StackTrieNode<Arena>, Arena>,
}

impl<Arena> HandoffAllocable for EpsilonNodeStateSubgraph<Arena>
where Arena: Allocator+Clone
{
  type Arena = Arena;

  fn allocator_handoff(&self) -> Arena { self.vertex_mapping.arena() }
}

impl<Arena> PartialEq for EpsilonNodeStateSubgraph<Arena>
where Arena: Allocator+Clone
{
  fn eq(&self, other: &Self) -> bool {
    self.vertex_mapping == other.vertex_mapping
      && self.trie_node_universe == other.trie_node_universe
  }
}

impl<Arena> Eq for EpsilonNodeStateSubgraph<Arena> where Arena: Allocator+Clone {}

impl<Arena> EpsilonNodeStateSubgraph<Arena>
where Arena: Allocator+Clone
{
  fn new_in(arena: Arena) -> Self {
    EpsilonNodeStateSubgraph {
      vertex_mapping: IndexMap::new_in(arena.clone()),
      trie_node_universe: Vec::new_in(arena),
    }
  }

  fn get_trie(&mut self, node_ref: TrieNodeRef) -> &mut StackTrieNode<Arena> {
    let TrieNodeRef(node_index) = node_ref;
    self
      .trie_node_universe
      .get_mut(node_index)
      .expect("trie node ref should never have been constructed out of bounds?")
  }

  fn trie_ref_for_vertex(&mut self, vtx: &EpsilonGraphVertex) -> TrieNodeRef {
    let basic_node = StackTrieNode::bare(*vtx, self.allocator_handoff());
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

#[derive(Debug, Clone)]
pub struct ContiguousNonterminalInterval<Arena>
where Arena: Allocator
{
  pub interval: Vec<EpsilonGraphVertex, Arena>,
  pub arena: Arena,
}

impl<Arena> PartialEq for ContiguousNonterminalInterval<Arena>
where Arena: Allocator
{
  fn eq(&self, other: &Self) -> bool { self.interval == other.interval }
}

impl<Arena> Eq for ContiguousNonterminalInterval<Arena> where Arena: Allocator {}

#[derive(Debug, Clone)]
pub struct CyclicGraphDecomposition<Arena>
where Arena: Allocator+Clone
{
  pub cyclic_subgraph: EpsilonNodeStateSubgraph<Arena>,
  pub pairwise_state_transitions: Vec<CompletedStatePairWithVertices<Arena>, Arena>,
  pub anon_step_mapping: IndexMap<AnonSym, UnflattenedProdCaseRef, Arena, DefaultHasher>,
}

impl<Arena> PartialEq for CyclicGraphDecomposition<Arena>
where Arena: Allocator+Clone
{
  fn eq(&self, other: &Self) -> bool {
    self.cyclic_subgraph == other.cyclic_subgraph
      && self.pairwise_state_transitions == other.pairwise_state_transitions
      && self.anon_step_mapping == other.anon_step_mapping
  }
}

impl<Arena> Eq for CyclicGraphDecomposition<Arena> where Arena: Allocator+Clone {}

/* Pointers to the appropriate "forests" of stack transitions
 * starting/completing at each state. "starting" and "completing" are
 * mirrored to allow working away at mapping states to input token indices
 * from either direction, which is intended to allow for parallelism. They're
 * not really "forests" because they *will* have cycles except in very simple
 * grammars (CFGs and below, I think? Unclear if the Chomsky hierarchy
 * still applies). */
// TODO: fix the above incorrect docstring!
#[derive(Debug, Clone)]
pub struct EpsilonIntervalGraph<Arena>
where Arena: Allocator+Clone
{
  pub all_intervals: Vec<ContiguousNonterminalInterval<Arena>, Arena>,
  pub anon_step_mapping: IndexMap<AnonSym, UnflattenedProdCaseRef, Arena, DefaultHasher>,
}

impl<Arena> HandoffAllocable for EpsilonIntervalGraph<Arena>
where Arena: Allocator+Clone
{
  type Arena = Arena;

  fn allocator_handoff(&self) -> Arena { self.anon_step_mapping.arena() }
}

impl<Arena> PartialEq for EpsilonIntervalGraph<Arena>
where Arena: Allocator+Clone
{
  fn eq(&self, other: &Self) -> bool {
    self.all_intervals == other.all_intervals && self.anon_step_mapping == other.anon_step_mapping
  }
}

impl<Arena> Eq for EpsilonIntervalGraph<Arena> where Arena: Allocator+Clone {}

impl<Arena> EpsilonIntervalGraph<Arena>
where Arena: Allocator+Clone
{
  pub fn find_start_end_indices(
    &self,
  ) -> IndexMap<gc::ProdRef, StartEndEpsilonIntervals<Arena>, Arena, DefaultHasher> {
    let arena = self.allocator_handoff();
    let mut epsilon_subscripts_index: IndexMap<
      gc::ProdRef,
      StartEndEpsilonIntervals<Arena>,
      Arena,
      DefaultHasher,
    > = IndexMap::new_in(arena.clone());
    let EpsilonIntervalGraph { all_intervals, .. } = self;
    for interval in all_intervals.iter() {
      let ContiguousNonterminalInterval {
        interval: vertices, ..
      } = interval.clone();
      /* We should always have a start and end node. */
      assert!(vertices.len() >= 2);
      let first = vertices[0];
      match first {
          EpsilonGraphVertex::Start(start_prod_ref) => {
            let intervals_for_this_prod = epsilon_subscripts_index.entry(start_prod_ref)
              .or_insert(StartEndEpsilonIntervals::new_in(arena.clone()));
            (*intervals_for_this_prod).start_epsilons.push(interval.clone());
          },
          EpsilonGraphVertex::End(end_prod_ref) => {
            let intervals_for_this_prod = epsilon_subscripts_index.entry(end_prod_ref)
              .or_insert(StartEndEpsilonIntervals::new_in(arena.clone()));
            (*intervals_for_this_prod).end_epsilons.push(interval.clone());
          },
          _ => unreachable!("the beginning of an interval should always be a start (epsilon) or end (epsilon prime) vertex"),
        }
    }
    epsilon_subscripts_index
  }

  pub fn connect_all_vertices(self) -> CyclicGraphDecomposition<Arena> {
    let intervals_indexed_by_start_and_end = self.find_start_end_indices();
    let arena = self.allocator_handoff();
    let EpsilonIntervalGraph {
      all_intervals,
      anon_step_mapping,
    } = self;

    let mut all_completed_pairs_with_vertices: Vec<CompletedStatePairWithVertices<Arena>, Arena> =
      Vec::new_in(arena.clone());
    /* NB: When finding token transitions, we keep track of which intermediate
     * transition states we've already seen by using this Hash impl. If any
     * stack cycles are detected when performing a single iteration, the
     * `todo` is dropped, but as there may be multiple paths to
     * the same intermediate transition state, we additionally require checking
     * the identity of intermediate transition states to avoid looping
     * forever. */
    let mut seen_transitions: IndexSet<IntermediateTokenTransition<Arena>, Arena, DefaultHasher> =
      IndexSet::new_in(arena.clone());

    let mut traversal_queue: Vec<IntermediateTokenTransition<Arena>, Arena> =
      Vec::with_capacity_in(all_intervals.len(), arena.clone());
    traversal_queue.extend(all_intervals.iter().map(IntermediateTokenTransition::new));

    let mut all_stack_cycles: Vec<SingleStackCycle<Arena>, Arena> = Vec::new_in(arena.clone());

    /* Find all the token transitions! */
    while !traversal_queue.is_empty() {
      let cur_transition = traversal_queue.remove(0);
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

    let merged_stack_cycles: EpsilonNodeStateSubgraph<Arena> = {
      let mut ret = EpsilonNodeStateSubgraph::new_in(arena);
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
#[derive(Debug, Clone)]
pub struct StartEndEpsilonIntervals<Arena>
where Arena: Allocator
{
  pub start_epsilons: Vec<ContiguousNonterminalInterval<Arena>, Arena>,
  pub end_epsilons: Vec<ContiguousNonterminalInterval<Arena>, Arena>,
}


impl<Arena> PartialEq for StartEndEpsilonIntervals<Arena>
where Arena: Allocator
{
  fn eq(&self, other: &Self) -> bool {
    self.start_epsilons == other.start_epsilons && self.end_epsilons == other.end_epsilons
  }
}

impl<Arena> Eq for StartEndEpsilonIntervals<Arena> where Arena: Allocator {}

impl<Arena> StartEndEpsilonIntervals<Arena>
where Arena: Allocator+Clone
{
  fn new_in(arena: Arena) -> Self {
    StartEndEpsilonIntervals {
      start_epsilons: Vec::new_in(arena.clone()),
      end_epsilons: Vec::new_in(arena),
    }
  }
}

#[derive(Debug, Clone)]
pub struct CompletedStatePairWithVertices<Arena>
where Arena: Allocator
{
  pub state_pair: StatePair,
  pub interval: ContiguousNonterminalInterval<Arena>,
}

impl<Arena> PartialEq for CompletedStatePairWithVertices<Arena>
where Arena: Allocator
{
  fn eq(&self, other: &Self) -> bool {
    self.state_pair == other.state_pair && self.interval == other.interval
  }
}

impl<Arena> Eq for CompletedStatePairWithVertices<Arena> where Arena: Allocator {}

#[derive(Debug, Clone)]
pub struct SingleStackCycle<Arena>(pub Vec<EpsilonGraphVertex, Arena>)
where Arena: Allocator;

#[derive(Debug, Clone)]
struct TransitionIterationResult<Arena>
where Arena: Allocator
{
  pub completed: Vec<CompletedStatePairWithVertices<Arena>, Arena>,
  pub todo: Vec<IntermediateTokenTransition<Arena>, Arena>,
  pub cycles: Vec<SingleStackCycle<Arena>, Arena>,
}

#[derive(Debug, Clone)]
struct IntermediateTokenTransition<Arena>
where Arena: Allocator
{
  cur_traversal_intermediate_nonterminals: Vec<EpsilonGraphVertex, Arena>,
  rest_of_interval: Vec<EpsilonGraphVertex, Arena>,
}

impl<Arena> HandoffAllocable for IntermediateTokenTransition<Arena>
where Arena: Allocator+Clone
{
  type Arena = Arena;

  fn allocator_handoff(&self) -> Arena {
    self
      .cur_traversal_intermediate_nonterminals
      .allocator()
      .clone()
  }
}

impl<Arena> PartialEq for IntermediateTokenTransition<Arena>
where Arena: Allocator
{
  fn eq(&self, other: &Self) -> bool {
    self.cur_traversal_intermediate_nonterminals == other.cur_traversal_intermediate_nonterminals
      && self.rest_of_interval == other.rest_of_interval
  }
}

impl<Arena> Eq for IntermediateTokenTransition<Arena> where Arena: Allocator {}

/// This [Hash] implementation is stable because the collection types in this
/// struct have a specific ordering.
impl<Arena> Hash for IntermediateTokenTransition<Arena>
where Arena: Allocator
{
  fn hash<H: Hasher>(&self, state: &mut H) {
    for intermediate_vertex in self.cur_traversal_intermediate_nonterminals.iter() {
      intermediate_vertex.hash(state);
    }
    for subsequent_vertex in self.rest_of_interval.iter() {
      subsequent_vertex.hash(state);
    }
  }
}

impl<Arena> IntermediateTokenTransition<Arena>
where Arena: Allocator+Clone
{
  fn new(wrapped_interval: &ContiguousNonterminalInterval<Arena>) -> Self {
    let ContiguousNonterminalInterval { interval, arena } = wrapped_interval;
    /* All intervals have a start and end node. */
    assert!(interval.len() >= 2);
    let start = interval[0];
    let mut cur_start: Vec<EpsilonGraphVertex, Arena> = Vec::with_capacity_in(1, arena.clone());
    cur_start.push(start);

    let rest_of_interval = &interval[1..];
    let mut cur_rest: Vec<EpsilonGraphVertex, Arena> =
      Vec::with_capacity_in(rest_of_interval.len(), arena.clone());
    cur_rest.extend_from_slice(rest_of_interval);

    IntermediateTokenTransition {
      cur_traversal_intermediate_nonterminals: cur_start,
      rest_of_interval: cur_rest,
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
  ) -> (
    IndexSet<EpsilonGraphVertex, Arena, DefaultHasher>,
    Option<SingleStackCycle<Arena>>,
  ) {
    let arena = self.allocator_handoff();
    let mut prev_nonterminals: IndexSet<EpsilonGraphVertex, Arena, DefaultHasher> =
      IndexSet::with_capacity_in(
        self.cur_traversal_intermediate_nonterminals.len(),
        arena.clone(),
      );
    prev_nonterminals.extend(self.cur_traversal_intermediate_nonterminals.iter().cloned());

    let (cur_vtx_ind, was_new_insert) = prev_nonterminals.insert_full(next);
    if was_new_insert {
      (prev_nonterminals, None)
    } else {
      /* If we have already seen this vertex, then a cycle was detected! */
      /* The cycle contains the start vertex and all the ones after it. */
      let mut cycle_elements: Vec<EpsilonGraphVertex, Arena> = Vec::new_in(arena.clone());
      cycle_elements.extend(prev_nonterminals.iter().skip(cur_vtx_ind).cloned());
      let cur_cycle = SingleStackCycle(cycle_elements);

      /* Shuffle all the intermediate vertices off, but keep the cycle start
       * vertex. */
      let len_to_take = cur_vtx_ind + 1;
      let mut remaining_elements: IndexSet<EpsilonGraphVertex, Arena, DefaultHasher> =
        IndexSet::with_capacity_in(len_to_take, arena);
      remaining_elements.extend(prev_nonterminals.into_iter().take(len_to_take));

      (remaining_elements, Some(cur_cycle))
    }
  }

  /// TODO: document this great method!!!
  fn process_next_vertex(
    &self,
    start: &EpsilonGraphVertex,
    next: EpsilonGraphVertex,
    indexed_intervals: &IndexMap<
      gc::ProdRef,
      StartEndEpsilonIntervals<Arena>,
      Arena,
      DefaultHasher,
    >,
    intermediate_nonterminals_for_next_step: IndexSet<EpsilonGraphVertex, Arena, DefaultHasher>,
  ) -> (
    Vec<CompletedStatePairWithVertices<Arena>, Arena>,
    Vec<IntermediateTokenTransition<Arena>, Arena>,
  ) {
    let arena = self.allocator_handoff();
    match next {
      /* Complete a transition, but also add more continuing from the start vertex. */
      EpsilonGraphVertex::Start(start_prod_ref) => {
        /* We only have this single next node, since we always start or end at a
         * start or end. */
        assert_eq!(1, self.rest_of_interval.len());
        /* NB: In the model we use for state transitions `A`, we never start from an
         * End node or end on a Start node, so we can skip completed paths
         * entirely here. */
        let interval = indexed_intervals.get(&start_prod_ref).expect(
          "all `ProdRef`s should have been accounted for when grouping by start and end intervals",
        );

        let mut passthrough_intermediates: Vec<IntermediateTokenTransition<Arena>, Arena> =
          Vec::with_capacity_in(interval.start_epsilons.len(), arena.clone());
        passthrough_intermediates.extend(interval.start_epsilons.iter().map(
          |ContiguousNonterminalInterval {
             interval: next_vertices,
             ..
           }| {
            let mut nonterminals: Vec<EpsilonGraphVertex, Arena> =
              Vec::with_capacity_in(intermediate_nonterminals_for_next_step.len(), arena.clone());
            nonterminals.extend(intermediate_nonterminals_for_next_step.clone().into_iter());

            /* Get the rest of the interval without the epsilon node that it starts with. */
            let rest_of_interval = &next_vertices[1..];
            let mut rest: Vec<EpsilonGraphVertex, Arena> =
              Vec::with_capacity_in(rest_of_interval.len(), arena.clone());
            rest.extend_from_slice(rest_of_interval);

            IntermediateTokenTransition {
              cur_traversal_intermediate_nonterminals: nonterminals,
              rest_of_interval: rest,
            }
          },
        ));
        (Vec::new_in(arena), passthrough_intermediates)
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
            unreachable!("an anonymous vertex should not be at the start of an interval!")
          },
        };

        let completed = if completed_path_makes_sense {
          let mut relevant_interval_with_terminals: Vec<EpsilonGraphVertex, Arena> =
            Vec::with_capacity_in(intermediate_nonterminals_for_next_step.len(), arena.clone());
          relevant_interval_with_terminals
            .extend(intermediate_nonterminals_for_next_step.clone().into_iter());

          let completed_state_pair = StatePair {
            left: LoweredState::from_vertex(*start),
            right: LoweredState::End,
          };
          let single_completion = CompletedStatePairWithVertices {
            state_pair: completed_state_pair,
            interval: ContiguousNonterminalInterval {
              interval: relevant_interval_with_terminals,
              arena: arena.clone(),
            },
          };

          let mut ret: Vec<CompletedStatePairWithVertices<Arena>, Arena> =
            Vec::with_capacity_in(1, arena.clone());
          ret.push(single_completion);
          ret
        } else {
          Vec::new_in(arena.clone())
        };

        let interval = indexed_intervals.get(&end_prod_ref).expect(
          "all `ProdRef`s should have been accounted for when grouping by start and end intervals",
        );

        let mut passthrough_intermediates: Vec<IntermediateTokenTransition<Arena>, Arena> =
          Vec::with_capacity_in(interval.end_epsilons.len(), arena.clone());
        passthrough_intermediates.extend(interval.end_epsilons.clone().into_iter().map(
          |ContiguousNonterminalInterval {
             interval: next_vertices,
             ..
           }| {
            let mut nonterminals: Vec<EpsilonGraphVertex, Arena> =
              Vec::with_capacity_in(intermediate_nonterminals_for_next_step.len(), arena.clone());
            nonterminals.extend(intermediate_nonterminals_for_next_step.clone().into_iter());

            /* Get the rest of the interval without the epsilon node that it starts with. */
            let rest_of_interval = &next_vertices[1..];
            let mut rest: Vec<EpsilonGraphVertex, Arena> =
              Vec::with_capacity_in(rest_of_interval.len(), arena.clone());
            rest.extend_from_slice(rest_of_interval);

            IntermediateTokenTransition {
              cur_traversal_intermediate_nonterminals: nonterminals,
              rest_of_interval: rest,
            }
          },
        ));

        (completed, passthrough_intermediates)
      },
      /* `next` is the anonymous vertex, which is all we need it for. */
      EpsilonGraphVertex::Anon(_) => {
        let mut nonterminals: Vec<EpsilonGraphVertex, Arena> =
          Vec::with_capacity_in(intermediate_nonterminals_for_next_step.len(), arena.clone());
        nonterminals.extend(intermediate_nonterminals_for_next_step.into_iter());

        /* Get the rest of the interval without the epsilon node that it starts with. */
        let rest_of_interval = &self.rest_of_interval[1..];
        let mut rest: Vec<EpsilonGraphVertex, Arena> =
          Vec::with_capacity_in(rest_of_interval.len(), arena.clone());
        rest.extend_from_slice(rest_of_interval);

        let mut ret: Vec<IntermediateTokenTransition<Arena>, Arena> =
          Vec::with_capacity_in(1, arena.clone());
        ret.push(IntermediateTokenTransition {
          cur_traversal_intermediate_nonterminals: nonterminals,
          rest_of_interval: rest,
        });

        (Vec::new_in(arena), ret)
      },
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
            unreachable!("an anonymous vertex should not be at the start of an interval!")
          },
        };
        let completed = if completed_path_makes_sense {
          let mut relevant_interval_with_terminals: Vec<EpsilonGraphVertex, Arena> =
            Vec::with_capacity_in(intermediate_nonterminals_for_next_step.len(), arena.clone());
          relevant_interval_with_terminals
            .extend(intermediate_nonterminals_for_next_step.into_iter());

          let mut ret: Vec<CompletedStatePairWithVertices<Arena>, Arena> =
            Vec::with_capacity_in(1, arena.clone());
          ret.push(CompletedStatePairWithVertices {
            state_pair: completed_state_pair,
            interval: ContiguousNonterminalInterval {
              interval: relevant_interval_with_terminals,
              arena: arena.clone(),
            },
          });
          ret
        } else {
          Vec::new_in(arena.clone())
        };

        let mut single_nonterminal: Vec<EpsilonGraphVertex, Arena> =
          Vec::with_capacity_in(1, arena.clone());
        single_nonterminal.push(next);

        let rest_of_interval = &self.rest_of_interval[1..];
        let mut rest: Vec<EpsilonGraphVertex, Arena> =
          Vec::with_capacity_in(rest_of_interval.len(), arena.clone());
        rest.extend(rest_of_interval.iter().cloned());

        let mut single_transition: Vec<IntermediateTokenTransition<Arena>, Arena> =
          Vec::with_capacity_in(1, arena);
        single_transition.push(IntermediateTokenTransition {
          cur_traversal_intermediate_nonterminals: single_nonterminal,
          rest_of_interval: rest,
        });

        (completed, single_transition)
      },
    }
  }

  fn iterate_and_maybe_complete(
    &self,
    indexed_intervals: &IndexMap<
      gc::ProdRef,
      StartEndEpsilonIntervals<Arena>,
      Arena,
      DefaultHasher,
    >,
  ) -> TransitionIterationResult<Arena> {
    let arena = self.allocator_handoff();

    let start = self.cur_traversal_intermediate_nonterminals[0];
    let next = self.rest_of_interval[0];

    let (intermediate_nonterminals_for_next_step, cycles) = self.check_for_cycles(next);
    let (completed, todo) = self.process_next_vertex(
      &start,
      next,
      indexed_intervals,
      intermediate_nonterminals_for_next_step,
    );

    let mut known_cycles: Vec<SingleStackCycle<Arena>, Arena> = Vec::new_in(arena.clone());
    let todo = match cycles {
      None => todo,
      /* NB: If cycles were detected, don't return any `todo` nodes, as we have already
       * traversed them! */
      Some(cycle) => {
        known_cycles.push(cycle);
        Vec::new_in(arena)
      },
    };
    TransitionIterationResult {
      completed,
      todo,
      cycles: known_cycles,
    }
  }
}

/// A [TokenGrammar][gb::TokenGrammar] after being parsed for cycles.
///
/// There is no intentionally no reference to any
/// [TokenGrammar][gb::TokenGrammar], in the hope that it becomes easier to have
/// the runtime we want just fall out of the code without too much work.
///
/// TODO: ^???
#[derive(Debug, Clone)]
pub struct PreprocessedGrammar<Tok, Arena>
where Arena: Allocator+Clone
{
  /// `M: T -> {Q}`, where `{Q}` is sets of states!
  ///
  /// These don't need to be quick to access or otherwise optimized for the
  /// algorithm until we create a `Parse` -- these are chosen to reduce
  /// redundancy.
  pub token_states_mapping: gb::AlphabetMapping<Arena>,
  /// `A: T x T -> {S}^+_-`
  ///
  /// where `{S}^+_-` (LaTeX formatting) is ordered sequences of signed
  /// stack symbols!
  pub cyclic_graph_decomposition: CyclicGraphDecomposition<Arena>,
  pub alphabet: gb::Alphabet<Tok, Arena>,
}

impl<Tok, Arena> HandoffAllocable for PreprocessedGrammar<Tok, Arena>
where Arena: Allocator+Clone
{
  type Arena = Arena;

  fn allocator_handoff(&self) -> Arena { self.alphabet.allocator_handoff() }
}

impl<Tok, Arena> PreprocessedGrammar<Tok, Arena>
where Arena: Allocator+Clone
{
  /// Intended to reduce visual clutter in the implementation of interval
  /// production.
  fn make_pos_neg_anon_steps(
    cur_index: &mut usize,
    anon_step_mapping: &mut IndexMap<AnonSym, UnflattenedProdCaseRef, Arena, DefaultHasher>,
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
  pub(crate) fn produce_terminals_interval_graph(
    grammar: gb::TokenGrammar<Tok, Arena>,
  ) -> (
    EpsilonIntervalGraph<Arena>,
    gb::Alphabet<Tok, Arena>,
    gb::AlphabetMapping<Arena>,
  ) {
    /* We would like to just accept a DetokenizedProductions here, but we call
     * this method directly in testing, and without the whole grammar object
     * the type ascription is ambiguous. */
    // TODO: what is "type ascription" referring to here^ lol
    let gb::TokenGrammar {
      graph: production_graph,
      alphabet,
      token_states,
    } = grammar;
    let arena = alphabet.allocator_handoff();
    let prods = production_graph.into_index_map();
    /* We would really like to use .flat_map()s here, but it's not clear how to
     * do that while mutating the global `cur_anon_sym_index` value. When
     * `move` is used on the inner loop, the value of `cur_anon_sym_index`
     * mysteriously gets reset, even if `move` is also used on the
     * outer loop. */
    let mut cur_anon_sym_index: usize = 0;
    let mut really_all_intervals: Vec<ContiguousNonterminalInterval<Arena>, Arena> =
      Vec::new_in(arena.clone());
    let mut anon_step_mapping: IndexMap<AnonSym, UnflattenedProdCaseRef, Arena, DefaultHasher> =
      IndexMap::new_in(arena.clone());
    for (cur_prod_ref, the_prod) in prods.iter() {
      let gb::Production(cases) = the_prod;
      for (case_ind, the_case) in cases.iter().enumerate() {
        let cur_case_ref: gc::CaseRef = case_ind.into();
        let gb::Case(elements_of_case) = the_case;
        let mut all_intervals_from_this_case: Vec<ContiguousNonterminalInterval<Arena>, Arena> =
          Vec::new_in(arena.clone());

        /* NB: make an anon sym whenever stepping onto a case! */
        let cur_prod_case = gc::ProdCaseRef {
          prod: *cur_prod_ref,
          case: cur_case_ref,
        };
        let (pos_case_anon, neg_case_anon) = Self::make_pos_neg_anon_steps(
          &mut cur_anon_sym_index,
          &mut anon_step_mapping,
          UnflattenedProdCaseRef::Case(cur_prod_case),
        );

        let mut cur_elements: Vec<EpsilonGraphVertex, Arena> = Vec::new_in(arena.clone());
        cur_elements.push(EpsilonGraphVertex::Start(*cur_prod_ref));
        cur_elements.push(pos_case_anon);

        for (element_of_case_ind, el) in elements_of_case.iter().enumerate() {
          let cur_el_ref: gc::CaseElRef = element_of_case_ind.into();
          let cur_pos = gc::TokenPosition {
            prod: *cur_prod_ref,
            case: cur_case_ref,
            el: cur_el_ref,
          };
          match el {
            /* Continue the current interval of nonterminals. */
            gc::CaseEl::Tok(_) => cur_elements.push(EpsilonGraphVertex::State(cur_pos)),
            /* The current case invokes a subproduction, so is split into two intervals
             * here, using anonymous symbols to keep track of where in
             * this case we jumped off of and where we can jump back
             * onto to satisfy this case of this production. */
            gc::CaseEl::Prod(target_subprod_ref) => {
              /* Generate anonymous steps for the current subprod split. */
              let (pos_anon, neg_anon) = Self::make_pos_neg_anon_steps(
                &mut cur_anon_sym_index,
                &mut anon_step_mapping,
                UnflattenedProdCaseRef::PassThrough,
              );

              /* Generate the interval terminating at the current subprod split. */
              let mut interval_upto_subprod: Vec<EpsilonGraphVertex, Arena> =
                Vec::with_capacity_in(cur_elements.len() + 2, arena.clone());
              /* NB: we empty out the state of `cur_elements` here! */
              interval_upto_subprod.extend(cur_elements.drain(..));
              /* We /end/ this interval with a "start" vertex because this is going
               * /into/ a subproduction! */
              interval_upto_subprod.push(pos_anon);
              interval_upto_subprod.push(EpsilonGraphVertex::Start(*target_subprod_ref));
              /* NB: Mutate the loop state! */
              /* Start a new interval of nonterminals which must come after the current
               * subprod split. */
              /* We /start/ with an "end" vertex because this comes /out/ of a
               * subproduction! */
              cur_elements.push(EpsilonGraphVertex::End(*target_subprod_ref));
              cur_elements.push(neg_anon);
              /* Register the interval we just cut off in the results list. */
              all_intervals_from_this_case.push(ContiguousNonterminalInterval {
                interval: interval_upto_subprod,
                arena: arena.clone(),
              });
            },
          }
        }
        /* Construct the interval of all remaining nonterminals to the end of the
         * production. */
        let mut final_interval: Vec<EpsilonGraphVertex, Arena> =
          Vec::with_capacity_in(cur_elements.len() + 2, arena.clone());
        final_interval.extend(cur_elements.into_iter());
        final_interval.push(neg_case_anon);
        final_interval.push(EpsilonGraphVertex::End(*cur_prod_ref));

        /* Register the interval of all remaining nonterminals in the results list. */
        all_intervals_from_this_case.push(ContiguousNonterminalInterval {
          interval: final_interval,
          arena: arena.clone(),
        });
        /* Return all the intervals from this case. */
        really_all_intervals.extend(all_intervals_from_this_case);
      }
    }
    (
      EpsilonIntervalGraph {
        all_intervals: really_all_intervals,
        anon_step_mapping,
      },
      alphabet,
      token_states,
    )
  }

  #[allow(dead_code)]
  pub fn new(grammar: gb::TokenGrammar<Tok, Arena>) -> Self {
    let (terminals_interval_graph, alphabet, token_states) =
      Self::produce_terminals_interval_graph(grammar);
    let cyclic_graph_decomposition: CyclicGraphDecomposition<Arena> =
      terminals_interval_graph.connect_all_vertices();
    PreprocessedGrammar {
      token_states_mapping: token_states,
      cyclic_graph_decomposition,
      alphabet,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    test_framework::{basic_productions, new_token_position, non_cyclic_productions},
    types::{DefaultHasher, Global},
  };

  #[test]
  fn token_grammar_state_indexing() {
    let prods = non_cyclic_productions();
    let grammar: gb::TokenGrammar<char, Global> = gb::TokenGrammar::new(prods, Global).unwrap();
    assert_eq!(
      grammar
        .token_states
        .into_index_map()
        .into_iter()
        .collect::<Vec<_>>(),
      [
        (
          gc::TokRef(0),
          [
            new_token_position(0, 0, 0),
            new_token_position(1, 0, 0),
            new_token_position(1, 1, 1),
          ]
          .as_ref()
          .to_vec()
        ),
        (
          gc::TokRef(1),
          [new_token_position(0, 0, 1), new_token_position(1, 0, 1)]
            .as_ref()
            .to_vec(),
        ),
      ]
      .iter()
      .cloned()
      .collect::<Vec<_>>(),
    )
  }


  #[test]
  fn terminals_interval_graph() {
    let noncyclic_prods = non_cyclic_productions();
    let noncyclic_grammar: gb::TokenGrammar<char, Global> =
      gb::TokenGrammar::new(noncyclic_prods, Global).unwrap();

    let (noncyclic_interval_graph, _, _) =
      PreprocessedGrammar::produce_terminals_interval_graph(noncyclic_grammar);

    let s_0 = new_token_position(0, 0, 0);
    let s_1 = new_token_position(0, 0, 1);
    let a_prod = gc::ProdRef(0);

    let s_2 = new_token_position(1, 0, 0);
    let s_3 = new_token_position(1, 0, 1);
    let s_4 = new_token_position(1, 1, 1);
    let b_prod = gc::ProdRef(1);

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

    let a_0 = ContiguousNonterminalInterval {
      interval: [
        a_start,
        a_prod_anon_start,
        a_0_0,
        a_0_1,
        a_prod_anon_end,
        a_end,
      ]
      .as_ref()
      .to_vec(),
      arena: Global,
    };
    let b_start_to_a_start_0 = ContiguousNonterminalInterval {
      interval: [
        b_start,
        b_prod_anon_start,
        b_0_0,
        b_0_1,
        b_0_anon_0_start,
        a_start,
      ]
      .as_ref()
      .to_vec(),
      arena: Global,
    };
    let a_end_to_b_end_0 = ContiguousNonterminalInterval {
      interval: [a_end, b_0_anon_0_end, b_prod_anon_end, b_end]
        .as_ref()
        .to_vec(),
      arena: Global,
    };
    let b_start_to_a_start_1 = ContiguousNonterminalInterval {
      interval: [b_start, b_1_anon_0_start, b_1_anon_0_start_2, a_start]
        .as_ref()
        .to_vec(),
      arena: Global,
    };
    let a_end_to_b_end_1 = ContiguousNonterminalInterval {
      interval: [a_end, b_1_anon_0_end_2, b_1_1, b_1_anon_0_end, b_end]
        .as_ref()
        .to_vec(),
      arena: Global,
    };

    assert_eq!(noncyclic_interval_graph, EpsilonIntervalGraph {
      all_intervals: [
        a_0.clone(),
        b_start_to_a_start_0.clone(),
        a_end_to_b_end_0.clone(),
        b_start_to_a_start_1.clone(),
        a_end_to_b_end_1.clone(),
      ]
      .as_ref()
      .to_vec(),
      anon_step_mapping: [
        (
          AnonSym(0),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: gc::ProdRef(0),
            case: gc::CaseRef(0)
          })
        ),
        (
          AnonSym(1),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: gc::ProdRef(1),
            case: gc::CaseRef(0)
          })
        ),
        (AnonSym(2), UnflattenedProdCaseRef::PassThrough),
        (
          AnonSym(3),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: gc::ProdRef(1),
            case: gc::CaseRef(1)
          })
        ),
        (AnonSym(4), UnflattenedProdCaseRef::PassThrough),
      ]
      .iter()
      .cloned()
      .collect::<IndexMap<_, _, Global, DefaultHasher>>(),
    });

    /* Now check for indices. */
    let intervals_by_start_and_end = noncyclic_interval_graph.find_start_end_indices();
    assert_eq!(
      intervals_by_start_and_end,
      [
        (a_prod, StartEndEpsilonIntervals {
          start_epsilons: [a_0.clone()].as_ref().to_vec(),
          end_epsilons: [a_end_to_b_end_0.clone(), a_end_to_b_end_1.clone()]
            .as_ref()
            .to_vec(),
        },),
        (b_prod, StartEndEpsilonIntervals {
          start_epsilons: [b_start_to_a_start_0.clone(), b_start_to_a_start_1.clone()]
            .as_ref()
            .to_vec(),
          end_epsilons: [].as_ref().to_vec(),
        },),
      ]
      .as_ref()
      .to_vec()
      .iter()
      .cloned()
      .collect::<IndexMap<_, _, Global, DefaultHasher>>()
    );

    /* Now check that the transition graph is as we expect. */
    let CyclicGraphDecomposition {
      cyclic_subgraph: merged_stack_cycles,
      pairwise_state_transitions: all_completed_pairs_with_vertices,
      anon_step_mapping,
    } = noncyclic_interval_graph.connect_all_vertices();
    /* There are no stack cycles in the noncyclic graph. */
    assert_eq!(merged_stack_cycles, EpsilonNodeStateSubgraph {
      vertex_mapping: IndexMap::new_in(Global),
      trie_node_universe: [].as_ref().to_vec(),
    });
    assert_eq!(
      anon_step_mapping,
      [
        (
          AnonSym(0),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: gc::ProdRef(0),
            case: gc::CaseRef(0)
          })
        ),
        (
          AnonSym(1),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: gc::ProdRef(1),
            case: gc::CaseRef(0)
          })
        ),
        (AnonSym(2), UnflattenedProdCaseRef::PassThrough),
        (
          AnonSym(3),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: gc::ProdRef(1),
            case: gc::CaseRef(1)
          })
        ),
        (AnonSym(4), UnflattenedProdCaseRef::PassThrough),
      ]
      .iter()
      .cloned()
      .collect::<IndexMap<_, _>>()
    );

    assert_eq!(
      all_completed_pairs_with_vertices,
      [
        /* 1 */
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Start,
            right: LoweredState::Within(s_0)
          },
          interval: ContiguousNonterminalInterval {
            interval: [a_start, a_prod_anon_start, a_0_0].as_ref().to_vec(),
            arena: Global
          },
        },
        /* 2 */
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Start,
            right: LoweredState::Within(s_2)
          },
          interval: ContiguousNonterminalInterval {
            interval: [b_start, b_prod_anon_start, b_0_0].as_ref().to_vec(),
            arena: Global
          },
        },
        /* 3 */
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(s_0),
            right: LoweredState::Within(s_1)
          },
          interval: ContiguousNonterminalInterval {
            interval: [a_0_0, a_0_1].as_ref().to_vec(),
            arena: Global
          },
        },
        /* 4 */
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(s_2),
            right: LoweredState::Within(s_3)
          },
          interval: ContiguousNonterminalInterval {
            interval: [b_0_0, b_0_1].as_ref().to_vec(),
            arena: Global
          },
        },
        /* 5 */
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(s_4),
            right: LoweredState::End
          },
          interval: ContiguousNonterminalInterval {
            interval: [b_1_1, b_1_anon_0_end, b_end].as_ref().to_vec(),
            arena: Global
          },
        },
        /* 6 */
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(s_1),
            right: LoweredState::End
          },
          interval: ContiguousNonterminalInterval {
            interval: [a_0_1, a_prod_anon_end, a_end].as_ref().to_vec(),
            arena: Global
          },
        },
        /* 7 */
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Start,
            right: LoweredState::Within(s_0)
          },
          interval: ContiguousNonterminalInterval {
            interval: [
              b_start,
              b_1_anon_0_start,
              b_1_anon_0_start_2,
              a_start,
              a_prod_anon_start,
              a_0_0
            ]
            .as_ref()
            .to_vec(),
            arena: Global,
          }
        },
        /* 8 */
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(s_1),
            right: LoweredState::Within(s_4)
          },
          interval: ContiguousNonterminalInterval {
            interval: [a_0_1, a_prod_anon_end, a_end, b_1_anon_0_end_2, b_1_1]
              .as_ref()
              .to_vec(),
            arena: Global
          },
        },
        /* 9 */
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(s_3),
            right: LoweredState::Within(s_0)
          },
          interval: ContiguousNonterminalInterval {
            interval: [b_0_1, b_0_anon_0_start, a_start, a_prod_anon_start, a_0_0]
              .as_ref()
              .to_vec(),
            arena: Global
          },
        },
        /* 10 */
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(s_1),
            right: LoweredState::End
          },
          interval: ContiguousNonterminalInterval {
            interval: [
              a_0_1,
              a_prod_anon_end,
              a_end,
              b_0_anon_0_end,
              b_prod_anon_end,
              b_end
            ]
            .as_ref()
            .to_vec(),
            arena: Global
          },
        },
      ]
      .as_ref()
      .to_vec()
    );

    /* Now do the same, but for `basic_productions()`. */
    /* TODO: test `.find_start_end_indices()` and `.connect_all_vertices()` here
     * too! */
    let prods = basic_productions();
    let grammar: gb::TokenGrammar<char, Global> = gb::TokenGrammar::new(prods, Global).unwrap();
    let (interval_graph, _, _) = PreprocessedGrammar::produce_terminals_interval_graph(grammar);
    assert_eq!(&interval_graph, &EpsilonIntervalGraph {
      all_intervals: [
        ContiguousNonterminalInterval {
          interval: [
            EpsilonGraphVertex::Start(gc::ProdRef(0)),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(0))),
            EpsilonGraphVertex::State(new_token_position(0, 0, 0)),
            EpsilonGraphVertex::State(new_token_position(0, 0, 1)),
            EpsilonGraphVertex::State(new_token_position(0, 0, 2)),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(0))),
            EpsilonGraphVertex::End(gc::ProdRef(0)),
          ]
          .as_ref()
          .to_vec(),
          arena: Global
        },
        ContiguousNonterminalInterval {
          interval: [
            EpsilonGraphVertex::Start(gc::ProdRef(0)),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(1))),
            EpsilonGraphVertex::State(new_token_position(0, 1, 0)),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(2))),
            EpsilonGraphVertex::Start(gc::ProdRef(0)),
          ]
          .as_ref()
          .to_vec(),
          arena: Global
        },
        ContiguousNonterminalInterval {
          interval: [
            EpsilonGraphVertex::End(gc::ProdRef(0)),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(2))),
            EpsilonGraphVertex::State(new_token_position(0, 1, 2)),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(1))),
            EpsilonGraphVertex::End(gc::ProdRef(0)),
          ]
          .as_ref()
          .to_vec(),
          arena: Global
        },
        ContiguousNonterminalInterval {
          interval: [
            EpsilonGraphVertex::Start(gc::ProdRef(0)),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(3))),
            EpsilonGraphVertex::State(new_token_position(0, 2, 0)),
            EpsilonGraphVertex::State(new_token_position(0, 2, 1)),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(4))),
            EpsilonGraphVertex::Start(gc::ProdRef(1)),
          ]
          .as_ref()
          .to_vec(),
          arena: Global
        },
        ContiguousNonterminalInterval {
          interval: [
            EpsilonGraphVertex::End(gc::ProdRef(1)),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(4))),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(3))),
            EpsilonGraphVertex::End(gc::ProdRef(0)),
          ]
          .as_ref()
          .to_vec(),
          arena: Global
        },
        ContiguousNonterminalInterval {
          interval: [
            EpsilonGraphVertex::Start(gc::ProdRef(1)),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(5))),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(6))),
            EpsilonGraphVertex::Start(gc::ProdRef(0)),
          ]
          .as_ref()
          .to_vec(),
          arena: Global
        },
        ContiguousNonterminalInterval {
          interval: [
            EpsilonGraphVertex::End(gc::ProdRef(0)),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(6))),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(5))),
            EpsilonGraphVertex::End(gc::ProdRef(1)),
          ]
          .as_ref()
          .to_vec(),
          arena: Global
        },
        ContiguousNonterminalInterval {
          interval: [
            EpsilonGraphVertex::Start(gc::ProdRef(1)),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(7))),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(8))),
            EpsilonGraphVertex::Start(gc::ProdRef(1)),
          ]
          .as_ref()
          .to_vec(),
          arena: Global
        },
        ContiguousNonterminalInterval {
          interval: [
            EpsilonGraphVertex::End(gc::ProdRef(1)),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(8))),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(7))),
            EpsilonGraphVertex::End(gc::ProdRef(1)),
          ]
          .as_ref()
          .to_vec(),
          arena: Global
        },
        ContiguousNonterminalInterval {
          interval: [
            EpsilonGraphVertex::Start(gc::ProdRef(1)),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(9))),
            EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(10))),
            EpsilonGraphVertex::Start(gc::ProdRef(0)),
          ]
          .as_ref()
          .to_vec(),
          arena: Global,
        },
        ContiguousNonterminalInterval {
          interval: [
            EpsilonGraphVertex::End(gc::ProdRef(0)),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(10))),
            EpsilonGraphVertex::State(new_token_position(1, 2, 1)),
            EpsilonGraphVertex::State(new_token_position(1, 2, 2)),
            EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(9))),
            EpsilonGraphVertex::End(gc::ProdRef(1)),
          ]
          .as_ref()
          .to_vec(),
          arena: Global,
        }
      ]
      .as_ref()
      .to_vec(),
      anon_step_mapping: [
        (
          AnonSym(0),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: gc::ProdRef(0),
            case: gc::CaseRef(0)
          })
        ),
        (
          AnonSym(1),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: gc::ProdRef(0),
            case: gc::CaseRef(1)
          })
        ),
        (AnonSym(2), UnflattenedProdCaseRef::PassThrough),
        (
          AnonSym(3),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: gc::ProdRef(0),
            case: gc::CaseRef(2)
          })
        ),
        (AnonSym(4), UnflattenedProdCaseRef::PassThrough),
        (
          AnonSym(5),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: gc::ProdRef(1),
            case: gc::CaseRef(0)
          })
        ),
        (AnonSym(6), UnflattenedProdCaseRef::PassThrough),
        (
          AnonSym(7),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: gc::ProdRef(1),
            case: gc::CaseRef(1)
          })
        ),
        (AnonSym(8), UnflattenedProdCaseRef::PassThrough),
        (
          AnonSym(9),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: gc::ProdRef(1),
            case: gc::CaseRef(2)
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
    let grammar: gb::TokenGrammar<char, Global> = gb::TokenGrammar::new(prods, Global).unwrap();
    let preprocessed_grammar = PreprocessedGrammar::new(grammar);
    let first_a = new_token_position(0, 0, 0);
    let first_b = new_token_position(0, 0, 1);
    let second_a = new_token_position(1, 0, 0);
    let second_b = new_token_position(1, 0, 1);
    let third_a = new_token_position(1, 1, 1);
    let a_prod = gc::ProdRef(0);
    let b_prod = gc::ProdRef(1);
    assert_eq!(
      preprocessed_grammar.token_states_mapping.into_index_map(),
      [
        (
          gc::TokRef(0),
          [first_a, second_a, third_a].as_ref().to_vec()
        ),
        (gc::TokRef(1), [first_b, second_b].as_ref().to_vec()),
      ]
      .iter()
      .cloned()
      .collect::<IndexMap<_, _, Global, DefaultHasher>>(),
    );

    let other_cyclic_graph_decomposition = CyclicGraphDecomposition {
      cyclic_subgraph: EpsilonNodeStateSubgraph {
        vertex_mapping: IndexMap::<_, _, Global, DefaultHasher>::new_in(Global),
        trie_node_universe: Vec::new_in(Global),
      },
      pairwise_state_transitions: [
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Start,
            right: LoweredState::Within(new_token_position(a_prod.0, 0, 0)),
          },
          interval: ContiguousNonterminalInterval {
            interval: [
              EpsilonGraphVertex::Start(a_prod),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(0))),
              EpsilonGraphVertex::State(new_token_position(a_prod.0, 0, 0)),
            ]
            .as_ref()
            .to_vec(),
            arena: Global,
          },
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Start,
            right: LoweredState::Within(new_token_position(b_prod.0, 0, 0)),
          },
          interval: ContiguousNonterminalInterval {
            interval: [
              EpsilonGraphVertex::Start(b_prod),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(1))),
              EpsilonGraphVertex::State(new_token_position(b_prod.0, 0, 0)),
            ]
            .as_ref()
            .to_vec(),
            arena: Global,
          },
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(new_token_position(a_prod.0, 0, 0)),
            right: LoweredState::Within(new_token_position(a_prod.0, 0, 1)),
          },
          interval: ContiguousNonterminalInterval {
            interval: [
              EpsilonGraphVertex::State(new_token_position(a_prod.0, 0, 0)),
              EpsilonGraphVertex::State(new_token_position(a_prod.0, 0, 1)),
            ]
            .as_ref()
            .to_vec(),
            arena: Global,
          },
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(new_token_position(b_prod.0, 0, 0)),
            right: LoweredState::Within(new_token_position(b_prod.0, 0, 1)),
          },
          interval: ContiguousNonterminalInterval {
            interval: [
              EpsilonGraphVertex::State(new_token_position(b_prod.0, 0, 0)),
              EpsilonGraphVertex::State(new_token_position(b_prod.0, 0, 1)),
            ]
            .as_ref()
            .to_vec(),
            arena: Global,
          },
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(new_token_position(b_prod.0, 1, 1)),
            right: LoweredState::End,
          },
          interval: ContiguousNonterminalInterval {
            interval: [
              EpsilonGraphVertex::State(new_token_position(b_prod.0, 1, 1)),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(3))),
              EpsilonGraphVertex::End(b_prod),
            ]
            .as_ref()
            .to_vec(),
            arena: Global,
          },
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(new_token_position(a_prod.0, 0, 1)),
            right: LoweredState::End,
          },
          interval: ContiguousNonterminalInterval {
            interval: [
              EpsilonGraphVertex::State(new_token_position(a_prod.0, 0, 1)),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(0))),
              EpsilonGraphVertex::End(a_prod),
            ]
            .as_ref()
            .to_vec(),
            arena: Global,
          },
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Start,
            right: LoweredState::Within(new_token_position(a_prod.0, 0, 0)),
          },
          interval: ContiguousNonterminalInterval {
            interval: [
              EpsilonGraphVertex::Start(b_prod),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(3))),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(4))),
              EpsilonGraphVertex::Start(a_prod),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(0))),
              EpsilonGraphVertex::State(new_token_position(a_prod.0, 0, 0)),
            ]
            .as_ref()
            .to_vec(),
            arena: Global,
          },
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(new_token_position(a_prod.0, 0, 1)),
            right: LoweredState::Within(new_token_position(b_prod.0, 1, 1)),
          },
          interval: ContiguousNonterminalInterval {
            interval: [
              EpsilonGraphVertex::State(new_token_position(a_prod.0, 0, 1)),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(0))),
              EpsilonGraphVertex::End(a_prod),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(4))),
              EpsilonGraphVertex::State(new_token_position(b_prod.0, 1, 1)),
            ]
            .as_ref()
            .to_vec(),
            arena: Global,
          },
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(new_token_position(b_prod.0, 0, 1)),
            right: LoweredState::Within(new_token_position(a_prod.0, 0, 0)),
          },
          interval: ContiguousNonterminalInterval {
            interval: [
              EpsilonGraphVertex::State(new_token_position(b_prod.0, 0, 1)),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(2))),
              EpsilonGraphVertex::Start(a_prod),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(0))),
              EpsilonGraphVertex::State(new_token_position(a_prod.0, 0, 0)),
            ]
            .as_ref()
            .to_vec(),
            arena: Global,
          },
        },
        CompletedStatePairWithVertices {
          state_pair: StatePair {
            left: LoweredState::Within(new_token_position(a_prod.0, 0, 1)),
            right: LoweredState::End,
          },
          interval: ContiguousNonterminalInterval {
            interval: [
              EpsilonGraphVertex::State(new_token_position(a_prod.0, 0, 1)),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(0))),
              EpsilonGraphVertex::End(a_prod),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(2))),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(1))),
              EpsilonGraphVertex::End(b_prod),
            ]
            .as_ref()
            .to_vec(),
            arena: Global,
          },
        },
      ]
      .as_ref()
      .to_vec(),
      anon_step_mapping: [
        (
          AnonSym(0),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: a_prod,
            case: gc::CaseRef(0),
          }),
        ),
        (
          AnonSym(1),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: b_prod,
            case: gc::CaseRef(0),
          }),
        ),
        (AnonSym(2), UnflattenedProdCaseRef::PassThrough),
        (
          AnonSym(3),
          UnflattenedProdCaseRef::Case(gc::ProdCaseRef {
            prod: b_prod,
            case: gc::CaseRef(1),
          }),
        ),
        (AnonSym(4), UnflattenedProdCaseRef::PassThrough),
      ]
      .iter()
      .cloned()
      .collect::<IndexMap<_, _, Global, DefaultHasher>>(),
    };

    assert_eq!(
      preprocessed_grammar.cyclic_graph_decomposition,
      other_cyclic_graph_decomposition,
    );
  }

  #[test]
  fn cyclic_transition_graph() {
    let prods = basic_productions();
    let grammar: gb::TokenGrammar<char, Global> = gb::TokenGrammar::new(prods, Global).unwrap();
    let preprocessed_grammar = PreprocessedGrammar::new(grammar);

    let first_a = new_token_position(0, 0, 0);
    let second_a = new_token_position(0, 1, 0);

    let first_b = new_token_position(0, 0, 1);
    let second_b = new_token_position(0, 2, 0);
    let third_b = new_token_position(1, 2, 1);

    let first_c = new_token_position(0, 0, 2);
    let second_c = new_token_position(0, 1, 2);
    let third_c = new_token_position(0, 2, 1);
    let fourth_c = new_token_position(1, 2, 2);

    let a_prod = gc::ProdRef(0);
    let b_prod = gc::ProdRef(1);
    let _c_prod = gc::ProdRef(2); /* unused */

    assert_eq!(
      preprocessed_grammar.token_states_mapping.into_index_map(),
      [
        (gc::TokRef(0), [first_a, second_a].as_ref().to_vec()),
        (
          gc::TokRef(1),
          [first_b, second_b, third_b].as_ref().to_vec()
        ),
        (
          gc::TokRef(2),
          [first_c, second_c, third_c, fourth_c].as_ref().to_vec()
        ),
      ]
      .iter()
      .cloned()
      .collect::<IndexMap<_, _, Global, DefaultHasher>>()
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
          EpsilonGraphVertex::State(new_token_position(a_prod.0, 1, 0)),
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
          EpsilonGraphVertex::State(new_token_position(a_prod.0, 1, 2)),
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
      .collect::<IndexMap<_, _, Global, DefaultHasher>>()
    );

    let all_trie_nodes: &[StackTrieNode<Global>] = preprocessed_grammar
      .cyclic_graph_decomposition
      .cyclic_subgraph
      .trie_node_universe
      .as_ref();
    assert_eq!(
      all_trie_nodes,
      [
        /* 0 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Named(StackStep::Positive(StackSym(
              b_prod
            )))]
            .as_ref()
            .to_vec()
          ),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(1))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(2))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 1 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(7)))]
              .as_ref()
              .to_vec()
          ),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(2))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(0))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 2 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(8)))]
              .as_ref()
              .to_vec()
          ),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(0))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(1))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 3 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Named(StackStep::Negative(StackSym(
              b_prod
            )))]
            .as_ref()
            .to_vec()
          ),
          next_nodes: [
            StackTrieNextEntry::Incomplete(TrieNodeRef(4)),
            StackTrieNextEntry::Incomplete(TrieNodeRef(14))
          ]
          .iter()
          .cloned()
          .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [
            StackTrieNextEntry::Incomplete(TrieNodeRef(5)),
            StackTrieNextEntry::Incomplete(TrieNodeRef(17))
          ]
          .iter()
          .cloned()
          .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 4 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(8)))]
              .as_ref()
              .to_vec()
          ),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(5))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(3))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 5 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(7)))]
              .as_ref()
              .to_vec()
          ),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(3))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(4))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 6 */
        StackTrieNode {
          stack_diff: StackDiffSegment(Vec::new()),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(7))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(9))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 7 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(2)))]
              .as_ref()
              .to_vec()
          ),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(8))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(6))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 8 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Named(StackStep::Positive(StackSym(
              a_prod
            )))]
            .as_ref()
            .to_vec()
          ),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(9))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(7))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 9 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(1)))]
              .as_ref()
              .to_vec()
          ),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(6))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(8))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 10 */
        StackTrieNode {
          stack_diff: StackDiffSegment(Vec::new()),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(11))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(13))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 11 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(1)))]
              .as_ref()
              .to_vec()
          ),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(12))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(10))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 12 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Named(StackStep::Negative(StackSym(
              a_prod
            )))]
            .as_ref()
            .to_vec()
          ),
          next_nodes: [
            StackTrieNextEntry::Incomplete(TrieNodeRef(13)),
            StackTrieNextEntry::Incomplete(TrieNodeRef(16))
          ]
          .iter()
          .cloned()
          .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [
            StackTrieNextEntry::Incomplete(TrieNodeRef(11)),
            StackTrieNextEntry::Incomplete(TrieNodeRef(15))
          ]
          .iter()
          .cloned()
          .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 13 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(2)))]
              .as_ref()
              .to_vec()
          ),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(10))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(12))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 14 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(4)))]
              .as_ref()
              .to_vec()
          ),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(15))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(3))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 15 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(3)))]
              .as_ref()
              .to_vec()
          ),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(12))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(14))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 16 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(6)))]
              .as_ref()
              .to_vec()
          ),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(17))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(12))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        },
        /* 17 */
        StackTrieNode {
          stack_diff: StackDiffSegment(
            [NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(5)))]
              .as_ref()
              .to_vec()
          ),
          next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(3))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>(),
          prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(16))]
            .iter()
            .cloned()
            .collect::<IndexSet<_, Global, DefaultHasher>>()
        }
      ]
      .as_ref()
    );
  }
}
