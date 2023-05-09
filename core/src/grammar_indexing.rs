/*
 * Description: Implementation for getting a PreprocessedGrammar.
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

//! Implementation for getting a [PreprocessedGrammar].
//!
//! This phase is **extremely complex** and **may not work yet!** This module
//! detects all *stack cycles* and *contiguous state
//! transitions* so that later in [super::parsing] we can ensure the parsing
//! will always terminate by *bounding context-sensitivity* **(TODO: link to
//! paper!)**.
//!
//! *Implementation Note: Performance doesn't matter here.*

use crate::lowering_to_indices::{grammar_building as gb, graph_coordinates as gc};

use displaydoc::Display;
use indexmap::{IndexMap, IndexSet};

use core::fmt;

/* TODO: what does "unflattened" mean? PassThrough appears relevant for
 * post-parse reconstruction? */
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum UnflattenedProdCaseRef {
  PassThrough,
  Case(gc::ProdCaseRef),
}

/// ~{0}~
#[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash)]
pub struct StackSym(pub gc::ProdRef);

/* TODO: is this used? */
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

#[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StackStep {
  /// +{0}
  Positive(StackSym),
  /// -{0}
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

/// !{0}!
#[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash)]
pub struct AnonSym(pub usize);

#[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AnonStep {
  /// +{0}
  Positive(AnonSym),
  /// -{0}
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

/* TODO: omfg document this vvv useful concept!!! */
#[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash)]
pub enum LoweredState {
  /// Start
  Start(gc::ProdRef),
  /// End
  End(gc::ProdRef),
  /// {0}
  Within(gc::TokenPosition),
}

impl LoweredState {
  pub fn from_vertex(vtx: EpsilonGraphVertex) -> Option<Self> {
    match vtx {
      EpsilonGraphVertex::Start(x) => Some(LoweredState::Start(x)),
      EpsilonGraphVertex::End(x) => Some(LoweredState::End(x)),
      EpsilonGraphVertex::State(pos) => Some(LoweredState::Within(pos)),
      EpsilonGraphVertex::Anon(_) => None,
    }
  }

  pub fn into_vertex(self) -> EpsilonGraphVertex {
    match self {
      Self::Start(x) => EpsilonGraphVertex::Start(x),
      Self::End(x) => EpsilonGraphVertex::End(x),
      Self::Within(x) => EpsilonGraphVertex::State(x),
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
#[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash)]
pub enum EpsilonGraphVertex {
  /// Start({0})
  Start(gc::ProdRef),
  /// End({0})
  End(gc::ProdRef),
  /// {0}
  Anon(AnonStep),
  /// {0}
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
      /* TODO: ^is the above true? */
      EpsilonGraphVertex::State(_) => None,
    }
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StackStepError {
  StepConcatenationError(NamedOrAnonStep, NamedOrAnonStep),
}

#[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NamedOrAnonStep {
  /// {0}
  Named(StackStep),
  /// {0}
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StackDiffSegment(pub Vec<NamedOrAnonStep>);

impl fmt::Display for StackDiffSegment {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    let mut s = String::new();
    s.push('[');
    for step in self.0.iter() {
      s.push_str(format!("{}", step).as_str());
    }
    s.push(']');
    write!(f, "{}", s)
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TrieNodeRef(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackTrieNode {
  pub step: Option<NamedOrAnonStep>,
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

impl StackTrieNode {
  fn bare(vtx: EpsilonGraphVertex) -> Self {
    StackTrieNode {
      step: vtx.get_step(),
      next_nodes: IndexSet::new(),
      prev_nodes: IndexSet::new(),
    }
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StackTrieNextEntry {
  Completed(LoweredState),
  Incomplete(TrieNodeRef),
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
      trie_node_universe: Vec::new(),
    }
  }

  fn get_trie(&mut self, node_ref: TrieNodeRef) -> &mut StackTrieNode {
    let TrieNodeRef(node_index) = node_ref;
    self
      .trie_node_universe
      .get_mut(node_index)
      .expect("trie node ref should never have been constructed out of bounds?")
  }

  fn trie_ref_for_vertex(&mut self, vtx: &EpsilonGraphVertex) -> TrieNodeRef {
    let basic_node = StackTrieNode::bare(*vtx);
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
      self.get_trie(trie_node_ref_for_vertex).step,
      basic_node.step
    );
    trie_node_ref_for_vertex
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpsilonGraphCase {
  pub interval: Vec<EpsilonGraphVertex>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CyclicGraphDecomposition {
  /* TODO: this isn't just the "cyclic" subgraph, it's needed to interpret the value of
   * pairwise_state_transitions! */
  pub trie_graph: EpsilonNodeStateSubgraph,
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
  pub all_cases: Vec<EpsilonGraphCase>,
}

impl EpsilonIntervalGraph {
  fn find_start_end_indices(
    all_cases: Vec<EpsilonGraphCase>,
  ) -> IndexMap<gc::ProdRef, StartEndEpsilonIntervals> {
    let mut epsilon_subscripts_index: IndexMap<gc::ProdRef, StartEndEpsilonIntervals> =
      IndexMap::new();
    for interval in all_cases.iter() {
      let EpsilonGraphCase {
        interval: vertices, ..
      } = interval.clone();
      assert!(
        vertices.len() >= 2,
        "we should always have a start and end node!"
      );
      match vertices.first().unwrap() {
          EpsilonGraphVertex::Start(start_prod_ref) => {
            let intervals_for_this_prod = epsilon_subscripts_index.entry(start_prod_ref.clone())
              .or_insert(StartEndEpsilonIntervals::new());
            (*intervals_for_this_prod).start_epsilons.push(interval.clone());
          },
          EpsilonGraphVertex::End(end_prod_ref) => {
            let intervals_for_this_prod = epsilon_subscripts_index.entry(end_prod_ref.clone())
              .or_insert(StartEndEpsilonIntervals::new());
            (*intervals_for_this_prod).end_epsilons.push(interval.clone());
          },
          _ => unreachable!("the beginning of an interval should always be a start (epsilon) or end (epsilon prime) vertex"),
      }
      match vertices.last().unwrap() {
        EpsilonGraphVertex::Start(_) | EpsilonGraphVertex::End(_) => { /* no-op */ },
        _ => {
          unreachable!(
            "all intervals should end in a start (epsilon) or end (epsilon prime) vertex"
          );
        },
      }
    }
    epsilon_subscripts_index
  }

  pub fn connect_all_vertices(self) -> CyclicGraphDecomposition {
    let EpsilonIntervalGraph { all_cases } = self;
    let all_cases = Self::find_start_end_indices(all_cases);

    let trie_graph: EpsilonNodeStateSubgraph = {
      let mut ret = EpsilonNodeStateSubgraph::new();
      for intervals in all_cases.into_values() {
        let StartEndEpsilonIntervals {
          start_epsilons,
          end_epsilons,
        } = intervals;
        /* Process all the epsilon intervals together. */
        for EpsilonGraphCase {
          interval: cur_interval,
        } in start_epsilons.into_iter().chain(end_epsilons.into_iter())
        {
          /* Set up a directed graph with each edge doubled. We reach forward into the next element
           * of cur_interval, so we avoid processing the final element of cur_interval in this
           * loop. */
          for (cur_vtx_index, cur_vtx) in
            cur_interval[..(cur_interval.len() - 1)].iter().enumerate()
          {
            let cur_trie_ref = ret.trie_ref_for_vertex(cur_vtx);
            let (next_trie_ref, next_vertex) = {
              let next_vtx_index = cur_vtx_index + 1;
              assert!(next_vtx_index >= 1);
              assert!(next_vtx_index <= (cur_interval.len() - 1));
              let next_vertex = cur_interval[next_vtx_index];
              let next_trie_ref = ret.trie_ref_for_vertex(&next_vertex);
              (next_trie_ref, next_vertex)
            };

            /* Add a forward edge on cur_trie_ref -> next_trie_ref */
            {
              let cur_trie = ret.get_trie(cur_trie_ref);
              let next_entry = if let Some(lowered_state) = LoweredState::from_vertex(next_vertex) {
                StackTrieNextEntry::Completed(lowered_state)
              } else {
                StackTrieNextEntry::Incomplete(next_trie_ref)
              };
              cur_trie.next_nodes.insert(next_entry);
            }
            /* Add a backward edge on next_trie_ref -> cur_trie_ref */
            {
              let next_trie = ret.get_trie(next_trie_ref);
              let prev_entry =
                if let Some(lowered_state) = LoweredState::from_vertex(cur_vtx.clone()) {
                  StackTrieNextEntry::Completed(lowered_state)
                } else {
                  StackTrieNextEntry::Incomplete(cur_trie_ref)
                };
              next_trie.prev_nodes.insert(prev_entry);
            }
          }
        }
      }
      ret
    };

    CyclicGraphDecomposition { trie_graph }
  }
}

/// The intervals of nonterminals which begin at epsilon (start) or epsilon
/// prime (end) for some ProdRef.
///
/// This is only a concept in the interval graph and is flattened to a single
/// epsilon/epsilon prime when the PreprocessedGrammar is finally
/// constructed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartEndEpsilonIntervals {
  pub start_epsilons: Vec<EpsilonGraphCase>,
  pub end_epsilons: Vec<EpsilonGraphCase>,
}

impl StartEndEpsilonIntervals {
  fn new() -> Self {
    StartEndEpsilonIntervals {
      start_epsilons: Vec::new(),
      end_epsilons: Vec::new(),
    }
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IndexedTrieNodeRef(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedTrieNode {
  pub next_nodes: IndexMap<Option<NamedOrAnonStep>, Vec<IndexedTrieNodeRef>>,
  pub prev_nodes: IndexMap<Option<NamedOrAnonStep>, Vec<IndexedTrieNodeRef>>,
}

impl IndexedTrieNode {
  fn bare() -> Self {
    Self {
      next_nodes: IndexMap::new(),
      prev_nodes: IndexMap::new(),
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedTrie {
  pub vertex_mapping: IndexMap<EpsilonGraphVertex, IndexedTrieNodeRef>,
  pub trie_node_universe: Vec<IndexedTrieNode>,
}

impl IndexedTrie {
  fn new() -> Self {
    Self {
      vertex_mapping: IndexMap::new(),
      trie_node_universe: Vec::new(),
    }
  }

  fn get_trie(&mut self, node_ref: IndexedTrieNodeRef) -> &mut IndexedTrieNode {
    let IndexedTrieNodeRef(node_index) = node_ref;
    self
      .trie_node_universe
      .get_mut(node_index)
      .expect("indexed trie node was out of bounds somehow?")
  }

  fn trie_ref_for_vertex(&mut self, vtx: &EpsilonGraphVertex) -> IndexedTrieNodeRef {
    let basic_node = IndexedTrieNode::bare();
    let trie_node_ref_for_vertex = if let Some(x) = self.vertex_mapping.get(vtx) {
      *x
    } else {
      let next_ref = IndexedTrieNodeRef(self.trie_node_universe.len());
      self.trie_node_universe.push(basic_node.clone());
      self.vertex_mapping.insert(*vtx, next_ref);
      next_ref
    };
    trie_node_ref_for_vertex
  }

  pub fn from_epsilon_graph(graph: EpsilonNodeStateSubgraph) -> Self {
    let EpsilonNodeStateSubgraph {
      vertex_mapping,
      trie_node_universe,
    } = graph;
    let reverse_vertex_mapping: IndexMap<TrieNodeRef, EpsilonGraphVertex> = vertex_mapping
      .iter()
      .map(|(x, y)| (y.clone(), x.clone()))
      .collect();
    let mut ret = Self::new();

    for (vtx, TrieNodeRef(old_trie_node_ref)) in vertex_mapping.into_iter() {
      let new_trie_ref = ret.trie_ref_for_vertex(&vtx);
      let IndexedTrieNode {
        next_nodes: mut new_next_nodes,
        prev_nodes: mut new_prev_nodes,
      } = ret.get_trie(new_trie_ref).clone();

      let StackTrieNode {
        next_nodes: old_next_nodes,
        prev_nodes: old_prev_nodes,
        ..
      } = trie_node_universe[old_trie_node_ref].clone();

      /* Copy over next nodes. */
      for old_next_entry in old_next_nodes.into_iter() {
        match old_next_entry {
          StackTrieNextEntry::Completed(lowered_state) => {
            let ref mut next_nodes = new_next_nodes.entry(None).or_insert_with(Vec::new);
            let old_next_vertex = lowered_state.into_vertex();
            let new_next_trie_ref = ret.trie_ref_for_vertex(&old_next_vertex);
            next_nodes.push(new_next_trie_ref);
          },
          StackTrieNextEntry::Incomplete(TrieNodeRef(old_next_node_ref)) => {
            let StackTrieNode { step, .. } = trie_node_universe[old_next_node_ref];
            let ref mut next_nodes = new_next_nodes.entry(step.clone()).or_insert_with(Vec::new);
            let old_next_vertex = reverse_vertex_mapping
              .get(&TrieNodeRef(old_next_node_ref))
              .unwrap();
            let new_next_trie_ref = ret.trie_ref_for_vertex(&old_next_vertex);
            next_nodes.push(new_next_trie_ref);
          },
        }
      }
      /* Copy over prev nodes. */
      for old_prev_entry in old_prev_nodes.into_iter() {
        match old_prev_entry {
          StackTrieNextEntry::Completed(lowered_state) => {
            let ref mut prev_nodes = new_prev_nodes.entry(None).or_insert_with(Vec::new);
            let old_prev_vertex = lowered_state.into_vertex();
            let new_prev_trie_ref = ret.trie_ref_for_vertex(&old_prev_vertex);
            prev_nodes.push(new_prev_trie_ref);
          },
          StackTrieNextEntry::Incomplete(TrieNodeRef(old_prev_node_ref)) => {
            let StackTrieNode { step, .. } = trie_node_universe[old_prev_node_ref];
            let ref mut prev_nodes = new_prev_nodes.entry(step.clone()).or_insert_with(Vec::new);
            let old_prev_vertex = reverse_vertex_mapping
              .get(&TrieNodeRef(old_prev_node_ref))
              .unwrap();
            let new_prev_trie_ref = ret.trie_ref_for_vertex(&old_prev_vertex);
            prev_nodes.push(new_prev_trie_ref);
          },
        }
      }

      *ret.get_trie(new_trie_ref) = IndexedTrieNode {
        next_nodes: new_next_nodes,
        prev_nodes: new_prev_nodes,
      };
    }

    ret
  }
}

/// A [TokenGrammar][gb::TokenGrammar] after being parsed for cycles.
///
/// There is no intentionally no reference to any
/// [TokenGrammar][gb::TokenGrammar], in the hope that it becomes easier to have
/// the runtime we want just fall out of the code without too much work.
#[derive(Debug, Clone)]
pub struct PreprocessedGrammar<Tok> {
  /// `A: T x T -> {S}^+_-`
  ///
  /// where `{S}^+_-` (LaTeX formatting) is ordered sequences of signed
  /// stack symbols!
  pub graph_transitions: IndexedTrie,
  /// `M: T -> {Q}`, where `{Q}` is sets of states!
  ///
  /// These don't need to be quick to access or otherwise optimized for the
  /// algorithm until we create a `Parse` -- these are chosen to reduce
  /// redundancy.
  pub token_states_mapping: gb::InternedLookupTable<Tok, gc::TokRef, gc::TokenPosition>,
}

impl<Tok> PreprocessedGrammar<Tok> {
  /// TODO: document this great method!!!
  pub(crate) fn produce_terminals_interval_graph(
    grammar: gb::TokenGrammar<Tok>,
  ) -> (
    EpsilonIntervalGraph,
    gb::InternedLookupTable<Tok, gc::TokRef, gc::TokenPosition>,
  ) {
    fn make_pos_neg_anon_steps(cur_index: &mut usize) -> (EpsilonGraphVertex, EpsilonGraphVertex) {
      let the_sym = AnonSym(*cur_index);
      *cur_index += 1;
      (
        EpsilonGraphVertex::Anon(AnonStep::Positive(the_sym)),
        EpsilonGraphVertex::Anon(AnonStep::Negative(the_sym)),
      )
    }

    /* We would like to just accept a DetokenizedProductions here, but we call
     * this method directly in testing, and without the whole grammar object
     * the type ascription is ambiguous. */
    // TODO: what is "type ascription" referring to here^ lol
    let gb::TokenGrammar {
      graph: production_graph,
      tokens,
      groups,
    } = grammar;
    let prods = production_graph.into_index_map();
    /* We would really like to use .flat_map()s here, but it's not clear how to
     * do that while mutating the global `cur_anon_sym_index` value. When
     * `move` is used on the inner loop, the value of `cur_anon_sym_index`
     * mysteriously gets reset, even if `move` is also used on the
     * outer loop. */
    let mut cur_anon_sym_index: usize = 0;
    let mut really_all_intervals: Vec<EpsilonGraphCase> = Vec::new();
    for (cur_prod_ref, the_prod) in prods.iter() {
      let gb::Production(cases) = the_prod;
      for (case_ind, the_case) in cases.iter().enumerate() {
        let cur_case_ref: gc::CaseRef = case_ind.into();
        let gb::Case(elements_of_case) = the_case;
        let mut all_intervals_from_this_case: Vec<EpsilonGraphCase> = Vec::new();

        /* NB: make an anon sym whenever stepping onto a case! */
        let (pos_case_anon, neg_case_anon) = make_pos_neg_anon_steps(&mut cur_anon_sym_index);

        let mut cur_elements: Vec<EpsilonGraphVertex> = Vec::new();
        cur_elements.push(EpsilonGraphVertex::Start(*cur_prod_ref));
        cur_elements.push(pos_case_anon);

        for (element_of_case_ind, el) in elements_of_case.iter().enumerate() {
          let cur_el_ref: gc::CaseElRef = element_of_case_ind.into();
          let cur_pos = gc::TokenPosition {
            prod: *cur_prod_ref,
            case: cur_case_ref,
            el: cur_el_ref,
          };

          /* Helper methods! */
          fn process_token(cur_pos: gc::TokenPosition, cur_elements: &mut Vec<EpsilonGraphVertex>) {
            cur_elements.push(EpsilonGraphVertex::State(cur_pos));
          }
          fn process_prod_ref(
            cur_anon_sym_index: &mut usize,
            cur_elements: &mut Vec<EpsilonGraphVertex>,
            all_intervals_from_this_case: &mut Vec<EpsilonGraphCase>,
            target_subprod_ref: gc::ProdRef,
          ) {
            /* Generate anonymous steps for the current subprod split. */
            let (pos_anon, neg_anon) = make_pos_neg_anon_steps(cur_anon_sym_index);

            /* Generate the interval terminating at the current subprod split. */
            let mut interval_upto_subprod: Vec<EpsilonGraphVertex> =
              Vec::with_capacity(cur_elements.len() + 2);
            /* NB: we empty out the state of `cur_elements` here! */
            interval_upto_subprod.extend(cur_elements.drain(..));
            /* We /end/ this interval with a "start" vertex because this is going
             * /into/ a subproduction! */
            interval_upto_subprod.push(pos_anon);
            interval_upto_subprod.push(EpsilonGraphVertex::Start(target_subprod_ref));
            /* NB: Mutate the loop state! */
            /* Start a new interval of nonterminals which must come after the current
             * subprod split. */
            /* We /start/ with an "end" vertex because this comes /out/ of a
             * subproduction! */
            cur_elements.push(EpsilonGraphVertex::End(target_subprod_ref));
            cur_elements.push(neg_anon);
            /* Register the interval we just cut off in the results list. */
            all_intervals_from_this_case.push(EpsilonGraphCase {
              interval: interval_upto_subprod,
            });
          }
          fn process_group(
            cur_anon_sym_index: &mut usize,
            cur_elements: &mut Vec<EpsilonGraphVertex>,
            all_intervals_from_this_case: &mut Vec<EpsilonGraphCase>,
            groups: &IndexMap<gc::GroupRef, gc::ProdRef>,
            target_group_ref: gc::GroupRef,
          ) {
            let target_prod_ref: gc::ProdRef = groups.get(&target_group_ref).unwrap().clone();
            process_prod_ref(
              cur_anon_sym_index,
              cur_elements,
              all_intervals_from_this_case,
              target_prod_ref,
            );
          }

          match el {
            /* Continue the current interval of nonterminals. */
            gc::CaseEl::Tok(_) => {
              process_token(cur_pos, &mut cur_elements);
            },
            /* The current case invokes a subproduction, so is split into two intervals
             * here, using anonymous symbols to keep track of where in
             * this case we jumped off of and where we can jump back
             * onto to satisfy this case of this production. */
            gc::CaseEl::Prod(target_subprod_ref) => {
              process_prod_ref(
                &mut cur_anon_sym_index,
                &mut cur_elements,
                &mut all_intervals_from_this_case,
                *target_subprod_ref,
              );
            },
            gc::CaseEl::Group(target_group_ref) => {
              process_group(
                &mut cur_anon_sym_index,
                &mut cur_elements,
                &mut all_intervals_from_this_case,
                &groups,
                *target_group_ref,
              );
            },
          }
        }
        /* Construct the interval of all remaining nonterminals to the end of the
         * production. */
        let mut final_interval: Vec<EpsilonGraphVertex> =
          Vec::with_capacity(cur_elements.len() + 2);
        final_interval.extend(cur_elements.into_iter());
        final_interval.push(neg_case_anon);
        final_interval.push(EpsilonGraphVertex::End(*cur_prod_ref));

        /* Register the interval of all remaining nonterminals in the results list. */
        all_intervals_from_this_case.push(EpsilonGraphCase {
          interval: final_interval,
        });
        /* Return all the intervals from this case. */
        really_all_intervals.extend(all_intervals_from_this_case);
      }
    }
    (
      EpsilonIntervalGraph {
        all_cases: really_all_intervals,
      },
      tokens,
    )
  }

  pub fn new(grammar: gb::TokenGrammar<Tok>) -> Self {
    let (terminals_interval_graph, token_states_mapping) =
      Self::produce_terminals_interval_graph(grammar);
    let CyclicGraphDecomposition { trie_graph } = terminals_interval_graph.connect_all_vertices();
    let graph_transitions = IndexedTrie::from_epsilon_graph(trie_graph);

    PreprocessedGrammar {
      token_states_mapping,
      graph_transitions,
    }
  }
}

impl<Tok> graphvizier::Graphable for PreprocessedGrammar<Tok> {
  fn build_graph(self) -> graphvizier::generator::GraphBuilder {
    todo!("fix!")
  }

  /* fn build_graph(self) -> graphvizier::generator::GraphBuilder { */
  /*   let mut gb = graphvizier::generator::GraphBuilder::new(); */

  /*   let Self { */
  /*     cyclic_graph_decomposition: */
  /*       CyclicGraphDecomposition { */
  /*         trie_graph, */
  /*         pairwise_state_transitions, */
  /*         .. */
  /*       }, */
  /*     .. */
  /*   } = self; */

  /*   let mut epsilon_graph_vertices: IndexMap<EpsilonGraphVertex, gv::Vertex> = IndexMap::new(); */
  /*   let mut cyclic_edges: Vec<gv::Edge> = Vec::new(); */
  /*   let mut stack_trie_vertices: IndexMap<TrieNodeRef, (StackTrieNode, gv::Vertex)> = */
  /*     IndexMap::new(); */

  /*   /\* (A) Process the stack trie node forest to get cyclic transitions between */
  /*    * states. *\/ */
  /*   { */
  /*     let EpsilonNodeStateSubgraph { */
  /*       vertex_mapping, */
  /*       trie_node_universe, */
  /*     } = trie_graph; */

  /*     /\* (1) Generate a graphviz vertex for each stack trie node in the universe. *\/ */
  /*     for (this_ref, node) in trie_node_universe.into_iter().enumerate() { */
  /*       let this_vertex = gv::Vertex { */
  /*         id: gv::Id::new(format!("stack_trie_node_{}", this_ref)), */
  /*         label: Some(gv::Label(format!("{}", node.step))), */
  /*         ..Default::default() */
  /*       }; */
  /*       /\* NB: Nodes are referenced by their index in the trie_node_universe vector. *\/ */
  /*       let this_ref = TrieNodeRef(this_ref); */

  /*       let entry = (node, this_vertex); */
  /*       assert!(stack_trie_vertices.insert(this_ref, entry).is_none()); */
  /*     } */

  /*     /\* (2) Generate a gv::Vertex for each EpsilonGraphVertex. *\/ */
  /*     let mut trie_node_vertex_mapping: IndexMap<gv::Id, gv::Id> = IndexMap::new(); */
  /*     for (this_id, (vtx, node_ref)) in vertex_mapping.into_iter().enumerate() { */
  /*       let this_id = gv::Id::new(format!("cyclic_epsilon_graph_vertex_{}", this_id)); */
  /*       let this_vertex = gv::Vertex { */
  /*         id: this_id.clone(), */
  /*         label: Some(gv::Label(format!("{}", vtx))), */
  /*         ..Default::default() */
  /*       }; */

  /*       assert!(epsilon_graph_vertices.insert(vtx, this_vertex).is_none()); */

  /*       /\* (2.1) Generate an edge to the trie node this vertex shadows! *\/ */
  /*       let (_, vtx) = stack_trie_vertices.get(&node_ref).unwrap(); */
  /*       assert!(trie_node_vertex_mapping */
  /*         .insert(vtx.id.clone(), this_id.clone()) */
  /*         .is_none()); */
  /*     } */
  /*     /\* (1.1) Add edges between all stack trie nodes! *\/ */
  /*     for (StackTrieNode { next_nodes, .. }, gv::Vertex { id: this_id, .. }) in */
  /*       stack_trie_vertices.values() */
  /*     { */
  /*       /\* (1.1.1) Add "next" edges. *\/ */
  /*       for next_node in next_nodes.iter() { */
  /*         let edge = match next_node { */
  /*           StackTrieNextEntry::Incomplete(next_ref) => { */
  /*             let (_, vtx) = stack_trie_vertices.get(next_ref).unwrap(); */
  /*             gv::Edge { */
  /*               source: trie_node_vertex_mapping.get(this_id).unwrap().clone(), */
  /*               target: trie_node_vertex_mapping.get(&vtx.id).unwrap().clone(), */
  /*               color: Some(gv::Color("red".to_string())), */
  /*               ..Default::default() */
  /*             } */
  /*           }, */
  /*           _ => unreachable!(), */
  /*         }; */
  /*         cyclic_edges.push(edge); */
  /*       } */
  /*       /\* (1.1.1) Add "prev" edges. *\/ */
  /*       /\* NB: these are always a mirror of the "next" edges, so we don't */
  /*        * need them for visualization. *\/ */
  /*     } */
  /*   } */

  /*   /\* (B) Extract all finite (non-looping) paths between states. *\/ */
  /*   let mut non_cyclic_edges: Vec<gv::Edge> = Vec::new(); */
  /*   { */
  /*     let mut vertex_id_counter_phase_2: usize = 0; */
  /*     for transition in pairwise_state_transitions.into_iter() { */
  /*       let CompletedStatePairWithVertices { */
  /*         interval: EpsilonGraphCase { interval }, */
  /*         .. */
  /*       } = transition; */

  /*       let mut prev_id: Option<gv::Id> = None; */

  /*       for vtx in interval.into_iter() { */
  /*         let next_id = epsilon_graph_vertices */
  /*           .entry(vtx) */
  /*           .or_insert_with(|| { */
  /*             let id = gv::Id::new(format!( */
  /*               "epsilon_graph_vertex_phase_2_{}", */
  /*               vertex_id_counter_phase_2 */
  /*             )); */
  /*             vertex_id_counter_phase_2 += 1; */
  /*             gv::Vertex { */
  /*               id, */
  /*               label: Some(gv::Label(format!("{}", vtx))), */
  /*               ..Default::default() */
  /*             } */
  /*           }) */
  /*           .id */
  /*           .clone(); */
  /*         if let Some(prev_id) = prev_id { */
  /*           let edge = gv::Edge { */
  /*             source: prev_id, */
  /*             target: next_id.clone(), */
  /*             color: Some(gv::Color("aqua".to_string())), */
  /*             ..Default::default() */
  /*           }; */
  /*           non_cyclic_edges.push(edge); */
  /*         } */
  /*         prev_id = Some(next_id); */
  /*       } */
  /*     } */
  /*   }; */

  /*   /\* Plot EpsilonGraphVertex instances. *\/ */
  /*   let mut border_vertices: Vec<gv::Vertex> = Vec::new(); */
  /*   let mut state_vertices: Vec<gv::Vertex> = Vec::new(); */
  /*   for (eg_vtx, mut gv_vtx) in epsilon_graph_vertices.into_iter() { */
  /*     match eg_vtx { */
  /*       EpsilonGraphVertex::Start(_) => { */
  /*         gv_vtx.color = Some(gv::Color("brown".to_string())); */
  /*         gv_vtx.fontcolor = Some(gv::Color("brown".to_string())); */
  /*         border_vertices.push(gv_vtx); */
  /*       }, */
  /*       EpsilonGraphVertex::End(_) => { */
  /*         gv_vtx.color = Some(gv::Color("darkgoldenrod".to_string())); */
  /*         gv_vtx.fontcolor = Some(gv::Color("darkgoldenrod".to_string())); */
  /*         border_vertices.push(gv_vtx); */
  /*       }, */
  /*       EpsilonGraphVertex::State(_) => { */
  /*         state_vertices.push(gv_vtx); */
  /*       }, */
  /*       EpsilonGraphVertex::Anon(_) => { */
  /*         gv_vtx.color = Some(gv::Color("blue".to_string())); */
  /*         gv_vtx.fontcolor = Some(gv::Color("blue".to_string())); */
  /*         gb.accept_entity(gv::Entity::Vertex(gv_vtx)); */
  /*       }, */
  /*     } */
  /*   } */
  /*   let border_vertices = gv::Subgraph { */
  /*     id: gv::Id::new("border_vertices"), */
  /*     label: Some(gv::Label("Borders".to_string())), */
  /*     color: Some(gv::Color("purple".to_string())), */
  /*     fontcolor: Some(gv::Color("purple".to_string())), */
  /*     entities: border_vertices */
  /*       .into_iter() */
  /*       .map(gv::Entity::Vertex) */
  /*       .collect(), */
  /*     ..Default::default() */
  /*   }; */
  /*   gb.accept_entity(gv::Entity::Subgraph(border_vertices)); */
  /*   let state_vertices = gv::Subgraph { */
  /*     id: gv::Id::new("state_vertices"), */
  /*     label: Some(gv::Label("States".to_string())), */
  /*     color: Some(gv::Color("green4".to_string())), */
  /*     fontcolor: Some(gv::Color("green4".to_string())), */
  /*     node_defaults: Some(gv::NodeDefaults { */
  /*       color: Some(gv::Color("green4".to_string())), */
  /*       fontcolor: Some(gv::Color("green4".to_string())), */
  /*     }), */
  /*     entities: state_vertices.into_iter().map(gv::Entity::Vertex).collect(), */
  /*     ..Default::default() */
  /*   }; */
  /*   gb.accept_entity(gv::Entity::Subgraph(state_vertices)); */

  /*   let mut seen: IndexSet<(gv::Id, gv::Id)> = IndexSet::new(); */
  /*   for edge in non_cyclic_edges.into_iter() { */
  /*     let key = (edge.source.clone(), edge.target.clone()); */
  /*     if !seen.contains(&key) { */
  /*       seen.insert(key); */
  /*       gb.accept_entity(gv::Entity::Edge(edge)); */
  /*     } */
  /*   } */
  /*   for edge in cyclic_edges.into_iter() { */
  /*     /\* If any cyclic edges overlap non-cyclic, don't print them! *\/ */
  /*     let key = (edge.source.clone(), edge.target.clone()); */
  /*     if !seen.contains(&key) { */
  /*       seen.insert(key); */
  /*       gb.accept_entity(gv::Entity::Edge(edge)); */
  /*     } */
  /*   } */

  /*   gb */
  /* } */
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    state,
    text_backend::{basic_productions, new_token_position, non_cyclic_productions},
  };

  use graphvizier::entities as gv;

  #[test]
  fn token_grammar_state_indexing() {
    let prods = non_cyclic_productions();
    let state::preprocessing::Detokenized::<char, _> {
      token_grammar: grammar,
      /* TODO: test prod_ref_map? */
      ..
    } = state::preprocessing::Init(prods).try_index().unwrap();
    assert_eq!(
      grammar
        .tokens
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

  #[ignore]
  #[test]
  fn terminals_interval_graph() {
    let noncyclic_prods = non_cyclic_productions();
    let state::preprocessing::Detokenized::<char, _> {
      token_grammar: noncyclic_grammar,
      ..
    } = state::preprocessing::Init(noncyclic_prods)
      .try_index()
      .unwrap();

    let (noncyclic_interval_graph, _) =
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

    let a_0 = EpsilonGraphCase {
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
    };
    let b_start_to_a_start_0 = EpsilonGraphCase {
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
    };
    let a_end_to_b_end_0 = EpsilonGraphCase {
      interval: [a_end, b_0_anon_0_end, b_prod_anon_end, b_end]
        .as_ref()
        .to_vec(),
    };
    let b_start_to_a_start_1 = EpsilonGraphCase {
      interval: [b_start, b_1_anon_0_start, b_1_anon_0_start_2, a_start]
        .as_ref()
        .to_vec(),
    };
    let a_end_to_b_end_1 = EpsilonGraphCase {
      interval: [a_end, b_1_anon_0_end_2, b_1_1, b_1_anon_0_end, b_end]
        .as_ref()
        .to_vec(),
    };

    assert_eq!(
      noncyclic_interval_graph,
      EpsilonIntervalGraph {
        all_cases: [
          a_0.clone(),
          b_start_to_a_start_0.clone(),
          a_end_to_b_end_0.clone(),
          b_start_to_a_start_1.clone(),
          a_end_to_b_end_1.clone(),
        ]
        .as_ref()
        .to_vec(),
      }
    );

    /* Now check for indices. */
    let intervals_by_start_and_end =
      EpsilonIntervalGraph::find_start_end_indices(noncyclic_interval_graph.all_cases.clone());
    assert_eq!(
      intervals_by_start_and_end,
      [
        (
          a_prod,
          StartEndEpsilonIntervals {
            start_epsilons: [a_0.clone()].as_ref().to_vec(),
            end_epsilons: [a_end_to_b_end_0.clone(), a_end_to_b_end_1.clone()]
              .as_ref()
              .to_vec(),
          },
        ),
        (
          b_prod,
          StartEndEpsilonIntervals {
            start_epsilons: [b_start_to_a_start_0.clone(), b_start_to_a_start_1.clone()]
              .as_ref()
              .to_vec(),
            end_epsilons: [].as_ref().to_vec(),
          },
        ),
      ]
      .as_ref()
      .to_vec()
      .iter()
      .cloned()
      .collect::<IndexMap<_, _>>()
    );

    /* Now check that the transition graph is as we expect. */
    let CyclicGraphDecomposition { trie_graph } = noncyclic_interval_graph.connect_all_vertices();
    let _ = trie_graph; /* TODO: uncomment this test! */
    /* There are no stack cycles in the noncyclic graph. */
    /* assert_eq!( */
    /*      trie_graph, */
    /*      EpsilonNodeStateSubgraph { */
    /*        vertex_mapping: [ */
    /*          (EpsilonGraphVertex::Start(gc::ProdRef(0)), TrieNodeRef(0)), */
    /*          (EpsilonGraphVertex::Anon(Positive(AnonSym(0))), TrieNodeRef(1)), */
    /*          (EpsilonGraphVertex::State(gc::TokenPosition::new(0, 0, 0)), TrieNodeRef(2)), */
    /*          (EpsilonGraphVertex::State(gv::TokenPosition::new(0, 0, 1)), TrieNodeRef(3)), */
    /*          (EpsilonGraphVertex::Anon(Negative(AnonSym(0))), TrieNodeRef(4)), */
    /*          (EpsilonGraphVertex::End(gc::ProdRef(0)), TrieNodeRef(5)), */
    /*          (EpsilonGraphVertex::Anon(Negative(AnonSym(2))), TrieNodeRef(6)), */
    /*          (EpsilonGraphVertex::Anon(Negative(AnonSym(1))), TrieNodeRef(7)), */
    /*          (EpsilonGraphVertex::End(gc::ProdRef(1)), TrieNodeRef(8)), */
    /*          (EpsilonGraphVertex::Anon(Negative(AnonSym(4))), TrieNodeRef(9)), */
    /*          (EpsilonGraphVertex::State(gc::TokenPosition::new(1, 1, 1)), TrieNodeRef(10)), */
    /*          (EpsilonGraphVertex::Anon(Negative(AnonSym(3))), TrieNodeRef(11)), */
    /*          (EpsilonGraphVertex::Start(gc::ProdRef(1)), TrieNodeRef(12)), */
    /*          (EpsilonGraphVertex::Anon(Positive(AnonSym(1))), TrieNodeRef(13)), */
    /*          (EpsilonGraphVertex::State(gc::TokenPosition::new(1, 0, 0)), TrieNodeRef(14)), */
    /*          (EpsilonGraphVertex::State(gc::TokenPosition::new(1, 0, 1)), TrieNodeRef(15)), */
    /*          (EpsilonGraphVertex::Anon(Positive(AnonSym(2))), TrieNodeRef(16)), */
    /*          (EpsilonGraphVertex::Anon(Positive(AnonSym(3))), TrieNodeRef(17)), */
    /*          (EpsilonGraphVertex::Anon(Positive(AnonSym(4))), TrieNodeRef(18)), */
    /*        ].iter().cloned().collect(), */
    /*        trie_node_universe: [ */
    /*          StackTrieNode { */
    /*            step: Some(Named(Positive(StackSym(ProdRef(0))))), */
    /*            next_nodes: [Incomplete(TrieNodeRef(1))].iter().cloned().collect(), */
    /*            prev_nodes: [Incomplete(TrieNodeRef(16)), Incomplete(TrieNodeRef(18))].iter().cloned().collect(), */
    /*          }, */
    /*          StackTrieNode { */
    /*            step: Some(Anon(Positive(AnonSym(0)))), */
    /*            next_nodes: {Completed(Within(TokenPosition { prod: ProdRef(0), case: CaseRef(0), el: CaseElRef(0) }))}, */
    /*            prev_nodes: {Completed(Start)} }, StackTrieNode { step: None, next_nodes: {Completed(Within(TokenPosition { prod: ProdRef(0), case: CaseRef(0), el: CaseElRef(1) }))}, prev_nodes: {Incomplete(TrieNodeRef(1))} }, StackTrieNode { step: None, next_nodes: {Incomplete(TrieNodeRef(4))}, prev_nodes: {Completed(Within(TokenPosition { prod: ProdRef(0), case: CaseRef(0), el: CaseElRef(0) }))} }, StackTrieNode { step: Some(Anon(Negative(AnonSym(0)))), next_nodes: {Completed(End)}, prev_nodes: {Completed(Within(TokenPosition { prod: ProdRef(0), case: CaseRef(0), el: CaseElRef(1) }))} }, StackTrieNode { step: Some(Named(Negative(StackSym(ProdRef(0))))), next_nodes: {Incomplete(TrieNodeRef(6)), Incomplete(TrieNodeRef(9))}, prev_nodes: {Incomplete(TrieNodeRef(4))} }, StackTrieNode { step: Some(Anon(Negative(AnonSym(2)))), next_nodes: {Incomplete(TrieNodeRef(7))}, prev_nodes: {Completed(End)} }, StackTrieNode { step: Some(Anon(Negative(AnonSym(1)))), next_nodes: {Completed(End)}, prev_nodes: {Incomplete(TrieNodeRef(6))} }, StackTrieNode { step: Some(Named(Negative(StackSym(ProdRef(1))))), next_nodes: {}, prev_nodes: {Incomplete(TrieNodeRef(7)), Incomplete(TrieNodeRef(11))} }, StackTrieNode { step: Some(Anon(Negative(AnonSym(4)))), next_nodes: {Completed(Within(TokenPosition { prod: ProdRef(1), case: CaseRef(1), el: CaseElRef(1) }))}, prev_nodes: {Completed(End)} }, StackTrieNode { step: None, next_nodes: {Incomplete(TrieNodeRef(11))}, prev_nodes: {Incomplete(TrieNodeRef(9))} }, StackTrieNode { step: Some(Anon(Negative(AnonSym(3)))), next_nodes: {Completed(End)}, prev_nodes: {Completed(Within(TokenPosition { prod: ProdRef(1), case: CaseRef(1), el: CaseElRef(1) }))} }, StackTrieNode { step: Some(Named(Positive(StackSym(ProdRef(1))))), next_nodes: {Incomplete(TrieNodeRef(13)), Incomplete(TrieNodeRef(17))}, prev_nodes: {} }, StackTrieNode { step: Some(Anon(Positive(AnonSym(1)))), next_nodes: {Completed(Within(TokenPosition { prod: ProdRef(1), case: CaseRef(0), el: CaseElRef(0) }))}, prev_nodes: {Completed(Start)} }, StackTrieNode { step: None, next_nodes: {Completed(Within(TokenPosition { prod: ProdRef(1), case: CaseRef(0), el: CaseElRef(1) }))}, prev_nodes: {Incomplete(TrieNodeRef(13))} }, StackTrieNode { step: None, next_nodes: {Incomplete(TrieNodeRef(16))}, prev_nodes: {Completed(Within(TokenPosition { prod: ProdRef(1), case: CaseRef(0), el: CaseElRef(0) }))} }, StackTrieNode { step: Some(Anon(Positive(AnonSym(2)))), next_nodes: {Completed(Start)}, prev_nodes: {Completed(Within(TokenPosition { prod: ProdRef(1), case: CaseRef(0), el: CaseElRef(1) }))} }, StackTrieNode { step: Some(Anon(Positive(AnonSym(3)))), next_nodes: {Incomplete(TrieNodeRef(18))}, prev_nodes: {Completed(Start)} }, StackTrieNode { step: Some(Anon(Positive(AnonSym(4)))), next_nodes: {Completed(Start)}, prev_nodes: {Incomplete(TrieNodeRef(17))} }].iter().cloned().collect(), */
    /* } */
    /*    ); */

    /* Now do the same, but for `basic_productions()`. */
    /* TODO: test `.find_start_end_indices()` and `.connect_all_vertices()` here
     * too! */
    let prods = basic_productions();
    let state::preprocessing::Detokenized::<char, _> {
      token_grammar: grammar,
      ..
    } = state::preprocessing::Init(prods).try_index().unwrap();
    let (interval_graph, _) = PreprocessedGrammar::produce_terminals_interval_graph(grammar);
    assert_eq!(
      &interval_graph,
      &EpsilonIntervalGraph {
        all_cases: [
          EpsilonGraphCase {
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
          },
          EpsilonGraphCase {
            interval: [
              EpsilonGraphVertex::Start(gc::ProdRef(0)),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(1))),
              EpsilonGraphVertex::State(new_token_position(0, 1, 0)),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(2))),
              EpsilonGraphVertex::Start(gc::ProdRef(0)),
            ]
            .as_ref()
            .to_vec(),
          },
          EpsilonGraphCase {
            interval: [
              EpsilonGraphVertex::End(gc::ProdRef(0)),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(2))),
              EpsilonGraphVertex::State(new_token_position(0, 1, 2)),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(1))),
              EpsilonGraphVertex::End(gc::ProdRef(0)),
            ]
            .as_ref()
            .to_vec(),
          },
          EpsilonGraphCase {
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
          },
          EpsilonGraphCase {
            interval: [
              EpsilonGraphVertex::End(gc::ProdRef(1)),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(4))),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(3))),
              EpsilonGraphVertex::End(gc::ProdRef(0)),
            ]
            .as_ref()
            .to_vec(),
          },
          EpsilonGraphCase {
            interval: [
              EpsilonGraphVertex::Start(gc::ProdRef(1)),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(5))),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(6))),
              EpsilonGraphVertex::Start(gc::ProdRef(0)),
            ]
            .as_ref()
            .to_vec(),
          },
          EpsilonGraphCase {
            interval: [
              EpsilonGraphVertex::End(gc::ProdRef(0)),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(6))),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(5))),
              EpsilonGraphVertex::End(gc::ProdRef(1)),
            ]
            .as_ref()
            .to_vec(),
          },
          EpsilonGraphCase {
            interval: [
              EpsilonGraphVertex::Start(gc::ProdRef(1)),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(7))),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(8))),
              EpsilonGraphVertex::Start(gc::ProdRef(1)),
            ]
            .as_ref()
            .to_vec(),
          },
          EpsilonGraphCase {
            interval: [
              EpsilonGraphVertex::End(gc::ProdRef(1)),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(8))),
              EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(7))),
              EpsilonGraphVertex::End(gc::ProdRef(1)),
            ]
            .as_ref()
            .to_vec(),
          },
          EpsilonGraphCase {
            interval: [
              EpsilonGraphVertex::Start(gc::ProdRef(1)),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(9))),
              EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(10))),
              EpsilonGraphVertex::Start(gc::ProdRef(0)),
            ]
            .as_ref()
            .to_vec(),
          },
          EpsilonGraphCase {
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
          }
        ]
        .as_ref()
        .to_vec(),
      }
    );
  }

  /* /\* TODO: consider creating/using a generic tree diffing algorithm in case */
  /*  * that speeds up debugging (this might conflict with the benefits of using */
  /*  * totally ordered IndexMaps though, namely determinism, as well as knowing */
  /*  * exactly which order your subtrees are created in)! *\/ */
  /* #[ignore] */
  /* #[test] */
  /* fn noncyclic_transition_graph() { */
  /*   let prods = non_cyclic_productions(); */
  /*   let detokenized = state::preprocessing::Init(prods).try_index().unwrap(); */
  /*   let state::preprocessing::Indexed(preprocessed_grammar) = detokenized.index(); */

  /*   let first_a = new_token_position(0, 0, 0); */
  /*   let first_b = new_token_position(0, 0, 1); */
  /*   let second_a = new_token_position(1, 0, 0); */
  /*   let second_b = new_token_position(1, 0, 1); */
  /*   let third_a = new_token_position(1, 1, 1); */
  /*   let a_prod = gc::ProdRef(0); */
  /*   let b_prod = gc::ProdRef(1); */
  /*   assert_eq!( */
  /*     preprocessed_grammar.token_states_mapping.into_index_map(), */
  /*     [ */
  /*       ( */
  /*         gc::TokRef(0), */
  /*         [first_a, second_a, third_a].as_ref().to_vec() */
  /*       ), */
  /*       (gc::TokRef(1), [first_b, second_b].as_ref().to_vec()), */
  /*     ] */
  /*     .iter() */
  /*     .cloned() */
  /*     .collect::<IndexMap<_, _>>(), */
  /*   ); */

  /*   let other_cyclic_graph_decomposition = CyclicGraphDecomposition { */
  /*     trie_graph: EpsilonNodeStateSubgraph { */
  /*       vertex_mapping: IndexMap::<_, _>::new(), */
  /*       trie_node_universe: Vec::new(), */
  /*     }, */
  /*   }; */

  /*   assert_eq!( */
  /*     preprocessed_grammar.graph_transitions, */
  /*     other_cyclic_graph_decomposition, */
  /*   ); */
  /* } */

  /* #[ignore] */
  /* #[test] */
  /* fn cyclic_transition_graph() { */
  /*   let prods = basic_productions(); */
  /*   let detokenized = state::preprocessing::Init(prods).try_index().unwrap(); */
  /*   let state::preprocessing::Indexed(preprocessed_grammar) = detokenized.index(); */

  /*   let first_a = new_token_position(0, 0, 0); */
  /*   let second_a = new_token_position(0, 1, 0); */

  /*   let first_b = new_token_position(0, 0, 1); */
  /*   let second_b = new_token_position(0, 2, 0); */
  /*   let third_b = new_token_position(1, 2, 1); */

  /*   let first_c = new_token_position(0, 0, 2); */
  /*   let second_c = new_token_position(0, 1, 2); */
  /*   let third_c = new_token_position(0, 2, 1); */
  /*   let fourth_c = new_token_position(1, 2, 2); */

  /*   let a_prod = gc::ProdRef(0); */
  /*   let b_prod = gc::ProdRef(1); */
  /*   let _c_prod = gc::ProdRef(2); /\* unused *\/ */

  /*   assert_eq!( */
  /*     preprocessed_grammar.token_states_mapping.into_index_map(), */
  /*     [ */
  /*       (gc::TokRef(0), [first_a, second_a].as_ref().to_vec()), */
  /*       ( */
  /*         gc::TokRef(1), */
  /*         [first_b, second_b, third_b].as_ref().to_vec() */
  /*       ), */
  /*       ( */
  /*         gc::TokRef(2), */
  /*         [first_c, second_c, third_c, fourth_c].as_ref().to_vec() */
  /*       ), */
  /*     ] */
  /*     .iter() */
  /*     .cloned() */
  /*     .collect::<IndexMap<_, _>>() */
  /*   ); */

  /*   assert_eq!( */
  /*     preprocessed_grammar */
  /*       .cyclic_graph_decomposition */
  /*       .trie_graph */
  /*       .vertex_mapping */
  /*       .clone(), */
  /*     [ */
  /*       /\* 0 *\/ */
  /*       (EpsilonGraphVertex::Start(b_prod), TrieNodeRef(0)), */
  /*       /\* 1 *\/ */
  /*       ( */
  /*         EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(7))), */
  /*         TrieNodeRef(1) */
  /*       ), */
  /*       /\* 2 *\/ */
  /*       ( */
  /*         EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(8))), */
  /*         TrieNodeRef(2) */
  /*       ), */
  /*       /\* 3 *\/ */
  /*       (EpsilonGraphVertex::End(b_prod), TrieNodeRef(3)), */
  /*       /\* 4 *\/ */
  /*       ( */
  /*         EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(8))), */
  /*         TrieNodeRef(4) */
  /*       ), */
  /*       /\* 5 *\/ */
  /*       ( */
  /*         EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(7))), */
  /*         TrieNodeRef(5) */
  /*       ), */
  /*       /\* 6 *\/ */
  /*       ( */
  /*         EpsilonGraphVertex::State(new_token_position(a_prod.0, 1, 0)), */
  /*         TrieNodeRef(6) */
  /*       ), */
  /*       /\* 7 *\/ */
  /*       ( */
  /*         EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(2))), */
  /*         TrieNodeRef(7) */
  /*       ), */
  /*       /\* 8 *\/ */
  /*       (EpsilonGraphVertex::Start(a_prod), TrieNodeRef(8)), */
  /*       /\* 9 *\/ */
  /*       ( */
  /*         EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(1))), */
  /*         TrieNodeRef(9) */
  /*       ), */
  /*       /\* 10 *\/ */
  /*       ( */
  /*         EpsilonGraphVertex::State(new_token_position(a_prod.0, 1, 2)), */
  /*         TrieNodeRef(10) */
  /*       ), */
  /*       /\* 11 *\/ */
  /*       ( */
  /*         EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(1))), */
  /*         TrieNodeRef(11) */
  /*       ), */
  /*       /\* 12 *\/ */
  /*       (EpsilonGraphVertex::End(a_prod), TrieNodeRef(12)), */
  /*       /\* 13 *\/ */
  /*       ( */
  /*         EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(2))), */
  /*         TrieNodeRef(13) */
  /*       ), */
  /*       /\* 14 *\/ */
  /*       ( */
  /*         EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(4))), */
  /*         TrieNodeRef(14) */
  /*       ), */
  /*       /\* 15 *\/ */
  /*       ( */
  /*         EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(3))), */
  /*         TrieNodeRef(15) */
  /*       ), */
  /*       /\* 16 *\/ */
  /*       ( */
  /*         EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(6))), */
  /*         TrieNodeRef(16) */
  /*       ), */
  /*       /\* 17 *\/ */
  /*       ( */
  /*         EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(5))), */
  /*         TrieNodeRef(17) */
  /*       ) */
  /*     ] */
  /*     .iter() */
  /*     .cloned() */
  /*     .collect::<IndexMap<_, _>>() */
  /*   ); */

  /*   let all_trie_nodes: &[StackTrieNode] = preprocessed_grammar */
  /*     .cyclic_graph_decomposition */
  /*     .trie_graph */
  /*     .trie_node_universe */
  /*     .as_ref(); */
  /*   assert_eq!( */
  /*     all_trie_nodes, */
  /*     [ */
  /*       /\* 0 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Named(StackStep::Positive(StackSym( */
  /*           b_prod */
  /*         )))), */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(1))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(2))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 1 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(7)))), */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(2))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(0))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 2 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(8)))), */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(0))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(1))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 3 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Named(StackStep::Negative(StackSym( */
  /*           b_prod */
  /*         )))), */
  /*         next_nodes: [ */
  /*           StackTrieNextEntry::Incomplete(TrieNodeRef(4)), */
  /*           StackTrieNextEntry::Incomplete(TrieNodeRef(14)) */
  /*         ] */
  /*         .iter() */
  /*         .cloned() */
  /*         .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [ */
  /*           StackTrieNextEntry::Incomplete(TrieNodeRef(5)), */
  /*           StackTrieNextEntry::Incomplete(TrieNodeRef(17)) */
  /*         ] */
  /*         .iter() */
  /*         .cloned() */
  /*         .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 4 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(8)))), */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(5))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(3))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 5 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(7)))), */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(3))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(4))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 6 *\/ */
  /*       StackTrieNode { */
  /*         step: None, */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(7))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(9))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 7 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(2)))), */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(8))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(6))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 8 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Named(StackStep::Positive(StackSym( */
  /*           a_prod */
  /*         )))), */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(9))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(7))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 9 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(1)))), */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(6))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(8))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 10 *\/ */
  /*       StackTrieNode { */
  /*         step: None, */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(11))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(13))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 11 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(1)))), */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(12))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(10))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 12 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Named(StackStep::Negative(StackSym( */
  /*           a_prod */
  /*         )))), */
  /*         next_nodes: [ */
  /*           StackTrieNextEntry::Incomplete(TrieNodeRef(13)), */
  /*           StackTrieNextEntry::Incomplete(TrieNodeRef(16)) */
  /*         ] */
  /*         .iter() */
  /*         .cloned() */
  /*         .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [ */
  /*           StackTrieNextEntry::Incomplete(TrieNodeRef(11)), */
  /*           StackTrieNextEntry::Incomplete(TrieNodeRef(15)) */
  /*         ] */
  /*         .iter() */
  /*         .cloned() */
  /*         .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 13 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(2)))), */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(10))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(12))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 14 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(4)))), */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(15))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(3))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 15 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(3)))), */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(12))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(14))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 16 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(6)))), */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(17))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(12))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       }, */
  /*       /\* 17 *\/ */
  /*       StackTrieNode { */
  /*         step: Some(NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(5)))), */
  /*         next_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(3))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>(), */
  /*         prev_nodes: [StackTrieNextEntry::Incomplete(TrieNodeRef(16))] */
  /*           .iter() */
  /*           .cloned() */
  /*           .collect::<IndexSet<_>>() */
  /*       } */
  /*     ] */
  /*     .as_ref() */
  /*   ); */
  /* } */

  #[ignore]
  #[test]
  fn non_cyclic_preprocessed_graphviz() {
    use graphvizier::Graphable;

    let prods = non_cyclic_productions();
    let detokenized = state::preprocessing::Init(prods).try_index().unwrap();
    let state::preprocessing::Indexed {
      preprocessed_grammar,
      ..
    } = detokenized.index();

    let gb = preprocessed_grammar.build_graph();
    let graphvizier::generator::DotOutput(output) = gb.build(gv::Id::new("test_graph"));

    assert_eq!(output, "digraph test_graph {\n  compound = true;\n\n  epsilon_graph_vertex_phase_2_1[label=\"+!0!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_4[label=\"+!1!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_9[label=\"-!3!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_11[label=\"-!0!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_13[label=\"+!3!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_14[label=\"+!4!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_15[label=\"-!4!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_16[label=\"+!2!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_17[label=\"-!2!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_18[label=\"-!1!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  subgraph border_vertices {\n    label = \"Borders\";\n    cluster = true;\n    rank = same;\n\n    color = \"purple\";\n    fontcolor = \"purple\";\n\n    epsilon_graph_vertex_phase_2_0[label=\"Start(0)\", color=\"brown\", fontcolor=\"brown\", ];\n    epsilon_graph_vertex_phase_2_3[label=\"Start(1)\", color=\"brown\", fontcolor=\"brown\", ];\n    epsilon_graph_vertex_phase_2_10[label=\"End(1)\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n    epsilon_graph_vertex_phase_2_12[label=\"End(0)\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n  }\n\n  subgraph state_vertices {\n    label = \"States\";\n    cluster = true;\n    rank = same;\n\n    color = \"green4\";\n    fontcolor = \"green4\";\n    node [color=\"green4\", fontcolor=\"green4\", ];\n\n    epsilon_graph_vertex_phase_2_2[label=\"0/0/0\", ];\n    epsilon_graph_vertex_phase_2_5[label=\"1/0/0\", ];\n    epsilon_graph_vertex_phase_2_6[label=\"0/0/1\", ];\n    epsilon_graph_vertex_phase_2_7[label=\"1/0/1\", ];\n    epsilon_graph_vertex_phase_2_8[label=\"1/1/1\", ];\n  }\n\n  epsilon_graph_vertex_phase_2_0 -> epsilon_graph_vertex_phase_2_1[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_1 -> epsilon_graph_vertex_phase_2_2[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_3 -> epsilon_graph_vertex_phase_2_4[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_4 -> epsilon_graph_vertex_phase_2_5[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_2 -> epsilon_graph_vertex_phase_2_6[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_5 -> epsilon_graph_vertex_phase_2_7[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_8 -> epsilon_graph_vertex_phase_2_9[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_9 -> epsilon_graph_vertex_phase_2_10[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_6 -> epsilon_graph_vertex_phase_2_11[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_11 -> epsilon_graph_vertex_phase_2_12[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_3 -> epsilon_graph_vertex_phase_2_13[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_13 -> epsilon_graph_vertex_phase_2_14[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_14 -> epsilon_graph_vertex_phase_2_0[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_12 -> epsilon_graph_vertex_phase_2_15[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_15 -> epsilon_graph_vertex_phase_2_8[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_7 -> epsilon_graph_vertex_phase_2_16[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_16 -> epsilon_graph_vertex_phase_2_0[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_12 -> epsilon_graph_vertex_phase_2_17[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_17 -> epsilon_graph_vertex_phase_2_18[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_18 -> epsilon_graph_vertex_phase_2_10[color=\"aqua\", ];\n}\n")
  }

  #[ignore]
  #[test]
  fn basic_preprocessed_graphviz() {
    use graphvizier::Graphable;

    let prods = basic_productions();
    let detokenized = state::preprocessing::Init(prods).try_index().unwrap();
    let state::preprocessing::Indexed {
      preprocessed_grammar,
      ..
    } = detokenized.index();

    let gb = preprocessed_grammar.build_graph();
    let graphvizier::generator::DotOutput(output) = gb.build(gv::Id::new("test_graph"));

    assert_eq!(output, "digraph test_graph {\n  compound = true;\n\n  cyclic_epsilon_graph_vertex_1[label=\"+!7!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  cyclic_epsilon_graph_vertex_2[label=\"+!8!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  cyclic_epsilon_graph_vertex_4[label=\"-!8!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  cyclic_epsilon_graph_vertex_5[label=\"-!7!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  cyclic_epsilon_graph_vertex_7[label=\"+!2!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  cyclic_epsilon_graph_vertex_9[label=\"+!1!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  cyclic_epsilon_graph_vertex_11[label=\"-!1!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  cyclic_epsilon_graph_vertex_13[label=\"-!2!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  cyclic_epsilon_graph_vertex_14[label=\"-!4!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  cyclic_epsilon_graph_vertex_15[label=\"-!3!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  cyclic_epsilon_graph_vertex_16[label=\"-!6!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  cyclic_epsilon_graph_vertex_17[label=\"-!5!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_0[label=\"+!0!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_2[label=\"+!3!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_9[label=\"+!5!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_10[label=\"+!6!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_11[label=\"+!9!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_12[label=\"+!10!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_13[label=\"-!9!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_14[label=\"-!0!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_15[label=\"-!10!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  epsilon_graph_vertex_phase_2_16[label=\"+!4!\", color=\"blue\", fontcolor=\"blue\", ];\n\n  subgraph border_vertices {\n    label = \"Borders\";\n    cluster = true;\n    rank = same;\n\n    color = \"purple\";\n    fontcolor = \"purple\";\n\n    cyclic_epsilon_graph_vertex_0[label=\"Start(1)\", color=\"brown\", fontcolor=\"brown\", ];\n    cyclic_epsilon_graph_vertex_3[label=\"End(1)\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n    cyclic_epsilon_graph_vertex_8[label=\"Start(0)\", color=\"brown\", fontcolor=\"brown\", ];\n    cyclic_epsilon_graph_vertex_12[label=\"End(0)\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n  }\n\n  subgraph state_vertices {\n    label = \"States\";\n    cluster = true;\n    rank = same;\n\n    color = \"green4\";\n    fontcolor = \"green4\";\n    node [color=\"green4\", fontcolor=\"green4\", ];\n\n    cyclic_epsilon_graph_vertex_6[label=\"0/1/0\", ];\n    cyclic_epsilon_graph_vertex_10[label=\"0/1/2\", ];\n    epsilon_graph_vertex_phase_2_1[label=\"0/0/0\", ];\n    epsilon_graph_vertex_phase_2_3[label=\"0/2/0\", ];\n    epsilon_graph_vertex_phase_2_4[label=\"0/0/1\", ];\n    epsilon_graph_vertex_phase_2_5[label=\"0/2/1\", ];\n    epsilon_graph_vertex_phase_2_6[label=\"1/2/1\", ];\n    epsilon_graph_vertex_phase_2_7[label=\"1/2/2\", ];\n    epsilon_graph_vertex_phase_2_8[label=\"0/0/2\", ];\n  }\n\n  cyclic_epsilon_graph_vertex_8 -> epsilon_graph_vertex_phase_2_0[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_0 -> epsilon_graph_vertex_phase_2_1[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_8 -> cyclic_epsilon_graph_vertex_9[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_9 -> cyclic_epsilon_graph_vertex_6[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_8 -> epsilon_graph_vertex_phase_2_2[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_2 -> epsilon_graph_vertex_phase_2_3[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_1 -> epsilon_graph_vertex_phase_2_4[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_3 -> epsilon_graph_vertex_phase_2_5[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_6 -> epsilon_graph_vertex_phase_2_7[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_4 -> epsilon_graph_vertex_phase_2_8[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_10 -> cyclic_epsilon_graph_vertex_11[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_11 -> cyclic_epsilon_graph_vertex_12[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_0 -> epsilon_graph_vertex_phase_2_9[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_9 -> epsilon_graph_vertex_phase_2_10[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_10 -> cyclic_epsilon_graph_vertex_8[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_0 -> epsilon_graph_vertex_phase_2_11[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_11 -> epsilon_graph_vertex_phase_2_12[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_12 -> cyclic_epsilon_graph_vertex_8[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_7 -> epsilon_graph_vertex_phase_2_13[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_13 -> cyclic_epsilon_graph_vertex_3[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_8 -> epsilon_graph_vertex_phase_2_14[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_14 -> cyclic_epsilon_graph_vertex_12[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_6 -> cyclic_epsilon_graph_vertex_7[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_7 -> cyclic_epsilon_graph_vertex_8[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_12 -> epsilon_graph_vertex_phase_2_15[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_15 -> epsilon_graph_vertex_phase_2_6[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_12 -> cyclic_epsilon_graph_vertex_16[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_16 -> cyclic_epsilon_graph_vertex_17[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_17 -> cyclic_epsilon_graph_vertex_3[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_12 -> cyclic_epsilon_graph_vertex_13[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_13 -> cyclic_epsilon_graph_vertex_10[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_3 -> cyclic_epsilon_graph_vertex_14[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_14 -> cyclic_epsilon_graph_vertex_15[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_15 -> cyclic_epsilon_graph_vertex_12[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_5 -> epsilon_graph_vertex_phase_2_16[color=\"aqua\", ];\n\n  epsilon_graph_vertex_phase_2_16 -> cyclic_epsilon_graph_vertex_0[color=\"aqua\", ];\n\n  cyclic_epsilon_graph_vertex_0 -> cyclic_epsilon_graph_vertex_1[color=\"red\", ];\n\n  cyclic_epsilon_graph_vertex_1 -> cyclic_epsilon_graph_vertex_2[color=\"red\", ];\n\n  cyclic_epsilon_graph_vertex_2 -> cyclic_epsilon_graph_vertex_0[color=\"red\", ];\n\n  cyclic_epsilon_graph_vertex_3 -> cyclic_epsilon_graph_vertex_4[color=\"red\", ];\n\n  cyclic_epsilon_graph_vertex_4 -> cyclic_epsilon_graph_vertex_5[color=\"red\", ];\n\n  cyclic_epsilon_graph_vertex_5 -> cyclic_epsilon_graph_vertex_3[color=\"red\", ];\n}\n")
  }
}
