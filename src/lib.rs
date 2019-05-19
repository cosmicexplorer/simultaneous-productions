/*
    Description: Implement the Simultaneous Productions general parsing method.
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
#![feature(fn_traits)]
/* These clippy lint descriptions are purely non-functional and do not affect the functionality
 * or correctness of the code.
 * TODO: rustfmt breaks multiline comments when used one on top of another! (each with its own
 * pair of delimiters)
 * Note: run clippy with: rustup run nightly cargo-clippy! */
#![deny(warnings)]
// Enable all clippy lints except for many of the pedantic ones. It's a shame this needs to be
// copied and pasted across crates, but there doesn't appear to be a way to include inner attributes
// from a common source.
#![deny(
  clippy::all,
  clippy::default_trait_access,
  clippy::expl_impl_clone_on_copy,
  clippy::if_not_else,
  clippy::needless_continue,
  clippy::single_match_else,
  clippy::unseparated_literal_suffix,
  clippy::used_underscore_binding
)]
// It is often more clear to show that nothing is being moved.
#![allow(clippy::match_ref_pats)]
// Subjective style.
#![allow(
  clippy::derive_hash_xor_eq,
  clippy::len_without_is_empty,
  clippy::redundant_field_names,
  clippy::too_many_arguments
)]
// Default isn't as big a deal as people seem to think it is.
#![allow(clippy::new_without_default, clippy::new_ret_no_self)]
// Arc<Mutex> can be more clear than needing to grok Orderings:
#![allow(clippy::mutex_atomic)]

extern crate indexmap;

use indexmap::{IndexMap, IndexSet};

use std::{
  collections::{HashMap, HashSet, VecDeque},
  hash::{Hash, Hasher},
};

pub mod user_api {
  use super::*;

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct Literal<Tok: PartialEq+Eq+Hash+Copy+Clone>(pub Vec<Tok>);

  // NB: a From impl is usually intended to denote that allocation is /not/
  // performed, I think: see https://doc.rust-lang.org/std/convert/trait.From.html -- fn new() makes more sense for this use
  // case.
  impl Literal<char> {
    pub fn new(s: &str) -> Self { Literal(s.chars().collect()) }
  }

  // A reference to another production -- the string must match the assigned name
  // of a production in a set of simultaneous productions.
  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct ProductionReference(String);

  impl ProductionReference {
    pub fn new(s: &str) -> Self { ProductionReference(s.to_string()) }
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub enum CaseElement<Tok: PartialEq+Eq+Hash+Copy+Clone> {
    Lit(Literal<Tok>),
    Prod(ProductionReference),
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct Case<Tok: PartialEq+Eq+Hash+Copy+Clone>(pub Vec<CaseElement<Tok>>);

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct Production<Tok: PartialEq+Eq+Hash+Copy+Clone>(pub Vec<Case<Tok>>);

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct SimultaneousProductions<Tok: PartialEq+Eq+Hash+Copy+Clone>(
    pub IndexMap<ProductionReference, Production<Tok>>,
  );
}

///
/// (I think this is a "model" graph class of some sort, where the model is
/// this "simultaneous productions" parsing formulation)
///
/// Vec<ProductionImpl> = [
///   Production([
///     Case([CaseEl(Lit("???")), CaseEl(ProdRef(?)), ...]),
///     ...,
///   ]),
///   ...,
/// ]
pub mod lowering_to_indices {
  use super::{user_api::*, *};

  /// Graph Coordinates
  // NB: all these Refs have nice properties, which includes being storeable
  // without reference to any particular graph, being totally ordered, and
  // being able to be incremented.

  // A version of `ProductionReference` which uses a `usize` for speed. We adopt
  // the convention of abbreviated names for things used in algorithms.
  // Points to a particular Production within a Vec<ProductionImpl>.
  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub struct ProdRef(pub usize);

  // Points to a particular case within a Production.
  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub struct CaseRef(pub usize);

  // Points to an element of a particular Case.
  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub struct CaseElRef(pub usize);

  /* This refers to a specific token, implying that we must be pointing to a
   * particular index of a particular Literal. This corresponds to a "state"
   * in the simultaneous productions terminology. */
  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub struct TokenPosition {
    pub prod: ProdRef,
    pub case: CaseRef,
    pub case_el: CaseElRef,
  }

  #[cfg(test)]
  impl TokenPosition {
    pub fn new(prod_ind: usize, case_ind: usize, case_el_ind: usize) -> Self {
      TokenPosition {
        prod: ProdRef(prod_ind),
        case: CaseRef(case_ind),
        case_el: CaseElRef(case_el_ind),
      }
    }
  }

  /// Graph Representation

  // TODO: describe!
  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub struct TokRef(pub usize);

  #[derive(Debug, Copy, Clone, PartialEq, Eq)]
  pub enum CaseEl {
    Tok(TokRef),
    Prod(ProdRef),
  }

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct CaseImpl(pub Vec<CaseEl>);

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct ProductionImpl(pub Vec<CaseImpl>);

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct LoweredProductions(pub Vec<ProductionImpl>);

  /// Mapping to Tokens

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct TokenGrammar<Tok: PartialEq+Eq+Hash+Copy+Clone> {
    pub graph: LoweredProductions,
    pub alphabet: Vec<Tok>,
  }

  impl<Tok: PartialEq+Eq+Hash+Copy+Clone> TokenGrammar<Tok> {
    fn walk_productions_and_split_literal_strings(prods: &SimultaneousProductions<Tok>) -> Self {
      // Mapping from strings -> indices (TODO: from a type-indexed map, where each
      // production returns the type!).
      let prod_ref_mapping: HashMap<ProductionReference, usize> = prods
        .0
        .iter()
        .map(|(prod_ref, _)| prod_ref)
        .cloned()
        .enumerate()
        .map(|(ind, p)| (p, ind))
        .collect();
      // Collect all the tokens (splitting up literals) as we traverse the
      // productions. So literal strings are "flattened" into their individual
      // tokens.
      let mut all_tokens: IndexSet<Tok> = IndexSet::new();
      // Pretty straightforwardly map the productions into the new space.
      let new_prods: Vec<_> = prods
        .0
        .iter()
        .map(|(_, prod)| {
          let cases: Vec<_> = prod
            .0
            .iter()
            .map(|case| {
              let case_els: Vec<_> = case
                .0
                .iter()
                .flat_map(|el| match el {
                  CaseElement::Lit(literal) => literal
                    .0
                    .iter()
                    .cloned()
                    .map(|cur_tok| {
                      let (tok_ind, _) = all_tokens.insert_full(cur_tok);
                      CaseEl::Tok(TokRef(tok_ind))
                    })
                    .collect::<Vec<_>>(),
                  CaseElement::Prod(prod_ref) => {
                    let prod_ref_ind = prod_ref_mapping
                      .get(prod_ref)
                      .unwrap_or_else(|| panic!("prod ref {:?} not found", prod_ref));
                    vec![CaseEl::Prod(ProdRef(*prod_ref_ind))]
                  },
                })
                .collect();
              CaseImpl(case_els)
            })
            .collect();
          ProductionImpl(cases)
        })
        .collect();
      TokenGrammar {
        graph: LoweredProductions(new_prods),
        alphabet: all_tokens.iter().cloned().collect(),
      }
    }

    pub fn new(prods: &SimultaneousProductions<Tok>) -> Self {
      Self::walk_productions_and_split_literal_strings(prods)
    }

    /* This is a tiny amount of complexity that we can reasonably conceal from
     * the preprocessing step, so we do it here. It could be done in the
     * same preprocessing pass, but we don't care about performance when
     * lowering. */
    pub fn index_token_states(&self) -> IndexMap<Tok, Vec<TokenPosition>> {
      let mut token_states_index: IndexMap<Tok, Vec<TokenPosition>> = IndexMap::new();
      let TokenGrammar {
        graph: LoweredProductions(prods),
        alphabet,
      } = self;
      /* TODO: consider making the iteration over the productions into a helper
       * method! */
      for (prod_ind, the_prod) in prods.iter().enumerate() {
        let cur_prod_ref = ProdRef(prod_ind);
        let ProductionImpl(cases) = the_prod;
        for (case_ind, the_case) in cases.iter().enumerate() {
          let cur_case_ref = CaseRef(case_ind);
          let CaseImpl(elements_of_case) = the_case;
          for (element_of_case_ind, the_element) in elements_of_case.iter().enumerate() {
            let cur_el_ref = CaseElRef(element_of_case_ind);
            match the_element {
              CaseEl::Tok(TokRef(alphabet_token_number)) => {
                let corresponding_token = alphabet.get(*alphabet_token_number)
                  .expect("token references are expected to be internally consistent with the alphabet of a TokenGrammar");
                let cur_pos = TokenPosition {
                  prod: cur_prod_ref,
                  case: cur_case_ref,
                  case_el: cur_el_ref,
                };
                let cur_tok_entry = token_states_index
                  .entry(*corresponding_token)
                  .or_insert(vec![]);
                (*cur_tok_entry).push(cur_pos);
              },
              CaseEl::Prod(_) => (),
            }
          }
        }
      }
      token_states_index
    }
  }
}

///
/// Implementation for getting a `PreprocessedGrammar`. Performance doesn't
/// matter here.
pub mod grammar_indexing {
  use super::{lowering_to_indices::*, *};

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub struct StackSym(pub ProdRef);

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub enum StackStep {
    Positive(StackSym),
    Negative(StackSym),
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub enum NamedOrAnonStep {
    Named(StackStep),
    Anon(AnonStep),
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct StackDiffSegment(pub Vec<NamedOrAnonStep>);

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub struct TrieNodeRef(pub usize);

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct StackTrieNode {
    pub stack_diff: StackDiffSegment,
    /* During parsing, the top of the stack will be a named or anonymous symbol. We can negate
     * that (it should always be a positive step on the top of the stack, so a negative
     * step, I think) to get a NamedOrAnonStep which can index into the relevant segments.
     * This supports stack cycles, as well as using an Rc<StackTrieNode> to manage state
     * during the parse. TODO: make a "build" method that removes the RefCell, coalesces
     * stack diffs, and makes the next nodes an IndexMap (??? on the last part given lex
     * BFS?!)! */
    pub next_nodes: Vec<StackTrieNextEntry>,
    /* Doubly-linked so that they can be traversed from either direction -- this is (maybe) key
     * to parallelism in parsing! */
    pub prev_nodes: Vec<StackTrieNextEntry>,
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
        next_nodes: vec![],
        prev_nodes: vec![],
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
      let trie_node_ref_for_vertex = self.vertex_mapping.get(vtx).cloned().unwrap_or_else(|| {
        let next_ref = TrieNodeRef(self.trie_node_universe.len());
        self.trie_node_universe.push(basic_node.clone());
        next_ref
      });
      /* All trie nodes corresponding to the same vertex should have the same stack
       * diff! */
      assert_eq!(
        self.get_trie(trie_node_ref_for_vertex).stack_diff,
        basic_node.stack_diff
      );
      trie_node_ref_for_vertex
    }
  }

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct ForestEntryExitPoints {
    pub entering_into: Vec<TrieNodeRef>,
    pub exiting_out_of: Vec<TrieNodeRef>,
  }

  impl ForestEntryExitPoints {
    fn new() -> Self {
      ForestEntryExitPoints {
        entering_into: vec![],
        exiting_out_of: vec![],
      }
    }

    fn add_exiting(&mut self, node_ref: TrieNodeRef) { self.exiting_out_of.push(node_ref); }

    fn add_entering(&mut self, node_ref: TrieNodeRef) { self.entering_into.push(node_ref); }
  }

  /* Pointers to the appropriate "forests" of stack transitions
   * starting/completing at each state. "starting" and "completing" are
   * mirrored to allow working away at mapping states to input token indices
   * from either direction, which is intended to allow for parallelism. They're
   * not really "forests" because they *will* have cycles except in very simple
   * grammars (CFGs and below, I think? Unclear if the Chomsky hierarchy
   * still applies). */
  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct StateTransitionGraph {
    pub state_forest_contact_points: IndexMap<LoweredState, ForestEntryExitPoints>,
    pub trie_node_mapping: Vec<StackTrieNode>,
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
    fn get_step(&self) -> Option<NamedOrAnonStep> {
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
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct EpsilonIntervalGraph(pub Vec<ContiguousNonterminalInterval>);

  impl EpsilonIntervalGraph {
    pub fn find_start_end_indices(&self) -> IndexMap<ProdRef, StartEndEpsilonIntervals> {
      let mut epsilon_subscripts_index: IndexMap<ProdRef, StartEndEpsilonIntervals> =
        IndexMap::new();
      let EpsilonIntervalGraph(all_intervals) = self;
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
          _ => panic!("the beginning of an interval should always be a start (epsilon) or end (epsilon prime) vertex"),
        }
      }
      epsilon_subscripts_index
    }

    pub fn connect_all_vertices(&self) -> CyclicGraphDecomposition {
      let intervals_indexed_by_start_and_end = self.find_start_end_indices();
      let EpsilonIntervalGraph(all_intervals) = self;

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
            let cur_trie_ref = ret.trie_ref_for_vertex(&vtx);
            let next_trie_ref = {
              let next_vtx_index = (cur_vtx_index + 1) % vertices.len();
              let next_vertex = vertices[next_vtx_index];
              ret.trie_ref_for_vertex(&next_vertex)
            };
            {
              /* Add a forward link from the current to the next vertex's node. */
              let mut cur_trie = ret.get_trie(cur_trie_ref);
              cur_trie
                .next_nodes
                .push(StackTrieNextEntry::Incomplete(next_trie_ref));
            }
            {
              /* Add a back edge from the next to the current. */
              let mut next_trie = ret.get_trie(next_trie_ref);
              next_trie
                .prev_nodes
                .push(StackTrieNextEntry::Incomplete(cur_trie_ref));
            }
          }
        }
        ret
      };

      CyclicGraphDecomposition {
        cyclic_subgraph: merged_stack_cycles,
        pairwise_state_transitions: all_completed_pairs_with_vertices,
      }
    }

    pub fn produce_transition_graph(&self) -> StateTransitionGraph {
      let CyclicGraphDecomposition {
        cyclic_subgraph: merged_stack_cycles,
        pairwise_state_transitions: all_completed_pairs_with_vertices,
      } = self.connect_all_vertices();

      let mut ret_mapping: IndexMap<LoweredState, ForestEntryExitPoints> = IndexMap::new();
      let mut ret_trie_subgraph = merged_stack_cycles;
      for completed_pair in all_completed_pairs_with_vertices.into_iter() {
        let CompletedStatePairWithVertices {
          state_pair,
          interval,
        } = completed_pair;
        let StatePair {
          left: left_state_in_pair,
          right: right_state_in_pair,
        } = state_pair;
        let ContiguousNonterminalInterval(vertices) = interval;
        for (vertex_index, vtx) in vertices.iter().enumerate() {
          let cur_trie_ref = ret_trie_subgraph.trie_ref_for_vertex(&vtx);
          let next_edge = if vertex_index == vertices.len() - 1 {
            /* Register the current trie as completing at the right state. */
            let right_entry = ret_mapping
              .entry(right_state_in_pair)
              .or_insert_with(ForestEntryExitPoints::new);
            (*right_entry).add_exiting(cur_trie_ref);
            /* To end at the `right` state, add a forward edge to a `Completed` entry. */
            StackTrieNextEntry::Completed(right_state_in_pair)
          } else {
            let next_vertex = vertices[vertex_index + 1];
            let next_trie_ref = ret_trie_subgraph.trie_ref_for_vertex(&next_vertex);
            let mut next_trie = ret_trie_subgraph.get_trie(next_trie_ref);
            next_trie
              .prev_nodes
              .push(StackTrieNextEntry::Incomplete(cur_trie_ref));
            StackTrieNextEntry::Incomplete(next_trie_ref)
          };
          let prev_edge = if vertex_index == 0 {
            /* Register the current trie as emanating from the left state. */
            let left_entry = ret_mapping
              .entry(left_state_in_pair)
              .or_insert_with(ForestEntryExitPoints::new);
            (*left_entry).add_entering(cur_trie_ref);
            /* To start at the `left` state, add a back edge to a `Completed` entry. */
            StackTrieNextEntry::Completed(left_state_in_pair)
          } else {
            let prev_vertex = vertices[vertex_index - 1];
            let prev_trie_ref = ret_trie_subgraph.trie_ref_for_vertex(&prev_vertex);
            let mut prev_trie = ret_trie_subgraph.get_trie(prev_trie_ref);
            prev_trie
              .next_nodes
              .push(StackTrieNextEntry::Incomplete(cur_trie_ref));
            StackTrieNextEntry::Incomplete(prev_trie_ref)
          };
          /* Link the forward and back edges from the current node. */
          let mut cur_trie = ret_trie_subgraph.get_trie(cur_trie_ref);
          cur_trie.next_nodes.push(next_edge);
          cur_trie.prev_nodes.push(prev_edge);
        }
      }

      StateTransitionGraph {
        state_forest_contact_points: ret_mapping,
        trie_node_mapping: ret_trie_subgraph.trie_node_universe,
      }
    }
  }

  /* For some given ProdRef, the intervals of nonterminals which begin at
   * epsilon (start) or epsilon prime (end) for the given ProdRef. This is
   * only a concept in the interval graph and is flattened to a single
   * epsilon/epsilon prime when the PreprocessedGrammar is finally
   * constructed. */
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
    state_pair: StatePair,
    interval: ContiguousNonterminalInterval,
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

  /* This Hash implementation is stable because the collection types in this
   * struct have a specific ordering. */
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

    fn iterate_and_maybe_complete(
      &self,
      indexed_intervals: &IndexMap<ProdRef, StartEndEpsilonIntervals>,
    ) -> TransitionIterationResult
    {
      assert!(!self.cur_traversal_intermediate_nonterminals.is_empty());
      let start = self
        .cur_traversal_intermediate_nonterminals
        .iter()
        .nth(0)
        .unwrap();
      assert!(!self.rest_of_interval.is_empty());
      let next = self.rest_of_interval[0];
      let (intermediate_nonterminals_for_next_step, cycles) = {
        /* Check for cycles. This method supports multiple paths to the same vertex,
         * each of which are a cycle, by pulling out the constituent
         * vertices from the current set of "intermediate" nonterminals. */
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
      };
      let (completed, todo) = match next {
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
          cur_traversal_intermediate_nonterminals: intermediate_nonterminals_for_next_step.clone(),
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
      };
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
  pub struct PreprocessedGrammar<Tok: PartialEq+Eq+Hash+Copy+Clone> {
    // These don't need to be quick to access or otherwise optimized for the algorithm until we
    // create a `Parse` -- these are chosen to reduce redundancy.
    // `M: T -> {Q}`, where `{Q}` is sets of states!
    pub token_states_mapping: IndexMap<Tok, Vec<TokenPosition>>,
    // `A: T x T -> {S}^+_-`, where `{S}^+_-` (LaTeX formatting) is ordered sequences of signed
    // stack symbols!
    pub state_transition_graph: StateTransitionGraph,
  }

  impl<Tok: PartialEq+Eq+Hash+Copy+Clone> PreprocessedGrammar<Tok> {
    /* Intended to reduce visual clutter in the implementation of interval
     * production. */
    fn make_pos_neg_anon_steps(cur_index: usize) -> (EpsilonGraphVertex, EpsilonGraphVertex) {
      (
        EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(cur_index))),
        EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(cur_index))),
      )
    }

    pub fn produce_terminals_interval_graph(grammar: &TokenGrammar<Tok>) -> EpsilonIntervalGraph {
      /* We would like to just accept a LoweredProductions here, but we call this
       * method directly in testing, and without the whole grammar object
       * the type ascription is ambiguous. */
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
      for (prod_ind, the_prod) in prods.iter().enumerate() {
        let cur_prod_ref = ProdRef(prod_ind);
        let ProductionImpl(cases) = the_prod;
        for (case_ind, the_case) in cases.iter().enumerate() {
          let cur_case_ref = CaseRef(case_ind);
          let CaseImpl(elements_of_case) = the_case;
          let mut all_intervals_from_this_case: Vec<ContiguousNonterminalInterval> = vec![];
          let mut cur_elements: Vec<EpsilonGraphVertex> =
            vec![EpsilonGraphVertex::Start(cur_prod_ref)];
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
                let (pos_anon, neg_anon) = Self::make_pos_neg_anon_steps(cur_anon_sym_index);
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
                /* Bump the loop-global anonymous symbol index! */
                cur_anon_sym_index += 1;
                /* Register the interval we just cut off in the results list. */
                all_intervals_from_this_case.push(interval_upto_subprod);
              },
            }
          }
          /* Construct the interval of all remaining nonterminals to the end of the
           * production. */
          let suffix_for_end_of_case = vec![EpsilonGraphVertex::End(cur_prod_ref)];
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
      EpsilonIntervalGraph(really_all_intervals)
    }

    pub fn new(grammar: &TokenGrammar<Tok>) -> Self {
      let terminals_interval_graph = Self::produce_terminals_interval_graph(&grammar);
      let state_transition_graph = terminals_interval_graph.produce_transition_graph();
      PreprocessedGrammar {
        token_states_mapping: grammar.index_token_states(),
        state_transition_graph,
      }
    }
  }
}

///
/// Implementation of parsing. Performance /does/ (eventually) matter here.
pub mod parsing {
  use super::{grammar_indexing::*, *};

  #[derive(Debug, Clone)]
  pub struct Input<Tok: PartialEq+Eq+Hash+Copy+Clone>(Vec<Tok>);

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub struct InputTokenIndex(usize);

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub struct InputRange {
    left: InputTokenIndex,
    right: InputTokenIndex,
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct SpanningSubtree {
    /* This is likely to be converted into a VecDeque<>. */
    contiguous_states: Vec<LoweredState>,
    input_spans: Vec<InputRange>,
    /* TODO: need pointers to the last pair of unions which merged to form this one! */
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub enum ParseGraphEdgeWeight {
    Stack(NamedOrAnonStep),
    State(SpanningSubtree),
  }

  /* #[derive(Debug, Clone, PartialEq, Eq, Hash)] */
  /* pub struct Parse { */
  /* input_mapping: IndexMap<ParseGraphEdgeWeight, Vec<InputRange>>, */
  /* } */
}

#[cfg(test)]
mod tests {
  use super::{grammar_indexing::*, lowering_to_indices::*, user_api::*, *};

  /* TODO: uncomment! */
  /* #[test] */
  /* fn initialize_parse_state() { */
  /* // TODO: figure out more complex parsing such as stack cycles/etc before
   * doing */
  /* // type-indexed maps, as well as syntax sugar for defining cases. */
  /* let prods = SimultaneousProductions( */
  /* [ */
  /* ( */
  /* ProductionReference::new("a"), */
  /* Production(vec![Case(vec![CaseElement::Lit(Literal::new("ab"))])]), */
  /* ), */
  /* ( */
  /* ProductionReference::new("b"), */
  /* Production(vec![Case(vec![ */
  /* CaseElement::Lit(Literal::new("ab")), */
  /* CaseElement::Prod(ProductionReference::new("a")), */
  /* ])]), */
  /* ), */
  /* ].iter() */
  /* .cloned() */
  /* .collect(), */
  /* ); */
  /* let grammar = TokenGrammar::new(&prods); */
  /* let preprocessed_grammar = PreprocessedGrammar::new(&grammar); */
  /* let input: Vec<char> = "abab".chars().collect(); */
  /* let parse = Parse::new(&preprocessed_grammar, input); */
  /* let first_a = TokenPosition { */
  /* prod: ProdRef(0), */
  /* case: CaseRef(0), */
  /* case_el: CaseElRef(0), */
  /* }; */
  /* let second_a = TokenPosition { */
  /* prod: ProdRef(1), */
  /* case: CaseRef(0), */
  /* case_el: CaseElRef(0), */
  /* }; */
  /* let first_b = TokenPosition { */
  /* prod: ProdRef(0), */
  /* case: CaseRef(0), */
  /* case_el: CaseElRef(1), */
  /* }; */
  /* let second_b = TokenPosition { */
  /* prod: ProdRef(1), */
  /* case: CaseRef(0), */
  /* case_el: CaseElRef(1), */
  /* }; */
  /* let into_a_prod = StackStep::Positive(StackSym(ProdRef(0))); */
  /* let out_of_a_prod = StackStep::Negative(StackSym(ProdRef(0))); */
  /* assert_eq!( */
  /* parse, */
  /* Parse(vec![ */
  /* StackTrie { */
  /* stack_steps: vec![StackDiffSegment(vec![])], */
  /* terminal_entries: vec![StackTrieTerminalEntry(vec![ */
  /* UnionRange::new(first_a, InputTokenIndex(1), first_b), */
  /* UnionRange::new(second_a, InputTokenIndex(1), second_b), */
  /* ])], */
  /* }, */
  /* // StackTrie {}, */
  /* StackTrie { */
  /* stack_steps: vec![StackDiffSegment(vec![]),
   * StackDiffSegment(vec![into_a_prod])], */
  /* terminal_entries: vec![StackTrieTerminalEntry(vec![ */
  /* UnionRange::new(first_a, InputTokenIndex(3), first_b), */
  /* UnionRange::new(second_a, InputTokenIndex(3), second_b), */
  /* ])], */
  /* }, */
  /* // StackTrie {}, */
  /* ]) */
  /* ); */
  /* } */

  #[test]
  fn token_grammar_unsorted_alphabet() {
    let prods = SimultaneousProductions(
      [(
        ProductionReference::new("xxx"),
        Production(vec![Case(vec![CaseElement::Lit(Literal::new("cab"))])]),
      )]
      .iter()
      .cloned()
      .collect(),
    );
    let grammar = TokenGrammar::new(&prods);
    assert_eq!(grammar.clone(), TokenGrammar {
      alphabet: vec!['c', 'a', 'b'],
      graph: LoweredProductions(vec![ProductionImpl(vec![CaseImpl(vec![
        CaseEl::Tok(TokRef(0)),
        CaseEl::Tok(TokRef(1)),
        CaseEl::Tok(TokRef(2)),
      ])])]),
    });
  }

  #[test]
  fn token_grammar_construction() {
    let prods = non_cyclic_productions();
    let grammar = TokenGrammar::new(&prods);
    assert_eq!(grammar.clone(), TokenGrammar {
      alphabet: vec!['a', 'b'],
      graph: LoweredProductions(vec![
        ProductionImpl(vec![CaseImpl(vec![
          CaseEl::Tok(TokRef(0)),
          CaseEl::Tok(TokRef(1)),
        ])]),
        ProductionImpl(vec![
          CaseImpl(vec![
            CaseEl::Tok(TokRef(0)),
            CaseEl::Tok(TokRef(1)),
            CaseEl::Prod(ProdRef(0)),
          ]),
          CaseImpl(vec![CaseEl::Prod(ProdRef(0)), CaseEl::Tok(TokRef(0))]),
        ]),
      ]),
    });
  }

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
    let a_start = EpsilonGraphVertex::Start(ProdRef(0));
    let a_0_0 = EpsilonGraphVertex::State(s_0);
    let a_0_1 = EpsilonGraphVertex::State(s_1);
    let a_end = EpsilonGraphVertex::End(ProdRef(0));

    let s_2 = TokenPosition::new(1, 0, 0);
    let s_3 = TokenPosition::new(1, 0, 1);
    let s_4 = TokenPosition::new(1, 1, 1);
    let b_start = EpsilonGraphVertex::Start(ProdRef(1));
    let b_0_0 = EpsilonGraphVertex::State(s_2);
    let b_0_1 = EpsilonGraphVertex::State(s_3);
    let b_0_anon_0_start = EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(0)));
    let b_0_anon_0_end = EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(0)));
    let b_1_anon_0_start = EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(1)));
    let b_1_anon_0_end = EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(1)));
    let b_1_1 = EpsilonGraphVertex::State(s_4);
    let b_end = EpsilonGraphVertex::End(ProdRef(1));

    let a_0 = ContiguousNonterminalInterval(vec![a_start, a_0_0, a_0_1, a_end]);
    let b_start_to_a_start_0 =
      ContiguousNonterminalInterval(vec![b_start, b_0_0, b_0_1, b_0_anon_0_start, a_start]);
    let a_end_to_b_end_0 = ContiguousNonterminalInterval(vec![a_end, b_0_anon_0_end, b_end]);
    let b_start_to_a_start_1 =
      ContiguousNonterminalInterval(vec![b_start, b_1_anon_0_start, a_start]);
    let a_end_to_b_end_1 = ContiguousNonterminalInterval(vec![a_end, b_1_anon_0_end, b_1_1, b_end]);

    assert_eq!(
      noncyclic_interval_graph,
      EpsilonIntervalGraph(vec![
        a_0.clone(),
        b_start_to_a_start_0.clone(),
        a_end_to_b_end_0.clone(),
        b_start_to_a_start_1.clone(),
        a_end_to_b_end_1.clone(),
      ])
    );

    /* Now check for indices. */
    let intervals_by_start_and_end = noncyclic_interval_graph.find_start_end_indices();
    assert_eq!(
      intervals_by_start_and_end,
      vec![
        (ProdRef(0), StartEndEpsilonIntervals {
          start_epsilons: vec![a_0.clone()],
          end_epsilons: vec![a_end_to_b_end_0.clone(), a_end_to_b_end_1.clone()],
        },),
        (ProdRef(1), StartEndEpsilonIntervals {
          start_epsilons: vec![b_start_to_a_start_0.clone(), b_start_to_a_start_1.clone()],
          end_epsilons: vec![],
        },),
      ]
      .iter()
      .cloned()
      .collect::<IndexMap<ProdRef, StartEndEpsilonIntervals>>()
    );

    /* /\* Now check that the transition graph is as we expect. *\/ */
    /* let CyclicGraphDecomposition { */
    /*   cyclic_subgraph: merged_stack_cycles, */
    /*   pairwise_state_transitions: all_completed_pairs_with_vertices, */
    /* } = noncyclic_interval_graph.connect_all_vertices(); */
    /* /\* There are no stack cycles in the noncyclic graph. *\/ */
    /* assert_eq!(merged_stack_cycles, EpsilonNodeStateSubgraph { */
    /*   vertex_mapping: IndexMap::new(), */
    /*   trie_node_universe: vec![], */
    /* }); */
    /* assert_eq!(all_completed_pairs_with_vertices, vec![ */
    /*   CompletedStatePairWithVertices::new( */
    /*     StatePair::new(LoweredState::Start, LoweredState::Within(s_0)), */
    /*     a_0, */
    /*   ), */
    /* ]); */

    /* Now do the same, but for `basic_productions()`. */
    let prods = basic_productions();
    let grammar = TokenGrammar::new(&prods);
    let interval_graph = PreprocessedGrammar::produce_terminals_interval_graph(&grammar);
    assert_eq!(
      interval_graph.clone(),
      EpsilonIntervalGraph(vec![
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(0)),
          EpsilonGraphVertex::State(TokenPosition::new(0, 0, 0)),
          EpsilonGraphVertex::State(TokenPosition::new(0, 0, 1)),
          EpsilonGraphVertex::State(TokenPosition::new(0, 0, 2)),
          EpsilonGraphVertex::End(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(0)),
          EpsilonGraphVertex::State(TokenPosition::new(0, 1, 0)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(0))),
          EpsilonGraphVertex::Start(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(0)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(0))),
          EpsilonGraphVertex::State(TokenPosition::new(0, 1, 2)),
          EpsilonGraphVertex::End(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(0)),
          EpsilonGraphVertex::State(TokenPosition::new(0, 2, 0)),
          EpsilonGraphVertex::State(TokenPosition::new(0, 2, 1)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(1))),
          EpsilonGraphVertex::Start(ProdRef(1)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(1))),
          EpsilonGraphVertex::End(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(2))),
          EpsilonGraphVertex::Start(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(0)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(2))),
          EpsilonGraphVertex::End(ProdRef(1)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(3))),
          EpsilonGraphVertex::Start(ProdRef(1)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(3))),
          EpsilonGraphVertex::End(ProdRef(1)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(4))),
          EpsilonGraphVertex::Start(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(0)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(4))),
          EpsilonGraphVertex::State(TokenPosition::new(1, 2, 1)),
          EpsilonGraphVertex::State(TokenPosition::new(1, 2, 2)),
          EpsilonGraphVertex::End(ProdRef(1)),
        ]),
      ])
    );
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
    let first_a = LoweredState::Within(TokenPosition::new(0, 0, 0));
    let first_b = LoweredState::Within(TokenPosition::new(0, 0, 1));
    let second_a = LoweredState::Within(TokenPosition::new(1, 0, 0));
    let second_b = LoweredState::Within(TokenPosition::new(1, 0, 1));
    let third_a = LoweredState::Within(TokenPosition::new(1, 1, 1));
    let a_prod = StackSym(ProdRef(0));
    let b_prod = StackSym(ProdRef(1));
    assert_eq!(
      preprocessed_grammar.token_states_mapping.clone(),
      vec![
        ('a', vec![
          TokenPosition::new(0, 0, 0),
          TokenPosition::new(1, 0, 0),
          TokenPosition::new(1, 1, 1),
        ],),
        ('b', vec![
          TokenPosition::new(0, 0, 1),
          TokenPosition::new(1, 0, 1)
        ],),
      ]
      .iter()
      .cloned()
      .collect::<IndexMap<char, Vec<TokenPosition>>>(),
    );

    let other_state_transition_graph = StateTransitionGraph {
      state_forest_contact_points: [
        (LoweredState::Start, ForestEntryExitPoints {
          entering_into: vec![TrieNodeRef(0), TrieNodeRef(2)],
          exiting_out_of: vec![],
        }),
        (LoweredState::End, ForestEntryExitPoints {
          entering_into: vec![],
          exiting_out_of: vec![TrieNodeRef(6), TrieNodeRef(7)],
        }),
        (first_a, ForestEntryExitPoints {
          entering_into: vec![TrieNodeRef(3)],
          exiting_out_of: vec![TrieNodeRef(1), TrieNodeRef(9)],
        }),
        (second_a, ForestEntryExitPoints {
          entering_into: vec![TrieNodeRef(4)],
          exiting_out_of: vec![TrieNodeRef(2)],
        }),
        (first_b, ForestEntryExitPoints {
          entering_into: vec![TrieNodeRef(5), TrieNodeRef(8)],
          exiting_out_of: vec![TrieNodeRef(3)],
        }),
        (second_b, ForestEntryExitPoints {
          entering_into: vec![TrieNodeRef(9)],
          exiting_out_of: vec![TrieNodeRef(4)],
        }),
        (third_a, ForestEntryExitPoints {
          entering_into: vec![TrieNodeRef(7)],
          exiting_out_of: vec![TrieNodeRef(8)],
        }),
      ]
      .into_iter()
      .map(|(s, t)| (s.clone(), t.clone()))
      .collect(),

      trie_node_mapping: vec![
        /* = {0,1} */
        /* ( */
        /*   StatePair { */
        /*     left: LoweredState::Start, */
        /*     right: first_a, */
        /*   }, */
        /*   vec![ */
        /*     StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::Positive(a_prod))]), */
        /*     StackDiffSegment(vec![ */
        /*       NamedOrAnonStep::Named(StackStep::Positive(b_prod)), */
        /*       NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(1))), */
        /*       NamedOrAnonStep::Named(StackStep::Positive(a_prod)), */
        /*     ]), */
        /*   ], */
        /* ), */
        /* 0 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::Positive(a_prod))]),
          next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(1))],
          prev_nodes: vec![StackTrieNextEntry::Completed(LoweredState::Start)],
        },
        /* 1 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![
              NamedOrAnonStep::Named(StackStep::Positive(b_prod)),
              NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(1))),
              NamedOrAnonStep::Named(StackStep::Positive(a_prod)),
          ]),
          next_nodes: vec![StackTrieNextEntry::Completed(first_a)],
          prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(0))],
        },
        /* 2 */
        /* ( */
        /*   StatePair { */
        /*     left: LoweredState::Start, */
        /*     right: second_a, */
        /*   }, */
        /*   vec![StackDiffSegment(vec![ */
        /*     NamedOrAnonStep::Named(StackStep::Positive(b_prod)), */
        /*   ])], */
        /* ), */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::Positive(b_prod))]),
          next_nodes: vec![StackTrieNextEntry::Completed(second_a)],
          prev_nodes: vec![StackTrieNextEntry::Completed(LoweredState::Start)],
        },
        /* 3 */
        /* ( */
        /*   StatePair { */
        /*     left: first_a, */
        /*     right: first_b, */
        /*   }, */
        /*   vec![StackDiffSegment(vec![])], */
        /* ), */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![]),
          next_nodes: vec![StackTrieNextEntry::Completed(first_b)],
          prev_nodes: vec![StackTrieNextEntry::Completed(first_a)],
        },
        /* 4 */
        /* ( */
        /*   StatePair { */
        /*     left: second_a, */
        /*     right: second_b, */
        /*   }, */
        /*   vec![StackDiffSegment(vec![])], */
        /* ), */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![]),
          next_nodes: vec![StackTrieNextEntry::Completed(second_b)],
          prev_nodes: vec![StackTrieNextEntry::Completed(second_a)],
        },
        /* = {5,6} */
        /* ( */
        /*   StatePair { */
        /*     left: first_b, */
        /*     right: LoweredState::End, */
        /*   }, */
        /*   vec![ */
        /*     StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::Negative(a_prod))]), */
        /*     StackDiffSegment(vec![ */
        /*       NamedOrAnonStep::Named(StackStep::Negative(a_prod)), */
        /*       NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(0))), */
        /*       NamedOrAnonStep::Named(StackStep::Negative(b_prod)), */
        /*     ]), */
        /*   ], */
        /* ), */
        /* 5 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::Negative(a_prod))]),
          next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(6))],
          prev_nodes: vec![StackTrieNextEntry::Completed(first_b)],
        },
        /* 6 */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![
            NamedOrAnonStep::Named(StackStep::Negative(a_prod)),
            NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(0))),
            NamedOrAnonStep::Named(StackStep::Negative(b_prod)),
          ]),
          next_nodes: vec![StackTrieNextEntry::Completed(LoweredState::End)],
          prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(5))],
        },
        /* 7 */
        /* ( */
        /*   StatePair { */
        /*     left: third_a, */
        /*     right: LoweredState::End, */
        /*   }, */
        /*   vec![StackDiffSegment(vec![NamedOrAnonStep::Named( */
        /*     StackStep::Negative(b_prod), */
        /*   )])], */
        /* ), */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![NamedOrAnonStep::Named(
            StackStep::Negative(b_prod),
          )]),
          next_nodes: vec![StackTrieNextEntry::Completed(LoweredState::End)],
          prev_nodes: vec![StackTrieNextEntry::Completed(third_a)],
        },
        /* 8 */
        /* ( */
        /*   StatePair { */
        /*     left: first_b, */
        /*     right: third_a, */
        /*   }, */
        /*   vec![StackDiffSegment(vec![ */
        /*     NamedOrAnonStep::Named(StackStep::Negative(a_prod)), */
        /*     NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(1))), */
        /*   ])], */
        /* ), */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![
            NamedOrAnonStep::Named(StackStep::Negative(a_prod)),
            NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(1))),
          ]),
          next_nodes: vec![StackTrieNextEntry::Completed(third_a)],
          prev_nodes: vec![StackTrieNextEntry::Completed(first_b)],
        },
        /* 9 */
        /* ( */
        /*   StatePair { */
        /*     left: second_b, */
        /*     right: first_a, */
        /*   }, */
        /*   vec![StackDiffSegment(vec![ */
        /*     NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(0))), */
        /*     NamedOrAnonStep::Named(StackStep::Positive(a_prod)), */
        /*   ])], */
        /* ), */
        StackTrieNode {
          stack_diff: StackDiffSegment(vec![
            NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(0))),
            NamedOrAnonStep::Named(StackStep::Positive(a_prod)),
          ]),
          next_nodes: vec![StackTrieNextEntry::Completed(first_a)],
          prev_nodes: vec![StackTrieNextEntry::Completed(second_b)],
        },
      ],
    };

    assert_eq!(
      preprocessed_grammar.state_transition_graph,
      other_state_transition_graph,
    );
  }

  /* #[test] */
  /* fn cyclic_transition_graph() { */
  /* let prods = basic_productions(); */
  /* let grammar = TokenGrammar::new(&prods); */
  /* let preprocessed_grammar = PreprocessedGrammar::new(&grammar); */
  /* /\* TODO: I've only worked out a few of the transitions right now --
   * circle */
  /*
   * * back after we're sure the cycles are right. *\/ */
  /* assert_eq!(preprocessed_grammar, PreprocessedGrammar { */
  /* token_states_mapping: IndexMap::new(), */
  /* state_transition_graph: StateTransitionGraph(IndexMap::new()), */
  /* },); */
  /* } */

  #[test]
  #[should_panic(expected = "prod ref ProductionReference(\"c\") not found")]
  fn missing_prod_ref() {
    let prods = SimultaneousProductions(
      [(
        ProductionReference::new("b"),
        Production(vec![Case(vec![
          CaseElement::Lit(Literal::new("ab")),
          CaseElement::Prod(ProductionReference::new("c")),
        ])]),
      )]
      .iter()
      .cloned()
      .collect(),
    );
    TokenGrammar::new(&prods);
  }

  fn non_cyclic_productions() -> SimultaneousProductions<char> {
    SimultaneousProductions(
      [
        (
          ProductionReference::new("a"),
          Production(vec![Case(vec![CaseElement::Lit(Literal::new("ab"))])]),
        ),
        (
          ProductionReference::new("b"),
          Production(vec![
            Case(vec![
              CaseElement::Lit(Literal::new("ab")),
              CaseElement::Prod(ProductionReference::new("a")),
            ]),
            Case(vec![
              CaseElement::Prod(ProductionReference::new("a")),
              CaseElement::Lit(Literal::new("a")),
            ]),
          ]),
        ),
      ]
      .iter()
      .cloned()
      .collect(),
    )
  }

  fn basic_productions() -> SimultaneousProductions<char> {
    SimultaneousProductions(
      [
        (
          ProductionReference::new("P_1"),
          Production(vec![
            Case(vec![CaseElement::Lit(Literal::new("abc"))]),
            Case(vec![
              CaseElement::Lit(Literal::new("a")),
              CaseElement::Prod(ProductionReference::new("P_1")),
              CaseElement::Lit(Literal::new("c")),
            ]),
            Case(vec![
              CaseElement::Lit(Literal::new("bc")),
              CaseElement::Prod(ProductionReference::new("P_2")),
            ]),
          ]),
        ),
        (
          ProductionReference::new("P_2"),
          Production(vec![
            Case(vec![CaseElement::Prod(ProductionReference::new("P_1"))]),
            Case(vec![CaseElement::Prod(ProductionReference::new("P_2"))]),
            Case(vec![
              CaseElement::Prod(ProductionReference::new("P_1")),
              CaseElement::Lit(Literal::new("bc")),
            ]),
          ]),
        ),
      ]
      .iter()
      .cloned()
      .collect(),
    )
  }
}
