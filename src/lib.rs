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
extern crate itertools;
extern crate priority_queue;

use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use priority_queue::PriorityQueue;

use std::{
  collections::{HashMap, HashSet, VecDeque},
  fmt::Debug,
  hash::{Hash, Hasher},
};

pub mod user_api {
  use super::*;

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct Literal<Tok: Debug+PartialEq+Eq+Hash+Copy+Clone>(pub Vec<Tok>);

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
  pub enum CaseElement<Tok: Debug+PartialEq+Eq+Hash+Copy+Clone> {
    Lit(Literal<Tok>),
    Prod(ProductionReference),
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct Case<Tok: Debug+PartialEq+Eq+Hash+Copy+Clone>(pub Vec<CaseElement<Tok>>);

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct Production<Tok: Debug+PartialEq+Eq+Hash+Copy+Clone>(pub Vec<Case<Tok>>);

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct SimultaneousProductions<Tok: Debug+PartialEq+Eq+Hash+Copy+Clone>(
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
  pub struct TokenGrammar<Tok: Debug+PartialEq+Eq+Hash+Copy+Clone> {
    pub graph: LoweredProductions,
    pub alphabet: Vec<Tok>,
  }

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct GrammarConstructionError(pub String);

  impl<Tok: Debug+PartialEq+Eq+Hash+Copy+Clone> TokenGrammar<Tok> {
    fn walk_productions_and_split_literal_strings(
      prods: &SimultaneousProductions<Tok>,
    ) -> Result<Self, GrammarConstructionError> {
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
      prods
        .0
        .iter()
        .map(|(_, prod)| {
          prod
            .0
            .iter()
            .map(|case| {
              case
                .0
                .iter()
                .map(|el| match el {
                  CaseElement::Lit(literal) => Ok(
                    literal
                      .0
                      .iter()
                      .cloned()
                      .map(|cur_tok| {
                        let (tok_ind, _) = all_tokens.insert_full(cur_tok);
                        CaseEl::Tok(TokRef(tok_ind))
                      })
                      .collect::<Vec<_>>(),
                  ),
                  CaseElement::Prod(prod_ref) => prod_ref_mapping
                    .get(prod_ref)
                    .map(|i| Ok(vec![CaseEl::Prod(ProdRef(*i))]))
                    .unwrap_or_else(|| {
                      Err(GrammarConstructionError(format!(
                        "prod ref {:?} not found!",
                        prod_ref
                      )))
                    }),
                })
                .collect::<Result<Vec<Vec<CaseEl>>, _>>()
                .map(|unflattened_case_els| {
                  let case_els = unflattened_case_els
                    .into_iter()
                    .flat_map(|els| els.into_iter())
                    .collect::<Vec<CaseEl>>();
                  CaseImpl(case_els)
                })
            })
            .collect::<Result<Vec<CaseImpl>, _>>()
            .map(|cases| ProductionImpl(cases))
        })
        .collect::<Result<Vec<ProductionImpl>, _>>()
        .map(|new_prods| TokenGrammar {
          graph: LoweredProductions(new_prods),
          alphabet: all_tokens.iter().cloned().collect(),
        })
    }

    pub fn new(prods: &SimultaneousProductions<Tok>) -> Result<Self, GrammarConstructionError> {
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
        x.clone()
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

  /* Pointers to the appropriate "forests" of stack transitions
   * starting/completing at each state. "starting" and "completing" are
   * mirrored to allow working away at mapping states to input token indices
   * from either direction, which is intended to allow for parallelism. They're
   * not really "forests" because they *will* have cycles except in very simple
   * grammars (CFGs and below, I think? Unclear if the Chomsky hierarchy
   * still applies). */
  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct StateTransitionGraph {
    pub state_forest_contact_points: IndexMap<LoweredState, TrieNodeRef>,
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
  pub struct PreprocessedGrammar<Tok: Debug+PartialEq+Eq+Hash+Copy+Clone> {
    // These don't need to be quick to access or otherwise optimized for the algorithm until we
    // create a `Parse` -- these are chosen to reduce redundancy.
    // `M: T -> {Q}`, where `{Q}` is sets of states!
    pub token_states_mapping: IndexMap<Tok, Vec<TokenPosition>>,
    // `A: T x T -> {S}^+_-`, where `{S}^+_-` (LaTeX formatting) is ordered sequences of signed
    // stack symbols!
    pub state_transition_graph: CyclicGraphDecomposition,
  }

  impl<Tok: Debug+PartialEq+Eq+Hash+Copy+Clone> PreprocessedGrammar<Tok> {
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
      let state_transition_graph = terminals_interval_graph.connect_all_vertices();
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
  use super::{grammar_indexing::*, lowering_to_indices::*, *};

  #[derive(Debug, Clone)]
  pub struct Input<Tok: Debug+PartialEq+Eq+Hash+Copy+Clone>(pub Vec<Tok>);

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub struct InputTokenIndex(pub usize);

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub struct InputRange {
    left_index: InputTokenIndex,
    right_index: InputTokenIndex,
  }

  impl InputRange {
    fn new(left_index: InputTokenIndex, right_index: InputTokenIndex) -> Self {
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
    state_pair: StatePair,
    input_range: InputRange,
    stack_diff: StackDiffSegment,
  }

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub struct SpanningSubtreeRef(pub usize);

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub struct ParentInfo {
    left_parent: SpanningSubtreeRef,
    right_parent: SpanningSubtreeRef,
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct SpanningSubtreeToCreate {
    pub input_span: FlattenedSpanInfo,
    pub parents: Option<ParentInfo>,
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct SpanningSubtree {
    pub input_span: FlattenedSpanInfo,
    pub parents: Option<ParentInfo>,
    pub id: SpanningSubtreeRef,
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
    /* TODO: ignore cycles for now! */
    pub pairwise_state_transition_table: IndexMap<StatePair, Vec<StackDiffSegment>>,
  }

  impl ParseableGrammar {
    /* TODO: get the transitive closure of this to get all the consecutive series
     * of states *over* length 2 and their corresponding stack diffs -- this
     * enables e.g. the use of SIMD instructions to find those series of
     * states! */
    fn connect_stack_diffs(
      transitions: &Vec<CompletedStatePairWithVertices>,
    ) -> IndexMap<StatePair, Vec<StackDiffSegment>> {
      transitions
        .iter()
        .map(
          |CompletedStatePairWithVertices {
             state_pair,
             interval: ContiguousNonterminalInterval(interval),
           }| {
            let diff: Vec<_> = interval.iter().flat_map(|vtx| vtx.get_step()).collect();
            (state_pair.clone(), StackDiffSegment(diff))
          },
        )
        .group_by(|(pair, _)| pair.clone())
        .into_iter()
        .map(|(pair, entries)| {
          let all_stack_diffs_for_pair: Vec<_> =
            entries.into_iter().map(|(_, segment)| segment).collect();
          (pair.clone(), all_stack_diffs_for_pair)
        })
        .collect::<IndexMap<_, _>>()
    }

    fn get_possible_states_for_input<Tok: Debug+PartialEq+Eq+Hash+Copy+Clone>(
      mapping: &IndexMap<Tok, Vec<TokenPosition>>,
      input: &Input<Tok>,
    ) -> Vec<PossibleStates>
    {
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

    pub fn new<Tok: Debug+PartialEq+Eq+Hash+Copy+Clone>(
      grammar: &PreprocessedGrammar<Tok>,
      input: &Input<Tok>,
    ) -> Self
    {
      let PreprocessedGrammar {
        state_transition_graph:
          CyclicGraphDecomposition {
            pairwise_state_transitions,
            ..
          },
        token_states_mapping,
      } = grammar;
      ParseableGrammar {
        input_as_states: Self::get_possible_states_for_input(token_states_mapping, input),
        pairwise_state_transition_table: Self::connect_stack_diffs(pairwise_state_transitions),
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
    /* NB: Need `left` and `right` indices to know when we're done parsing! */
    pub left_index: InputTokenIndex,
    pub right_index: InputTokenIndex,
    pub spans: PriorityQueue<SpanningSubtree, usize>,
    pub grammar: ParseableGrammar,
    pub finishes_at_left: IndexMap<InputTokenIndex, IndexSet<SpanningSubtree>>,
    pub finishes_at_right: IndexMap<InputTokenIndex, IndexSet<SpanningSubtree>>,
    pub spanning_subtree_table: Vec<SpanningSubtree>,
  }

  impl Parse {
    fn new(
      left_index: InputTokenIndex,
      right_index: InputTokenIndex,
      grammar: &ParseableGrammar,
    ) -> Self
    {
      Parse {
        left_index,
        right_index,
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
            input_range:
              InputRange {
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
    ) -> IndexSet<SpanningSubtreeToCreate>
    {
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
      } = grammar;

      let left_index = InputTokenIndex(0);
      let right_index = InputTokenIndex(input_as_states.len() - 1);

      let mut parse = Self::new(left_index, right_index, grammar);

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
     * so return the resulting stack diff from joining them. */
    fn stack_diff_pair_zipper(
      left_diff: StackDiffSegment,
      right_diff: StackDiffSegment,
    ) -> Option<StackDiffSegment>
    {
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

      (0..min_length)
        .map(|i| {
          (
            cmp_left.get(i).unwrap().clone(),
            cmp_right.get(i).unwrap().clone(),
          )
        })
        .map(|(left_step, right_step)| left_step.sequence(right_step))
        .collect::<Result<Vec<Vec<NamedOrAnonStep>>, _>>()
        .map(|all_steps| {
          all_steps
            .iter()
            .flat_map(|steps| steps.iter().cloned())
            .collect()
        })
        .ok()
        .map(|steps: Vec<NamedOrAnonStep>| {
          /* Put the leftover left and right on the left and right of the resulting
           * stack steps! */
          let all_steps: Vec<_> = leftover_left
            .into_iter()
            .chain(steps)
            .chain(leftover_right.into_iter())
            .collect();
          StackDiffSegment(all_steps)
        })
    }

    pub fn get_spanning_subtree(&self, span_ref: SpanningSubtreeRef) -> Option<&SpanningSubtree> {
      self.spanning_subtree_table.get(span_ref.0)
    }

    pub fn advance(&mut self) -> Result<ParseResult, ParsingFailure> {
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

        /* TODO: ensure all entries of `.finishes_at_left` and `.finishes_at_right`
         * are lexicographically sorted! */
        /* Check all right-neighbors for compatible stack diffs. */
        for right_neighbor in self
          .finishes_at_left
          .get(&InputTokenIndex(cur_right_index + 1))
          .cloned()
          .unwrap_or_else(IndexSet::new)
          .iter()
        {
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
          assert_eq!(right_left_index, (cur_right_index + 1));

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

        /* Check all left-neighbors for compatible stack diffs. */
        for left_neighbor in self
          .finishes_at_right
          .get(&InputTokenIndex(cur_left_index - 1))
          .cloned()
          .unwrap_or_else(IndexSet::new)
          .iter()
        {
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
          assert_eq!(left_right_index, (cur_left_index - 1));

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

        if (InputTokenIndex(cur_left_index) == self.left_index)
          && (InputTokenIndex(cur_right_index) == self.right_index)
        {
          Ok(ParseResult::Complete(cur_span.id))
        } else {
          Ok(ParseResult::Incomplete)
        }
      } else {
        Err(ParsingFailure("no more spans to iterate over!".to_string()))
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::{grammar_indexing::*, lowering_to_indices::*, parsing::*, user_api::*, *};

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
    assert_eq!(
      grammar.clone(),
      Ok(TokenGrammar {
        alphabet: vec!['c', 'a', 'b'],
        graph: LoweredProductions(vec![ProductionImpl(vec![CaseImpl(vec![
          CaseEl::Tok(TokRef(0)),
          CaseEl::Tok(TokRef(1)),
          CaseEl::Tok(TokRef(2)),
        ])])]),
      })
    );
  }

  #[test]
  fn token_grammar_construction() {
    let prods = non_cyclic_productions();
    let grammar = TokenGrammar::new(&prods);
    assert_eq!(
      grammar.clone(),
      Ok(TokenGrammar {
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
      })
    );
  }

  #[test]
  fn token_grammar_state_indexing() {
    let prods = non_cyclic_productions();
    let grammar = TokenGrammar::new(&prods);
    assert_eq!(
      grammar.unwrap().index_token_states(),
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
    let noncyclic_grammar = TokenGrammar::new(&noncyclic_prods).unwrap();
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
    let a_0_0 = EpsilonGraphVertex::State(s_0);
    let a_0_1 = EpsilonGraphVertex::State(s_1);
    let a_end = EpsilonGraphVertex::End(a_prod);

    let b_start = EpsilonGraphVertex::Start(b_prod);
    let b_0_0 = EpsilonGraphVertex::State(s_2);
    let b_0_1 = EpsilonGraphVertex::State(s_3);
    let b_0_anon_0_start = EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(0)));
    let b_0_anon_0_end = EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(0)));
    let b_1_anon_0_start = EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(1)));
    let b_1_anon_0_end = EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(1)));
    let b_1_1 = EpsilonGraphVertex::State(s_4);
    let b_end = EpsilonGraphVertex::End(b_prod);

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
    } = noncyclic_interval_graph.connect_all_vertices();
    /* There are no stack cycles in the noncyclic graph. */
    assert_eq!(merged_stack_cycles, EpsilonNodeStateSubgraph {
      vertex_mapping: IndexMap::new(),
      trie_node_universe: vec![],
    });
    assert_eq!(all_completed_pairs_with_vertices, vec![
      /* 1 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Start, LoweredState::Within(s_0)),
        ContiguousNonterminalInterval(vec![a_start, a_0_0]),
      ),
      /* 2 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Start, LoweredState::Within(s_2)),
        ContiguousNonterminalInterval(vec![b_start, b_0_0]),
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
        StatePair::new(LoweredState::Within(s_1), LoweredState::End),
        ContiguousNonterminalInterval(vec![a_0_1, a_end]),
      ),
      /* 6 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Start, LoweredState::Within(s_0)),
        ContiguousNonterminalInterval(vec![b_start, b_1_anon_0_start, a_start, a_0_0]),
      ),
      /* 7 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Within(s_4), LoweredState::End),
        ContiguousNonterminalInterval(vec![b_1_1, b_end]),
      ),
      /* 8 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Within(s_1), LoweredState::End),
        ContiguousNonterminalInterval(vec![a_0_1, a_end, b_0_anon_0_end, b_end]),
      ),
      /* 9 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Within(s_1), LoweredState::Within(s_4)),
        ContiguousNonterminalInterval(vec![a_0_1, a_end, b_1_anon_0_end, b_1_1]),
      ),
      /* 10 */
      CompletedStatePairWithVertices::new(
        StatePair::new(LoweredState::Within(s_3), LoweredState::Within(s_0)),
        ContiguousNonterminalInterval(vec![b_0_1, b_0_anon_0_start, a_start, a_0_0]),
      ),
    ]);

    /* Now do the same, but for `basic_productions()`. */
    /* TODO: test `.find_start_end_indices()` and `.connect_all_vertices()` here
     * too! */
    let prods = basic_productions();
    let grammar = TokenGrammar::new(&prods).unwrap();
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
    let grammar = TokenGrammar::new(&prods).unwrap();
    let preprocessed_grammar = PreprocessedGrammar::new(&grammar);
    /* let first_a = LoweredState::Within(TokenPosition::new(0, 0, 0)); */
    /* let first_b = LoweredState::Within(TokenPosition::new(0, 0, 1)); */
    /* let second_a = LoweredState::Within(TokenPosition::new(1, 0, 0)); */
    /* let second_b = LoweredState::Within(TokenPosition::new(1, 0, 1)); */
    /* let third_a = LoweredState::Within(TokenPosition::new(1, 1, 1)); */
    /* let a_prod = StackSym(ProdRef(0)); */
    /* let b_prod = StackSym(ProdRef(1)); */
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

    /* let other_state_transition_graph = StateTransitionGraph { */
    /* state_forest_contact_points: [ */
    /* (LoweredState::Start, TrieNodeRef(0)), */
    /* (first_a, TrieNodeRef(1)), */
    /* (second_a, TrieNodeRef(3)), */
    /* (first_b, TrieNodeRef(4)), */
    /* (second_b, TrieNodeRef(5)), */
    /* (LoweredState::End, TrieNodeRef(6)), */
    /* (third_a, TrieNodeRef(8)), */
    /* ] */
    /* .into_iter() */
    /* .map(|(s, t)| (s.clone(), t.clone())) */
    /* .collect(), */

    /* trie_node_mapping: vec![ */
    /* /\* 0 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::
     * Positive(a_prod))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(1))] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Completed(LoweredState::Start), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(7)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(12)), */
    /* ] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* }, */
    /* /\* 1 *\/ */
    /* StackTrieNode { */
    /* stack_diff: StackDiffSegment(vec![]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Completed(first_a), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(4)), */
    /* ] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(0)), */
    /* StackTrieNextEntry::Completed(first_a), */
    /* ] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* }, */
    /* /\* 2 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::
     * Positive(b_prod))]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(3)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(7)), */
    /* ] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Completed(LoweredState::Start)] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* }, */
    /* /\* 3 *\/ */
    /* StackTrieNode { */
    /* stack_diff: StackDiffSegment(vec![]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Completed(second_a), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(5)), */
    /* ] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(2)), */
    /* StackTrieNextEntry::Completed(second_a), */
    /* ] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* }, */
    /* /\* 4 *\/ */
    /* StackTrieNode { */
    /* stack_diff: StackDiffSegment(vec![]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Completed(first_b), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(6)), */
    /* ] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(1)), */
    /* StackTrieNextEntry::Completed(first_b), */
    /* ] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* }, */
    /* /\* 5 *\/ */
    /* StackTrieNode { */
    /* stack_diff: StackDiffSegment(vec![]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Completed(second_b), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(12)), */
    /* ] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(3)), */
    /* StackTrieNextEntry::Completed(second_b), */
    /* ] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* }, */
    /* /\* 6 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::
     * Negative(a_prod))]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Completed(LoweredState::End), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(10)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(11)), */
    /* ] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(4))] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* }, */
    /* /\* 7 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::
     * Positive(AnonSym(1)))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(0))] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(2))] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* }, */
    /* /\* 8 *\/ */
    /* StackTrieNode { */
    /* stack_diff: StackDiffSegment(vec![]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(9)), */
    /* StackTrieNextEntry::Completed(third_a), */
    /* ] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Completed(third_a), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(11)), */
    /* ] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* }, */
    /* /\* 9 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::
     * Negative(b_prod))]), */
    /* next_nodes: vec![StackTrieNextEntry::Completed(LoweredState::End)] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(8)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(10)), */
    /* ] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* }, */
    /* /\* 10 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::
     * Negative(AnonSym(0)))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(9))] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(6))] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* }, */
    /* /\* 11 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::
     * Negative(AnonSym(1)))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(8))] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(6))] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* }, */
    /* /\* 12 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::
     * Positive(AnonSym(0)))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(0))] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(5))] */
    /* .iter() */
    /* .cloned() */
    /* .collect::<IndexSet<_>>(), */
    /* }, */
    /* ], */
    /* }; */

    /* assert_eq!( */
    /* preprocessed_grammar.state_transition_graph, */
    /* other_state_transition_graph, */
    /* ); */
  }

  #[test]
  fn cyclic_transition_graph() {
    let prods = basic_productions();
    let grammar = TokenGrammar::new(&prods).unwrap();
    let preprocessed_grammar = PreprocessedGrammar::new(&grammar);

    /* let first_a = LoweredState::Within(TokenPosition::new(0, 0, 0)); */
    /* let second_a = LoweredState::Within(TokenPosition::new(0, 1, 0)); */

    /* let first_b = LoweredState::Within(TokenPosition::new(0, 0, 1)); */
    /* let second_b = LoweredState::Within(TokenPosition::new(0, 2, 0)); */
    /* let third_b = LoweredState::Within(TokenPosition::new(1, 2, 1)); */

    /* let first_c = LoweredState::Within(TokenPosition::new(0, 0, 2)); */
    /* let second_c = LoweredState::Within(TokenPosition::new(0, 1, 2)); */
    /* let third_c = LoweredState::Within(TokenPosition::new(0, 2, 1)); */
    /* let fourth_c = LoweredState::Within(TokenPosition::new(1, 2, 2)); */

    assert_eq!(
      preprocessed_grammar.token_states_mapping.clone(),
      vec![
        ('a', vec![
          TokenPosition::new(0, 0, 0),
          TokenPosition::new(0, 1, 0)
        ]),
        ('b', vec![
          TokenPosition::new(0, 0, 1),
          TokenPosition::new(0, 2, 0),
          TokenPosition::new(1, 2, 1)
        ]),
        ('c', vec![
          TokenPosition::new(0, 0, 2),
          TokenPosition::new(0, 1, 2),
          TokenPosition::new(0, 2, 1),
          TokenPosition::new(1, 2, 2)
        ]),
      ]
      .into_iter()
      .collect::<IndexMap<_, _>>()
    );

    /* assert_eq!( */
    /* preprocessed_grammar */
    /* .state_transition_graph */
    /* .state_forest_contact_points */
    /* .clone(), */
    /* [ */
    /* (LoweredState::Start, TrieNodeRef(6)), */
    /* (first_a, TrieNodeRef(12)), */
    /* (first_b, TrieNodeRef(14)), */
    /* (second_a, TrieNodeRef(4)), */
    /* (second_b, TrieNodeRef(13)), */
    /* (third_c, TrieNodeRef(15)), */
    /* (first_c, TrieNodeRef(16)), */
    /* (second_c, TrieNodeRef(10)), */
    /* (LoweredState::End, TrieNodeRef(8)), */
    /* (third_b, TrieNodeRef(19)), */
    /* (fourth_c, TrieNodeRef(20)), */
    /* ] */
    /* .into_iter() */
    /* .map(|(s, t)| (s.clone(), t.clone())) */
    /* .collect::<IndexMap<_, _>>() */
    /* ); */

    /* assert_eq!( */
    /* preprocessed_grammar */
    /* .state_transition_graph */
    /* .trie_node_mapping, */
    /* vec![ */
    /* /\* 0 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::Positive( */
    /* StackSym(ProdRef(1)) */
    /* ))]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(1)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(17)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(18)) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(1)), */
    /* StackTrieNextEntry::Completed(LoweredState::Start), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(22)) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 1 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::
     * Positive(AnonSym(3)))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(0))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(0))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 2 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::Negative( */
    /* StackSym(ProdRef(1)) */
    /* ))]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(3)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(7)), */
    /* StackTrieNextEntry::Completed(LoweredState::End) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(3)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(9)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(20)) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 3 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::
     * Negative(AnonSym(3)))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(2))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(2))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 4 *\/ */
    /* StackTrieNode { */
    /* stack_diff: StackDiffSegment(vec![]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(5)), */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(0), */
    /* case: CaseRef(1), */
    /* case_el: CaseElRef(0) */
    /* })) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(6)), */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(0), */
    /* case: CaseRef(1), */
    /* case_el: CaseElRef(0) */
    /* })) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 5 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::
     * Positive(AnonSym(0)))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(6))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(4))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 6 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::Positive( */
    /* StackSym(ProdRef(0)) */
    /* ))]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(4)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(12)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(13)) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(5)), */
    /* StackTrieNextEntry::Completed(LoweredState::Start), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(17)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(18)) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 7 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::
     * Negative(AnonSym(1)))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(8))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(2))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 8 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Named(StackStep::Negative( */
    /* StackSym(ProdRef(0)) */
    /* ))]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(9)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(11)), */
    /* StackTrieNextEntry::Completed(LoweredState::End), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(21)) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(7)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(10)), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(16)) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 9 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::
     * Negative(AnonSym(2)))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(2))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(8))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 10 *\/ */
    /* StackTrieNode { */
    /* stack_diff: StackDiffSegment(vec![]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(8)), */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(0), */
    /* case: CaseRef(1), */
    /* case_el: CaseElRef(2) */
    /* })) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(11)), */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(0), */
    /* case: CaseRef(1), */
    /* case_el: CaseElRef(2) */
    /* })) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 11 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::
     * Negative(AnonSym(0)))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(10))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(8))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 12 *\/ */
    /* StackTrieNode { */
    /* stack_diff: StackDiffSegment(vec![]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(0), */
    /* case: CaseRef(0), */
    /* case_el: CaseElRef(0) */
    /* })), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(14)) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(6)), */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(0), */
    /* case: CaseRef(0), */
    /* case_el: CaseElRef(0) */
    /* })) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 13 *\/ */
    /* StackTrieNode { */
    /* stack_diff: StackDiffSegment(vec![]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(0), */
    /* case: CaseRef(2), */
    /* case_el: CaseElRef(0) */
    /* })), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(15)) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(6)), */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(0), */
    /* case: CaseRef(2), */
    /* case_el: CaseElRef(0) */
    /* })) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 14 *\/ */
    /* StackTrieNode { */
    /* stack_diff: StackDiffSegment(vec![]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(0), */
    /* case: CaseRef(0), */
    /* case_el: CaseElRef(1) */
    /* })), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(16)) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(12)), */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(0), */
    /* case: CaseRef(0), */
    /* case_el: CaseElRef(1) */
    /* })) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 15 *\/ */
    /* StackTrieNode { */
    /* stack_diff: StackDiffSegment(vec![]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(0), */
    /* case: CaseRef(2), */
    /* case_el: CaseElRef(1) */
    /* })), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(22)) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(13)), */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(0), */
    /* case: CaseRef(2), */
    /* case_el: CaseElRef(1) */
    /* })) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 16 *\/ */
    /* StackTrieNode { */
    /* stack_diff: StackDiffSegment(vec![]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(0), */
    /* case: CaseRef(0), */
    /* case_el: CaseElRef(2) */
    /* })), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(8)) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(14)), */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(0), */
    /* case: CaseRef(0), */
    /* case_el: CaseElRef(2) */
    /* })) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 17 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::
     * Positive(AnonSym(2)))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(6))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(0))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 18 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::
     * Positive(AnonSym(4)))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(6))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(0))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 19 *\/ */
    /* StackTrieNode { */
    /* stack_diff: StackDiffSegment(vec![]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(20)), */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(1), */
    /* case: CaseRef(2), */
    /* case_el: CaseElRef(1) */
    /* })) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(1), */
    /* case: CaseRef(2), */
    /* case_el: CaseElRef(1) */
    /* })), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(21)) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 20 *\/ */
    /* StackTrieNode { */
    /* stack_diff: StackDiffSegment(vec![]), */
    /* next_nodes: vec![ */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(1), */
    /* case: CaseRef(2), */
    /* case_el: CaseElRef(2) */
    /* })), */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(2)) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![ */
    /* StackTrieNextEntry::Incomplete(TrieNodeRef(19)), */
    /* StackTrieNextEntry::Completed(LoweredState::Within(TokenPosition { */
    /* prod: ProdRef(1), */
    /* case: CaseRef(2), */
    /* case_el: CaseElRef(2) */
    /* })) */
    /* ] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 21 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::
     * Negative(AnonSym(4)))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(19))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(8))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* /\* 22 *\/ */
    /* StackTrieNode { */
    /* stack_diff:
     * StackDiffSegment(vec![NamedOrAnonStep::Anon(AnonStep::
     * Positive(AnonSym(1)))]), */
    /* next_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(0))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>(), */
    /* prev_nodes: vec![StackTrieNextEntry::Incomplete(TrieNodeRef(15))] */
    /* .into_iter() */
    /* .collect::<IndexSet<_>>() */
    /* }, */
    /* ] */
    /* ); */
  }

  #[test]
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
    assert_eq!(
      TokenGrammar::new(&prods),
      Err(GrammarConstructionError(format!(
        "prod ref ProductionReference(\"c\") not found!"
      )))
    );
  }

  #[test]
  fn initial_parse_state() {
    let prods = non_cyclic_productions();
    let token_grammar = TokenGrammar::new(&prods).unwrap();
    let preprocessed_grammar = PreprocessedGrammar::new(&token_grammar);
    let string_input = "ab";
    let input = Input(string_input.chars().collect());
    let parseable_grammar = ParseableGrammar::new::<char>(&preprocessed_grammar, &input);
    let Parse {
      left_index,
      right_index,
      spans,
      grammar: new_parseable_grammar,
      finishes_at_left,
      finishes_at_right,
      spanning_subtree_table,
    } = Parse::initialize_with_trees_for_adjacent_pairs(&parseable_grammar);
    assert_eq!(new_parseable_grammar, parseable_grammar);

    assert_eq!(left_index, InputTokenIndex(0));
    /* We have to add two because the process creates a start and end token! */
    assert_eq!(right_index, InputTokenIndex((string_input.len() - 1) + 2));

    assert_eq!(spans.into_iter().collect::<Vec<_>>(), vec![]);

    /* NB: These explicit type ascriptions are necessary for some reason... */
    let expected_at_left: IndexMap<InputTokenIndex, IndexSet<SpanningSubtree>> = IndexMap::new();
    assert_eq!(finishes_at_left, expected_at_left);
    let expected_at_right: IndexMap<InputTokenIndex, IndexSet<SpanningSubtree>> = IndexMap::new();
    assert_eq!(finishes_at_right, expected_at_right);

    assert_eq!(spanning_subtree_table, vec![]);
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
