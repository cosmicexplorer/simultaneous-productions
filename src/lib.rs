/*
    Description: Implement the Simultaneous Productions general parsing method.
    Copyright (C) 2019, 2021  Danny McClanahan (https://twitter.com/hipsterelectron)

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
#![feature(trace_macros)]
#![feature(trait_alias)]
/* These clippy lint descriptions are purely non-functional and do not affect the functionality
 * or correctness of the code.
 * TODO: rustfmt breaks multiline comments when used one on top of another! (each with its own
 * pair of delimiters)
 * Note: run clippy with: rustup run nightly cargo-clippy! */
#![allow(missing_docs)]
#![doc(test(attr(deny(warnings))))]
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

/* #[macro_use] */
/* extern crate frunk; */
/* #[macro_use] */
/* extern crate gensym; */
extern crate indexmap;
extern crate priority_queue;
/* #[macro_use] */
/* extern crate quote; */
extern crate typename;

/* use gensym::gensym; */
use indexmap::{IndexMap, IndexSet};
use priority_queue::PriorityQueue;
use typename::TypeName;

/* use frunk::hlist::*; */
use std::{
  collections::{HashMap, HashSet, VecDeque},
  fmt::{self, Debug},
  hash::{Hash, Hasher},
  rc::Rc,
};

pub mod token {
  use typename::TypeName;

  use std::{fmt::Debug, hash::Hash};

  /// The constraints required for any token stream parsed by this crate.
  pub trait Token = Debug+PartialEq+Eq+Hash+Copy+Clone+TypeName;
}

pub mod api {
  use super::{token::*, *};

  #[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
  pub struct Literal<Tok: Token>(pub Vec<Tok>);

  impl From<&str> for Literal<char> {
    fn from(s: &str) -> Self { Self(s.chars().collect()) }
  }

  impl<Tok: Token> From<&[Tok]> for Literal<Tok> {
    fn from(s: &[Tok]) -> Self { Self(s.iter().cloned().collect()) }
  }

  /// A reference to another production.
  ///
  /// The string must match the assigned name of a production in a set of
  /// simultaneous productions.
  #[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
  pub struct ProductionReference(String);

  impl ProductionReference {
    pub fn new(s: &str) -> Self { ProductionReference(s.to_string()) }
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
  pub enum CaseElement<Tok: Token> {
    Lit(Literal<Tok>),
    Prod(ProductionReference),
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
  pub struct Case<Tok: Token>(pub Vec<CaseElement<Tok>>);

  #[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
  pub struct Production<Tok: Token>(pub Vec<Case<Tok>>);

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct SimultaneousProductions<Tok: Token>(
    pub IndexMap<ProductionReference, Production<Tok>>,
  );
}

/// ???
///
/// (I think this is a "model" graph class of some sort, where the model is
/// this "simultaneous productions" parsing formulation. See Spinrad's book
/// [???]!)
///
/// Vec<ProductionImpl> = [
///   Production([
///     Case([CaseEl(Lit("???")), CaseEl(ProdRef(?)), ...]),
///     ...,
///   ]),
///   ...,
/// ]
pub mod lowering_to_indices {
  /// Graph Coordinates
  ///
  /// All these `Ref` types have nice properties, like being storeable without
  /// reference to any particular graph, being totally ordered, and being able
  /// to be incremented.
  ///
  /// We adopt the convention of abbreviated names for things used in
  /// algorithms.
  pub mod graph_coordinates {
    #[cfg(doc)]
    use super::{
      super::api::{Case, Literal, Production, ProductionReference},
      graph_representation::ProductionImpl,
    };

    /// Points to a particular Production within a sequence of [ProductionImpl].
    ///
    /// A version of [ProductionReference] which uses a [usize] for speed.
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct ProdRef(pub usize);

    /// Points to a particular case within a [Production].
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct CaseRef(pub usize);

    /// Points to an element of a particular [Case].
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct CaseElRef(pub usize);

    /// This corresponds to a "state" in the simultaneous productions
    /// terminology.
    ///
    /// This refers to a specific token, implying that we must be pointing to a
    /// particular index of a particular [Literal].
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
  }

  /// Graph Representation
  pub mod graph_representation {
    use super::graph_coordinates::*;

    /// TODO: describe why this struct is here and not in
    /// [super::graph_coordinates]!
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

    impl LoweredProductions {
      pub fn new_production(&mut self) -> (ProdRef, &mut ProductionImpl) {
        let new_end_index = ProdRef(self.0.len());
        self.0.push(ProductionImpl(vec![]));
        (new_end_index, self.0.last_mut().unwrap())
      }
    }
  }

  /// Mapping to Tokens
  pub mod mapping_to_tokens {
    use super::{
      super::{api::*, token::*, *},
      graph_coordinates::*,
      graph_representation::*,
    };

    /// TODO: ???
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct TokenGrammar<Tok: Token> {
      pub graph: LoweredProductions,
      pub alphabet: Vec<Tok>,
    }

    impl<Tok: Token> TokenGrammar<Tok> {
      fn walk_productions_and_split_literal_strings(prods: &SimultaneousProductions<Tok>) -> Self {
        // Mapping from strings -> indices (TODO: from a type-indexed map, where each
        // production returns the type!).
        let prod_ref_mapping: HashMap<ProductionReference, usize> = prods
          .0
          .iter()
          .enumerate()
          .map(|(index, (prod_ref, _))| (prod_ref.clone(), index))
          .collect();
        // Collect all the tokens (splitting up literals) as we traverse the
        // productions. So literal strings are "flattened" into their individual
        // tokens.
        let mut all_tokens: IndexSet<Tok> = IndexSet::new();
        // Pretty straightforwardly map the productions into the new space.
        let mut ret_prods: Vec<ProductionImpl> = Vec::new();
        for (_, prod) in prods.0.iter() {
          let mut ret_cases: Vec<CaseImpl> = Vec::new();
          for case in prod.0.iter() {
            let mut ret_els: Vec<CaseEl> = Vec::new();
            for el in case.0.iter() {
              match el {
                CaseElement::Lit(literal) => {
                  ret_els.extend(literal.0.iter().map(|cur_tok| {
                    let (tok_ind, _) = all_tokens.insert_full(cur_tok.clone());
                    CaseEl::Tok(TokRef(tok_ind))
                  }));
                },
                CaseElement::Prod(prod_ref) => {
                  let matching_production_index = prod_ref_mapping
                    .get(prod_ref)
                    .expect("we assume all prod refs exist at this point");
                  ret_els.push(CaseEl::Prod(ProdRef(*matching_production_index)));
                },
              }
            }
            let cur_case = CaseImpl(ret_els);
            ret_cases.push(cur_case);
          }
          let cur_prod = ProductionImpl(ret_cases);
          ret_prods.push(cur_prod);
        }
        TokenGrammar {
          graph: LoweredProductions(ret_prods),
          alphabet: all_tokens.iter().cloned().collect(),
        }
      }

      pub fn new(prods: &SimultaneousProductions<Tok>) -> Self {
        Self::walk_productions_and_split_literal_strings(prods)
      }

      /// ???
      ///
      /// This is a tiny amount of complexity that we can reasonably conceal
      /// from the preprocessing step, so we do it here. It could be done
      /// in the same preprocessing pass, but we don't care
      /// about performance when lowering.
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
}

/// Implementation for getting a [grammar_indexing::PreprocessedGrammar].
///
/// Performance doesn't matter here.
pub mod grammar_indexing {
  use super::{
    lowering_to_indices::{graph_coordinates::*, graph_representation::*, mapping_to_tokens::*},
    token::*,
    *,
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
      let mut epsilon_subscripts_index: IndexMap<ProdRef, StartEndEpsilonIntervals> =
        IndexMap::new();
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
        .nth(0)
        .unwrap();
      assert!(!self.rest_of_interval.is_empty());
      let next = self.rest_of_interval[0];

      let (intermediate_nonterminals_for_next_step, cycles) = self.check_for_cycles(next.clone());
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
        Self::produce_terminals_interval_graph(&grammar);
      let cyclic_graph_decomposition: CyclicGraphDecomposition =
        terminals_interval_graph.connect_all_vertices();
      PreprocessedGrammar {
        token_states_mapping: grammar.index_token_states(),
        cyclic_graph_decomposition,
      }
    }
  }
}

///
/// Implementation of parsing. Performance /does/ (eventually) matter here.
pub mod parsing {
  use super::{grammar_indexing::*, lowering_to_indices::graph_coordinates::*, token::*, *};

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
}

pub mod reconstruction {
  use super::{grammar_indexing::*, parsing::*, *};

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
}


///
/// Syntax sugar for inline modifications to productions.
pub mod operators {
  use super::lowering_to_indices::graph_representation::*;

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct OperatorResult {
    pub result: Vec<CaseEl>,
  }

  pub trait UnaryOperator {
    fn operate(&self, prods: &mut LoweredProductions) -> OperatorResult;
  }

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct KleeneStar {
    pub group: Vec<CaseEl>,
  }

  impl UnaryOperator for KleeneStar {
    fn operate(&self, prods: &mut LoweredProductions) -> OperatorResult {
      let (new_prod_ref, &mut ProductionImpl(ref mut new_prod)) = prods.new_production();
      /* Add an empty case. */
      new_prod.push(CaseImpl(vec![]));
      /* Add a case traversing the initial group! */
      new_prod.push(CaseImpl(
        self
          .group
          .iter()
          .cloned()
          /* Allow circling back at the end! */
          .chain(vec![CaseEl::Prod(new_prod_ref)])
          .collect(),
      ));
      /* The result is just a single reference to the new production! */
      OperatorResult {
        result: vec![CaseEl::Prod(new_prod_ref)],
      }
    }
  }

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct Repeated {
    pub lower_bound: Option<usize>,
    pub upper_bound: Option<usize>,
    pub group: Vec<CaseEl>,
  }

  impl UnaryOperator for Repeated {
    fn operate(&self, prods: &mut LoweredProductions) -> OperatorResult {
      let prologue_length = self
        .lower_bound
        .map(|i| if i > 0 { i - 1 } else { i })
        .unwrap_or(0);
      let prologue: Vec<CaseEl> = (0..prologue_length)
        .flat_map(|_| self.group.clone())
        .collect();

      let epilogue: Vec<CaseEl> = match self.upper_bound {
        /* If we have a definite upper bound, make up the difference in length from the initial
         * left side. */
        Some(upper_bound) => (0..(upper_bound - prologue_length))
          .flat_map(|_| self.group.clone())
          .collect(),
        /* If not, we can go forever, or not at all, so we can just apply a Kleene star to this! */
        None => {
          let starred = KleeneStar {
            group: self.group.clone(),
          };
          let OperatorResult { result } = starred.operate(prods);
          result
        },
      };

      OperatorResult {
        result: prologue.into_iter().chain(epilogue.into_iter()).collect(),
      }
    }
  }

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct Optional {
    pub group: Vec<CaseEl>,
  }

  impl UnaryOperator for Optional {
    fn operate(&self, prods: &mut LoweredProductions) -> OperatorResult {
      let (new_prod_ref, &mut ProductionImpl(ref mut new_prod)) = prods.new_production();
      /* Add an empty case. */
      new_prod.push(CaseImpl(vec![]));
      /* Add a non-empty case. */
      new_prod.push(CaseImpl(self.group.clone()));
      OperatorResult {
        result: vec![CaseEl::Prod(new_prod_ref)],
      }
    }
  }
}


#[macro_use]
pub mod binding {
  use super::{
    api::*, grammar_indexing::*, lowering_to_indices::graph_coordinates::*, reconstruction::*,
    token::*, *,
  };

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct BindingError(String);

  pub trait ProvidesProduction<Tok: Token> {
    fn as_production(&self) -> Production<Tok>;
    fn get_type_name(&self) -> TypeNameWrapper;
    fn get_acceptors(&self) -> Vec<Rc<Box<dyn PointerBoxingAcceptor>>>;
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
  pub struct TypeNameWrapper(String);

  impl TypeNameWrapper {
    pub fn for_type<T: TypeName>() -> Self { TypeNameWrapper(T::type_name()) }

    pub fn as_production_reference(&self) -> ProductionReference {
      ProductionReference::new(&self.0)
    }
  }

  #[derive(Debug, Clone, PartialEq, Eq, TypeName)]
  pub struct TypedCase<Tok: Token> {
    pub case: Case<Tok>,
    pub acceptor: Rc<Box<dyn PointerBoxingAcceptor>>,
  }

  #[derive(Debug, Clone, PartialEq, Eq, TypeName)]
  pub struct TypedProduction<Tok: Token> {
    cases: Vec<TypedCase<Tok>>,
    output_type: TypeNameWrapper,
  }

  impl<Tok: Token> TypedProduction<Tok> {
    pub fn new<Output: TypeName>(cases: Vec<TypedCase<Tok>>) -> Self {
      TypedProduction {
        cases,
        output_type: TypeNameWrapper::for_type::<Output>(),
      }
    }
  }

  impl<Tok: Token> ProvidesProduction<Tok> for TypedProduction<Tok> {
    fn as_production(&self) -> Production<Tok> {
      Production(
        self
          .cases
          .iter()
          .map(|TypedCase { case, .. }| case)
          .cloned()
          .collect(),
      )
    }

    fn get_type_name(&self) -> TypeNameWrapper { self.output_type.clone() }

    fn get_acceptors(&self) -> Vec<Rc<Box<dyn PointerBoxingAcceptor>>> {
      self
        .cases
        .iter()
        .cloned()
        .map(|TypedCase { acceptor, .. }| acceptor)
        .collect()
    }
  }

  #[derive(Debug, PartialEq, Eq, TypeName)]
  pub struct TypedSimultaneousProductions<
    Tok: Token,
    /* Members: HList, */
  > {
    pub underlying: SimultaneousProductions<Tok>,
    pub bindings: IndexMap<ProdCaseRef, Rc<Box<dyn PointerBoxingAcceptor>>>,
  }

  impl<
      Tok: Token,
      /* Members: HList, */
    >
    TypedSimultaneousProductions<
      Tok,
      /* Members, */
    >
  {
    pub fn reconstruct<Output: TypeName+'static>(
      &self,
      reconstruction: &CompletedWholeReconstruction,
    ) -> Result<Output, BindingError> {
      let mut reconstruction = reconstruction
        .clone()
        .0
        .into_iter()
        .collect::<VecDeque<_>>();
      if reconstruction.len() == 3
        && reconstruction.pop_front().unwrap()
          == CompleteSubReconstruction::State(LoweredState::Start)
        && reconstruction.pop_back().unwrap() == CompleteSubReconstruction::State(LoweredState::End)
      {
        match reconstruction.pop_front().unwrap() {
          CompleteSubReconstruction::Completed(CompletedCaseReconstruction { prod_case, args }) => {
            let acceptor_for_outer = self.bindings.get(&prod_case).ok_or_else(|| {
              BindingError(format!("no case found for prod case ref {:?}!", prod_case))
            })?;
            let TypedProductionParamsDescription { output_type, .. } =
              acceptor_for_outer.type_params();
            let expected_output_type = TypeNameWrapper::for_type::<Output>();
            if output_type != expected_output_type {
              /* FIXME: how do we reasonably accept a type parameter upon reconstruction of
               * a parse? */
              Err(BindingError(format!(
                "output type {:?} for case {:?} did not match expected output type {:?}",
                output_type, prod_case, expected_output_type
              )))
            } else {
              self
                .reconstruct_sub(acceptor_for_outer.clone(), &args)
                .and_then(|result_rc: Box<dyn std::any::Any>| {
                  result_rc.downcast::<Output>().or_else(|_| {
                    Err(BindingError(format!(
                      "prod case {:?} with args {:?} could not be downcast to {:?}",
                      prod_case,
                      args,
                      TypeNameWrapper::for_type::<Output>()
                    )))
                  })
                })
                .map(|x| *x)
            }
          },
          x => Err(BindingError(format!(
            "element {:?} in complete reconstruction {:?} was not recognized",
            x, reconstruction
          ))),
        }
      } else {
        Err(BindingError(format!("reconstruction {:?} was not recognized as a top-level reconstruction (with 3 elements, beginning at Start and ending at End)", reconstruction)))
      }
    }

    fn reconstruct_sub(
      &self,
      acceptor: Rc<Box<dyn PointerBoxingAcceptor>>,
      args: &Vec<CompleteSubReconstruction>,
    ) -> Result<Box<dyn std::any::Any>, BindingError> {
      let sub_args: Vec<Box<dyn std::any::Any>> = args
        .iter()
        .flat_map(|x| match x {
          CompleteSubReconstruction::State(_) => Ok(None),
          CompleteSubReconstruction::Completed(CompletedCaseReconstruction { prod_case, args }) => {
            let acceptor_for_outer = self.bindings.get(prod_case).ok_or_else(|| {
              BindingError(format!("no case found for prod case ref {:?}!", prod_case))
            })?;
            self
              .reconstruct_sub(acceptor_for_outer.clone(), args)
              .map(|x| Some(x))
          },
        })
        .flat_map(|x| x)
        .collect();
      let TypedProductionParamsDescription { params, .. } = acceptor.type_params();
      if sub_args.len() != params.len() {
        Err(BindingError(format!(
          "{:?} args for acceptor {:?} (expected {:?})",
          sub_args.len(),
          &acceptor,
          params.len()
        )))
      } else {
        acceptor
          .accept_erased(sub_args)
          .or_else(|e| Err(BindingError(format!("acceptance error {:?}", e))))
      }
    }

    pub fn new(production_boxes: Vec<Rc<Box<dyn ProvidesProduction<Tok>>>>) -> Self {
      let underlying = SimultaneousProductions(
        production_boxes
          .iter()
          .cloned()
          .map(|prod| {
            (
              prod.get_type_name().as_production_reference(),
              prod.as_production(),
            )
          })
          .collect(),
      );
      let bindings: IndexMap<ProdCaseRef, Rc<Box<dyn PointerBoxingAcceptor>>> = production_boxes
        .iter()
        .cloned()
        .enumerate()
        .flat_map(|(prod_ind, prod)| {
          let cur_prod_ref = ProdRef(prod_ind);
          prod
            .get_acceptors()
            .into_iter()
            .enumerate()
            .map(move |(case_ind, acceptor)| {
              let cur_prod_case_ref = ProdCaseRef {
                prod: cur_prod_ref,
                case: CaseRef(case_ind),
              };
              (cur_prod_case_ref, acceptor)
            })
            .collect::<Vec<_>>()
        })
        .collect();
      TypedSimultaneousProductions {
        underlying,
        bindings,
      }
    }
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
  pub struct ParamName(String);

  impl ParamName {
    pub fn new(s: &str) -> Self { ParamName(s.to_string()) }
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
  pub struct TypedParam {
    arg_type: TypeNameWrapper,
    arg_name: ParamName,
  }

  impl TypedParam {
    pub fn new<T: TypeName>(arg_name: ParamName) -> Self {
      TypedParam {
        arg_type: TypeNameWrapper::for_type::<T>(),
        arg_name,
      }
    }
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
  pub struct TypedProductionParamsDescription {
    output_type: TypeNameWrapper,
    params: Vec<TypedParam>,
  }

  impl TypedProductionParamsDescription {
    pub fn new<T: TypeName>(params: Vec<TypedParam>) -> Self {
      TypedProductionParamsDescription {
        output_type: TypeNameWrapper::for_type::<T>(),
        params,
      }
    }
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct AcceptanceError(String);

  pub trait PointerBoxingAcceptor {
    fn identity_salt(&self) -> &str;
    fn type_params(&self) -> TypedProductionParamsDescription;
    fn accept_erased(
      &self,
      args: Vec<Box<dyn std::any::Any>>,
    ) -> Result<Box<dyn std::any::Any>, AcceptanceError>;
  }

  impl Debug for dyn PointerBoxingAcceptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(f, "PointerBoxingAcceptor({:?})", self.type_params())
    }
  }
  impl PartialEq for dyn PointerBoxingAcceptor {
    fn eq(&self, other: &Self) -> bool { self.identity_salt() == other.identity_salt() }
  }
  impl Eq for dyn PointerBoxingAcceptor {}
  impl Hash for dyn PointerBoxingAcceptor {
    fn hash<H: Hasher>(&self, state: &mut H) { self.identity_salt().hash(state); }
  }
  impl TypeName for dyn PointerBoxingAcceptor {
    fn fmt(f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(f, "dyn {}::PointerBoxingAcceptor", module_path!())
    }
  }

  #[macro_export]
  macro_rules! vec_box_rc {
    ($($x:expr),+) => {
      vec![
        $(
          Rc::new(Box::new($x))
        ),+
      ]
    };
  }

  /* #[macro_export] */
  /* macro_rules! _merge { */
  /* /* This alows merging the head/tail! */ */
  /* (@merge [], [$($rest:tt)*]) => { [$($rest)*] }; */
  /* (@merge [$($cur:tt)*], []) => { [$($cur)*] }; */
  /* (@merge */
  /* [$($cur_arg_name:ident: $cur_arg_type:ty),+], */
  /* [$($rest_arg_name:ident: $rest_arg_type:ty),+]) => { */
  /* [ */
  /* $($cur_arg_name: $cur_arg_type),+ */
  /* , */
  /* $($rest_arg_name: $rest_arg_type),+ */
  /* ] */
  /* }; */
  /* ([$(rest:tt)+]) => { [$($rest)+] } */
  /* } */

  /* #[macro_export] */
  /* macro_rules! _extract_typed_params { */
  /* ([$arg_name:ident: $arg_type:ty]) => { */
  /* [$arg_name: $arg_type] */
  /* }; */
  /* ([$_literal:expr]) => { [] }; */

  /* ($([$arg_name:ident: $arg_type:ty]),+) => { */
  /* [ */
  /* $( */
  /* $arg_name: $arg_type */
  /* ),+ */
  /* ] */
  /* }; */

  /* ([$arg_name:ident: $arg_type:ty], [$_literal:expr], $($rest:tt)+) => { */
  /* _extract_typed_params![[$arg_name: $arg_type], $($rest:tt)+] */
  /* }; */
  /* ([$arg_name:ident: $arg_type:ty], [$_literal:expr]) => { */
  /* [$arg_name: $arg_type] */
  /* }; */

  /* ([$_literal:expr], $($rest:tt)+) => { _extract_typed_params![$($rest)+] */
  /* }; */
  /* } */

  /* fn wow() { */
  /* trace_macros!(true); */
  /* _extract_typed_params![["a"]]; */
  /* _extract_typed_params![[y: u32]]; */
  /* _extract_typed_params![["a"], [y: u32]]; */
  /* _extract_typed_params![[y: u32], ["XXX"]]; */
  /* trace_macros!(false); */
  /* } */

  /* #[macro_export] */
  /* macro_rules! _generate_typed_params_description { */
  /* ($production_type:ty, [$($arg_name:ident: $arg_type:ty),+]) => { */
  /* TypedProductionParamsDescription::new::<$production_type>(vec![ */
  /* $( */
  /* TypedParam::new::<$arg_type>(ParamName::new(stringify!($a))) */
  /* ),+ */
  /* ]) */
  /* }; */
  /* } */

  /* #[macro_export] */
  /* macro_rules! _generate_case { */
  /* ($gen_id:ident, $production_type:ty => [$($decl:tt)+] => $body:block) => */
  /* {{ */
  /* pub struct $gen_id(pub String); */
  /* impl PointerBoxingAcceptor> for $gen_id { */
  /* fn identity_salt(&self) -> &str { */
  /* self.0.as_str() */
  /* } */

  /* fn type_params(&self) -> TypedProductionParamsDescription { */
  /* _generate_typed_params_description![$production_type, */
  /* _extract_typed_params![$($decl)+]] */
  /* } */

  /* fn accept(args: Vec<Box<dyn std::any::Any>>) -> $production_type { */
  /* let rev_args: Vec<_> = args.into_iter().rev().collect(); */
  /* $( */
  /* let $a: $in = rev_args.pop() */
  /* .expect("failed to pop from argument vector!") */
  /* .downcast::<$in>() */
  /* .expect("invalid downcast!"); */
  /* )* */
  /* $body */
  /* } */
  /* } */
  /* let acceptor = Rc::<dyn */
  /* PointerBoxingAcceptor>>::new( */
  /* $gen_id(format!("anonymous class at {}::{}", module_path!(), */
  /* stringify!($gen_id)))); */
  /* let case = Case(vec![$($x),*]); */
  /* TypedCase { case, acceptor } */
  /* }}; */
  /* } */

  /* #[macro_export] */
  /* macro_rules! productions { */
  /* ($($production_type:ty => [ */
  /* $(case ($($decl:tt)+) => $body:block),+ */
  /* ]),+) => { */
  /* TypedSimultaneousProductions::new(vec![ */
  /* $((Box::new(TypedProduction(vec![ */
  /* $( */
  /* gensym!{ _generate_case!{ $production_type => [$($decl)+] => $body } } */
  /* ),* */
  /* ])) */
  /* )),* */
  /* ]) */
  /* }; */
  /* } */
}


#[cfg(test)]
mod tests {
  use super::{
    api::*,
    binding::*,
    grammar_indexing::*,
    lowering_to_indices::{graph_coordinates::*, graph_representation::*, mapping_to_tokens::*},
    parsing::*,
    reconstruction::*,
    token::*,
    *,
  };

  #[test]
  fn token_grammar_unsorted_alphabet() {
    let prods = SimultaneousProductions(
      [(
        ProductionReference::new("xxx"),
        Production(vec![Case(vec![CaseElement::Lit(Literal::from("cab"))])]),
      )]
      .iter()
      .cloned()
      .collect(),
    );
    let grammar = TokenGrammar::new(&prods);
    assert_eq!(grammar, TokenGrammar {
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
    assert_eq!(grammar, TokenGrammar {
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

  #[test]
  fn missing_prod_ref() {
    let prods = SimultaneousProductions(
      [(
        ProductionReference::new("b"),
        Production(vec![Case(vec![
          CaseElement::Lit(Literal::from("ab")),
          CaseElement::Prod(ProductionReference::new("c")),
        ])]),
      )]
      .iter()
      .cloned()
      .collect(),
    );
    let _grammar = TokenGrammar::new(&prods);
    assert!(
      false,
      "ensure production references all exist as a prerequisite on the type level!"
    );
    // assert_eq!(
    //   TokenGrammar::new(&prods),
    //   Err(GrammarConstructionError(format!(
    //     "prod ref ProductionReference(\"c\") not found!"
    //   )))
    // );
  }

  #[test]
  fn dynamic_parse_state() {
    let prods = non_cyclic_productions();

    let token_grammar = TokenGrammar::new(&prods);
    let preprocessed_grammar = PreprocessedGrammar::new(&token_grammar);
    let string_input = "ab";
    let input = Input(string_input.chars().collect());
    let parseable_grammar = ParseableGrammar::new::<char>(preprocessed_grammar, &input);

    assert_eq!(parseable_grammar.input_as_states.clone(), vec![
      PossibleStates(vec![LoweredState::Start]),
      PossibleStates(vec![
        LoweredState::Within(TokenPosition::new(0, 0, 0)),
        LoweredState::Within(TokenPosition::new(1, 0, 0)),
        LoweredState::Within(TokenPosition::new(1, 1, 1)),
      ]),
      PossibleStates(vec![
        LoweredState::Within(TokenPosition::new(0, 0, 1)),
        LoweredState::Within(TokenPosition::new(1, 0, 1)),
      ]),
      PossibleStates(vec![LoweredState::End]),
    ]);

    assert_eq!(
      parseable_grammar.pairwise_state_transition_table.clone(),
      vec![
        (
          StatePair {
            left: LoweredState::Start,
            right: LoweredState::Within(TokenPosition {
              prod: ProdRef(0),
              case: CaseRef(0),
              case_el: CaseElRef(0)
            }),
          },
          vec![
            StackDiffSegment(vec![
              NamedOrAnonStep::Named(StackStep::Positive(StackSym(ProdRef(0)))),
              NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(0))),
            ]),
            StackDiffSegment(vec![
              NamedOrAnonStep::Named(StackStep::Positive(StackSym(ProdRef(1)))),
              NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(3))),
              NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(4))),
              NamedOrAnonStep::Named(StackStep::Positive(StackSym(ProdRef(0)))),
              NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(0))),
            ]),
          ]
        ),
        (
          StatePair {
            left: LoweredState::Start,
            right: LoweredState::Within(TokenPosition {
              prod: ProdRef(1),
              case: CaseRef(0),
              case_el: CaseElRef(0)
            }),
          },
          vec![StackDiffSegment(vec![
            NamedOrAnonStep::Named(StackStep::Positive(StackSym(ProdRef(1)))),
            NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(1))),
          ])],
        ),
        (
          StatePair {
            left: LoweredState::Within(TokenPosition {
              prod: ProdRef(0),
              case: CaseRef(0),
              case_el: CaseElRef(0)
            }),
            right: LoweredState::Within(TokenPosition {
              prod: ProdRef(0),
              case: CaseRef(0),
              case_el: CaseElRef(1)
            }),
          },
          vec![StackDiffSegment(vec![]),]
        ),
        (
          StatePair {
            left: LoweredState::Within(TokenPosition {
              prod: ProdRef(1),
              case: CaseRef(0),
              case_el: CaseElRef(0)
            }),
            right: LoweredState::Within(TokenPosition {
              prod: ProdRef(1),
              case: CaseRef(0),
              case_el: CaseElRef(1)
            }),
          },
          vec![StackDiffSegment(vec![]),]
        ),
        (
          StatePair {
            left: LoweredState::Within(TokenPosition {
              prod: ProdRef(1),
              case: CaseRef(1),
              case_el: CaseElRef(1)
            }),
            right: LoweredState::End,
          },
          vec![StackDiffSegment(vec![
            NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(3))),
            NamedOrAnonStep::Named(StackStep::Negative(StackSym(ProdRef(1)))),
          ])],
        ),
        (
          StatePair {
            left: LoweredState::Within(TokenPosition {
              prod: ProdRef(0),
              case: CaseRef(0),
              case_el: CaseElRef(1)
            }),
            right: LoweredState::End,
          },
          vec![
            StackDiffSegment(vec![
              NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(0))),
              NamedOrAnonStep::Named(StackStep::Negative(StackSym(ProdRef(0)))),
            ]),
            StackDiffSegment(vec![
              NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(0))),
              NamedOrAnonStep::Named(StackStep::Negative(StackSym(ProdRef(0)))),
              NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(2))),
              NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(1))),
              NamedOrAnonStep::Named(StackStep::Negative(StackSym(ProdRef(1)))),
            ]),
          ]
        ),
        (
          StatePair {
            left: LoweredState::Within(TokenPosition {
              prod: ProdRef(0),
              case: CaseRef(0),
              case_el: CaseElRef(1)
            }),
            right: LoweredState::Within(TokenPosition {
              prod: ProdRef(1),
              case: CaseRef(1),
              case_el: CaseElRef(1)
            }),
          },
          vec![StackDiffSegment(vec![
            NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(0))),
            NamedOrAnonStep::Named(StackStep::Negative(StackSym(ProdRef(0)))),
            NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(4))),
          ]),]
        ),
        (
          StatePair {
            left: LoweredState::Within(TokenPosition {
              prod: ProdRef(1),
              case: CaseRef(0),
              case_el: CaseElRef(1)
            }),
            right: LoweredState::Within(TokenPosition {
              prod: ProdRef(0),
              case: CaseRef(0),
              case_el: CaseElRef(0)
            }),
          },
          vec![StackDiffSegment(vec![
            NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(2))),
            NamedOrAnonStep::Named(StackStep::Positive(StackSym(ProdRef(0)))),
            NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(0))),
          ]),]
        ),
      ]
      .into_iter()
      .collect::<IndexMap<StatePair, Vec<StackDiffSegment>>>()
    );

    let mut parse = Parse::initialize_with_trees_for_adjacent_pairs(&parseable_grammar);
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
      vec![
        (
          SpanningSubtree {
            input_span: FlattenedSpanInfo {
              state_pair: StatePair {
                left: LoweredState::Start,
                right: LoweredState::Within(TokenPosition {
                  prod: ProdRef(0),
                  case: CaseRef(0),
                  case_el: CaseElRef(0)
                })
              },
              input_range: InputRange {
                left_index: InputTokenIndex(0),
                right_index: InputTokenIndex(1)
              },
              stack_diff: StackDiffSegment(vec![
                NamedOrAnonStep::Named(StackStep::Positive(StackSym(ProdRef(0)))),
                NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(0))),
              ]),
            },
            parents: None,
            id: SpanningSubtreeRef(0)
          },
          1
        ),
        (
          SpanningSubtree {
            input_span: FlattenedSpanInfo {
              state_pair: StatePair {
                left: LoweredState::Start,
                right: LoweredState::Within(TokenPosition {
                  prod: ProdRef(0),
                  case: CaseRef(0),
                  case_el: CaseElRef(0)
                })
              },
              input_range: InputRange {
                left_index: InputTokenIndex(0),
                right_index: InputTokenIndex(1)
              },
              stack_diff: StackDiffSegment(vec![
                NamedOrAnonStep::Named(StackStep::Positive(StackSym(ProdRef(1)))),
                NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(3))),
                NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(4))),
                NamedOrAnonStep::Named(StackStep::Positive(StackSym(ProdRef(0)))),
                NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(0))),
              ])
            },
            parents: None,
            id: SpanningSubtreeRef(1)
          },
          1
        ),
        (
          SpanningSubtree {
            input_span: FlattenedSpanInfo {
              state_pair: StatePair {
                left: LoweredState::Start,
                right: LoweredState::Within(TokenPosition {
                  prod: ProdRef(1),
                  case: CaseRef(0),
                  case_el: CaseElRef(0)
                })
              },
              input_range: InputRange {
                left_index: InputTokenIndex(0),
                right_index: InputTokenIndex(1)
              },
              stack_diff: StackDiffSegment(vec![
                NamedOrAnonStep::Named(StackStep::Positive(StackSym(ProdRef(1)))),
                NamedOrAnonStep::Anon(AnonStep::Positive(AnonSym(1))),
              ]),
            },
            parents: None,
            id: SpanningSubtreeRef(2)
          },
          1
        ),
        (
          SpanningSubtree {
            input_span: FlattenedSpanInfo {
              state_pair: StatePair {
                left: LoweredState::Within(TokenPosition {
                  prod: ProdRef(0),
                  case: CaseRef(0),
                  case_el: CaseElRef(0)
                }),
                right: LoweredState::Within(TokenPosition {
                  prod: ProdRef(0),
                  case: CaseRef(0),
                  case_el: CaseElRef(1)
                })
              },
              input_range: InputRange {
                left_index: InputTokenIndex(1),
                right_index: InputTokenIndex(2)
              },
              stack_diff: StackDiffSegment(vec![])
            },
            parents: None,
            id: SpanningSubtreeRef(3)
          },
          1
        ),
        (
          SpanningSubtree {
            input_span: FlattenedSpanInfo {
              state_pair: StatePair {
                left: LoweredState::Within(TokenPosition {
                  prod: ProdRef(1),
                  case: CaseRef(0),
                  case_el: CaseElRef(0)
                }),
                right: LoweredState::Within(TokenPosition {
                  prod: ProdRef(1),
                  case: CaseRef(0),
                  case_el: CaseElRef(1)
                })
              },
              input_range: InputRange {
                left_index: InputTokenIndex(1),
                right_index: InputTokenIndex(2)
              },
              stack_diff: StackDiffSegment(vec![])
            },
            parents: None,
            id: SpanningSubtreeRef(4)
          },
          1
        ),
        (
          SpanningSubtree {
            input_span: FlattenedSpanInfo {
              state_pair: StatePair {
                left: LoweredState::Within(TokenPosition {
                  prod: ProdRef(0),
                  case: CaseRef(0),
                  case_el: CaseElRef(1)
                }),
                right: LoweredState::End
              },
              input_range: InputRange {
                left_index: InputTokenIndex(2),
                right_index: InputTokenIndex(3)
              },
              stack_diff: StackDiffSegment(vec![
                NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(0))),
                NamedOrAnonStep::Named(StackStep::Negative(StackSym(ProdRef(0)))),
              ])
            },
            parents: None,
            id: SpanningSubtreeRef(5)
          },
          1
        ),
        (
          SpanningSubtree {
            input_span: FlattenedSpanInfo {
              state_pair: StatePair {
                left: LoweredState::Within(TokenPosition {
                  prod: ProdRef(0),
                  case: CaseRef(0),
                  case_el: CaseElRef(1)
                }),
                right: LoweredState::End
              },
              input_range: InputRange {
                left_index: InputTokenIndex(2),
                right_index: InputTokenIndex(3)
              },
              stack_diff: StackDiffSegment(vec![
                NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(0))),
                NamedOrAnonStep::Named(StackStep::Negative(StackSym(ProdRef(0)))),
                NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(2))),
                NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(1))),
                NamedOrAnonStep::Named(StackStep::Negative(StackSym(ProdRef(1))))
              ]),
            },
            parents: None,
            id: SpanningSubtreeRef(6)
          },
          1
        )
      ]
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
    let expected_at_left: IndexMap<InputTokenIndex, IndexSet<SpanningSubtree>> = vec![
      (InputTokenIndex(0), collect_spans(&all_spans, vec![0, 1, 2])),
      (InputTokenIndex(1), collect_spans(&all_spans, vec![3, 4])),
      (InputTokenIndex(2), collect_spans(&all_spans, vec![5, 6])),
    ]
    .into_iter()
    .collect();
    assert_eq!(finishes_at_left, expected_at_left);

    let expected_at_right: IndexMap<InputTokenIndex, IndexSet<SpanningSubtree>> = vec![
      (InputTokenIndex(1), collect_spans(&all_spans, vec![0, 1, 2])),
      (InputTokenIndex(2), collect_spans(&all_spans, vec![3, 4])),
      (InputTokenIndex(3), collect_spans(&all_spans, vec![5, 6])),
    ]
    .into_iter()
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
        state_pair: StatePair {
          left: LoweredState::Start,
          right: LoweredState::End,
        },
        input_range: InputRange::new(InputTokenIndex(0), InputTokenIndex(3)),
        stack_diff: StackDiffSegment(vec![]),
      },
      parents: Some(ParentInfo {
        left_parent: SpanningSubtreeRef(7),
        right_parent: SpanningSubtreeRef(5),
      }),
      id: SpanningSubtreeRef(9),
    };

    let expected_subtree = SpanningSubtree {
      input_span: FlattenedSpanInfo {
        state_pair: StatePair {
          left: LoweredState::Start,
          right: LoweredState::End,
        },
        input_range: InputRange::new(InputTokenIndex(0), InputTokenIndex(3)),
        stack_diff: StackDiffSegment(vec![
          NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(2))),
          NamedOrAnonStep::Anon(AnonStep::Negative(AnonSym(1))),
          NamedOrAnonStep::Named(StackStep::Negative(StackSym(ProdRef(1)))),
        ]),
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
        states: vec![
          LoweredState::Start,
          LoweredState::Within(TokenPosition {
            prod: ProdRef(0),
            case: CaseRef(0),
            case_el: CaseElRef(0)
          }),
          LoweredState::Within(TokenPosition {
            prod: ProdRef(0),
            case: CaseRef(0),
            case_el: CaseElRef(1)
          }),
          LoweredState::End,
        ],
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

  #[test]
  fn extract_typed_production() {
    /* FIXME: turn this into a really neat macro!!! */
    let example = TypedSimultaneousProductions::new(vec_box_rc![
      TypedProduction::new::<u64>(vec![TypedCase {
        /* FIXME: this breaks when we try to use a 1-length string!!! */
        case: Case(vec![CaseElement::Lit(Literal::from("2"))]),
        acceptor: Rc::new(Box::new({
          struct GeneratedStruct;
          impl PointerBoxingAcceptor for GeneratedStruct {
            fn identity_salt(&self) -> &str { "salt1!" }

            fn type_params(&self) -> TypedProductionParamsDescription {
              TypedProductionParamsDescription::new::<u64>(vec![])
            }

            fn accept_erased(
              &self,
              _args: Vec<Box<dyn std::any::Any>>,
            ) -> Result<Box<dyn std::any::Any>, AcceptanceError> {
              /* FIXME: how do i get access to the states we've traversed at all? Do I
               * care? */
              Ok(Box::new({
                let res: u64 = { 2 as u64 };
                res
              }))
            }
          }
          GeneratedStruct
        }))
      }]),
      TypedProduction::new::<usize>(vec![TypedCase {
        /* FIXME: this breaks when we try to use a 1-length string!!! */
        case: Case(vec![
          CaseElement::Prod(TypeNameWrapper::for_type::<u64>().as_production_reference()),
          CaseElement::Lit(Literal::from("+")),
          CaseElement::Prod(TypeNameWrapper::for_type::<u64>().as_production_reference()),
        ]),
        acceptor: Rc::new(Box::new({
          struct GeneratedStruct;
          impl PointerBoxingAcceptor for GeneratedStruct {
            fn identity_salt(&self) -> &str { "salt2!" }

            fn type_params(&self) -> TypedProductionParamsDescription {
              TypedProductionParamsDescription::new::<usize>(vec![
                TypedParam::new::<u64>(ParamName::new("x")),
                TypedParam::new::<u64>(ParamName::new("y")),
              ])
            }

            fn accept_erased(
              &self,
              args: Vec<Box<dyn std::any::Any>>,
            ) -> Result<Box<dyn std::any::Any>, AcceptanceError> {
              let mut args: VecDeque<_> = args.into_iter().collect();
              assert_eq!(args.len(), 2);
              let x: u64 = *args.pop_front().unwrap().downcast::<u64>().unwrap();
              let y: u64 = *args.pop_back().unwrap().downcast::<u64>().unwrap();
              Ok(Box::new({
                use std::convert::TryInto;
                let res: usize = { (x + y).try_into().unwrap() };
                res
              }))
            }
          }
          GeneratedStruct
        }))
      }])
    ]);
    let token_grammar = TokenGrammar::new(&example.underlying);
    let preprocessed_grammar = PreprocessedGrammar::new(&token_grammar);
    /* FIXME: THE ERROR OUTPUT FOR THIS IS INCREDIBLE -- PLEASE TEST IT!!!!

        let string_input = "2+1";

    `cargo test` then produces:

        thread 'tests::extract_typed_production' panicked at 'no tokens found for token '1' in input Input(['2', '+', '1'])', src/libcore/option.rs:1166:5

     */
    let string_input = "2+2";
    let input = Input(string_input.chars().collect());
    let parseable_grammar = ParseableGrammar::new::<char>(preprocessed_grammar, &input);
    let mut parse = Parse::initialize_with_trees_for_adjacent_pairs(&parseable_grammar);
    let parsed_string = parse.get_next_parse();
    let reconstructed_parse = InProgressReconstruction::new(parsed_string, &parse);
    let completely_reconstructed_parse = CompletedWholeReconstruction::new(reconstructed_parse);
    assert_eq!(
      example
        .reconstruct::<usize>(&completely_reconstructed_parse)
        .unwrap(),
      4 as usize
    );

    /* assert_eq!( */
    /* { */
    /* trace_macros!(true); */
    /* let res = productions![ */
    /* u32 => [ */
    /* case ( */
    /* _x: Vec<char> => CaseElement::Lit(Literal::from("1")) */
    /* ) => { */
    /* 1 */
    /* } */
    /* ], */
    /* Vec<i64> => [ */
    /* case ( */
    /* _x: Vec<char> => CaseElement::Lit(Literal::from("a")), */
    /* y: u32 => CaseElement::Prod(ProductionReference::<u32>::new()), */
    /* _z: Vec<char> => CaseElement::Lit(Literal::from("a")) */
    /* ) => { */
    /* asdf(); */
    /* } */
    /* ] */
    /* ]; */
    /* trace_macros!(false); */
    /* }, */
    /* example */
    /* ); */
  }

  fn non_cyclic_productions() -> SimultaneousProductions<char> {
    SimultaneousProductions(
      [
        (
          ProductionReference::new("a"),
          Production(vec![Case(vec![CaseElement::Lit(Literal::from("ab"))])]),
        ),
        (
          ProductionReference::new("b"),
          Production(vec![
            Case(vec![
              CaseElement::Lit(Literal::from("ab")),
              CaseElement::Prod(ProductionReference::new("a")),
            ]),
            Case(vec![
              CaseElement::Prod(ProductionReference::new("a")),
              CaseElement::Lit(Literal::from("a")),
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
            Case(vec![CaseElement::Lit(Literal::from("abc"))]),
            Case(vec![
              CaseElement::Lit(Literal::from("a")),
              CaseElement::Prod(ProductionReference::new("P_1")),
              CaseElement::Lit(Literal::from("c")),
            ]),
            Case(vec![
              CaseElement::Lit(Literal::from("bc")),
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
              CaseElement::Lit(Literal::from("bc")),
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
