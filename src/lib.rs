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

extern crate indexmap;

use indexmap::{IndexMap, IndexSet};

use std::{
  collections::{HashMap, VecDeque},
  hash::Hash,
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
///
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
                      .expect(&format!("prod ref {:?} not found", prod_ref));
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
///
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
  pub enum ParseTimeStateMachine {
    Named(StackStep),
    Anon(AnonStep),
    /* This makes it recursive! SingleStackCycle currently creates input for Kleene(), and it
     * does not itself contain Kleene stars (TODO: ???). */
    Kleene(StackDiff),
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct StackDiff(pub Vec<ParseTimeStateMachine>);

  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct StateTransitionGraph {
    pub graph: IndexMap<StatePair, Vec<StackDiff>>,
    /* NB: This is a formulation of stack cycles which is usable in both parsing directions! */
    pub cycles: IndexMap<ProdRef, Vec<StackDiff>>,
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
    fn to_steps(&self) -> Vec<ParseTimeStateMachine> {
      match self {
        EpsilonGraphVertex::Start(prod_ref) => vec![ParseTimeStateMachine::Named(
          StackStep::Positive(StackSym(*prod_ref)),
        )],
        EpsilonGraphVertex::End(prod_ref) => vec![ParseTimeStateMachine::Named(
          StackStep::Negative(StackSym(*prod_ref)),
        )],
        EpsilonGraphVertex::Anon(anon_step) => vec![ParseTimeStateMachine::Anon(*anon_step)],
        EpsilonGraphVertex::State(_) => {
          /* NB: This should always be at the end of the "nonterminals"! */
          vec![]
        },
      }
    }
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct ContiguousNonterminalInterval(pub Vec<EpsilonGraphVertex>);

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
        let first = vertices.get(0).unwrap();
        match first {
          EpsilonGraphVertex::Start(start_prod_ref) => {
            let intervals_for_this_prod = epsilon_subscripts_index.entry(*start_prod_ref)
              .or_insert(StartEndEpsilonIntervals::new());
            (*intervals_for_this_prod).start_epsilons.push(interval.clone());
          },
          EpsilonGraphVertex::End(end_prod_ref) => {
            let intervals_for_this_prod = epsilon_subscripts_index.entry(*end_prod_ref)
              .or_insert(StartEndEpsilonIntervals::new());
            (*intervals_for_this_prod).end_epsilons.push(interval.clone());
          },
          _ => panic!("the beginning of an interval should always be a start (epsilon) or end (epsilon prime) vertex"),
        }
      }
      epsilon_subscripts_index
    }

    pub fn produce_transition_graph(&self) -> StateTransitionGraph {
      let intervals_indexed_by_start_and_end = self.find_start_end_indices();
      let EpsilonIntervalGraph(all_intervals) = self;
      let mut all_completed_pairs_with_vertices: Vec<CompletedStatePairWithVertices> = vec![];
      let mut traversal_queue: VecDeque<IntermediateTokenTransition> = all_intervals
        .iter()
        .map(IntermediateTokenTransition::new)
        .collect();
      let mut all_stack_cycles: Vec<SingleStackCycle> = vec![];
      while !traversal_queue.is_empty() {
        let cur_transition = traversal_queue.pop_front().unwrap();
        let TransitionIterationResult {
          completed,
          todo,
          cycles,
        } = cur_transition.iterate_and_maybe_complete(&intervals_indexed_by_start_and_end);
        all_completed_pairs_with_vertices.extend(completed);
        traversal_queue.extend(todo);
        all_stack_cycles.extend(cycles);
      }
      let mut grouped_nonterminal_strings: IndexMap<
        StatePair,
        Vec<ContiguousNonterminalInterval>,
      > = IndexMap::new();
      for completed_pair in all_completed_pairs_with_vertices.into_iter() {
        let CompletedStatePairWithVertices {
          state_pair,
          interval,
        } = completed_pair;
        let entry = grouped_nonterminal_strings
          .entry(state_pair)
          .or_insert(vec![]);
        (*entry).push(interval);
      }
      /* TODO: /keep/ cycles in TransitionIterationResult (they should never have
       * left there anyway), and give that struct the power to create its
       * own transitions (`StackDiff`s), and then for EACH node of EVERY
       * cycle, replace ALL instances of it with itself + the
       * ParseTimeStateMachine::Kleene() of the cycle's vertices (that should be
       * itself a "flat" StackDiff (with no sub-Kleene()s (TODO: ???))). */
      let converted_into_transitions: IndexMap<StatePair, Vec<StackDiff>> =
        grouped_nonterminal_strings
          .into_iter()
          .map(|(pair, intervals)| {
            let stack_diffs: Vec<StackDiff> = intervals
              .iter()
              .map(|ContiguousNonterminalInterval(nonterminals)| {
                let cur_stack_steps: Vec<ParseTimeStateMachine> = nonterminals
                  .iter()
                  .flat_map(|vtx| vtx.to_steps())
                  .collect();
                StackDiff(cur_stack_steps)
              })
              /* NB: Make unique, keeping order. */
              .collect::<IndexSet<_>>()
              .into_iter()
              .collect();
            (pair, stack_diffs)
          })
          .collect();
      StateTransitionGraph {
        graph: converted_into_transitions,
        /* TODO: fill this out for stack cycles! */
        cycles: IndexMap::new(),
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
  struct CompletedStatePairWithVertices {
    state_pair: StatePair,
    interval: ContiguousNonterminalInterval,
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  struct SingleStackCycle(Vec<EpsilonGraphVertex>);

  impl SingleStackCycle {
    fn to_diff(&self) -> StackDiff {
      let SingleStackCycle(vertices) = self;
      let stack_steps: Vec<ParseTimeStateMachine> =
        vertices.iter().flat_map(|vtx| vtx.to_steps()).collect();
      StackDiff(stack_steps)
    }
  }

  #[derive(Debug, Clone, PartialEq, Eq)]
  struct TransitionIterationResult {
    completed: Vec<CompletedStatePairWithVertices>,
    todo: Vec<IntermediateTokenTransition>,
    cycles: Vec<SingleStackCycle>,
  }

  #[derive(Debug, Clone, PartialEq, Eq)]
  struct IntermediateTokenTransition {
    start: EpsilonGraphVertex,
    cur_traversal_intermediate_nonterminals: IndexSet<EpsilonGraphVertex>,
    rest_of_interval: Vec<EpsilonGraphVertex>,
  }

  impl IntermediateTokenTransition {
    fn new(wrapped_interval: &ContiguousNonterminalInterval) -> Self {
      let ContiguousNonterminalInterval(interval) = wrapped_interval;
      /* All intervals have a start and end node. */
      assert!(interval.len() >= 2);
      let start = interval.get(0).unwrap();
      let rest_of_interval = interval.get(1..).unwrap().to_vec();
      IntermediateTokenTransition {
        start: *start,
        cur_traversal_intermediate_nonterminals: IndexSet::new(),
        rest_of_interval,
      }
    }

    fn iterate_and_maybe_complete(
      &self,
      indexed_intervals: &IndexMap<ProdRef, StartEndEpsilonIntervals>,
    ) -> TransitionIterationResult
    {
      assert!(!self.rest_of_interval.is_empty());
      let next = self.rest_of_interval.get(0).unwrap();
      let (intermediate_nonterminals_for_next_step, cycles) = {
        /* Check for cycles. This method supports multiple paths to the same vertex,
         * each of which are a cycle, by pulling out the constituent
         * vertices from the current set of "intermediate" nonterminals. */
        let mut prev_nonterminals = self.cur_traversal_intermediate_nonterminals.clone();
        let (cur_vtx_ind, was_new_insert) = prev_nonterminals.insert_full(next.clone());
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
            .get(start_prod_ref)
            .expect("all `ProdRef`s should have been accounted for when grouping by start and end intervals")
            .start_epsilons
            .iter()
            .map(|ContiguousNonterminalInterval(next_vertices)| {
              IntermediateTokenTransition {
                start: self.start,
                cur_traversal_intermediate_nonterminals: intermediate_nonterminals_for_next_step.clone(),
                /* Get the rest of the interval without the epsilon node that it starts with. */
                rest_of_interval: next_vertices.get(1..).unwrap().to_vec(),
              }
            })
            .collect();
          TransitionIterationResult {
            completed: vec![],
            todo: passthrough_intermediates,
            cycles,
          }
        },
        /* Similarly to ending on a Start vertex. */
        EpsilonGraphVertex::End(end_prod_ref) => {
          /* We only have this single next node, since we always start or end at a
           * start or end. */
          assert_eq!(self.rest_of_interval.len(), 1);
          let completed_path_makes_sense = match self.start {
            EpsilonGraphVertex::State(_) => true,
            EpsilonGraphVertex::Start(_) => true,
            EpsilonGraphVertex::End(_) => false,
            EpsilonGraphVertex::Anon(_) => {
              panic!("an anonymous vertex should not be at the start of an interval!")
            },
          };
          let completed = if completed_path_makes_sense {
            let completed_state_pair = StatePair {
              left: LoweredState::from_vertex(self.start),
              right: LoweredState::End,
            };
            let relevant_interval_with_terminals: Vec<EpsilonGraphVertex> = vec![
              self.start.clone(),
            ].iter()
              .chain(intermediate_nonterminals_for_next_step.iter())
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
            .get(end_prod_ref)
            .expect("all `ProdRef`s should have been accounted for when grouping by start and end intervals")
            .end_epsilons
            .iter()
            .map(|ContiguousNonterminalInterval(next_vertices)| {
              IntermediateTokenTransition {
                start: self.start,
                cur_traversal_intermediate_nonterminals: intermediate_nonterminals_for_next_step.clone(),
                /* Get the rest of the interval without the epsilon node that it starts with. */
                rest_of_interval: next_vertices.get(1..).unwrap().to_vec(),
              }
            })
            .collect();
          TransitionIterationResult {
            completed,
            todo: passthrough_intermediates,
            cycles,
          }
        },
        /* `next` is the anonymous vertex, which is all we need it for. */
        EpsilonGraphVertex::Anon(_) => TransitionIterationResult {
          completed: vec![],
          todo: vec![IntermediateTokenTransition {
            start: self.start,
            cur_traversal_intermediate_nonterminals: intermediate_nonterminals_for_next_step
              .clone(),
            rest_of_interval: self.rest_of_interval.get(1..).unwrap().to_vec(),
          }],
          cycles,
        },
        /* Similar to start and end, but the `todo` starts off at the state. */
        EpsilonGraphVertex::State(state_pos) => {
          let completed_state_pair = StatePair {
            left: LoweredState::from_vertex(self.start),
            right: LoweredState::Within(*state_pos),
          };
          let completed_path_makes_sense = match self.start {
            EpsilonGraphVertex::State(_) => true,
            EpsilonGraphVertex::Start(_) => true,
            EpsilonGraphVertex::End(_) => false,
            EpsilonGraphVertex::Anon(_) => {
              panic!("an anonymous vertex should not be at the start of an interval!")
            },
          };
          let completed = if completed_path_makes_sense {
            let relevant_interval_with_terminals: Vec<EpsilonGraphVertex> = vec![
              self.start.clone(),
            ].iter()
              .chain(intermediate_nonterminals_for_next_step.iter())
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
          TransitionIterationResult {
            completed,
            todo: vec![IntermediateTokenTransition {
              /* NB: starting off /at/ the current state vertex! */
              start: next.clone(),
              cur_traversal_intermediate_nonterminals: IndexSet::new(),
              rest_of_interval: self.rest_of_interval.get(1..).unwrap().to_vec(),
            }],
            cycles,
          }
        },
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
                cur_anon_sym_index = cur_anon_sym_index + 1;
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
  /* stack_steps: vec![StackDiff(vec![])], */
  /* terminal_entries: vec![StackTrieTerminalEntry(vec![ */
  /* UnionRange::new(first_a, InputTokenIndex(1), first_b), */
  /* UnionRange::new(second_a, InputTokenIndex(1), second_b), */
  /* ])], */
  /* }, */
  /* // StackTrie {}, */
  /* StackTrie { */
  /* stack_steps: vec![StackDiff(vec![]), StackDiff(vec![into_a_prod])], */
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
      )].iter()
        .cloned()
        .collect(),
    );
    let grammar = TokenGrammar::new(&prods);
    assert_eq!(
      grammar.clone(),
      TokenGrammar {
        alphabet: vec!['c', 'a', 'b'],
        graph: LoweredProductions(vec![ProductionImpl(vec![CaseImpl(vec![
          CaseEl::Tok(TokRef(0)),
          CaseEl::Tok(TokRef(1)),
          CaseEl::Tok(TokRef(2)),
        ])])]),
      }
    );
  }

  #[test]
  fn token_grammar_construction() {
    let prods = non_cyclic_productions();
    let grammar = TokenGrammar::new(&prods);
    assert_eq!(
      grammar.clone(),
      TokenGrammar {
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
      }
    );
  }

  #[test]
  fn token_grammar_state_indexing() {
    let prods = non_cyclic_productions();
    let grammar = TokenGrammar::new(&prods);
    assert_eq!(
      grammar.index_token_states(),
      [
        (
          'a',
          vec![
            TokenPosition::new(0, 0, 0),
            TokenPosition::new(1, 0, 0),
            TokenPosition::new(1, 1, 1),
          ]
        ),
        (
          'b',
          vec![TokenPosition::new(0, 0, 1), TokenPosition::new(1, 0, 1)],
        ),
      ].iter()
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
    assert_eq!(
      noncyclic_interval_graph,
      EpsilonIntervalGraph(vec![
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(0)),
          EpsilonGraphVertex::State(TokenPosition::new(0, 0, 0)),
          EpsilonGraphVertex::State(TokenPosition::new(0, 0, 1)),
          EpsilonGraphVertex::End(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(1)),
          EpsilonGraphVertex::State(TokenPosition::new(1, 0, 0)),
          EpsilonGraphVertex::State(TokenPosition::new(1, 0, 1)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(0))),
          EpsilonGraphVertex::Start(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(0)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(0))),
          EpsilonGraphVertex::End(ProdRef(1)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(1))),
          EpsilonGraphVertex::Start(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(0)),
          EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(1))),
          EpsilonGraphVertex::State(TokenPosition::new(1, 1, 1)),
          EpsilonGraphVertex::End(ProdRef(1)),
        ]),
      ])
    );

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

  #[test]
  fn terminals_interval_graph_start_end_indices() {
    let noncyclic_prods = non_cyclic_productions();
    let noncyclic_grammar = TokenGrammar::new(&noncyclic_prods);
    let noncyclic_interval_graph =
      PreprocessedGrammar::produce_terminals_interval_graph(&noncyclic_grammar);
    let intervals_by_start_and_end = noncyclic_interval_graph.find_start_end_indices();
    assert_eq!(
      intervals_by_start_and_end,
      vec![
        (
          ProdRef(0),
          StartEndEpsilonIntervals {
            start_epsilons: vec![ContiguousNonterminalInterval(vec![
              EpsilonGraphVertex::Start(ProdRef(0)),
              EpsilonGraphVertex::State(TokenPosition::new(0, 0, 0)),
              EpsilonGraphVertex::State(TokenPosition::new(0, 0, 1)),
              EpsilonGraphVertex::End(ProdRef(0)),
            ])],
            end_epsilons: vec![
              ContiguousNonterminalInterval(vec![
                EpsilonGraphVertex::End(ProdRef(0)),
                EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(0))),
                EpsilonGraphVertex::End(ProdRef(1)),
              ]),
              ContiguousNonterminalInterval(vec![
                EpsilonGraphVertex::End(ProdRef(0)),
                EpsilonGraphVertex::Anon(AnonStep::Negative(AnonSym(1))),
                EpsilonGraphVertex::State(TokenPosition::new(1, 1, 1)),
                EpsilonGraphVertex::End(ProdRef(1)),
              ]),
            ],
          },
        ),
        (
          ProdRef(1),
          StartEndEpsilonIntervals {
            start_epsilons: vec![
              ContiguousNonterminalInterval(vec![
                EpsilonGraphVertex::Start(ProdRef(1)),
                EpsilonGraphVertex::State(TokenPosition::new(1, 0, 0)),
                EpsilonGraphVertex::State(TokenPosition::new(1, 0, 1)),
                EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(0))),
                EpsilonGraphVertex::Start(ProdRef(0)),
              ]),
              ContiguousNonterminalInterval(vec![
                EpsilonGraphVertex::Start(ProdRef(1)),
                EpsilonGraphVertex::Anon(AnonStep::Positive(AnonSym(1))),
                EpsilonGraphVertex::Start(ProdRef(0)),
              ]),
            ],
            end_epsilons: vec![],
          },
        ),
      ].iter()
        .cloned()
        .collect::<IndexMap<ProdRef, StartEndEpsilonIntervals>>()
    );
  }


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
      preprocessed_grammar.clone(),
      PreprocessedGrammar {
        token_states_mapping: vec![
          (
            'a',
            vec![
              TokenPosition::new(0, 0, 0),
              TokenPosition::new(1, 0, 0),
              TokenPosition::new(1, 1, 1),
            ],
          ),
          (
            'b',
            vec![TokenPosition::new(0, 0, 1), TokenPosition::new(1, 0, 1)],
          ),
        ].iter()
          .cloned()
          .collect::<IndexMap<char, Vec<TokenPosition>>>(),
        state_transition_graph: StateTransitionGraph {
          cycles: IndexMap::new(),
          graph: vec![
            (
              StatePair {
                left: LoweredState::Start,
                right: first_a,
              },
              vec![
                StackDiff(vec![ParseTimeStateMachine::Named(StackStep::Positive(
                  a_prod,
                ))]),
                StackDiff(vec![
                  ParseTimeStateMachine::Named(StackStep::Positive(b_prod)),
                  ParseTimeStateMachine::Anon(AnonStep::Positive(AnonSym(1))),
                  ParseTimeStateMachine::Named(StackStep::Positive(a_prod)),
                ]),
              ],
            ),
            (
              StatePair {
                left: LoweredState::Start,
                right: second_a,
              },
              vec![StackDiff(vec![ParseTimeStateMachine::Named(
                StackStep::Positive(b_prod),
              )])],
            ),
            (
              StatePair {
                left: first_a,
                right: first_b,
              },
              vec![StackDiff(vec![])],
            ),
            (
              StatePair {
                left: second_a,
                right: second_b,
              },
              vec![StackDiff(vec![])],
            ),
            (
              StatePair {
                left: first_b,
                right: LoweredState::End,
              },
              vec![
                StackDiff(vec![ParseTimeStateMachine::Named(StackStep::Negative(
                  a_prod,
                ))]),
                StackDiff(vec![
                  ParseTimeStateMachine::Named(StackStep::Negative(a_prod)),
                  ParseTimeStateMachine::Anon(AnonStep::Negative(AnonSym(0))),
                  ParseTimeStateMachine::Named(StackStep::Negative(b_prod)),
                ]),
              ],
            ),
            (
              StatePair {
                left: third_a,
                right: LoweredState::End,
              },
              vec![StackDiff(vec![ParseTimeStateMachine::Named(
                StackStep::Negative(b_prod),
              )])],
            ),
            (
              StatePair {
                left: first_b,
                right: third_a,
              },
              vec![StackDiff(vec![
                ParseTimeStateMachine::Named(StackStep::Negative(a_prod)),
                ParseTimeStateMachine::Anon(AnonStep::Negative(AnonSym(1))),
              ])],
            ),
            (
              StatePair {
                left: second_b,
                right: first_a,
              },
              vec![StackDiff(vec![
                ParseTimeStateMachine::Anon(AnonStep::Positive(AnonSym(0))),
                ParseTimeStateMachine::Named(StackStep::Positive(a_prod)),
              ])],
            ),
          ].iter()
            .cloned()
            .collect::<IndexMap<StatePair, Vec<StackDiff>>>(),
        },
      },
    );
  }

  #[test]
  fn cyclic_transition_graph() {
    let prods = basic_productions();
    let grammar = TokenGrammar::new(&prods);
    let preprocessed_grammar = PreprocessedGrammar::new(&grammar);
    assert_eq!(
      preprocessed_grammar,
      PreprocessedGrammar {
        token_states_mapping: IndexMap::new(),
        state_transition_graph: StateTransitionGraph {
          graph: IndexMap::new(),
          cycles: IndexMap::new(),
        },
      },
    );
  }

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
      )].iter()
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
      ].iter()
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
      ].iter()
        .cloned()
        .collect(),
    )
  }
}
