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

use std::{collections::HashMap, hash::Hash};

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
  pub struct StackDiff(pub Vec<StackStep>);

  impl StackDiff {
    fn sequence(&self, other: &Self) -> Self {
      let combined: Vec<StackStep> = self.0.iter().chain(other.0.iter()).cloned().collect();
      StackDiff(combined)
    }
  }

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub enum LoweredState {
    Start,
    End,
    Within(TokenPosition),
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
  pub enum AnonStepVertex {
    Positive(AnonSym),
    Negative(AnonSym),
  }

  /* Fun fact: I'm pretty sure this /is/ actually an interval graph,
   * describing the continuous strings of terminals in a TokenGrammar! */
  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  pub enum EpsilonGraphVertex {
    Start(ProdRef),
    End(ProdRef),
    Anon(AnonStepVertex),
    State(TokenPosition),
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct ContiguousNonterminalInterval(pub Vec<EpsilonGraphVertex>);

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  pub struct EpsilonIntervalGraph(pub Vec<ContiguousNonterminalInterval>);

  // NB: There is no reference to any `TokenGrammar` -- this is intentional, and
  // I believe makes it easier to have the runtime we want just fall out of the
  // code without too much work.
  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct PreprocessedGrammar<Tok: PartialEq+Eq+Hash+Copy+Clone> {
    // These don't need to be quick to access or otherwise optimized for the algorithm until we
    // create a `Parse` -- these are chosen to reduce redundancy.
    // `M: T -> {Q}`, where `{Q}` is sets of states!
    pub token_states_mapping: IndexMap<Tok, Vec<TokenPosition>>,
    // TODO: we don't yet support stack cycles (ignored), or multiple stack paths to the same
    // succeeding state from an initial state (also ignored) -- details in
    // build_pairwise_transitions_table().
    // `A: T x T -> {S}^+_-`, where `{S}^+_-` (LaTeX formatting) is ordered sequences of signed
    // stack symbols!
    pub pairwise_state_transition_table: IndexMap<StatePair, Vec<StackDiff>>,
  }

  impl<Tok: PartialEq+Eq+Hash+Copy+Clone> PreprocessedGrammar<Tok> {
    /* Intended to reduce visual clutter in the implementation of interval
     * production. */
    fn make_pos_neg_anon_steps(cur_index: usize) -> (EpsilonGraphVertex, EpsilonGraphVertex) {
      (
        EpsilonGraphVertex::Anon(AnonStepVertex::Positive(AnonSym(cur_index))),
        EpsilonGraphVertex::Anon(AnonStepVertex::Negative(AnonSym(cur_index))),
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

    fn produce_token_transition_graph(
      interval_graph: &EpsilonIntervalGraph,
    ) -> IndexMap<StatePair, Vec<StackDiff>> {
      panic!("not yet implemented!");
    }

    pub fn new(grammar: &TokenGrammar<Tok>) -> Self {
      let terminals_interval_graph = Self::produce_terminals_interval_graph(&grammar);
      let token_transition_graph = Self::produce_token_transition_graph(&terminals_interval_graph);
      PreprocessedGrammar {
        token_states_mapping: grammar.index_token_states(),
        pairwise_state_transition_table: token_transition_graph,
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
          EpsilonGraphVertex::Anon(AnonStepVertex::Positive(AnonSym(0))),
          EpsilonGraphVertex::Start(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(0)),
          EpsilonGraphVertex::Anon(AnonStepVertex::Negative(AnonSym(0))),
          EpsilonGraphVertex::State(TokenPosition::new(0, 1, 2)),
          EpsilonGraphVertex::End(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(0)),
          EpsilonGraphVertex::State(TokenPosition::new(0, 2, 0)),
          EpsilonGraphVertex::State(TokenPosition::new(0, 2, 1)),
          EpsilonGraphVertex::Anon(AnonStepVertex::Positive(AnonSym(1))),
          EpsilonGraphVertex::Start(ProdRef(1)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStepVertex::Negative(AnonSym(1))),
          EpsilonGraphVertex::End(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStepVertex::Positive(AnonSym(2))),
          EpsilonGraphVertex::Start(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(0)),
          EpsilonGraphVertex::Anon(AnonStepVertex::Negative(AnonSym(2))),
          EpsilonGraphVertex::End(ProdRef(1)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStepVertex::Positive(AnonSym(3))),
          EpsilonGraphVertex::Start(ProdRef(1)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStepVertex::Negative(AnonSym(3))),
          EpsilonGraphVertex::End(ProdRef(1)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::Start(ProdRef(1)),
          EpsilonGraphVertex::Anon(AnonStepVertex::Positive(AnonSym(4))),
          EpsilonGraphVertex::Start(ProdRef(0)),
        ]),
        ContiguousNonterminalInterval(vec![
          EpsilonGraphVertex::End(ProdRef(0)),
          EpsilonGraphVertex::Anon(AnonStepVertex::Negative(AnonSym(4))),
          EpsilonGraphVertex::State(TokenPosition::new(1, 2, 1)),
          EpsilonGraphVertex::State(TokenPosition::new(1, 2, 2)),
          EpsilonGraphVertex::End(ProdRef(1)),
        ]),
      ])
    );
  }


  #[test]
  fn noncyclic_token_transition_graph() {
    let prods = non_cyclic_productions();
    let grammar = TokenGrammar::new(&prods);
    let preprocessed_grammar = PreprocessedGrammar::new(&grammar);
    let first_a = LoweredState::Within(TokenPosition {
      prod: ProdRef(0),
      case: CaseRef(0),
      case_el: CaseElRef(0),
    });
    let first_b = LoweredState::Within(TokenPosition {
      prod: ProdRef(0),
      case: CaseRef(0),
      case_el: CaseElRef(1),
    });
    let second_a = LoweredState::Within(TokenPosition {
      prod: ProdRef(1),
      case: CaseRef(0),
      case_el: CaseElRef(0),
    });
    let second_b = LoweredState::Within(TokenPosition {
      prod: ProdRef(1),
      case: CaseRef(0),
      case_el: CaseElRef(1),
    });
    let third_a = LoweredState::Within(TokenPosition {
      prod: ProdRef(1),
      case: CaseRef(1),
      case_el: CaseElRef(1),
    });
    let a_prod = StackSym(ProdRef(0));
    let b_prod = StackSym(ProdRef(1));
    assert_eq!(
      preprocessed_grammar.clone(),
      PreprocessedGrammar {
        token_states_mapping: vec![
          (
            'a',
            vec![
              TokenPosition {
                prod: ProdRef(0),
                case: CaseRef(0),
                case_el: CaseElRef(0),
              },
              TokenPosition {
                prod: ProdRef(1),
                case: CaseRef(0),
                case_el: CaseElRef(0),
              },
            ],
          ),
          (
            'b',
            vec![
              TokenPosition {
                prod: ProdRef(0),
                case: CaseRef(0),
                case_el: CaseElRef(1),
              },
              TokenPosition {
                prod: ProdRef(1),
                case: CaseRef(0),
                case_el: CaseElRef(1),
              },
            ],
          ),
        ].iter()
          .cloned()
          .collect::<IndexMap<char, Vec<TokenPosition>>>(),
        pairwise_state_transition_table: vec![
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
              StackDiff(vec![StackStep::Negative(a_prod)]),
              // TODO: this is currently missing! this happens because a prod ref to
              // "a" is at the
              // end of the single case of the "b" production -- we can recognize
              // this case in index_tokens() (ugh) and propagate it (probably not
              // that hard, could be done by adding an "end" case to the
              // `GrammarVertex` enum!)!
              StackDiff(vec![
                StackStep::Negative(a_prod),
                StackStep::Negative(b_prod),
              ]),
            ],
          ),
          (
            StatePair {
              left: third_a,
              right: LoweredState::End,
            },
            vec![StackDiff(vec![StackStep::Negative(b_prod)])],
          ),
          (
            StatePair {
              left: first_b,
              right: first_a,
            },
            vec![StackDiff(vec![
              StackStep::Negative(a_prod),
              StackStep::Positive(a_prod),
            ])],
          ),
          (
            StatePair {
              left: first_b,
              right: second_a,
            },
            vec![StackDiff(vec![
              StackStep::Negative(a_prod),
              StackStep::Positive(b_prod),
            ])],
          ),
          (
            StatePair {
              left: first_b,
              right: third_a,
            },
            vec![StackDiff(vec![StackStep::Negative(a_prod)])],
          ),
          (
            StatePair {
              left: second_b,
              right: first_a,
            },
            vec![StackDiff(vec![StackStep::Positive(a_prod)])],
          ),
          (
            StatePair {
              left: LoweredState::Start,
              right: first_a,
            },
            vec![
              StackDiff(vec![StackStep::Positive(a_prod)]),
              StackDiff(vec![
                StackStep::Positive(b_prod),
                StackStep::Positive(a_prod),
              ]),
            ],
          ),
          (
            StatePair {
              left: LoweredState::Start,
              right: second_a,
            },
            vec![StackDiff(vec![StackStep::Positive(b_prod)])],
          ),
        ].iter()
          .cloned()
          .collect::<IndexMap<StatePair, Vec<StackDiff>>>(),
      }
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
