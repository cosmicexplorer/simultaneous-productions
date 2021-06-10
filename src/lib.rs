/*
    Description: Implement the Simultaneous Productions general parsing method.
    Copyright (C) 2019-2021 Danny McClanahan <dmcC2@hypnicjerk.ai>
    SPDX-License-Identifier: GPL-3.0

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

#[macro_use]
pub mod binding;
pub mod grammar_indexing;
pub mod parsing;
pub mod reconstruction;

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
use typename::TypeName;

/* use frunk::hlist::*; */
use std::collections::HashMap;

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

  /// A reference to another production within the same set.
  ///
  /// The string must match the assigned name of a [Production] in a set of
  /// [SimultaneousProductions].
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
  /// ???
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
    /// This refers to a specific token within the graph, implying that we must
    /// be pointing to a particular index of a particular [Literal].
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

    /// Points to a particular token value within an alphabet.
    ///
    /// Differs from [TokenPosition], which points to an individual *state* in
    /// the graph (which may be satisfied by exactly one token *value*).
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct TokRef(pub usize);
  }

  /// ???
  pub mod graph_representation {
    use super::graph_coordinates::*;

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

  /// ???
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

  use std::{collections::VecDeque, rc::Rc};

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
