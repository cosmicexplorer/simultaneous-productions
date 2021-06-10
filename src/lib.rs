/*
 * Description: Implement the Simultaneous Productions general parsing
 * method.
 *
 * Copyright (C) 2019-2021 Danny McClanahan <dmcC2@hypnicjerk.ai>
 * SPDX-License-Identifier: GPL-3.0
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
/* Ensure any doctest warnings fails the doctest! */
#![doc(test(attr(deny(warnings))))]
/* Enable all clippy lints except for many of the pedantic ones. It's a shame this needs to be
 * copied and pasted across crates, but there doesn't appear to be a way to include inner
 * attributes from a common source. */
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
/* It is often more clear to show that nothing is being moved. */
#![allow(clippy::match_ref_pats)]
/* Subjective style. */
#![allow(
  clippy::derive_hash_xor_eq,
  clippy::len_without_is_empty,
  clippy::redundant_field_names,
  clippy::too_many_arguments
)]
/* Default isn't as big a deal as people seem to think it is. */
#![allow(clippy::new_without_default, clippy::new_ret_no_self)]
/* Arc<Mutex> can be more clear than needing to grok Orderings. */
#![allow(clippy::mutex_atomic)]

extern crate indexmap;
extern crate priority_queue;
extern crate typename;

#[macro_use]
pub mod binding;
pub mod grammar_indexing;
pub mod parsing;
pub mod reconstruction;

pub mod token {
  use typename::TypeName;

  use std::{fmt::Debug, hash::Hash};

  /// The constraints required for any token stream parsed by this crate.
  pub trait Token = Debug+PartialEq+Eq+Hash+Copy+Clone+TypeName;
}

pub mod api {
  use super::token::*;

  use indexmap::IndexMap;
  use typename::TypeName;

  #[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
  pub struct Literal<Tok: Token>(pub Vec<Tok>);

  impl From<&str> for Literal<char> {
    fn from(s: &str) -> Self { Self(s.chars().collect()) }
  }

  impl<Tok: Token> From<&[Tok]> for Literal<Tok> {
    fn from(s: &[Tok]) -> Self { Self(s.to_vec()) }
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
      super::{api::*, token::*},
      graph_coordinates::*,
      graph_representation::*,
    };

    use indexmap::{IndexMap, IndexSet};

    use std::collections::HashMap;

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
                    let (tok_ind, _) = all_tokens.insert_full(*cur_tok);
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
      let prologue = (0..prologue_length).flat_map(|_| self.group.clone());

      let epilogue: Box<dyn std::iter::Iterator<Item=CaseEl>> =
        if let Some(upper_bound) = self.upper_bound {
          /* If we have a definite upper bound, make up the difference in length from
           * the initial left side. */
          Box::new((0..(upper_bound - prologue_length)).flat_map(|_| self.group.clone()))
        } else {
          /* If not, we can go forever, or not at all, so we can just apply a Kleene
           * star to this! */
          let starred = KleeneStar {
            group: self.group.clone(),
          };
          let OperatorResult { result } = starred.operate(prods);
          Box::new(result.into_iter())
        };

      let result = prologue.chain(epilogue).collect();
      OperatorResult { result }
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
pub mod tests {
  use super::{
    api::*,
    lowering_to_indices::{graph_coordinates::*, graph_representation::*, mapping_to_tokens::*},
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

  pub fn non_cyclic_productions() -> SimultaneousProductions<char> {
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

  pub fn basic_productions() -> SimultaneousProductions<char> {
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
