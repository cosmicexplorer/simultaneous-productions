/* Copyright (C) 2021 Danny McClanahan <dmcC2@hypnicjerk.ai> */
/* SPDX-License-Identifier: GPL-3.0 */

//! ???
//!
//! (I think this is a "model" graph class of some sort, where the model is
//! this "simultaneous productions" parsing formulation. See Spinrad's book
//! [???]!)
//!
//! Vec<ProductionImpl> = [
//!   Production([
//!     Case([CaseEl(Lit("???")), CaseEl(ProdRef(?)), ...]),
//!     ...,
//!   ]),
//!   ...,
//! ]

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

#[cfg(test)]
mod tests {
  use super::{graph_coordinates::*, graph_representation::*, mapping_to_tokens::*};
  use crate::{api::*, test_framework::non_cyclic_productions};

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
}
