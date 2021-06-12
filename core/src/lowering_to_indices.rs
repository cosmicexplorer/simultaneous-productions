/* Copyright (C) 2021 Danny McClanahan <dmcC2@hypnicjerk.ai> */
/* SPDX-License-Identifier: GPL-3.0 */

//! Map locations of [Token]s and [ProductionReference]s.
//!
//! This phase mainly consists of calling [TokenGrammar::new].
//!
//! *(I think this is a "model" graph class of some sort, where the model is
//! this "simultaneous productions" parsing formulation. See Spinrad's book
//! [???]!)*
//!
//!     Vec<ProductionImpl> = [
//!       Production([
//!         Case([CaseEl(Lit("???")), CaseEl(ProdRef::new(?)), ...]),
//!         ...,
//!       ]),
//!       ...,
//!     ]

#[cfg(doc)]
use crate::{grammar_specification::ProductionReference, token::Token};
#[cfg(doc)]
use grammar_building::TokenGrammar;

/// ???
///
/// All these `Ref` types have nice properties, like being storeable without
/// reference to any particular graph, being totally ordered, and being able
/// to be incremented.
///
/// We adopt the convention of abbreviated names for things used in
/// algorithms.
pub mod graph_coordinates {
  /// Points to a particular Production within a sequence of [Production].
  ///
  /// A version of [ProductionReference] which uses a [usize] for speed.
  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub struct ProdRef(pub usize);

  impl From<usize> for ProdRef {
    fn from(value: usize) -> Self { Self(value) }
  }

  impl Into<usize> for ProdRef {
    fn into(self) -> usize { self.0 }
  }

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub struct CaseRef(pub usize);

  impl From<usize> for CaseRef {
    fn from(value: usize) -> Self { Self(value) }
  }

  impl Into<usize> for CaseRef {
    fn into(self) -> usize { self.0 }
  }

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub struct CaseElRef(pub usize);

  impl From<usize> for CaseElRef {
    fn from(value: usize) -> Self { Self(value) }
  }

  impl Into<usize> for CaseElRef {
    fn into(self) -> usize { self.0 }
  }

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub struct TokenPosition {
    pub prod: ProdRef,
    pub case: CaseRef,
    pub el: CaseElRef,
  }

  /// Points to a particular token value within an alphabet.
  ///
  /// Differs from [TokenPosition], which points to an individual *state* in
  /// the graph (which may be satisfied by exactly one token *value*).
  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub struct TokRef(pub usize);

  impl From<usize> for TokRef {
    fn from(value: usize) -> Self { Self(value) }
  }

  impl Into<usize> for TokRef {
    fn into(self) -> usize { self.0 }
  }

  #[derive(Debug, Copy, Clone)]
  pub enum CaseEl {
    Tok(TokRef),
    Prod(ProdRef),
  }
}

pub mod grammar_building {
  use super::graph_coordinates as gc;
  #[cfg(doc)]
  use super::graph_coordinates::{ProdRef, TokenPosition};
  #[cfg(doc)]
  use crate::grammar_specification::{Literal, ProductionReference};
  use crate::{grammar_specification as gs, interns::InternArena, vec::Vec};

  /* use heapless::{FnvIndexMap, FnvIndexSet, IndexMap, IndexSet, Vec}; */
  use indexmap::IndexMap;
  use twox_hash::XxHash64;

  use core::{
    alloc::Allocator,
    fmt,
    hash::{BuildHasherDefault, Hash},
  };

  #[derive(Debug)]
  pub struct Case<Arena>(pub Vec<gc::CaseEl, Arena>)
  where Arena: Allocator;

  #[derive(Debug)]
  pub struct Production<Arena>(pub Vec<Case<Arena>, Arena>)
  where Arena: Allocator;

  #[derive(Debug)]
  pub struct DetokenizedProductions<Arena>(
    IndexMap<gc::ProdRef, Production<Arena>, Arena, BuildHasherDefault<XxHash64>>,
  )
  where Arena: Allocator+Clone;

  impl<Arena> DetokenizedProductions<Arena>
  where Arena: Allocator+Clone
  {
    pub fn new_in(arena: Arena) -> Self { Self(IndexMap::new_in(arena)) }

    pub fn insert_new_production(&mut self, entry: (gc::ProdRef, Production<Arena>)) {
      let (key, value) = entry;
      match self.0.insert_full(key) {
        (_, Some(_)) => unreachable!("expected all productions to have unique IDs"),
        (_, None) => (),
      }
    }
  }

  /// An alphabet of tokens for a grammar.
  #[derive(Debug)]
  pub struct Alphabet<Tok, Arena>(pub InternArena<Tok, gc::TokRef, Arena>)
  where Arena: Allocator;

  #[derive(Debug)]
  pub struct AlphabetMapping<Arena>(
    IndexMap<gc::TokRef, Vec<gc::TokenPosition, Arena>, Arena, BuildHasherDefault<XxHash64>>,
  )
  where Arena: Allocator+Clone;

  impl<Arena> AlphabetMapping<Arena>
  where Arena: Allocator+Clone
  {
    pub fn new_in(arena: Arena) -> Self { Self(IndexMap::new_in(arena)) }

    pub fn insert_new_position(&mut self, entry: (gc::TokRef, gc::TokenPosition)) {
      let (key, new_value) = entry;
      let arena = self.0.arena();
      let entry = self.0.entry(key).or_insert_with(|| Vec::new_in(arena));
      (*entry).push(new_value);
    }
  }

  pub enum GrammarConstructionError<ID> {
    DuplicateProductionId(ID),
    UnrecognizedProdRefId(ID),
  }

  impl<ID> fmt::Debug for GrammarConstructionError<ID>
  where ID: fmt::Debug
  {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
      match self {
        Self::DuplicateProductionId(id) => {
          write!(
            f,
            "GrammarConstructionError::DuplicateProductionId({:?})",
            id
          )
        },
      }
    }
  }

  #[derive(Debug)]
  pub struct TokenGrammar<Tok, Arena>
  where Arena: Allocator+Clone
  {
    graph: DetokenizedProductions<Arena>,
    alphabet: Alphabet<Tok, Arena>,
    token_states: AlphabetMapping<Arena>,
  }

  impl<Tok, Arena> TokenGrammar<Tok, Arena>
  where
    Tok: Hash+Eq,
    Arena: Allocator+Clone,
  {
    /// Walk productions and split literal strings.
    ///
    /// This method does two things:
    /// 1. Flatten out [Literal]s into individual tokens, and store a mapping of
    /// all    the [locations][TokenPosition] each token is located at.
    /// 2. Match up [ProductionReference]s to [ProdRef]s, or error out.
    pub fn new<ID, Lit, PR, C, P, SP>(
      sp: SP,
      arena: Arena,
    ) -> Result<Self, GrammarConstructionError<ID>>
    where
      ID: Hash+Eq+Clone,
      Lit: gs::Literal<Tok>,
      PR: gs::ProductionReference<ID>,
      C: gs::Case<ID, Tok, Lit, PR>,
      P: gs::Production<ID, Tok, PR, C>,
      SP: gs::SimultaneousProductions<ID, Tok, Lit, PR, C, P>,
    {
      let (all_prods, id_prod_mapping) = {
        let mut all_prods: InternArena<P, gc::ProdRef, Arena> = InternArena::new(arena.clone());
        let mut id_prod_mapping: IndexMap<ID, gc::ProdRef, Arena, BuildHasherDefault<XxHash64>> =
          IndexMap::new_in(arena.clone());
        for (prod_ref, prod) in sp.into_iter() {
          let intern_token = all_prods.intern_always_new_increasing(prod);
          let id: ID = prod_ref.into();
          if let Some(_) = id_prod_mapping.insert(id.clone(), intern_token) {
            return Err(GrammarConstructionError::DuplicateProductionId(id));
          }
        }
        (all_prods.into_vec(), id_prod_mapping)
      };

      // Collect all the tokens (splitting up literals) as we traverse the
      // productions. So literal strings are "flattened" into their individual
      // tokens.
      let mut alphabet: InternArena<Tok, gc::TokRef, Arena> = InternArena::new(arena.clone());
      let mut token_states: AlphabetMapping<Arena> = AlphabetMapping::new_in(arena.clone());
      let mut ret_prods: DetokenizedProductions<Arena> =
        DetokenizedProductions::new_in(arena.clone());

      for (prod_ref, prod) in all_prods.into_iter() {
        let mut ret_cases: Vec<Case<Arena>, Arena> = Vec::new_in(arena.clone());
        for (case_ind, case) in prod.into_iter().enumerate() {
          let case_ref: gc::CaseRef = case_ind.into();
          let mut ret_els: Vec<gc::CaseEl, Arena> = Vec::new_in(arena.clone());
          /* We want to track the positions of each token within each literal as well,
           * so we can't directly use .enumerate() */
          let mut case_el_ind: usize = 0;
          for el in case.into_iter() {
            match el {
              gs::CaseElement::Lit(lit) => {
                for tok in lit.into_iter() {
                  let tok_ref = alphabet.intern_exclusive(tok);

                  ret_els.push(gc::CaseEl::Tok(tok_ref));

                  let el_ref: gc::CaseElRef = case_el_ind.into();
                  let cur_pos = gc::TokenPosition {
                    prod: prod_ref,
                    case: case_ref,
                    el: el_ref,
                  };
                  token_states.insert_new_position((tok_ref, cur_pos));

                  case_el_ind += 1;
                }
              },
              gs::CaseElement::Prod(prod_ref) => {
                let id: ID = prod_ref.into();
                let pr: gc::ProdRef = match id_prod_mapping.get(&id) {
                  Some(pr) => pr,
                  None => {
                    return Err(GrammarConstructionError::UnrecognizedProdRefId(id));
                  },
                };

                ret_els.push(gc::CaseEl::Prod(pr));

                case_el_ind += 1;
              },
            }
          }
          ret_cases.push(Case(ret_els));
        }
        ret_prods.insert_new_production((prod_ref, Production(ret_cases)));
      }

      Ok(Self {
        graph: ret_prods,
        alphabet: Alphabet(alphabet),
        token_states,
      })
    }
  }
}

#[cfg(test)]
mod tests {
  use super::{grammar_building::*, graph_coordinates::*};
  use crate::{grammar_specification::*, test_framework::non_cyclic_productions};

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
      graph: DetokenizedProductions(vec![ProductionImpl(vec![CaseImpl(vec![
        CaseEl::Tok(TokRef::new(0)),
        CaseEl::Tok(TokRef::new(1)),
        CaseEl::Tok(TokRef::new(2)),
      ])])]),
    });
  }

  #[test]
  fn token_grammar_construction() {
    let prods = non_cyclic_productions();
    let grammar = TokenGrammar::new(&prods);
    assert_eq!(grammar, TokenGrammar {
      alphabet: vec!['a', 'b'],
      graph: DetokenizedProductions(vec![
        ProductionImpl(vec![CaseImpl(vec![
          CaseEl::Tok(TokRef::new(0)),
          CaseEl::Tok(TokRef::new(1)),
        ])]),
        ProductionImpl(vec![
          CaseImpl(vec![
            CaseEl::Tok(TokRef::new(0)),
            CaseEl::Tok(TokRef::new(1)),
            CaseEl::Prod(ProdRef::new(0)),
          ]),
          CaseImpl(vec![
            CaseEl::Prod(ProdRef::new(0)),
            CaseEl::Tok(TokRef::new(0))
          ]),
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
