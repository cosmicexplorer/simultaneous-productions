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
//! Vec<ProductionImpl> = [
//!   Production([
//!     Case([CaseEl(Lit("???")), CaseEl(ProdRef(?)), ...]),
//!     ...,
//!   ]),
//!   ...,
//! ]

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

  impl From<ProdRef> for usize {
    fn from(value: ProdRef) -> Self { value.0 }
  }

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub struct CaseRef(pub usize);

  impl From<usize> for CaseRef {
    fn from(value: usize) -> Self { Self(value) }
  }

  impl From<CaseRef> for usize {
    fn from(value: CaseRef) -> Self { value.0 }
  }

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub struct CaseElRef(pub usize);

  impl From<usize> for CaseElRef {
    fn from(value: usize) -> Self { Self(value) }
  }

  impl From<CaseElRef> for usize {
    fn from(value: CaseElRef) -> Self { value.0 }
  }

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub struct ProdCaseRef {
    pub prod: ProdRef,
    pub case: CaseRef,
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

  impl From<TokRef> for usize {
    fn from(value: TokRef) -> Self { value.0 }
  }

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub enum CaseEl {
    Tok(TokRef),
    Prod(ProdRef),
  }
}

pub mod grammar_building {
  pub use containers::*;
  pub use result::*;

  pub(super) mod containers {
    use super::super::graph_coordinates as gc;
    use crate::{
      allocation::HandoffAllocable,
      interns::InternArena,
      types::{DefaultHasher, Vec},
    };

    /* use heapless::{FnvIndexMap, FnvIndexSet, IndexMap, IndexSet, Vec}; */
    use indexmap::IndexMap;

    use core::alloc::Allocator;

    #[derive(Debug)]
    pub struct Case<Arena>(pub Vec<gc::CaseEl, Arena>)
    where Arena: Allocator;


    impl<Arena> PartialEq for Case<Arena>
    where Arena: Allocator
    {
      fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
    }

    impl<Arena> Eq for Case<Arena> where Arena: Allocator {}

    impl<Arena> Clone for Case<Arena>
    where Arena: Allocator+Clone
    {
      fn clone(&self) -> Self { Self(self.0.clone()) }
    }

    #[derive(Debug)]
    pub struct Production<Arena>(pub Vec<Case<Arena>, Arena>)
    where Arena: Allocator;

    impl<Arena> PartialEq for Production<Arena>
    where Arena: Allocator
    {
      fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
    }

    impl<Arena> Eq for Production<Arena> where Arena: Allocator {}

    impl<Arena> Clone for Production<Arena>
    where Arena: Allocator+Clone
    {
      fn clone(&self) -> Self { Self(self.0.clone()) }
    }

    #[derive(Debug)]
    pub struct DetokenizedProductions<Arena>(
      IndexMap<gc::ProdRef, Production<Arena>, Arena, DefaultHasher>,
    )
    where Arena: Allocator+Clone;

    impl<Arena> PartialEq for DetokenizedProductions<Arena>
    where Arena: Allocator+Clone
    {
      fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
    }

    impl<Arena> Eq for DetokenizedProductions<Arena> where Arena: Allocator+Clone {}

    impl<Arena> Clone for DetokenizedProductions<Arena>
    where Arena: Allocator+Clone
    {
      fn clone(&self) -> Self { Self(self.0.clone()) }
    }

    impl<Arena> DetokenizedProductions<Arena>
    where Arena: Allocator+Clone
    {
      pub fn new_in(arena: Arena) -> Self { Self(IndexMap::new_in(arena)) }

      pub fn insert_new_production(&mut self, entry: (gc::ProdRef, Production<Arena>)) {
        let (key, value) = entry;
        match self.0.insert_full(key, value) {
          (_, Some(_)) => unreachable!("expected all productions to have unique IDs"),
          (_, None) => (),
        }
      }

      #[allow(dead_code)]
      pub fn into_index_map(
        self,
      ) -> IndexMap<gc::ProdRef, Production<Arena>, Arena, DefaultHasher> {
        self.0
      }
    }

    /// An alphabet of tokens for a grammar.
    #[derive(Debug, Clone)]
    pub struct Alphabet<Tok, Arena>(pub InternArena<Tok, gc::TokRef, Arena>)
    where Arena: Allocator;

    impl<Tok, Arena> HandoffAllocable for Alphabet<Tok, Arena>
    where Arena: Allocator+Clone
    {
      type Arena = Arena;

      fn allocator_handoff(&self) -> Arena { self.0.allocator_handoff() }
    }

    impl<Tok, Arena> PartialEq for Alphabet<Tok, Arena>
    where
      Tok: Eq,
      Arena: Allocator,
    {
      fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
    }

    impl<Tok, Arena> Eq for Alphabet<Tok, Arena>
    where
      Tok: Eq,
      Arena: Allocator,
    {
    }

    #[derive(Debug)]
    pub struct AlphabetMapping<Arena>(
      IndexMap<gc::TokRef, Vec<gc::TokenPosition, Arena>, Arena, DefaultHasher>,
    )
    where Arena: Allocator+Clone;

    impl<Arena> HandoffAllocable for AlphabetMapping<Arena>
    where Arena: Allocator+Clone
    {
      type Arena = Arena;

      fn allocator_handoff(&self) -> Arena { self.0.arena() }
    }

    impl<Arena> PartialEq for AlphabetMapping<Arena>
    where Arena: Allocator+Clone
    {
      fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
    }

    impl<Arena> Eq for AlphabetMapping<Arena> where Arena: Allocator+Clone {}

    impl<Arena> Clone for AlphabetMapping<Arena>
    where Arena: Allocator+Clone
    {
      fn clone(&self) -> Self { Self(self.0.clone()) }
    }

    impl<Arena> AlphabetMapping<Arena>
    where Arena: Allocator+Clone
    {
      pub fn new_in(arena: Arena) -> Self { Self(IndexMap::new_in(arena)) }

      pub fn insert_new_position(&mut self, entry: (gc::TokRef, gc::TokenPosition)) {
        let (key, new_value) = entry;
        let arena = self.allocator_handoff();
        let entry = self.0.entry(key).or_insert_with(|| Vec::new_in(arena));
        (*entry).push(new_value);
      }

      pub fn get(&self, tok_ref: gc::TokRef) -> Option<&[gc::TokenPosition]> {
        self.0.get(&tok_ref).map(|ps| ps.as_ref())
      }

      #[allow(dead_code)]
      pub fn into_index_map(
        self,
      ) -> IndexMap<gc::TokRef, Vec<gc::TokenPosition, Arena>, Arena, DefaultHasher> {
        self.0
      }
    }
  }

  pub(super) mod result {
    use super::{super::graph_coordinates as gc, containers::*};
    use crate::{
      allocation::HandoffAllocable,
      grammar_specification as gs,
      interns::InternArena,
      types::{DefaultHasher, Vec},
    };

    #[cfg(doc)]
    use super::super::graph_coordinates::{ProdRef, TokenPosition};
    #[cfg(doc)]
    use crate::grammar_specification::{Literal, ProductionReference};

    use indexmap::IndexMap;

    use core::{
      alloc::Allocator,
      fmt,
      hash::Hash,
      iter::{IntoIterator, Iterator},
    };

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
          Self::UnrecognizedProdRefId(id) => {
            write!(
              f,
              "GrammarConstructionError::UnrecognizedProdRefId({:?})",
              id
            )
          },
        }
      }
    }

    impl<ID> PartialEq for GrammarConstructionError<ID>
    where ID: Eq
    {
      fn eq(&self, other: &Self) -> bool {
        match (self, other) {
          (Self::DuplicateProductionId(id1), Self::DuplicateProductionId(id2)) if id1 == id2 => {
            true
          },
          (Self::UnrecognizedProdRefId(id1), Self::UnrecognizedProdRefId(id2)) if id1 == id2 => {
            true
          },
          _ => false,
        }
      }
    }

    impl<ID> Eq for GrammarConstructionError<ID> where ID: Eq {}

    #[derive(Debug)]
    pub struct TokenGrammar<Tok, Arena>
    where Arena: Allocator+Clone
    {
      pub graph: DetokenizedProductions<Arena>,
      pub alphabet: Alphabet<Tok, Arena>,
      pub token_states: AlphabetMapping<Arena>,
    }

    impl<Tok, Arena> PartialEq for TokenGrammar<Tok, Arena>
    where
      Tok: Eq,
      Arena: Allocator+Clone,
    {
      fn eq(&self, other: &Self) -> bool {
        self.graph == other.graph
          && self.alphabet == other.alphabet
          && self.token_states == other.token_states
      }
    }

    impl<Tok, Arena> Eq for TokenGrammar<Tok, Arena>
    where
      Tok: Eq,
      Arena: Allocator+Clone,
    {
    }

    impl<Tok, Arena> HandoffAllocable for TokenGrammar<Tok, Arena>
    where Arena: Allocator+Clone
    {
      type Arena = Arena;

      fn allocator_handoff(&self) -> Arena { self.alphabet.allocator_handoff() }
    }

    impl<Tok, Arena> Clone for TokenGrammar<Tok, Arena>
    where
      Tok: Clone,
      Arena: Allocator+Clone,
    {
      fn clone(&self) -> Self {
        Self {
          graph: self.graph.clone(),
          alphabet: self.alphabet.clone(),
          token_states: self.token_states.clone(),
        }
      }
    }

    impl<Tok, Arena> TokenGrammar<Tok, Arena>
    where
      Tok: Hash+Eq,
      Arena: Allocator+Clone,
    {
      /// Walk productions and split literal strings.
      ///
      /// This method does two things:
      /// 1. Flatten out [Literal]s into individual tokens, and store a mapping
      /// of all    the [locations][TokenPosition] each token is located
      /// at. 2. Match up [ProductionReference]s to [ProdRef]s, or error
      /// out.
      #[allow(dead_code)]
      pub fn new<ID, PR, C, P, SP, Lit>(
        sp: SP,
        arena: Arena,
      ) -> Result<Self, GrammarConstructionError<ID>>
      where
        Lit: gs::Literal<Tok=Tok>+IntoIterator<Item=Tok>,
        ID: Hash+Eq+Clone,
        PR: gs::ProductionReference<ID=ID>,
        C: gs::Case<PR=PR>+IntoIterator<Item=gs::CaseElement<Lit, PR>>,
        P: gs::Production<C=C>+IntoIterator<Item=C>,
        SP: gs::SimultaneousProductions<P=P>+IntoIterator<Item=(PR, P)>,
      {
        let (all_prods, id_prod_mapping) = {
          let mut all_prods: InternArena<P, gc::ProdRef, Arena> = InternArena::new(arena.clone());
          let mut id_prod_mapping: IndexMap<ID, gc::ProdRef, Arena, DefaultHasher> =
            IndexMap::new_in(arena.clone());
          for (prod_ref, prod) in sp.into_iter() {
            let intern_token = all_prods.intern_always_new_increasing(prod);
            let id: PR::ID = prod_ref.into();
            if id_prod_mapping.insert(id.clone(), intern_token).is_some() {
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
                    Some(pr) => *pr,
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
}

#[cfg(test)]
mod tests {
  use super::{grammar_building as gb, graph_coordinates as gc};
  use crate::{
    interns::InternArena,
    test_framework::*,
    types::{Global, Vec},
  };

  #[test]
  fn token_grammar_unsorted_alphabet() {
    let prods = SP::from(
      [(
        ProductionReference::from("xxx"),
        Production::from([Case::from([CE::Lit(Lit::from("cabc"))].as_ref())].as_ref()),
      )]
      .as_ref(),
    );
    let grammar = gb::TokenGrammar::new(prods, Global).unwrap();
    let mut dt = gb::DetokenizedProductions::new_in(Global);
    dt.insert_new_production((
      gc::ProdRef(0),
      gb::Production(
        [gb::Case(
          [
            gc::CaseEl::Tok(gc::TokRef(0)),
            gc::CaseEl::Tok(gc::TokRef(1)),
            gc::CaseEl::Tok(gc::TokRef(2)),
            gc::CaseEl::Tok(gc::TokRef(0)),
          ]
          .as_ref()
          .to_vec(),
        )]
        .as_ref()
        .to_vec(),
      ),
    ));
    let mut ts = gb::AlphabetMapping::new_in(Global);
    ts.insert_new_position((gc::TokRef(0), new_token_position(0, 0, 0)));
    ts.insert_new_position((gc::TokRef(1), new_token_position(0, 0, 1)));
    ts.insert_new_position((gc::TokRef(2), new_token_position(0, 0, 2)));
    ts.insert_new_position((gc::TokRef(0), new_token_position(0, 0, 3)));
    assert_eq!(grammar, gb::TokenGrammar {
      alphabet: gb::Alphabet(InternArena::from(
        ['c', 'a', 'b'].iter().cloned().collect::<Vec<_>>()
      )),
      graph: dt,
      token_states: ts
    });
  }

  #[test]
  fn token_grammar_construction() {
    let prods = non_cyclic_productions();
    let grammar = gb::TokenGrammar::new(prods, Global).unwrap();
    let mut dt = gb::DetokenizedProductions::new_in(Global);
    dt.insert_new_production((
      gc::ProdRef(0),
      gb::Production(
        [gb::Case(
          [
            gc::CaseEl::Tok(gc::TokRef(0)),
            gc::CaseEl::Tok(gc::TokRef(1)),
          ]
          .as_ref()
          .to_vec(),
        )]
        .as_ref()
        .to_vec(),
      ),
    ));
    dt.insert_new_production((
      gc::ProdRef(1),
      gb::Production(
        [
          gb::Case(
            [
              gc::CaseEl::Tok(gc::TokRef(0)),
              gc::CaseEl::Tok(gc::TokRef(1)),
              gc::CaseEl::Prod(gc::ProdRef(0)),
            ]
            .as_ref()
            .to_vec(),
          ),
          gb::Case(
            [
              gc::CaseEl::Prod(gc::ProdRef(0)),
              gc::CaseEl::Tok(gc::TokRef(0)),
            ]
            .as_ref()
            .to_vec(),
          ),
        ]
        .as_ref()
        .to_vec(),
      ),
    ));
    let mut ts = gb::AlphabetMapping::new_in(Global);
    ts.insert_new_position((gc::TokRef(0), new_token_position(0, 0, 0)));
    ts.insert_new_position((gc::TokRef(1), new_token_position(0, 0, 1)));
    ts.insert_new_position((gc::TokRef(0), new_token_position(1, 0, 0)));
    ts.insert_new_position((gc::TokRef(1), new_token_position(1, 0, 1)));
    ts.insert_new_position((gc::TokRef(0), new_token_position(1, 1, 1)));
    assert_eq!(grammar, gb::TokenGrammar {
      alphabet: gb::Alphabet(InternArena::from(
        ['a', 'b'].iter().cloned().collect::<Vec<_>>()
      )),
      graph: dt,
      token_states: ts,
    });
  }

  #[test]
  fn missing_prod_ref() {
    let prods = SP::from(
      [(
        ProductionReference::from("b"),
        Production::from(
          [Case::from(
            [
              CE::Lit(Lit::from("ab")),
              CE::Prod(ProductionReference::from("c")),
            ]
            .as_ref(),
          )]
          .as_ref(),
        ),
      )]
      .as_ref(),
    );
    let grammar: Result<gb::TokenGrammar<char, Global>, _> = gb::TokenGrammar::new(prods, Global);
    assert_eq!(
      grammar,
      Err(gb::GrammarConstructionError::UnrecognizedProdRefId(
        ProductionReference::from("c")
      ))
    );
  }
}
