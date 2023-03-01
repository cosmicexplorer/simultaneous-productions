/*
 * Description: Map locations of tokens and production references.
 *
 * Copyright (C) 2021-2023 Danny McClanahan <dmcC2@hypnicjerk.ai>
 * SPDX-License-Identifier: AGPL-3.0
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

//! Map locations of [tok][Literal::Tok]ens and [ProductionReference]s.
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
use crate::grammar_specification::{Literal, ProductionReference};
#[cfg(doc)]
use grammar_building::TokenGrammar;

/// Specification for internal graph model of the grammar.
///
/// All these `Ref` types have nice properties, like being storeable without
/// reference to any particular graph, being totally ordered, and being able
/// to be incremented.
///
/// We adopt the convention of abbreviated names for things used in
/// algorithms.
pub mod graph_coordinates {
  #[cfg(doc)]
  use crate::grammar_specification as gs;

  macro_rules! via_primitive {
    ($type_name:ident, $primitive:ident) => {
      /* #[doc = $doc] */
      #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
      pub struct $type_name(pub $primitive);

      impl From<$primitive> for $type_name {
        fn from(value: $primitive) -> Self { Self(value) }
      }

      impl From<$type_name> for $primitive {
        fn from(value: $type_name) -> Self { value.0 }
      }
    };
  }

  /* FIXME: make doc comment apply to the macro expansion!! */
  /// Points to a particular Production within a sequence of
  /// [gs::synthesis::Production]s.
  ///
  /// A version of [gs::synthesis::ProductionReference] which uses a [usize] for
  /// speed.
  via_primitive![ProdRef, usize];

  via_primitive![CaseRef, usize];

  via_primitive![CaseElRef, usize];

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

  /* FIXME: make doc comment apply to the macro expansion!! */
  /// Points to a particular token value within an alphabet.
  ///
  /// Differs from [TokenPosition], which points to an individual *state* in
  /// the graph (which may be satisfied by exactly one token *value*).
  via_primitive![TokRef, usize];

  via_primitive![SMRef, usize];

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub enum CaseEl {
    Tok(TokRef),
    Prod(ProdRef),
    SM(SMRef),
  }
}

pub mod grammar_building {
  pub use containers::*;
  pub use result::*;

  pub(super) mod containers {
    use super::super::graph_coordinates as gc;
    use crate::interns::InternArena;

    use core::hash::Hash;

    /* use heapless::{FnvIndexMap, FnvIndexSet, IndexMap, IndexSet, Vec}; */
    use indexmap::IndexMap;

    macro_rules! vec_type {
      ($type_name:ident, $collection_type:ty) => {
        #[derive(Debug)]
        pub struct $type_name(pub $collection_type);

        impl PartialEq for $type_name {
          fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
        }

        impl Eq for $type_name {}

        impl Clone for $type_name {
          fn clone(&self) -> Self { Self(self.0.clone()) }
        }
      };
    }

    vec_type![Case, Vec<gc::CaseEl>];

    vec_type![Production, Vec<Case>];

    macro_rules! indexmap_type {
      ($type_name:ident, $collection_type:ty) => {
        #[derive(Debug)]
        pub struct $type_name(pub $collection_type);

        impl PartialEq for $type_name {
          fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
        }

        impl Eq for $type_name {}

        impl Clone for $type_name {
          fn clone(&self) -> Self { Self(self.0.clone()) }
        }
      };
    }

    indexmap_type![DetokenizedProductions,
                   IndexMap<gc::ProdRef, Production>];

    impl DetokenizedProductions {
      pub fn new() -> Self { Self(IndexMap::new()) }

      pub fn insert_new_production(&mut self, entry: (gc::ProdRef, Production)) {
        let (key, value) = entry;
        match self.0.insert_full(key, value) {
          (_, Some(_)) => unreachable!("expected all productions to have unique IDs"),
          (_, None) => (),
        }
      }

      pub fn into_index_map(self) -> IndexMap<gc::ProdRef, Production> { self.0 }
    }

    /// Merge an intern table with a mapping of locations in the input where the
    /// interned object is found.
    #[derive(Debug, Clone)]
    pub struct InternedLookupTable<Tok, Key> {
      alphabet: InternArena<Tok, Key>,
      locations: IndexMap<Key, Vec<gc::TokenPosition>>,
    }

    impl<Tok, Key> PartialEq for InternedLookupTable<Tok, Key>
    where
      Tok: Eq,
      Key: Hash+Eq,
    {
      fn eq(&self, other: &Self) -> bool {
        self.alphabet == other.alphabet && self.locations == other.locations
      }
    }

    impl<Tok, Key> Eq for InternedLookupTable<Tok, Key>
    where
      Tok: Eq,
      Key: Hash+Eq,
    {
    }

    impl<Tok, Key> InternedLookupTable<Tok, Key>
    where
      Tok: Eq,
      Key: From<usize>,
    {
      pub fn retrieve_intern(&self, x: &Tok) -> Option<Key> { self.alphabet.retrieve_intern(x) }

      pub fn intern_exclusive(&mut self, tok: Tok) -> Key { self.alphabet.intern_exclusive(tok) }
    }

    impl<Tok, Key> InternedLookupTable<Tok, Key>
    where
      Tok: Eq,
      Key: Hash+Eq,
    {
      pub fn insert_new_position(&mut self, entry: (Key, gc::TokenPosition)) {
        let (key, new_value) = entry;
        let entry = self.locations.entry(key).or_insert_with(|| Vec::new());
        (*entry).push(new_value);
      }

      pub fn get(&self, tok_ref: Key) -> Option<&[gc::TokenPosition]> {
        self.locations.get(&tok_ref).map(|ps| ps.as_ref())
      }
    }

    impl<Tok, Key> InternedLookupTable<Tok, Key> {
      pub fn new() -> Self {
        Self {
          alphabet: InternArena::new(),
          locations: IndexMap::new(),
        }
      }

      pub fn into_index_map(self) -> IndexMap<Key, Vec<gc::TokenPosition>> { self.locations }
    }
  }

  pub(super) mod result {
    use super::{super::graph_coordinates as gc, containers::*};
    use crate::{grammar_specification as gs, interns::InternArena};

    #[cfg(doc)]
    use super::super::graph_coordinates::{ProdRef, TokenPosition};
    #[cfg(doc)]
    use crate::grammar_specification::{Literal, ProductionReference};

    use indexmap::IndexMap;

    use core::{
      cmp::{Ord, Ordering, PartialOrd},
      fmt,
      iter::{IntoIterator, Iterator},
    };

    #[derive(Copy, Clone)]
    pub enum GrammarConstructionError<ID> {
      DuplicateProductionId(ID),
      UnrecognizedProdRefId(ID),
    }

    impl<ID> GrammarConstructionError<ID> {
      pub fn get_id(&self) -> &ID {
        match self {
          Self::DuplicateProductionId(id) => id,
          Self::UnrecognizedProdRefId(id) => id,
        }
      }
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

    impl<ID> PartialOrd for GrammarConstructionError<ID>
    where ID: PartialOrd+Eq
    {
      fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.get_id().partial_cmp(other.get_id())
      }
    }

    impl<ID> Ord for GrammarConstructionError<ID>
    where ID: Ord
    {
      fn cmp(&self, other: &Self) -> Ordering { self.get_id().cmp(other.get_id()) }
    }

    #[derive(Debug, Clone)]
    pub struct TokenGrammar<Tok> {
      pub graph: DetokenizedProductions,
      pub tokens: InternedLookupTable<Tok, gc::TokRef>,
    }

    impl<Tok> PartialEq for TokenGrammar<Tok>
    where Tok: Eq
    {
      fn eq(&self, other: &Self) -> bool {
        self.graph == other.graph && self.tokens == other.tokens
      }
    }

    impl<Tok> Eq for TokenGrammar<Tok> where Tok: Eq {}

    impl<Tok> TokenGrammar<Tok>
    where Tok: gs::constraints::Hashable
    {
      /// Walk productions and split literal strings.
      ///
      /// This method does two things:
      /// 1. Flatten out [Literal]s into individual tokens, and store a mapping
      ///    of all the [locations][TokenPosition] each token is located
      ///    at.
      /// 2. Match up [ProductionReference]s to [ProdRef]s, or error
      ///    out.
      pub fn new<Lit, ID, PR, C, P, SP>(sp: SP) -> Result<Self, GrammarConstructionError<ID>>
      where
        Lit: gs::direct::Literal<Tok=Tok>+IntoIterator<Item=Tok>,
        ID: gs::constraints::Hashable+Clone,
        PR: gs::indirect::ProductionReference<ID=ID>,
        C: gs::synthesis::Case<PR=PR>+IntoIterator<Item=gs::synthesis::CaseElement<Lit, PR>>,
        P: gs::synthesis::Production<C=C>+IntoIterator<Item=C>,
        SP: gs::synthesis::SimultaneousProductions<P=P>+IntoIterator<Item=(PR, P)>,
      {
        let (all_prods, id_prod_mapping) = {
          let mut all_prods: InternArena<P, gc::ProdRef> = InternArena::new();
          let mut id_prod_mapping: IndexMap<ID, gc::ProdRef> = IndexMap::new();
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
        let mut tokens: InternedLookupTable<Tok, gc::TokRef> = InternedLookupTable::new();
        let mut ret_prods: DetokenizedProductions = DetokenizedProductions::new();

        for (prod_ref, prod) in all_prods.into_iter() {
          let mut ret_cases: Vec<Case> = Vec::new();
          for (case_ind, case) in prod.into_iter().enumerate() {
            let case_ref: gc::CaseRef = case_ind.into();
            let mut ret_els: Vec<gc::CaseEl> = Vec::new();
            /* We want to track the positions of each token within each literal as well,
             * so we can't directly use .enumerate() */
            let mut case_el_ind: usize = 0;
            for el in case.into_iter() {
              match el {
                gs::synthesis::CaseElement::Lit(lit) => {
                  for tok in lit.into_iter() {
                    let tok_ref = tokens.intern_exclusive(tok);

                    ret_els.push(gc::CaseEl::Tok(tok_ref));

                    let el_ref: gc::CaseElRef = case_el_ind.into();
                    let cur_pos = gc::TokenPosition {
                      prod: prod_ref,
                      case: case_ref,
                      el: el_ref,
                    };
                    tokens.insert_new_position((tok_ref, cur_pos));

                    case_el_ind += 1;
                  }
                },
                gs::synthesis::CaseElement::Prod(prod_ref) => {
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
          tokens,
        })
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::{grammar_building as gb, graph_coordinates as gc};
  use crate::{state, test_framework::*};

  #[test]
  fn token_grammar_unsorted_alphabet() {
    let prods = SP::from(
      [(
        ProductionReference::from("xxx"),
        Production::from([Case::from([CE::Lit(Lit::from("cabc"))].as_ref())].as_ref()),
      )]
      .as_ref(),
    );

    let state::preprocessing::Detokenized(grammar) =
      state::preprocessing::Init(prods).try_index().unwrap();
    let mut dt = gb::DetokenizedProductions::new();
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
    let mut ts = gb::InternedLookupTable::<char, gc::TokRef>::new();
    /* NB: The tokens are allocated in the order they are encountered in the
     * grammar! */
    let c_ref = ts.intern_exclusive('c');
    let a_ref = ts.intern_exclusive('a');
    assert_eq!(a_ref, ts.intern_exclusive('a'));
    let b_ref = ts.intern_exclusive('b');
    ts.insert_new_position((c_ref, new_token_position(0, 0, 0)));
    ts.insert_new_position((a_ref, new_token_position(0, 0, 1)));
    ts.insert_new_position((b_ref, new_token_position(0, 0, 2)));
    ts.insert_new_position((c_ref, new_token_position(0, 0, 3)));
    assert_eq!(grammar, gb::TokenGrammar {
      graph: dt,
      tokens: ts,
    });
  }

  #[test]
  fn token_grammar_construction() {
    let prods = non_cyclic_productions();
    let state::preprocessing::Detokenized(grammar) =
      state::preprocessing::Init(prods).try_index().unwrap();
    let mut dt = gb::DetokenizedProductions::new();
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
    let mut ts = gb::InternedLookupTable::<char, gc::TokRef>::new();
    let a_ref = ts.intern_exclusive('a');
    assert_eq!(a_ref, ts.intern_exclusive('a'));
    let b_ref = ts.intern_exclusive('b');
    ts.insert_new_position((a_ref, new_token_position(0, 0, 0)));
    ts.insert_new_position((b_ref, new_token_position(0, 0, 1)));
    ts.insert_new_position((a_ref, new_token_position(1, 0, 0)));
    ts.insert_new_position((b_ref, new_token_position(1, 0, 1)));
    ts.insert_new_position((a_ref, new_token_position(1, 1, 1)));
    assert_eq!(grammar, gb::TokenGrammar {
      graph: dt,
      tokens: ts,
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
    let grammar: Result<gb::TokenGrammar<char>, _> = state::preprocessing::Init(prods)
      .try_index()
      .map(|state::preprocessing::Detokenized(grammar)| grammar);
    assert_eq!(
      grammar,
      Result::<gb::TokenGrammar::<char>, gb::GrammarConstructionError<ProductionReference>>::Err(
        gb::GrammarConstructionError::UnrecognizedProdRefId(ProductionReference::from("c"))
      )
    );
  }
}
