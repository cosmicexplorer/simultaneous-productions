/*
 * Description: Map locations of tokens and production references.
 *
 * Copyright (C) 2021-2022 Danny McClanahan <dmcC2@hypnicjerk.ai>
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

  via_primitive![ZCRef, usize];

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub enum CaseEl {
    Tok(TokRef),
    Prod(ProdRef),
    SM(SMRef),
    ZC(ZCRef),
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
      types::{Allocator, DefaultHasher, Vec},
    };

    use core::hash::Hash;

    /* use heapless::{FnvIndexMap, FnvIndexSet, IndexMap, IndexSet, Vec}; */
    use indexmap::IndexMap;

    macro_rules! vec_type {
      ($type_name:ident, $collection_type:ty) => {
        #[derive(Debug)]
        pub struct $type_name<Arena>(pub $collection_type)
        where Arena: Allocator+Clone;

        impl<Arena> PartialEq for $type_name<Arena>
        where Arena: Allocator+Clone
        {
          fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
        }

        impl<Arena> Eq for $type_name<Arena> where Arena: Allocator+Clone {}

        impl<Arena> Clone for $type_name<Arena>
        where Arena: Allocator+Clone
        {
          fn clone(&self) -> Self { Self(self.0.clone()) }
        }
      };
    }

    vec_type![Case, Vec<gc::CaseEl, Arena>];

    vec_type![Production, Vec<Case<Arena>, Arena>];

    macro_rules! indexmap_type {
      ($type_name:ident, $collection_type:ty) => {
        #[derive(Debug)]
        pub struct $type_name<Arena>(pub $collection_type)
        where Arena: Allocator+Clone;

        impl<Arena> HandoffAllocable for $type_name<Arena>
        where Arena: Allocator+Clone
        {
          type Arena = Arena;

          fn allocator_handoff(&self) -> Arena { self.0.arena() }
        }

        impl<Arena> PartialEq for $type_name<Arena>
        where Arena: Allocator+Clone
        {
          fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
        }

        impl<Arena> Eq for $type_name<Arena> where Arena: Allocator+Clone {}

        impl<Arena> Clone for $type_name<Arena>
        where Arena: Allocator+Clone
        {
          fn clone(&self) -> Self { Self(self.0.clone()) }
        }
      };
    }

    indexmap_type![DetokenizedProductions,
                   IndexMap<gc::ProdRef, Production<Arena>, Arena, DefaultHasher>];

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

      pub fn into_index_map(
        self,
      ) -> IndexMap<gc::ProdRef, Production<Arena>, Arena, DefaultHasher> {
        self.0
      }
    }

    /// Merge an intern table with a mapping of locations in the input where the
    /// interned object is found.
    #[derive(Debug, Clone)]
    pub struct InternedLookupTable<Tok, Key, Arena>
    where Arena: Allocator+Clone
    {
      alphabet: InternArena<Tok, Key, Arena>,
      locations: IndexMap<Key, Vec<gc::TokenPosition, Arena>, Arena, DefaultHasher>,
    }

    impl<Tok, Key, Arena> PartialEq for InternedLookupTable<Tok, Key, Arena>
    where
      Tok: Eq,
      Key: Hash+Eq,
      Arena: Allocator+Clone,
    {
      fn eq(&self, other: &Self) -> bool {
        self.alphabet == other.alphabet && self.locations == other.locations
      }
    }

    impl<Tok, Key, Arena> Eq for InternedLookupTable<Tok, Key, Arena>
    where
      Tok: Eq,
      Key: Hash+Eq,
      Arena: Allocator+Clone,
    {
    }

    impl<Tok, Key, Arena> HandoffAllocable for InternedLookupTable<Tok, Key, Arena>
    where Arena: Allocator+Clone
    {
      type Arena = Arena;

      fn allocator_handoff(&self) -> Self::Arena { self.alphabet.allocator_handoff() }
    }

    impl<Tok, Key, Arena> InternedLookupTable<Tok, Key, Arena>
    where
      Arena: Allocator+Clone,
      Tok: Eq,
      Key: From<usize>,
    {
      pub fn retrieve_intern(&self, x: &Tok) -> Option<Key> { self.alphabet.retrieve_intern(x) }

      pub fn intern_exclusive(&mut self, tok: Tok) -> Key { self.alphabet.intern_exclusive(tok) }
    }

    impl<Tok, Key, Arena> InternedLookupTable<Tok, Key, Arena>
    where
      Arena: Allocator+Clone,
      Tok: Eq,
      Key: Hash+Eq,
    {
      pub fn insert_new_position(&mut self, entry: (Key, gc::TokenPosition)) {
        let (key, new_value) = entry;
        let arena = self.allocator_handoff();
        let entry = self
          .locations
          .entry(key)
          .or_insert_with(|| Vec::new_in(arena));
        (*entry).push(new_value);
      }

      pub fn get(&self, tok_ref: Key) -> Option<&[gc::TokenPosition]> {
        self.locations.get(&tok_ref).map(|ps| ps.as_ref())
      }
    }

    impl<Tok, Key, Arena> InternedLookupTable<Tok, Key, Arena>
    where Arena: Allocator+Clone
    {
      pub fn new_in(arena: Arena) -> Self {
        Self {
          alphabet: InternArena::new(arena.clone()),
          locations: IndexMap::new_in(arena),
        }
      }

      pub fn into_index_map(
        self,
      ) -> IndexMap<Key, Vec<gc::TokenPosition, Arena>, Arena, DefaultHasher> {
        self.locations
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
    pub struct TokenGrammar<Tok, Arena>
    where Arena: Allocator+Clone
    {
      pub graph: DetokenizedProductions<Arena>,
      pub tokens: InternedLookupTable<Tok, gc::TokRef, Arena>,
    }

    impl<Tok, Arena> PartialEq for TokenGrammar<Tok, Arena>
    where
      Tok: Eq,
      Arena: Allocator+Clone,
    {
      fn eq(&self, other: &Self) -> bool {
        self.graph == other.graph && self.tokens == other.tokens
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

      fn allocator_handoff(&self) -> Arena { self.tokens.allocator_handoff() }
    }

    impl<Tok, Arena> TokenGrammar<Tok, Arena>
    where
      Tok: gs::types::Hashable,
      Arena: Allocator+Clone,
    {
      /// Walk productions and split literal strings.
      ///
      /// This method does two things:
      /// 1. Flatten out [Literal]s into individual tokens, and store a mapping
      ///    of all the [locations][TokenPosition] each token is located
      ///    at.
      /// 2. Match up [ProductionReference]s to [ProdRef]s, or error
      ///    out.
      pub fn new<Lit, ID, PR, S, Sym, SymSet, N, Name, NS, SM, ZC, C, P, SP>(
        sp: SP,
        arena: Arena,
      ) -> Result<Self, GrammarConstructionError<ID>>
      where
        Lit: gs::direct::Literal<Tok=Tok>+IntoIterator<Item=Tok>,
        ID: gs::types::Hashable+Clone,
        PR: gs::indirect::ProductionReference<ID=ID>,
        S: gs::types::Hashable,
        Sym: gs::explicit::StackSym<S=S>,
        SymSet: gs::explicit::SymbolSet<Sym=Sym>+IntoIterator<Item=Sym>,
        N: gs::types::Hashable,
        Name: gs::explicit::StackName<N=N>,
        NS: gs::explicit::NamedStack<Name=Name, SymSet=SymSet>,
        SM: gs::explicit::StackManipulation<NS=NS>+IntoIterator<Item=gs::explicit::StackStep<NS>>,
        ZC: gs::undecidable::ZipperCondition<SM=SM>,
        C:
          gs::synthesis::Case<PR=PR>+IntoIterator<Item=gs::synthesis::CaseElement<Lit, PR, SM, ZC>>,
        P: gs::synthesis::Production<C=C>+IntoIterator<Item=C>,
        SP: gs::synthesis::SimultaneousProductions<P=P>+IntoIterator<Item=(PR, P)>,
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
        let mut tokens: InternedLookupTable<Tok, gc::TokRef, Arena> =
          InternedLookupTable::new_in(arena.clone());
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
                gs::synthesis::CaseElement::Stack(_) => {
                  todo!("can't handle stack manipulations yet")
                },
                gs::synthesis::CaseElement::Zipper(_) => {
                  todo!("can't handle zipper conditions yet")
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
  use crate::{state, test_framework::*, types::Global};

  #[test]
  fn token_grammar_unsorted_alphabet() {
    let prods = SP::from(
      [(
        ProductionReference::from("xxx"),
        Production::from([Case::from([CE::Lit(Lit::from("cabc"))].as_ref())].as_ref()),
      )]
      .as_ref(),
    );

    let state::preprocessing::Detokenized(grammar) = state::preprocessing::Init(prods)
      .try_index_with_allocator(Global)
      .unwrap();
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
    let mut ts = gb::InternedLookupTable::<char, gc::TokRef, Global>::new_in(Global);
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
    let state::preprocessing::Detokenized(grammar) = state::preprocessing::Init(prods)
      .try_index_with_allocator(Global)
      .unwrap();
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
    let mut ts = gb::InternedLookupTable::<char, gc::TokRef, Global>::new_in(Global);
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
    let grammar: Result<gb::TokenGrammar<char, Global>, _> = state::preprocessing::Init(prods)
      .try_index_with_allocator(Global)
      .map(|state::preprocessing::Detokenized(grammar)| grammar);
    assert_eq!(
      grammar,
      Result::<
        gb::TokenGrammar::<char, Global>,
        gb::GrammarConstructionError<ProductionReference>,
      >::Err(gb::GrammarConstructionError::UnrecognizedProdRefId(
        ProductionReference::from("c")
      ))
    );
  }
}
