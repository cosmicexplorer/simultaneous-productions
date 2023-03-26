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
use crate::grammar_specification::{direct::Literal, indirect::ProductionReference};
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

  use displaydoc::Display;

  macro_rules! via_primitive {
    ($type_name:ident, $primitive:ident) => {
      /* #[doc = $doc] */
      #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
      pub struct $type_name(pub $primitive);

      impl AsRef<$primitive> for $type_name {
        fn as_ref(&self) -> &$primitive {
          &self.0
        }
      }

      impl From<$primitive> for $type_name {
        fn from(value: $primitive) -> Self {
          Self(value)
        }
      }

      impl From<$type_name> for $primitive {
        fn from(value: $type_name) -> Self {
          value.0
        }
      }

      impl ::core::fmt::Display for $type_name {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
          write!(f, "{}", self.0)
        }
      }
    };
  }

  /* FIXME: make doc comment apply to the macro expansion!!
   * Points to a particular Production within a sequence of
   * [gs::synthesis::Production]s.
   *
   * A version of [gs::synthesis::ProductionReference] which uses a [usize] for
   * speed.
   */
  via_primitive![ProdRef, usize];

  via_primitive![CaseRef, usize];

  via_primitive![CaseElRef, usize];

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub struct ProdCaseRef {
    pub prod: ProdRef,
    pub case: CaseRef,
  }

  /// {prod}/{case}/{el}
  #[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub struct TokenPosition {
    pub prod: ProdRef,
    pub case: CaseRef,
    pub el: CaseElRef,
  }

  /* FIXME: make doc comment apply to the macro expansion!!
   * Points to a particular token value within an alphabet.
   *
   * Differs from [TokenPosition], which points to an individual *state* in
   * the graph (which may be satisfied by exactly one token *value*).
   */
  via_primitive![TokRef, usize];

  via_primitive![GroupRef, usize];

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
  pub enum CaseEl {
    Tok(TokRef),
    Prod(ProdRef),
    Group(GroupRef),
  }
}

pub mod grammar_building {
  pub use containers::*;
  pub use result::*;

  pub(super) mod containers {
    use super::super::graph_coordinates as gc;
    use crate::interns::InternArena;

    use core::hash::Hash;

    use indexmap::IndexMap;

    macro_rules! collection_type {
      ($type_name:ident, $collection_type:ty) => {
        #[derive(Debug, PartialEq, Eq, Clone, Default)]
        pub struct $type_name(pub $collection_type);

        impl $type_name {
          pub fn new() -> Self {
            Self::default()
          }
        }
      };
    }

    macro_rules! index_map_type {
      ($type_name:ident, $collection_type:ty) => {
        collection_type![$type_name, $collection_type];

        impl $type_name {
          pub fn into_index_map(self) -> $collection_type {
            self.0
          }
        }
      };
    }

    collection_type![Case, Vec<gc::CaseEl>];

    collection_type![Production, Vec<Case>];

    index_map_type![DetokenizedProductions, IndexMap<gc::ProdRef, Production>];

    impl DetokenizedProductions {
      pub fn insert_new_production(&mut self, entry: (gc::ProdRef, Production)) {
        let (key, value) = entry;
        match self.0.insert_full(key, value) {
          (_, Some(_)) => unreachable!("expected all productions to have unique IDs"),
          (_, None) => (),
        }
      }
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
      Key: Hash + Eq,
    {
      fn eq(&self, other: &Self) -> bool {
        self.alphabet == other.alphabet && self.locations == other.locations
      }
    }

    impl<Tok, Key> Eq for InternedLookupTable<Tok, Key>
    where
      Tok: Eq,
      Key: Hash + Eq,
    {
    }

    impl<Tok, Key> InternedLookupTable<Tok, Key>
    where
      Tok: Eq,
      Key: From<usize>,
    {
      pub fn key_for(&self, x: &Tok) -> Option<Key> {
        self.alphabet.key_for(x)
      }

      pub fn intern_exclusive(&mut self, tok: Tok) -> Key {
        self.alphabet.intern_exclusive(tok)
      }
    }

    impl<Tok, Key> InternedLookupTable<Tok, Key>
    where
      Tok: Eq,
      Key: Hash + Eq,
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

      pub fn into_index_map(self) -> IndexMap<Key, Vec<gc::TokenPosition>> {
        self.locations
      }
    }

    impl<Tok, Key> InternedLookupTable<Tok, Key>
    where
      Key: From<usize> + Eq + ::core::hash::Hash,
    {
      pub fn into_alphabet_index_map(self) -> IndexMap<Key, Tok> {
        self.alphabet.into_vec_with_keys().into_iter().collect()
      }
    }
  }

  pub(super) mod result {
    use super::{super::graph_coordinates as gc, containers::*};
    use crate::{grammar_specification as gs, interns::InternArena};

    #[cfg(doc)]
    use super::super::graph_coordinates::{ProdRef, TokenPosition};
    #[cfg(doc)]
    use crate::grammar_specification::{direct::Literal, indirect::ProductionReference};

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
    where
      ID: fmt::Debug,
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
    where
      ID: Eq,
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
    where
      ID: PartialOrd + Eq,
    {
      fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.get_id().partial_cmp(other.get_id())
      }
    }

    impl<ID> Ord for GrammarConstructionError<ID>
    where
      ID: Ord,
    {
      fn cmp(&self, other: &Self) -> Ordering {
        self.get_id().cmp(other.get_id())
      }
    }

    #[derive(Debug, Clone)]
    pub struct TokenGrammar<Tok> {
      pub graph: DetokenizedProductions,
      pub tokens: InternedLookupTable<Tok, gc::TokRef>,
      pub groups: IndexMap<gc::GroupRef, gc::ProdRef>,
    }

    impl<Tok> PartialEq for TokenGrammar<Tok>
    where
      Tok: Eq,
    {
      fn eq(&self, other: &Self) -> bool {
        self.graph == other.graph && self.tokens == other.tokens
      }
    }

    impl<Tok> Eq for TokenGrammar<Tok> where Tok: Eq {}

    impl<Tok> TokenGrammar<Tok>
    where
      Tok: gs::constraints::Hashable,
    {
      /// Walk productions and split literal strings.
      ///
      /// This method does two things:
      /// 1. Flatten out [Literal]s into individual tokens, and store a mapping
      ///    of all the [locations][TokenPosition] each token is located
      ///    at.
      /// 2. Match up [ProductionReference]s to [ProdRef]s, or error
      ///    out.
      pub fn new<Lit, ID, PR, Group, C, P, SP>(sp: SP) -> Result<Self, GrammarConstructionError<ID>>
      where
        Lit: gs::direct::Literal<Tok = Tok> + IntoIterator<Item = Tok>,
        ID: gs::constraints::Hashable + Clone,
        PR: gs::indirect::ProductionReference<ID = ID>,
        Group: gs::synthesis::Group<Lit = Lit, PR = PR>
          + IntoIterator<Item = gs::synthesis::CaseElement<Lit, PR, Group>>,
        C: gs::synthesis::Case<Lit = Lit, PR = PR, Group = Group>
          + IntoIterator<Item = gs::synthesis::CaseElement<Lit, PR, Group>>,
        P: gs::synthesis::Production<C = C> + IntoIterator<Item = C>,
        SP: gs::synthesis::SimultaneousProductions<P = P> + IntoIterator<Item = (PR, P)>,
      {
        use core::hash::Hash;

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
          (all_prods.into_vec_with_keys(), id_prod_mapping)
        };

        // Collect all the tokens (splitting up literals) as we traverse the
        // productions. So literal strings are "flattened" into their individual
        // tokens.
        let mut tokens: InternedLookupTable<Tok, gc::TokRef> = InternedLookupTable::new();
        /* Collect all the recursive groups, and replace them with CaseElement::GroupRef. */
        let mut groups: InternArena<gc::ProdRef, gc::GroupRef> = InternArena::new();
        let mut ret_prods: DetokenizedProductions = DetokenizedProductions::new();
        let mut group_prods_index: usize = all_prods.len();

        for (cur_prod_ref, prod) in all_prods.into_iter() {
          let mut ret_cases: Vec<Case> = Vec::new();
          for (case_ind, case) in prod.into_iter().enumerate() {
            let cur_case_ref: gc::CaseRef = case_ind.into();
            let mut ret_els: Vec<gc::CaseEl> = Vec::new();
            /* We want to track the positions of each token within each literal as well,
             * so we can't directly use .enumerate() */
            let mut case_el_ind: usize = 0;
            for el in case.into_iter() {
              /* Helper methods! */
              fn process_lit<
                Tok: Hash + Eq,
                Lit: gs::direct::Literal<Tok = Tok> + IntoIterator<Item = Tok>,
              >(
                ret_els: &mut Vec<gc::CaseEl>,
                tokens: &mut InternedLookupTable<Tok, gc::TokRef>,
                case_el_ind: &mut usize,
                cur_prod_ref: gc::ProdRef,
                cur_case_ref: gc::CaseRef,
                lit: Lit,
              ) {
                for tok in lit.into_iter() {
                  let tok_ref = tokens.intern_exclusive(tok);

                  ret_els.push(gc::CaseEl::Tok(tok_ref));

                  let el_ref = gc::CaseElRef(*case_el_ind);
                  *case_el_ind += 1;
                  let cur_pos = gc::TokenPosition {
                    prod: cur_prod_ref,
                    case: cur_case_ref,
                    el: el_ref,
                  };
                  tokens.insert_new_position((tok_ref, cur_pos));
                }
              }
              fn process_prod_ref<
                ID: gs::constraints::Hashable + Clone,
                PR: gs::indirect::ProductionReference<ID = ID>,
              >(
                id_prod_mapping: &IndexMap<ID, gc::ProdRef>,
                ret_els: &mut Vec<gc::CaseEl>,
                case_el_ind: &mut usize,
                target_prod_ref: PR,
              ) -> Result<(), GrammarConstructionError<ID>> {
                let id: ID = target_prod_ref.into();
                let pr: gc::ProdRef = match id_prod_mapping.get(&id) {
                  Some(pr) => *pr,
                  None => {
                    return Err(GrammarConstructionError::UnrecognizedProdRefId(id));
                  },
                };

                ret_els.push(gc::CaseEl::Prod(pr));
                *case_el_ind += 1;

                Ok(())
              }
              fn process_group<
                Tok: Hash + Eq,
                Lit: gs::direct::Literal<Tok = Tok> + IntoIterator<Item = Tok>,
                ID: gs::constraints::Hashable + Clone,
                PR: gs::indirect::ProductionReference<ID = ID>,
                Group: gs::synthesis::Group<Lit = Lit, PR = PR>
                  + IntoIterator<Item = gs::synthesis::CaseElement<Lit, PR, Group>>,
              >(
                id_prod_mapping: &IndexMap<ID, gc::ProdRef>,
                ret_prods: &mut DetokenizedProductions,
                ret_els: &mut Vec<gc::CaseEl>,
                tokens: &mut InternedLookupTable<Tok, gc::TokRef>,
                group_prods_index: &mut usize,
                groups: &mut InternArena<gc::ProdRef, gc::GroupRef>,
                case_el_ind: &mut usize,
                group: Group,
              ) -> Result<(), GrammarConstructionError<ID>> {
                let mut inner_ret_els: Vec<gc::CaseEl> = Vec::new();
                let mut inner_case_el_ind: usize = 0;

                let group_prod_ref = gc::ProdRef(*group_prods_index);
                *group_prods_index += 1;
                let only_case_ref = gc::CaseRef(0);
                let group_id = groups.intern_always_new_increasing(group_prod_ref.clone());

                for el in group.into_iter() {
                  match el {
                    gs::synthesis::CaseElement::Lit(lit) => {
                      process_lit::<Tok, Lit>(
                        &mut inner_ret_els,
                        tokens,
                        &mut inner_case_el_ind,
                        group_prod_ref,
                        only_case_ref,
                        lit,
                      );
                    },
                    gs::synthesis::CaseElement::Prod(target_prod_ref) => {
                      process_prod_ref::<ID, PR>(
                        &id_prod_mapping,
                        &mut inner_ret_els,
                        &mut inner_case_el_ind,
                        target_prod_ref,
                      )?;
                    },
                    gs::synthesis::CaseElement::Group(group) => {
                      process_group::<Tok, Lit, ID, PR, Group>(
                        id_prod_mapping,
                        ret_prods,
                        &mut inner_ret_els,
                        tokens,
                        group_prods_index,
                        groups,
                        &mut inner_case_el_ind,
                        group,
                      )?;
                    },
                  }
                }

                ret_prods
                  .insert_new_production((group_prod_ref, Production(vec![Case(inner_ret_els)])));
                ret_els.push(gc::CaseEl::Group(group_id));
                *case_el_ind += 1;

                Ok(())
              }

              match el {
                gs::synthesis::CaseElement::Lit(lit) => {
                  process_lit::<Tok, Lit>(
                    &mut ret_els,
                    &mut tokens,
                    &mut case_el_ind,
                    cur_prod_ref,
                    cur_case_ref,
                    lit,
                  );
                },
                gs::synthesis::CaseElement::Prod(target_prod_ref) => {
                  process_prod_ref::<ID, PR>(
                    &id_prod_mapping,
                    &mut ret_els,
                    &mut case_el_ind,
                    target_prod_ref,
                  )?;
                },
                gs::synthesis::CaseElement::Group(group) => {
                  process_group::<Tok, Lit, ID, PR, Group>(
                    &id_prod_mapping,
                    &mut ret_prods,
                    &mut ret_els,
                    &mut tokens,
                    &mut group_prods_index,
                    &mut groups,
                    &mut case_el_ind,
                    group,
                  )?;
                },
              }
            }
            ret_cases.push(Case(ret_els));
          }
          ret_prods.insert_new_production((cur_prod_ref, Production(ret_cases)));
        }

        Ok(Self {
          graph: ret_prods,
          tokens,
          groups: groups.into_vec_with_keys().into_iter().collect(),
        })
      }
    }

    impl<Tok> graphvizier::Graphable for TokenGrammar<Tok>
    where
      Tok: ::core::fmt::Display,
    {
      fn build_graph(self) -> graphvizier::generator::GraphBuilder {
        use graphvizier::entities as gv;

        let mut gb = graphvizier::generator::GraphBuilder::new();

        let Self {
          graph,
          tokens,
          groups,
        } = self;

        let mut token_edges: Vec<gv::Edge> = Vec::new();
        let mut token_vertices: Vec<gv::Vertex> = Vec::new();
        let mut tok_ref_vertices: IndexMap<gc::TokRef, gv::Vertex> = IndexMap::new();
        for (tok_ind, (tok_ref, tok)) in tokens.into_alphabet_index_map().into_iter().enumerate() {
          let tok_id = gv::Id::new(format!("token_{}", tok_ind));
          let tok_vertex = gv::Vertex {
            id: tok_id.clone(),
            label: Some(gv::Label(format!("<{}>", &tok))),
            ..Default::default()
          };
          token_vertices.push(tok_vertex);

          let gv::Vertex {
            id: ref mut tok_ref_id,
            ..
          } = tok_ref_vertices
            .entry(tok_ref.clone())
            .or_insert_with(|| gv::Vertex {
              id: gv::Id::new(format!("tok_ref_{}", &tok_ref)),
              label: Some(gv::Label(format!("{:?}", &tok_ref))),
              ..Default::default()
            });
          let tok_ref_edge = gv::Edge {
            source: tok_id.clone(),
            target: tok_ref_id.clone(),
            ..Default::default()
          };
          token_edges.push(tok_ref_edge);
        }

        let mut prod_ref_vertices: IndexMap<gc::ProdRef, gv::Vertex> = IndexMap::new();

        let mut group_edges: Vec<gv::Edge> = Vec::new();
        let mut group_ref_vertices: IndexMap<gc::GroupRef, gv::Vertex> = IndexMap::new();
        for (group_ref, prod_ref) in groups.into_iter() {
          let group_id = gv::Id::new(format!("group_{}", &group_ref));
          let group_vertex = gv::Vertex {
            id: group_id.clone(),
            label: Some(gv::Label(format!("({})", &group_ref))),
            ..Default::default()
          };
          assert!(group_ref_vertices
            .insert(group_ref.clone(), group_vertex)
            .is_none());

          let gv::Vertex {
            id: ref mut prod_ref_id,
            ..
          } = prod_ref_vertices
            .entry(prod_ref.clone())
            .or_insert_with(|| gv::Vertex {
              id: gv::Id::new(format!("prod_ref_{}", &prod_ref)),
              label: Some(gv::Label(format!("{:?}", &prod_ref))),
              ..Default::default()
            });
          let group_ref_edge = gv::Edge {
            source: group_id,
            target: prod_ref_id.clone(),
            ..Default::default()
          };
          group_edges.push(group_ref_edge);
        }

        let mut case_el_index: usize = 0;
        let mut case_el_vertices: Vec<gv::Vertex> = Vec::new();
        let mut case_el_edges: Vec<gv::Edge> = Vec::new();
        for (prod_ref, prod) in graph.into_index_map().into_iter() {
          let gv::Vertex {
            id: ref mut prod_ref_id,
            ..
          } = prod_ref_vertices
            .entry(prod_ref.clone())
            .or_insert_with(|| gv::Vertex {
              id: gv::Id::new(format!("prod_ref_{}", &prod_ref)),
              label: Some(gv::Label(format!("{:?}", &prod_ref))),
              ..Default::default()
            });

          for case in prod.0.into_iter() {
            let mut prev = prod_ref_id.clone();
            for case_el in case.0.into_iter() {
              let cur_id = {
                let id = gv::Id::new(format!("case_el_{}", case_el_index));
                case_el_index += 1;
                id
              };
              let cur_vertex = gv::Vertex {
                id: cur_id.clone(),
                label: Some(gv::Label(format!("{:?}", &case_el))),
                ..Default::default()
              };
              case_el_vertices.push(cur_vertex);

              let next_edge = gv::Edge {
                source: prev.clone(),
                target: cur_id.clone(),
                ..Default::default()
              };
              case_el_edges.push(next_edge);

              prev = cur_id;
            }

            let final_edge = gv::Edge {
              source: prev,
              target: prod_ref_id.clone(),
              ..Default::default()
            };
            case_el_edges.push(final_edge);
          }
        }

        /* Plot things now. */
        let token_vertices = gv::Subgraph {
          id: gv::Id::new("token_vertices"),
          label: Some(gv::Label("tokens".to_string())),
          color: Some(gv::Color("purple".to_string())),
          fontcolor: Some(gv::Color("purple".to_string())),
          entities: token_vertices.into_iter().map(gv::Entity::Vertex).collect(),
          ..Default::default()
        };
        gb.accept_entity(gv::Entity::Subgraph(token_vertices));

        let tok_ref_vertices = gv::Subgraph {
          id: gv::Id::new("tok_ref_vertices"),
          label: Some(gv::Label("TokRefs".to_string())),
          color: Some(gv::Color("green4".to_string())),
          fontcolor: Some(gv::Color("green4".to_string())),
          entities: tok_ref_vertices
            .into_iter()
            .map(|(_, vtx)| gv::Entity::Vertex(vtx))
            .collect(),
          ..Default::default()
        };
        gb.accept_entity(gv::Entity::Subgraph(tok_ref_vertices));

        let group_ref_vertices = gv::Subgraph {
          id: gv::Id::new("group_ref_vertices"),
          label: Some(gv::Label("group refs".to_string())),
          color: Some(gv::Color("red".to_string())),
          fontcolor: Some(gv::Color("red".to_string())),
          entities: group_ref_vertices
            .into_iter()
            .map(|(_, vtx)| gv::Entity::Vertex(vtx))
            .collect(),
          ..Default::default()
        };
        gb.accept_entity(gv::Entity::Subgraph(group_ref_vertices));

        let prod_ref_vertices = gv::Subgraph {
          id: gv::Id::new("prod_ref_vertices"),
          label: Some(gv::Label("prod refs".to_string())),
          color: Some(gv::Color("aqua".to_string())),
          fontcolor: Some(gv::Color("aqua".to_string())),
          entities: prod_ref_vertices
            .into_iter()
            .map(|(_, vtx)| gv::Entity::Vertex(vtx))
            .collect(),
          ..Default::default()
        };
        gb.accept_entity(gv::Entity::Subgraph(prod_ref_vertices));

        for case_el_vertex in case_el_vertices.into_iter() {
          gb.accept_entity(gv::Entity::Vertex(case_el_vertex));
        }

        for token_edge in token_edges.into_iter() {
          gb.accept_entity(gv::Entity::Edge(token_edge));
        }
        for group_edge in group_edges.into_iter() {
          gb.accept_entity(gv::Entity::Edge(group_edge));
        }
        for case_el_edge in case_el_edges.into_iter() {
          gb.accept_entity(gv::Entity::Edge(case_el_edge));
        }

        gb
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::{grammar_building as gb, graph_coordinates as gc};
  use crate::{state, text_backend::*};

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
    assert_eq!(
      grammar,
      gb::TokenGrammar {
        graph: dt,
        tokens: ts,
        groups: Default::default(),
      }
    );
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
    assert_eq!(
      grammar,
      gb::TokenGrammar {
        graph: dt,
        tokens: ts,
        groups: Default::default()
      }
    );
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

  #[test]
  fn non_cyclic_tokenized_graphviz() {
    use graphvizier::entities as gv;
    use graphvizier::Graphable;

    let prods = non_cyclic_productions();
    let state::preprocessing::Detokenized(token_grammar) =
      state::preprocessing::Init(prods).try_index().unwrap();

    let gb = token_grammar.build_graph();
    let graphvizier::generator::DotOutput(output) = gb.build(gv::Id::new("test_graph"));

    assert_eq!(output, "digraph test_graph {\n  compound = true;\n\n  subgraph token_vertices {\n    label = \"tokens\";\n    cluster = true;\n    rank = same;\n\n    color = \"purple\";\n    fontcolor = \"purple\";\n\n    token_0[label=\"<a>\", ];\n    token_1[label=\"<b>\", ];\n  }\n\n  subgraph tok_ref_vertices {\n    label = \"TokRefs\";\n    cluster = true;\n    rank = same;\n\n    color = \"green4\";\n    fontcolor = \"green4\";\n\n    tok_ref_0[label=\"TokRef(0)\", ];\n    tok_ref_1[label=\"TokRef(1)\", ];\n  }\n\n  subgraph group_ref_vertices {\n    label = \"group refs\";\n    cluster = true;\n    rank = same;\n\n    color = \"red\";\n    fontcolor = \"red\";\n\n  }\n\n  subgraph prod_ref_vertices {\n    label = \"prod refs\";\n    cluster = true;\n    rank = same;\n\n    color = \"aqua\";\n    fontcolor = \"aqua\";\n\n    prod_ref_0[label=\"ProdRef(0)\", ];\n    prod_ref_1[label=\"ProdRef(1)\", ];\n  }\n\n  case_el_0[label=\"Tok(TokRef(0))\", ];\n\n  case_el_1[label=\"Tok(TokRef(1))\", ];\n\n  case_el_2[label=\"Tok(TokRef(0))\", ];\n\n  case_el_3[label=\"Tok(TokRef(1))\", ];\n\n  case_el_4[label=\"Prod(ProdRef(0))\", ];\n\n  case_el_5[label=\"Prod(ProdRef(0))\", ];\n\n  case_el_6[label=\"Tok(TokRef(0))\", ];\n\n  token_0 -> tok_ref_0;\n\n  token_1 -> tok_ref_1;\n\n  prod_ref_0 -> case_el_0;\n\n  case_el_0 -> case_el_1;\n\n  case_el_1 -> prod_ref_0;\n\n  prod_ref_1 -> case_el_2;\n\n  case_el_2 -> case_el_3;\n\n  case_el_3 -> case_el_4;\n\n  case_el_4 -> prod_ref_1;\n\n  prod_ref_1 -> case_el_5;\n\n  case_el_5 -> case_el_6;\n\n  case_el_6 -> prod_ref_1;\n}\n");
  }

  #[test]
  fn basic_tokenized_graphviz() {
    use graphvizier::entities as gv;
    use graphvizier::Graphable;

    let prods = basic_productions();
    let state::preprocessing::Detokenized(token_grammar) =
      state::preprocessing::Init(prods).try_index().unwrap();

    let gb = token_grammar.build_graph();
    let graphvizier::generator::DotOutput(output) = gb.build(gv::Id::new("test_graph"));

    assert_eq!(output, "digraph test_graph {\n  compound = true;\n\n  subgraph token_vertices {\n    label = \"tokens\";\n    cluster = true;\n    rank = same;\n\n    color = \"purple\";\n    fontcolor = \"purple\";\n\n    token_0[label=\"<a>\", ];\n    token_1[label=\"<b>\", ];\n    token_2[label=\"<c>\", ];\n  }\n\n  subgraph tok_ref_vertices {\n    label = \"TokRefs\";\n    cluster = true;\n    rank = same;\n\n    color = \"green4\";\n    fontcolor = \"green4\";\n\n    tok_ref_0[label=\"TokRef(0)\", ];\n    tok_ref_1[label=\"TokRef(1)\", ];\n    tok_ref_2[label=\"TokRef(2)\", ];\n  }\n\n  subgraph group_ref_vertices {\n    label = \"group refs\";\n    cluster = true;\n    rank = same;\n\n    color = \"red\";\n    fontcolor = \"red\";\n\n  }\n\n  subgraph prod_ref_vertices {\n    label = \"prod refs\";\n    cluster = true;\n    rank = same;\n\n    color = \"aqua\";\n    fontcolor = \"aqua\";\n\n    prod_ref_0[label=\"ProdRef(0)\", ];\n    prod_ref_1[label=\"ProdRef(1)\", ];\n  }\n\n  case_el_0[label=\"Tok(TokRef(0))\", ];\n\n  case_el_1[label=\"Tok(TokRef(1))\", ];\n\n  case_el_2[label=\"Tok(TokRef(2))\", ];\n\n  case_el_3[label=\"Tok(TokRef(0))\", ];\n\n  case_el_4[label=\"Prod(ProdRef(0))\", ];\n\n  case_el_5[label=\"Tok(TokRef(2))\", ];\n\n  case_el_6[label=\"Tok(TokRef(1))\", ];\n\n  case_el_7[label=\"Tok(TokRef(2))\", ];\n\n  case_el_8[label=\"Prod(ProdRef(1))\", ];\n\n  case_el_9[label=\"Prod(ProdRef(0))\", ];\n\n  case_el_10[label=\"Prod(ProdRef(1))\", ];\n\n  case_el_11[label=\"Prod(ProdRef(0))\", ];\n\n  case_el_12[label=\"Tok(TokRef(1))\", ];\n\n  case_el_13[label=\"Tok(TokRef(2))\", ];\n\n  token_0 -> tok_ref_0;\n\n  token_1 -> tok_ref_1;\n\n  token_2 -> tok_ref_2;\n\n  prod_ref_0 -> case_el_0;\n\n  case_el_0 -> case_el_1;\n\n  case_el_1 -> case_el_2;\n\n  case_el_2 -> prod_ref_0;\n\n  prod_ref_0 -> case_el_3;\n\n  case_el_3 -> case_el_4;\n\n  case_el_4 -> case_el_5;\n\n  case_el_5 -> prod_ref_0;\n\n  prod_ref_0 -> case_el_6;\n\n  case_el_6 -> case_el_7;\n\n  case_el_7 -> case_el_8;\n\n  case_el_8 -> prod_ref_0;\n\n  prod_ref_1 -> case_el_9;\n\n  case_el_9 -> prod_ref_1;\n\n  prod_ref_1 -> case_el_10;\n\n  case_el_10 -> prod_ref_1;\n\n  prod_ref_1 -> case_el_11;\n\n  case_el_11 -> case_el_12;\n\n  case_el_12 -> case_el_13;\n\n  case_el_13 -> prod_ref_1;\n}\n");
  }

  #[test]
  fn group_tokenized_graphviz() {
    use graphvizier::entities as gv;
    use graphvizier::Graphable;

    let prods = group_productions();
    let state::preprocessing::Detokenized(token_grammar) =
      state::preprocessing::Init(prods).try_index().unwrap();

    let gb = token_grammar.build_graph();
    let graphvizier::generator::DotOutput(output) = gb.build(gv::Id::new("test_graph"));

    /* FIXME: the ::Optional operator isn't getting synced here! */
    assert_eq!(output, "asfd");
  }
}
