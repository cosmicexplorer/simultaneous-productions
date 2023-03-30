/*
 * Description: Implement the Simultaneous Productions general parsing
 * method.
 *
 * Copyright (C) 2019-2023 Danny McClanahan <dmcC2@hypnicjerk.ai>
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

//! Implement the Simultaneous Productions general parsing method.

/* These clippy lint descriptions are purely non-functional and do not affect the functionality
 * or correctness of the code.
 * TODO: #![warn(missing_docs)]
 * TODO: rustfmt breaks multiline comments when used one on top of another! (each with its own
 * pair of delimiters)
 * Note: run clippy with: rustup run nightly cargo-clippy! */
#![deny(unsafe_code)]
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

pub mod grammar_grammar;
mod grammar_indexing;
mod interns;
mod lowering_to_indices;
pub mod text_backend;
mod transitions;

/// The basic traits which define an input *grammar* (TODO: link to paper!).
///
/// *Implementation note: While macros may be able to streamline the process of
/// declaring a grammar, their stability guarantees can be much lower than the
/// definitions in this module.*
pub mod grammar_specification {
  /// Aliases used in the grammar specification.
  pub mod constraints {
    use core::hash::Hash;

    /// Necessary requirement to hash an object, but not e.g. to
    /// lexicographically sort it.
    pub trait Hashable: Hash + Eq {}

    pub trait SerializableGrammar {
      type Out;

      type ParseError;
      fn parse(out: &Self::Out) -> Result<Self, Self::ParseError>
      where
        Self: Sized;

      type SerializeError;
      fn serialize(&self) -> Result<Self::Out, Self::SerializeError>;
    }
  }

  /// Grammar components which expand into exactly one specific token.
  pub mod direct {
    use core::iter::IntoIterator;

    /// A contiguous sequence of tokens.
    pub trait Literal: IntoIterator {
      /// Specifies the type of "token" to iterate over when constructing a
      /// grammar.
      ///
      /// This parameter is *separate from, but may be the same as* the tokens
      /// we can actually parse with
      /// [Input::InChunk][super::execution::Input].
      type Tok: super::constraints::Hashable;
      /// Override [IntoIterator::Item] with this trait's parameter.
      type Item: Into<Self::Tok>;
    }
  }

  /// Grammar components which expand into the content of another production
  /// within the grammar.
  pub mod indirect {
    /// A type representing a [Production] that the grammar should satisfy at
    /// that position.
    pub trait ProductionReference: Into<Self::ID> {
      /// Parameterized type to reference the identity of some particular
      /// [Production].
      type ID: super::constraints::Hashable;
    }
  }

  /// Grammar components which synthesize the lower-level elements from
  /// [direct], [indirect], [explicit], and [undecidable].
  pub mod synthesis {
    use super::{direct::Literal, indirect::ProductionReference};

    use displaydoc::Display;
    use graphvizier::{entities as gv, Graphable};

    use core::{fmt, iter::IntoIterator};

    #[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
    pub enum GroupOperator {
      #[default]
      #[doc = ""]
      NoOp,
      /// ?
      Optional,
    }

    pub trait Group: IntoIterator + AsRef<GroupOperator> + Sized {
      type Lit: Literal;
      type PR: ProductionReference;
      type Item: Into<CaseElement<Self::Lit, Self::PR, Self>>;
    }

    /// Each individual element that can be matched against some input in a
    /// case.
    #[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum CaseElement<Lit, PR, Group> {
      /// literal value {0}
      Lit(Lit),
      /// production reference {0}
      Prod(PR),
      /// group {0}
      Group(Group),
    }

    /// A sequence of *elements* which, if successfully matched against some
    /// *input*, represents some *production*.
    pub trait Case: IntoIterator {
      /// Literal tokens used. in this case.
      type Lit: Literal;
      /// References to productions used in this case.
      type PR: ProductionReference;
      type Group: Group;
      /// Override of [Iterator::Item].
      type Item: Into<CaseElement<Self::Lit, Self::PR, Self::Group>>;
    }

    /// A disjunction of cases.
    pub trait Production: IntoIterator {
      /// Cases used in this production.
      type C: Case;
      /// Override of [Iterator::Item].
      type Item: Into<Self::C>;
    }

    /// A conjunction of productions (a grammar!).
    pub trait SimultaneousProductions: IntoIterator {
      /// Productions used in this grammar.
      type P: Production;
      /// Override of [Iterator::Item].
      type Item: Into<(<<Self::P as Production>::C as Case>::PR, Self::P)>;
    }

    pub struct SPGrapher<SP>(pub SP);

    impl<Tok, ID, PR, Lit, G, C, P, SP> Graphable for SPGrapher<SP>
    where
      Tok: fmt::Display,
      ID: fmt::Display,
      PR: ProductionReference<ID = ID> + Into<ID> + Clone,
      Lit: Literal<Tok = Tok> + IntoIterator<Item = Tok>,
      G: Group<Lit = Lit, PR = PR>,
      C: Case<Lit = Lit, PR = PR, Group = G> + IntoIterator<Item = CaseElement<Lit, PR, G>>,
      P: Production<C = C> + IntoIterator<Item = C>,
      SP: SimultaneousProductions<P = P> + IntoIterator<Item = (PR, P)>,
    {
      fn build_graph(self) -> graphvizier::generator::GraphBuilder {
        let mut gb = graphvizier::generator::GraphBuilder::new();
        let mut vertex_id_counter: usize = 0;
        let mut prod_vertices: Vec<gv::Vertex> = Vec::new();
        let mut prod_entities: Vec<gv::Entity> = Vec::new();

        let Self(sp) = self;

        for prod in sp.into_iter() {
          let (prod_ref, prod): (<<SP::P as Production>::C as Case>::PR, SP::P) = prod.into();
          // (1) Add vertex corresponding to any references to this production by name.
          let ref_id = format!("prod_{}", prod_ref.clone().into());
          let ref_vertex = gv::Vertex {
            id: gv::Id::new(&ref_id),
            label: Some(gv::Label(format!("#{}", prod_ref.clone().into()))),
            color: None,
            fontcolor: None,
          };
          let this_prod_ref_id = ref_vertex.id.clone();
          // (1.1) Record a vertex for each production for their own subgraph at the end
          // of this loop.
          prod_vertices.push(ref_vertex);

          // (1.2) Accumulate edges and add them after each production's subgraph.
          let mut edges: Vec<gv::Edge> = Vec::new();

          // (1.3) Create a subgraph for each production!
          let mut cur_prod_subgraph = gv::Subgraph {
            id: gv::Id::new(format!("{}_prod", prod_ref.clone().into())),
            label: Some(gv::Label(format!("Cases: \\#{}", prod_ref.clone().into()))),
            color: Some(gv::Color("purple".to_string())),
            fontcolor: Some(gv::Color("purple".to_string())),
            ..Default::default()
          };

          // (2) Traverse the productions, accumulating case elements and edges between
          //     each other and the prod refs!
          for (case_index, case) in prod.into_iter().enumerate() {
            // (2.1) Link each consecutive pair of case elements with a (directed) edge.
            let mut prev_id = this_prod_ref_id.clone();
            let mut cur_edge_color = gv::Color("red".to_string());

            // (1.3)
            let mut cur_case_subgraph = gv::Subgraph {
              id: gv::Id::new(format!("{}_case_{}", prod_ref.clone().into(), case_index)),
              label: Some(gv::Label(format!("{}", case_index))),
              color: Some(gv::Color("green4".to_string())),
              fontcolor: Some(gv::Color("green4".to_string())),
              ..Default::default()
            };

            for case_el in case.into_iter() {
              let case_el = CaseElement::from(case_el);

              // (2.2) Create a new vertex for each case element.
              let new_id = gv::Id::new(format!("vertex_{}", vertex_id_counter));
              vertex_id_counter += 1;

              match case_el {
                CaseElement::Lit(lit) => {
                  let mut joined_tokens = String::new();
                  for tok in lit.into_iter() {
                    joined_tokens.push_str(format!("{}", tok).as_str());
                  }
                  let label = gv::Label(format!("<{}>", joined_tokens));
                  let new_vertex = gv::Vertex {
                    id: new_id.clone(),
                    label: Some(label),
                    color: Some(gv::Color("brown".to_string())),
                    fontcolor: Some(gv::Color("brown".to_string())),
                  };

                  cur_case_subgraph
                    .entities
                    .push(gv::Entity::Vertex(new_vertex));
                },
                CaseElement::Prod(pr) => {
                  let prod_id: ID = pr.into();
                  let label = gv::Label(format!("ref: {}", prod_id));
                  let new_vertex = gv::Vertex {
                    id: new_id.clone(),
                    label: Some(label),
                    color: Some(gv::Color("darkgoldenrod".to_string())),
                    fontcolor: Some(gv::Color("darkgoldenrod".to_string())),
                  };

                  cur_case_subgraph
                    .entities
                    .push(gv::Entity::Vertex(new_vertex));

                  // (2.3) If this is a prod ref, then add another edge from this to the prod
                  // ref's id!
                  /* FIXME: remove duplicate format!("prod_{}", ...) calls! */
                  let target_id = gv::Id::new(format!("prod_{}", prod_id));
                  edges.push(gv::Edge {
                    source: new_id.clone(),
                    target: target_id,
                    color: Some(gv::Color("darkgoldenrod".to_string())),
                    ..Default::default()
                  });
                },
                /* FIXME: do this!!! */
                CaseElement::Group(_) => todo!("we still need to support groups in graphviz!"),
              }

              // See (2.1).
              let new_edge = gv::Edge {
                source: prev_id,
                target: new_id.clone(),
                color: Some(cur_edge_color),
                ..Default::default()
              };
              prev_id = new_id.clone();
              cur_edge_color = gv::Color("aqua".to_string());
              edges.push(new_edge);
            }

            // (2.4) Link this final case element back with the production!
            edges.push(gv::Edge {
              source: prev_id,
              target: this_prod_ref_id.clone(),
              color: Some(gv::Color("black".to_string())),
              ..Default::default()
            });

            cur_prod_subgraph
              .entities
              .push(gv::Entity::Subgraph(cur_case_subgraph));
          }
          prod_entities.push(gv::Entity::Subgraph(cur_prod_subgraph));

          for edge in edges.into_iter() {
            prod_entities.push(gv::Entity::Edge(edge));
          }
        }

        // See (1.1).
        gb.accept_entity(gv::Entity::Subgraph(gv::Subgraph {
          id: gv::Id::new("prods"),
          label: Some(gv::Label("Productions".to_string())),
          color: Some(gv::Color("blue".to_string())),
          fontcolor: Some(gv::Color("blue".to_string())),
          node_defaults: Some(gv::NodeDefaults {
            color: Some(gv::Color("blue".to_string())),
            fontcolor: Some(gv::Color("blue".to_string())),
          }),
          entities: prod_vertices.into_iter().map(gv::Entity::Vertex).collect(),
        }));

        for entity in prod_entities.into_iter() {
          gb.accept_entity(entity);
        }

        gb
      }
    }
  }
}

/// The basic traits which define the *input*, *actions*, and *output* of a
/// parse.
///
/// The basic trait [`execution::Transformer`] allows constructing pipelines of
/// multiple separate monadic interfaces:
/// 1. **Iterators:** see [`execution::iterator_api`].
/// 2. **Generators:** see the `sp_generator_api` crate.
/// 3. **Streams:** see the `sp_stream_api` crate..
pub mod execution {
  /// A "stream-like" type.
  ///
  /// A "stream-like" type has a method that returns one instance of
  /// [Self::InChunk] at a time, possibly in a blocking fashion.
  pub trait Input {
    /// Type of object to iterate over.
    type InChunk;
  }

  /// Another stream-like type.
  pub trait Output {
    /// Type of object to iterate over.
    type OutChunk;
  }

  /// A stream-like type which transforms [Self::I] into [Self::O].
  ///
  /// See the node.js [transform stream API docs] as inspiration!
  ///
  /// [transform stream API docs]: https://nodejs.org/api/stream.html#stream_implementing_a_transform_stream
  pub trait Transformer {
    /// Input stream for this transformer to consume.
    type I: Input;
    /// Output stream for this transformer to produce.
    type O: Output;
    /// The return value of [Self::transform].
    ///
    /// This type is intentionally not constrained at all in order to conform to
    /// multiple monadic APIs in a [prototypal] way. *See [iterator_api].*
    ///
    /// [prototypal]: https://en.wikipedia.org/wiki/Prototype_pattern
    type R;
    /// Consume a single block of `input`, modify any internal state, and
    /// produce a result.
    fn transform(&mut self, input: <Self::I as Input>::InChunk) -> Self::R;
  }

  /// An [`Iterator`][core::iter::Iterator]-based API to a [`Transformer`].
  pub mod iterator_api {
    use super::*;

    /// A wrapper struct which consumes a transformer `ST` and an input iterable
    /// `I`.
    ///
    /// Implements [`Iterator`] such that [`Iterator::Item`] is equal to
    /// [`Transformer::O`] when `ST` implements [`Transformer`].
    #[derive(Debug, Default, Copy, Clone)]
    pub struct STIterator<ST, I> {
      state: ST,
      iter: I,
    }

    impl<ST, I> STIterator<ST, I> {
      /// Create a new instance from a [`Transformer`] `ST` and an [`Iterator`]
      /// `I`.
      pub fn new(state: ST, iter: I) -> Self {
        Self { state, iter }
      }
    }

    impl<ST, I> From<I> for STIterator<ST, I>
    where
      ST: Default,
    {
      fn from(value: I) -> Self {
        Self::new(ST::default(), value)
      }
    }

    impl<ST, I, II, O, OO, R> Iterator for STIterator<ST, I>
    where
      I: Input<InChunk = II> + Iterator<Item = II>,
      O: Output<OutChunk = OO> + Iterator<Item = OO>,
      R: Into<Option<OO>>,
      ST: Transformer<I = I, O = O, R = R>,
    {
      type Item = OO;

      fn next(&mut self) -> Option<Self::Item> {
        self
          .iter
          .next()
          .and_then(|input| self.state.transform(input).into())
      }
    }
  }
}

/// The various phases that a grammar (in [preprocessing][state::preprocessing])
/// and then a parse (in [active][state::active]) goes through.
pub mod state {
  #[cfg(doc)]
  use crate::execution::Input;
  #[cfg(doc)]
  use preprocessing::{Detokenized, Indexed, Init};

  /// Phases of interpreting an S.P. grammar into an executable specification.
  ///
  /// `[Init] -> [Detokenized] -> [Indexed] (-> [Ready])`
  pub mod preprocessing {
    use crate::{
      grammar_indexing as gi, grammar_specification as gs,
      lowering_to_indices::grammar_building as gb,
    };

    use core::{fmt, iter::IntoIterator};

    /// Container for an implementor of
    /// [gs::synthesis::SimultaneousProductions].
    #[derive(Debug, Copy, Clone)]
    pub struct Init<SP>(pub SP);

    impl<Tok, Lit, ID, PR, Group, C, P, SP> Init<SP>
    where
      Tok: gs::constraints::Hashable,
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
      /// Create a [`gb::TokenGrammar`] and convert it to [`Detokenized`] for
      /// further preprocessing.
      pub fn try_index(self) -> Result<Detokenized<Tok>, gb::GrammarConstructionError<ID>> {
        Ok(Detokenized(gb::TokenGrammar::new(self.0)?))
      }
    }

    /// Container after converting the tokens into [gc::TokenPosition]s.
    #[derive(Debug, Clone)]
    pub struct Detokenized<Tok>(pub gb::TokenGrammar<Tok>);

    impl<Tok> Detokenized<Tok> {
      /// Create a [`gi::PreprocessedGrammar`] and convert it to [`Indexed`] for
      /// further preprocessing.
      pub fn index(self) -> Indexed<Tok> {
        Indexed(gi::PreprocessedGrammar::new(self.0))
      }
    }

    /// Container for an immediately executable grammar.
    #[derive(Debug, Clone)]
    pub struct Indexed<Tok>(pub gi::PreprocessedGrammar<Tok>);
  }
}
