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

mod grammar_indexing;
mod interns;
mod lowering_to_indices;
mod parsing;
mod reconstruction;

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

    /// Each individual element that can be matched against some input in a
    /// case.
    #[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum CaseElement<Lit, PR> {
      /// literal value {0}
      Lit(Lit),
      /// production reference {0}
      Prod(PR),
    }

    /// A sequence of *elements* which, if successfully matched against some
    /// *input*, represents some *production*.
    pub trait Case: IntoIterator {
      /// Literal tokens used. in this case.
      type Lit: Literal;
      /// References to productions used in this case.
      type PR: ProductionReference;
      /// Override of [Iterator::Item].
      type Item: Into<CaseElement<Self::Lit, Self::PR>>;
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

    impl<Tok, ID, PR, Lit, C, P, SP> Graphable for SPGrapher<SP>
    where
      Tok: fmt::Display,
      ID: fmt::Display,
      PR: ProductionReference<ID = ID> + Into<ID> + Clone,
      Lit: Literal<Tok = Tok> + IntoIterator<Item = Tok>,
      C: Case<PR = PR> + IntoIterator<Item = CaseElement<Lit, PR>>,
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
  use active::{InProgress, Ready};
  #[cfg(doc)]
  use preprocessing::{Detokenized, Indexed, Init};

  /// Phases of interpreting an S.P. grammar into an executable specification.
  ///
  /// `[Init] -> [Detokenized] -> [Indexed] (-> [Ready])`
  pub mod preprocessing {
    use crate::{
      grammar_indexing as gi, grammar_specification as gs,
      lowering_to_indices::grammar_building as gb, parsing as p,
    };

    use core::{fmt, iter::IntoIterator};

    /// Container for an implementor of
    /// [gs::synthesis::SimultaneousProductions].
    #[derive(Debug, Copy, Clone)]
    pub struct Init<SP>(pub SP);

    impl<Tok, Lit, ID, PR, C, P, SP> Init<SP>
    where
      Tok: gs::constraints::Hashable,
      Lit: gs::direct::Literal<Tok = Tok> + IntoIterator<Item = Tok>,
      ID: gs::constraints::Hashable + Clone,
      PR: gs::indirect::ProductionReference<ID = ID>,
      C: gs::synthesis::Case<PR = PR> + IntoIterator<Item = gs::synthesis::CaseElement<Lit, PR>>,
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

    impl<Tok> Indexed<Tok>
    where
      Tok: gs::constraints::Hashable + fmt::Debug + Clone,
    {
      /// Create a [`p::ParseableGrammar`] and convert to a parseable state.
      ///
      /// **FIXME: `input` should be a [crate::execution::Input]!!**
      pub fn attach_input(
        &self,
        input: &p::Input<Tok>,
      ) -> Result<super::active::Ready<'_>, p::ParsingInputFailure<Tok>> {
        Ok(super::active::Ready::new(p::ParseableGrammar::new(
          self.0.clone(),
          input,
        )?))
      }
    }
  }

  /// Phases of receiving an [Input] and parsing something useful out of it.
  ///
  /// `([Indexed] ->) [Ready] -> [InProgress]`
  pub mod active {
    use crate::parsing as p;

    use core::marker::PhantomData;

    /// Container for a parseable grammar that propagates the lifetime of an
    /// input.
    #[derive(Debug, Clone)]
    pub struct Ready<'a>(pub p::ParseableGrammar, PhantomData<&'a u8>);

    impl<'a> Ready<'a> {
      #[allow(missing_docs)]
      pub fn new(grammar: p::ParseableGrammar) -> Self {
        Self(grammar, PhantomData)
      }

      /// "Detokenize" *(TODO: cite!)* the input and produce a [`p::Parse`]
      /// instance!
      pub fn initialize_parse(self) -> InProgress<'a> {
        InProgress::new(p::Parse::initialize_with_trees_for_adjacent_pairs(self.0))
      }
    }

    /// The final form of an initialized parse, ready to iterate over the input!
    #[derive(Debug, Clone)]
    pub struct InProgress<'a>(pub p::Parse, PhantomData<&'a u8>);

    impl<'a> InProgress<'a> {
      #[allow(missing_docs)]
      pub fn new(parse: p::Parse) -> Self {
        Self(parse, PhantomData)
      }
    }
  }
}

/// An implementation of S.P. that works with finite blocks of text.
///
/// Helper methods are also provided to improve the ergonomics of testing in a
/// [`no_std`] environment.
///
/// [`no_std`]: https://docs.rust-embedded.org/book/intro/no-std.html
pub mod text_backend {
  use super::grammar_specification as gs;
  use crate::lowering_to_indices::graph_coordinates as gc;

  use core::{
    hash::{Hash, Hasher},
    iter::{IntoIterator, Iterator},
    str,
  };

  pub fn new_token_position(
    prod_ind: usize,
    case_ind: usize,
    case_el_ind: usize,
  ) -> gc::TokenPosition {
    gc::TokenPosition {
      prod: gc::ProdRef(prod_ind),
      case: gc::CaseRef(case_ind),
      el: gc::CaseElRef(case_el_ind),
    }
  }

  /// Declare a type backed by [Vec::IntoIter] which forwards trait
  /// implementations to a newly constructed vector type.
  ///
  /// This allows us to implement [Iterator] without having to create a name for
  /// an intermediate `IntoIter` type.
  macro_rules! into_iter {
    ($type_name:ident, $item:ty) => {
      #[derive(Debug, Clone)]
      pub struct $type_name(<Vec<$item> as IntoIterator>::IntoIter);

      impl $type_name {
        fn as_new_vec(&self) -> Vec<$item> {
          self.0.clone().collect()
        }
      }

      impl From<&[$item]> for $type_name {
        fn from(value: &[$item]) -> Self {
          Self(value.iter().cloned().collect::<Vec<_>>().into_iter())
        }
      }

      impl Iterator for $type_name {
        type Item = $item;

        fn next(&mut self) -> Option<Self::Item> {
          self.0.next()
        }
      }

      impl Hash for $type_name {
        fn hash<H: Hasher>(&self, state: &mut H) {
          self.as_new_vec().hash(state);
        }
      }

      impl PartialEq for $type_name {
        fn eq(&self, other: &Self) -> bool {
          self.as_new_vec() == other.as_new_vec()
        }
      }

      impl Eq for $type_name {}

      impl Default for $type_name {
        fn default() -> Self {
          Self(Vec::new().into_iter())
        }
      }
    };
  }

  /// A specialization of [into_iter] for strings.
  macro_rules! string_iter {
    ($type_name:ident) => {
      into_iter![$type_name, char];

      impl From<&str> for $type_name {
        fn from(value: &str) -> Self {
          let chars = value.chars().collect::<Vec<_>>();
          Self::from(&chars[..])
        }
      }

      impl ::core::fmt::Display for $type_name {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
          write!(f, "{}", self.into_string())
        }
      }

      impl $type_name {
        pub fn into_string(&self) -> String {
          String::from_iter(self.0.clone())
        }
      }
    };
  }

  string_iter![Lit];

  impl gs::constraints::Hashable for char {}

  impl gs::direct::Literal for Lit {
    type Item = char;
    type Tok = char;
  }

  string_iter![ProductionReference];

  impl gs::constraints::Hashable for ProductionReference {}

  impl gs::indirect::ProductionReference for ProductionReference {
    type ID = Self;
  }

  pub type CE = gs::synthesis::CaseElement<Lit, ProductionReference>;

  into_iter![Case, CE];

  impl gs::synthesis::Case for Case {
    type Item = CE;
    type Lit = Lit;
    type PR = ProductionReference;
  }

  into_iter![Production, Case];

  impl gs::synthesis::Production for Production {
    type C = Case;
    type Item = Case;
  }

  into_iter![SP, (ProductionReference, Production)];

  impl gs::synthesis::SimultaneousProductions for SP {
    type Item = (ProductionReference, Self::P);
    type P = Production;
  }

  pub use grammar_grammar::{SPTextFormat, SerializableGrammar};
  pub mod grammar_grammar {
    use super::*;

    use displaydoc::Display;
    use regex::Regex;
    use thiserror::Error;

    #[derive(Debug, Display, Error, Clone)]
    pub enum GrammarGrammarParsingError {
      /// line {0} didn't match LINE: '{1}'
      LineMatchFailed(String, &'static Regex),
      /// case {0} didn't match CASE: '{1}'
      CaseMatchFailed(String, &'static Regex),
    }

    pub trait SerializableGrammar {
      type Out;

      type ParseError;
      fn parse(out: &Self::Out) -> Result<Self, Self::ParseError>
      where
        Self: Sized;

      type SerializeError;
      fn serialize(&self) -> Result<Self::Out, Self::SerializeError>;
    }

    /// grammar definition: "{0}"
    ///
    /// Convert an EBNF-like syntax into an executable
    /// [`SimultaneousProductions`](gs::synthesis::SimultaneousProductions) instance!
    ///
    /// # Line Format
    /// The format expects any number of lines of text formatted like:
    /// 1. `Line` = `ProductionName: CaseDefinition`
    /// 1. `ProductionName` = `/\$([^\$]|\$\$)*\$/`
    /// 1. `CaseDefinition` = `CaseHead( -> CaseHead)*`
    /// 1. `CaseHead` = `ProductionRef|Literal`
    /// 1. `ProductionRef` = `ProductionName`
    /// 1. `Literal` = `/<([^>]|>>)*>/`
    ///
    /// Each `Line` appends a new `CaseDefinition` to the list of [`Case`](Case)s registered for the
    /// production named `ProductionName`! Each case is a sequence of `CaseHead`s demarcated by the
    /// ` -> ` literal, and each `CaseHead` may either be a `ProductionRef` (which is just
    /// a `ProductionName`) or a `Literal` (which is a sequence of
    /// unicode characters).
    ///
    /// **Note that leading and trailing whitespace is trimmed from each line, and lines that are
    /// empty or contain only whitespace are skipped.**
    ///
    /// Here's an example of a parser that accepts the input `/abab?/` for the production `$B`:
    ///```
    /// use sp_core::text_backend::*;
    ///
    /// let sp = SP::parse(&SPTextFormat::from(
    ///   "\
    /// $A$: <ab>
    /// $B$: <ab> -> $A$
    /// $B$: $A$ -> <a>
    /// ".to_string()
    /// )).unwrap();
    ///
    /// assert_eq!(
    ///   sp,
    ///   SP::from(
    ///     [
    ///       (
    ///         ProductionReference::from("A"),
    ///         Production::from([Case::from([CE::Lit(Lit::from("ab"))].as_ref())].as_ref()),
    ///       ),
    ///       (
    ///         ProductionReference::from("B"),
    ///         Production::from(
    ///           [
    ///             Case::from(
    ///               [
    ///                 CE::Lit(Lit::from("ab")),
    ///                 CE::Prod(ProductionReference::from("A")),
    ///               ]
    ///               .as_ref(),
    ///             ),
    ///             Case::from(
    ///               [
    ///                 CE::Prod(ProductionReference::from("A")),
    ///                 CE::Lit(Lit::from("a")),
    ///               ]
    ///               .as_ref(),
    ///             ),
    ///           ]
    ///           .as_ref(),
    ///         ),
    ///       ),
    ///     ]
    ///     .as_ref(),
    ///   )
    /// );
    ///```
    ///
    /// ## Literal `>` and `$`
    /// To form a single literal `>` character, provide `>>` within a `Literal`. To form a single
    /// literal `$` character for a production name, provide `$$` within a `ProductionName`:
    ///```
    /// use sp_core::text_backend::*;
    ///
    /// let sp = SP::parse(&SPTextFormat::from(
    ///   "\
    /// $A$$B$: <a>>b>".to_string()
    /// )).unwrap();
    ///
    /// assert_eq!(
    ///   sp,
    ///   SP::from(
    ///     [(
    ///       ProductionReference::from("A$B"),
    ///       Production::from([Case::from([CE::Lit(Lit::from("a>b"))].as_ref())].as_ref())
    ///     )]
    ///     .as_ref()
    ///   )
    /// );
    ///```
    #[derive(Debug, Clone, Display, Eq, PartialEq, Hash)]
    #[ignore_extra_doc_attributes]
    pub struct SPTextFormat(String);

    impl From<String> for SPTextFormat {
      fn from(s: String) -> Self {
        Self(s)
      }
    }

    impl AsRef<str> for SPTextFormat {
      fn as_ref(&self) -> &str {
        self.0.as_str()
      }
    }

    impl SerializableGrammar for SP {
      type Out = SPTextFormat;

      type ParseError = GrammarGrammarParsingError;
      fn parse(out: &Self::Out) -> Result<Self, Self::ParseError>
      where
        Self: Sized,
      {
        use indexmap::IndexMap;
        use lazy_static::lazy_static;

        let grammar: &str = out.as_ref();

        lazy_static! {
          static ref MAYBE_SPACE: Regex = Regex::new("^[[:space:]]*$").unwrap();
          static ref LINE: Regex = Regex::new(
            "^[[:space:]]*(?P<prod>\\$(?:[^\\$]|\\$\\$)*\\$):[[:space:]]*(?P<rest>.+)[[:space:]]*$"
          ).unwrap();
          static ref CASE: Regex = Regex::new(
            "^(?P<head>\\$(?:[^\\$]|\\$\\$)*\\$|<(?:[^>]|>>)*>)(?:[[:space:]]*->[[:space:]]*(?P<tail>.+))?[[:space:]]*$"
          )
          .unwrap();
        }

        fn parse_doubled_escape(escape_char: char, s: &str) -> String {
          /* (1) Strip the end marker. */
          assert!(s.ends_with(escape_char));
          let s = &s[..(s.len() - 1)];
          dbg!(s);
          /* (2) Un-escape any doubled `escape_char`. */
          let mut prior_escape_char: bool = false;
          let mut chars: Vec<char> = Vec::new();
          for c in s.chars() {
            if c == escape_char {
              if prior_escape_char {
                chars.push(c);
                prior_escape_char = false;
              } else {
                prior_escape_char = true;
              }
            } else {
              assert!(
                !prior_escape_char,
                "no undoubled escape char should be here!"
              );
              chars.push(c);
            }
          }
          chars.into_iter().collect()
        }

        let mut cases: IndexMap<String, Vec<Vec<CE>>> = IndexMap::new();

        for line in grammar.lines() {
          /* LINE trims off any leading or trailing whitespace. */
          let caps = match LINE.captures(line) {
            Some(caps) => caps,
            None => {
              /* Ignore lines that are empty or contain only spaces. */
              if MAYBE_SPACE.is_match(line) {
                continue;
              } else {
                return Err(GrammarGrammarParsingError::LineMatchFailed(
                  line.to_string(),
                  &LINE,
                ));
              }
            },
          };
          let prod = caps.name("prod").unwrap().as_str();
          dbg!(prod);
          assert!(prod.starts_with('$'));
          let prod = parse_doubled_escape('$', &prod[1..]);
          dbg!(&prod);
          let rest = caps.name("rest").unwrap().as_str();

          let mut case_els: Vec<CE> = Vec::new();
          dbg!(rest);
          /* CASE trims off any trailing whitespace that wasn't caught by the LINE pattern. */
          /* (This is likely due to longest-first matching.) */
          let caps = CASE
            .captures(rest)
            .ok_or_else(|| GrammarGrammarParsingError::CaseMatchFailed(rest.to_string(), &CASE))?;
          let head = caps.name("head").unwrap().as_str();
          let mut tail = caps.name("tail").map(|c| c.as_str());

          fn parse_case_element(case_el: &str) -> CE {
            if case_el.starts_with('$') {
              let prod_ref = parse_doubled_escape('$', &case_el[1..]);
              CE::Prod(ProductionReference::from(prod_ref.as_str()))
            } else {
              assert!(case_el.starts_with('<'));
              let lit = parse_doubled_escape('>', &case_el[1..]);
              CE::Lit(Lit::from(lit.as_str()))
            }
          }

          let cur_ce = parse_case_element(head);
          case_els.push(cur_ce);

          while let Some(cur_tail_nonempty) = tail {
            dbg!(cur_tail_nonempty);
            let caps = CASE.captures(cur_tail_nonempty).ok_or_else(|| {
              GrammarGrammarParsingError::CaseMatchFailed(cur_tail_nonempty.to_string(), &CASE)
            })?;
            let head = caps.name("head").unwrap().as_str();
            /* Mutate `tail` here, which will affect the `while` condition. */
            tail = caps.name("tail").map(|c| c.as_str());

            let cur_ce = parse_case_element(head);
            case_els.push(cur_ce);
          }

          cases
            .entry(prod.to_string())
            .or_insert_with(Vec::new)
            .push(case_els);
        }

        let cases: Vec<(ProductionReference, Production)> = cases
          .into_iter()
          .map(|(pr, prod)| {
            let cases: Vec<Case> = prod
              .into_iter()
              .map(|case_els| Case::from(&case_els[..]))
              .collect();
            (
              ProductionReference::from(pr.as_str()),
              Production::from(&cases[..]),
            )
          })
          .collect();
        Ok(SP::from(&cases[..]))
      }

      type SerializeError = ();
      fn serialize(&self) -> Result<Self::Out, Self::SerializeError> {
        use itertools::Itertools; /* for intersperse() */

        let mut lines: Vec<String> = Vec::new();
        for (cur_prod_ref, prod) in self.clone() {
          let cur_prod_ref = cur_prod_ref.into_string().replace('$', "$$");
          for case in prod {
            let mut case_elements: Vec<String> = Vec::new();
            for case_el in case {
              match case_el {
                CE::Lit(lit) => {
                  case_elements.push(format!("<{0}>", lit.into_string().replace('>', ">>")));
                },
                CE::Prod(prod_ref) => {
                  case_elements.push(format!("${0}$", prod_ref.into_string().replace('$', "$$")));
                },
              }
            }
            let case_elements: String = case_elements
              .into_iter()
              .intersperse(" -> ".to_string())
              .collect();
            let case_line = format!("${0}$: {1}", cur_prod_ref, case_elements);
            lines.push(case_line);
          }
        }

        let lines: String = lines.into_iter().intersperse("\n".to_string()).collect();
        Ok(SPTextFormat::from(lines))
      }
    }

    #[test]
    fn test_serialize() {
      let sp = SP::from(
        [(
          ProductionReference::from("A"),
          Production::from(
            [
              Case::from([CE::Lit(Lit::from("a"))].as_ref()),
              Case::from(
                [
                  CE::Prod(ProductionReference::from("A")),
                  CE::Lit(Lit::from("b")),
                ]
                .as_ref(),
              ),
            ]
            .as_ref(),
          ),
        )]
        .as_ref(),
      );

      assert_eq!(
        sp.serialize().unwrap(),
        SPTextFormat::from(
          "\
$A$: <a>
$A$: $A$ -> <b>"
            .to_string()
        )
      );
    }

    #[test]
    fn test_serialize_escapes_double_cash() {
      let sp = SP::from(
        [(
          ProductionReference::from("A$A"),
          Production::from(
            [Case::from(
              [
                CE::Prod(ProductionReference::from("A$A")),
                CE::Lit(Lit::from("b")),
              ]
              .as_ref(),
            )]
            .as_ref(),
          ),
        )]
        .as_ref(),
      );

      let text_grammar = sp.serialize().unwrap();

      assert_eq!(
        &text_grammar,
        &SPTextFormat::from("$A$$A$: $A$$A$ -> <b>".to_string())
      );

      assert_eq!(SP::parse(&text_grammar).unwrap(), sp);
    }

    #[test]
    fn test_empty() {
      let sp = SP::parse(&SPTextFormat::from("".to_string())).unwrap();

      assert_eq!(sp, SP::from([].as_ref()));
    }

    #[test]
    fn test_strips_whitespace() {
      let sp = SP::parse(&SPTextFormat::from(" $A$: <a> ".to_string())).unwrap();

      assert_eq!(
        sp,
        SP::from(
          [(
            ProductionReference::from("A"),
            Production::from([Case::from([CE::Lit(Lit::from("a"))].as_ref())].as_ref())
          )]
          .as_ref()
        )
      );
    }

    #[test]
    fn test_escapes_double_right_arrow() {
      let sp = SP::parse(&SPTextFormat::from("$A$: <a>>b>".to_string())).unwrap();

      assert_eq!(
        sp,
        SP::from(
          [(
            ProductionReference::from("A"),
            Production::from([Case::from([CE::Lit(Lit::from("a>b"))].as_ref())].as_ref())
          )]
          .as_ref()
        )
      );

      assert_eq!(sp.serialize().unwrap().as_ref(), "$A$: <a>>b>");
    }

    #[test]
    fn test_production_ref() {
      let sp = SP::parse(&SPTextFormat::from("$A$: $B$".to_string())).unwrap();

      assert_eq!(
        sp,
        SP::from(
          [(
            ProductionReference::from("A"),
            Production::from(
              [Case::from(
                [CE::Prod(ProductionReference::from("B"))].as_ref()
              )]
              .as_ref()
            )
          )]
          .as_ref()
        ),
      );
    }

    #[test]
    fn test_line_match_fail() {
      match SP::parse(&SPTextFormat::from("A = B".to_string())) {
        Err(GrammarGrammarParsingError::LineMatchFailed(line, _)) => {
          assert_eq!(&line, "A = B");
        },
        _ => unreachable!(),
      };
    }

    #[test]
    fn test_case_match_fail() {
      match SP::parse(&SPTextFormat::from("$A$: asdf".to_string())) {
        Err(GrammarGrammarParsingError::CaseMatchFailed(case, _)) => {
          assert_eq!(&case, "asdf");
        },
        _ => unreachable!(),
      }
    }
  }

  pub fn non_cyclic_productions() -> SP {
    SP::from(
      [
        (
          ProductionReference::from("A"),
          Production::from([Case::from([CE::Lit(Lit::from("ab"))].as_ref())].as_ref()),
        ),
        (
          ProductionReference::from("B"),
          Production::from(
            [
              Case::from(
                [
                  CE::Lit(Lit::from("ab")),
                  CE::Prod(ProductionReference::from("A")),
                ]
                .as_ref(),
              ),
              Case::from(
                [
                  CE::Prod(ProductionReference::from("A")),
                  CE::Lit(Lit::from("a")),
                ]
                .as_ref(),
              ),
            ]
            .as_ref(),
          ),
        ),
      ]
      .as_ref(),
    )
  }

  #[test]
  fn non_cyclic_parse() {
    let sp = SP::parse(&SPTextFormat::from(
      "\
$A$: <ab>
$B$: <ab> -> $A$
$B$: $A$ -> <a>
"
      .to_string(),
    ))
    .unwrap();

    assert_eq!(sp, non_cyclic_productions());
  }

  #[test]
  fn non_cyclic_graphviz() {
    use graphvizier::{entities as gv, Graphable};

    let sp = non_cyclic_productions();
    let grapher = gs::synthesis::SPGrapher(sp);
    let gb = grapher.build_graph();
    let graphvizier::generator::DotOutput(output) = gb.build(gv::Id::new("test_sp_graph"));

    assert_eq!(
      output,
      "digraph test_sp_graph {\n  compound = true;\n\n  subgraph prods {\n    label = \"Productions\";\n    cluster = true;\n    rank = same;\n\n    color = \"blue\";\n    fontcolor = \"blue\";\n    node [color=\"blue\", fontcolor=\"blue\", ];\n\n    prod_A[label=\"#A\", ];\n    prod_B[label=\"#B\", ];\n  }\n\n  subgraph A_prod {\n    label = \"Cases: \\#A\";\n    cluster = true;\n    rank = same;\n\n    color = \"purple\";\n    fontcolor = \"purple\";\n\n    subgraph A_case_0 {\n      label = \"0\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_0[label=\"<ab>\", color=\"brown\", fontcolor=\"brown\", ];\n    }\n  }\n\n  prod_A -> vertex_0[color=\"red\", ];\n\n  vertex_0 -> prod_A[color=\"black\", ];\n\n  subgraph B_prod {\n    label = \"Cases: \\#B\";\n    cluster = true;\n    rank = same;\n\n    color = \"purple\";\n    fontcolor = \"purple\";\n\n    subgraph B_case_0 {\n      label = \"0\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_1[label=\"<ab>\", color=\"brown\", fontcolor=\"brown\", ];\n      vertex_2[label=\"ref: A\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n    }\n    subgraph B_case_1 {\n      label = \"1\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_3[label=\"ref: A\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n      vertex_4[label=\"<a>\", color=\"brown\", fontcolor=\"brown\", ];\n    }\n  }\n\n  prod_B -> vertex_1[color=\"red\", ];\n\n  vertex_2 -> prod_A[color=\"darkgoldenrod\", ];\n\n  vertex_1 -> vertex_2[color=\"aqua\", ];\n\n  vertex_2 -> prod_B[color=\"black\", ];\n\n  vertex_3 -> prod_A[color=\"darkgoldenrod\", ];\n\n  prod_B -> vertex_3[color=\"red\", ];\n\n  vertex_3 -> vertex_4[color=\"aqua\", ];\n\n  vertex_4 -> prod_B[color=\"black\", ];\n}\n"
    );
  }

  pub fn basic_productions() -> SP {
    SP::from(
      [
        (
          ProductionReference::from("P_1"),
          Production::from(
            [
              Case::from([CE::Lit(Lit::from("abc"))].as_ref()),
              Case::from(
                [
                  CE::Lit(Lit::from("a")),
                  CE::Prod(ProductionReference::from("P_1")),
                  CE::Lit(Lit::from("c")),
                ]
                .as_ref(),
              ),
              Case::from(
                [
                  CE::Lit(Lit::from("bc")),
                  CE::Prod(ProductionReference::from("P_2")),
                ]
                .as_ref(),
              ),
            ]
            .as_ref(),
          ),
        ),
        (
          ProductionReference::from("P_2"),
          Production::from(
            [
              Case::from([CE::Prod(ProductionReference::from("P_1"))].as_ref()),
              Case::from([CE::Prod(ProductionReference::from("P_2"))].as_ref()),
              Case::from(
                [
                  CE::Prod(ProductionReference::from("P_1")),
                  CE::Lit(Lit::from("bc")),
                ]
                .as_ref(),
              ),
            ]
            .as_ref(),
          ),
        ),
      ]
      .as_ref(),
    )
  }

  #[test]
  fn basic_parse() {
    let sp = SP::parse(&SPTextFormat::from(
      "\
$P_1$: <abc>
$P_1$: <a> -> $P_1$ -> <c>
$P_1$: <bc> -> $P_2$
$P_2$: $P_1$
$P_2$: $P_2$
$P_2$: $P_1$ -> <bc>
"
      .to_string(),
    ))
    .unwrap();

    assert_eq!(sp, basic_productions());
  }

  #[test]
  fn basic_graphviz() {
    use graphvizier::{entities as gv, Graphable};

    let sp = basic_productions();
    let grapher = gs::synthesis::SPGrapher(sp);
    let gb = grapher.build_graph();
    let graphvizier::generator::DotOutput(output) = gb.build(gv::Id::new("test_sp_graph"));

    assert_eq!(output, "digraph test_sp_graph {\n  compound = true;\n\n  subgraph prods {\n    label = \"Productions\";\n    cluster = true;\n    rank = same;\n\n    color = \"blue\";\n    fontcolor = \"blue\";\n    node [color=\"blue\", fontcolor=\"blue\", ];\n\n    prod_P_1[label=\"#P_1\", ];\n    prod_P_2[label=\"#P_2\", ];\n  }\n\n  subgraph P_1_prod {\n    label = \"Cases: \\#P_1\";\n    cluster = true;\n    rank = same;\n\n    color = \"purple\";\n    fontcolor = \"purple\";\n\n    subgraph P_1_case_0 {\n      label = \"0\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_0[label=\"<abc>\", color=\"brown\", fontcolor=\"brown\", ];\n    }\n    subgraph P_1_case_1 {\n      label = \"1\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_1[label=\"<a>\", color=\"brown\", fontcolor=\"brown\", ];\n      vertex_2[label=\"ref: P_1\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n      vertex_3[label=\"<c>\", color=\"brown\", fontcolor=\"brown\", ];\n    }\n    subgraph P_1_case_2 {\n      label = \"2\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_4[label=\"<bc>\", color=\"brown\", fontcolor=\"brown\", ];\n      vertex_5[label=\"ref: P_2\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n    }\n  }\n\n  prod_P_1 -> vertex_0[color=\"red\", ];\n\n  vertex_0 -> prod_P_1[color=\"black\", ];\n\n  prod_P_1 -> vertex_1[color=\"red\", ];\n\n  vertex_2 -> prod_P_1[color=\"darkgoldenrod\", ];\n\n  vertex_1 -> vertex_2[color=\"aqua\", ];\n\n  vertex_2 -> vertex_3[color=\"aqua\", ];\n\n  vertex_3 -> prod_P_1[color=\"black\", ];\n\n  prod_P_1 -> vertex_4[color=\"red\", ];\n\n  vertex_5 -> prod_P_2[color=\"darkgoldenrod\", ];\n\n  vertex_4 -> vertex_5[color=\"aqua\", ];\n\n  vertex_5 -> prod_P_1[color=\"black\", ];\n\n  subgraph P_2_prod {\n    label = \"Cases: \\#P_2\";\n    cluster = true;\n    rank = same;\n\n    color = \"purple\";\n    fontcolor = \"purple\";\n\n    subgraph P_2_case_0 {\n      label = \"0\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_6[label=\"ref: P_1\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n    }\n    subgraph P_2_case_1 {\n      label = \"1\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_7[label=\"ref: P_2\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n    }\n    subgraph P_2_case_2 {\n      label = \"2\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_8[label=\"ref: P_1\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n      vertex_9[label=\"<bc>\", color=\"brown\", fontcolor=\"brown\", ];\n    }\n  }\n\n  vertex_6 -> prod_P_1[color=\"darkgoldenrod\", ];\n\n  prod_P_2 -> vertex_6[color=\"red\", ];\n\n  vertex_6 -> prod_P_2[color=\"black\", ];\n\n  vertex_7 -> prod_P_2[color=\"darkgoldenrod\", ];\n\n  prod_P_2 -> vertex_7[color=\"red\", ];\n\n  vertex_7 -> prod_P_2[color=\"black\", ];\n\n  vertex_8 -> prod_P_1[color=\"darkgoldenrod\", ];\n\n  prod_P_2 -> vertex_8[color=\"red\", ];\n\n  vertex_8 -> vertex_9[color=\"aqua\", ];\n\n  vertex_9 -> prod_P_2[color=\"black\", ];\n}\n");
  }
}
