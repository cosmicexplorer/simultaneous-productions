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
 * TODO: rustfmt breaks multiline comments when used one on top of another! (each with its own
 * pair of delimiters)
 * Note: run clippy with: rustup run nightly cargo-clippy! */
#![warn(missing_docs)]
/* There should be no need to use unsafe code here! */
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
    pub trait Hashable: Hash+Eq {}
  }

  pub mod graphviz {
    use uuid::Uuid;

    #[derive(Debug, Hash, PartialEq, Eq, Clone)]
    pub struct Id(pub String);

    #[derive(Debug, Clone)]
    pub struct Label(pub String);

    #[derive(Debug, Clone)]
    pub struct Color(pub String);

    #[derive(Debug, Clone)]
    pub struct Vertex {
      pub id: Id,
      pub label: Option<Label>,
      pub color: Option<Color>,
      pub fontcolor: Option<Color>,
    }

    impl Default for Vertex {
      fn default() -> Self {
        let id = Id(Uuid::new_v4().to_string());
        Self {
          id,
          label: None,
          color: None,
          fontcolor: None,
        }
      }
    }

    #[cfg(test)]
    impl Vertex {
      fn numeric(index: usize) -> Self {
        let key = format!("node_{}", index);
        Self {
          id: Id(key.clone()),
          label: Some(Label(key)),
          color: None,
          fontcolor: None,
        }
      }
    }

    #[derive(Debug, Clone, Default)]
    pub struct NodeDefaults {
      pub color: Option<Color>,
      pub fontcolor: Option<Color>,
    }

    #[derive(Debug, Clone)]
    pub enum Entity {
      Subgraph(Subgraph),
      Vertex(Vertex),
      Edge(Edge),
    }

    #[derive(Debug, Clone)]
    pub struct Subgraph {
      pub id: Id,
      pub label: Option<Label>,
      pub color: Option<Color>,
      pub fontcolor: Option<Color>,
      pub node_defaults: Option<NodeDefaults>,
      pub entities: Vec<Entity>,
    }

    impl Default for Subgraph {
      fn default() -> Self {
        let id = Id(Uuid::new_v4().to_string());
        Self {
          id,
          label: None,
          color: None,
          fontcolor: None,
          node_defaults: None,
          entities: Vec::new(),
        }
      }
    }

    #[derive(Debug, Clone)]
    pub struct Edge {
      pub source: Id,
      pub target: Id,
      pub label: Option<Label>,
      pub color: Option<Color>,
      pub fontcolor: Option<Color>,
    }

    impl Default for Edge {
      fn default() -> Self {
        Self {
          source: Id("".to_string()),
          target: Id("".to_string()),
          label: None,
          color: None,
          fontcolor: None,
        }
      }
    }

    #[derive(Debug, Hash, PartialEq, Eq, Clone)]
    pub struct DotOutput(pub String);

    pub struct GraphBuilder {
      entities: Vec<Entity>,
    }

    impl GraphBuilder {
      pub fn new() -> Self {
        Self {
          entities: Vec::new(),
        }
      }

      pub fn accept_entity(&mut self, e: Entity) { self.entities.push(e); }

      fn newline(output: &mut String) { output.push('\n'); }

      fn newline_indent(output: &mut String, indent: usize) {
        Self::newline(output);
        for _ in 0..indent {
          output.push(' ');
        }
      }

      fn bump_indent(indent: &mut usize) { *indent += 2; }

      fn unbump_indent(indent: &mut usize) {
        assert!(*indent >= 2);
        *indent -= 2;
      }

      fn print_entity(entity: Entity, mut indent: usize) -> String {
        match entity {
          Entity::Vertex(Vertex {
            id,
            label,
            color,
            fontcolor,
          }) => {
            let mut output = id.0;

            let mut modifiers: Vec<String> = Vec::new();
            if let Some(Label(label)) = label {
              modifiers.push(format!("label=\"{}\"", label));
            }
            if let Some(Color(color)) = color {
              modifiers.push(format!("color=\"{}\"", color));
            }
            if let Some(Color(fontcolor)) = fontcolor {
              modifiers.push(format!("fontcolor=\"{}\"", fontcolor));
            }

            if !modifiers.is_empty() {
              output.push('[');

              for m in modifiers.into_iter() {
                output.push_str(format!("{}, ", m).as_str());
              }

              output.push(']');
            }

            output.push(';');

            output
          },
          Entity::Edge(Edge {
            source,
            target,
            label,
            color,
            fontcolor,
          }) => {
            let mut output = format!("{} -> {}", source.0, target.0);

            let mut modifiers: Vec<String> = Vec::new();
            if let Some(Label(label)) = label {
              modifiers.push(format!("label=\"{}\"", label));
            }
            if let Some(Color(color)) = color {
              modifiers.push(format!("color=\"{}\"", color));
            }
            if let Some(Color(fontcolor)) = fontcolor {
              modifiers.push(format!("fontcolor=\"{}\"", fontcolor));
            }

            if !modifiers.is_empty() {
              output.push('[');

              for m in modifiers.into_iter() {
                output.push_str(format!("{}, ", m).as_str());
              }

              output.push(']');
            }

            output.push(';');

            output
          },
          Entity::Subgraph(Subgraph {
            id,
            label,
            color,
            fontcolor,
            node_defaults,
            entities,
          }) => {
            let mut output = format!("subgraph {} {{", id.0);
            Self::bump_indent(&mut indent);

            Self::newline_indent(&mut output, indent);
            if let Some(Label(label)) = label {
              output.push_str(format!("label = \"{}\";", label).as_str());
              Self::newline_indent(&mut output, indent);
            }
            output.push_str("cluster = true;");
            Self::newline_indent(&mut output, indent);
            output.push_str("rank = same;");
            Self::newline(&mut output);

            if let Some(Color(color)) = color {
              Self::newline_indent(&mut output, indent);
              output.push_str(format!("color = \"{}\";", color).as_str());
            }
            if let Some(Color(fontcolor)) = fontcolor {
              Self::newline_indent(&mut output, indent);
              output.push_str(format!("fontcolor = \"{}\";", fontcolor).as_str());
            }
            if let Some(NodeDefaults { color, fontcolor }) = node_defaults {
              let mut modifiers: Vec<String> = Vec::new();
              if let Some(Color(color)) = color {
                modifiers.push(format!("color=\"{}\"", color));
              }
              if let Some(Color(fontcolor)) = fontcolor {
                modifiers.push(format!("fontcolor=\"{}\"", fontcolor));
              }
              if !modifiers.is_empty() {
                Self::newline_indent(&mut output, indent);
                output.push_str("node [");
                for m in modifiers.into_iter() {
                  output.push_str(format!("{}, ", m).as_str());
                }
                output.push_str("];")
              }
            }
            Self::newline(&mut output);

            for e in entities.into_iter() {
              Self::newline_indent(&mut output, indent);
              let expr = Self::print_entity(e, indent);
              output.push_str(expr.as_str());
            }

            Self::unbump_indent(&mut indent);
            Self::newline_indent(&mut output, indent);
            output.push('}');

            output
          },
        }
      }

      pub fn build(self, graph_name: Id) -> DotOutput {
        let mut output: String = String::new();
        let mut indent: usize = 0;

        output.push_str(format!("digraph {} {{", graph_name.0).as_str());
        Self::bump_indent(&mut indent);

        Self::newline_indent(&mut output, indent);
        output.push_str("compound = true;");

        for entity in self.entities.into_iter() {
          Self::newline(&mut output);
          Self::newline_indent(&mut output, indent);

          let expr = Self::print_entity(entity, indent);
          output.push_str(expr.as_str());
        }

        Self::unbump_indent(&mut indent);
        assert_eq!(indent, 0);
        Self::newline_indent(&mut output, indent);
        output.push('}');
        Self::newline(&mut output);

        DotOutput(output)
      }
    }

    #[cfg(test)]
    mod test {
      use super::*;

      #[test]
      fn render_single_vertex() {
        let mut gb = GraphBuilder::new();
        gb.accept_entity(Entity::Vertex(Vertex::numeric(0)));
        let DotOutput(output) = gb.build(Id("test_graph".to_string()));

        assert_eq!(
          output,
          "digraph test_graph {\n  \
             compound = true;\n\n  \
             node_0[label=\"node_0\", ];\n\
           }\n"
        );
      }

      #[test]
      fn render_single_edge() {
        let mut gb = GraphBuilder::new();
        gb.accept_entity(Entity::Vertex(Vertex::numeric(0)));
        gb.accept_entity(Entity::Vertex(Vertex::numeric(1)));
        gb.accept_entity(Entity::Edge(Edge {
          source: Vertex::numeric(0).id,
          target: Vertex::numeric(1).id,
          label: Some(Label("asdf".to_string())),
          ..Default::default()
        }));

        let DotOutput(output) = gb.build(Id("test_graph".to_string()));

        assert_eq!(
          output,
          "digraph test_graph {\n  \
             compound = true;\n\n  \
             node_0[label=\"node_0\", ];\n\n  \
             node_1[label=\"node_1\", ];\n\n  \
             node_0 -> node_1[label=\"asdf\", ];\n\
           }\n"
        );
      }
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

    use core::iter::IntoIterator;

    use displaydoc::Display;

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
      pub fn new(state: ST, iter: I) -> Self { Self { state, iter } }
    }

    impl<ST, I> From<I> for STIterator<ST, I>
    where ST: Default
    {
      fn from(value: I) -> Self { Self::new(ST::default(), value) }
    }

    impl<ST, I, II, O, OO, R> Iterator for STIterator<ST, I>
    where
      I: Input<InChunk=II>+Iterator<Item=II>,
      O: Output<OutChunk=OO>+Iterator<Item=OO>,
      R: Into<Option<OO>>,
      ST: Transformer<I=I, O=O, R=R>,
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
      Lit: gs::direct::Literal<Tok=Tok>+IntoIterator<Item=Tok>,
      ID: gs::constraints::Hashable+Clone,
      PR: gs::indirect::ProductionReference<ID=ID>,
      C: gs::synthesis::Case<PR=PR>+IntoIterator<Item=gs::synthesis::CaseElement<Lit, PR>>,
      P: gs::synthesis::Production<C=C>+IntoIterator<Item=C>,
      SP: gs::synthesis::SimultaneousProductions<P=P>+IntoIterator<Item=(PR, P)>,
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
      pub fn index(self) -> Indexed<Tok> { Indexed(gi::PreprocessedGrammar::new(self.0)) }
    }

    /// Container for an immediately executable grammar.
    #[derive(Debug, Clone)]
    pub struct Indexed<Tok>(pub gi::PreprocessedGrammar<Tok>);

    impl<Tok> Indexed<Tok>
    where Tok: gs::constraints::Hashable+fmt::Debug+Clone
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
      pub fn new(grammar: p::ParseableGrammar) -> Self { Self(grammar, PhantomData) }

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
      pub fn new(parse: p::Parse) -> Self { Self(parse, PhantomData) }
    }
  }
}

/// Helper methods to improve the ergonomics of testing in a [`no_std`]
/// environment.
///
/// [`no_std`]: https://docs.rust-embedded.org/book/intro/no-std.html
#[cfg(test)]
pub mod test_framework {
  use super::grammar_specification::{self as gs, graphviz as gv};
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
        fn as_new_vec(&self) -> Vec<$item> { self.0.clone().collect() }
      }

      impl From<&[$item]> for $type_name {
        fn from(value: &[$item]) -> Self {
          Self(value.iter().cloned().collect::<Vec<_>>().into_iter())
        }
      }

      impl Iterator for $type_name {
        type Item = $item;

        fn next(&mut self) -> Option<Self::Item> { self.0.next() }
      }

      impl Hash for $type_name {
        fn hash<H: Hasher>(&self, state: &mut H) { self.as_new_vec().hash(state); }
      }

      impl PartialEq for $type_name {
        fn eq(&self, other: &Self) -> bool { self.as_new_vec() == other.as_new_vec() }
      }

      impl Eq for $type_name {}

      impl Default for $type_name {
        fn default() -> Self { Self(Vec::new().into_iter()) }
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

      #[allow(dead_code)]
      impl $type_name {
        fn into_string(&self) -> String { String::from_iter(self.0.clone()) }
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

  pub fn parse_sp_text_format(grammar: &str) -> SP {
    use indexmap::IndexMap;
    use lazy_static::lazy_static;
    use regex::Regex;

    lazy_static! {
      static ref LINE: Regex =
        Regex::new("^(?P<prod>[A-Z][a-z0-9_-]*):[[:space:]]*(?P<rest>.+)$").unwrap();
      static ref CASE: Regex = Regex::new(
        "^(?P<head>\\$[A-Z][a-z0-9_-]*|<[^>]*>)(?:[[:space:]]*->[[:space:]]*(?P<tail>.+))?$"
      )
      .unwrap();
    }

    let mut cases: IndexMap<String, Vec<Vec<CE>>> = IndexMap::new();

    for line in grammar.lines() {
      let caps = LINE.captures(line).expect("line didn't match LINE");
      let prod = caps.name("prod").unwrap().as_str();
      dbg!(prod);
      let rest = caps.name("rest").unwrap().as_str();

      let mut case_els: Vec<CE> = Vec::new();
      dbg!(rest);
      let caps = CASE.captures(rest).expect("rest didn't match CASE");
      let head = caps.name("head").unwrap().as_str();
      let mut tail = caps.name("tail").map(|c| c.as_str());

      fn parse_case_element(case_el: &str) -> CE {
        if case_el.starts_with('$') {
          CE::Prod(ProductionReference::from(&case_el[1..]))
        } else {
          assert!(case_el.starts_with('<'));
          assert!(case_el.ends_with('>'));
          CE::Lit(Lit::from(&case_el[1..(case_el.len() - 1)]))
        }
      }

      let cur_ce = parse_case_element(head);
      case_els.push(cur_ce);

      while tail.is_some() {
        dbg!(tail);
        let caps = CASE
          .captures(tail.unwrap())
          .expect("tail didn't match CASE");
        let head = caps.name("head").unwrap().as_str();
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
    SP::from(&cases[..])
  }

  /* FIXME: provide this as a generic method for any implementor of the SP
   * trait! */
  pub fn build_sp_graph(sp: SP) -> gv::GraphBuilder {
    let mut gb = gv::GraphBuilder::new();
    let mut vertex_id_counter: usize = 0;
    let mut prod_vertices: Vec<gv::Vertex> = Vec::new();
    let mut prod_entities: Vec<gv::Entity> = Vec::new();

    for (prod_ref, prod) in sp.into_iter() {
      // (1) Add vertex corresponding to any references to this production by name.
      let ref_id = format!("prod_{}", prod_ref.into_string());
      let ref_vertex = gv::Vertex {
        id: gv::Id(ref_id.clone()),
        label: Some(gv::Label(prod_ref.into_string())),
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
        id: gv::Id(format!("{}_prod", prod_ref.into_string())),
        label: Some(gv::Label(format!("Cases: {}", prod_ref.into_string()))),
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
          id: gv::Id(format!("{}_case_{}", prod_ref.into_string(), case_index)),
          label: Some(gv::Label(format!("{}", case_index))),
          color: Some(gv::Color("green4".to_string())),
          fontcolor: Some(gv::Color("green4".to_string())),
          ..Default::default()
        };

        for case_el in case.into_iter() {
          // (2.2) Create a new vertex for each case element.
          let new_id = gv::Id(format!("vertex_{}", vertex_id_counter));
          vertex_id_counter += 1;

          match case_el {
            CE::Lit(lit) => {
              let label = gv::Label(format!("<{}>", lit.into_string()));
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
            CE::Prod(pr) => {
              let label = gv::Label(format!("ref: {}", pr.into_string()));
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
              let target_id = gv::Id(format!("prod_{}", pr.into_string()));
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
      id: gv::Id("prods".to_string()),
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
    let sp = parse_sp_text_format(
      "\
A: <ab>
B: <ab> -> $A
B: $A -> <a>
",
    );

    assert_eq!(sp, non_cyclic_productions());
  }

  #[test]
  fn non_cyclic_graphviz() {
    let sp = non_cyclic_productions();
    let gb = build_sp_graph(sp);
    let gv::DotOutput(output) = gb.build(gv::Id("test_sp_graph".to_string()));

    assert_eq!(
      output,
      "digraph test_sp_graph {\n  compound = true;\n\n  subgraph prods {\n    label = \"Productions\";\n    cluster = true;\n    rank = same;\n\n    color = \"blue\";\n    fontcolor = \"blue\";\n    node [color=\"blue\", fontcolor=\"blue\", ];\n\n    prod_A[label=\"A\", ];\n    prod_B[label=\"B\", ];\n  }\n\n  subgraph A_prod {\n    label = \"Cases: A\";\n    cluster = true;\n    rank = same;\n\n    color = \"purple\";\n    fontcolor = \"purple\";\n\n    subgraph A_case_0 {\n      label = \"0\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_0[label=\"<ab>\", color=\"brown\", fontcolor=\"brown\", ];\n    }\n  }\n\n  prod_A -> vertex_0[color=\"red\", ];\n\n  vertex_0 -> prod_A[color=\"black\", ];\n\n  subgraph B_prod {\n    label = \"Cases: B\";\n    cluster = true;\n    rank = same;\n\n    color = \"purple\";\n    fontcolor = \"purple\";\n\n    subgraph B_case_0 {\n      label = \"0\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_1[label=\"<ab>\", color=\"brown\", fontcolor=\"brown\", ];\n      vertex_2[label=\"ref: A\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n    }\n    subgraph B_case_1 {\n      label = \"1\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_3[label=\"ref: A\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n      vertex_4[label=\"<a>\", color=\"brown\", fontcolor=\"brown\", ];\n    }\n  }\n\n  prod_B -> vertex_1[color=\"red\", ];\n\n  vertex_2 -> prod_A[color=\"darkgoldenrod\", ];\n\n  vertex_1 -> vertex_2[color=\"aqua\", ];\n\n  vertex_2 -> prod_B[color=\"black\", ];\n\n  vertex_3 -> prod_A[color=\"darkgoldenrod\", ];\n\n  prod_B -> vertex_3[color=\"red\", ];\n\n  vertex_3 -> vertex_4[color=\"aqua\", ];\n\n  vertex_4 -> prod_B[color=\"black\", ];\n}\n"
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
    let sp = parse_sp_text_format(
      "\
P_1: <abc>
P_1: <a> -> $P_1 -> <c>
P_1: <bc> -> $P_2
P_2: $P_1
P_2: $P_2
P_2: $P_1 -> <bc>
",
    );

    assert_eq!(sp, basic_productions());
  }

  #[test]
  fn basic_graphvis() {
    let sp = basic_productions();
    let gb = build_sp_graph(sp);
    let gv::DotOutput(output) = gb.build(gv::Id("test_sp_graph".to_string()));

    assert_eq!(output, "digraph test_sp_graph {\n  compound = true;\n\n  subgraph prods {\n    label = \"Productions\";\n    cluster = true;\n    rank = same;\n\n    color = \"blue\";\n    fontcolor = \"blue\";\n    node [color=\"blue\", fontcolor=\"blue\", ];\n\n    prod_P_1[label=\"P_1\", ];\n    prod_P_2[label=\"P_2\", ];\n  }\n\n  subgraph P_1_prod {\n    label = \"Cases: P_1\";\n    cluster = true;\n    rank = same;\n\n    color = \"purple\";\n    fontcolor = \"purple\";\n\n    subgraph P_1_case_0 {\n      label = \"0\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_0[label=\"<abc>\", color=\"brown\", fontcolor=\"brown\", ];\n    }\n    subgraph P_1_case_1 {\n      label = \"1\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_1[label=\"<a>\", color=\"brown\", fontcolor=\"brown\", ];\n      vertex_2[label=\"ref: P_1\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n      vertex_3[label=\"<c>\", color=\"brown\", fontcolor=\"brown\", ];\n    }\n    subgraph P_1_case_2 {\n      label = \"2\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_4[label=\"<bc>\", color=\"brown\", fontcolor=\"brown\", ];\n      vertex_5[label=\"ref: P_2\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n    }\n  }\n\n  prod_P_1 -> vertex_0[color=\"red\", ];\n\n  vertex_0 -> prod_P_1[color=\"black\", ];\n\n  prod_P_1 -> vertex_1[color=\"red\", ];\n\n  vertex_2 -> prod_P_1[color=\"darkgoldenrod\", ];\n\n  vertex_1 -> vertex_2[color=\"aqua\", ];\n\n  vertex_2 -> vertex_3[color=\"aqua\", ];\n\n  vertex_3 -> prod_P_1[color=\"black\", ];\n\n  prod_P_1 -> vertex_4[color=\"red\", ];\n\n  vertex_5 -> prod_P_2[color=\"darkgoldenrod\", ];\n\n  vertex_4 -> vertex_5[color=\"aqua\", ];\n\n  vertex_5 -> prod_P_1[color=\"black\", ];\n\n  subgraph P_2_prod {\n    label = \"Cases: P_2\";\n    cluster = true;\n    rank = same;\n\n    color = \"purple\";\n    fontcolor = \"purple\";\n\n    subgraph P_2_case_0 {\n      label = \"0\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_6[label=\"ref: P_1\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n    }\n    subgraph P_2_case_1 {\n      label = \"1\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_7[label=\"ref: P_2\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n    }\n    subgraph P_2_case_2 {\n      label = \"2\";\n      cluster = true;\n      rank = same;\n\n      color = \"green4\";\n      fontcolor = \"green4\";\n\n      vertex_8[label=\"ref: P_1\", color=\"darkgoldenrod\", fontcolor=\"darkgoldenrod\", ];\n      vertex_9[label=\"<bc>\", color=\"brown\", fontcolor=\"brown\", ];\n    }\n  }\n\n  vertex_6 -> prod_P_1[color=\"darkgoldenrod\", ];\n\n  prod_P_2 -> vertex_6[color=\"red\", ];\n\n  vertex_6 -> prod_P_2[color=\"black\", ];\n\n  vertex_7 -> prod_P_2[color=\"darkgoldenrod\", ];\n\n  prod_P_2 -> vertex_7[color=\"red\", ];\n\n  vertex_7 -> prod_P_2[color=\"black\", ];\n\n  vertex_8 -> prod_P_1[color=\"darkgoldenrod\", ];\n\n  prod_P_2 -> vertex_8[color=\"red\", ];\n\n  vertex_8 -> vertex_9[color=\"aqua\", ];\n\n  vertex_9 -> prod_P_2[color=\"black\", ];\n}\n");
  }
}
