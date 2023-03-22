/*
 * Description: An implementation of S.P. that works with finite blocks of text.
 *
 * Copyright (C) 2023 Danny McClanahan <dmcC2@hypnicjerk.ai>
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

//! An implementation of S.P. that works with finite blocks of text.
//!
//! Helper methods are also provided to improve the ergonomics of testing in a
//! [`no_std`] environment.
//!
//! [`no_std`]: https://docs.rust-embedded.org/book/intro/no-std.html

use crate::{grammar_specification as gs, lowering_to_indices::graph_coordinates as gc};

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
      pub fn as_new_vec(&self) -> Vec<$item> {
        self.0.clone().collect()
      }

      pub fn via_into_iter(x: <Vec<$item> as IntoIterator>::IntoIter) -> Self {
        Self(x)
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

    impl ::core::cmp::PartialOrd for $type_name {
      fn partial_cmp(&self, other: &Self) -> Option<::core::cmp::Ordering> {
        self.as_new_vec().partial_cmp(&other.as_new_vec())
      }
    }

    impl ::core::cmp::Ord for $type_name {
      fn cmp(&self, other: &Self) -> ::core::cmp::Ordering {
        self.as_new_vec().cmp(&other.as_new_vec())
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

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Group {
  pub elements: Vec<CE>,
}

impl Default for Group {
  fn default() -> Self {
    Self {
      elements: Vec::new(),
    }
  }
}

impl IntoIterator for Group {
  type Item = CE;
  type IntoIter = <Vec<Self::Item> as IntoIterator>::IntoIter;

  fn into_iter(self) -> Self::IntoIter {
    self.elements.into_iter()
  }
}

impl gs::synthesis::Group for Group {
  type Lit = Lit;
  type PR = ProductionReference;
  type Item = CE;
}

pub type CE = gs::synthesis::CaseElement<Lit, ProductionReference, Group>;

into_iter![Case, CE];

impl gs::synthesis::Case for Case {
  type Item = CE;
  type Lit = Lit;
  type Group = Group;
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
  use crate::grammar_grammar::SPTextFormat;

  use gs::constraints::SerializableGrammar;
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
  use crate::grammar_grammar::SPTextFormat;
  use gs::constraints::SerializableGrammar;

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
