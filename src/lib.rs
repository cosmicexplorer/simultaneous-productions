/*
    Description: Implement the Simultaneous Productions general parsing method.
    Copyright (C) 2019  Danny McClanahan (https://twitter.com/hipsterelectron)

    TODO: Get Twitter to sign a copyright disclaimer!

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#![feature(fn_traits)]

extern crate indexmap;

use indexmap::{IndexMap, IndexSet};

use std::{
  collections::{HashMap, VecDeque},
  convert::From,
  hash::Hash,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Literal<Tok: PartialEq+Eq+Hash+Copy+Clone>(Vec<Tok>);

// NB: a From impl is usually intended to denote that allocation is /not/ performed, I think: see
// https://doc.rust-lang.org/std/convert/trait.From.html -- fn new() makes more sense for this use
// case.
impl Literal<char> {
  fn new(s: &str) -> Self { Literal(s.chars().collect()) }
}

// A reference to another production -- the string must match the assigned name of a production in a
// set of simultaneous productions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProductionReference(String);

impl ProductionReference {
  fn new(s: &str) -> Self { ProductionReference(s.to_string()) }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CaseElement<Tok: PartialEq+Eq+Hash+Copy+Clone> {
  Lit(Literal<Tok>),
  Prod(ProductionReference),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Case<Tok: PartialEq+Eq+Hash+Copy+Clone>(Vec<CaseElement<Tok>>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Production<Tok: PartialEq+Eq+Hash+Copy+Clone>(Vec<Case<Tok>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimultaneousProductions<Tok: PartialEq+Eq+Hash+Copy+Clone>(
  IndexMap<ProductionReference, Production<Tok>>,
);

///
/// Here comes the algorithm!
///
/// (I think this is a "model" graph class of some sort, where the model is
/// this "simultaneous productions" parsing formulation)
///
/// Vec<ProductionImpl> = [
///   Production([
///     Case([CaseEl(Lit("???")), CaseEl(ProdRef(?)), ...]),
///     ...,
///   ]),
///   ...,
/// ]
///

/// Graph Coordinates

// NB: all these Refs have nice properties, which includes being storeable without reference to any
// particular graph, being totally ordered, and being able to be incremented.
// TODO: should probably stick all of this inside a private module.

// A version of `ProductionReference` which uses a `usize` for speed. We adopt the convention of
// abbreviated names for things used in algorithms.
// Points to a particular Production within a Vec<ProductionImpl>.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct ProdRef(usize);

// Points to a particular case within a Production.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct CaseRef(usize);

// Points to an element of a particular Case.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct CaseElRef(usize);

// This refers to a specific token, implying that we must be pointing to a
// particular index of a particular Literal. This corresponds to a "state" in
// the simultaneous productions terminology.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct TokenPosition {
  prod: ProdRef,
  case: CaseRef,
  case_el: CaseElRef,
}

/// Graph Representation

// TODO: describe!
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct TokRef(usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum CaseEl {
  Tok(TokRef),
  Prod(ProdRef),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CaseImpl(Vec<CaseEl>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProductionImpl(Vec<CaseImpl>);

/// Mapping to Tokens

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenGrammar<Tok: PartialEq+Eq+Hash+Copy+Clone> {
  graph: Vec<ProductionImpl>,
  tokens: Vec<Tok>,
}

impl<Tok: PartialEq+Eq+Hash+Copy+Clone> TokenGrammar<Tok> {
  fn new(prods: &SimultaneousProductions<Tok>) -> Self {
    // Mapping from strings -> indices (TODO: from a type-indexed map, where each
    // production returns the type!).
    let prod_ref_mapping: HashMap<ProductionReference, usize> = prods
      .0
      .iter()
      .map(|(prod_ref, _)| prod_ref)
      .cloned()
      .enumerate()
      .map(|(ind, p)| (p, ind))
      .collect();
    // Collect all the tokens (splitting up literals) as we traverse the
    // productions. So literal strings are "flattened" into their individual
    // tokens.
    let mut all_tokens: IndexSet<Tok> = IndexSet::new();
    // Pretty straightforwardly map the productions into the new space.
    let new_prods: Vec<_> = prods
      .0
      .iter()
      .map(|(_, prod)| {
        let cases: Vec<_> = prod
          .0
          .iter()
          .map(|case| {
            let case_els: Vec<_> = case
              .0
              .iter()
              .flat_map(|el| match el {
                CaseElement::Lit(literal) => literal
                  .0
                  .iter()
                  .cloned()
                  .map(|cur_tok| {
                    let (tok_ind, _) = all_tokens.insert_full(cur_tok);
                    CaseEl::Tok(TokRef(tok_ind))
                  })
                  .collect::<Vec<_>>(),
                CaseElement::Prod(prod_ref) => {
                  let prod_ref_ind = prod_ref_mapping
                    .get(prod_ref)
                    .expect(&format!("prod ref {:?} not found", prod_ref));
                  vec![CaseEl::Prod(ProdRef(*prod_ref_ind))]
                },
              })
              .collect();
            CaseImpl(case_els)
          })
          .collect();
        ProductionImpl(cases)
      })
      .collect();
    TokenGrammar {
      graph: new_prods,
      tokens: all_tokens.iter().cloned().collect(),
    }
  }
}

///
/// Implementation for getting a `PreprocessedGrammar`. Performance doesn't
/// matter here.
///
mod grammar_indexing {
  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  struct StackSym(ProdRef);

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  enum StackStep {
    Positive(StackSym),
    Negative(StackSym),
  }

  #[derive(Debug, Clone, PartialEq, Eq, Hash)]
  struct StackDiff(Vec<StackStep>);

  impl StackDiff {
    fn sequence(&self, other: &Self) -> Self {
      let combined: Vec<StackStep> = self.0.iter().chain(other.0.iter()).cloned().collect();
      StackDiff(combined)
    }
  }

  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  enum LoweredState {
    Start,
    End,
    Within(TokenPosition),
  }

  // TODO: consider the relationship between populating token transitions in the
  // lookbehind cache to some specific depth (e.g. strings of 3, 4, 5 tokens)
  // and SIMD type 1 instructions (my notations: meaning recognizing a specific
  // contiguous sequence of tokens (bytes)). SIMD type 2 (finding a
  // specific token in a longer string of bytes) can already easily be used with
  // just token pairs (and others).
  // TODO: consider GPU parsing before the above!
  #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
  struct StatePair {
    left: LoweredState,
    right: LoweredState,
  }

  // NB: There is no reference to any `TokenGrammar` -- this is intentional, and
  // I believe makes it easier to have the runtime we want just fall out of the
  // code without too much work.
  #[derive(Debug, Clone, PartialEq, Eq)]
  struct PreprocessedGrammar<Tok: PartialEq+Eq+Hash+Copy+Clone> {
    // These don't need to be quick to access or otherwise optimized for the algorithm until we
    // create a `Parse` -- these are chosen to reduce redundancy.
    // `M: T -> {Q}`, where `{Q}` is sets of states!
    states: IndexMap<Tok, Vec<TokenPosition>>,
    // TODO: we don't yet support stack cycles (ignored), or multiple stack paths to the same
    // succeeding state from an initial state (also ignored) -- details in
    // build_pairwise_transitions_table().
    // `A: T x T -> {S}^+_-`, where `{S}^+_-` (LaTeX formatting) is ordered sequences of signed
    // stack symbols!
    transitions: IndexMap<StatePair, Vec<StackDiff>>,
  }

}

impl<Tok: PartialEq+Eq+Hash+Copy+Clone> PreprocessedGrammar<Tok> {
  fn new(grammar: &TokenGrammar<Tok>) -> Self {
    let (states, neighbors) = Self::index_tokens(grammar);
    let transitions = Self::build_pairwise_transitions_table(grammar, &states, &neighbors);
    PreprocessedGrammar {
      states,
      transitions,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn initialize_parse_state() {
    // TODO: figure out more complex parsing such as stack cycles/etc before doing
    // type-indexed maps, as well as syntax sugar for defining cases.
    let prods = SimultaneousProductions(
      [
        (
          ProductionReference::new("a"),
          Production(vec![Case(vec![CaseElement::Lit(Literal::new("ab"))])]),
        ),
        (
          ProductionReference::new("b"),
          Production(vec![Case(vec![
            CaseElement::Lit(Literal::new("ab")),
            CaseElement::Prod(ProductionReference::new("a")),
          ])]),
        ),
      ].iter()
        .cloned()
        .collect(),
    );
    let grammar = TokenGrammar::new(&prods);
    let preprocessed_grammar = PreprocessedGrammar::new(&grammar);
    let input: Vec<char> = "abab".chars().collect();
    let parse = Parse::new(&preprocessed_grammar, input);
    let first_a = TokenPosition {
      prod: ProdRef(0),
      case: CaseRef(0),
      case_el: CaseElRef(0),
    };
    let second_a = TokenPosition {
      prod: ProdRef(1),
      case: CaseRef(0),
      case_el: CaseElRef(0),
    };
    let first_b = TokenPosition {
      prod: ProdRef(0),
      case: CaseRef(0),
      case_el: CaseElRef(1),
    };
    let second_b = TokenPosition {
      prod: ProdRef(1),
      case: CaseRef(0),
      case_el: CaseElRef(1),
    };
    let into_a_prod = StackStep::Positive(StackSym(ProdRef(0)));
    let out_of_a_prod = StackStep::Negative(StackSym(ProdRef(0)));
    assert_eq!(
      parse,
      Parse(vec![
        StackTrie {
          stack_steps: vec![StackDiff(vec![])],
          terminal_entries: vec![StackTrieTerminalEntry(vec![
            UnionRange::new(first_a, InputTokenIndex(1), first_b),
            UnionRange::new(second_a, InputTokenIndex(1), second_b),
          ])],
        },
        // StackTrie {},
        StackTrie {
          stack_steps: vec![StackDiff(vec![]), StackDiff(vec![into_a_prod])],
          terminal_entries: vec![StackTrieTerminalEntry(vec![
            UnionRange::new(first_a, InputTokenIndex(3), first_b),
            UnionRange::new(second_a, InputTokenIndex(3), second_b),
          ])],
        },
        // StackTrie {},
      ])
    );
  }

  #[test]
  fn preprocessed_state_for_non_cyclic_productions() {
    let prods = SimultaneousProductions(
      [
        (
          ProductionReference::new("a"),
          Production(vec![Case(vec![CaseElement::Lit(Literal::new("ab"))])]),
        ),
        (
          ProductionReference::new("b"),
          Production(vec![
            Case(vec![
              CaseElement::Lit(Literal::new("ab")),
              CaseElement::Prod(ProductionReference::new("a")),
            ]),
            Case(vec![
              CaseElement::Prod(ProductionReference::new("a")),
              CaseElement::Lit(Literal::new("a")),
            ]),
          ]),
        ),
      ].iter()
        .cloned()
        .collect(),
    );
    let grammar = TokenGrammar::new(&prods);
    assert_eq!(
      grammar.clone(),
      TokenGrammar {
        tokens: vec!['a', 'b'],
        graph: vec![
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
        ],
      }
    );
    let preprocessed_grammar = PreprocessedGrammar::new(&grammar);
    let first_a = LoweredState::Within(TokenPosition {
      prod: ProdRef(0),
      case: CaseRef(0),
      case_el: CaseElRef(0),
    });
    let first_b = LoweredState::Within(TokenPosition {
      prod: ProdRef(0),
      case: CaseRef(0),
      case_el: CaseElRef(1),
    });
    let second_a = LoweredState::Within(TokenPosition {
      prod: ProdRef(1),
      case: CaseRef(0),
      case_el: CaseElRef(0),
    });
    let second_b = LoweredState::Within(TokenPosition {
      prod: ProdRef(1),
      case: CaseRef(0),
      case_el: CaseElRef(1),
    });
    let third_a = LoweredState::Within(TokenPosition {
      prod: ProdRef(1),
      case: CaseRef(1),
      case_el: CaseElRef(1),
    });
    let a_prod = StackSym(ProdRef(0));
    let b_prod = StackSym(ProdRef(1));
    assert_eq!(
      preprocessed_grammar.clone(),
      PreprocessedGrammar {
        states: vec![
          (
            'a',
            vec![
              TokenPosition {
                prod: ProdRef(0),
                case: CaseRef(0),
                case_el: CaseElRef(0),
              },
              TokenPosition {
                prod: ProdRef(1),
                case: CaseRef(0),
                case_el: CaseElRef(0),
              },
            ],
          ),
          (
            'b',
            vec![
              TokenPosition {
                prod: ProdRef(0),
                case: CaseRef(0),
                case_el: CaseElRef(1),
              },
              TokenPosition {
                prod: ProdRef(1),
                case: CaseRef(0),
                case_el: CaseElRef(1),
              },
            ],
          ),
        ].iter()
          .cloned()
          .collect::<IndexMap<char, Vec<TokenPosition>>>(),
        transitions: vec![
          (
            StatePair {
              left: first_a,
              right: first_b,
            },
            vec![StackDiff(vec![])],
          ),
          (
            StatePair {
              left: second_a,
              right: second_b,
            },
            vec![StackDiff(vec![])],
          ),
          (
            StatePair {
              left: first_b,
              right: LoweredState::End,
            },
            vec![
              StackDiff(vec![StackStep::Negative(a_prod)]),
              // TODO: this is currently missing! this happens because a prod ref to "a" is at the
              // end of the single case of the "b" production -- we can recognize
              // this case in index_tokens() (ugh) and propagate it (probably not
              // that hard, could be done by adding an "end" case to the
              // `GrammarVertex` enum!)!
              StackDiff(vec![
                StackStep::Negative(a_prod),
                StackStep::Negative(b_prod),
              ]),
            ],
          ),
          (
            StatePair {
              left: third_a,
              right: LoweredState::End,
            },
            vec![StackDiff(vec![StackStep::Negative(b_prod)])],
          ),
          (
            StatePair {
              left: first_b,
              right: first_a,
            },
            vec![StackDiff(vec![
              StackStep::Negative(a_prod),
              StackStep::Positive(a_prod),
            ])],
          ),
          (
            StatePair {
              left: first_b,
              right: second_a,
            },
            vec![StackDiff(vec![
              StackStep::Negative(a_prod),
              StackStep::Positive(b_prod),
            ])],
          ),
          (
            StatePair {
              left: first_b,
              right: third_a,
            },
            vec![StackDiff(vec![StackStep::Negative(a_prod)])],
          ),
          (
            StatePair {
              left: second_b,
              right: first_a,
            },
            vec![StackDiff(vec![StackStep::Positive(a_prod)])],
          ),
          (
            StatePair {
              left: LoweredState::Start,
              right: first_a,
            },
            vec![
              StackDiff(vec![StackStep::Positive(a_prod)]),
              StackDiff(vec![
                StackStep::Positive(b_prod),
                StackStep::Positive(a_prod),
              ]),
            ],
          ),
          (
            StatePair {
              left: LoweredState::Start,
              right: second_a,
            },
            vec![StackDiff(vec![StackStep::Positive(b_prod)])],
          ),
        ].iter()
          .cloned()
          .collect::<IndexMap<StatePair, Vec<StackDiff>>>(),
      }
    );
  }

  #[test]
  #[should_panic(expected = "prod ref ProductionReference(\"c\") not found")]
  fn missing_prod_ref() {
    let prods = SimultaneousProductions(
      [(
        ProductionReference::new("b"),
        Production(vec![Case(vec![
          CaseElement::Lit(Literal::new("ab")),
          CaseElement::Prod(ProductionReference::new("c")),
        ])]),
      )].iter()
        .cloned()
        .collect(),
    );
    TokenGrammar::new(&prods);
  }
}
