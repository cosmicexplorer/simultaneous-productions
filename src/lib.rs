#![feature(fn_traits)]
// #![feature(trait_alias)]

extern crate indexmap;

// TODO: indexmap here is only used for testing purposes, so we can compare the results (see
// `basic_productions()`) -- figure out if there is a better way to do this.
use indexmap::{IndexMap, IndexSet};

use std::hash::{Hash, Hasher};
use std::iter::FromIterator;

// TODO: trait aliases are not fully implemented!
// trait TokenBound = Sized + PartialEq + Eq + Hash + Clone;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Literal<Tok: Sized + PartialEq + Eq + Hash + Clone>(Vec<Tok>);

impl Literal<char> {
  fn from(s: &str) -> Self {
    Literal(s.chars().collect())
  }
}

// A reference to another production -- the string must match the assigned name of a production in a
// set of simultaneous productions.
// NB: The `Ord` derivation lets us reliably hash `SimultaneousProductions<Tok>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProductionReference(String);

impl ProductionReference {
  fn new(s: &str) -> Self {
    ProductionReference(s.to_string())
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CaseElement<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  Lit(Literal<Tok>),
  Prod(ProductionReference),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Case<Tok: Sized + PartialEq + Eq + Hash + Clone>(Vec<CaseElement<Tok>>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Production<Tok: Sized + PartialEq + Eq + Hash + Clone>(Vec<Case<Tok>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimultaneousProductions<Tok: Sized + PartialEq + Eq + Hash + Clone>(
  IndexMap<ProductionReference, Production<Tok>>);

impl <Tok: Sized + PartialEq + Eq + Hash + Clone> Hash for SimultaneousProductions<Tok> {
  fn hash<H: Hasher>(&self, state: &mut H) {
    let mut keys: Vec<_> = self.0.keys().collect();
    keys.sort();
    for k in keys.iter().cloned() {
      let val = self.0.get(k).unwrap();
      k.hash(state);
      val.hash(state);
    }
  }
}

// This can be used to search backwards and/or forwards for allowed states during the execution of
// the partitioning algorithm!
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConsecutiveTokenPair<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  // TODO: consider a better naming scheme -- this is probably fine for now!
  left_tok: Tok,
  right_tok: Tok,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TokenPositionInProduction<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  productions_context: SimultaneousProductions<Tok>,
  case_context: Case<Tok>,
  case_pos: usize,
  literal_context: Literal<Tok>,
  literal_pos: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StackSymbol(ProductionReference);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StackChangeUnit {
  Positive(StackSymbol),
  Negative(StackSymbol),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TokenWithPosition<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  tok: Tok,
  pos: TokenPositionInProduction<Tok>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackChangeSet(Vec<StackChangeUnit>);

impl StackChangeSet {
  fn reversed(&self) -> Self {
    let rev_changes = self.0.clone().into_iter().rev().collect::<Vec<_>>();
    StackChangeSet(rev_changes)
  }
}

// The specific ordering of the vector is meaningful, so this Hash impl makes sense.
impl Hash for StackChangeSet {
  fn hash<H: Hasher>(&self, state: &mut H) {
    for unit_change in self.0.iter() {
      unit_change.hash(state);
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StateChange<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  // TODO: map `left_state` and `right_state` to simply unique strings, which are the value of some
  // Map<Tok, String> -- we don't need all the information from `TokenPositionInProduction` *during
  // parsing*.
  left_state: TokenWithPosition<Tok>,
  right_state: TokenWithPosition<Tok>,
  // NB: when going backwards, this should be reversed!
  stack_changes: StackChangeSet,
}

// This contains all of the possible state changes that can result from moving from one token to
// another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowedTransitions<Tok: Sized + PartialEq + Eq + Hash + Clone>(IndexSet<StateChange<Tok>>);

impl <Tok: Sized + PartialEq + Eq + Hash + Clone> TokenWithPosition<Tok> {
  fn collect_backward_forward_transitions(&self) -> AllowedTransitions<Tok> {
    // forward literals
    let forward_transitions: Vec<StateChange<Tok>> = if self.pos.literal_pos < (self.pos.literal_context.0.len() - 1) {
      let next_index = self.pos.literal_pos + 1;
      let next_token = self.pos.literal_context.0.get(next_index).unwrap().clone();
      let next_pos = TokenPositionInProduction {
        productions_context: self.pos.productions_context.clone(),
        case_context: self.pos.case_context.clone(),
        case_pos: self.pos.case_pos,
        literal_context: self.pos.literal_context.clone(),
        literal_pos: next_index,
      };
      let next_token_with_pos = TokenWithPosition {
        tok: next_token,
        pos: next_pos,
      };
      let next_state_change = StateChange {
        left_state: self.clone(),
        right_state: next_token_with_pos,
        stack_changes: StackChangeSet(vec![]),
      };
      vec![next_state_change]
    } else if self.pos.case_pos < (self.pos.case_context.0.len() - 1) {
      let next_case_index = self.pos.case_pos + 1;
      let next_case_element = self.pos.case_context.0.get(next_case_index).unwrap().clone();
      match next_case_element.clone() {
        CaseElement::Lit(next_literal) => {
          // We assert!(next_literal.0.len() > 0) in generate_token_index()!
          let new_next_case_el_init_pos = 0;
          let next_case_el_lit_pos = TokenPositionInProduction {
            productions_context: self.pos.productions_context.clone(),
            case_context: self.pos.case_context.clone(),
            case_pos: next_case_index,
            literal_context: next_literal.clone(),
            literal_pos: new_next_case_el_init_pos,
          };
          let next_case_el_tok_with_pos = TokenWithPosition {
            tok: next_literal.0.get(new_next_case_el_init_pos).unwrap().clone(),
            pos: next_case_el_lit_pos,
          };
          let next_case_el_state_change = StateChange {
            left_state: self.clone(),
            right_state: next_case_el_tok_with_pos,
            stack_changes: StackChangeSet(vec![]),
          };
          vec![next_case_el_state_change]
        },
        CaseElement::Prod(next_prod_ref) => {
          panic!("???/next_prod_ref");
        },
      }
    } else {
      // We're at the end of a case -- do nothing (this is supposed to be covered by moving
      // backward/forward from other states).
      vec![]
    };
    // backward literals
    let backward_transitions: Vec<StateChange<Tok>> = if self.pos.literal_pos > 0 {
      let prev_index = self.pos.literal_pos - 1;
      let prev_token = self.pos.literal_context.0.get(prev_index).unwrap().clone();
      let prev_pos = TokenPositionInProduction {
        productions_context: self.pos.productions_context.clone(),
        case_context: self.pos.case_context.clone(),
        case_pos: self.pos.case_pos,
        literal_context: self.pos.literal_context.clone(),
        literal_pos: prev_index,
      };
      let prev_token_with_pos = TokenWithPosition {
        tok: prev_token,
        pos: prev_pos,
      };
      let prev_state_change = StateChange {
        left_state: prev_token_with_pos,
        right_state: self.clone(),
        stack_changes: StackChangeSet(vec![]),
      };
      vec![prev_state_change]
    } else if self.pos.case_pos > 0 {
      let prev_case_index = self.pos.case_pos - 1;
      let prev_case_element = self.pos.case_context.0.get(prev_case_index).unwrap().clone();
      match prev_case_element.clone() {
        CaseElement::Lit(prev_literal) => {
          // We assert!(prev_literal.0.len() > 0) in generate_token_index()!
          let new_prev_case_el_init_pos = prev_literal.0.len() - 1;
          let prev_case_el_lit_pos = TokenPositionInProduction {
            productions_context: self.pos.productions_context.clone(),
            case_context: self.pos.case_context.clone(),
            case_pos: prev_case_index,
            literal_context: prev_literal.clone(),
            literal_pos: new_prev_case_el_init_pos,
          };
          let prev_case_el_tok_with_pos = TokenWithPosition {
            tok: prev_literal.0.get(new_prev_case_el_init_pos).unwrap().clone(),
            pos: prev_case_el_lit_pos,
          };
          let prev_case_el_state_change = StateChange {
            left_state: prev_case_el_tok_with_pos,
            right_state: self.clone(),
            stack_changes: StackChangeSet(vec![]),
          };
          vec![prev_case_el_state_change]
        },
        CaseElement::Prod(prev_prod_ref) => {
          panic!("???/prev_prod_ref");
        },
      }
    } else {
      // We're at the beginning of a case -- do nothing (this is supposed to be covered by moving
      // backward/forward from other states).
      vec![]
    };
    let cur_transitions: IndexSet<StateChange<Tok>> = IndexSet::from_iter(
      forward_transitions.into_iter().chain(backward_transitions.into_iter())
    );
    AllowedTransitions(cur_transitions)
  }
}

// TODO: to generate this, move forward and backward "one step" for all states (all
// TokenPositionInProduction instances) -- meaning, find all the other TokenPositionInProduction
// instances reachable from this one with a single token transition, and as you reach into
// sub-productions, build up the appropriate `stack_changes`! This should be done in
// `generate_token_index()`, I think.
// TODO: IndexMap is only used here so we can compare the hashmap results -- for testing only --
// this should be fixed (?).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenIndex<Tok: Sized + PartialEq + Eq + Hash + Clone>(IndexMap<
    ConsecutiveTokenPair<Tok>, AllowedTransitions<Tok>>);

pub trait Tokenizeable<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  fn generate_token_index(&self) -> TokenIndex<Tok>;
}

impl <Tok: Sized + PartialEq + Eq + Hash + Clone> Tokenizeable<Tok>
  for SimultaneousProductions<Tok> {
    fn generate_token_index(&self) -> TokenIndex<Tok> {
      let other_self = self.clone();
      let states: Vec<TokenWithPosition<Tok>> = self.0.iter().flat_map(|(_, production)| {
        production.0.iter().flat_map(|case| {
          case.0.iter().enumerate().flat_map(|(element_index, element)| match element {
            CaseElement::Lit(literal) => {
              assert!(literal.0.len() > 0);
              literal.0.clone().into_iter().enumerate().flat_map(|(token_index, tok)| {
                let cur_new_state = TokenPositionInProduction {
                  productions_context: self.clone(),
                  case_context: case.clone(),
                  case_pos: element_index,
                  literal_context: literal.clone(),
                  literal_pos: token_index,
                };
                vec![TokenWithPosition {
                  tok: tok,
                  pos: cur_new_state,
                }]
              }).collect::<Vec<_>>()
            },
            CaseElement::Prod(prod_ref) => {
              // TODO: make this whole thing into a Result<_, _>!
              other_self.0.get(prod_ref).expect("prod ref not found");
              vec![]
            },
          }).collect::<Vec<_>>()
        }).collect::<Vec<_>>()
      }).collect::<Vec<_>>();
      // `states` has been populated -- let each TokenWithPosition take care of finding the
      // neighboring states for itself.
      let all_state_changes: IndexSet<StateChange<Tok>> = states.iter().flat_map(|token_with_position| {
        token_with_position.collect_backward_forward_transitions().0
      }).collect::<IndexSet<_>>();
      // TODO: this can probably be done immutably pretty easily?
      let mut pair_map: IndexMap<ConsecutiveTokenPair<Tok>, Vec<StateChange<Tok>>> =
        IndexMap::new();
      for state_change in all_state_changes.iter() {
        let left_tok = state_change.left_state.tok.clone();
        let right_tok = state_change.right_state.tok.clone();
        let consecutive_pair_key = ConsecutiveTokenPair { left_tok, right_tok };
        let pair_entry = pair_map.entry(consecutive_pair_key).or_insert(vec![]);
        (*pair_entry).push(state_change.clone());
      }
      TokenIndex(pair_map.into_iter().map(|(k, changes)| {
        let transitions_deduplicated = changes.iter().cloned().collect::<IndexSet<_>>();
        (k, AllowedTransitions(transitions_deduplicated))
      })
                 .collect::<IndexMap<_, _>>())
    }
  }

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn basic_productions() {
    let prods = SimultaneousProductions([
      (ProductionReference::new("a"), Production(vec![
        Case(vec![
          CaseElement::Lit(Literal::from("ab"))])])),
      (ProductionReference::new("b"), Production(vec![
        Case(vec![
          CaseElement::Lit(Literal::from("ab")),
          CaseElement::Prod(ProductionReference::new("a")),
        ])]))
    ].iter().cloned().collect());
    let token_index = prods.generate_token_index();
    // TODO: actually implement everything!
    assert_eq!(token_index, TokenIndex([
      (ConsecutiveTokenPair {
        left_tok: 'a',
        right_tok: 'b',
      }, AllowedTransitions(vec![
        StateChange {
          left_state: TokenWithPosition {
            tok: 'a',
            pos: TokenPositionInProduction {
              productions_context: prods.clone(),
              case_context: Case(vec![
                CaseElement::Lit(Literal(vec!['a', 'b']))]),
              case_pos: 0,
              literal_context: Literal(vec!['a', 'b']),
              literal_pos: 0,
            }
          },
          right_state: TokenWithPosition {
            tok: 'b',
            pos: TokenPositionInProduction {
              productions_context: prods.clone(),
              case_context: Case(vec![
                CaseElement::Lit(Literal(vec!['a', 'b']))]),
              case_pos: 0,
              literal_context: Literal(vec!['a', 'b']),
              literal_pos: 1,
            }
          },
          stack_changes: StackChangeSet(vec![]),
        },
        StateChange {
          left_state: TokenWithPosition {
            tok: 'a',
            pos: TokenPositionInProduction {
              productions_context: prods.clone(),
              case_context: Case(vec![
                CaseElement::Lit(Literal(vec!['a', 'b'])),
                CaseElement::Prod(ProductionReference::new("a")),
              ]),
              case_pos: 0,
              literal_context: Literal(vec!['a', 'b']),
              literal_pos: 0,
            }
          },
          right_state: TokenWithPosition {
            tok: 'b',
            pos: TokenPositionInProduction {
              productions_context: prods.clone(),
              case_context: Case(vec![
                CaseElement::Lit(Literal(vec!['a', 'b'])),
                CaseElement::Prod(ProductionReference::new("a")),
              ]),
              case_pos: 0,
              literal_context: Literal(vec!['a', 'b']),
              literal_pos: 1,
            }
          },
          stack_changes: StackChangeSet(vec![]),
        },
      ].iter().cloned().collect::<IndexSet<_>>()))
    ].iter().cloned().collect::<IndexMap<_, _>>()));
  }

  #[test]
  #[should_panic(expected = "prod ref not found")]
  fn missing_prod_ref() {
    let prods = SimultaneousProductions([
      (ProductionReference::new("b"), Production(vec![
        Case(vec![
          CaseElement::Lit(Literal::from("ab")),
          CaseElement::Prod(ProductionReference::new("c")),
        ])]))
    ].iter().cloned().collect());
    prods.generate_token_index();
  }
}
