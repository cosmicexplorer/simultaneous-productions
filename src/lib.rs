#![feature(fn_traits)]
// #![feature(trait_alias)]

extern crate indexmap;

// TODO: indexmap here is only used for testing purposes, so we can compare the results (see
// `basic_productions()`) -- figure out if there is a better way to do this.
use indexmap::IndexMap;

// use std::collections::HashMap;
use std::hash::Hash;

// TODO: trait aliases are not fully implemented!
// trait TokenBound = Sized + PartialEq + Eq + Hash + Clone;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Literal<Tok: Sized + PartialEq + Eq + Hash + Clone>(Vec<Tok>);

// A reference to another production -- the string must match the assigned name of a production in a
// set of simultaneous productions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProductionReference(String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaseElement<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  Lit(Literal<Tok>),
  Prod(ProductionReference),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Case<Tok: Sized + PartialEq + Eq + Hash + Clone>(Vec<CaseElement<Tok>>);

// TODO: the Eq/Hash impls are going to be very expensive here! memoization is the right idea, or
// pairing it with some uuid?
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Production<Tok: Sized + PartialEq + Eq + Hash + Clone>(Vec<Case<Tok>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimultaneousProductions<Tok: Sized + PartialEq + Eq + Hash + Clone>(
  IndexMap<ProductionReference, Production<Tok>>);

// This can be used to search backwards and/or forwards for allowed states during the execution of
// the partitioning algorithm!
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConsecutiveTokenPair<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  // TODO: consider a better naming scheme -- this is probably fine for now!
  left_tok: Tok,
  right_tok: Tok,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenPositionInProduction<Tok: Sized + PartialEq + Eq + Hash + Clone> {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenWithPosition<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  tok: Tok,
  pos: TokenPositionInProduction<Tok>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateChange<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  // TODO: should we include the `left`/`right` tokens here as well? It's redundant, but it may help
  // during debugging of the algorithm / help avoid mapping state changes to incorrect token
  // pairs. For now, we will separate these.
  left_state: TokenWithPosition<Tok>,
  right_state: TokenWithPosition<Tok>,
  // NB: when going backwards, this should be reversed!
  stack_changes: Vec<StackChangeUnit>,
}

// This contains all of the possible state changes that can result from moving from one token to
// another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowedTransitions<Tok: Sized + PartialEq + Eq + Hash + Clone>(Vec<StateChange<Tok>>);

impl <Tok: Sized + PartialEq + Eq + Hash + Clone> TokenWithPosition<Tok> {
  fn collect_backward_forward_transitions(&self) -> AllowedTransitions<Tok> {
    let mut cur_transitions: Vec<StateChange<Tok>> = Vec::new();
    // forward literals
    if self.pos.literal_pos < (self.pos.literal_context.0.len() - 1) {
      let next_index = self.pos.literal_pos + 1;
      let next_token = self.pos.literal_context.0.get(next_index).unwrap().clone();
      let next_pos = TokenPositionInProduction {
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
        stack_changes: vec![],
      };
      cur_transitions.push(next_state_change);
    } else {
      println!("forward past end of literal (not implemented yet)!");
    }
    // backward literals
    if self.pos.literal_pos > 0 {
      let prev_index = self.pos.literal_pos - 1;
      let prev_token = self.pos.literal_context.0.get(prev_index).unwrap().clone();
      let prev_pos = TokenPositionInProduction {
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
        stack_changes: vec![],
      };
      cur_transitions.push(prev_state_change);
    } else {
      println!("backward past start of literal (not implemented yet)!");
    }
    // panic!("not implemented yet!");
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
              literal.0.clone().into_iter().enumerate().flat_map(|(token_index, tok)| {
                let cur_new_state = TokenPositionInProduction {
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
              other_self.0.get(prod_ref).expect("prod ref not found");
              vec![]
            },
          }).collect::<Vec<_>>()
        }).collect::<Vec<_>>()
      }).collect::<Vec<_>>();
      // `states` has been populated -- let each TokenWithPosition take care of finding the
      // neighboring states for itself.
      let all_state_changes: Vec<StateChange<Tok>> = states.iter().flat_map(|token_with_position| {
        token_with_position.collect_backward_forward_transitions().0
      }).collect::<Vec<_>>();
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
      TokenIndex(pair_map.into_iter().map(|(k, changes)| (k, AllowedTransitions(changes.to_vec())))
                 .collect::<IndexMap<_, _>>())
    }
  }

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn basic_productions() {
    let prods = SimultaneousProductions([
      (ProductionReference("a".to_string()), Production(vec![
        Case(vec![
          CaseElement::Lit(Literal(vec!['a', 'b']))])])),
      (ProductionReference("b".to_string()), Production(vec![
        Case(vec![
          CaseElement::Lit(Literal(vec!['a', 'b'])),
          CaseElement::Prod(ProductionReference("a".to_string())),
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
              case_context: Case(vec![
                CaseElement::Lit(Literal(vec!['a', 'b']))]),
              case_pos: 0,
              literal_context: Literal(vec!['a', 'b']),
              literal_pos: 1,
            }
          },
          stack_changes: vec![],
        },
        StateChange {
          left_state: TokenWithPosition {
            tok: 'a',
            pos: TokenPositionInProduction {
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
              case_context: Case(vec![
                CaseElement::Lit(Literal(vec!['a', 'b']))]),
              case_pos: 0,
              literal_context: Literal(vec!['a', 'b']),
              literal_pos: 1,
            }
          },
          stack_changes: vec![],
        },
        StateChange {
          left_state: TokenWithPosition {
            tok: 'a',
            pos: TokenPositionInProduction {
              case_context: Case(vec![
                CaseElement::Lit(Literal(vec!['a', 'b'])),
                CaseElement::Prod(ProductionReference("a".to_string())),
              ]),
              case_pos: 0,
              literal_context: Literal(vec!['a', 'b']),
              literal_pos: 0,
            }
          },
          right_state: TokenWithPosition {
            tok: 'b',
            pos: TokenPositionInProduction {
              case_context: Case(vec![
                CaseElement::Lit(Literal(vec!['a', 'b'])),
                CaseElement::Prod(ProductionReference("a".to_string())),
              ]),
              case_pos: 0,
              literal_context: Literal(vec!['a', 'b']),
              literal_pos: 1,
            }
          },
          stack_changes: vec![],
        },
        StateChange {
          left_state: TokenWithPosition {
            tok: 'a',
            pos: TokenPositionInProduction {
              case_context: Case(vec![
                CaseElement::Lit(Literal(vec!['a', 'b'])),
                CaseElement::Prod(ProductionReference("a".to_string())),
              ]),
              case_pos: 0,
              literal_context: Literal(vec!['a', 'b']),
              literal_pos: 0,
            }
          },
          right_state: TokenWithPosition {
            tok: 'b',
            pos: TokenPositionInProduction {
              case_context: Case(vec![
                CaseElement::Lit(Literal(vec!['a', 'b'])),
                CaseElement::Prod(ProductionReference("a".to_string())),
              ]),
              case_pos: 0,
              literal_context: Literal(vec!['a', 'b']),
              literal_pos: 1,
            }
          },
          stack_changes: vec![],
        },
      ]))
    ].iter().cloned().collect::<IndexMap<_, _>>()));
  }

  #[test]
  #[should_panic(expected = "prod ref not found")]
  fn missing_prod_ref() {
    let prods = SimultaneousProductions([
      (ProductionReference("b".to_string()), Production(vec![
        Case(vec![
          CaseElement::Lit(Literal(vec!['a', 'b'])),
          CaseElement::Prod(ProductionReference("c".to_string())),
        ])]))
    ].iter().cloned().collect());
    prods.generate_token_index();
  }
}
