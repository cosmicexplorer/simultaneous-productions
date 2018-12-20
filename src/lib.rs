#![feature(fn_traits)]
// #![feature(trait_alias)]

use std::collections::HashMap;
use std::hash::Hash;

// TODO: trait aliases are not fully implemented!
// trait TokenBound = Sized + PartialEq + Eq + Hash;

// NB: this is to make it clear how the simultaneous productions technique can be parameterized by
// arbitrary binary data -- but we'll make it work with unicode `char`s first!
// trait Tokenizable<Tok: Sized + PartialEq + Eq + Hash> {
//   fn tokens(&self) -> Vec<Tok>;
// }

// trait Parseable<T> {
//   fn result(&self) ->
// }

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Literal<Tok: Sized + PartialEq + Eq + Hash>(Vec<Tok>);

// impl Literal {
//   fn from(s: String) -> Self {
//     Literal(s.as_str().chars().collect())
//   }
// }

// impl Tokenizable<char> for Literal {
//   fn tokens(&self) -> Vec<char> {
//     self.0.as_str().chars().collect()
//   }
// }

// A reference to another production -- the string must match the assigned name of a production in a
// set of simultaneous productions.
// TODO: make this do type-indexed maps and stuff!
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProductionReference(String);

#[derive(Debug, Clone)]
enum CaseElement<Tok: Sized + PartialEq + Eq + Hash> {
  Lit(Literal<Tok>),
  Prod(ProductionReference),
}

// TODO: figure out collect / return type / type-indexed mapping later!
// #[derive(Debug)]
// struct Case<ReturnType, ElementTypeTuple>;
// impl <ReturnType, ElementTypeTuple> Case<ReturnType, ElementTypeTuple> {
//   fn run(t: ElementTypeTuple) -> ReturnType {}
// }

#[derive(Debug, Clone)]
struct Case<Tok: Sized + PartialEq + Eq + Hash>(Vec<CaseElement<Tok>>);

// TODO: the Eq/Hash impls are going to be very expensive here! memoization is the right idea, or
// pairing it with some uuid?
#[derive(Debug, Clone)]
struct Production<Tok: Sized + PartialEq + Eq + Hash>(Vec<Case<Tok>>);

#[derive(Debug, Clone)]
struct SimultaneousProductions<Tok: Sized + PartialEq + Eq + Hash>(
  HashMap<ProductionReference, Production<Tok>>);

// This can be used to search backwards and/or forwards for allowed states during the execution of
// the partitioning algorithm!
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ConsecutiveTokenPair<Tok: Sized + PartialEq + Eq + Hash> {
  // TODO: consider a better naming scheme -- this is probably fine for now!
  left_tok: Tok,
  right_tok: Tok,
}

#[derive(Debug, Clone)]
struct TokenPositionInProduction<Tok: Sized + PartialEq + Eq + Hash> {
  case_context: Case<Tok>,
  case_pos: usize,
  literal_context: Literal<Tok>,
  literal_pos: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StackSymbol(ProductionReference);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum StackChangeUnit {
  Positive(StackSymbol),
  Negative(StackSymbol),
}

#[derive(Debug, Clone)]
struct TokenWithPosition<Tok: Sized + PartialEq + Eq + Hash> {
  tok: Tok,
  pos: TokenPositionInProduction<Tok>,
}

#[derive(Debug, Clone)]
struct StateChange<Tok: Sized + PartialEq + Eq + Hash> {
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
#[derive(Debug, Clone)]
struct AllowedTransitions<Tok: Sized + PartialEq + Eq + Hash>(Vec<StateChange<Tok>>);


impl <Tok: Sized + PartialEq + Eq + Hash> TokenWithPosition<Tok> {
  fn collect_backward_forward_transitions(&self) -> AllowedTransitions<Tok> {
    AllowedTransitions(Vec::new())
  }
}

// TODO: to generate this, move forward and backward "one step" for all states (all
// TokenPositionInProduction instances) -- meaning, find all the other TokenPositionInProduction
// instances reachable from this one with a single token transition, and as you reach into
// sub-productions, build up the appropriate `stack_changes`! This should be done in
// `generate_token_index()`, I think.
#[derive(Debug, Clone)]
struct PairwiseTransitionTable<Tok: Sized + PartialEq + Eq + Hash>(Vec<AllowedTransitions<Tok>>);

#[derive(Debug, Clone)]
struct TokenIndex<Tok: Sized + PartialEq + Eq + Hash>(HashMap<
    ConsecutiveTokenPair<Tok>, AllowedTransitions<Tok>>);

trait Tokenizeable<Tok: Sized + PartialEq + Eq + Hash> {
  fn generate_token_index(&self) -> TokenIndex<Tok>;
}

// NB: we add the `+ Clone` here so we can clone the vector -- is this necessary?
impl <Tok: Sized + PartialEq + Eq + Hash + Clone> Tokenizeable<Tok>
  for SimultaneousProductions<Tok> {
    fn generate_token_index(&self) -> TokenIndex<Tok> {
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
            CaseElement::Prod(_) => vec![],
          }).collect::<Vec<_>>()
        }).collect::<Vec<_>>()
      }).collect::<Vec<_>>();
      // `states` has been populated -- let each TokenWithPosition take care of finding the
      // neighboring states for itself.
      let all_state_changes: Vec<StateChange<Tok>> = states.iter().flat_map(|token_with_position| {
        token_with_position.collect_backward_forward_transitions().0
      }).collect::<Vec<_>>();
      // TODO: this can probably be done immutably pretty easily?
      let mut pair_map: HashMap<ConsecutiveTokenPair<Tok>, Vec<StateChange<Tok>>> =
        HashMap::new();
      for state_change in all_state_changes.iter() {
        let left_tok = state_change.left_state.tok.clone();
        let right_tok = state_change.right_state.tok.clone();
        let consecutive_pair_key = ConsecutiveTokenPair { left_tok, right_tok };
        let pair_entry = pair_map.entry(consecutive_pair_key).or_insert(vec![]);
        (*pair_entry).push(state_change.clone());
      }
      TokenIndex(pair_map.into_iter().map(|(k, changes)| (k, AllowedTransitions(changes.to_vec())))
                 .collect::<HashMap<_, _>>())
    }
  }

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
