#![feature(fn_traits)]

extern crate indexmap;

// TODO: indexmap here is used for testing purposes, so we can compare the results (see
// `basic_productions()`) -- figure out if there is a better way to do this.
use indexmap::{IndexMap, IndexSet};

use std::convert::From;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

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

///
/// Here comes the algorithm!
///
/// (I think this is a "model" graph class of some sort, where the model is this "simultaneous
/// productions" parsing formulation)
///
/// ImplicitRepresentation = [
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

// A version of `ProductionReference` which uses a `usize` for speed. We adopt the convention of
// abbreviated names for things used in algorithms.
// Points to a particular Production within an ImplicitRepresentation.
#[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
struct ProdRef(usize);

// Points to a particular case within a Production.
#[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
struct CaseRef(usize);

// Points to an element of a particular Case.
#[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
struct CaseElRef(usize);

// This refers to a specific token, implying that we must be pointing to a particular index of a
// particular Literal. This corresponds to a "state" in the simultaneous productions terminology.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct TokenPosition {
  prod: ProdRef,
  case: CaseRef,
  case_el: CaseElRef,
}

/// Graph Representation

// TODO: describe!
#[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ImplicitRepresentation(Vec<ProductionImpl>);

/// Mapping to Tokens

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenGrammar<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  graph: ImplicitRepresentation,
  tokens: Vec<Tok>,
}

impl <Tok: Sized + PartialEq + Eq + Hash + Clone>
  From<SimultaneousProductions<Tok>>
  for TokenGrammar<Tok> {
    fn from(prods: SimultaneousProductions<Tok>) -> Self {
      // Mapping from strings -> indices (TODO: from a type-indexed map, where each production
      // returns the type!).
      let prod_ref_mapping: HashMap<ProductionReference, usize> = prods.0.iter()
        .map(|(prod_ref, _)| prod_ref).cloned().enumerate().map(|(ind, p)| (p, ind)).collect();
      // Collect all the tokens (splitting up literals) as we traverse the productions.
      let mut all_tokens: IndexSet<Tok> = IndexSet::new();
      // Pretty straightforwardly map the productions into the new space.
      let new_prods: Vec<_> = prods.0.iter().map(|(_, prod)| {
        let cases: Vec<_> = prod.0.iter().map(|case| {
          let case_els: Vec<_> = case.0.iter().flat_map(|el| match el {
            CaseElement::Lit(literal) => {
              literal.0.iter().cloned().map(|cur_tok| {
                let (tok_ind, _) = all_tokens.insert_full(cur_tok);
                CaseEl::Tok(TokRef(tok_ind))
              }).collect::<Vec<_>>()
            },
            CaseElement::Prod(prod_ref) => {
              let prod_ref_ind = prod_ref_mapping.get(prod_ref).expect("prod ref not found");
              vec![CaseEl::Prod(ProdRef(*prod_ref_ind))]
            },
          }).collect();
          CaseImpl(case_els)
        }).collect();
        ProductionImpl(cases)
      }).collect();
      TokenGrammar {
        graph: ImplicitRepresentation(new_prods),
        tokens: all_tokens.iter().cloned().collect(),
      }
    }
  }

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct StackSym(ProdRef);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum StackStep {
  Positive(StackSym),
  Negative(StackSym),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StackDiff(Vec<StackStep>);

// TODO: consider the relationship between populating token transitions in the lookbehind cache to
// some specific depth (e.g. strings of 3, 4, 5 tokens) and SIMD type 1 instructions (my
// notations: meaning recognizing a specific sequence of tokens). SIMD type 2 (finding a specific
// token in a longer string of bytes) can already easily be used with just token pairs (and
// others).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StatePair {
  left: TokenPosition,
  right: TokenPosition,
}

// NB: There is no reference to any `TokenGrammar` -- this is intentional, and I believe makes it
// easier to have the runtime we want just fall out of the code without too much work.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PreprocessedGrammar<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  // These don't need to be quick to access or otherwise optimized for the algorithm until we create
  // a `Parse` -- these are chosen to reduce redundancy.
  states: IndexMap<Tok, Vec<TokenPosition>>,
  // TODO: we don't yet support stack cycles (ignored), or multiple stack paths to the same
  // succeeding state from an initial state (also ignored) -- details in index_tokens().
  transitions: IndexMap<StatePair, Vec<StackDiff>>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum GrammarVertex {
  State(TokenPosition),
  Prod(ProdRef),
  Epsilon,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct AnonSym(usize);

impl AnonSym {
  fn inc(&self) -> Self {
    AnonSym(self.0 + 1)
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum AnonStep {
  Positive(AnonSym),
  Negative(AnonSym),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum GrammarEdgeWeightStep {
  Named(StackStep),
  Anon(AnonStep),
}

impl GrammarEdgeWeightStep {
  fn is_anon(&self) -> bool {
    match self {
      &GrammarEdgeWeightStep::Anon(_) => true,
      _ => false,
    }
  }

  fn is_negative(&self) -> bool {
    match self {
      &GrammarEdgeWeightStep::Anon(AnonStep::Negative(_)) => true,
      &GrammarEdgeWeightStep::Named(StackStep::Negative(_)) => true,
      _ => false,
    }
  }

  fn is_negated_by(&self, rhs: &Self) -> bool {
    match self {
      GrammarEdgeWeightStep::Anon(AnonStep::Positive(cur_sym)) => match rhs {
        GrammarEdgeWeightStep::Anon(AnonStep::Negative(other_sym)) => cur_sym == other_sym,
        _ => false,
      },
      GrammarEdgeWeightStep::Anon(AnonStep::Negative(cur_sym)) => match rhs {
        GrammarEdgeWeightStep::Anon(AnonStep::Positive(other_sym)) => cur_sym == other_sym,
        _ => false,
      },
      GrammarEdgeWeightStep::Named(StackStep::Positive(cur_sym)) => match rhs {
        GrammarEdgeWeightStep::Named(StackStep::Negative(other_sym)) => cur_sym == other_sym,
        _ => false,
      },
      GrammarEdgeWeightStep::Named(StackStep::Negative(cur_sym)) => match rhs {
        GrammarEdgeWeightStep::Named(StackStep::Positive(other_sym)) => cur_sym == other_sym,
        _ => false,
      },
    }
  }
}

fn shuffle_negative_steps_returning_whether_matched(
  stack: &mut VecDeque<GrammarEdgeWeightStep>,
  edge_steps: &mut VecDeque<GrammarEdgeWeightStep>,
) -> bool {
  let mut match_has_failed = false;
  while edge_steps.front().map_or(false, |front_step| front_step.is_negative()) {
    let neg_step = edge_steps.pop_front().unwrap();
    if let Some(last_step) = stack.pop_back() {
      if !last_step.is_negated_by(&neg_step) {
        stack.push_back(last_step);
        match_has_failed = true;
        break;
      }
      // The stack symbols negate correctly.
      continue;
    } else {
      // The stack is currently empty, so just add everything to the end of it and finish. We check
      // that there are no anonymous steps left in the final diff elsewhere.
      stack.push_back(neg_step);
      // On a successful match, we push back the remaining steps anyway.
      break;
    }
  }
  !match_has_failed
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GrammarEdge {
  weight: Vec<GrammarEdgeWeightStep>,
  target: GrammarVertex,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GrammarTraversalState {
  traversal_target: GrammarVertex,
  prev_stack: VecDeque<GrammarEdgeWeightStep>,
}

impl <Tok: Sized + PartialEq + Eq + Hash + Clone> PreprocessedGrammar<Tok> {
  fn index_tokens(grammar: TokenGrammar<Tok>) -> Self {
    let mut toks_with_locs: IndexMap<Tok, Vec<TokenPosition>> = IndexMap::new();
    let mut neighbors: IndexMap<GrammarVertex, Vec<GrammarEdge>> = IndexMap::new();
    let mut cur_anon_sym = AnonSym(0);
    // Map all the tokens to states (`TokenPosition`s) which reference them, and build up a graph in
    // `neighbors` of the elements in each case.
    for (prod_ind, prod) in grammar.graph.0.iter().cloned().enumerate() {
      let prod_ref = ProdRef(prod_ind);
      for (case_ind, case) in prod.0.iter().cloned().enumerate() {
        let case_ref = CaseRef(case_ind);
        let mut is_start_of_case = true;
        let mut prev_vtx = GrammarVertex::Prod(prod_ref);
        let mut prev_anon_sym = cur_anon_sym;
        for (case_el_ind, case_el) in case.0.iter().cloned().enumerate() {
          let case_el_ref = CaseElRef(case_el_ind);
          // Make the appropriate weight steps to add to the current edge.
          let mut weight_steps: Vec<GrammarEdgeWeightStep> = match prev_vtx {
            GrammarVertex::Prod(_) => if !is_start_of_case {
              // Add a negative anonymous step, but only if we have just come from a ProdRef (and
              // not at the start of a case, only to cancel out a previous ProdRef case element).
              vec![GrammarEdgeWeightStep::Anon(AnonStep::Negative(prev_anon_sym))]
            } else {
              vec![]
            },
            _ => {
              assert!(!is_start_of_case,
                      "the start of a case should always have a ProdRef as prev_vtx");
              vec![]
            },
          };
          // Get a new anonymous symbol.
          cur_anon_sym = cur_anon_sym.inc();
          // Analyze the current case element.
          let cur_vtx = match case_el {
            CaseEl::Tok(tok_ref) => {
              let cur_tok = grammar.tokens.get(tok_ref.0).unwrap().clone();
              let tok_loc_entry = toks_with_locs.entry(cur_tok).or_insert(vec![]);
              let cur_pos = TokenPosition {
                prod: prod_ref,
                case: case_ref,
                case_el: case_el_ref,
              };
              (*tok_loc_entry).push(cur_pos);
              GrammarVertex::State(cur_pos)
            },
            CaseEl::Prod(cur_el_prod_ref) => {
              // Add a positive anonymous step, only if we are stepping onto a ProdRef.
              weight_steps.extend([
                GrammarEdgeWeightStep::Anon(AnonStep::Positive(cur_anon_sym)),
                // Also add a positive stack step (to be canceled out by a later step onto the
                // epsilon vertex from the final case element of a case of the target ProdRef).
                GrammarEdgeWeightStep::Named(StackStep::Positive(StackSym(cur_el_prod_ref))),
              ].iter().cloned());
              GrammarVertex::Prod(cur_el_prod_ref)
            },
          };
          // Add appropriate edges to neighbors for the current state.
          let edge = GrammarEdge {
            weight: weight_steps,
            target: cur_vtx,
          };
          let prev_neighborhood = neighbors.entry(prev_vtx).or_insert(vec![]);
          (*prev_neighborhood).push(edge);
          is_start_of_case = false;
          prev_vtx = cur_vtx;
          prev_anon_sym = cur_anon_sym;
        }
        // Add edge to epsilon vertex at end of case.
        let mut epsilon_edge_weight = match prev_vtx {
          // Only add a negative anonymous symbol if we are stepping off of a ProdRef (as above, for
          // in between case elements).
          GrammarVertex::Prod(_) => if !is_start_of_case {
            vec![GrammarEdgeWeightStep::Anon(AnonStep::Negative(cur_anon_sym))]
          } else {
            vec![]
          },
          _ => vec![],
        };
        epsilon_edge_weight.push(
          GrammarEdgeWeightStep::Named(StackStep::Negative(StackSym(prod_ref))));
        let epsilon_edge = GrammarEdge {
          weight: epsilon_edge_weight,
          target: GrammarVertex::Epsilon,
        };
        let final_case_el_neighborhood = neighbors.entry(prev_vtx).or_insert(vec![]);
        (*final_case_el_neighborhood).push(epsilon_edge);
      }
    }
    eprintln!("neighbors: {:?}", neighbors);

    // Crawl `neighbors` to get the `StackDiff` between each pair of states.
    // TODO: add support for stack cycles! Right now we just disallow stack cycles, but it's not
    // difficult to support this (in addition to iterating over all the known tokens, also iterate
    // over all the `ProdRef`s to find cycles (and record the stack diffs occuring during a single
    // iteration of the cycle), and in matching, remember that any instance of a cycling `ProdRef`
    // can have an indefinite number of iterations of its cycle)!
    let mut transitions: IndexMap<StatePair, Vec<StackDiff>> = IndexMap::new();
    for left_tok_pos in toks_with_locs.values().flatten() {
      eprintln!("left_tok_pos: {:?}", left_tok_pos);
      let mut queue: VecDeque<GrammarTraversalState> = VecDeque::new();
      // TODO: Store not just productions we've found, but the different paths taken to them (don't
      // try to do anything with cycles here, though)!
      let mut seen_productions: IndexSet<ProdRef> = IndexSet::new();
      queue.push_back(GrammarTraversalState {
        traversal_target: GrammarVertex::State(*left_tok_pos),
        prev_stack: VecDeque::new(),
      });
      while !queue.is_empty() {
        eprintln!("queue: {:?}", queue);
        let GrammarTraversalState {
          traversal_target,
          prev_stack,
        } = queue.pop_front().unwrap();
        for new_edge in neighbors.get(&traversal_target).unwrap().iter().cloned() {
          eprintln!("new_edge: {:?}", new_edge);
          let GrammarEdge {
            weight,
            target,
          } = new_edge;
          let mut new_stack = prev_stack.clone();
          eprintln!("new_stack: {:?}", new_stack);
          // I love move construction being the default. This is why Rust is the best language to
          // implement algorithms. I don't have to worry about accidental aliasing causing
          // correctness issues.
          let mut cur_edge_steps = VecDeque::from(weight);
          eprintln!("cur_edge_steps: {:?}", cur_edge_steps);
          // Match any negative elements at the beginning of the edge weights, or else fail.
          // NB: We assume that if there are negative stack steps, they are *only* located at the
          // beginning of the edge's weight! This is ok because we are generating the stack steps
          // ourselves earlier in this same method.
          if shuffle_negative_steps_returning_whether_matched(&mut new_stack, &mut cur_edge_steps) {
            new_stack.extend(cur_edge_steps.into_iter());
            eprintln!("next new_stack: {:?}", new_stack);
            eprintln!("target: {:?}", target);
            match target {
              // We have completed a traversal -- there should be no "anon" steps.
              GrammarVertex::State(right_tok_pos) => {
                let state_pair = StatePair {
                  left: *left_tok_pos,
                  right: right_tok_pos,
                };
                let mut diff: Vec<StackStep> = Vec::new();
                for el in new_stack.into_iter() {
                  match el {
                    GrammarEdgeWeightStep::Named(s) => diff.push(s),
                    _ => (),
                  }
                }
                eprintln!("diff: {:?}", diff);
                let transitions_for_pair = transitions.entry(state_pair).or_insert(vec![]);
                (*transitions_for_pair).push(StackDiff(diff));
              },
              // Add the target vertex with the new stack appended to the end (minus any initial
              // negatives).
              GrammarVertex::Prod(next_prod_ref) => {
                let (_, was_not_previously_seen) = seen_productions.insert_full(next_prod_ref);
                if was_not_previously_seen {
                  queue.push_back(GrammarTraversalState {
                    traversal_target: GrammarVertex::Prod(next_prod_ref),
                    prev_stack: new_stack,
                  });
                }
              },
              // The epsilon vertex has a zero-weight edge to all productions.
              GrammarVertex::Epsilon => {
                for next_prod_ind in 0..grammar.graph.0.len() {
                  queue.push_back(GrammarTraversalState {
                    traversal_target: GrammarVertex::Prod(ProdRef(next_prod_ind)),
                    prev_stack: new_stack.clone(),
                  });
                }
              },
            }
          }
        }
      }
    }
    PreprocessedGrammar {
      states: toks_with_locs,
      transitions,
    }
  }
}

/// Problem Instance

// May have to "reach into" the stack vec here at times when incrementally finding stack diffs to
// traverse to the next/prev token.
// TODO: this should probably be an index into the lookbehind cache!
#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenTransition<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  stack: Vec<StackStep>,
  // The previous token.
  // TODO: this may not be necessary in comparison to an index into the input, actually.
  tok: Tok,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AccumulatedTransitions<Tok: Sized + PartialEq + Eq + Hash + Clone>(
  VecDeque<TokenTransition<Tok>>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct KnownPathsToInputToken<Tok: Sized + PartialEq + Eq + Hash + Clone>(
  Vec<AccumulatedTransitions<Tok>>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct Parse<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  // TODO: this could be bound to an external lifetime -- but if it's not, that opens the
  // possibility of varying the grammar for different parses (which is something you might want to
  // do if you're /learning/ a grammar) (!!!!!!!!!?!).
  grammar: PreprocessedGrammar<Tok>,
  // NB: Don't worry too much about this right now. The grammar is idempotent -- the parse can be
  // too (without any modifications??!!??!!!!!!!).
  input: Vec<Tok>,
  // The term "state" is used very loosely here.
  // NB: same length as `input`!
  state: Vec<KnownPathsToInputToken<Tok>>,
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
    let grammar = TokenGrammar::from(prods);
    assert_eq!(grammar.clone(), TokenGrammar {
      tokens: vec!['a', 'b'],
      graph: ImplicitRepresentation(vec![
        ProductionImpl(vec![
          CaseImpl(vec![CaseEl::Tok(TokRef(0)), CaseEl::Tok(TokRef(1))]),
        ]),
        ProductionImpl(vec![
          CaseImpl(vec![
            CaseEl::Tok(TokRef(0)),
            CaseEl::Tok(TokRef(1)),
            CaseEl::Prod(ProdRef(0)),
          ]),
        ]),
      ]),
    });
    let preprocessed_grammar = PreprocessedGrammar::index_tokens(grammar);
    assert_eq!(preprocessed_grammar.clone(), PreprocessedGrammar {
      states: vec![
        ('a', vec![
          TokenPosition { prod: ProdRef(0), case: CaseRef(0), case_el: CaseElRef(0) },
          TokenPosition { prod: ProdRef(1), case: CaseRef(0), case_el: CaseElRef(0) },
        ]),
        ('b', vec![
          TokenPosition { prod: ProdRef(0), case: CaseRef(0), case_el: CaseElRef(1) },
          TokenPosition { prod: ProdRef(1), case: CaseRef(0), case_el: CaseElRef(1) },
        ]),
      ].iter().cloned().collect::<IndexMap<char, Vec<TokenPosition>>>(),
      transitions: vec![
        (StatePair {
          left: TokenPosition { prod: ProdRef(0), case: CaseRef(0), case_el: CaseElRef(0) },
          right: TokenPosition { prod: ProdRef(0), case: CaseRef(0), case_el: CaseElRef(1) },
        }, vec![StackDiff(vec![])]),
        (StatePair {
          left: TokenPosition { prod: ProdRef(1), case: CaseRef(0), case_el: CaseElRef(0) },
          right: TokenPosition { prod: ProdRef(1), case: CaseRef(0), case_el: CaseElRef(1) },
        }, vec![StackDiff(vec![])]),
        (StatePair {
          left: TokenPosition { prod: ProdRef(1), case: CaseRef(0), case_el: CaseElRef(1) },
          right: TokenPosition { prod: ProdRef(0), case: CaseRef(0), case_el: CaseElRef(0) },
        }, vec![StackDiff(vec![StackStep::Positive(StackSym(ProdRef(0)))])]),
        (StatePair {
          left: TokenPosition { prod: ProdRef(0), case: CaseRef(0), case_el: CaseElRef(1) },
          right: TokenPosition { prod: ProdRef(1), case: CaseRef(0), case_el: CaseElRef(0) },
        }, vec![StackDiff(vec![StackStep::Negative(StackSym(ProdRef(0)))])]),
        (StatePair {
          left: TokenPosition { prod: ProdRef(0), case: CaseRef(0), case_el: CaseElRef(1) },
          right: TokenPosition { prod: ProdRef(0), case: CaseRef(0), case_el: CaseElRef(0) },
        }, vec![StackDiff(vec![StackStep::Negative(StackSym(ProdRef(0)))])])
      ].iter().cloned().collect::<IndexMap<StatePair, Vec<StackDiff>>>(),
    });
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
    TokenGrammar::from(prods);
  }
}
