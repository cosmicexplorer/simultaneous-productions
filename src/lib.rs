#![feature(fn_traits)]

extern crate indexmap;

// TODO: indexmap here is used for testing purposes, so we can compare the results (see
// `basic_productions()`) -- figure out if there is a better way to do this.
use indexmap::{IndexMap, IndexSet};

use std::convert::From;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;

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
#[derive(Debug, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
struct ProdRef(usize);

// Points to a particular case within a Production.
#[derive(Debug, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
struct CaseRef(usize);

// Points to an element of a particular Case.
#[derive(Debug, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
struct CaseElRef(usize);

// This refers to a specific token, implying that we must be pointing to a particular index of a
// particular Literal. This corresponds to a "state" in the simultaneous productions terminology.
#[derive(Debug, Copy, PartialEq, Eq, Hash)]
struct TokenPosition {
  prod: ProdRef,
  case: CaseRef,
  case_el: CaseElRef,
}

/// Graph Representation

// TODO: describe!
#[derive(Debug, Copy, PartialOrd, Ord, PartialEq, Eq)]
struct TokRef(usize);

#[derive(Debug, Copy, PartialEq, Eq)]
enum CaseEl {
  Tok(TokRef),
  Prod(ProdRef),
};

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
      let prod_ref_mapping: HashMap<ProductionReference, usize> = prods.0.iter().cloned()
        .map(|(prod_ref, _)| prod_ref).enumerate(|(ind, p)| (p, ind)).collect();
      // Collect all the tokens (splitting up literals) as we traverse the productions.
      let mut all_tokens: IndexSet<Tok> = IndexSet::new();
      // Pretty straightforwardly map the productions into the new space.
      let new_prods: Vec<_> = prods.0.iter().map(|(_, prod)| {
        prod.0.iter().map(|(_, case)| {
          case.0.iter().flat_map(|(_, el)| match el {
            CaseElement::Lit(literal) => {
              literal.0.iter().map(|cur_tok| {
                let (tok_ind, _) = all_tokens.insert_full(cur_tok);
                CaseEl::Tok(TokRef(tok_ind))
              }).collect::<Vec<_>>()
            },
            CaseElement::Prod(prod_ref) => {
              let prod_ref_ind = prod_ref_mapping.get(prod_ref).unwrap();
              vec![CaseEl::Prod(ProdRef(prod_ref_ind))]
            },
          }).collect::<Vec<_>>()
        }).collect::<Vec<_>>()
      }).collect();
      TokenGrammar {
        graph: ImplicitRepresentation(new_prods),
        tokens: all_tokens.iter().collect(),
      }
    }
  }

#[derive(Debug, Copy, PartialEq, Eq)]
struct StackSym(ProdRef);

#[derive(Debug, Copy, PartialEq, Eq)]
enum StackStep {
  Positive(StackSym),
  Negative(StackSym),
}

// This refers to the position of a particular ProdRef within the grammar (its location within a
// specific Case).
#[derive(Debug, Copy, PartialEq, Eq)]
struct ProdRefPosition {
  prod: ProdRef,
  case: CaseRef,
  case_el: CaseElRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StackDiff {
  Empty,
  Single(Vec<StackStep>),
  Cycle(Vec<StackStep>),
}

// TODO: Some merge method for parallel parses!
// TODO: consider the relationship between populating token transitions in the lookbehind cache to
// some specific depth (e.g. strings of 3, 4, 5 tokens) and SIMD type 1 instructions (my
// notations: meaning recognizing a specific sequence of tokens). SIMD type 2 (finding a specific
// token in a longer string of bytes) can already easily be used with just token pairs (and
// others).
#[derive(Debug, Clone, PartialEq, Eq)]
struct KnownStateTraversals(Vec<StackDiff>);

#[derive(Debug, Clone, PartialEq, Eq)]
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
  transitions: IndexMap<StatePair, KnownStateTraversals>,
}

#[derive(Debug, Copy, PartialEq, Eq, Hash)]
enum GrammarVertex {
  State(TokenPosition),
  Prod(ProdRef),
  Epsilon,
}

#[derive(Debug, Copy, PartialEq, Eq)]
struct AnonSym(usize);

impl AnonSym {
  fn inc(&self) -> Self {
    AnonSym(self.0 + 1)
  }
}

#[derive(Debug, Copy, PartialEq, Eq)]
enum AnonStep {
  Positive(AnonSym),
  Negative(AnonSym),
}

#[derive(Debug, Copy, PartialEq, Eq)]
enum GrammarEdgeWeightStep {
  Named(StackStep),
  Anon(AnonStep),
}

impl GrammarEdgeWeightStep {
  fn negates(&self, rhs: &Self) -> bool {
    match self {
      Self::Anon(this_step) => match this_step {
        AnonStep::Positive(this_pos_sym) => match rhs {
          Self::Anon(AnonStep::Negative(other_neg_sym)) => this_pos_sym == other_neg_sym,
          _ => false,
        },
        AnonStep::Negative(this_neg_sym) => match rhs {
          Self::Anon()
        }
      }
      Self::Anon(AnonStep::Positive(this_sym)) => match rhs {
        Self::Anon(AnonStep::Negative(other_sym)) => this_sym == other_sym,
        _ => false,
      },
      Self::Named(StackStep::Positive(this_sym)) =>
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GrammarEdge {
  weight: Vec<GrammarEdgeWeightStep>,
  target: GrammarVertex,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GrammarTraversalState {
  target: GrammarVertex,
  prev_stack: VecDeque<GrammarEdgeWeightStep>,
}

impl GrammarTraversalState {
  fn advance(&self, neighbors: &IndexMap<GrammarVertex, Vec<GrammarEdge>>) -> Vec<Self> {
    let edges = neighbors.get(self.target).unwrap().clone();
    edges.iter().flat_map(|edge| {
      let cur_stack = self.prev_stack.clone();
      for edge_step in edge.weight.into_iter() {
        match edge_step {
          GrammarEdgeWeightStep::Anon()
        }
      }
    }).collect::<Vec<_>>()
  }
}

impl <Tok: Sized + PartialEq + Eq + Hash + Clone> PreprocessedGrammar<Tok> {
  fn index_tokens(grammar: TokenGrammar<Tok>) -> Self {
    let mut toks_with_locs: IndexMap<Tok, Vec<TokenPosition>> = IndexMap::new();
    let mut neighbors: IndexMap<GrammarVertex, Vec<GrammarEdge>> = IndexMap::new();
    let mut cur_anon_sym = AnonSym(0);
    grammar.graph.0.iter().cloned().enumerate(|(prod_ind, prod)| {
      let prod_ref = ProdRef(prod_ind);
      prod.0.iter().cloned().enumerate(|(case_ind, case)| {
        let case_ref = CaseRef(case_ind);
        let mut is_start_of_case = true;
        let mut prev_vtx = GrammarVertex::Prod(prod_ref);
        case.0.iter().cloned().enumerate(|(case_el_ind, case_el)| {
          let case_el_ref = CaseElRef(case_el_ind);
          let cur_pos = TokenPosition {
            prod: prod_ref,
            case: case_ref,
            case_el: case_el_ref,
          };
          // Make the appropriate weight steps to add to the current edge.
          let mut weight_steps: Vec<GrammarEdgeWeightStep> = Vec::new();
          if !is_start_of_case {
            weight_steps.push(GrammarEdgeWeightStep::Anon(AnonStep::Negative(cur_anon_sym)));
          }
          // Get a new anonymous symbol.
          cur_anon_sym = cur_anon_sym.inc();
          weight_steps.push(GrammarEdgeWeightStep::Anon(AnonStep::Positive(cur_anon_sym)));
          if is_start_of_case {
            weight_steps.push(
              GrammarEdgeWeightStep::Named(StackStep::Positive(StackSym(prod_ref))));
          }
          is_start_of_case = false;
          // Analyze the current case element.
          let cur_vtx = match case_el {
            CaseEl::Tok(tok_ref) => {
              let cur_tok = grammar.tokens.get(tok_ref.0).unwrap().clone();
              let tok_loc_entry = toks_with_locs.entry(tok).or_insert(vec![]);
              (*tok_loc_entry).push(cur_pos);
              GrammarVertex::State(cur_pos)
            },
            CaseEl::Prod(cur_el_prod_ref) => GrammarVertex::Prod(cur_el_prod_ref),
          };
          // Add appropriate edges to neighbors for the current state.
          let edge = GrammarEdge {
            weight: weight_steps,
            target: cur_vtx,
          };
          let prev_neighborhood = neighbors.entry(prev_vtx).or_insert(vec![]);
          (*prev_neighborhood).push(edge);
          prev_vtx = cur_vtx;
        });
        // Add edge to epsilon vertex at end of case.
        let epsilon_edge = GrammarEdge {
          weight: vec![
            GrammarEdgeWeightStep::Anon(AnonStep::Negative(cur_anon_sym)),
            GrammarEdgeWeightStep::Named(StackStep::Negative(StackSym(prod_ref))),
          ],
          target: GrammarVertex::Epsilon,
        };
        let final_case_el_neighborhood = neighbors.entry(prev_vtx).or_insert(vec![]);
        (*final_case_el_neighborhood).push(epsilon_edge);
      });
    });
    // TODO: add Epsilon vertex with zero-weight edges to everything (or do it implicitly?)!
    // Crawl `neighbors` to get the `KnownStateTraversals` for each state.
    let mut transitions: IndexMap<StatePair, KnownStateTraversals> = IndexMap::new();
    for tok_pos in toks_with_locs.values().flatten() {
      let mut queue: VecDeque<GrammarTraversalState> = VecDeque::new();
      queue.push(GrammarTraversalState {
        target: GrammarVertex::State(tok_pos),
        prev_stack: VecDeque::new(),
      });
      while !queue.is_empty() {
        
      }
    }
    PreprocessedGrammar {
      states: toks_with_locs,
      transitions: ???,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct UnionFind<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  parse: Parse<Tok>,
  lookbehind_cache: IndexMap<ConsecutiveTokenPair<Tok>, KnownStateTraversals>,
}

impl <Tok: Sized + PartialEq + Eq + Hash + Clone> UnionFind<Tok> {
  fn init(grammar: PreprocessedGrammar<Tok>, input: Vec<Tok>) -> Self {
    // TODO: initialize all the `completed_traversals` and some `intermediate_traversals` (do a
    // depth-1 BFS forward and backward from each token kinda like we do below, but in a single
    // pass)! (and do this as part of PreprocessedGrammar<Tok>, honestly)
    let mut per_token_init_state: Vec<KnownPathsToInputToken<Tok>> = Vec::with_capacity(input.len());
    for 0..input.len() {
      per_token_init_state.push(KnownPathsToInputToken(vec![]));
    }
    let parse = Parse {
      grammar,
      input,
      state: per_token_init_state,
    };
    UnionFind {
      parse,
      lookbehind_cache: IndexMap::new(),
    }
  }

  // TODO: as we advance the parse state, we can independently advance the lookbehind cache state!
  // (!!!!!!)
  // TODO: some way to know when it's done (ish)!
  fn iterate(&mut self) {
    // TODO: pick a consecutive pair of indices in the input!
    // TODO: see if the lookbehind cache has any candidates for matching the two tokens!
    // TODO: if not, see if the lookbehind cache has any intermediate positions!
  }
}

/// Old Stuff

// NB: This is the "state".
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TokenPositionInProduction<Tok: Sized + PartialEq + Eq + Hash + Clone> {
  productions_context: SimultaneousProductions<Tok>,
  production_pos: ProductionReference,
  case_context: Case<Tok>,
  case_pos: usize,
  literal_context: Literal<Tok>,
  literal_pos: usize,
}

impl <Tok: Sized + PartialEq + Eq + Hash + Clone> TokenPositionInProduction<Tok> {
  fn with_new_literal_index(&self, new_index: usize) -> Self {
    assert!(0 <= new_index <= self.literal_context.0.len());
    TokenPositionInProduction {
      productions_context: self.productions_context.clone(),
      case_context: self.case_context.clone(),
      case_pos: self.case_pos,
      literal_context: self.literal_context.clone(),
      literal_pos: new_index,
    }
  }

  // We are always within some literal, because we are always on a token, which is always associated
  // with a specific position within a specific literal.
  fn forward_within_literal(&self) -> Option<StateChange<Tok>> {
    let next_index = self.literal_pos + 1;
    self.literal_context.0.get(next_index).map(|next_token| {
      let next_pos = self.with_new_literal_index(next_index);
      StateChange {
        left_state: self.clone(),
        right_state: next_pos,
        stack_changes: AccumulatedTransitions::empty(),
      }
    })
  }

  fn backward_within_literal(&self) -> Option<StateChange<Tok>> {
    assert!(self.literal_pos >= 0);
    if self.literal_pos == 0 {
      None
    } else {
      let prev_index = self.literal_pos - 1;
      let prev_token = self.literal_context.0.get(prev_index).unwrap();
      let prev_pos = self.with_new_literal_index(prev_index);
      Some(StateChange {
        left_state: prev_pos,
        right_state: self.clone(),
        stack_changes: AccumulatedTransitions::empty(),
      })
    }
  }

  fn forward_within_case(&self) ->
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StackSymbol(ProductionReference);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StackChangeUnit {
  Positive(StackSymbol),
  Negative(StackSymbol),
}

// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
// pub struct TokenWithPosition<Tok: Sized + PartialEq + Eq + Hash + Clone> {
//   tok: Tok,
//   pos: TokenPositionInProduction<Tok>,
// }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccumulatedTransitions(Vec<StackChangeUnit>);

impl AccumulatedTransitions {
  fn empty() -> Self {
    AccumulatedTransitions(vec![])
  }

  fn init(st: StackChangeUnit) -> Self {
    AccumulatedTransitions(vec![st])
  }

  fn append(&self, st: StackChangeUnit) -> Self {
    AccumulatedTransitions(self.0.iter().cloned().chain(vec![st].into_iter()).collect())
  }

  fn reversed(&self) -> Self {
    let rev_changes = self.0.clone().into_iter().rev().collect::<Vec<_>>();
    AccumulatedTransitions(rev_changes)
  }
}

// The specific ordering of the vector is meaningful, so this Hash impl makes sense.
impl Hash for AccumulatedTransitions {
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
  left_state: TokenPositionInProduction<Tok>,
  right_state: TokenPositionInProduction<Tok>,
  // NB: when going backwards, this should be reversed!
  stack_changes: AccumulatedTransitions,
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
        stack_changes: AccumulatedTransitions::empty(),
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
            stack_changes: AccumulatedTransitions::empty(),
          };
          vec![next_case_el_state_change]
        },
        CaseElement::Prod(next_prod_ref) => {
          // We check that all production refs exist in generate_token_index()!
          let next_prod = self.pos.productions_context.0.get(&next_prod_ref.clone()).unwrap().clone();
          // Find all reachable states that are one token transition away -- a BFS with depth 1.
          let mut reachable_cases: Vec<(Case<Tok>, AccumulatedTransitions)> =
            next_prod.0.iter().cloned().map(|case| {
              let new_stack_sym = StackSymbol(next_prod_ref.clone());
              (case, AccumulatedTransitions::init(StackChangeUnit::Positive(new_stack_sym)))
            }).collect();
          let mut one_step_state_changes: IndexSet<StateChange<Tok>> = IndexSet::new();
          while !reachable_cases.is_empty() {
            let new_reachable_cases = reachable_cases.drain(..).flat_map(|(cur_case, cur_stack_changes)| {
              // We assert!(case.0.len() > 0) in generate_token_index()!
              // TODO: we commented that out -- we need to be able to support empty cases! This may
              // also require some munging of the data model.
              let new_next_stepped_case_el_init_pos = 0;
              match cur_case.0.get(new_next_stepped_case_el_init_pos).unwrap().clone() {
                CaseElement::Lit(next_literal) => {
                  // We assert!(next_literal.0.len() > 0) in generate_token_index()!
                  let new_next_stepped_literal_init_pos = 0;
                  let stepped_to_pos = TokenPositionInProduction {
                    productions_context: self.pos.productions_context.clone(),
                    case_context: cur_case.clone(),
                    case_pos: new_next_stepped_case_el_init_pos,
                    literal_context: next_literal.clone(),
                    literal_pos: new_next_stepped_literal_init_pos,
                  };
                  let stepped_to_state = TokenWithPosition {
                    tok: next_literal.0.get(new_next_stepped_literal_init_pos).unwrap().clone(),
                    pos: stepped_to_pos,
                  };
                  let stepped_state_change = StateChange {
                    left_state: self.clone(),
                    right_state: stepped_to_state,
                    stack_changes: cur_stack_changes,
                  };
                  one_step_state_changes.insert(stepped_state_change);
                  vec![]
                },
                CaseElement::Prod(further_next_prod_ref) => {
                  let further_next_prod = self.pos.productions_context.0
                    .get(&further_next_prod_ref.clone()).unwrap().clone();
                  // TODO: this can cause an infinite loop on normal inputs unless we can reasonably
                  // extend `AccumulatedTransitions` to hold a cycle so that it can be resumed during
                  // parsing.
                  // NB: The above might be the "pathological" case that causes the algorithm to be
                  // nonlinear -- BUT ONLY IF WE LET THAT HAPPEN (and don't e.g. amortize!).
                  further_next_prod.0.into_iter().map(|case| {
                    let new_stack_sym = StackSymbol(further_next_prod_ref.clone());
                    (case, cur_stack_changes.append(StackChangeUnit::Positive(new_stack_sym)))
                  }).collect::<Vec<_>>()
                },
              }
            }).collect::<Vec<_>>();
            reachable_cases.extend(new_reachable_cases.into_iter());
          }
          one_step_state_changes.into_iter().collect::<Vec<_>>()
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
        stack_changes: AccumulatedTransitions::empty(),
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
            stack_changes: AccumulatedTransitions::empty(),
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
      let states: Vec<TokenPositionInProduction<Tok>> = self.0.iter().flat_map(|(_, production)| {
        assert!(production.0.len() > 0);
        production.0.iter().flat_map(|case| {
          // assert!(case.0.len() > 0);
          case.0.iter().enumerate().flat_map(|(element_index, element)| match element {
            CaseElement::Lit(literal) => {
              assert!(literal.0.len() > 0);
              literal.0.clone().into_iter().enumerate().flat_map(|(token_index, tok)| {
                vec![TokenPositionInProduction {
                  productions_context: self.clone(),
                  case_context: case.clone(),
                  case_pos: element_index,
                  literal_context: literal.clone(),
                  literal_pos: token_index,
                }];
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
      // `states` has been populated -- let each TokenPositionInProduction take care of finding the
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
          stack_changes: AccumulatedTransitions::empty(),
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
          stack_changes: AccumulatedTransitions::empty(),
        },
      ].iter().cloned().collect::<IndexSet<_>>())),
      (ConsecutiveTokenPair {
        left_tok: 'b',
        right_tok: 'a',
      }, AllowedTransitions(vec![
        StateChange {
          left_state: TokenWithPosition {
            tok: 'b',
            pos: TokenPositionInProduction {
              productions_context: prods.clone(),
              case_context: Case(vec![
                CaseElement::Lit(Literal(vec!['a', 'b'])),
                CaseElement::Prod(ProductionReference::new("a"))]),
              case_pos: 0,
              literal_context: Literal(vec!['a', 'b']),
              literal_pos: 1,
            }
          },
          right_state: TokenWithPosition {
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
          stack_changes: AccumulatedTransitions(vec![
            StackChangeUnit::Positive(
              StackSymbol(ProductionReference::new("a"))),
          ]),
        },
      ].iter().cloned().collect::<IndexSet<_>>())),
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
