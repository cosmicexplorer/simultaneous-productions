#![feature(fn_traits)]

extern crate indexmap;

// TODO: indexmap here is used for testing purposes, so we can compare the results (see
// `non_cyclic_productions()`) -- figure out if there is a better way to do this.
use indexmap::{IndexMap, IndexSet};

use std::convert::From;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Literal<Tok: Sized + PartialEq + Eq + Hash + Copy + Clone>(Vec<Tok>);

// NB: a From impl is usually intended to denote that allocation is /not/ performed, I think: see
// https://doc.rust-lang.org/std/convert/trait.From.html -- fn new() makes more sense for this use
// case.
impl Literal<char> {
  fn new(s: &str) -> Self {
    Literal(s.chars().collect())
  }
}

// A reference to another production -- the string must match the assigned name of a production in a
// set of simultaneous productions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProductionReference(String);

impl ProductionReference {
  fn new(s: &str) -> Self {
    ProductionReference(s.to_string())
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CaseElement<Tok: Sized + PartialEq + Eq + Hash + Copy + Clone> {
  Lit(Literal<Tok>),
  Prod(ProductionReference),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Case<Tok: Sized + PartialEq + Eq + Hash + Copy + Clone>(Vec<CaseElement<Tok>>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Production<Tok: Sized + PartialEq + Eq + Hash + Copy + Clone>(Vec<Case<Tok>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimultaneousProductions<Tok: Sized + PartialEq + Eq + Hash + Copy + Clone>(
  IndexMap<ProductionReference, Production<Tok>>);

///
/// Here comes the algorithm!
///
/// (I think this is a "model" graph class of some sort, where the model is this "simultaneous
/// productions" parsing formulation)
///
/// LoweredProductions = [
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
// TODO: see if auto-implementing Clone (as well as Copy) means Clone is used over copy and if
// that's somehow slower???

// A version of `ProductionReference` which uses a `usize` for speed. We adopt the convention of
// abbreviated names for things used in algorithms.
// Points to a particular Production within a LoweredProductions.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct ProdRef(usize);

// Points to a particular case within a Production.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct CaseRef(usize);

// Points to an element of a particular Case.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct LoweredProductions(Vec<ProductionImpl>);

/// Mapping to Tokens

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenGrammar<Tok: Sized + PartialEq + Eq + Hash + Copy + Clone> {
  graph: LoweredProductions,
  tokens: Vec<Tok>,
}

impl <Tok: Sized + PartialEq + Eq + Hash + Copy + Clone> TokenGrammar<Tok> {
  fn new(prods: &SimultaneousProductions<Tok>) -> Self {
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
            let prod_ref_ind = prod_ref_mapping.get(prod_ref)
              .expect(&format!("prod ref {:?} not found", prod_ref));
            vec![CaseEl::Prod(ProdRef(*prod_ref_ind))]
          },
        }).collect();
        CaseImpl(case_els)
      }).collect();
      ProductionImpl(cases)
    }).collect();
    TokenGrammar {
      graph: LoweredProductions(new_prods),
      tokens: all_tokens.iter().cloned().collect(),
    }
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct StackSym(ProdRef);

// NB: I can't BELIEVE rust can auto-derive PartialOrd and Ord for structs and enums! Note that this
// would order all of the positive stack symbols first, and /then/ moves on to the negative symbols.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum StackStep {
  Positive(StackSym),
  Negative(StackSym),
}

// Automatically deriving Ord for vectors is incredible.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StackDiff(Vec<StackStep>);

impl StackDiff {
  fn sequence(&self, other: &Self) -> Self {
    // TODO: are there perf concerns with .iter().cloned() vs .clone().iter()?
    let combined: Vec<StackStep> = self.0.iter().chain(other.0.iter()).cloned().collect();
    StackDiff(combined)
  }
}

// TODO: consider the relationship between populating token transitions in the lookbehind cache to
// some specific depth (e.g. strings of 3, 4, 5 tokens) and SIMD type 1 instructions (my notations:
// meaning recognizing a specific contiguous sequence of tokens (bytes)). SIMD type 2 (finding a
// specific token in a longer string of bytes) can already easily be used with just token pairs (and
// others).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StatePair {
  left: TokenPosition,
  right: TokenPosition,
}

// NB: There is no reference to any `TokenGrammar` -- this is intentional, and I believe makes it
// easier to have the runtime we want just fall out of the code without too much work.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PreprocessedGrammar<Tok: Sized + PartialEq + Eq + Hash + Copy + Clone> {
  // These don't need to be quick to access or otherwise optimized for the algorithm until we create
  // a `Parse` -- these are chosen to reduce redundancy.
  states: IndexMap<Tok, Vec<TokenPosition>>,
  // TODO: we don't yet support stack cycles (ignored), or multiple stack paths to the same
  // succeeding state from an initial state (also ignored) -- details in
  // build_pairwise_transitions_table().
  transitions: IndexMap<StatePair, Vec<StackDiff>>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum GrammarVertex {
  State(TokenPosition),
  Prod(ProdRef),
  Epsilon,
}

// "anonymous" stack symbols are only used in constructing a `PreprocessedGrammar`.
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

// I love move construction being the default. This is why Rust is the best language to
// implement algorithms. I don't have to worry about accidental aliasing causing
// correctness issues.
fn shuffling_negative_steps_successfully_matched(
  stack: &mut VecDeque<GrammarEdgeWeightStep>,
  mut edge_steps: VecDeque<GrammarEdgeWeightStep>,
) -> bool {
  eprintln!("stack: {:?}", stack);
  eprintln!("edge_steps: {:?}", edge_steps);
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
  // We have now moved all of the edge steps into the stack.
  // NB: This *should* be fine to do even on a failed match, but watch this carefully!
  stack.extend(edge_steps.into_iter());
  eprintln!("new stack: {:?}", stack);
  eprintln!("match_has_failed: {:?}", match_has_failed);
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

impl <Tok: Sized + PartialEq + Eq + Hash + Copy + Clone> PreprocessedGrammar<Tok> {
  fn index_tokens(
    grammar: &TokenGrammar<Tok>,
  ) -> (IndexMap<Tok, Vec<TokenPosition>>, IndexMap<GrammarVertex, Vec<GrammarEdge>>) {
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
    (toks_with_locs, neighbors)
  }

  fn build_pairwise_transitions_table(
    grammar: &TokenGrammar<Tok>,
    toks_with_locs: &IndexMap<Tok, Vec<TokenPosition>>,
    neighbors: &IndexMap<GrammarVertex, Vec<GrammarEdge>>,
  ) -> IndexMap<StatePair, Vec<StackDiff>> {
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
          // Match any negative elements at the beginning of the edge weights, or else fail.
          // NB: We assume that if there are negative stack steps, they are *only* located at the
          // beginning of the edge's weight! This is ok because we are generating the stack steps
          // ourselves earlier in this same method.
          if shuffling_negative_steps_successfully_matched(&mut new_stack, VecDeque::from(weight)) {
            eprintln!("target: {:?}", target);
            match target {
              // We have completed a traversal -- there should be no "anon" steps in the result
              // (those are only for bookkeeping during this construction phase).
              GrammarVertex::State(right_tok_pos) => {
                let diff: Vec<StackStep> = new_stack.into_iter().filter_map(|step| match step {
                  GrammarEdgeWeightStep::Named(s) => Some(s),
                  _ => None,
                }).collect();
                eprintln!("diff: {:?}", diff);
                let transitions_for_pair = transitions.entry(StatePair {
                  left: *left_tok_pos,
                  right: right_tok_pos,
                }).or_insert(vec![]);
                eprintln!("transitions_for_pair: {:?}", transitions_for_pair);
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
    transitions
  }

  fn new(grammar: &TokenGrammar<Tok>) -> Self {
    let (states, neighbors) = Self::index_tokens(grammar);
    let transitions = Self::build_pairwise_transitions_table(grammar, &states, &neighbors);
    PreprocessedGrammar {
      states,
      transitions,
    }
  }
}

/// Problem Instance (optimize for perf later!!!!)

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct InputTokenIndex(usize);

#[derive(Debug, Clone, PartialEq, Eq)]
struct StackTrieTerminalEntry(Vec<UnionRange>);

// This isn't a lot of information, but I'm actually pretty sure it can be walked backwards to
// reconstruct the sequence of states in the input upon completion of a parse pretty
// easily/efficiently. This begins as both pointing to the same input index/state -- see
// Parse::initial_state(). This is the root of a tree. I think this traversal can be easily
// integrated with the aforementioned type-safe parse API with "collect" methods which accept tuples
// of arguments (or something) corresponding to the hierarchical reconstruction (and the stack diffs
// might be useful for that as well, or they might be just for bookkeeping during the parse).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct UnionRange {
  this_grammar_state: TokenPosition,
  other_input_index: InputTokenIndex,
  other_grammar_state: TokenPosition,
}

// TODO: List of known stack paths, for both "sides", per input token (sorted lexicographically)!
// TODO: Lexicographic sorting can be accomplished using a trie (of stack syms!!!!!!!!)! A sparse
// trie might work too -- can do this with a vec of filled indices alongside the trie vec (probably
// need two vecs, one with each each character, the other full of the next level of vecs).
#[derive(Debug, Clone)]
struct StackTrie {
  // TODO: can make this a reference to a stack alphabet and keep indices to stack steps as a
  // Vec<usize> for perf later (maybe).
  // NB: Keeping this as a (lexicographically) sorted vec of stack diffs is much more
  // likely to be efficient than nested trie structs, and should maintain the same runtime if we
  // don't explicitly call .sort() or whatever (making use of the lexicographical ordering at the
  // beginning of one step to produce a lexicographic ordering at the end of that step). It might be
  // appropriate to call it something other than a "trie", however.
  // TODO(!!!): traverse the list of stack diffs using a separate VecDeque of individual stack
  // steps, and keep a separate vector of "planned" entries to `stack_steps` and the appropriate
  // indices, and sort the queue of unions to try (see notebook) by some measure of the length of
  // the "planned" entries as a fraction (?) of the length of `stack_steps`.
  stack_steps: Vec<StackDiff>,
  // NB: Same length as `stack_steps`!
  terminal_entries: Vec<StackTrieTerminalEntry>,
}

#[derive(Debug, Clone)]
struct Parse(Vec<StackTrie>);

#[derive(Debug, Clone)]
struct SlowStackTrieForSetup(IndexMap<StackDiff, Vec<UnionRange>>);

impl SlowStackTrieForSetup {
  fn new() -> Self {
    SlowStackTrieForSetup(IndexMap::new())
  }

  // A valid stack trie may return true for this at any point, hence "probably" -- this is just
  // paranoia.
  fn is_probably_newly_initialized(&self) -> bool {
    let has_exactly_one_entry = 1 == self.0.len();
    // The .unwrap() is part of the assertion here.
    let entry_for_zero_diff = self.0.get(&StackDiff(vec![])).unwrap();
    let has_nonempty_terminal_entry_set = !entry_for_zero_diff.is_empty();
    has_exactly_one_entry && has_nonempty_terminal_entry_set
  }

  fn apply_to_following_token(
    left_index: InputTokenIndex,
    right_index: InputTokenIndex,
    cur_state: &mut Vec<SlowStackTrieForSetup>,
    transitions: &IndexMap<StatePair, Vec<StackDiff>>,
  ) {
    // Call .drain(..) so we can use the state and then update the objects without interference.
    let all_right_terminal_entries: Vec<UnionRange> = cur_state.get_mut(right_index.0).unwrap().0
      .drain(..)
      .flat_map(|(_, es)| es)
      .collect();
    let left_state_unrolled: Vec<(StackDiff, Vec<UnionRange>)> = cur_state
      .get_mut(left_index.0).unwrap().0
      .drain(..)
      .collect();
    for (cur_left_diff, cur_left_terminal_entry) in left_state_unrolled.into_iter() {
        for cur_left_union_range in cur_left_terminal_entry.into_iter() {
          for cur_right_union_range in all_right_terminal_entries.iter().cloned() {
            // TODO: using the StatePair map here is "cheating" in that it doesn't represent the
            // walking we have to do in the actual parsing -- this can be fixed later.
            let cur_pair = StatePair {
              left: cur_left_union_range.this_grammar_state.clone(),
              right: cur_right_union_range.this_grammar_state.clone(),
            };
            // Some (most?) state pairs don't exist in the grammar. We can cull these at this start
            // phase before getting into parsing.
            // TODO: see if this is relevant for good error messaging on a failed parse -- I think
            // this isn't relevant for the error that would be useful to a user.
            if let Some(ref all_cur_diffs) = transitions.get(&cur_pair) {
              for right_diff in all_cur_diffs.iter() {
                let joined_diff = cur_left_diff.sequence(&right_diff);
                (*cur_state.get_mut(left_index.0).unwrap().0.entry(joined_diff.clone())
                 .or_insert(vec![]))
                  .push(UnionRange {
                    this_grammar_state: cur_pair.left,
                    other_input_index: right_index,
                    other_grammar_state: cur_pair.right,
                  });
                (*cur_state.get_mut(right_index.0).unwrap().0.entry(joined_diff.clone())
                 .or_insert(vec![]))
                  .push(UnionRange {
                    this_grammar_state: cur_pair.right,
                    other_input_index: left_index,
                    other_grammar_state: cur_pair.left,
                  });
              }
            }
          }
        }
      }
  }

  fn build(self) -> StackTrie {
    let mut stack_steps: Vec<StackDiff> = vec![];
    let mut terminal_entries: Vec<StackTrieTerminalEntry> = vec![];
    for (cur_diff, cur_entries) in self.0.into_iter() {
      stack_steps.push(cur_diff);
      terminal_entries.push(StackTrieTerminalEntry(cur_entries));
    }
    StackTrie {
      stack_steps,
      terminal_entries,
    }
  }
}

impl Parse {
  fn initial_state<Tok: Sized + PartialEq + Eq + Hash + Copy + Clone>(
    states: &IndexMap<Tok, Vec<TokenPosition>>,
    input: Vec<Tok>,
  ) -> Vec<SlowStackTrieForSetup> {
    input.iter().enumerate().map(|(tok_ind, tok)| {
      let possible_states: Vec<TokenPosition> = states.get(tok)
        // TODO: make this into a Result!
        // TODO: specialize this error message for the case when Tok: Debug somehow!
        .expect(&format!("unrecognized token at index {:?} in input", tok_ind))
        .clone();
      (tok_ind, possible_states)
    }).map(|(tok_ind, possible_states)| {
      let input_index = InputTokenIndex(tok_ind);
      // The Vec<Vec<_>>s are a little confusing right now, but this reduces to "there is one
      // possible stack diff at this token (empty), which can map to any of several possible states,
      // all at this one input token". As we compare input elements we will begin populating
      // terminal entries from other indices into this trie.
      let union_ranges: Vec<UnionRange> = possible_states.into_iter().map(|tok_pos_in_grammar| {
        UnionRange {
          this_grammar_state: tok_pos_in_grammar.clone(),
          other_input_index: input_index.clone(),
          other_grammar_state: tok_pos_in_grammar.clone(),
        }
      }).collect();
      let mut map: IndexMap<StackDiff, Vec<UnionRange>> = IndexMap::new();
      map.insert(StackDiff(vec![]), union_ranges);
      SlowStackTrieForSetup(map)
    }).collect::<Vec<_>>()
  }

  // This method exists so we can drop all references to the original grammar when actually parsing.
  fn apply_initial_pairwise_transitions(
    // NB: `initial_state` is assumed to be what is returned by `initial_state()` above!
    initial_state: &mut Vec<SlowStackTrieForSetup>,
    transitions: &IndexMap<StatePair, Vec<StackDiff>>,
  ) {
    assert!(initial_state.get(0).unwrap().is_probably_newly_initialized());
    // I'm pretty sure iterating over pairs left to right gives us the same result as any other
    // order in this stage.
    // NB: Up to .len() - 1 because we are iterating over pairs.
    // TODO: after doing the pairwise transitions, we can "throw away" the initial (trivial)
    // `UnionRange`s -- these are necessary for reconstructing the final parse tree, but they are no
    // longer necessary for the actual algorithm (I think???) -- we should represent this in the
    // `StackTrie` struct somehow (...and this might be how we get lexicographic sorting without
    // doing anything with partially sorted lists at all......)!
    for left_ind in 0..(initial_state.len() - 1) {
      let right_ind = left_ind + 1;
      assert!(initial_state.get(right_ind).unwrap().is_probably_newly_initialized());
      SlowStackTrieForSetup::apply_to_following_token(
        InputTokenIndex(left_ind),
        InputTokenIndex(right_ind),
        initial_state,
        transitions,
      )
    }
  }

  // NB: define what one iteration means, and make that definition very flexible, but ensure it is
  // well-defined at all times.
  fn iterate(&mut self) {
    
  }

  // The pattern of defining new() methods which consume some inputs is *strictly* better than
  // generating them in the impl of a previous class with some (&self) method. One, you get to
  // dispatch on multiple argument types, two, you can use ownership and lifetimes of the arguments
  // in a meaningful way, instead of requiring that everything be (&self) (this is possible with
  // (self) args, but not in C++), three, you *very* clearly separate different phases of lowering
  // from the high-level `SimultaneousProductions` representation all the way down to the actual
  // parsing. Letting the caller control these also makes it more clear what transformations are
  // being performed as the lowering occurs, similar to how the iterate(&mut self) method below
  // helps make the runtime much easier to analyze, theoretically and on the actual computer.
  fn new<Tok: Sized + PartialEq + Eq + Hash + Copy + Clone>(
    grammar: &PreprocessedGrammar<Tok>,
    input: Vec<Tok>,
  ) -> Self {
    let mut initial_state: Vec<SlowStackTrieForSetup> = Self::initial_state(&grammar.states, input);
    Self::apply_initial_pairwise_transitions(&mut initial_state, &grammar.transitions);
    let real_tries: Vec<StackTrie> = initial_state.into_iter()
      .map(|slow_trie| slow_trie.build())
      .collect();
    Parse(real_tries)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn simple_parse() {
    // TODO: figure out more complex parsing such as stack cycles/etc before doing type-indexed
    // maps, as well as syntax sugar for defining cases.
    let prods = SimultaneousProductions([
      (ProductionReference::new("a"), Production(vec![
        Case(vec![
          CaseElement::Lit(Literal::new("ab"))])])),
      (ProductionReference::new("b"), Production(vec![
        Case(vec![
          CaseElement::Lit(Literal::new("ab")),
          CaseElement::Prod(ProductionReference::new("a")),
        ])]))
    ].iter().cloned().collect());
    let grammar = TokenGrammar::new(&prods);
    let preprocessed_grammar = PreprocessedGrammar::new(&grammar);
    let input: Vec<char> = "abab".chars().collect();
    let mut parse = Parse::new(&preprocessed_grammar, input);
    panic!("call iterate method until we have a full parse!");
  }

  #[test]
  fn preprocessed_state_for_non_cyclic_productions() {
    let prods = SimultaneousProductions([
      (ProductionReference::new("a"), Production(vec![
        Case(vec![
          CaseElement::Lit(Literal::new("ab"))])])),
      (ProductionReference::new("b"), Production(vec![
        Case(vec![
          CaseElement::Lit(Literal::new("ab")),
          CaseElement::Prod(ProductionReference::new("a")),
        ])]))
    ].iter().cloned().collect());
    let grammar = TokenGrammar::new(&prods);
    assert_eq!(grammar.clone(), TokenGrammar {
      tokens: vec!['a', 'b'],
      graph: LoweredProductions(vec![
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
    let preprocessed_grammar = PreprocessedGrammar::new(&grammar);
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
        }, vec![StackDiff(vec![StackStep::Negative(StackSym(ProdRef(0)))])]),
      ].iter().cloned().collect::<IndexMap<StatePair, Vec<StackDiff>>>(),
    });
  }

  #[test]
  #[should_panic(expected = "prod ref ProductionReference(\"c\") not found")]
  fn missing_prod_ref() {
    let prods = SimultaneousProductions([
      (ProductionReference::new("b"), Production(vec![
        Case(vec![
          CaseElement::Lit(Literal::new("ab")),
          CaseElement::Prod(ProductionReference::new("c")),
        ])]))
    ].iter().cloned().collect());
    TokenGrammar::new(&prods);
  }
}
