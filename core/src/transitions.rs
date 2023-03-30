/*
 * Description: A queryable database for traversing state transition paths.
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

//! A queryable database for traversing state transition paths.

use crate::{
  grammar_indexing as gi,
  lowering_to_indices::{grammar_building as gb, graph_coordinates as gc},
};

use indexmap::IndexMap;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IndexedTrieNodeRef(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedTrieNode {
  pub next_nodes: IndexMap<Option<gi::NamedOrAnonStep>, Vec<IndexedTrieNodeRef>>,
  pub prev_nodes: IndexMap<Option<gi::NamedOrAnonStep>, Vec<IndexedTrieNodeRef>>,
}

impl IndexedTrieNode {
  fn bare() -> Self {
    Self {
      next_nodes: IndexMap::new(),
      prev_nodes: IndexMap::new(),
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedTrie {
  pub vertex_mapping: IndexMap<gi::EpsilonGraphVertex, IndexedTrieNodeRef>,
  pub trie_node_universe: Vec<IndexedTrieNode>,
}

impl IndexedTrie {
  fn new() -> Self {
    Self {
      vertex_mapping: IndexMap::new(),
      trie_node_universe: Vec::new(),
    }
  }

  fn get_trie(&mut self, node_ref: IndexedTrieNodeRef) -> &mut IndexedTrieNode {
    let IndexedTrieNodeRef(node_index) = node_ref;
    self
      .trie_node_universe
      .get_mut(node_index)
      .expect("indexed trie node was out of bounds somehow?")
  }

  fn trie_ref_for_vertex(&mut self, vtx: &gi::EpsilonGraphVertex) -> IndexedTrieNodeRef {
    let basic_node = IndexedTrieNode::bare();
    let trie_node_ref_for_vertex = if let Some(x) = self.vertex_mapping.get(vtx) {
      *x
    } else {
      let next_ref = IndexedTrieNodeRef(self.trie_node_universe.len());
      self.trie_node_universe.push(basic_node.clone());
      self.vertex_mapping.insert(*vtx, next_ref);
      next_ref
    };
    trie_node_ref_for_vertex
  }

  pub fn from_epsilon_graph(graph: gi::EpsilonNodeStateSubgraph) -> Self {
    let gi::EpsilonNodeStateSubgraph {
      vertex_mapping,
      trie_node_universe,
    } = graph;
    let reverse_vertex_mapping: IndexMap<gi::TrieNodeRef, gi::EpsilonGraphVertex> = vertex_mapping
      .iter()
      .map(|(x, y)| (y.clone(), x.clone()))
      .collect();
    let mut ret = Self::new();

    for (vtx, gi::TrieNodeRef(old_trie_node_ref)) in vertex_mapping.into_iter() {
      let new_trie_ref = ret.trie_ref_for_vertex(&vtx);
      let IndexedTrieNode {
        next_nodes: mut new_next_nodes,
        prev_nodes: mut new_prev_nodes,
      } = ret.get_trie(new_trie_ref).clone();

      let gi::StackTrieNode {
        next_nodes: old_next_nodes,
        prev_nodes: old_prev_nodes,
        ..
      } = trie_node_universe[old_trie_node_ref].clone();

      /* Copy over next nodes. */
      for old_next_entry in old_next_nodes.into_iter() {
        match old_next_entry {
          gi::StackTrieNextEntry::Completed(lowered_state) => {
            let ref mut next_nodes = new_next_nodes.entry(None).or_insert_with(Vec::new);
            let old_next_vertex = lowered_state.into_vertex();
            let new_next_trie_ref = ret.trie_ref_for_vertex(&old_next_vertex);
            next_nodes.push(new_next_trie_ref);
          },
          gi::StackTrieNextEntry::Incomplete(gi::TrieNodeRef(old_next_node_ref)) => {
            let gi::StackTrieNode { step, .. } = trie_node_universe[old_next_node_ref];
            let ref mut next_nodes = new_next_nodes.entry(step.clone()).or_insert_with(Vec::new);
            let old_next_vertex = reverse_vertex_mapping
              .get(&gi::TrieNodeRef(old_next_node_ref))
              .unwrap();
            let new_next_trie_ref = ret.trie_ref_for_vertex(&old_next_vertex);
            next_nodes.push(new_next_trie_ref);
          },
        }
      }
      /* Copy over prev nodes. */
      for old_prev_entry in old_prev_nodes.into_iter() {
        match old_prev_entry {
          gi::StackTrieNextEntry::Completed(lowered_state) => {
            let ref mut prev_nodes = new_prev_nodes.entry(None).or_insert_with(Vec::new);
            let old_prev_vertex = lowered_state.into_vertex();
            let new_prev_trie_ref = ret.trie_ref_for_vertex(&old_prev_vertex);
            prev_nodes.push(new_prev_trie_ref);
          },
          gi::StackTrieNextEntry::Incomplete(gi::TrieNodeRef(old_prev_node_ref)) => {
            let gi::StackTrieNode { step, .. } = trie_node_universe[old_prev_node_ref];
            let ref mut prev_nodes = new_prev_nodes.entry(step.clone()).or_insert_with(Vec::new);
            let old_prev_vertex = reverse_vertex_mapping
              .get(&gi::TrieNodeRef(old_prev_node_ref))
              .unwrap();
            let new_prev_trie_ref = ret.trie_ref_for_vertex(&old_prev_vertex);
            prev_nodes.push(new_prev_trie_ref);
          },
        }
      }

      *ret.get_trie(new_trie_ref) = IndexedTrieNode {
        next_nodes: new_next_nodes,
        prev_nodes: new_prev_nodes,
      };
    }

    ret
  }
}
