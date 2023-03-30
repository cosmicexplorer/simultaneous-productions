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
}

/* pub struct */
