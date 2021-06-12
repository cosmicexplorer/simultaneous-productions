/* Copyright (C) 2021 Danny McClanahan <dmcC2@hypnicjerk.ai> */
/* SPDX-License-Identifier: GPL-3.0 */

//! Implementation for getting a [super::grammar_indexing::PreprocessedGrammar].
//!
//! Performance doesn't matter here.

use crate::lowering_to_indices::{graph_representation::*, mapping_to_tokens::*};
use sp_core::{graph_coordinates::*, token::Token};

use indexmap::{IndexMap, IndexSet};
use typename::TypeName;

use std::{
  collections::{HashSet, VecDeque},
  hash::{Hash, Hasher},
};
