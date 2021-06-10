/* Copyright (C) 2021 Danny McClanahan <dmcC2@hypnicjerk.ai> */
/* SPDX-License-Identifier: GPL-3.0 */

use super::{
  api::*, grammar_indexing::*, lowering_to_indices::graph_coordinates::*, reconstruction::*,
  token::*,
};

use indexmap::IndexMap;
use typename::TypeName;

use std::{
  collections::VecDeque,
  fmt::{self, Debug},
  hash::{Hash, Hasher},
  rc::Rc,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindingError(String);

pub trait ProvidesProduction<Tok: Token> {
  fn as_production(&self) -> Production<Tok>;
  fn get_type_name(&self) -> TypeNameWrapper;
  fn get_acceptors(&self) -> Vec<Rc<Box<dyn PointerBoxingAcceptor>>>;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
pub struct TypeNameWrapper(String);

impl TypeNameWrapper {
  pub fn for_type<T: TypeName>() -> Self { TypeNameWrapper(T::type_name()) }

  pub fn as_production_reference(&self) -> ProductionReference { ProductionReference::new(&self.0) }
}

#[derive(Debug, Clone, PartialEq, Eq, TypeName)]
pub struct TypedCase<Tok: Token> {
  pub case: Case<Tok>,
  pub acceptor: Rc<Box<dyn PointerBoxingAcceptor>>,
}

#[derive(Debug, Clone, PartialEq, Eq, TypeName)]
pub struct TypedProduction<Tok: Token> {
  cases: Vec<TypedCase<Tok>>,
  output_type: TypeNameWrapper,
}

impl<Tok: Token> TypedProduction<Tok> {
  pub fn new<Output: TypeName>(cases: Vec<TypedCase<Tok>>) -> Self {
    TypedProduction {
      cases,
      output_type: TypeNameWrapper::for_type::<Output>(),
    }
  }
}

impl<Tok: Token> ProvidesProduction<Tok> for TypedProduction<Tok> {
  fn as_production(&self) -> Production<Tok> {
    Production(
      self
        .cases
        .iter()
        .map(|TypedCase { case, .. }| case)
        .cloned()
        .collect(),
    )
  }

  fn get_type_name(&self) -> TypeNameWrapper { self.output_type.clone() }

  fn get_acceptors(&self) -> Vec<Rc<Box<dyn PointerBoxingAcceptor>>> {
    self
      .cases
      .iter()
      .cloned()
      .map(|TypedCase { acceptor, .. }| acceptor)
      .collect()
  }
}

#[derive(Debug, PartialEq, Eq, TypeName)]
pub struct TypedSimultaneousProductions<
  Tok: Token,
  /* Members: HList, */
> {
  pub underlying: SimultaneousProductions<Tok>,
  pub bindings: IndexMap<ProdCaseRef, Rc<Box<dyn PointerBoxingAcceptor>>>,
}

impl<
    Tok: Token,
    /* Members: HList, */
  >
  TypedSimultaneousProductions<
    Tok,
    /* Members, */
  >
{
  pub fn reconstruct<Output: TypeName+'static>(
    &self,
    reconstruction: &CompletedWholeReconstruction,
  ) -> Result<Output, BindingError> {
    let mut reconstruction = reconstruction
      .clone()
      .0
      .into_iter()
      .collect::<VecDeque<_>>();
    if reconstruction.len() == 3
      && reconstruction.pop_front().unwrap()
        == CompleteSubReconstruction::State(LoweredState::Start)
      && reconstruction.pop_back().unwrap() == CompleteSubReconstruction::State(LoweredState::End)
    {
      match reconstruction.pop_front().unwrap() {
        CompleteSubReconstruction::Completed(CompletedCaseReconstruction { prod_case, args }) => {
          let acceptor_for_outer = self.bindings.get(&prod_case).ok_or_else(|| {
            BindingError(format!("no case found for prod case ref {:?}!", prod_case))
          })?;
          let TypedProductionParamsDescription { output_type, .. } =
            acceptor_for_outer.type_params();
          let expected_output_type = TypeNameWrapper::for_type::<Output>();
          if output_type != expected_output_type {
            /* FIXME: how do we reasonably accept a type parameter upon reconstruction of
             * a parse? */
            Err(BindingError(format!(
              "output type {:?} for case {:?} did not match expected output type {:?}",
              output_type, prod_case, expected_output_type
            )))
          } else {
            self
              .reconstruct_sub(acceptor_for_outer.clone(), &args)
              .and_then(|result_rc: Box<dyn std::any::Any>| {
                result_rc.downcast::<Output>().or_else(|_| {
                  Err(BindingError(format!(
                    "prod case {:?} with args {:?} could not be downcast to {:?}",
                    prod_case,
                    args,
                    TypeNameWrapper::for_type::<Output>()
                  )))
                })
              })
              .map(|x| *x)
          }
        },
        x => Err(BindingError(format!(
          "element {:?} in complete reconstruction {:?} was not recognized",
          x, reconstruction
        ))),
      }
    } else {
      Err(BindingError(format!("reconstruction {:?} was not recognized as a top-level reconstruction (with 3 elements, beginning at Start and ending at End)", reconstruction)))
    }
  }

  fn reconstruct_sub(
    &self,
    acceptor: Rc<Box<dyn PointerBoxingAcceptor>>,
    args: &Vec<CompleteSubReconstruction>,
  ) -> Result<Box<dyn std::any::Any>, BindingError> {
    let sub_args: Vec<Box<dyn std::any::Any>> = args
      .iter()
      .flat_map(|x| match x {
        CompleteSubReconstruction::State(_) => Ok(None),
        CompleteSubReconstruction::Completed(CompletedCaseReconstruction { prod_case, args }) => {
          let acceptor_for_outer = self.bindings.get(prod_case).ok_or_else(|| {
            BindingError(format!("no case found for prod case ref {:?}!", prod_case))
          })?;
          self
            .reconstruct_sub(acceptor_for_outer.clone(), args)
            .map(|x| Some(x))
        },
      })
      .flat_map(|x| x)
      .collect();
    let TypedProductionParamsDescription { params, .. } = acceptor.type_params();
    if sub_args.len() != params.len() {
      Err(BindingError(format!(
        "{:?} args for acceptor {:?} (expected {:?})",
        sub_args.len(),
        &acceptor,
        params.len()
      )))
    } else {
      acceptor
        .accept_erased(sub_args)
        .or_else(|e| Err(BindingError(format!("acceptance error {:?}", e))))
    }
  }

  pub fn new(production_boxes: Vec<Rc<Box<dyn ProvidesProduction<Tok>>>>) -> Self {
    let underlying = SimultaneousProductions(
      production_boxes
        .iter()
        .cloned()
        .map(|prod| {
          (
            prod.get_type_name().as_production_reference(),
            prod.as_production(),
          )
        })
        .collect(),
    );
    let bindings: IndexMap<ProdCaseRef, Rc<Box<dyn PointerBoxingAcceptor>>> = production_boxes
      .iter()
      .cloned()
      .enumerate()
      .flat_map(|(prod_ind, prod)| {
        let cur_prod_ref = ProdRef(prod_ind);
        prod
          .get_acceptors()
          .into_iter()
          .enumerate()
          .map(move |(case_ind, acceptor)| {
            let cur_prod_case_ref = ProdCaseRef {
              prod: cur_prod_ref,
              case: CaseRef(case_ind),
            };
            (cur_prod_case_ref, acceptor)
          })
          .collect::<Vec<_>>()
      })
      .collect();
    TypedSimultaneousProductions {
      underlying,
      bindings,
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
pub struct ParamName(String);

impl ParamName {
  pub fn new(s: &str) -> Self { ParamName(s.to_string()) }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
pub struct TypedParam {
  arg_type: TypeNameWrapper,
  arg_name: ParamName,
}

impl TypedParam {
  pub fn new<T: TypeName>(arg_name: ParamName) -> Self {
    TypedParam {
      arg_type: TypeNameWrapper::for_type::<T>(),
      arg_name,
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeName)]
pub struct TypedProductionParamsDescription {
  output_type: TypeNameWrapper,
  params: Vec<TypedParam>,
}

impl TypedProductionParamsDescription {
  pub fn new<T: TypeName>(params: Vec<TypedParam>) -> Self {
    TypedProductionParamsDescription {
      output_type: TypeNameWrapper::for_type::<T>(),
      params,
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AcceptanceError(String);

pub trait PointerBoxingAcceptor {
  fn identity_salt(&self) -> &str;
  fn type_params(&self) -> TypedProductionParamsDescription;
  fn accept_erased(
    &self,
    args: Vec<Box<dyn std::any::Any>>,
  ) -> Result<Box<dyn std::any::Any>, AcceptanceError>;
}

impl Debug for dyn PointerBoxingAcceptor {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "PointerBoxingAcceptor({:?})", self.type_params())
  }
}
impl PartialEq for dyn PointerBoxingAcceptor {
  fn eq(&self, other: &Self) -> bool { self.identity_salt() == other.identity_salt() }
}
impl Eq for dyn PointerBoxingAcceptor {}
impl Hash for dyn PointerBoxingAcceptor {
  fn hash<H: Hasher>(&self, state: &mut H) { self.identity_salt().hash(state); }
}
impl TypeName for dyn PointerBoxingAcceptor {
  fn fmt(f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "dyn {}::PointerBoxingAcceptor", module_path!())
  }
}

#[macro_export]
macro_rules! vec_box_rc {
    ($($x:expr),+) => {
      vec![
        $(
          std::rc::Rc::new(Box::new($x))
        ),+
      ]
    };
  }

/* #[macro_export] */
/* macro_rules! _merge { */
/* /* This alows merging the head/tail! */ */
/* (@merge [], [$($rest:tt)*]) => { [$($rest)*] }; */
/* (@merge [$($cur:tt)*], []) => { [$($cur)*] }; */
/* (@merge */
/* [$($cur_arg_name:ident: $cur_arg_type:ty),+], */
/* [$($rest_arg_name:ident: $rest_arg_type:ty),+]) => { */
/* [ */
/* $($cur_arg_name: $cur_arg_type),+ */
/* , */
/* $($rest_arg_name: $rest_arg_type),+ */
/* ] */
/* }; */
/* ([$(rest:tt)+]) => { [$($rest)+] } */
/* } */

/* #[macro_export] */
/* macro_rules! _extract_typed_params { */
/* ([$arg_name:ident: $arg_type:ty]) => { */
/* [$arg_name: $arg_type] */
/* }; */
/* ([$_literal:expr]) => { [] }; */

/* ($([$arg_name:ident: $arg_type:ty]),+) => { */
/* [ */
/* $( */
/* $arg_name: $arg_type */
/* ),+ */
/* ] */
/* }; */

/* ([$arg_name:ident: $arg_type:ty], [$_literal:expr], $($rest:tt)+) => { */
/* _extract_typed_params![[$arg_name: $arg_type], $($rest:tt)+] */
/* }; */
/* ([$arg_name:ident: $arg_type:ty], [$_literal:expr]) => { */
/* [$arg_name: $arg_type] */
/* }; */

/* ([$_literal:expr], $($rest:tt)+) => { _extract_typed_params![$($rest)+] */
/* }; */
/* } */

/* fn wow() { */
/* trace_macros!(true); */
/* _extract_typed_params![["a"]]; */
/* _extract_typed_params![[y: u32]]; */
/* _extract_typed_params![["a"], [y: u32]]; */
/* _extract_typed_params![[y: u32], ["XXX"]]; */
/* trace_macros!(false); */
/* } */

/* #[macro_export] */
/* macro_rules! _generate_typed_params_description { */
/* ($production_type:ty, [$($arg_name:ident: $arg_type:ty),+]) => { */
/* TypedProductionParamsDescription::new::<$production_type>(vec![ */
/* $( */
/* TypedParam::new::<$arg_type>(ParamName::new(stringify!($a))) */
/* ),+ */
/* ]) */
/* }; */
/* } */

/* #[macro_export] */
/* macro_rules! _generate_case { */
/* ($gen_id:ident, $production_type:ty => [$($decl:tt)+] => $body:block) => */
/* {{ */
/* pub struct $gen_id(pub String); */
/* impl PointerBoxingAcceptor> for $gen_id { */
/* fn identity_salt(&self) -> &str { */
/* self.0.as_str() */
/* } */

/* fn type_params(&self) -> TypedProductionParamsDescription { */
/* _generate_typed_params_description![$production_type, */
/* _extract_typed_params![$($decl)+]] */
/* } */

/* fn accept(args: Vec<Box<dyn std::any::Any>>) -> $production_type { */
/* let rev_args: Vec<_> = args.into_iter().rev().collect(); */
/* $( */
/* let $a: $in = rev_args.pop() */
/* .expect("failed to pop from argument vector!") */
/* .downcast::<$in>() */
/* .expect("invalid downcast!"); */
/* )* */
/* $body */
/* } */
/* } */
/* let acceptor = Rc::<dyn */
/* PointerBoxingAcceptor>>::new( */
/* $gen_id(format!("anonymous class at {}::{}", module_path!(), */
/* stringify!($gen_id)))); */
/* let case = Case(vec![$($x),*]); */
/* TypedCase { case, acceptor } */
/* }}; */
/* } */

/* #[macro_export] */
/* macro_rules! productions { */
/* ($($production_type:ty => [ */
/* $(case ($($decl:tt)+) => $body:block),+ */
/* ]),+) => { */
/* TypedSimultaneousProductions::new(vec![ */
/* $((Box::new(TypedProduction(vec![ */
/* $( */
/* gensym!{ _generate_case!{ $production_type => [$($decl)+] => $body } } */
/* ),* */
/* ])) */
/* )),* */
/* ]) */
/* }; */
/* } */
