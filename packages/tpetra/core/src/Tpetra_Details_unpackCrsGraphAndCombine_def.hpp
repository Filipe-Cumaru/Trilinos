// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef TPETRA_DETAILS_UNPACKCRSGRAPHANDCOMBINE_DEF_HPP
#define TPETRA_DETAILS_UNPACKCRSGRAPHANDCOMBINE_DEF_HPP

#include "TpetraCore_config.h"
#include "Teuchos_Array.hpp"
#include "Teuchos_ArrayView.hpp"
#include "Tpetra_Details_castAwayConstDualView.hpp"
#include "Tpetra_Details_computeOffsets.hpp"
#include "Tpetra_Details_createMirrorView.hpp"
#include "Tpetra_Details_OrdinalTraits.hpp"
#include "Tpetra_Details_Behavior.hpp"
#include "Tpetra_CrsGraph_decl.hpp"
#include "Tpetra_Details_getEntryOnHost.hpp"
#include "Kokkos_Core.hpp"
#include <memory>
#include <string>

/// \file Tpetra_Details_unpackCrsGraphAndCombine_def.hpp
/// \brief Definition of functions for unpacking the entries of a
///   Tpetra::CrsGraph for communication, in the case where it is
///   valid to go to the KokkosSparse::CrsGraph (local sparse graph
///   data structure) directly.
/// \warning This file, and its contents, are implementation details
///   of Tpetra.  The file itself or its contents may disappear or
///   change at any time.
///
/// Data (bytes) describing the row of the CRS graph are "packed"
/// (concatenated) in to a (view of) Packet* object in the following order:
///
///   1. global column indices (GlobalOrdinal)
///   2. proces IDs (optional, int)
///
/// The functions in this file are companions to
/// Tpetra_Details_packCrsGraph.hpp, i.e., Tpetra_Details_packCrsGraph.hpp
/// implements the packing order described above to ensure proper unpacking.

namespace Tpetra {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// Forward declaration of Distributor
class Distributor;
#endif // DOXYGEN_SHOULD_SKIP_THIS

//
// Users must never rely on anything in the Details namespace.
//
namespace Details {

namespace UnpackAndCombineCrsGraphImpl {

/// \brief Unpack a single row of a CrsGraph
///
/// \tparam LO The type of local indices.  See the
///   documentation of Map for requirements.
/// \tparam GO The type of global indices.  See the
///   documentation of Map for requirements.
/// \tparam Device The Kokkos device type.  See the documentation of Map
///   for requirements.
/// \tparam BufferDevice The "buffer device type."
template<class Packet, class GO, class Device, class BufferDevice>
KOKKOS_FUNCTION int
unpackRow(typename Kokkos::View<GO*,Device,Kokkos::MemoryUnmanaged>& gids_out,
          typename Kokkos::View<int*,Device,Kokkos::MemoryUnmanaged>& pids_out,
          const Kokkos::View<const Packet*,BufferDevice>& imports,
          const size_t offset,
          const size_t num_ent)
{
  typedef typename Kokkos::View<GO*,Device>::size_type size_type;

  if (num_ent == 0) {
    // Empty rows always take zero bytes, to ensure sparsity.
    return 0;
  }

  // Unpack GIDs
  for (size_type k=0; k<num_ent; k++)
    gids_out(k) = imports(offset+k);

  // Unpack PIDs
  if (pids_out.size() > 0) {
    for (size_type k=0; k<num_ent; k++)
      pids_out(k) = static_cast<int>(imports(offset+num_ent+k));
  }

  return 0;
}

/// \brief Unpacks and combines a single row of the CrsGraph.
///
/// \tparam LocalGraph KokkosSparse::CrsGraph specialization.
/// \tparam LocalMap Type of the "local" column map
/// \tparam BufferDevice Type of the "buffer device type."
///   See Trilinos GitHub Issue #1088 for details.
///
/// Data (bytes) describing the row of the CRS graph are "unpacked"
/// from a single (concatenated) (view of) Packet* directly into the
/// row of the graph.
template<class Packet, class LocalGraph, class LocalMap, class BufferDevice>
class UnpackAndCombineFunctor {

  typedef Packet packet_type;
  typedef LocalMap local_map_type;
  typedef LocalGraph local_graph_type;
  typedef BufferDevice buffer_device_type;

  typedef typename local_map_type::local_ordinal_type LO;
  typedef typename local_map_type::global_ordinal_type GO;
  // Kokkos::parallel_reduce fails to compile if named device_type and typedef
  // is public
  typedef typename local_map_type::device_type device_type;
  typedef typename device_type::execution_space execution_space;

  typedef Kokkos::View<const size_t*, buffer_device_type> num_packets_per_lid_type;
  typedef Kokkos::View<const size_t*, device_type> offsets_type;
  typedef Kokkos::View<const packet_type*, buffer_device_type> input_buffer_type;
  typedef Kokkos::View<const LO*, device_type> import_lids_type;

  typedef Kokkos::View<LO*, device_type> lids_scratch_type;
  typedef Kokkos::View<GO*, device_type> gids_scratch_type;
  typedef Kokkos::View<int*,device_type> pids_scratch_type;

  static_assert(std::is_same<LO, typename local_graph_type::data_type>::value,
                "LocalMap::local_ordinal_type and "
                "LocalGraph::data_type must be the same.");

  local_graph_type local_graph;
  local_map_type local_col_map;
  input_buffer_type imports;
  num_packets_per_lid_type num_packets_per_lid;
  import_lids_type import_lids;
  offsets_type offsets;
  Tpetra::CombineMode combine_mode;
  size_t max_num_ent;
  bool unpack_pids;
  bool atomic;
  Kokkos::Experimental::UniqueToken<execution_space,
                                    Kokkos::Experimental::UniqueTokenScope::Global> tokens;
  lids_scratch_type lids_scratch;
  gids_scratch_type gids_scratch;
  pids_scratch_type pids_scratch;

 public:
  typedef Kokkos::pair<int, LO> value_type;

  UnpackAndCombineFunctor(
      const local_graph_type& local_graph_in,
      const local_map_type& local_col_map_in,
      const input_buffer_type& imports_in,
      const num_packets_per_lid_type& num_packets_per_lid_in,
      const import_lids_type& import_lids_in,
      const offsets_type& offsets_in,
      const Tpetra::CombineMode combine_mode_in,
      const size_t max_num_ent_in,
      const bool unpack_pids_in,
      const bool atomic_in) :
    local_graph(local_graph_in),
    local_col_map(local_col_map_in),
    imports(imports_in),
    num_packets_per_lid(num_packets_per_lid_in),
    import_lids(import_lids_in),
    offsets(offsets_in),
    combine_mode(combine_mode_in),
    max_num_ent(max_num_ent_in),
    unpack_pids(unpack_pids_in),
    atomic(atomic_in),
    tokens(execution_space()),
    lids_scratch("pids_scratch", tokens.size() * max_num_ent),
    gids_scratch("gids_scratch", tokens.size() * max_num_ent),
    pids_scratch("lids_scratch", tokens.size() * max_num_ent)
  {}

  KOKKOS_INLINE_FUNCTION void init(value_type& dst) const
  {
    using Tpetra::Details::OrdinalTraits;
    dst = Kokkos::make_pair(0, OrdinalTraits<LO>::invalid());
  }

  KOKKOS_INLINE_FUNCTION void
  join(volatile value_type& dst, const volatile value_type& src) const
  {
    // `dst` should reflect the first (least) bad index and
    // all other associated error codes and data.  Thus, we need only
    // check if the `src` object shows an error and if its associated
    // bad index is less than `dst`'s bad index.
    using Tpetra::Details::OrdinalTraits;
    if (src.second != OrdinalTraits<LO>::invalid()) {
      // An error in the src; check if
      //   1. `dst` shows errors
      //   2. If `dst` does show errors, if src's bad index is less than
      //      *this' bad index
      if (dst.second == OrdinalTraits<LO>::invalid() ||
          src.second < dst.second) {
        dst = src;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const LO i, value_type& dst) const
  {
    using Kokkos::View;
    using Kokkos::subview;
    using Kokkos::MemoryUnmanaged;
    typedef typename execution_space::size_type size_type;
    typedef typename Kokkos::pair<size_type, size_type> slice;

    typedef View<LO*, device_type, MemoryUnmanaged> lids_out_type;
    typedef View<int*,device_type, MemoryUnmanaged> pids_out_type;
    typedef View<GO*, device_type, MemoryUnmanaged> gids_out_type;

    const size_t num_packets_this_lid = num_packets_per_lid(i);
    const size_t num_ent = (unpack_pids) ? num_packets_this_lid/2
                                         : num_packets_this_lid;
    if (unpack_pids && num_packets_this_lid%2 != 0) {
      // Attempting to unpack PIDs, but num_packets_this_lid is not even; this
      // should never
      dst = Kokkos::make_pair(1, i);
      return;
    }

    // Only unpack data if there is a nonzero number to unpack
    if (num_ent == 0) {
      return;
    }

    // there is actually something in the row
    const size_t buf_size = imports.size();
    const size_t offset = offsets(i);

    if (offset > buf_size || offset + num_packets_this_lid > buf_size) {
      dst = Kokkos::make_pair(2, i); // out of bounds
      return;
    }

    // Get subviews in to the scratch arrays.  The token returned from acquire
    // is an integer in [0, tokens.size()).  It is used to grab a unique (to
    // this thread) subview of the scratch arrays.
    const size_type token = tokens.acquire();
    const size_t a = static_cast<size_t>(token) * max_num_ent;
    const size_t b = a + num_ent;
    lids_out_type lids_out = subview(lids_scratch, slice(a, b));
    gids_out_type gids_out = subview(gids_scratch, slice(a, b));
    pids_out_type pids_out = subview(pids_scratch, slice(a, (unpack_pids ? b : a)));

    // Unpack this row!
    int err = unpackRow<packet_type,GO,device_type,buffer_device_type>(
        gids_out, pids_out, imports, offset, num_ent);

    if (err != 0) {
      dst = Kokkos::make_pair(3, i);
      return;
    }

    // Column indices come in as global indices, in case the
    // source object's column Map differs from the target object's
    // (this's) column Map, and must be converted local index values
    for (size_t k = 0; k < num_ent; ++k) {
      lids_out(k) = local_col_map.getLocalElement(gids_out(k));
    }

    tokens.release(token);
  }
};

/// \brief Perform the unpack operation for the graph
///
/// \tparam LocalGraph the specialization of the KokkosSparse::CrsGraph
///   local graph
/// \tparam LocalMap the type of the local column map
///
/// This is a higher level interface to the UnpackAndCombineFunctor
template<class Packet, class LocalGraph, class LocalMap, class BufferDevice>
void
unpackAndCombine(
    const LocalGraph& local_graph,
    const LocalMap& local_map,
    const Kokkos::View<const Packet*, BufferDevice, Kokkos::MemoryUnmanaged>& imports,
    const Kokkos::View<const size_t*, BufferDevice, Kokkos::MemoryUnmanaged>& num_packets_per_lid,
    const Kokkos::View<const typename LocalMap::local_ordinal_type*,
                       typename LocalMap::device_type,
                       Kokkos::MemoryUnmanaged>& import_lids,
    const Tpetra::CombineMode combine_mode,
    const bool unpack_pids,
    const bool atomic)
{

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
      "unpackAndCombine[New] should not (yet) be called, the method is "
      "incomplete.  To complete, indices need to be inserted (unpacked) in to "
      "the destination graph.  The local graph, a Kokkos::StaticCrsGraph, does "
      "not support insertion of indices");

  typedef typename LocalMap::local_ordinal_type LO;
  typedef typename LocalMap::device_type device_type;
  typedef typename device_type::execution_space execution_space;
  typedef Kokkos::RangePolicy<execution_space, Kokkos::IndexType<LO> > range_policy;
  typedef UnpackAndCombineFunctor<Packet,LocalGraph,LocalMap,BufferDevice> unpack_functor_type;

  const char prefix[] =
    "Tpetra::Details::UnpackAndCombineCrsGraphImpl::unpackAndCombine: ";

  const size_t num_import_lids = static_cast<size_t>(import_lids.extent(0));
  if (num_import_lids == 0) {
    // Nothing to unpack
    return;
  }

  {
    TEUCHOS_TEST_FOR_EXCEPTION(combine_mode == INSERT,
        std::invalid_argument,
        prefix << "INSERT combine mode is not allowed if the graph has a static graph "
        "(i.e., was constructed with the CrsGraph constructor that takes a "
        "const CrsGraph pointer).");

    // Unknown combine mode!
    TEUCHOS_TEST_FOR_EXCEPTION(combine_mode != REPLACE,
        std::invalid_argument,
        prefix << "Invalid combine mode; should never get "
        "here!  Please report this bug to the Tpetra developers.");

    // Check that sizes of input objects are consistent.
    bool bad_num_import_lids =
      num_import_lids != static_cast<size_t>(num_packets_per_lid.extent(0));
    TEUCHOS_TEST_FOR_EXCEPTION(bad_num_import_lids,
        std::invalid_argument,
        prefix << "importLIDs.size() (" << num_import_lids << ") != "
        "numPacketsPerLID.size() (" << num_packets_per_lid.extent(0) << ").");
  } // end QA error checking

  // Get the offsets
  Kokkos::View<size_t*, device_type> offsets("offsets", num_import_lids+1);
  computeOffsetsFromCounts(offsets, num_packets_per_lid);

  // Determine the maximum number of entries in any row in the graph.  The
  // maximum number of entries is needed to allocate unpack buffers on the
  // device.
  size_t max_num_ent;
  Kokkos::parallel_reduce("MaxReduce",
    num_packets_per_lid.size(),
    KOKKOS_LAMBDA(const int& i, size_t& running_max_num_ent) {
      size_t num_packets_this_lid = num_packets_per_lid(i);
      size_t num_ent = (unpack_pids) ? num_packets_this_lid/2
                                     : num_packets_this_lid;
      if (num_ent > running_max_num_ent) running_max_num_ent = num_ent;
    }, Kokkos::Experimental::Max<size_t>(max_num_ent));

  // Now do the actual unpack!
  unpack_functor_type f(local_graph, local_map,
      imports, num_packets_per_lid, import_lids, offsets, combine_mode,
      max_num_ent, unpack_pids, atomic);

  typename unpack_functor_type::value_type x;
  Kokkos::parallel_reduce(range_policy(0, static_cast<LO>(num_import_lids)), f, x);
  auto x_h = x.to_std_pair();
  TEUCHOS_TEST_FOR_EXCEPTION(x_h.first != 0, std::runtime_error,
      prefix << "UnpackAndCombineFunctor reported error code "
             << x_h.first << " for the first bad row " << x_h.second);

  return;
}

template<class Packet, class LocalGraph, class BufferDevice>
size_t
unpackAndCombineWithOwningPIDsCount(
  const LocalGraph& local_graph,
  const Kokkos::View<const typename LocalGraph::data_type*,
                     typename LocalGraph::device_type,
                     Kokkos::MemoryUnmanaged> permute_from_lids,
  const Kokkos::View<const Packet*, BufferDevice>& imports,
  const Kokkos::View<const size_t*, BufferDevice>& num_packets_per_lid,
  const size_t num_same_ids)
{
  using Kokkos::parallel_reduce;
  typedef LocalGraph local_graph_type;
  typedef typename local_graph_type::data_type LO;
  typedef typename local_graph_type::device_type device_type;
  typedef typename device_type::execution_space execution_space;
  typedef Kokkos::RangePolicy<execution_space, Kokkos::IndexType<LO> > range_policy;

  size_t count = 0;
  LO num_items;

  // Number of graph entries to unpack (returned by this function).
  num_items = static_cast<LO>(num_same_ids);
  if (num_items) {
    size_t kcnt = 0;
    parallel_reduce(
      range_policy(0, num_items),
      KOKKOS_LAMBDA(const LO lid, size_t& update) {
        update += static_cast<size_t>(local_graph.row_map[lid+1]
                                     -local_graph.row_map[lid]);
      }, kcnt);
    count += kcnt;
  }

  // Count entries copied directly from the source graph with permuting.
  num_items = static_cast<LO>(permute_from_lids.extent(0));
  if (num_items) {
    size_t kcnt = 0;
    parallel_reduce(
      range_policy(0, num_items),
      KOKKOS_LAMBDA(const LO i, size_t& update) {
        const LO lid = permute_from_lids(i);
        update += static_cast<size_t>(local_graph.row_map[lid+1]
                                     - local_graph.row_map[lid]);
      }, kcnt);
    count += kcnt;
  }

  {
    // Count entries received from other MPI processes.
    size_t tot_num_ent = 0;
    Kokkos::parallel_reduce("SumReduce",
        num_packets_per_lid.size(),
        KOKKOS_LAMBDA(const int& i, size_t& lsum) {
          lsum += num_packets_per_lid(i) / 2;
        }, Kokkos::Experimental::Sum<size_t>(tot_num_ent));
    count += tot_num_ent;
  }

  return count;
}

/// \brief Setup row pointers for remotes
template<class Packet, class LO, class Device, class BufferDevice>
void
setupRowPointersForRemotes(
  const Kokkos::View<size_t*, Device>& tgt_rowptr,
  const Kokkos::View<const LO*, Device>& import_lids,
  const Kokkos::View<const Packet*, BufferDevice>& imports,
  const Kokkos::View<const size_t*, BufferDevice>& num_packets_per_lid)
{
  using Kokkos::parallel_reduce;
  typedef Device device_type;
  typedef typename device_type::execution_space execution_space;
  typedef typename Kokkos::View<size_t*,device_type>::size_type size_type;
  typedef Kokkos::RangePolicy<execution_space, Kokkos::IndexType<size_type> > range_policy;

  const size_type N = num_packets_per_lid.extent(0);
  parallel_for("Setup row pointers for remotes",
    range_policy(0, N),
    KOKKOS_LAMBDA(const size_t i){
      typedef typename std::remove_reference<decltype(tgt_rowptr(0))>::type atomic_incr_type;
      const size_t num_packets_this_lid = num_packets_per_lid(i);
      const size_t num_ent = num_packets_this_lid / 2;
      Kokkos::atomic_fetch_add(&tgt_rowptr(import_lids(i)), atomic_incr_type(num_ent));
    });
}

// Convert array of row lengths to a CRS pointer array
template<class Device>
void
makeCrsRowPtrFromLengths(
    const Kokkos::View<size_t*,Device,Kokkos::MemoryUnmanaged>& tgt_rowptr,
    const Kokkos::View<size_t*,Device>& new_start_row)
{
  using Kokkos::parallel_scan;
  typedef Device device_type;
  typedef typename device_type::execution_space execution_space;
  typedef typename Kokkos::View<size_t*,device_type>::size_type size_type;
  typedef Kokkos::RangePolicy<execution_space, Kokkos::IndexType<size_type> > range_policy;
  const size_type N = new_start_row.extent(0);
  parallel_scan(
    range_policy(0, N),
    KOKKOS_LAMBDA(const size_t& i, size_t& update, const bool& final) {
      auto cur_val = tgt_rowptr(i);
      if (final) {
        tgt_rowptr(i) = update;
        new_start_row(i) = tgt_rowptr(i);
      }
      update += cur_val;
    }
  );
}

template<class LocalGraph, class LocalMap>
void
copyDataFromSameIDs(
    const Kokkos::View<typename LocalMap::global_ordinal_type*,
                       typename LocalMap::device_type>& tgt_colind,
    const Kokkos::View<int*, typename LocalMap::device_type>& tgt_pids,
    const Kokkos::View<size_t*,typename LocalMap::device_type>& new_start_row,
    const Kokkos::View<size_t*, typename LocalMap::device_type>& tgt_rowptr,
    const Kokkos::View<const int*, typename LocalMap::device_type>& src_pids,
    const LocalGraph& local_graph,
    const LocalMap& local_col_map,
    const size_t num_same_ids,
    const int my_pid)
{
  using Kokkos::parallel_for;
  typedef typename LocalMap::device_type device_type;
  typedef typename LocalMap::local_ordinal_type LO;
  typedef typename device_type::execution_space execution_space;
  typedef Kokkos::RangePolicy<execution_space, Kokkos::IndexType<size_t> > range_policy;

  parallel_for(
    range_policy(0, num_same_ids),
    KOKKOS_LAMBDA(const size_t i) {
      typedef typename std::remove_reference<decltype(new_start_row(0))>::type atomic_incr_type;

      const LO src_lid    = static_cast<LO>(i);
      size_t src_row = local_graph.row_map(src_lid);

      const LO tgt_lid      = static_cast<LO>(i);
      const size_t tgt_row = tgt_rowptr(tgt_lid);

      const size_t nsr = local_graph.row_map(src_lid+1)
                       - local_graph.row_map(src_lid);
      Kokkos::atomic_fetch_add(&new_start_row(tgt_lid), atomic_incr_type(nsr));

      for (size_t j=local_graph.row_map(src_lid);
                  j<local_graph.row_map(src_lid+1); ++j) {
        LO src_col = local_graph.entries(j);
        tgt_colind(tgt_row + j - src_row) = local_col_map.getGlobalElement(src_col);
        tgt_pids(tgt_row + j - src_row) = (src_pids(src_col) != my_pid) ? src_pids(src_col) : -1;
      }
    }
  );
}

template<class LocalGraph, class LocalMap>
void
copyDataFromPermuteIDs(
    const Kokkos::View<typename LocalMap::global_ordinal_type*,
                       typename LocalMap::device_type>& tgt_colind,
    const Kokkos::View<int*,
                       typename LocalMap::device_type>& tgt_pids,
    const Kokkos::View<size_t*,
                       typename LocalMap::device_type>& new_start_row,
    const Kokkos::View<size_t*,
                       typename LocalMap::device_type>& tgt_rowptr,
    const Kokkos::View<const int*,
                       typename LocalMap::device_type>& src_pids,
    const Kokkos::View<const typename LocalMap::local_ordinal_type*,
                       typename LocalMap::device_type>& permute_to_lids,
    const Kokkos::View<const typename LocalMap::local_ordinal_type*,
                       typename LocalMap::device_type>& permute_from_lids,
    const LocalGraph& local_graph,
    const LocalMap& local_col_map,
    const int my_pid)
{
  using Kokkos::parallel_for;
  typedef typename LocalMap::device_type device_type;
  typedef typename LocalMap::local_ordinal_type LO;
  typedef typename device_type::execution_space execution_space;
  typedef typename Kokkos::View<LO*,device_type>::size_type size_type;
  typedef Kokkos::RangePolicy<execution_space, Kokkos::IndexType<size_type> > range_policy;

  const size_type num_permute_to_lids = permute_to_lids.extent(0);

  parallel_for(
    range_policy(0, num_permute_to_lids),
    KOKKOS_LAMBDA(const size_t i) {
      typedef typename std::remove_reference<decltype(new_start_row(0)) >::type atomic_incr_type;

      const LO src_lid = permute_from_lids(i);
      const size_t src_row = local_graph.row_map(src_lid);

      const LO tgt_lid = permute_to_lids(i);
      const size_t tgt_row = tgt_rowptr(tgt_lid);

      size_t nsr = local_graph.row_map(src_lid+1)
                 - local_graph.row_map(src_lid);
      Kokkos::atomic_fetch_add(&new_start_row(tgt_lid), atomic_incr_type(nsr));

      for (size_t j=local_graph.row_map(src_lid);
                  j<local_graph.row_map(src_lid+1); ++j) {
        LO src_col = local_graph.entries(j);
        tgt_colind(tgt_row + j - src_row) = local_col_map.getGlobalElement(src_col);
        tgt_pids(tgt_row + j - src_row) = (src_pids(src_col) != my_pid) ? src_pids(src_col) : -1;
      }
    }
  );
}

template<class Packet, class LocalGraph, class LocalMap, class BufferDevice>
void
unpackAndCombineIntoCrsArrays2(
    const Kokkos::View<typename LocalMap::global_ordinal_type*, typename LocalMap::device_type>& tgt_colind,
    const Kokkos::View<int*, typename LocalMap::device_type>& tgt_pids,
    const Kokkos::View<size_t*,typename LocalMap::device_type>& new_start_row,
    const Kokkos::View<const size_t*, typename LocalMap::device_type>& offsets,
    const Kokkos::View<const typename LocalMap::local_ordinal_type*, typename LocalMap::device_type>& import_lids,
    const Kokkos::View<const Packet*, BufferDevice>& imports,
    const Kokkos::View<const size_t*, BufferDevice>& num_packets_per_lid,
    const LocalGraph& local_graph,
    const LocalMap /*& local_col_map*/,
    const int my_pid)
{
  using Kokkos::View;
  using Kokkos::subview;
  using Kokkos::MemoryUnmanaged;
  using Kokkos::parallel_reduce;
  using Kokkos::atomic_fetch_add;

  typedef Packet packet_type;
  typedef BufferDevice buffer_device_type;
  typedef typename LocalMap::device_type device_type;
  typedef typename LocalMap::local_ordinal_type LO;
  typedef typename LocalMap::global_ordinal_type GO;
  typedef typename device_type::execution_space execution_space;
  typedef typename Kokkos::View<LO*, device_type>::size_type size_type;
  typedef typename Kokkos::pair<size_type, size_type> slice;
  typedef Kokkos::RangePolicy<execution_space, Kokkos::IndexType<size_type> > range_policy;

  typedef View<int*,device_type, MemoryUnmanaged> pids_out_type;
  typedef View<GO*, device_type, MemoryUnmanaged> gids_out_type;

  const size_type num_import_lids = import_lids.size();
  const char prefix[] = "UnpackAndCombineCrsGraphImpl::unpackAndCombineIntoCrsArrays2: ";

  // RemoteIDs: Loop structure following UnpackAndCombine
  int gbl_err_count;
  parallel_reduce("Unpack and combine into CRS",
    range_policy(0, num_import_lids),
    KOKKOS_LAMBDA(const size_t i, int& err) {
      typedef typename std::remove_reference< decltype( new_start_row(0) ) >::type atomic_incr_type;
      const size_t num_packets_this_lid = num_packets_per_lid(i);
      const size_t num_ent = num_packets_this_lid / 2;
      const size_t offset = offsets(i);
      const LO lcl_row = import_lids(i);
      const size_t start_row = atomic_fetch_add(&new_start_row(lcl_row), atomic_incr_type(num_ent));
      const size_t end_row = start_row + num_ent;

      gids_out_type gids_out = subview(tgt_colind, slice(start_row, end_row));
      pids_out_type pids_out = subview(tgt_pids, slice(start_row, end_row));

      err += unpackRow<packet_type,GO,device_type,buffer_device_type>(
          gids_out, pids_out, imports, offset, num_ent);

      // Correct target PIDs.
      for (size_t j = 0; j < static_cast<size_t>(num_ent); ++j) {
        const int pid = pids_out(j);
        pids_out(j) = (pid != my_pid) ? pid : -1;
      }
    }, gbl_err_count);

  TEUCHOS_TEST_FOR_EXCEPTION(gbl_err_count != 0,
      std::invalid_argument, prefix <<
      "Attempting to unpack PIDs, but num_ent is not even; this should never "
      "happen!  Please report this bug to the Tpetra developers.");

  return;
}

template<class Packet, class LocalGraph, class LocalMap, class BufferDevice>
void
unpackAndCombineIntoCrsArrays(
    const LocalGraph & local_graph,
    const LocalMap & local_col_map,
    const Kokkos::View<const typename LocalMap::local_ordinal_type*,
                       typename LocalMap::device_type,
                       Kokkos::MemoryUnmanaged>& import_lids,
    const Kokkos::View<const Packet*, BufferDevice>& imports,
    const Kokkos::View<const size_t*, BufferDevice>& num_packets_per_lid,
    const Kokkos::View<const typename LocalMap::local_ordinal_type*,
                       typename LocalMap::device_type,
                       Kokkos::MemoryUnmanaged>& permute_to_lids,
    const Kokkos::View<const typename LocalMap::local_ordinal_type*,
                       typename LocalMap::device_type,
                       Kokkos::MemoryUnmanaged>& permute_from_lids,
    const Kokkos::View<size_t*,
                       typename LocalMap::device_type,
                       Kokkos::MemoryUnmanaged>& tgt_rowptr,
    const Kokkos::View<typename LocalMap::global_ordinal_type*,
                       typename LocalMap::device_type,
                       Kokkos::MemoryUnmanaged>& tgt_colind,
    const Kokkos::View<const int*,
                       typename LocalMap::device_type,
                       Kokkos::MemoryUnmanaged>& src_pids,
    const Kokkos::View<int*,
                       typename LocalMap::device_type,
                       Kokkos::MemoryUnmanaged>& tgt_pids,
    const size_t num_same_ids,
    const size_t tgt_num_rows,
    const size_t tgt_num_nonzeros,
    const int my_tgt_pid)
{
  using Kokkos::View;
  using Kokkos::subview;
  using Kokkos::parallel_for;
  using Kokkos::MemoryUnmanaged;
  typedef Packet packet_type;
  typedef LocalMap local_map_type;
  typedef LocalGraph local_graph_type;
  typedef BufferDevice buffer_device_type;
  typedef typename LocalMap::device_type device_type;
  typedef typename LocalMap::local_ordinal_type LO;
  typedef typename device_type::execution_space execution_space;
  typedef typename Kokkos::View<LO*, device_type>::size_type size_type;
  typedef Kokkos::RangePolicy<execution_space, Kokkos::IndexType<size_t> > range_policy;

  const char prefix[] = "UnpackAndCombineCrsGraphImpl::unpackAndCombineIntoCrsArrays: ";

  const size_t N = tgt_num_rows;
  const size_t mynnz = tgt_num_nonzeros;

  // In the case of reduced communicators, the sourceGraph won't have
  // the right "my_pid", so thus we have to supply it.
  const int my_pid = my_tgt_pid;

  // Zero the rowptr
  parallel_for(
    range_policy(0, N+1),
    KOKKOS_LAMBDA(const size_t i) {
      tgt_rowptr(i) = 0;
    }
  );

  // same IDs: Always first, always in the same place
  parallel_for(
    range_policy(0, num_same_ids),
    KOKKOS_LAMBDA(const size_t i) {
      const LO tgt_lid = static_cast<LO>(i);
      const LO src_lid = static_cast<LO>(i);
      tgt_rowptr(tgt_lid) = local_graph.row_map(src_lid+1)
                          - local_graph.row_map(src_lid);
    }
  );

  // Permute IDs: Still local, but reordered
  const size_type num_permute_to_lids = permute_to_lids.extent(0);
  parallel_for(
    range_policy(0, num_permute_to_lids),
    KOKKOS_LAMBDA(const size_t i) {
      const LO tgt_lid = permute_to_lids(i);
      const LO src_lid = permute_from_lids(i);
      tgt_rowptr(tgt_lid) = local_graph.row_map(src_lid+1)
                          - local_graph.row_map(src_lid);
    }
  );

  // Get the offsets from the number of packets per LID
  const size_type num_import_lids = import_lids.extent(0);
  View<size_t*, device_type> offsets("offsets", num_import_lids+1);
  computeOffsetsFromCounts(offsets, num_packets_per_lid);

#ifdef HAVE_TPETRA_DEBUG
  {
    auto nth_offset_h = getEntryOnHost(offsets, num_import_lids);
    const bool condition =
      nth_offset_h != static_cast<size_t>(imports.extent(0));
    TEUCHOS_TEST_FOR_EXCEPTION
      (condition, std::logic_error, prefix
       << "The final offset in bytes " << nth_offset_h
       << " != imports.size() = " << imports.extent(0)
       << ".  Please report this bug to the Tpetra developers.");
  }
#endif // HAVE_TPETRA_DEBUG

  // Setup row pointers for remotes
  setupRowPointersForRemotes<packet_type,LO,device_type,buffer_device_type>(
      tgt_rowptr, import_lids, imports, num_packets_per_lid);

  // If multiple processes contribute to the same row, we may need to
  // update row offsets.  This tracks that.
  View<size_t*, device_type> new_start_row("new_start_row", N+1);

  // Turn row length into a real CRS row pointer
  makeCrsRowPtrFromLengths(tgt_rowptr, new_start_row);
  {
    auto nth_tgt_rowptr_h = getEntryOnHost(tgt_rowptr, N);
    bool condition = nth_tgt_rowptr_h != mynnz;
    TEUCHOS_TEST_FOR_EXCEPTION(condition, std::invalid_argument,
      prefix << "CRS_rowptr[last] = " <<
      nth_tgt_rowptr_h << "!= mynnz = " << mynnz << ".");
  }

  // SameIDs: Copy the data over
  copyDataFromSameIDs<LocalGraph,LocalMap>(tgt_colind, tgt_pids, new_start_row,
      tgt_rowptr, src_pids, local_graph, local_col_map, num_same_ids, my_pid);

  copyDataFromPermuteIDs<LocalGraph,LocalMap>(tgt_colind, tgt_pids, new_start_row,
      tgt_rowptr, src_pids, permute_to_lids, permute_from_lids,
      local_graph, local_col_map, my_pid);

  if (imports.extent(0) <= 0) {
    return;
  }

  unpackAndCombineIntoCrsArrays2<
    packet_type,local_graph_type,local_map_type,buffer_device_type>(
        tgt_colind, tgt_pids, new_start_row, offsets, import_lids, imports,
        num_packets_per_lid, local_graph, local_col_map, my_pid);

  return;
}

} // namespace UnpackAndCombineCrsGraphImpl

/// \brief Unpack the imported column indices and combine into graph.
///
/// \tparam LO The type of local indices.  See the
///   documentation of Map for requirements.
/// \tparam GO The type of global indices.  See the
///   documentation of Map for requirements.
/// \tparam Node The Kokkos Node type.  See the documentation of Map
///   for requirements.
///
/// \param sourceGraph [in] the CrsGraph source
///
/// \param imports [in] Input pack buffer
///
/// \param numPacketsPerLID [out] Entry k gives the number of bytes
///   packed for row exportLIDs[k] of the local graph.
///
/// \param importLIDs [in] Local indices of the rows to pack.
///
/// \param constantNumPackets [out] Setting this to zero tells the caller
///   to expect a possibly /// different ("nonconstant") number of packets per local index
///   (i.e., a possibly different number of entries per row).
///
/// \param distor [in] The distributor (not used)
///
/// \param combineMode [in] the mode to use for combining indices
///
/// \param atomic [in] whether or not do atomic adds/replaces in to the graph
///
/// \warning The allowed \c combineMode are:
///   REPLACE. INSERT is not allowed.
///
/// This is the public interface to the unpack and combine machinery and
/// converts passed Teuchos::ArrayView objects to Kokkos::View objects (and
/// copies back in to the Teuchos::ArrayView objects, if needed).  When
/// CrsGraph migrates fully to adopting Kokkos::DualView objects for its storage
/// of data, this procedure could be bypassed.
template<class LO, class GO, class Node>
void
unpackCrsGraphAndCombine(
    const CrsGraph<LO, GO, Node>& sourceGraph,
    const Teuchos::ArrayView<const typename CrsGraph<LO,GO,Node>::packet_type>& imports,
    const Teuchos::ArrayView<const size_t>& numPacketsPerLID,
    const Teuchos::ArrayView<const LO>& importLIDs,
    size_t constantNumPackets,
    Distributor & distor,
    CombineMode combineMode,
    const bool atomic)
{
  using Kokkos::View;
  typedef typename Node::device_type device_type;
  typedef typename CrsGraph<LO,GO,Node>::packet_type packet_type;
  typedef typename CrsGraph<LO, GO, Node>::local_graph_type local_graph_type;
  typedef typename CrsGraph<LO, GO, Node>::buffer_device_type buffer_device_type;
  static_assert(std::is_same<device_type, typename local_graph_type::device_type>::value,
                 "Node::device_type and LocalGraph::device_type must be the same.");

  typedef typename device_type::execution_space execution_space;
  typename execution_space::device_type outputDevice;

  typedef typename buffer_device_type::execution_space buffer_execution_space;
  typename buffer_execution_space::device_type bufferOutputDevice;

  // Convert all Teuchos::Array to Kokkos::View.

  // numPacketsPerLID, importLIDs, and imports are input, so we have to copy
  // them to device.  Since unpacking is done directly in to the local graph
  // (lclGraph), no copying needs to be performed after unpacking.
  auto imports_d =
    create_mirror_view_from_raw_host_array(bufferOutputDevice,
        imports.getRawPtr(), imports.size(),
        true, "imports");

  auto num_packets_per_lid_d =
    create_mirror_view_from_raw_host_array(bufferOutputDevice,
        numPacketsPerLID.getRawPtr(), numPacketsPerLID.size(),
        true, "num_packets_per_lid");

  auto import_lids_d =
    create_mirror_view_from_raw_host_array(outputDevice,
        importLIDs.getRawPtr(), importLIDs.size(),
        true, "import_lids");

  auto local_graph = sourceGraph.getLocalGraph();
  auto local_col_map = sourceGraph.getColMap()->getLocalMap();

  // Now do the actual unpack!
  typedef decltype(local_col_map) local_map_type;
  UnpackAndCombineCrsGraphImpl::unpackAndCombine<
    packet_type,local_graph_type,local_map_type,buffer_device_type>(
        local_graph, local_col_map, imports_d, num_packets_per_lid_d,
        import_lids_d, combineMode, false, atomic);

  return;
}

template<class LO, class GO, class Node>
void
unpackCrsGraphAndCombineNew(
    const CrsGraph<LO, GO, Node>& sourceGraph,
    const Kokkos::DualView<const typename CrsGraph<LO,GO,Node>::packet_type*,
                           typename CrsGraph<LO,GO,Node>::buffer_device_type>& imports,
    const Kokkos::DualView<const size_t*,
                           typename CrsGraph<LO,GO,Node>::buffer_device_type>& numPacketsPerLID,
    const Kokkos::DualView<const LO*, typename Node::device_type>& importLIDs,
    const size_t constantNumPackets,
    Distributor& distor,
    const CombineMode combineMode,
    const bool atomic)
{
  using Tpetra::Details::castAwayConstDualView;
  using Kokkos::View;
  typedef typename Node::device_type device_type;
  typedef CrsGraph<LO, GO, Node> crs_graph_type;
  typedef typename crs_graph_type::packet_type packet_type;
  typedef typename crs_graph_type::local_graph_type local_graph_type;
  typedef typename crs_graph_type::buffer_device_type buffer_device_type;
  typedef typename buffer_device_type::memory_space buffer_memory_space;
  typedef typename device_type::memory_space memory_space;

  static_assert(std::is_same<device_type, typename local_graph_type::device_type>::value,
                "Node::device_type and LocalGraph::device_type must be "
                "the same.");

  {
    auto numPacketsPerLID_nc = castAwayConstDualView(numPacketsPerLID);
    numPacketsPerLID_nc.template sync<buffer_memory_space>();
  }
  auto num_packets_per_lid_d = numPacketsPerLID.template view<buffer_memory_space>();

  {
    auto importLIDs_nc = castAwayConstDualView(importLIDs);
    importLIDs_nc.template sync<memory_space>();
  }
  auto import_lids_d = importLIDs.template view<memory_space>();

  {
    auto imports_nc = castAwayConstDualView(imports);
    imports_nc.template sync<buffer_memory_space>();
  }
  auto imports_d = imports.template view<buffer_memory_space>();

  auto local_graph = sourceGraph.getLocalGraph();
  auto local_col_map = sourceGraph.getColMap()->getLocalMap();
  typedef decltype(local_col_map) local_map_type;

  // Now do the actual unpack!
  UnpackAndCombineCrsGraphImpl::unpackAndCombine<
    packet_type,local_graph_type,local_map_type,buffer_device_type>(
      local_graph, local_col_map, imports_d, num_packets_per_lid_d,
      import_lids_d, combineMode, false, atomic);
}

/// \brief Special version of Tpetra::Details::unpackCrsGraphAndCombine
///   that also unpacks owning process ranks.
///
/// Perform the count for unpacking the imported column indices and pids,
/// and combining them into graph.  Return (a ceiling on)
/// the number of local stored entries ("nonzeros") in the graph.  If
/// there are no shared rows in the sourceGraph this count is exact.
///
/// Note: This routine also counts the copyAndPermute nonzeros in
/// addition to those that come in via import.
///
/// \tparam LO The type of local indices.  See the
///   documentation of Map for requirements.
/// \tparam GO The type of global indices.  See the
///   documentation of Map for requirements.
/// \tparam Node The Kokkos Node type.  See the documentation of Map
///   for requirements.
///
/// \param sourceGraph [in] the CrsGraph source
///
/// \param imports [in] Input pack buffer
///
/// \param numPacketsPerLID [out] Entry k gives the number of bytes
///   packed for row exportLIDs[k] of the local graph.
///
/// \param importLIDs [in] Local indices of the rows to pack.
///
/// \param constantNumPackets [out] Setting this to zero tells the caller
///   to expect a possibly /// different ("nonconstant") number of packets per local index
///   (i.e., a possibly different number of entries per row).
///
/// \param distor [in] The distributor (not used)
///
/// \param combineMode [in] the mode to use for combining
///
/// \param numSameIds [in]
///
/// \param permuteToLIDs [in]
///
/// \param permuteFromLIDs [in]
///
/// \warning The allowed \c combineMode are:
///   ADD, REPLACE, and ABSMAX. INSERT is not allowed.
//
/// \warning This method is intended for expert developer use
///   only, and should never be called by user code.
///
/// Note: This is the public interface to the unpack and combine machinery and
/// converts passed Teuchos::ArrayView objects to Kokkos::View objects (and
/// copies back in to the Teuchos::ArrayView objects, if needed).  When
/// CrsGraph migrates fully to adopting Kokkos::DualView objects for its storage
/// of data, this procedure could be bypassed.
template<class LocalOrdinal, class GlobalOrdinal, class Node>
size_t
unpackAndCombineWithOwningPIDsCount(
    const CrsGraph<LocalOrdinal, GlobalOrdinal, Node> & sourceGraph,
    const Teuchos::ArrayView<const LocalOrdinal> &importLIDs,
    const Teuchos::ArrayView<const typename CrsGraph<LocalOrdinal,GlobalOrdinal,Node>::packet_type> &imports,
    const Teuchos::ArrayView<const size_t>& numPacketsPerLID,
    size_t constantNumPackets,
    Distributor &distor,
    CombineMode combineMode,
    size_t numSameIDs,
    const Teuchos::ArrayView<const LocalOrdinal>& permuteToLIDs,
    const Teuchos::ArrayView<const LocalOrdinal>& permuteFromLIDs)
{
  using Kokkos::MemoryUnmanaged;
  using Kokkos::View;
  typedef typename Node::device_type device_type;
  typedef typename CrsGraph<LocalOrdinal,GlobalOrdinal,Node>::packet_type packet_type;
  typedef typename CrsGraph<LocalOrdinal,GlobalOrdinal,Node>::local_graph_type local_graph_type;
  typedef typename CrsGraph<LocalOrdinal,GlobalOrdinal,Node>::buffer_device_type buffer_device_type;
  const char prefix[] = "unpackAndCombineWithOwningPIDsCount: ";

  TEUCHOS_TEST_FOR_EXCEPTION
    (permuteToLIDs.size() != permuteFromLIDs.size(), std::invalid_argument,
     prefix << "permuteToLIDs.size() = " << permuteToLIDs.size() << " != "
     "permuteFromLIDs.size() = " << permuteFromLIDs.size() << ".");
  // FIXME (mfh 26 Jan 2015) If there are no entries on the calling
  // process, then the graph is neither locally nor globally indexed.
  const bool locallyIndexed = sourceGraph.isLocallyIndexed();
  TEUCHOS_TEST_FOR_EXCEPTION
    (! locallyIndexed, std::invalid_argument, prefix << "The input "
    "CrsGraph 'sourceGraph' must be locally indexed.");
  TEUCHOS_TEST_FOR_EXCEPTION
    (importLIDs.size() != numPacketsPerLID.size(), std::invalid_argument,
     prefix << "importLIDs.size() = " << importLIDs.size() << " != "
     "numPacketsPerLID.size() = " << numPacketsPerLID.size() << ".");

  auto local_graph = sourceGraph.getLocalGraph();
  auto permute_from_lids_d =
    create_mirror_view_from_raw_host_array(device_type(),
                                           permuteFromLIDs.getRawPtr(),
                                           permuteFromLIDs.size(), true,
                                           "permute_from_lids");
  auto imports_d =
    create_mirror_view_from_raw_host_array(buffer_device_type(),
                                           imports.getRawPtr(),
                                           imports.size(), true,
                                           "imports");
  auto num_packets_per_lid_d =
    create_mirror_view_from_raw_host_array(buffer_device_type(),
                                           numPacketsPerLID.getRawPtr(),
                                           numPacketsPerLID.size(), true,
                                           "num_packets_per_lid");

  return UnpackAndCombineCrsGraphImpl::unpackAndCombineWithOwningPIDsCount<
    packet_type,local_graph_type,buffer_device_type>(
      local_graph, permute_from_lids_d, imports_d, num_packets_per_lid_d, numSameIDs);
}

/// \brief unpackAndCombineIntoCrsArrays
///
/// \note You should call unpackAndCombineWithOwningPIDsCount first
///   and allocate all arrays accordingly, before calling this
///   function.
///
/// Note: The SourcePids vector (on input) should contam
/// Tpetra::Import_Util::getPids, with the "-1 for local" option being
/// used.
///
/// Note: The TargetPids vector (on output) will contain owning PIDs
/// for each entry in the graph, with the "-1 for local" for locally
/// owned entries.
template<class LocalOrdinal, class GlobalOrdinal, class Node>
void
unpackAndCombineIntoCrsArrays(
    const CrsGraph<LocalOrdinal, GlobalOrdinal, Node> & sourceGraph,
    const Teuchos::ArrayView<const LocalOrdinal>& importLIDs,
    const Teuchos::ArrayView<const typename CrsGraph<LocalOrdinal,GlobalOrdinal,Node>::packet_type>& imports,
    const Teuchos::ArrayView<const size_t>& numPacketsPerLID,
    const size_t constantNumPackets,
    Distributor& distor,
    const CombineMode combineMode,
    const size_t numSameIDs,
    const Teuchos::ArrayView<const LocalOrdinal>& permuteToLIDs,
    const Teuchos::ArrayView<const LocalOrdinal>& permuteFromLIDs,
    size_t TargetNumRows,
    size_t TargetNumNonzeros,
    const int MyTargetPID,
    const Teuchos::ArrayView<size_t>& CRS_rowptr,
    const Teuchos::ArrayView<GlobalOrdinal>& CRS_colind,
    const Teuchos::ArrayView<const int>& SourcePids,
    Teuchos::Array<int>& TargetPids)
{
  using Kokkos::View;
  using Kokkos::deep_copy;
  using Teuchos::ArrayView;
  using Teuchos::outArg;
  using Teuchos::REDUCE_MAX;
  using Teuchos::reduceAll;
  typedef LocalOrdinal LO;
  typedef typename CrsGraph<LocalOrdinal,GlobalOrdinal,Node>::packet_type packet_type;
  typedef typename CrsGraph<LocalOrdinal,GlobalOrdinal,Node>::local_graph_type local_graph_type;
  typedef typename CrsGraph<LocalOrdinal,GlobalOrdinal,Node>::buffer_device_type buffer_device_type;
  typedef typename Node::device_type device_type;
  typedef typename device_type::execution_space execution_space;
  typedef typename buffer_device_type::execution_space buffer_execution_space;
  typedef typename ArrayView<const LO>::size_type size_type;

  const char prefix[] = "Tpetra::Details::unpackAndCombineIntoCrsArrays: ";

  TEUCHOS_TEST_FOR_EXCEPTION(
    TargetNumRows + 1 != static_cast<size_t>(CRS_rowptr.size()),
    std::invalid_argument, prefix << "CRS_rowptr.size() = " <<
    CRS_rowptr.size() << "!= TargetNumRows+1 = " << TargetNumRows+1 << ".");

  TEUCHOS_TEST_FOR_EXCEPTION(
    permuteToLIDs.size() != permuteFromLIDs.size(), std::invalid_argument,
    prefix << "permuteToLIDs.size() = " << permuteToLIDs.size()
    << "!= permuteFromLIDs.size() = " << permuteFromLIDs.size() << ".");
  const size_type numImportLIDs = importLIDs.size();

  TEUCHOS_TEST_FOR_EXCEPTION(
    numImportLIDs != numPacketsPerLID.size(), std::invalid_argument,
    prefix << "importLIDs.size() = " << numImportLIDs << " != "
    "numPacketsPerLID.size() = " << numPacketsPerLID.size() << ".");

  // Preseed TargetPids with -1 for local
  if (static_cast<size_t>(TargetPids.size()) != TargetNumNonzeros) {
    TargetPids.resize(TargetNumNonzeros);
  }
  TargetPids.assign(TargetNumNonzeros, -1);

  // Grab pointers for sourceGraph
  auto local_graph = sourceGraph.getLocalGraph();
  auto local_col_map = sourceGraph.getColMap()->getLocalMap();

  // Convert input arrays to Kokkos::View
  typename execution_space::device_type outputDevice;
  typename buffer_execution_space::device_type bufferOutputDevice;

  auto import_lids_d = create_mirror_view_from_raw_host_array(outputDevice,
        importLIDs.getRawPtr(), importLIDs.size(),
        true, "import_lids");

  auto imports_d = create_mirror_view_from_raw_host_array(bufferOutputDevice,
        imports.getRawPtr(), imports.size(),
        true, "imports");

  auto num_packets_per_lid_d = create_mirror_view_from_raw_host_array(bufferOutputDevice,
      numPacketsPerLID.getRawPtr(), numPacketsPerLID.size(),
      true, "num_packets_per_lid");

  auto permute_from_lids_d = create_mirror_view_from_raw_host_array(outputDevice,
      permuteFromLIDs.getRawPtr(), permuteFromLIDs.size(),
      true, "permute_from_lids");

  auto permute_to_lids_d = create_mirror_view_from_raw_host_array(outputDevice,
      permuteToLIDs.getRawPtr(), permuteToLIDs.size(),
      true, "permute_to_lids");

  auto crs_rowptr_d = create_mirror_view_from_raw_host_array(outputDevice,
      CRS_rowptr.getRawPtr(), CRS_rowptr.size(),
      true, "crs_rowptr");

  auto crs_colind_d = create_mirror_view_from_raw_host_array(outputDevice,
      CRS_colind.getRawPtr(), CRS_colind.size(),
      true, "crs_colidx");

  auto src_pids_d = create_mirror_view_from_raw_host_array(outputDevice,
      SourcePids.getRawPtr(), SourcePids.size(),
      true, "src_pids");

  auto tgt_pids_d = create_mirror_view_from_raw_host_array(outputDevice,
      TargetPids.getRawPtr(), TargetPids.size(),
      true, "tgt_pids");

  typedef decltype(local_col_map) local_map_type;
  UnpackAndCombineCrsGraphImpl::unpackAndCombineIntoCrsArrays<
    packet_type,local_graph_type,local_map_type,buffer_device_type>(
      local_graph, local_col_map, import_lids_d, imports_d, num_packets_per_lid_d,
      permute_to_lids_d, permute_from_lids_d, crs_rowptr_d, crs_colind_d, src_pids_d,
      tgt_pids_d, numSameIDs, TargetNumRows, TargetNumNonzeros, MyTargetPID);

  // Copy outputs back to host
  typename decltype(crs_rowptr_d)::HostMirror crs_rowptr_h(
      CRS_rowptr.getRawPtr(), CRS_rowptr.size());
  deep_copy(crs_rowptr_h, crs_rowptr_d);

  typename decltype(crs_colind_d)::HostMirror crs_colind_h(
      CRS_colind.getRawPtr(), CRS_colind.size());
  deep_copy(crs_colind_h, crs_colind_d);

  typename decltype(tgt_pids_d)::HostMirror tgt_pids_h(
      TargetPids.getRawPtr(), TargetPids.size());
  deep_copy(tgt_pids_h, tgt_pids_d);

}

} // namespace Details
} // namespace Tpetra

#define TPETRA_DETAILS_UNPACKCRSGRAPHANDCOMBINE_INSTANT( LO, GO, NT ) \
  template void \
  Details::unpackCrsGraphAndCombine<LO, GO, NT>( \
    const CrsGraph<LO, GO, NT>&, \
    const Teuchos::ArrayView<const typename CrsGraph<LO,GO,NT>::packet_type>&, \
    const Teuchos::ArrayView<const size_t>&, \
    const Teuchos::ArrayView<const LO>&, \
    size_t, \
    Distributor&, \
    CombineMode, \
    const bool); \
  template void \
  Details::unpackCrsGraphAndCombineNew<LO, GO, NT>( \
    const CrsGraph<LO, GO, NT>&, \
    const Kokkos::DualView<const typename CrsGraph<LO,GO,NT>::packet_type*, \
                           typename CrsGraph<LO, GO, NT>::buffer_device_type>&, \
    const Kokkos::DualView<const size_t*, typename CrsGraph<LO, GO, NT>::buffer_device_type>&, \
    const Kokkos::DualView<const LO*, NT::device_type>&, \
    const size_t, \
    Distributor&, \
    const CombineMode, \
    const bool); \
  template void \
  Details::unpackAndCombineIntoCrsArrays<LO, GO, NT>( \
    const CrsGraph<LO, GO, NT> &, \
    const Teuchos::ArrayView<const LO>&, \
    const Teuchos::ArrayView<const typename CrsGraph<LO,GO,NT>::packet_type>&, \
    const Teuchos::ArrayView<const size_t>&, \
    const size_t, \
    Distributor&, \
    const CombineMode, \
    const size_t, \
    const Teuchos::ArrayView<const LO>&, \
    const Teuchos::ArrayView<const LO>&, \
    size_t, \
    size_t, \
    const int, \
    const Teuchos::ArrayView<size_t>&, \
    const Teuchos::ArrayView<GO>&, \
    const Teuchos::ArrayView<const int>&, \
    Teuchos::Array<int>&); \
  template size_t \
  Details::unpackAndCombineWithOwningPIDsCount<LO, GO, NT>( \
    const CrsGraph<LO, GO, NT> &, \
    const Teuchos::ArrayView<const LO> &, \
    const Teuchos::ArrayView<const typename CrsGraph<LO,GO,NT>::packet_type> &, \
    const Teuchos::ArrayView<const size_t>&, \
    size_t, \
    Distributor &, \
    CombineMode, \
    size_t, \
    const Teuchos::ArrayView<const LO>&, \
    const Teuchos::ArrayView<const LO>&);

#endif // TPETRA_DETAILS_UNPACKCRSGRAPHANDCOMBINE_DEF_HPP