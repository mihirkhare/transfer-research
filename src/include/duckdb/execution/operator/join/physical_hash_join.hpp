//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/physical_hash_join.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/value_operations/value_operations.hpp"
#include "duckdb/execution/join_hashtable.hpp"
#include "duckdb/execution/operator/join/perfect_hash_join_executor.hpp"
#include "duckdb/execution/operator/join/physical_comparison_join.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/optimizer/predicate_transfer/bloom_filter/bloom_filter.hpp"
#include "duckdb/planner/operator/logical_join.hpp"

// #include <iostream>

namespace duckdb {

#define LIP_NONE 0
#define LIP_BUILD 1
#define LIP_PROBE 2

class LIPProbeInfo {
public:
	LIPProbeInfo() = delete;

	explicit LIPProbeInfo(const vector<shared_ptr<BloomFilterUsage>>& lip_filters) :
		lip_filters(lip_filters), probe_order(lip_filters.size()), bf_hit_counts(lip_filters.size()),
		bf_total_counts(lip_filters.size()), bf_hit_percentages(lip_filters.size()),
		hash_staging(LogicalType::HASH), probe_res(STANDARD_VECTOR_SIZE), probe_sel(STANDARD_VECTOR_SIZE) {
		for (idx_t i = 0; i < lip_filters.size(); i++) {
			probe_order[i] = i;
		}
		// probe_seconds = std::chrono::microseconds(0);
	}

	// std::chrono::microseconds probe_seconds;

	//! Reference to filters being probed
	const vector<shared_ptr<BloomFilterUsage>>& lip_filters;
	//! Order in which to probe lip_filters
	vector<idx_t> probe_order;
	//! Current chunks processed in batch
	size_t chunks_processed = 0;
	//! Total chunks in batch
	size_t batch_size = 32;
	//! Hit count of each BF
	vector<size_t> bf_hit_counts;
	//! Total counts of each BF
	vector<size_t> bf_total_counts;
	//! Percentage of each BF
	vector<pair<double, idx_t>> bf_hit_percentages;

	//! staging area for hashes
	Vector hash_staging;
	//! results vector for probes (no need to clear between probes)
	vector<uint32_t> probe_res;
	//! sel vector for probe results (no need to clear between probes)
	SelectionVector probe_sel;

	//! Probe BFs in current order
	void ProbeBFs(DataChunk &input) {
		if (input.size() == 0) {
			return;
		}

		// auto start = high_resolution_clock::now();

		for (size_t idx = 0; idx < probe_order.size(); idx++) {
			// TODO: why is this needed???????
			// probe_sel.Initialize();

			auto &filter = lip_filters[probe_order[idx]];
			// TODO: there has to be a better way to do this lmao
			//  e.g. Lookup can just return a new chunk ? or at least a sel vector
			// filter->Lookup(chunk, probe_res, hash_staging);
			filter->Lookup(input, probe_res, hash_staging);

			idx_t result_count = 0;
			for (idx_t i = 0; i < input.size(); i++) {
				probe_sel.set_index(result_count, i);
				result_count += probe_res[i];
				// if (probe_res[i] > 0) {
				// 	probe_sel.set_index(result_count, i);
				// 	result_count++;
				// }
			}
			bf_hit_counts[idx] += result_count;
			bf_total_counts[idx] += input.size();

			if (result_count == 0) {
				input.Slice(0, 0);
				break;
			}
			if (result_count != input.size()) {
				input.Slice(probe_sel, result_count);
			}
		}

		chunks_processed++;
		if (chunks_processed == batch_size) {
			ReorderProbes();
		}

		// auto end = high_resolution_clock::now();
		// probe_seconds += duration_cast<std::chrono::microseconds>(end - start);
		// std::cout << "operator " << this << " probe_seconds " << probe_seconds.count() << '\n';
	}

	//! On batch completion, reorder BFs such that more selective BFs are first
	void ReorderProbes() {
		D_ASSERT(chunks_processed == batch_size);

		for (size_t i = 0; i < bf_hit_counts.size(); i++) {
			size_t bf_hit_count = bf_hit_counts[i];
			size_t bf_total_count = bf_total_counts[i];
			// minumum threshold to avoid noisy reorders
			if (bf_total_count <= 32 * batch_size) {
				bf_hit_percentages[i] = {1.0, probe_order[i]};
				continue;
			}
			double bf_hit_percentage = static_cast<double>(bf_hit_count) / static_cast<double>(bf_total_count);
			bf_hit_percentages[i] = {bf_hit_percentage, probe_order[i]};
		}

		// sort by increasing hit rate
		std::sort(bf_hit_percentages.begin(), bf_hit_percentages.end());
		for (idx_t i = 0; i < bf_hit_percentages.size(); i++) {
			probe_order[i] = bf_hit_percentages[i].second;
		}

		// Reset state for next batch
		chunks_processed = 0;
		batch_size *= 2;
		for (size_t i = 0; i < bf_hit_counts.size(); i++) {
			bf_hit_counts[i] = 0;
		}
		for (size_t i = 0; i < bf_total_counts.size(); i++) {
			bf_total_counts[i] = 0;
		}
	}
};

//! PhysicalHashJoin represents a hash loop join between two tables
class PhysicalHashJoin : public PhysicalComparisonJoin {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::HASH_JOIN;

	struct JoinProjectionColumns {
		vector<idx_t> col_idxs;
		vector<LogicalType> col_types;
	};

public:
	PhysicalHashJoin(LogicalOperator &op, PhysicalOperator &left, PhysicalOperator &right, vector<JoinCondition> cond,
	                 JoinType join_type, const vector<idx_t> &left_projection_map,
	                 const vector<idx_t> &right_projection_map, vector<LogicalType> delim_types,
	                 idx_t estimated_cardinality, unique_ptr<JoinFilterPushdownInfo> pushdown_info);
	PhysicalHashJoin(LogicalOperator &op, PhysicalOperator &left, PhysicalOperator &right, vector<JoinCondition> cond,
	                 JoinType join_type, idx_t estimated_cardinality);

	//! Initialize HT for this operator
	unique_ptr<JoinHashTable> InitializeHashTable(ClientContext &context) const;

	//! The types of the join keys
	vector<LogicalType> condition_types;

	//! The indices/types of the payload columns
	JoinProjectionColumns payload_columns;
	//! The indices/types of the lhs columns that need to be output
	JoinProjectionColumns lhs_output_columns;
	//! The indices/types of the rhs columns that need to be output
	JoinProjectionColumns rhs_output_columns;

	//! Duplicate eliminated types; only used for delim_joins (i.e. correlated subqueries)
	vector<LogicalType> delim_types;

	//! Join Keys statistics (optional)
	vector<unique_ptr<BaseStatistics>> join_stats;

	/** LIP ******************************************************************/
public:
	uint8_t lip_type = LIP_NONE;
	shared_ptr<BloomFilterUsage> bf_build;
	vector<shared_ptr<BloomFilterUsage>> bf_probe;

public:
	InsertionOrderPreservingMap<string> ParamsToString() const override;

public:
	// Operator Interface
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;

	bool ParallelOperator() const override {
		return true;
	}

protected:
	// CachingOperator Interface
	OperatorResultType ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                                   GlobalOperatorState &gstate, OperatorState &state) const override;

	// Source interface
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override;

	ProgressData GetProgress(ClientContext &context, GlobalSourceState &gstate) const override;

	//! Becomes a source when it is an external join
	bool IsSource() const override {
		return true;
	}

	bool ParallelSource() const override {
		return true;
	}

public:
	// Sink Interface
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	void PrepareFinalize(ClientContext &context, GlobalSinkState &global_state) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return true;
	}
};

} // namespace duckdb
