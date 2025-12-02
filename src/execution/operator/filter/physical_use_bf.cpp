#include "duckdb/execution/operator/filter/physical_use_bf.hpp"

#include "duckdb/execution/adaptive_filter.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/table_filter_state.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/storage/table/column_segment.hpp"

#include <iostream>

namespace duckdb {
PhysicalUseBF::PhysicalUseBF(vector<LogicalType> types, const shared_ptr<FilterPlan> &filter_plan,
                             unique_ptr<BloomFilterUsage> bf, PhysicalCreateBF *related_create_bfs,
                             idx_t estimated_cardinality, shared_ptr<DynamicTableFilterSet> min_max_to_use)
    : CachingPhysicalOperator(PhysicalOperatorType::USE_BF, std::move(types), estimated_cardinality),
      filter_plan(filter_plan), related_creator(related_create_bfs), bf_to_use(std::move(bf)),
      min_max_to_use(std::move(min_max_to_use)) {
}

class UseBFState : public CachingOperatorState {
public:
	static constexpr int64_t NUM_CHUNK_FOR_CHECK = 32;
	static constexpr double SELECTIVITY_THRESHOLD = 0.9;

public:
	static void PopulateConstantFilters(idx_t col_idx, unique_ptr<TableFilter> filter, vector<pair<idx_t, unique_ptr<TableFilter>>> &filters) {
		if (filter->filter_type == TableFilterType::CONSTANT_COMPARISON) {
			filters.emplace_back(col_idx, std::move(filter));
		} else if (filter->filter_type == TableFilterType::CONJUNCTION_AND) {
			for (auto &f : filter->Cast<ConjunctionAndFilter>().child_filters) {
				PopulateConstantFilters(col_idx, std::move(f), filters);
			}
		} else {
			throw NotImplementedException("TableFilterType not implemented in UseBFState::PopulateConstantFilters");
		}
	}

	UseBFState(const PhysicalUseBF &op, ClientContext &context, bool valid_bf)
	    : sel_vector(STANDARD_VECTOR_SIZE), lookup_results(STANDARD_VECTOR_SIZE), use_bf(valid_bf) {
		if (op.min_max_to_use && op.min_max_to_use->HasFilters()) {
			auto filter_set = op.min_max_to_use->GetFinalTableFilters(nullptr);
			filter_set->UnifyFilters();
			for (auto &entry : filter_set->filters) {
				min_max_states.push_back(TableFilterState::Initialize(context, *entry.second));
				min_max_to_use.push_back(std::move(entry));
				// PopulateConstantFilters(entry.first, std::move(entry.second), min_max_to_use);
			}
			if (ClientConfig::GetConfig(context).filter_mode == FILTER_ADAPT) {
				adaptive_filter = make_uniq<AdaptiveFilter>(min_max_to_use);
			}
			// min_max_to_use.reserve(filter_set->filters.size());
			// std::cout << "For UseBF:\n" << op.ToString();
			// for (auto &entry : filter_set->filters) {
			// 	std::cout << "- column " << entry.first << " has type " << op.GetTypes()[entry.first].ToString() << '\n';
			// 	min_max_to_use.push_back(std::move(entry));
			// }
			min_max_chunk.Initialize(context, op.types);
		}
	}

	SelectionVector sel_vector;
	vector<uint32_t> lookup_results;

	vector<pair<idx_t, unique_ptr<TableFilter>>> min_max_to_use;
	vector<unique_ptr<TableFilterState>> min_max_states;
	unique_ptr<AdaptiveFilter> adaptive_filter;
	DataChunk min_max_chunk;

	bool use_bf;
	bool is_checked = false;
	int64_t num_chunk = 0;
	uint64_t num_received = 0;
	uint64_t num_sent = 0;

public:
	void CheckBFSelectivity(uint64_t num_in, uint64_t num_out) {
		num_received += num_in;
		num_sent += num_out;
		num_chunk++;

		if (num_chunk > NUM_CHUNK_FOR_CHECK) {
			is_checked = true;

			double selectivity = static_cast<double>(num_sent) / static_cast<double>(num_received);
			if (selectivity > SELECTIVITY_THRESHOLD && selectivity < 1) {
				use_bf = false;
			}
		}
	}

	void Finalize(const PhysicalOperator &op, ExecutionContext &context) override {
		context.thread.profiler.Flush(op);
	}
};

unique_ptr<OperatorState> PhysicalUseBF::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<UseBFState>(*this, context.client, bf_to_use->IsValid());
}

InsertionOrderPreservingMap<string> PhysicalUseBF::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["BF Creators"] = "0x" + std::to_string(reinterpret_cast<size_t>(related_creator)) + "\n";

	string bound_cols_apply;
	for (const auto &col : filter_plan->bound_cols_apply) {
		bound_cols_apply += std::to_string(col) + " ";
	}
	result["Apply (Bound)"] = bound_cols_apply;

	if (min_max_to_use && min_max_to_use->HasFilters()) {
		string dynamic_info;
		bool first_item = true;
		auto filters = min_max_to_use->GetFinalTableFilters(nullptr);
		if (filters) {
			for (auto &f : filters->filters) {
				auto &column_index = f.first;
				auto &filter = f.second;
				if (!first_item) {
					dynamic_info += "\n";
				}
				first_item = false;

				dynamic_info += filter->ToString("col" + std::to_string(column_index));
			}
		}
		result["Dynamic Filters"] = dynamic_info;
	} else {
		result["Dynamic Filters"] = "None";
	}

	return result;
}

void PhysicalUseBF::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();

	auto &state = meta_pipeline.GetState();
	state.AddPipelineOperator(current, *this);
	related_creator->BuildPipelinesFromRelated(current, meta_pipeline);
	children[0].get().BuildPipelines(current, meta_pipeline);
}

OperatorResultType PhysicalUseBF::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                  GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = state_p.Cast<UseBFState>();

	// Begin adaptive min-max filtering
	SelectionVector min_max_sel;
	idx_t input_size = input.size();
	idx_t approved_tuple_count = input_size;
	if (state.adaptive_filter) {
		auto start = state.adaptive_filter->BeginFilter();
		for (idx_t i = 0; i < state.min_max_to_use.size(); i++) {
			idx_t perm_idx = state.adaptive_filter->permutation[i];
			auto &info = state.min_max_to_use[perm_idx];
			auto column_idx = info.first;
			auto &filter = *info.second;
			auto &filter_state = *state.min_max_states[perm_idx];

			auto &col_vec = input.data[column_idx];

			UnifiedVectorFormat vdata;
			col_vec.ToUnifiedFormat(input.size(), vdata);
			ColumnSegment::FilterSelection(min_max_sel, col_vec, vdata, filter, filter_state, input_size, approved_tuple_count);

			// auto &filter = info.second->Cast<ConstantFilter>();
			//
			// auto &col = input.data[column_idx];
			// auto new_sel = SelectionVector(approved_tuple_count);
			//
			// idx_t result_count = 0;
			// for (idx_t j = 0; j < approved_tuple_count; j++) {
			// 	auto idx = min_max_sel.get_index(j);
			// 	auto value = col.GetValue(idx);
			// 	bool comparison_result = !value.IsNull() && filter.Compare(value);
			// 	new_sel.set_index(result_count, idx);
			// 	result_count += comparison_result;
			// }
			// approved_tuple_count = result_count;

			// min_max_sel.Initialize(new_sel);
		}
		state.adaptive_filter->EndFilter(start);
		if (approved_tuple_count != input.size()) {
			input.Slice(min_max_sel, approved_tuple_count);
		}
	} else {
		for (idx_t i = 0; i < state.min_max_to_use.size(); i++) {
			idx_t perm_idx = state.adaptive_filter->permutation[i];
			auto &info = state.min_max_to_use[perm_idx];
			auto column_idx = info.first;
			auto &filter = *info.second;
			auto &filter_state = *state.min_max_states[perm_idx];

			auto &col_vec = input.data[column_idx];

			UnifiedVectorFormat vdata;
			col_vec.ToUnifiedFormat(input.size(), vdata);
			ColumnSegment::FilterSelection(min_max_sel, col_vec, vdata, filter, filter_state, input_size, approved_tuple_count);

			// auto &filter = info.second->Cast<ConstantFilter>();
			//
			// auto &col = input.data[column_idx];
			// auto new_sel = SelectionVector(approved_tuple_count);
			//
			// idx_t result_count = 0;
			// for (idx_t j = 0; j < approved_tuple_count; j++) {
			// 	auto idx = min_max_sel.get_index(j);
			// 	auto value = col.GetValue(idx);
			// 	bool comparison_result = !value.IsNull() && filter.Compare(value);
			// 	new_sel.set_index(result_count, idx);
			// 	result_count += comparison_result;
			// }
			// approved_tuple_count = result_count;

			// min_max_sel.Initialize(new_sel);
		}
		if (approved_tuple_count != input.size()) {
			input.Slice(min_max_sel, approved_tuple_count);
		}
	}

	// This operator has no BloomFilter to use
	if (input.size() == 0 || !state.use_bf) {
		chunk.Reference(input);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// 1. Lookup the BloomFilter
	bf_to_use->Lookup(input, state.lookup_results);

	// 2. Fill results
	idx_t result_count = 0;
	auto &sel = state.sel_vector;
	for (size_t i = 0; i < input.size(); i++) {
		sel.set_index(result_count, i);
		result_count += state.lookup_results[i];
	}
	if (result_count == input.size()) {
		// nothing was filtered: skip adding any selection vectors
		chunk.Reference(input);
	} else {
		chunk.Slice(input, sel, result_count);
	}

	// 3. Update statistics
	if (!state.is_checked) {
		state.CheckBFSelectivity(input.size(), result_count);
	}

	return OperatorResultType::NEED_MORE_INPUT;
}
} // namespace duckdb
