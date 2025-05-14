<template>
  <dashboard-layout>
    <template #header>
      Data Quality Monitor
    </template>
    
    <!-- KPI Metrics -->
<div class="dashboard-grid mb-6">
  <kpi-card
    title="Categorization Rate"
    :value="dataQuality.categorization_rate * 100"
    format="percent"
    description="Transactions categorized"
    :trend="dataQuality.recent_improvements[0].improvement * 100"
    trendDescription="improvement"
    color="green"
  />
  
  <kpi-card
    title="High Confidence"
    :value="dataQuality.high_confidence_categories * 100"
    format="percent"
    description="Categories with high confidence"
    :trend="dataQuality.recent_improvements[1].improvement * 100"
    trendDescription="improvement"
    color="primary"
  />
  
  <kpi-card
    title="Data Completeness"
    :value="getAverageCompleteness() * 100"
    format="percent"
    description="Average across all data fields"
    color="secondary"
  />
</div>

<!-- Data Scan Info -->
<div class="card mb-6">
  <div class="flex justify-between items-start mb-4">
    <div>
      <h3 class="card-title">Data Quality Scan</h3>
      <p class="text-gray-500">Latest data quality assessment</p>
    </div>
    <div class="bg-primary-50 text-primary-700 px-3 py-1 rounded-full text-sm font-medium">
      {{ dataQuality.scan_date }}
    </div>
  </div>
  
  <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
    <div class="bg-gray-50 p-4 rounded-lg">
      <p class="text-sm text-gray-500">Total Transactions</p>
      <p class="font-medium">{{ dataQuality.total_transactions.toLocaleString() }}</p>
    </div>
    <div class="bg-gray-50 p-4 rounded-lg">
      <p class="text-sm text-gray-500">Total Transactions</p>
      <p class="font-medium">{{ dataQuality.total_transactions.toLocaleString() }}</p>
    </div>
    <div class="bg-gray-50 p-4 rounded-lg">
      <p class="text-sm text-gray-500">Total Spend</p>
      <p class="font-medium">${{ dataQuality.total_spend.toLocaleString() }}</p>
    </div>
    <div class="bg-gray-50 p-4 rounded-lg">
      <p class="text-sm text-gray-500">Categorized Transactions</p>
      <p class="font-medium">{{ dataQuality.categorized_transactions.toLocaleString() }}</p>
    </div>
    <div class="bg-gray-50 p-4 rounded-lg">
      <p class="text-sm text-gray-500">Uncategorized Items</p>
      <p class="font-medium">{{ (dataQuality.total_transactions - dataQuality.categorized_transactions).toLocaleString() }}</p>
    </div>
  </div>
</div>

<!-- Data Completeness by Field -->
<div class="card mb-6">
  <h3 class="card-title mb-4">Data Completeness by Field</h3>
  <bar-chart :chart-data="completenessData" :options="horizontalBarOptions" />
</div>

<!-- Data Quality Issues and Improvements -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
  <div class="card">
    <h3 class="card-title mb-4">Data Quality Issues</h3>
    <div class="overflow-x-auto">
      <table class="min-w-full divide-y divide-gray-200">
        <thead class="bg-gray-50">
          <tr>
            <th class="table-header">Issue Type</th>
            <th class="table-header">Count</th>
            <th class="table-header">Impact</th>
            <th class="table-header">Status</th>
          </tr>
        </thead>
        <tbody class="bg-white divide-y divide-gray-200">
          <tr v-for="(issue, index) in dataQuality.quality_issues" :key="index">
            <td class="table-cell">{{ issue.issue_type }}</td>
            <td class="table-cell">{{ issue.count.toLocaleString() }}</td>
            <td class="table-cell">
              <span class="px-2 py-1 text-xs rounded-full" 
                :class="getImpactClass(issue.impact)">
                {{ issue.impact }}
              </span>
            </td>
            <td class="table-cell">
              <button class="px-3 py-1 bg-primary-100 text-primary-600 rounded-full text-xs font-medium">
                Review
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
  
  <div class="card">
    <h3 class="card-title mb-4">Recent Improvements</h3>
    <div class="space-y-4">
      <div v-for="(improvement, index) in dataQuality.recent_improvements" :key="index" 
        class="bg-gray-50 p-4 rounded-lg">
        <div class="flex justify-between mb-1">
          <span class="font-medium">{{ improvement.metric }}</span>
          <span class="text-green-600">+{{ (improvement.improvement * 100).toFixed(1) }}%</span>
        </div>
        <div class="w-full bg-gray-200 rounded-full h-2.5">
          <div class="bg-primary-600 h-2.5 rounded-full" 
            :style="`width: ${improvement.current * 100}%`"></div>
        </div>
        <div class="flex justify-between mt-1 text-xs text-gray-500">
          <span>Previous: {{ (improvement.previous * 100).toFixed(1) }}%</span>
          <span>Current: {{ (improvement.current * 100).toFixed(1) }}%</span>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Items Requiring Review -->
<div class="card">
  <div class="flex justify-between items-center mb-4">
    <h3 class="card-title">Items Requiring Review</h3>
    <div>
      <button class="px-4 py-2 bg-primary-600 text-white rounded-lg text-sm font-medium mr-2">
        Review All
      </button>
      <button class="px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium">
        Filter
      </button>
    </div>
  </div>
  
  <div class="overflow-x-auto">
    <table class="min-w-full divide-y divide-gray-200">
      <thead class="bg-gray-50">
        <tr>
          <th class="table-header">Item ID</th>
          <th class="table-header">Description</th>
          <th class="table-header">Issue Type</th>
          <th class="table-header">Suggested Action</th>
          <th class="table-header">Actions</th>
        </tr>
      </thead>
      <tbody class="bg-white divide-y divide-gray-200">
        <tr v-for="(item, index) in itemsRequiringReview" :key="index">
          <td class="table-cell">{{ item.id }}</td>
          <td class="table-cell">{{ item.description }}</td>
          <td class="table-cell">{{ item.issueType }}</td>
          <td class="table-cell">{{ item.suggestedAction }}</td>
          <td class="table-cell">
            <button class="px-3 py-1 bg-green-100 text-green-600 rounded-full text-xs font-medium mr-1">
              Approve
            </button>
            <button class="px-3 py-1 bg-red-100 text-red-600 rounded-full text-xs font-medium">
              Reject
            </button>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</div>
  
    
  </dashboard-layout>
</template>

<script>
import DashboardLayout from '@/layouts/DashboardLayout.vue';
import KpiCard from '@/components/ui/KpiCard.vue';
import BarChart from '@/components/charts/BarChart.vue';
import { dataQuality } from '@/data/procurementData';

// At the top of each component file
import { 
  dataQualityData 
} from '@/data/dummyData';

export default {
  name: 'DataQuality',
  components: {
    DashboardLayout,
    KpiCard,
    BarChart
  },
  data() {
  return {
    dataQuality: {
      scan_date: "2025-05-01",
      total_transactions: dataQualityData.summary.totalTransactions,
      total_spend: 25487651.25,
      categorized_transactions: dataQualityData.summary.categorizedTransactions,
      categorization_rate: dataQualityData.summary.categorization_rate, 
      high_confidence_categories: dataQualityData.summary.high_confidence_categories,
      completeness_metrics: dataQualityData.completeness,
      quality_issues: dataQualityData.issues,
      recent_improvements: dataQualityData.improvements
    },
    
    // Chart data for Data Completeness
    completenessData: {
      labels: Object.keys(dataQualityData.completeness).map(this.formatMetricName),
      datasets: [{
        label: 'Completeness Rate',
        data: Object.values(dataQualityData.completeness).map(value => value * 100),
        backgroundColor: function(context) {
          const value = context.dataset.data[context.dataIndex];
          return value >= 99 ? '#22c55e' : value >= 95 ? '#f59e0b' : '#ef4444';
        },
      }]
    },
    
    // Horizontal bar chart options
    horizontalBarOptions: {
      indexAxis: 'y',
      scales: {
        x: {
          beginAtZero: true,
          max: 100,
          title: {
            display: true,
            text: 'Completeness (%)'
          }
        }
      }
    },
    
    // Items needing review
    itemsRequiringReview: [
      { id: "ITM-5423", description: "INDUSTRIAL BELT FOR HEAVY LOAD", issueType: "Missing Category", suggestedAction: "Assign to 'Leather straps'" },
      { id: "ITM-6712", description: "CONVEYOR SYSTEM REPLACEMENT PART", issueType: "Low Confidence", suggestedAction: "Review classification" },
      { id: "ITM-8901", description: "AKRON MFCG CUSTOM PART #AB123-45", issueType: "Incomplete Description", suggestedAction: "Enrich description" }
    ]
  };
},
  methods: {
    formatMetricName(name) {
      // Format metric names for display (e.g., convert snake_case to Title Case)
      return name.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
      ).join(' ');
    },
    getAverageCompleteness() {
      const metrics = dataQuality.completeness_metrics;
      const values = Object.values(metrics);
      return values.reduce((acc, val) => acc + val, 0) / values.length;
    },
    getImpactClass(impact) {
      const impactClassMap = {
        'Low': 'bg-green-100 text-green-800',
        'Medium': 'bg-yellow-100 text-yellow-800',
        'High': 'bg-red-100 text-red-800'
      };
      return impactClassMap[impact] || 'bg-gray-100 text-gray-800';
    }
  }
};
</script>