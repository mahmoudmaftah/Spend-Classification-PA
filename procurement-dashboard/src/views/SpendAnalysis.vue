<template>
  <dashboard-layout>
    <template #header>
      Spend Analysis Dashboard
    </template>
    
    <!-- KPI Metrics -->
<div class="dashboard-grid mb-6">
  <kpi-card
    title="Total Spend"
    :value="categoryAnalytics.total_spend"
    format="currency"
    description="For Leather straps category"
    :trend="categoryAnalytics.spend_change_percent"
    trendDescription="vs previous year"
  />
  
  <kpi-card
    title="Transactions"
    :value="categoryAnalytics.transaction_count"
    description="Across all suppliers"
  />
  
  <kpi-card
    title="Average Transaction"
    :value="categoryAnalytics.average_transaction_value"
    format="currency"
    description="Per transaction value"
  />
</div>

<!-- Main Visualizations -->
<div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
  <!-- Hierarchical Spend Breakdown -->
  <div class="card lg:col-span-2">
    <h3 class="card-title mb-4">Hierarchical Spend Breakdown</h3>
    <div class="chart-container">
      <treemap-chart :data="hierarchicalSpendData" />
    </div>
  </div>
  
  <!-- Supplier Distribution -->
  <div class="card">
    <h3 class="card-title mb-2">Supplier Distribution</h3>
    <p class="text-sm text-gray-500 mb-4">Top suppliers by spend</p>
    <doughnut-chart :chart-data="supplierDistributionData" />
  </div>
</div>

<!-- Spend Trends & Category Analysis -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
  <!-- Spend Trend Over Time -->
  <div class="card">
    <h3 class="card-title mb-4">Spend Trend Over Time</h3>
    <line-chart :chart-data="spendTrendData" />
  </div>
  
  <!-- Price Analysis -->
  <div class="card">
    <h3 class="card-title mb-4">Price Analysis</h3>
    <bar-chart :chart-data="priceAnalysisData" />
  </div>
</div>

<!-- Supplier-Category Matrix -->
<div class="card mb-6">
  <h3 class="card-title mb-4">Supplier-Category Matrix</h3>
  <div class="chart-container">
    <heatmap-chart :data="supplierCategoryMatrixData" />
  </div>
</div>

<!-- Top Suppliers Table -->
<div class="card">
  <h3 class="card-title mb-4">Top Suppliers by Spend</h3>
  <div class="overflow-x-auto">
    <table class="min-w-full divide-y divide-gray-200">
      <thead class="bg-gray-50">
        <tr>
          <th class="table-header">Supplier</th>
          <th class="table-header">Spend</th>
          <th class="table-header">% of Category</th>
          <th class="table-header">Transactions</th>
          <th class="table-header">Avg. Price</th>
        </tr>
      </thead>
      <tbody class="bg-white divide-y divide-gray-200">
        <tr v-for="(supplier, index) in categoryAnalytics.top_suppliers" :key="index">
          <td class="table-cell font-medium">{{ supplier.name }}</td>
          <td class="table-cell">${{ supplier.spend.toLocaleString() }}</td>
          <td class="table-cell">{{ supplier.percent }}%</td>
          <td class="table-cell">{{ Math.round(supplier.spend / categoryAnalytics.average_transaction_value) }}</td>
          <td class="table-cell">${{ Math.round(supplier.spend / Math.round(supplier.spend / categoryAnalytics.average_transaction_value)) }}</td>
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
import LineChart from '@/components/charts/LineChart.vue';
import DoughnutChart from '@/components/charts/DoughnutChart.vue';
import TreemapChart from '@/components/charts/TreemapChart.vue';
import HeatmapChart from '@/components/charts/HeatmapChart.vue';
// import { categoryAnalytics } from '@/data/procurementData';
// At the top of each component file
import { 
  spendAnalysisData, 
} from '@/data/dummyData';

export default {
  name: 'SpendAnalysisView',
  components: {
    DashboardLayout,
    KpiCard,
    BarChart,
    LineChart,
    DoughnutChart,
    TreemapChart,
    HeatmapChart
  },
  data() {
  return {
    categoryAnalytics: {
      total_spend: 158432.75,
      transaction_count: 245,
      average_transaction_value: 646.66,
      spend_change_percent: 8.53,
      supplier_count: 12,
      top_suppliers: spendAnalysisData.topSuppliers,
      price_range: {
        min: 87.50, 
        max: 250.00, 
        average: 125.75, 
        median: 115.50
      }
    },
    
    // Hierarchical Spend Data
    hierarchicalSpendData: spendAnalysisData.hierarchicalSpend,
    
    // Chart data for Supplier Distribution
    supplierDistributionData: {
      labels: spendAnalysisData.topSuppliers.map(supplier => supplier.name).concat(['Others']),
      datasets: [{
        data: [
          ...spendAnalysisData.topSuppliers.map(supplier => supplier.spend),
          // Calculate 'Others' as the remaining percentage
          158432.75 - spendAnalysisData.topSuppliers.reduce((acc, supplier) => acc + supplier.spend, 0)
        ],
        backgroundColor: [
          '#4338ca', // primary-700
          '#6366f1', // primary-500
          '#818cf8', // primary-400
          '#e0e7ff'  // primary-100
        ],
        borderWidth: 0
      }]
    },
    
    // Chart data for Spend Trend
    spendTrendData: {
      labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
      datasets: [{
        label: '2025 YTD',
        data: [7500, 8200, 7800, 9500, 9800, 0], // Example data
        borderColor: '#4338ca',
        backgroundColor: 'rgba(67, 56, 202, 0.1)',
        fill: true,
        tension: 0.4
      }, {
        label: '2024',
        data: [7000, 7800, 7200, 8800, 9200, 10000], // Example data
        borderColor: '#0ea5e9',
        backgroundColor: 'rgba(14, 165, 233, 0.1)',
        fill: true,
        tension: 0.4
      }]
    },
    
    // Chart data for Price Analysis
    priceAnalysisData: {
      labels: spendAnalysisData.topSuppliers.map(supplier => supplier.name),
      datasets: [{
        label: 'Avg. Price',
        data: spendAnalysisData.topSuppliers.map(supplier => 
          Math.round(supplier.spend / Math.round(supplier.spend / 646.66))
        ),
        backgroundColor: '#4338ca',
      }, {
        label: 'Market Avg.',
        data: spendAnalysisData.topSuppliers.map(() => 125.75),
        backgroundColor: '#0ea5e9',
        type: 'line',
        fill: false,
        tension: 0
      }]
    },
    
    // Supplier-Category Matrix Data
    supplierCategoryMatrixData: spendAnalysisData.supplierCategoryMatrix
  };
}
};
</script>