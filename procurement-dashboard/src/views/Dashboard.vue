<template>
  <dashboard-layout>
    <template #header>
      PAVUS AI Procurement Dashboard
    </template>
    
    <div class="mb-6">
      <h1 class="text-2xl font-bold text-gray-800 mb-2">Welcome to the Procurement Analytics Platform</h1>
      <p class="text-gray-600">
        Explore procurement data, supplier insights, and cost-saving opportunities powered by AI.
      </p>
    </div>
    
    <!-- Overview Statistics -->
    <div class="dashboard-grid mb-6">
      <kpi-card
        title="Total Spend"
        :value="overviewData.totalSpend"
        format="currency"
        description="Across all categories"
        :trend="overviewData.spendTrend"
        trendDescription="increase from last year"
      />
      
      <kpi-card
        title="Categorization Rate"
        :value="dataQualityData.summary.categorization_rate * 100"
        format="percent"
        description="Transactions successfully categorized"
        :trend="dataQualityData.improvements[0].improvement * 100"
        trendDescription="improvement from previous"
        color="green"
      />
      
      <kpi-card
        title="Active Suppliers"
        :value="overviewData.suppliersCount"
        description="Across multiple categories"
        color="secondary"
      />
    </div>
    
    <!-- Quick Insights -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
      <div class="card">
        <h3 class="card-title mb-4">Category Distribution</h3>
        <doughnut-chart :chart-data="categoryDistributionData" />
      </div>
      
      <div class="card">
        <h3 class="card-title mb-4">Monthly Spend Trend</h3>
        <line-chart :chart-data="monthlySpendData" />
      </div>
    </div>
    
    <!-- Model Performance and Savings -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
      <div class="card">
        <h3 class="card-title mb-4">Model Accuracy by Level</h3>
        <bar-chart :chart-data="modelAccuracyData" />
      </div>
      
      <div class="card">
        <h3 class="card-title mb-4">Cost Saving Opportunities</h3>
        <table class="min-w-full divide-y divide-gray-200">
          <thead class="bg-gray-50">
            <tr>
              <th class="table-header">Opportunity</th>
              <th class="table-header">Category</th>
              <th class="table-header">Estimated Savings</th>
            </tr>
          </thead>
          <tbody class="bg-white divide-y divide-gray-200">
            <tr v-for="(opportunity, index) in costSavingOpportunities" :key="index">
              <td class="table-cell">{{ opportunity.type }}</td>
              <td class="table-cell">{{ opportunity.category }}</td>
              <td class="table-cell">${{ opportunity.savings.toLocaleString() }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
    
    <!-- Recent Activity -->
    <div class="card">
      <h3 class="card-title mb-4">Recent Activity</h3>
      <div class="space-y-4">
        <div class="flex items-start p-3 bg-gray-50 rounded">
          <div class="text-primary-500 text-2xl mr-3">🔄</div>
          <div>
            <h4 class="font-medium">Model Updated</h4>
            <p class="text-sm text-gray-600">Model version 1.2.5 has been deployed with 2% improved accuracy</p>
            <p class="text-xs text-gray-500 mt-1">2 hours ago</p>
          </div>
        </div>
        
        <div class="flex items-start p-3 bg-gray-50 rounded">
          <div class="text-green-500 text-2xl mr-3">💰</div>
          <div>
            <h4 class="font-medium">New Cost Saving Opportunity</h4>
            <p class="text-sm text-gray-600">
              {{ costSavingOpportunities && costSavingOpportunities.length > 0 ? 
                `${costSavingOpportunities[0].type} for ${costSavingOpportunities[0].category} could save $${costSavingOpportunities[0].savings.toLocaleString()}` : 
                'New cost saving opportunity identified' }}
            </p>
            <p class="text-xs text-gray-500 mt-1">Yesterday</p>
          </div>
        </div>
        
        <div class="flex items-start p-3 bg-gray-50 rounded">
          <div class="text-yellow-500 text-2xl mr-3">⚠️</div>
          <div>
            <h4 class="font-medium">Low Confidence Items</h4>
            <p class="text-sm text-gray-600">
              {{ dataQualityData.issues && dataQualityData.issues.length > 3 ? 
                `${dataQualityData.issues[3].count} items require manual review` : 
                'Items require manual review for categorization' }}
            </p>
            <p class="text-xs text-gray-500 mt-1">2 days ago</p>
          </div>
        </div>
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
import { 
  overviewData, 
  kpiData, 
  modelPerformanceData, 
  dataQualityData,
  costSavingsData
} from '@/data/dummyData';

export default {
  name: 'DashboardView',
  components: {
    DashboardLayout,
    KpiCard,
    BarChart,
    LineChart,
    DoughnutChart
  },
  data() {
    // Create model accuracy data safely
    const modelAccuracyData = {
      labels: modelPerformanceData && modelPerformanceData.accuracyByLevel ? 
        modelPerformanceData.accuracyByLevel.map(level => `Level ${level.level}`) : 
        ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
      datasets: [{
        label: 'Accuracy',
        data: modelPerformanceData && modelPerformanceData.accuracyByLevel ? 
          modelPerformanceData.accuracyByLevel.map(level => level.accuracy * 100) : 
          [96, 93, 90, 87, 81],
        backgroundColor: '#4338ca',
      }, {
        label: 'F1 Score',
        data: modelPerformanceData && modelPerformanceData.accuracyByLevel ? 
          modelPerformanceData.accuracyByLevel.map(level => level.f1_score * 100) : 
          [95, 92, 89, 85, 79],
        backgroundColor: '#0ea5e9',
      }]
    };
    
    return {
      // Use overviewData for KPI cards
      overviewData,
      dataQualityData,
      
      // Chart data for Category Distribution
      categoryDistributionData: {
        labels: kpiData.categoryDistribution.map(item => item.category),
        datasets: [{
          data: kpiData.categoryDistribution.map(item => item.value),
          backgroundColor: [
            '#4338ca', // primary-700
            '#6366f1', // primary-500
            '#818cf8', // primary-400
            '#a5b4fc', // primary-300
            '#e0e7ff'  // primary-100
          ],
          borderWidth: 0
        }]
      },
      
      // Chart data for Monthly Spend
      monthlySpendData: {
        labels: kpiData.spendByMonth.map(item => item.month),
        datasets: [{
          label: '2025 Spend',
          data: kpiData.spendByMonth.map(item => item.spend2025),
          borderColor: '#4338ca',
          backgroundColor: 'rgba(67, 56, 202, 0.1)',
          fill: true,
          tension: 0.4
        }, {
          label: '2024 Spend',
          data: kpiData.spendByMonth.map(item => item.spend2024),
          borderColor: '#0ea5e9',
          backgroundColor: 'rgba(14, 165, 233, 0.1)',
          fill: true,
          tension: 0.4
        }]
      },
      
      // Chart data for Model Accuracy (created safely above)
      modelAccuracyData,
      
      // Cost saving opportunities
      costSavingOpportunities: costSavingsData && costSavingsData.opportunities ? 
        costSavingsData.opportunities : []
    };
  }
};
</script>