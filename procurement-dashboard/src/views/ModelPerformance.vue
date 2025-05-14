<template>
  <dashboard-layout>
    <template #header>
      Model Performance Metrics
    </template>
    
    <!-- KPI Metrics -->
<div class="dashboard-grid mb-6">
  <kpi-card
    title="Overall Accuracy"
    :value="modelPerformance.overall_accuracy * 100"
    format="percent"
    description="Across all category levels"
    color="green"
  />
  
  <kpi-card
    title="F1 Score"
    :value="modelPerformance.overall_f1_score"
    description="Balanced precision and recall"
    color="primary"
  />
  
  <kpi-card
    title="Low Confidence"
    :value="modelPerformance.low_confidence_predictions * 100"
    format="percent"
    description="Predictions requiring review"
    color="yellow"
  />
</div>

<!-- Model Details -->
<div class="card mb-6">
  <div class="flex justify-between items-start mb-4">
    <div>
      <h3 class="card-title">Model Information</h3>
      <p class="text-gray-500">Current deployed model details</p>
    </div>
    <div class="bg-primary-50 text-primary-700 px-3 py-1 rounded-full text-sm font-medium">
      v{{ modelPerformance.model_version }}
    </div>
  </div>
  
  <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
    <div class="bg-gray-50 p-4 rounded-lg">
      <p class="text-sm text-gray-500">Model Version</p>
      <p class="font-medium">{{ modelPerformance.model_version }}</p>
    </div>
    <div class="bg-gray-50 p-4 rounded-lg">
      <p class="text-sm text-gray-500">Training Date</p>
      <p class="font-medium">{{ modelPerformance.training_date }}</p>
    </div>
    <div class="bg-gray-50 p-4 rounded-lg">
      <p class="text-sm text-gray-500">Training Data Points</p>
      <p class="font-medium">{{ modelPerformance.data_points_trained.toLocaleString() }}</p>
    </div>
    <div class="bg-gray-50 p-4 rounded-lg">
      <p class="text-sm text-gray-500">Model Type</p>
      <p class="font-medium">Deep Learning Classifier</p>
    </div>
  </div>
</div>

<!-- Accuracy by Level -->
<div class="card mb-6">
  <h3 class="card-title mb-4">Performance by Taxonomy Level</h3>
  <bar-chart :chart-data="accuracyByLevelData" />
</div>

<!-- Feature Importance and Confidence Distribution -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
  <div class="card">
    <h3 class="card-title mb-4">Feature Importance</h3>
    <bar-chart :chart-data="featureImportanceData" :options="horizontalBarOptions" />
  </div>
  
  <div class="card">
    <h3 class="card-title mb-4">Confidence Distribution</h3>
    <bar-chart :chart-data="confidenceDistributionData" />
  </div>
</div>

<!-- Common Misclassifications -->
<div class="card">
  <h3 class="card-title mb-4">Common Misclassifications</h3>
  <div class="overflow-x-auto">
    <table class="min-w-full divide-y divide-gray-200">
      <thead class="bg-gray-50">
        <tr>
          <th class="table-header">From Category</th>
          <th class="table-header">To Category</th>
          <th class="table-header">Count</th>
          <th class="table-header">Confusion Rate</th>
          <th class="table-header">Status</th>
        </tr>
      </thead>
      <tbody class="bg-white divide-y divide-gray-200">
        <tr v-for="(misclass, index) in modelPerformance.common_misclassifications" :key="index">
          <td class="table-cell">{{ getCategoryName(misclass.from_category) }}</td>
          <td class="table-cell">{{ getCategoryName(misclass.to_category) }}</td>
          <td class="table-cell">{{ misclass.count }}</td>
          <td class="table-cell">{{ (misclass.confusion_rate * 100).toFixed(1) }}%</td>
          <td class="table-cell">
            <span class="px-2 py-1 text-xs rounded-full" 
              :class="misclass.confusion_rate > 0.1 ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800'">
              {{ misclass.confusion_rate > 0.1 ? 'Needs Attention' : 'Acceptable' }}
            </span>
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
// import { modelPerformance } from '@/data/procurementData';

// At the top of each component file
import { 
  modelPerformanceData, 
} from '@/data/dummyData';

export default {
  name: 'ModelPerformance',
  components: {
    DashboardLayout,
    KpiCard,
    BarChart
  },
  data() {
  return {
    modelPerformance: {
      model_version: "1.2.5",
      training_date: "2025-04-01",
      data_points_trained: 50432,
      overall_accuracy: 0.89,
      overall_f1_score: 0.87,
      low_confidence_predictions: 0.08,
      level_performance: modelPerformanceData.accuracyByLevel,
      feature_importance: modelPerformanceData.featureImportance,
      common_misclassifications: modelPerformanceData.commonMisclassifications
    },
    
    // Chart data for Accuracy by Level
    accuracyByLevelData: {
      labels: modelPerformanceData.accuracyByLevel.map(level => `Level ${level.level} (${level.number_of_classes} classes)`),
      datasets: [{
        label: 'Accuracy',
        data: modelPerformanceData.accuracyByLevel.map(level => level.accuracy * 100),
        backgroundColor: '#4338ca',
      }, {
        label: 'F1 Score',
        data: modelPerformanceData.accuracyByLevel.map(level => level.f1_score * 100),
        backgroundColor: '#0ea5e9',
      }]
    },
    
    // Chart data for Feature Importance
    featureImportanceData: {
      labels: modelPerformanceData.featureImportance.map(feature => this.formatFeatureName(feature.feature)),
      datasets: [{
        label: 'Importance',
        data: modelPerformanceData.featureImportance.map(feature => feature.importance * 100),
        backgroundColor: '#4338ca',
      }]
    },
    
    // Chart data for Confidence Distribution
    confidenceDistributionData: {
      labels: modelPerformanceData.confidenceDistribution.map(d => d.range),
      datasets: [{
        label: 'Predictions',
        data: modelPerformanceData.confidenceDistribution.map(d => d.count),
        backgroundColor: function(context) {
          const index = context.dataIndex;
          // Color gradient based on the confidence level
          return index < 5 ? '#f87171' : index < 8 ? '#facc15' : '#4ade80';
        },
      }]
    },
    
    // Horizontal bar chart options
    horizontalBarOptions: {
      indexAxis: 'y',
      scales: {
        x: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Importance (%)'
          }
        }
      }
    },
    
    // Category name mapping
    categoryNames: {
      '102993': 'Leather straps',
      '102994': 'Nylon straps',
      '103245': 'Conveyor belts',
      '103246': 'Rubber belts',
    }
  };
},
  methods: {
    formatFeatureName(name) {
      // Format feature names for display (e.g., convert snake_case to Title Case)
      return name.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
      ).join(' ');
    },
    getCategoryName(categoryId) {
      return this.categoryNames[categoryId] || `Category ${categoryId}`;
    }
  }
};
</script>