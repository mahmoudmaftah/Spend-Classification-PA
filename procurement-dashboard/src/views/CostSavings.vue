<template>
  <dashboard-layout>
    <template #header>
      Cost Saving Opportunities
    </template>
    
    <!-- KPI Metrics -->
<div class="dashboard-grid mb-6">
  <kpi-card
    title="Total Savings"
    :value="costSavingOpportunities.estimated_savings"
    format="currency"
    description="From identified opportunities"
    color="green"
  />
  
  <kpi-card
    title="Implementation"
    :value="costSavingOpportunities.recommended_action.implementation_difficulty"
    description="Average difficulty level"
    color="yellow"
  />
  
  <kpi-card
    title="Confidence"
    :value="costSavingOpportunities.confidence_level"
    description="Based on market analysis"
    color="primary"
  />
</div>

<!-- Price Variation Chart -->
<div class="card mb-6">
  <h3 class="card-title mb-4">Price Variation Analysis</h3>
  <div class="h-64">
    <bar-chart :chart-data="priceVariationData" :options="boxPlotOptions" />
  </div>
  <div class="mt-4 bg-primary-50 p-4 rounded-lg">
    <h4 class="font-medium text-primary-700">Insight:</h4>
    <p class="text-sm text-gray-600">
      Significant price variation detected across suppliers for the same category. 
      Standardizing to the lowest competitive price could yield ${{ costSavingOpportunities.estimated_savings.toLocaleString() }} in savings.
    </p>
  </div>
</div>

<!-- Detailed Opportunity -->
<div class="card mb-6">
  <div class="flex justify-between items-start mb-4">
    <div>
      <h3 class="card-title">{{ costSavingOpportunities.opportunity_type }}</h3>
      <p class="text-gray-500">{{ costSavingOpportunities.description }}</p>
    </div>
    <div class="bg-green-50 text-green-700 px-3 py-1 rounded-full text-sm font-medium">
      {{ costSavingOpportunities.confidence_level }} confidence
    </div>
  </div>
  
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-4">
    <div>
      <h4 class="font-medium text-gray-700 mb-3">Current State</h4>
      <div class="overflow-x-auto">
        <table class="min-w-full divide-y divide-gray-200">
          <thead class="bg-gray-50">
            <tr>
              <th class="table-header">Supplier</th>
              <th class="table-header">Avg. Price</th>
              <th class="table-header">Difference</th>
            </tr>
          </thead>
          <tbody class="bg-white divide-y divide-gray-200">
            <tr v-for="(supplier, index) in costSavingOpportunities.current_state.suppliers" :key="index">
              <td class="table-cell">{{ supplier.name }}</td>
              <td class="table-cell">${{ supplier.avg_price.toFixed(2) }}</td>
              <td class="table-cell">
                <span :class="getSupplierPriceDiffClass(supplier.avg_price)">
                  {{ getSupplierPriceDiff(supplier.avg_price) }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
    
    <div>
      <h4 class="font-medium text-gray-700 mb-3">Recommended Action</h4>
      <div class="bg-gray-50 p-4 rounded-lg mb-4">
        <div class="flex justify-between items-center mb-2">
          <span class="text-sm text-gray-500">Target Price:</span>
          <span class="font-medium text-green-600">${{ costSavingOpportunities.recommended_action.target_price }}</span>
        </div>
        <div class="flex justify-between items-center mb-2">
          <span class="text-sm text-gray-500">Preferred Supplier:</span>
          <span class="font-medium">{{ costSavingOpportunities.recommended_action.preferred_supplier }}</span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-sm text-gray-500">Implementation:</span>
          <span class="font-medium">{{ costSavingOpportunities.recommended_action.implementation_difficulty }}</span>
        </div>
      </div>
      
      <h4 class="font-medium text-gray-700 mb-2">Next Steps:</h4>
      <ul class="space-y-1">
        <li v-for="(step, index) in costSavingOpportunities.recommended_action.next_steps" :key="index" class="flex items-center">
          <span class="h-2 w-2 rounded-full bg-primary-500 mr-2"></span>
          <span class="text-sm">{{ step }}</span>
        </li>
      </ul>
    </div>
  </div>
  
  <div class="bg-gray-50 p-4 rounded-lg">
    <h4 class="font-medium text-gray-700 mb-2">Supporting Evidence:</h4>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div>
        <p class="text-sm text-gray-500">Price Trend:</p>
        <p class="font-medium">{{ costSavingOpportunities.evidence.price_trend }}</p>
      </div>
      <div>
        <p class="text-sm text-gray-500">Market Conditions:</p>
        <p class="font-medium">{{ costSavingOpportunities.evidence.market_conditions }}</p>
      </div>
      <div>
        <p class="text-sm text-gray-500">Benchmark:</p>
        <p class="font-medium">{{ costSavingOpportunities.evidence.benchmark_data }}</p>
      </div>
    </div>
  </div>
</div>

<!-- Additional Opportunities -->
<div class="card">
  <h3 class="card-title mb-4">More Cost Saving Opportunities</h3>
  <div class="overflow-x-auto">
    <table class="min-w-full divide-y divide-gray-200">
      <thead class="bg-gray-50">
        <tr>
          <th class="table-header">Opportunity</th>
          <th class="table-header">Category</th>
          <th class="table-header">Estimated Savings</th>
          <th class="table-header">Difficulty</th>
          <th class="table-header">Confidence</th>
          <th class="table-header">Actions</th>
        </tr>
      </thead>
      <tbody class="bg-white divide-y divide-gray-200">
        <!-- Additional opportunities from dummy data -->
        <tr v-for="(opportunity, index) in additionalOpportunities" :key="index">
          <td class="table-cell font-medium">{{ opportunity.type }}</td>
          <td class="table-cell">{{ opportunity.category }}</td>
          <td class="table-cell">${{ opportunity.savings.toLocaleString() }}</td>
          <td class="table-cell">{{ opportunity.difficulty }}</td>
          <td class="table-cell">{{ opportunity.confidence }}</td>
          <td class="table-cell">
            <button :class="[
    'px-3 py-1 rounded-full text-xs font-medium',
    index === 0 ? 'bg-primary-100 text-primary-600' : 'bg-gray-100 text-gray-600'
]">
    View Details
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
// import { costSavingOpportunities } from '@/data/procurementData';

// At the top of each component file
import { 
  costSavingsData, 
} from '@/data/dummyData';

export default {
  name: 'CostSavings',
  components: {
    DashboardLayout,
    KpiCard,
    BarChart
  },
  data() {
  return {
    costSavingOpportunities: {
      opportunity_id: "OPP123456",
      opportunity_type: "Price Standardization",
      description: "Standardize pricing for leather straps across suppliers",
      affected_category: {"id": "102993", "name": "Leather straps"},
      estimated_savings: 18500.00,
      confidence_level: "High",
      current_state: {
        suppliers: costSavingsData.priceVariation.map(item => ({
          id: item.supplier.substring(0, 4).toUpperCase(),
          name: item.supplier,
          avg_price: item.currentPrice
        })),
        volume: 245,
        current_spend: 158432.75
      },
      recommended_action: {
        target_price: 118.25,
        preferred_supplier: "CONVEX",
        implementation_difficulty: "Medium",
        next_steps: [
          "Review contracts with current suppliers",
          "Negotiate price match with preferred suppliers",
          "Consolidate volume with best-price supplier"
        ]
      },
      evidence: {
        price_trend: "Stable",
        market_conditions: "Competitive",
        benchmark_data: "Industry average price is $120.50"
      }
    },
    
    // Chart data for Price Variation
    priceVariationData: {
      labels: costSavingsData.priceVariation.map(item => item.supplier),
      datasets: [{
        label: 'Current Price',
        data: costSavingsData.priceVariation.map(item => item.currentPrice),
        backgroundColor: '#4338ca',
      }, {
        label: 'Target Price',
        data: costSavingsData.priceVariation.map(item => item.targetPrice),
        type: 'line',
        borderColor: '#22c55e',
        borderWidth: 2,
        pointBackgroundColor: '#22c55e',
        fill: false
      }, {
        label: 'Industry Average',
        data: costSavingsData.priceVariation.map(item => item.industryAvg),
        type: 'line',
        borderColor: '#f59e0b',
        borderDash: [5, 5],
        borderWidth: 2,
        pointBackgroundColor: '#f59e0b',
        fill: false
      }]
    },
    
    // Box Plot options
    boxPlotOptions: {
      scales: {
        y: {
          beginAtZero: false,
          title: {
            display: true,
            text: 'Price ($)'
          }
        }
      }
    },
    
    // Additional opportunities
    additionalOpportunities: costSavingsData.opportunities
  };
},
  methods: {
    getSupplierPriceDiff(price) {
      const targetPrice = this.costSavingOpportunities.recommended_action.target_price;
      const diff = price - targetPrice;
      if (diff === 0) return 'Baseline';
      return diff > 0 ? `+$${diff.toFixed(2)}` : `-$${Math.abs(diff).toFixed(2)}`;
    },
    getSupplierPriceDiffClass(price) {
      const targetPrice = this.costSavingOpportunities.recommended_action.target_price;
      const diff = price - targetPrice;
      if (diff === 0) return 'text-green-600';
      return diff > 0 ? 'text-red-600' : 'text-green-600';
    }
  }
};
</script>