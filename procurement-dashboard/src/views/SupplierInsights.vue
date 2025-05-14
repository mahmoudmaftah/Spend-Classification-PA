<template>
  <dashboard-layout>
    <template #header>
      Supplier Insights Dashboard
    </template>

    <!-- KPI Metrics -->
    <div class="dashboard-grid mb-6">
      <kpi-card
        title="Total Suppliers"
        :value="categoryAnalytics.supplier_count"
        description="For current category"
      />

      <kpi-card
        title="Supplier Concentration"
        :value="categoryAnalytics.supplier_concentration"
        format="percent"
        description="Top 3 suppliers by spend"
        color="yellow"
      />

      <kpi-card
        title="Price Volatility"
        :value="categoryAnalytics.price_volatility * 100"
        format="percent"
        description="Coefficient of variation"
        color="secondary"
      />
    </div>

    <!-- Supplier Risk Map -->
    <div class="card mb-6">
      <h3 class="card-title mb-4">Supplier Risk Map</h3>
      <div class="chart-container">
        <div class="h-64 bg-primary-50 p-4 flex items-center justify-center rounded-lg">
          <!-- This would be a scatter plot, using D3.js or ECharts -->
          <div class="text-center">
            <p class="text-lg text-primary-700 font-medium">Risk/Spend Matrix</p>
            <p class="text-gray-500">Positioning suppliers by risk level and spend volume</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Supplier Analysis -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
      <!-- Supplier Category Distribution -->
      <div class="card">
        <h3 class="card-title mb-4">Supplier Category Specialization</h3>
        <bar-chart :chart-data="categorySpecializationData" />
      </div>

      <!-- Supplier Sustainability -->
      <div class="card">
        <h3 class="card-title mb-4">Supplier Sustainability Ratings</h3>
        <bar-chart :chart-data="sustainabilityData" />
      </div>
    </div>

    <!-- Selected Supplier Details -->
    <div class="card mb-6">
      <div class="flex justify-between items-center mb-4">
        <h3 class="card-title">Supplier: {{ supplierAnalytics.supplier_name }}</h3>
        <select class="px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500">
          <option
            v-for="(supplier, index) in categoryAnalytics.top_suppliers"
            :key="index"
          >
            {{ supplier.name }}
          </option>
        </select>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
        <div class="bg-gray-50 p-4 rounded-lg">
          <p class="text-sm text-gray-500">Industry</p>
          <p class="font-medium">{{ supplierAnalytics.industry }}</p>
        </div>
        <div class="bg-gray-50 p-4 rounded-lg">
          <p class="text-sm text-gray-500">Location</p>
          <p class="font-medium">{{ supplierAnalytics.location }}</p>
        </div>
        <div class="bg-gray-50 p-4 rounded-lg">
          <p class="text-sm text-gray-500">Risk Level</p>
          <p class="font-medium">{{ supplierAnalytics.risk_level }}</p>
        </div>
        <div class="bg-gray-50 p-4 rounded-lg">
          <p class="text-sm text-gray-500">Financial Health</p>
          <p class="font-medium">{{ supplierAnalytics.financial_health }}/10</p>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h4 class="font-medium text-gray-700 mb-2">Category Distribution</h4>
          <doughnut-chart :chart-data="supplierCategoryData" />
        </div>
        <div>
          <h4 class="font-medium text-gray-700 mb-2">Risk Factors</h4>
          <ul class="space-y-2">
            <li
              v-for="(factor, index) in supplierAnalytics.risk_factors"
              :key="index"
              class="flex items-center"
            >
              <span class="h-2 w-2 rounded-full bg-yellow-500 mr-2"></span>
              <span>{{ factor }}</span>
            </li>
          </ul>

          <h4 class="font-medium text-gray-700 mt-4 mb-2">Contract Information</h4>
          <p class="text-sm"><span class="text-gray-500">Status:</span> {{ supplierAnalytics.contract_status }}</p>
          <p class="text-sm"><span class="text-gray-500">Expiration:</span> {{ supplierAnalytics.contract_expiration }}</p>
          <p class="text-sm"><span class="text-gray-500">Payment Terms:</span> {{ supplierAnalytics.payment_terms }}</p>
        </div>
      </div>
    </div>

    <!-- Top Suppliers Table -->
    <div class="card">
      <h3 class="card-title mb-4">All Suppliers Performance</h3>
      <div class="overflow-x-auto">
        <table class="min-w-full divide-y divide-gray-200">
          <thead class="bg-gray-50">
            <tr>
              <th class="table-header">Supplier</th>
              <th class="table-header">Total Spend</th>
              <th class="table-header">Categories</th>
              <th class="table-header">Risk Level</th>
              <th class="table-header">Sustainability</th>
              <th class="table-header">Financial Health</th>
            </tr>
          </thead>
          <tbody class="bg-white divide-y divide-gray-200">
            <tr
              v-for="(supplier, index) in supplierInsightsData.supplierPerformance.slice(0, 3)"
              :key="index"
            >
              <td class="table-cell font-medium">
                {{ supplierInsightsData.supplierRisk[index].supplier }}
              </td>
              <td class="table-cell">
                ${{ supplierInsightsData.supplierRisk[index].spend.toLocaleString() }}
              </td>
              <td class="table-cell">
                {{ supplierInsightsData.categorySpecialization[index].categories.length }}
              </td>
              <td class="table-cell">
                {{ supplierInsightsData.supplierRisk[index].risk > 0.5 ? 'Medium' : 'Low' }}
              </td>
              <td class="table-cell">{{ supplier.sustainability }}/10</td>
              <td class="table-cell">{{ supplier.financialHealth }}/10</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </dashboard-layout>
</template>

<script>
import DashboardLayout     from '@/layouts/DashboardLayout.vue';
import KpiCard             from '@/components/ui/KpiCard.vue';
import BarChart            from '@/components/charts/BarChart.vue';
import DoughnutChart       from '@/components/charts/DoughnutChart.vue';

import {
  spendAnalysisData,
  supplierInsightsData   // <- imported dataset
} from '@/data/dummyData';

export default {
  name: 'SupplierInsights',
  components: {
    DashboardLayout,
    KpiCard,
    BarChart,
    DoughnutChart
  },
  data() {
    return {
      /* --------  FIX: make the dataset reactive & visible to the template  -------- */
      supplierInsightsData,

      /* --------------------------------------------------------------------------- */
      categoryAnalytics: {
        supplier_count: 12,
        supplier_concentration: 72.01,
        price_volatility: 0.15,
        top_suppliers: spendAnalysisData.topSuppliers
      },

      supplierAnalytics: {
        supplier_id: 'AKRB',
        supplier_name: 'Akron Belting',
        standard_name: 'AKRON BELTING',
        industry: 'Industrial Manufacturing',
        market_segment: 'Conveyor Systems and Components',
        company_size: 'Small',
        location: 'Akron, OH, USA',
        relationship_status: 'Preferred',
        total_spend: 452718.25,
        spend_year_to_date: 125431.50,
        spend_previous_year: 415980.25,
        spend_change_percent: 8.83,
        transaction_count: 587,
        average_transaction_value: 771.24,
        category_distribution: [
          { category_id: '102993', name: 'Leather straps',  spend:  58432.50, percent: 12.91 },
          { category_id: '102994', name: 'Nylon straps',    spend:  45781.25, percent: 10.11 },
          { category_id: '103245', name: 'Conveyor belts', spend: 215432.75, percent: 47.59 }
        ],
        category_concentration: 70.61,
        risk_level: 'Medium',
        risk_factors: ['Single source for key items', 'Regional supplier'],
        sustainability_rating: 4,
        digital_presence_score: 3,
        financial_health: 6,
        innovation_score: 5,
        contract_status: 'Active',
        contract_expiration: '2025-08-15',
        payment_terms: 'Net 45',
        discounts_available: ['2% 10 Net 30', 'Volume discount > $50,000']
      },

      /* -------- Chart Data -------- */
      categorySpecializationData: {
        labels: spendAnalysisData.topSuppliers.map(s => s.name),
        datasets: [
          { label: 'Leather straps',  data: [58432, 32451, 24789], backgroundColor: '#4338ca' },
          { label: 'Nylon straps',    data: [45781, 28750, 12340], backgroundColor: '#6366f1' },
          { label: 'Conveyor belts',  data: [215432,145670, 98540], backgroundColor: '#818cf8' }
        ]
      },

      sustainabilityData: {
        labels: spendAnalysisData.topSuppliers.map(s => s.name),
        datasets: [
          {
            label: 'Sustainability Rating',
            data: supplierInsightsData.supplierPerformance.map(s => s.sustainability).slice(0, 3),
            backgroundColor: '#065f46'
          },
          {
            label: 'Financial Health',
            data: supplierInsightsData.supplierPerformance.map(s => s.financialHealth).slice(0, 3),
            backgroundColor: '#0ea5e9'
          }
        ]
      },

      supplierCategoryData: {
        labels: ['Leather straps', 'Nylon straps', 'Conveyor belts'],
        datasets: [
          {
            data: [58432.50, 45781.25, 215432.75],
            backgroundColor: ['#4338ca', '#6366f1', '#818cf8'],
            borderWidth: 0
          }
        ]
      }
    };
  }
};
</script>
