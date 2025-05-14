<template>
  <div class="flex h-screen bg-gray-50">
    <!-- Sidebar -->
    <div class="w-64 bg-primary-700 text-white">
      <div class="p-4">
        <h1 class="text-xl font-bold flex items-center">
          PAVUS AI Procurement
        </h1>
      </div>
      <nav class="mt-8">
        <router-link 
          to="/" 
          class="flex items-center w-full px-4 py-3 hover:bg-primary-600"
          active-class="bg-primary-800"
          exact
        >
          <span class="mr-3">📊</span>
          <span>Overview</span>
        </router-link>
        
        <router-link 
          to="/spend-analysis" 
          class="flex items-center w-full px-4 py-3 hover:bg-primary-600"
          active-class="bg-primary-800"
        >
          <span class="mr-3">📈</span>
          <span>Spend Analysis</span>
        </router-link>
        
        <router-link 
          to="/supplier-insights" 
          class="flex items-center w-full px-4 py-3 hover:bg-primary-600"
          active-class="bg-primary-800"
        >
          <span class="mr-3">👥</span>
          <span>Supplier Insights</span>
        </router-link>
        
        <router-link 
          to="/model-performance" 
          class="flex items-center w-full px-4 py-3 hover:bg-primary-600"
          active-class="bg-primary-800"
        >
          <span class="mr-3">🧠</span>
          <span>Model Performance</span>
        </router-link>
        
        <router-link 
          to="/cost-savings" 
          class="flex items-center w-full px-4 py-3 hover:bg-primary-600"
          active-class="bg-primary-800"
        >
          <span class="mr-3">💰</span>
          <span>Cost Savings</span>
        </router-link>
        
        <router-link 
          to="/data-quality" 
          class="flex items-center w-full px-4 py-3 hover:bg-primary-600"
          active-class="bg-primary-800"
        >
          <span class="mr-3">✓</span>
          <span>Data Quality</span>
        </router-link>
      </nav>
    </div>
    
    <!-- Main Content -->
    <div class="flex-1 overflow-auto">
      <!-- Header -->
      <header class="bg-white shadow">
        <div class="px-6 py-4 flex justify-between items-center">
          <h2 class="text-xl font-semibold text-gray-800">
            <slot name="header">Dashboard</slot>
          </h2>
          <div class="flex items-center">
            <div class="relative">
              <input 
                type="text" 
                placeholder="Search..." 
                class="px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
            </div>
            <div class="ml-4 relative">
              <button class="flex items-center text-gray-700 focus:outline-none">
                <span class="mr-2">Admin</span>
                <div class="h-8 w-8 rounded-full bg-gray-300 flex items-center justify-center">
                  A
                </div>
              </button>
            </div>
          </div>
        </div>
      </header>
      
      <!-- Dashboard Content -->
      <main class="p-6">
        <slot></slot>
      </main>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DashboardLayout'
}
</script>