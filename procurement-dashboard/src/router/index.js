import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from '../views/Dashboard.vue'

const routes = [
  {
    path: '/',
    name: 'Dashboard',
    component: Dashboard
  },
  {
    path: '/spend-analysis',
    name: 'SpendAnalysis',
    component: () => import('../views/SpendAnalysis.vue')
  },
  {
    path: '/supplier-insights',
    name: 'SupplierInsights',
    component: () => import('../views/SupplierInsights.vue')
  },
  {
    path: '/model-performance',
    name: 'ModelPerformance',
    component: () => import('../views/ModelPerformance.vue')
  },
  {
    path: '/cost-savings',
    name: 'CostSavings',
    component: () => import('../views/CostSavings.vue')
  },
  {
    path: '/data-quality',
    name: 'DataQuality',
    component: () => import('../views/DataQuality.vue')
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router