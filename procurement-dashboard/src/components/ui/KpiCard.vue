<template>
  <div class="card">
    <h3 class="card-title">{{ title }}</h3>
    <p class="kpi-value" :class="valueColor">{{ formattedValue }}</p>
    <p class="kpi-description">{{ description }}</p>
    <div v-if="showTrend" class="mt-4 flex items-center">
      <span :class="trendColor" class="inline-flex items-center">
        <span v-if="trend > 0">↑</span>
        <span v-else-if="trend < 0">↓</span>
        <span class="ml-1">{{ Math.abs(trend) }}%</span>
      </span>
      <span class="ml-2 text-gray-500 text-sm">{{ trendDescription }}</span>
    </div>
  </div>
</template>

<script>
export default {
  name: 'KpiCard',
  props: {
    title: {
      type: String,
      required: true
    },
    value: {
      type: [Number, String],
      required: true
    },
    description: {
      type: String,
      default: ''
    },
    trend: {
      type: Number,
      default: null
    },
    trendDescription: {
      type: String,
      default: 'vs previous period'
    },
    format: {
      type: String,
      default: 'number' // 'number', 'currency', 'percent'
    },
    color: {
      type: String,
      default: 'primary' // primary, secondary, green, yellow, red
    }
  },
  computed: {
    formattedValue() {
      if (this.format === 'currency') {
        return typeof this.value === 'number' 
          ? `$${this.value.toLocaleString()}` 
          : this.value;
      } else if (this.format === 'percent') {
        return typeof this.value === 'number' 
          ? `${this.value}%` 
          : this.value;
      } else {
        return typeof this.value === 'number' 
          ? this.value.toLocaleString() 
          : this.value;
      }
    },
    valueColor() {
      const colorMap = {
        primary: 'text-primary-600',
        secondary: 'text-secondary-600',
        green: 'text-green-600',
        yellow: 'text-yellow-600',
        red: 'text-red-600'
      };
      return colorMap[this.color] || colorMap.primary;
    },
    showTrend() {
      return this.trend !== null;
    },
    trendColor() {
      if (this.trend > 0) {
        return 'text-green-500';
      } else if (this.trend < 0) {
        return 'text-red-500';
      } else {
        return 'text-gray-500';
      }
    }
  }
}
</script>