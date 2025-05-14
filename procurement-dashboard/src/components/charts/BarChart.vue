<template>
  <div class="chart-container">
    <canvas ref="barChart"></canvas>
  </div>
</template>

<script>
import { Chart, registerables } from 'chart.js';
import { onMounted, ref, watch } from 'vue';

Chart.register(...registerables);

export default {
  name: 'BarChart',
  props: {
    chartData: {
      type: Object,
      required: true
    },
    options: {
      type: Object,
      default: () => ({})
    }
  },
  setup(props) {
    const barChart = ref(null);
    let chart = null;

    const createChart = () => {
      if (chart) {
        chart.destroy();
      }

      const ctx = barChart.value.getContext('2d');
      
      const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
          },
          title: {
            display: false,
          },
          tooltip: {
            mode: 'index',
            intersect: false,
          },
        },
        scales: {
          y: {
            beginAtZero: true
          }
        }
      };

      chart = new Chart(ctx, {
        type: 'bar',
        data: props.chartData,
        options: { ...defaultOptions, ...props.options }
      });
    };

    onMounted(() => {
      createChart();
    });

    watch(() => props.chartData, () => {
      createChart();
    }, { deep: true });

    return {
      barChart
    };
  }
}
</script>