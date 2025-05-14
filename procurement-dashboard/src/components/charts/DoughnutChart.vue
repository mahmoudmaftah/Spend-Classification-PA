<template>
  <div class="chart-container">
    <canvas ref="doughnutChart"></canvas>
  </div>
</template>

<script>
import { Chart, registerables } from 'chart.js';
import { onMounted, ref, watch } from 'vue';

Chart.register(...registerables);

export default {
  name: 'DoughnutChart',
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
    const doughnutChart = ref(null);
    let chart = null;

    const createChart = () => {
      if (chart) {
        chart.destroy();
      }

      const ctx = doughnutChart.value.getContext('2d');
      
      const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                let label = context.label || '';
                if (label) {
                  label += ': ';
                }
                if (context.parsed !== null) {
                  label += context.parsed;
                }
                return label;
              }
            }
          }
        },
        cutout: '70%'
      };

      chart = new Chart(ctx, {
        type: 'doughnut',
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
      doughnutChart
    };
  }
}
</script>