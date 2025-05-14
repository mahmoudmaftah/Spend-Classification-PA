<template>
  <div ref="chartContainer" class="chart-container"></div>
</template>

<script>
import * as d3 from 'd3';
import { onMounted, ref, watch } from 'vue';

export default {
  name: 'HeatmapChart',
  props: {
    data: {
      type: Array,
      required: true
    },
    options: {
      type: Object,
      default: () => ({})
    }
  },
  setup(props) {
    const chartContainer = ref(null);
    
    const createChart = () => {
      if (!chartContainer.value) return;
      
      // Clear previous chart
      d3.select(chartContainer.value).selectAll('*').remove();
      
      const width = chartContainer.value.clientWidth;
      const height = chartContainer.value.clientHeight;
      const margin = { top: 50, right: 50, bottom: 100, left: 100 };
      
      // Get unique suppliers and categories
      const suppliers = [...new Set(props.data.map(d => d.supplier))];
      const categories = [...new Set(props.data.map(d => d.category))];
      
      // Create scales
      const xScale = d3.scaleBand()
        .domain(categories)
        .range([margin.left, width - margin.right])
        .padding(0.05);
        
      const yScale = d3.scaleBand()
        .domain(suppliers)
        .range([margin.top, height - margin.bottom])
        .padding(0.05);
        
      // Color scale for spend values
      const colorScale = d3.scaleSequential()
        .interpolator(d3.interpolateBlues)
        .domain([0, d3.max(props.data, d => d.spend)]);
        
      // Create SVG
      const svg = d3.select(chartContainer.value)
        .append('svg')
        .attr('width', width)
        .attr('height', height);
        
      // Create heatmap cells
      svg.selectAll('rect')
        .data(props.data)
        .enter()
        .append('rect')
        .attr('x', d => xScale(d.category))
        .attr('y', d => yScale(d.supplier))
        .attr('width', xScale.bandwidth())
        .attr('height', yScale.bandwidth())
        .attr('fill', d => colorScale(d.spend))
        .attr('stroke', 'white')
        .attr('stroke-width', 0.5)
        .append('title')
        .text(d => `${d.supplier} - ${d.category}: $${d.spend.toLocaleString()}`);
        
      // Add x-axis
      svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(xScale))
        .selectAll('text')
        .attr('transform', 'rotate(-45)')
        .style('text-anchor', 'end')
        .attr('dx', '-.8em')
        .attr('dy', '.15em');
        
      // Add y-axis
      svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(yScale));
        
      // Add title
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', margin.top / 2)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .style('font-weight', 'bold')
        .text('Supplier-Category Matrix');
    };
    
    onMounted(() => {
      createChart();
      
      // Handle resize
      const handleResize = () => {
        createChart();
      };
      
      window.addEventListener('resize', handleResize);
      
      // Cleanup on unmount
      return () => {
        window.removeEventListener('resize', handleResize);
      };
    });
    
    watch(() => props.data, () => {
      createChart();
    }, { deep: true });
    
    return {
      chartContainer
    };
  }
}
</script>