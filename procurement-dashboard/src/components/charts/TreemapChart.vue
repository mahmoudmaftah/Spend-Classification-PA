<template>
  <div ref="chartContainer" class="chart-container"></div>
</template>

<script>
import * as d3 from 'd3';
import { onMounted, ref, watch } from 'vue';

export default {
  name: 'TreemapChart',
  props: {
    data: {
      type: Object,
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
      
      // Create SVG
      const svg = d3.select(chartContainer.value)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .append('g')
        .attr('transform', 'translate(0,0)');
      
      // Hierarchy and treemap layout
      const root = d3.hierarchy(props.data)
        .sum(d => d.value)
        .sort((a, b) => b.value - a.value);
      
      // Create treemap layout - FIXED: formatting of function call
      const treemap = d3.treemap()
        .size([width, height])
        .paddingInner(3)
        .paddingTop(20)
        .paddingBottom(2)
        .paddingRight(2)
        .paddingLeft(2)
        .round(true);
        
      // Apply the treemap layout to the hierarchy
      treemap(root);
      
      // Color scale for different depths
      const colorScale = d3.scaleOrdinal()
        .domain([0, 1, 2, 3, 4])
        .range(['#4338ca', '#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe']);
        
      // Create cells for each node
      const cell = svg.selectAll('g')
        .data(root.descendants())
        .enter()
        .append('g')
        .attr('transform', d => `translate(${d.x0},${d.y0})`);
      
      // Add rectangles
      cell.append('rect')
        .attr('width', d => d.x1 - d.x0)
        .attr('height', d => d.y1 - d.y0)
        .attr('fill', d => colorScale(d.depth))
        .attr('stroke', '#fff')
        .attr('stroke-width', 1);
      
      // Add text labels
      cell.append('text')
        .attr('x', 5)
        .attr('y', 15)
        .attr('fill', 'white')
        .attr('font-size', '10px')
        .attr('font-weight', 'bold')
        .text(d => d.data.name)
        .attr('clip-path', d => `inset(0px 0px 0px ${d.x1 - d.x0}px)`);
        
      // Add value labels (only for cells large enough)
      cell
        .filter(d => (d.x1 - d.x0) > 40 && (d.y1 - d.y0) > 40)
        .append('text')
        .attr('x', 5)
        .attr('y', 35)
        .attr('fill', 'rgba(255,255,255,0.7)')
        .attr('font-size', '9px')
        .text(d => d.value ? `$${d.value.toLocaleString()}` : '');
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