<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  imageSize: {
    type: Array,
    required: true,
  },
  markers: {
    type: Array,
    default: () => [],
  },
  colors: {
    type: Array,
    default: () => [],
  },
  selectedKey: {
    type: [String, Number],
    default: null,
  },
  adding: {
    type: Boolean,
    default: false,
  },
  disabled: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits(['canvas-click', 'select', 'move'])
const svgEl = ref(null)
const drag = ref(null)

function pointFromEvent(event) {
  const svg = svgEl.value
  const width = Number(props.imageSize?.[0])
  const height = Number(props.imageSize?.[1])
  if (!svg || !width || !height) return null

  const rect = svg.getBoundingClientRect()
  const scale = Math.min(rect.width / width, rect.height / height)
  const renderedWidth = width * scale
  const renderedHeight = height * scale
  const x = event.clientX - rect.left - (rect.width - renderedWidth) / 2
  const y = event.clientY - rect.top - (rect.height - renderedHeight) / 2
  if (x < 0 || y < 0 || x > renderedWidth || y > renderedHeight) return null

  return [
    Math.max(0, Math.min(width - 1, x / scale)),
    Math.max(0, Math.min(height - 1, y / scale)),
  ]
}

function markerColor(marker) {
  const option = props.colors.find((color) =>
    String(color.id) === String(marker.id) || String(color.id) === String(marker.color),
  )
  return option?.display_color || marker.display_color || marker.color || '#4f8cff'
}

function markerLabel(marker) {
  const option = props.colors.find((color) =>
    String(color.id) === String(marker.id) || String(color.id) === String(marker.color),
  )
  return option?.label || marker.color || marker.id
}

function markerFlags(marker) {
  if (Array.isArray(marker.flags)) return marker.flags
  if (marker.flags && typeof marker.flags === 'object') {
    return Object.entries(marker.flags).filter(([, active]) => active).map(([name]) => name)
  }
  return marker.flags ? [String(marker.flags)] : []
}

function isUncertain(marker) {
  const noisyFlags = markerFlags(marker).filter(
    (flag) => flag !== 'already_saved' && flag !== 'manual_added',
  )
  return Number(marker.confidence ?? 1) < 0.65
    || Number(marker.color_confidence ?? 1) < 0.65
    || noisyFlags.length > 0
}

function isSaved(marker) {
  return marker.source === 'saved' || markerFlags(marker).includes('already_saved')
}

const labelLayouts = computed(() => {
  const width = Number(props.imageSize?.[0]) || 1
  const height = Number(props.imageSize?.[1]) || 1
  const labelWidth = 86
  const labelHeight = 23
  const labelGap = 8
  const margin = 10
  const markers = props.markers.map((marker) => ({
    marker,
    x: Number(marker.center?.[0]) || 0,
    y: Number(marker.center?.[1]) || 0,
  }))
  if (!markers.length) return {}

  const averageX = markers.reduce((sum, item) => sum + item.x, 0) / markers.length
  const placeRight = averageX < width / 2
  const outerX = placeRight
    ? Math.max(...markers.map((item) => item.x))
    : Math.min(...markers.map((item) => item.x))
  const labelX = placeRight
    ? Math.min(width - labelWidth - margin, outerX + 65)
    : Math.max(margin, outerX - labelWidth - 65)
  const sorted = [...markers].sort((a, b) => a.y - b.y)

  let previousBottom = margin - labelGap
  const positioned = sorted.map((item) => {
    const y = Math.max(margin, item.y - labelHeight / 2, previousBottom + labelGap)
    previousBottom = y + labelHeight
    return { ...item, labelX, labelY: y }
  })
  const overflow = Math.max(0, previousBottom - (height - margin))

  return Object.fromEntries(positioned.map((item) => {
    const labelY = Math.max(margin, item.labelY - overflow)
    const lineX = placeRight ? labelX : labelX + labelWidth
    const lineY = labelY + labelHeight / 2
    const dx = item.x - lineX
    const dy = item.y - lineY
    const distance = Math.max(1, Math.hypot(dx, dy))
    const radius = Math.max(Number(item.marker.radius_px) || 12, 8) + 6
    return [item.marker._key, {
      x: labelX,
      y: labelY,
      lineX,
      lineY,
      markerX: item.x - (dx / distance) * radius,
      markerY: item.y - (dy / distance) * radius,
    }]
  }))
})

function labelLayout(marker) {
  return labelLayouts.value[marker._key] || {
    x: marker.center[0] + 65,
    y: marker.center[1] - 12,
    lineX: marker.center[0] + 65,
    lineY: marker.center[1],
    markerX: marker.center[0],
    markerY: marker.center[1],
  }
}

function onCanvasPointerDown(event) {
  if (props.disabled) return
  const point = pointFromEvent(event)
  if (point) emit('canvas-click', point)
}

function onMarkerPointerDown(event, marker) {
  if (props.disabled) return
  event.preventDefault()
  event.stopPropagation()
  emit('select', marker._key)
  if (isSaved(marker)) return
  drag.value = { key: marker._key, pointerId: event.pointerId }
  svgEl.value?.setPointerCapture?.(event.pointerId)
}

function onPointerMove(event) {
  if (!drag.value || drag.value.pointerId !== event.pointerId) return
  const point = pointFromEvent(event)
  if (point) emit('move', { key: drag.value.key, center: point })
}

function endDrag(event) {
  if (!drag.value || drag.value.pointerId !== event.pointerId) return
  svgEl.value?.releasePointerCapture?.(event.pointerId)
  drag.value = null
}
</script>

<template>
  <svg
    ref="svgEl"
    class="marker-overlay"
    :class="{ adding, disabled }"
    :viewBox="`0 0 ${imageSize[0]} ${imageSize[1]}`"
    preserveAspectRatio="xMidYMid meet"
    @pointermove="onPointerMove"
    @pointerup="endDrag"
    @pointercancel="endDrag"
  >
    <defs>
      <marker
        id="leader-arrow"
        viewBox="0 0 6 6"
        refX="5"
        refY="3"
        markerWidth="5"
        markerHeight="5"
        orient="auto"
      >
        <path d="M 0 0 L 6 3 L 0 6 z" />
      </marker>
    </defs>
    <rect
      class="overlay-hit-area"
      x="0"
      y="0"
      :width="imageSize[0]"
      :height="imageSize[1]"
      @pointerdown="onCanvasPointerDown"
    />
    <g
      v-for="marker in markers"
      :key="marker._key"
      class="marker-node"
      :class="{
        selected: marker._key === selectedKey,
        uncertain: isUncertain(marker),
        saved: isSaved(marker),
      }"
      @pointerdown="onMarkerPointerDown($event, marker)"
    >
      <title>
        {{ markerLabel(marker) }}
        · 置信度 {{ Math.round(Number(marker.confidence ?? 0) * 100) }}%
        {{ markerFlags(marker).length ? `· ${markerFlags(marker).join(', ')}` : '' }}
      </title>
      <line
        class="marker-leader"
        :x1="labelLayout(marker).lineX"
        :y1="labelLayout(marker).lineY"
        :x2="labelLayout(marker).markerX"
        :y2="labelLayout(marker).markerY"
        marker-end="url(#leader-arrow)"
      />
      <circle
        class="marker-halo"
        :cx="marker.center[0]"
        :cy="marker.center[1]"
        :r="Math.max(Number(marker.radius_px) || 12, 8) + 3"
      />
      <circle
        class="marker-circle"
        :cx="marker.center[0]"
        :cy="marker.center[1]"
        :r="Math.max(Number(marker.radius_px) || 12, 8)"
        :style="{ stroke: markerColor(marker) }"
      />
      <line
        class="marker-center"
        :x1="marker.center[0] - 5"
        :x2="marker.center[0] + 5"
        :y1="marker.center[1]"
        :y2="marker.center[1]"
      />
      <line
        class="marker-center"
        :x1="marker.center[0]"
        :x2="marker.center[0]"
        :y1="marker.center[1] - 5"
        :y2="marker.center[1] + 5"
      />
      <g :transform="`translate(${labelLayout(marker).x} ${labelLayout(marker).y})`">
        <rect class="marker-label-bg" x="0" y="0" width="86" height="23" rx="5" />
        <circle cx="11" cy="11.5" r="5" :fill="markerColor(marker)" />
        <text class="marker-label" x="20" y="15">{{ markerLabel(marker) }}{{ isSaved(marker) ? '✓' : '' }}</text>
        <text v-if="isUncertain(marker)" class="marker-warning" x="73" y="15">!</text>
      </g>
    </g>
  </svg>
</template>

<style scoped>
.marker-overlay {
  position: absolute;
  inset: 0;
  z-index: 2;
  width: 100%;
  height: 100%;
  cursor: default;
  touch-action: none;
  user-select: none;
}

.marker-overlay.adding {
  cursor: crosshair;
}

.marker-overlay.disabled {
  cursor: wait;
}

.overlay-hit-area {
  fill: transparent;
  pointer-events: all;
}

.marker-node {
  cursor: grab;
}

.marker-node:active {
  cursor: grabbing;
}

.marker-halo {
  fill: rgba(0, 0, 0, .12);
  stroke: rgba(255, 255, 255, .9);
  stroke-width: 2;
}

.marker-circle {
  fill: rgba(0, 0, 0, .08);
  stroke-width: 4;
}

.marker-node.selected .marker-halo {
  stroke: #4f8cff;
  stroke-width: 5;
}

.marker-node.saved .marker-halo {
  stroke: #3ecf8e;
}

.marker-node.uncertain .marker-halo {
  stroke: #f7b955;
  stroke-dasharray: 7 5;
}

.marker-node.uncertain.selected .marker-halo {
  stroke: #ff6b6b;
}

.marker-node.saved {
  cursor: default;
}

.marker-center {
  stroke: white;
  stroke-width: 2;
  pointer-events: none;
}

.marker-leader {
  stroke: rgba(255, 255, 255, .82);
  stroke-width: 1;
  fill: none;
  pointer-events: none;
}

#leader-arrow path {
  fill: rgba(255, 255, 255, .82);
}

.marker-label-bg {
  fill: rgba(15, 17, 23, .9);
  stroke: rgba(255, 255, 255, .45);
  stroke-width: 1;
}

.marker-label {
  fill: white;
  font-size: 12px;
  font-weight: 600;
  pointer-events: none;
}

.marker-warning {
  fill: #f7b955;
  font-size: 14px;
  font-weight: 800;
  pointer-events: none;
}
</style>
