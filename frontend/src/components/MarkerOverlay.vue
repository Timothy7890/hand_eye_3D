<script setup>
import { ref } from 'vue'

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
      <g :transform="`translate(${marker.center[0] + 10} ${marker.center[1] - 25})`">
        <rect class="marker-label-bg" x="0" y="0" width="78" height="21" rx="5" />
        <circle cx="10" cy="10.5" r="5" :fill="markerColor(marker)" />
        <text class="marker-label" x="19" y="14">{{ markerLabel(marker) }}{{ isSaved(marker) ? '✓' : '' }}</text>
        <text v-if="isUncertain(marker)" class="marker-warning" x="66" y="14">!</text>
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
