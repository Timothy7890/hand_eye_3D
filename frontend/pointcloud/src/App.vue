<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'

const viewerHost = ref(null)
const status = ref(null)
const episodes = ref([])
const samples = ref([])
const markerColors = ref([])
const selectedEpisode = ref('')
const activeColor = ref('')
const selections = ref([])
const cloudBusy = ref(false)
const saveBusy = ref(false)
const solveBusy = ref(false)
const errorMsg = ref('')
const infoMsg = ref('')
const cloudId = ref('')
const cloudStride = ref(2)
const pointCount = ref(0)
const pointSize = ref(4)
const solveResult = ref(null)
const imageFrontendUrl = `${window.location.protocol}//${window.location.hostname}:7012`

let scene
let camera
let renderer
let controls
let cloudObject
let cloudMaterial
let markerGroup
let resizeObserver
let requestSerial = 0
let pointerStart = null

const currentEpisode = computed(() =>
  episodes.value.find((item) => item.name === selectedEpisode.value) || null,
)
const episodeSamples = computed(() =>
  samples.value.filter((sample) =>
    (sample.episode || sample.pose_id || sample.provenance?.episode) === selectedEpisode.value,
  ),
)
const savedColors = computed(() => new Set(episodeSamples.value.map((sample) => sample.color)))
const selectedColors = computed(() => new Set(selections.value.map((item) => item.color)))
const selectableColors = computed(() => markerColors.value)
const canSave = computed(() =>
  (selections.value.length > 0 || episodeSamples.value.length > 0)
  && !cloudBusy.value
  && !saveBusy.value
  && Boolean(selectedEpisode.value),
)
const residualSummary = computed(() => {
  const residual = solveResult.value?.residual_mm
  if (!residual) return null
  return {
    rms: Number(residual.rms).toFixed(2),
    median: Number(residual.median).toFixed(2),
    max: Number(residual.max).toFixed(2),
  }
})

function colorInfo(color) {
  return markerColors.value.find((item) => item.color === color) || {
    color,
    label_zh: color,
    display_color: '#f8fafc',
  }
}

function setError(error) {
  errorMsg.value = error instanceof Error ? error.message : String(error)
}

async function responseError(response, fallback) {
  try {
    const data = await response.json()
    return new Error(data.error || fallback)
  } catch {
    return new Error(`${fallback}（HTTP ${response.status}）`)
  }
}

function initViewer() {
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x080c14)

  camera = new THREE.PerspectiveCamera(48, 1, 0.005, 50)
  // 相机坐标系为 X 右、Y 下、Z 前。Three.js 相机默认看向 -Z，
  // 当观察方向改为 +Z 时需同时把 up 设为 -Y，才能保持画面不镜像。
  camera.up.set(0, -1, 0)
  camera.position.set(0, 0, -2)

  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.outputColorSpace = THREE.SRGBColorSpace
  viewerHost.value.appendChild(renderer.domElement)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.08
  controls.screenSpacePanning = true

  const grid = new THREE.GridHelper(2, 20, 0x334155, 0x1e293b)
  scene.add(grid)
  markerGroup = new THREE.Group()
  scene.add(markerGroup)

  renderer.domElement.addEventListener('pointerdown', onPointerDown)
  renderer.domElement.addEventListener('pointerup', onPointerUp)
  resizeObserver = new ResizeObserver(resizeViewer)
  resizeObserver.observe(viewerHost.value)
  renderer.setAnimationLoop(() => {
    controls.update()
    renderer.render(scene, camera)
  })
  resizeViewer()
}

function resizeViewer() {
  if (!renderer || !viewerHost.value) return
  const width = Math.max(1, viewerHost.value.clientWidth)
  const height = Math.max(1, viewerHost.value.clientHeight)
  renderer.setSize(width, height, false)
  camera.aspect = width / height
  camera.updateProjectionMatrix()
}

function disposeCloud() {
  if (!cloudObject) return
  scene.remove(cloudObject)
  cloudObject.geometry.dispose()
  cloudMaterial.dispose()
  cloudObject = null
  cloudMaterial = null
}

function frameCloud(geometry) {
  geometry.computeBoundingSphere()
  const sphere = geometry.boundingSphere
  if (!sphere) return
  const center = sphere.center.clone()
  const radius = Math.max(sphere.radius, 0.12)
  controls.target.copy(center)
  camera.position.set(center.x, center.y, center.z - radius * 2.8)
  camera.near = Math.max(0.001, radius / 200)
  camera.far = Math.max(10, radius * 30)
  camera.updateProjectionMatrix()
  controls.update()
}

function restoreSavedSelections(geometry) {
  const saved = episodeSamples.value.filter((sample) =>
    Array.isArray(sample.p_camera) && sample.p_camera.length === 3,
  )
  const position = geometry?.getAttribute('position')
  if (!position || !saved.length) {
    selections.value = []
    refreshHighlights()
    return 0
  }

  const restored = saved.map((sample) => {
    const target = sample.p_camera.map(Number)
    const savedCloudId = sample.cloud_id || sample.provenance?.cloud_id
    const savedStride = Number(sample.point_cloud_stride || sample.provenance?.point_cloud_stride)
    const savedIndex = Number(sample.vertex_index ?? sample.provenance?.vertex_index)
    let vertexIndex = -1

    if (
      savedCloudId === cloudId.value
      && savedStride === cloudStride.value
      && Number.isInteger(savedIndex)
      && savedIndex >= 0
      && savedIndex < position.count
    ) {
      vertexIndex = savedIndex
    } else {
      let nearestDistanceSq = Infinity
      for (let index = 0; index < position.count; index += 1) {
        const dx = position.getX(index) - target[0]
        const dy = position.getY(index) - target[1]
        const dz = position.getZ(index) - target[2]
        const distanceSq = dx * dx + dy * dy + dz * dz
        if (distanceSq < nearestDistanceSq) {
          nearestDistanceSq = distanceSq
          vertexIndex = index
        }
      }
    }

    const point = vertexIndex >= 0
      ? [position.getX(vertexIndex), position.getY(vertexIndex), position.getZ(vertexIndex)]
      : target
    return {
      color: sample.color,
      vertexIndex,
      point,
      displayPoint: [...point],
      sampleIndex: sample.index,
      restored: true,
    }
  }).filter((item) => item.vertexIndex >= 0)

  selections.value = restored
  refreshHighlights()
  return restored.length
}

async function loadPointCloud() {
  const episode = selectedEpisode.value
  if (!episode || !renderer) return
  const serial = ++requestSerial
  cloudBusy.value = true
  errorMsg.value = ''
  infoMsg.value = ''
  solveResult.value = null
  selections.value = []
  cloudId.value = ''
  pointCount.value = 0
  refreshHighlights()
  disposeCloud()
  try {
    const response = await fetch(
      `/api/offline/episodes/${encodeURIComponent(episode)}/point-cloud.ply`
      + `?stride=${cloudStride.value}&v=${Date.now()}`,
    )
    if (!response.ok) throw await responseError(response, '点云加载失败')
    const id = response.headers.get('X-Point-Cloud-Id')
    const count = Number(response.headers.get('X-Point-Count'))
    const stride = Number(response.headers.get('X-Point-Cloud-Stride'))
    const buffer = await response.arrayBuffer()
    const geometry = new PLYLoader().parse(buffer)
    if (serial !== requestSerial || episode !== selectedEpisode.value) {
      geometry.dispose()
      return
    }
    if (!geometry.getAttribute('position') || geometry.getAttribute('position').count === 0) {
      geometry.dispose()
      throw new Error('点云中没有可选顶点')
    }
    cloudMaterial = new THREE.PointsMaterial({
      size: pointSize.value / 1000,
      vertexColors: Boolean(geometry.getAttribute('color')),
      sizeAttenuation: true,
    })
    cloudObject = new THREE.Points(geometry, cloudMaterial)
    scene.add(cloudObject)
    cloudId.value = id || ''
    pointCount.value = Number.isFinite(count) ? count : geometry.getAttribute('position').count
    cloudStride.value = Number.isFinite(stride) ? stride : cloudStride.value
    frameCloud(geometry)
    const restoredCount = restoreSavedSelections(geometry)
    infoMsg.value = restoredCount
      ? `已恢复 ${restoredCount} 个已保存选点，可选择颜色后重新选点`
      : `已加载 ${pointCount.value.toLocaleString()} 个稳定点`
  } catch (error) {
    if (serial === requestSerial) setError(error)
  } finally {
    if (serial === requestSerial) cloudBusy.value = false
  }
}

function onPointerDown(event) {
  pointerStart = { x: event.clientX, y: event.clientY }
}

function onPointerUp(event) {
  if (!pointerStart || !cloudObject || !activeColor.value) return
  const moved = Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y)
  pointerStart = null
  if (moved > 5) return

  const rect = renderer.domElement.getBoundingClientRect()
  const mouse = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  )
  const raycaster = new THREE.Raycaster()
  raycaster.params.Points.threshold = Math.max(0.003, pointSize.value / 700)
  raycaster.setFromCamera(mouse, camera)
  cloudObject.updateMatrixWorld(true)
  const position = cloudObject.geometry.getAttribute('position')
  const candidate = raycaster
    .intersectObject(cloudObject, false)
    .filter((hit) => hit.index != null)
    .map((hit) => {
      const projected = new THREE.Vector3(
        position.getX(hit.index),
        position.getY(hit.index),
        position.getZ(hit.index),
      )
        .applyMatrix4(cloudObject.matrixWorld)
        .project(camera)
      const dx = (projected.x - mouse.x) * rect.width / 2
      const dy = (projected.y - mouse.y) * rect.height / 2
      return { hit, screenDistancePx: Math.hypot(dx, dy) }
    })
    .sort((a, b) => a.screenDistancePx - b.screenDistancePx)[0]
  if (!candidate) {
    infoMsg.value = '没有命中点，请放大后重试'
    return
  }

  const { hit, screenDistancePx } = candidate
  const point = [position.getX(hit.index), position.getY(hit.index), position.getZ(hit.index)]
  const displayPoint = new THREE.Vector3(...point)
    .applyMatrix4(cloudObject.matrixWorld)
    .toArray()
  const color = activeColor.value
  const next = selections.value.filter((item) => item.color !== color)
  const previous = selections.value.find((item) => item.color === color)
  next.push({
    color,
    vertexIndex: hit.index,
    point,
    displayPoint,
    sampleIndex: previous?.sampleIndex,
    restored: false,
  })
  selections.value = next
  infoMsg.value = `${colorInfo(color).label_zh}选点已更新（距点击 ${screenDistancePx.toFixed(1)} px）`
  refreshHighlights()
  chooseNextColor()
}

function chooseNextColor() {
  const next = selectableColors.value.find((item) => !selectedColors.value.has(item.color))
  if (next) activeColor.value = next.color
}

function refreshHighlights() {
  if (!markerGroup) return
  while (markerGroup.children.length) {
    const child = markerGroup.children[0]
    markerGroup.remove(child)
    child.geometry?.dispose()
    child.material?.dispose()
  }
  for (const selection of selections.value) {
    const marker = colorInfo(selection.color)
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.0035, 16, 12),
      new THREE.MeshBasicMaterial({ color: marker.display_color }),
    )
    const displayPoint = selection.displayPoint || [
      selection.point[0],
      selection.point[1],
      selection.point[2],
    ]
    mesh.position.fromArray(displayPoint)
    markerGroup.add(mesh)
  }
}

function removeSelection(color) {
  selections.value = selections.value.filter((item) => item.color !== color)
  activeColor.value = color
  refreshHighlights()
}

function clearSelections() {
  selections.value = []
  chooseNextColor()
  refreshHighlights()
}

async function loadWorkspace() {
  errorMsg.value = ''
  try {
    const [statusResponse, episodesResponse, colorsResponse, samplesResponse] = await Promise.all([
      fetch('/api/status'),
      fetch('/api/offline/episodes'),
      fetch('/api/markers/colors'),
      fetch('/api/samples'),
    ])
    for (const [response, label] of [
      [statusResponse, '状态加载失败'],
      [episodesResponse, 'episode 加载失败'],
      [colorsResponse, 'marker 颜色加载失败'],
      [samplesResponse, '样本加载失败'],
    ]) {
      if (!response.ok) throw await responseError(response, label)
    }
    const [statusData, episodeData, colorData, sampleData] = await Promise.all([
      statusResponse.json(),
      episodesResponse.json(),
      colorsResponse.json(),
      samplesResponse.json(),
    ])
    if (statusData.mode !== 'offline') {
      throw new Error('7013 点云页面只支持 --teleop-task-dir 离线模式')
    }
    status.value = statusData
    episodes.value = episodeData.episodes || []
    markerColors.value = colorData.colors || []
    samples.value = sampleData.samples || []
    if (!episodes.value.some((item) => item.name === selectedEpisode.value)) {
      selectedEpisode.value = episodes.value[0]?.name || ''
    }
    activeColor.value = episodeSamples.value[0]?.color || selectableColors.value[0]?.color || ''
  } catch (error) {
    setError(error)
  }
}

async function selectEpisode(name) {
  if (name === selectedEpisode.value || cloudBusy.value) return
  selectedEpisode.value = name
  activeColor.value = episodeSamples.value[0]?.color || selectableColors.value[0]?.color || ''
  await nextTick()
  await loadPointCloud()
}

async function confirmAndSave() {
  if (!canSave.value) return
  saveBusy.value = true
  errorMsg.value = ''
  infoMsg.value = ''
  try {
    if (!selections.value.length) {
      const deleteResponse = await fetch(
        `/api/samples/by-episode/${encodeURIComponent(selectedEpisode.value)}`,
        { method: 'DELETE' },
      )
      if (!deleteResponse.ok) {
        throw await responseError(deleteResponse, '清空已保存观测失败')
      }
      const deleted = await deleteResponse.json()
      await loadWorkspace()
      activeColor.value = selectableColors.value[0]?.color || ''
      infoMsg.value = `已清空 ${selectedEpisode.value} 的 ${deleted.deleted_count || 0} 个观测`
      return
    }

    const confirmResponse = await fetch('/api/offline/confirm-points', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        episode: selectedEpisode.value,
        cloud_id: cloudId.value,
        stride: cloudStride.value,
        selections: selections.value.map((item) => ({
          id: `marker-${item.color}`,
          color: item.color,
          vertex_index: item.vertexIndex,
        })),
      }),
    })
    if (!confirmResponse.ok) {
      throw await responseError(confirmResponse, '点云选点确认失败')
    }
    const confirmation = await confirmResponse.json()
    const saveResponse = await fetch('/api/samples/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        episode: selectedEpisode.value,
        observations: confirmation.observations,
        replace_existing: true,
      }),
    })
    if (!saveResponse.ok) throw await responseError(saveResponse, '样本保存失败')
    const saved = await saveResponse.json()
    await loadWorkspace()
    if (cloudObject) restoreSavedSelections(cloudObject.geometry)
    infoMsg.value = saved.updated_count
      ? `已更新 ${saved.updated_count} 个点云观测`
      : `已保存 ${saved.indices?.length || selections.value.length} 个点云观测`
  } catch (error) {
    setError(error)
  } finally {
    saveBusy.value = false
  }
}

async function solve() {
  solveBusy.value = true
  errorMsg.value = ''
  infoMsg.value = ''
  try {
    const response = await fetch('/api/solve', { method: 'POST' })
    if (!response.ok) throw await responseError(response, '解算失败')
    solveResult.value = await response.json()
    infoMsg.value = '联合解算完成'
  } catch (error) {
    setError(error)
  } finally {
    solveBusy.value = false
  }
}

watch(pointSize, (value) => {
  if (cloudMaterial) cloudMaterial.size = value / 1000
})

watch(cloudStride, async (value, oldValue) => {
  if (value !== oldValue && selectedEpisode.value && renderer) await loadPointCloud()
})

onMounted(async () => {
  initViewer()
  await loadWorkspace()
  if (selectedEpisode.value) await loadPointCloud()
})

onBeforeUnmount(() => {
  requestSerial += 1
  resizeObserver?.disconnect()
  if (renderer) {
    renderer.setAnimationLoop(null)
    renderer.domElement.removeEventListener('pointerdown', onPointerDown)
    renderer.domElement.removeEventListener('pointerup', onPointerUp)
    renderer.dispose()
  }
  controls?.dispose()
  disposeCloud()
})
</script>

<template>
  <main class="app-shell">
    <header class="topbar">
      <div>
        <h1>Hand-Eye 3D · 点云选点</h1>
        <p>离线 episode · 相机系 X 右 / Y 下 / Z 前 · 单位 m</p>
      </div>
      <div class="topbar-status">
        <span class="status-dot" :class="{ ready: status?.mode === 'offline' }"></span>
        {{ status?.mode === 'offline' ? '离线后端已连接' : '等待后端' }}
        <a :href="imageFrontendUrl">打开 7012 图像版</a>
      </div>
    </header>

    <section class="workspace">
      <aside class="episode-panel">
        <div class="panel-heading">
          <div>
            <h2>采集姿态</h2>
            <span>{{ episodes.length }} episodes</span>
          </div>
          <button class="icon-button" title="刷新" @click="loadWorkspace">↻</button>
        </div>
        <div class="episode-list">
          <button
            v-for="episode in episodes"
            :key="episode.name"
            class="episode-item"
            :class="{ active: episode.name === selectedEpisode }"
            :disabled="cloudBusy"
            @click="selectEpisode(episode.name)"
          >
            <span>{{ episode.name }}</span>
            <small>{{ episode.imported_marker_count || 0 }} 已保存</small>
            <small v-if="episode.warnings?.length" class="episode-warning">
              ⚠ {{ episode.warnings[0] }}
            </small>
          </button>
          <p v-if="!episodes.length" class="empty-state">没有可用 episode</p>
        </div>
      </aside>

      <section class="viewer-column">
        <div class="viewer-toolbar">
          <div>
            <strong>{{ selectedEpisode || '未选择 episode' }}</strong>
            <span v-if="pointCount">{{ pointCount.toLocaleString() }} points</span>
          </div>
          <label>
            点大小
            <input v-model.number="pointSize" type="range" min="1" max="12" step="1" />
          </label>
          <label>
            采样
            <select v-model.number="cloudStride" :disabled="cloudBusy">
              <option :value="1">1× 精细</option>
              <option :value="2">2× 默认</option>
              <option :value="3">3× 流畅</option>
              <option :value="4">4× 快速</option>
            </select>
          </label>
          <button class="secondary-button" :disabled="cloudBusy || !selectedEpisode" @click="loadPointCloud">
            重新加载
          </button>
        </div>
        <div ref="viewerHost" class="viewer">
          <div v-if="cloudBusy" class="viewer-overlay">
            <span class="spinner"></span>
            正在对齐五帧深度并生成点云…
          </div>
          <div class="axis-legend">
            <span class="x">X 右</span>
            <span class="y">Y 下</span>
            <span class="z">Z 前</span>
          </div>
          <div class="viewer-help">单击选点 · 左键拖动旋转 · 右键平移 · 滚轮缩放</div>
        </div>
        <div v-if="errorMsg" class="message error">{{ errorMsg }}</div>
        <div v-else-if="infoMsg" class="message success">{{ infoMsg }}</div>
      </section>

      <aside class="selection-panel">
        <section class="side-card">
          <div class="panel-heading compact">
            <div>
              <h2>1. 选择标记颜色</h2>
              <span>再到点云中单击对应中心</span>
            </div>
          </div>
          <div class="color-grid">
            <button
              v-for="marker in markerColors"
              :key="marker.color"
              class="color-chip"
              :class="{
                active: activeColor === marker.color,
                selected: selectedColors.has(marker.color),
                saved: savedColors.has(marker.color),
              }"
              @click="activeColor = marker.color"
            >
              <i :style="{ background: marker.display_color }"></i>
              <span>{{ marker.label_zh }}</span>
              <small v-if="savedColors.has(marker.color)">已保存</small>
              <small v-else-if="selectedColors.has(marker.color)">已选</small>
            </button>
          </div>
        </section>

        <section class="side-card grow">
          <div class="panel-heading compact">
            <div>
              <h2>2. 待保存选点</h2>
              <span>{{ selections.length }} 个</span>
            </div>
            <button class="text-button" :disabled="!selections.length" @click="clearSelections">清空</button>
          </div>
          <div class="selection-list">
            <article v-for="item in selections" :key="item.color" class="selection-row">
              <i :style="{ background: colorInfo(item.color).display_color }"></i>
              <div>
                <strong>{{ colorInfo(item.color).label_zh }}</strong>
                <code>
                  {{ item.point.map((value) => Number(value).toFixed(4)).join(', ') }}
                </code>
                <small>vertex #{{ item.vertexIndex }}</small>
              </div>
              <button title="删除" @click="removeSelection(item.color)">×</button>
            </article>
            <p v-if="!selections.length" class="empty-state">
              先选择颜色，再单击点云中的 marker 中心。
            </p>
          </div>
          <button class="primary-button" :disabled="!canSave" @click="confirmAndSave">
            {{
              saveBusy
                ? '确认并保存中…'
                : (!selections.length && episodeSamples.length
                  ? '保存清空结果'
                  : '确认并保存本姿态')
            }}
          </button>
        </section>

        <section class="side-card solve-card">
          <div>
            <h2>3. 联合解算</h2>
            <span>当前共 {{ samples.length }} 条有效标记观测</span>
          </div>
          <button class="solve-button" :disabled="solveBusy || samples.length < 6" @click="solve">
            {{ solveBusy ? '解算中…' : '运行手眼标定' }}
          </button>
          <div v-if="residualSummary" class="result-summary">
            <div><span>RMS</span><strong>{{ residualSummary.rms }} mm</strong></div>
            <div><span>Median</span><strong>{{ residualSummary.median }} mm</strong></div>
            <div><span>Max</span><strong>{{ residualSummary.max }} mm</strong></div>
          </div>
          <p v-if="solveResult?.saved_to" class="saved-path">{{ solveResult.saved_to }}</p>
        </section>
      </aside>
    </section>
  </main>
</template>
