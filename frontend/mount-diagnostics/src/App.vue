<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { STLLoader } from 'three/addons/loaders/STLLoader.js'

const viewerHost = ref(null)
const diagnostics = ref(null)
const loading = ref(false)
const errorMessage = ref('')
const selectedPoseId = ref('')
const selectedPointId = ref('')
const showModel = ref(true)
const showExpected = ref(true)
const showObserved = ref(true)
const showLines = ref(true)
const showLabels = ref(true)
const pointCloudFrontendUrl = `${window.location.protocol}//${window.location.hostname}:7013`

const pointInfo = (pointId) => {
  const number = Number(pointId?.split('-').at(-1))
  const red = pointId?.startsWith('palm-red-')
  return {
    label: `${red ? '红' : '绿'}${number}`,
    color: red ? '#ef4444' : '#22c55e',
  }
}

const selectedPose = computed(() =>
  diagnostics.value?.poses?.find((pose) => pose.pose_id === selectedPoseId.value) || null,
)
const selectedObservation = computed(() =>
  selectedPose.value?.observations?.find(
    (observation) => observation.point_id === selectedPointId.value,
  ) || null,
)
const sortedPointStats = computed(() =>
  [...(diagnostics.value?.point_stats || [])].sort((left, right) => right.rms - left.rms),
)
const looStats = computed(() => {
  const loo = diagnostics.value?.summary?.leave_one_pose_out
  return loo?.feasible ? loo.stats_mm : null
})
const quality = computed(() => {
  if (!diagnostics.value) return null
  if (diagnostics.value.stale) {
    return {
      tone: 'bad',
      label: '结果已过期',
      detail: '样本已改变，请回到 7013 重新解算。',
    }
  }
  const rms = Number(diagnostics.value.summary?.rms)
  if (rms <= 5) {
    return { tone: 'good', label: '拟合良好', detail: '整体 RMS 不高于 5 mm。' }
  }
  if (rms <= 10) {
    return {
      tone: 'warning',
      label: '误差偏大',
      detail: '建议优先检查红色标出的姿态和长连线。',
    }
  }
  return { tone: 'bad', label: '不建议使用', detail: '请修正选点后重新解算。' }
})

let scene
let camera
let renderer
let controls
let modelGroup
let diagnosticGroup
let resizeObserver
let animationFrame
const stlCache = new Map()

function matrixFromRows(rows) {
  const matrix = new THREE.Matrix4()
  matrix.set(
    rows[0][0], rows[0][1], rows[0][2], rows[0][3],
    rows[1][0], rows[1][1], rows[1][2], rows[1][3],
    rows[2][0], rows[2][1], rows[2][2], rows[2][3],
    rows[3][0], rows[3][1], rows[3][2], rows[3][3],
  )
  return matrix
}

function urdfOriginMatrix(xyz, rpy, scale) {
  const matrix = new THREE.Matrix4()
  matrix.compose(
    new THREE.Vector3(...xyz),
    new THREE.Quaternion().setFromEuler(
      new THREE.Euler(rpy[0], rpy[1], rpy[2], 'ZYX'),
    ),
    new THREE.Vector3(...scale),
  )
  return matrix
}

function loadStl(url) {
  if (!stlCache.has(url)) {
    stlCache.set(
      url,
      new Promise((resolve, reject) => {
        new STLLoader().load(
          url,
          resolve,
          undefined,
          () => reject(new Error(`模型加载失败：${url}`)),
        )
      }),
    )
  }
  return stlCache.get(url)
}

function initViewer() {
  if (renderer || !viewerHost.value) return
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x07101c)
  camera = new THREE.PerspectiveCamera(44, 1, 0.001, 10)
  camera.position.set(0.22, -0.2, 0.18)

  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.outputColorSpace = THREE.SRGBColorSpace
  viewerHost.value.appendChild(renderer.domElement)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.08
  scene.add(new THREE.HemisphereLight(0xe5f0ff, 0x17253a, 1.3))
  const key = new THREE.DirectionalLight(0xffffff, 1.5)
  key.position.set(0.5, -0.4, 0.8)
  scene.add(key)
  const fill = new THREE.DirectionalLight(0x73a9ff, 0.55)
  fill.position.set(-0.6, 0.5, -0.3)
  scene.add(fill)
  scene.add(new THREE.AxesHelper(0.05))

  modelGroup = new THREE.Group()
  diagnosticGroup = new THREE.Group()
  scene.add(modelGroup)
  scene.add(diagnosticGroup)
  renderer.domElement.addEventListener('pointerup', onViewerClick)
  resizeObserver = new ResizeObserver(resizeViewer)
  resizeObserver.observe(viewerHost.value)
  const render = () => {
    controls.update()
    renderer.render(scene, camera)
    animationFrame = requestAnimationFrame(render)
  }
  render()
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

function clearDiagnosticGroup() {
  if (!diagnosticGroup) return
  diagnosticGroup.traverse((item) => {
    item.element?.remove()
    item.geometry?.dispose()
    item.material?.map?.dispose()
    item.material?.dispose()
  })
  diagnosticGroup.clear()
}

function makeLabel(text, tone, pointId) {
  const palette = {
    good: { border: '#28875c', text: '#8af0bb', background: '#071910' },
    warning: { border: '#9a6b25', text: '#ffd27c', background: '#211707' },
    bad: { border: '#9d3b49', text: '#ff9caa', background: '#240b10' },
  }[tone]
  const canvas = document.createElement('canvas')
  canvas.width = 384
  canvas.height = 84
  const context = canvas.getContext('2d')
  context.fillStyle = palette.background
  context.strokeStyle = palette.border
  context.lineWidth = 4
  context.beginPath()
  context.roundRect(3, 3, canvas.width - 6, canvas.height - 6, 12)
  context.fill()
  context.stroke()
  context.fillStyle = palette.text
  context.font = '600 30px monospace'
  context.textAlign = 'center'
  context.textBaseline = 'middle'
  context.fillText(text, canvas.width / 2, canvas.height / 2)
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.minFilter = THREE.LinearFilter
  const label = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    }),
  )
  label.center.set(0.5, 0)
  label.scale.set(0.038, 0.0083, 1)
  label.renderOrder = 8
  label.userData.pointId = pointId
  return label
}

function residualColor(residualMm) {
  if (residualMm <= 5) return '#4ade80'
  if (residualMm <= 10) return '#f5b942'
  return '#fb7185'
}

function renderPoseDiagnostics() {
  clearDiagnosticGroup()
  const pose = selectedPose.value
  if (!pose || !diagnosticGroup) return
  for (const observation of pose.observations) {
    const selected = !selectedPointId.value || selectedPointId.value === observation.point_id
    const semantic = pointInfo(observation.point_id)
    const expected = new THREE.Vector3(...observation.model_point_hand_m)
    const observed = new THREE.Vector3(...observation.observed_point_hand_m)
    const opacity = selected ? 1 : 0.16

    if (showExpected.value) {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(selected ? 0.0042 : 0.0032, 20, 14),
        new THREE.MeshBasicMaterial({
          color: semantic.color,
          transparent: true,
          opacity,
          depthTest: false,
        }),
      )
      sphere.position.copy(expected)
      sphere.renderOrder = 3
      sphere.userData.pointId = observation.point_id
      diagnosticGroup.add(sphere)
    }

    if (showObserved.value) {
      const measured = new THREE.Mesh(
        new THREE.SphereGeometry(selected ? 0.0049 : 0.0038, 18, 12),
        new THREE.MeshBasicMaterial({
          color: 0xffffff,
          wireframe: true,
          transparent: true,
          opacity,
          depthTest: false,
        }),
      )
      measured.position.copy(observed)
      measured.renderOrder = 4
      measured.userData.pointId = observation.point_id
      diagnosticGroup.add(measured)
    }

    if (showLines.value) {
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([expected, observed]),
        new THREE.LineBasicMaterial({
          color: residualColor(observation.residual_mm),
          transparent: true,
          opacity: selected ? 0.95 : 0.12,
          depthTest: false,
        }),
      )
      line.renderOrder = 2
      line.userData.pointId = observation.point_id
      diagnosticGroup.add(line)
    }

    if (showLabels.value && selected) {
      const label = makeLabel(
        `${semantic.label}  ${observation.residual_mm.toFixed(1)} mm`,
        observation.residual_mm > 10 ? 'bad' : (observation.residual_mm > 5 ? 'warning' : 'good'),
        observation.point_id,
      )
      label.position.copy(expected)
      diagnosticGroup.add(label)
    }
  }
}

async function loadHandModel(handId) {
  const response = await fetch(`/api/hands/${encodeURIComponent(handId)}/model`)
  if (!response.ok) throw new Error(`手模型加载失败（HTTP ${response.status}）`)
  const payload = await response.json()
  const meshes = []
  for (const link of payload.links || []) {
    const THandLink = matrixFromRows(link.T_hand_link)
    for (const visual of link.visuals || []) {
      const geometry = await loadStl(visual.mesh_url)
      meshes.push({
        geometry,
        matrix: THandLink.clone().multiply(
          urdfOriginMatrix(visual.xyz, visual.rpy, visual.scale),
        ),
        color: visual.color,
      })
    }
  }
  modelGroup.clear()
  for (const item of meshes) {
    const mesh = new THREE.Mesh(
      item.geometry,
      new THREE.MeshStandardMaterial({
        color: item.color
          ? new THREE.Color(item.color[0], item.color[1], item.color[2])
          : 0x8b9bb0,
        transparent: true,
        opacity: showModel.value ? 0.48 : 0,
        metalness: 0.08,
        roughness: 0.7,
        depthWrite: false,
      }),
    )
    mesh.matrixAutoUpdate = false
    mesh.matrix.copy(item.matrix)
    modelGroup.add(mesh)
  }
}

function frameScene() {
  if (!camera || !controls) return
  const box = new THREE.Box3().setFromObject(modelGroup)
  if (box.isEmpty() && diagnosticGroup) box.setFromObject(diagnosticGroup)
  if (box.isEmpty()) return
  const uniqueModelPoints = new Map()
  for (const pose of diagnostics.value?.poses || []) {
    for (const observation of pose.observations || []) {
      uniqueModelPoints.set(observation.point_id, observation.model_point_hand_m)
    }
  }
  const center = new THREE.Vector3()
  for (const point of uniqueModelPoints.values()) {
    center.add(new THREE.Vector3(...point))
  }
  if (uniqueModelPoints.size) {
    center.multiplyScalar(1 / uniqueModelPoints.size)
  } else {
    box.getCenter(center)
  }
  const corners = [
    new THREE.Vector3(box.min.x, box.min.y, box.min.z),
    new THREE.Vector3(box.min.x, box.min.y, box.max.z),
    new THREE.Vector3(box.min.x, box.max.y, box.min.z),
    new THREE.Vector3(box.min.x, box.max.y, box.max.z),
    new THREE.Vector3(box.max.x, box.min.y, box.min.z),
    new THREE.Vector3(box.max.x, box.min.y, box.max.z),
    new THREE.Vector3(box.max.x, box.max.y, box.min.z),
    new THREE.Vector3(box.max.x, box.max.y, box.max.z),
  ]
  const radius = Math.max(
    ...corners.map((corner) => corner.distanceTo(center)),
    0.07,
  )
  controls.target.copy(center)
  camera.position.set(
    center.x + radius * 1.75,
    center.y - radius * 1.75,
    center.z + radius * 1.4,
  )
  camera.near = Math.max(0.0005, radius / 150)
  camera.far = radius * 60
  camera.updateProjectionMatrix()
  controls.update()
}

function selectPose(poseId) {
  selectedPoseId.value = poseId
  selectedPointId.value = ''
  nextTick(renderPoseDiagnostics)
}

function selectPoint(pointId) {
  selectedPointId.value = selectedPointId.value === pointId ? '' : pointId
  nextTick(renderPoseDiagnostics)
}

function onViewerClick(event) {
  if (!renderer || !camera || !diagnosticGroup) return
  const rect = renderer.domElement.getBoundingClientRect()
  const pointer = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  )
  const raycaster = new THREE.Raycaster()
  raycaster.params.Line.threshold = 0.003
  raycaster.setFromCamera(pointer, camera)
  const hit = raycaster
    .intersectObjects(diagnosticGroup.children, false)
    .find((item) => item.object.userData.pointId)
  if (hit) selectPoint(hit.object.userData.pointId)
}

async function refreshDiagnostics() {
  loading.value = true
  errorMessage.value = ''
  try {
    const response = await fetch('/api/mount/diagnostics')
    const payload = await response.json()
    if (!response.ok) throw new Error(payload.error || '诊断数据加载失败')
    diagnostics.value = payload.available ? payload : null
    if (!payload.available) {
      errorMessage.value = payload.reason || '尚无可视化结果'
      clearDiagnosticGroup()
      return
    }
    const worstPose = [...payload.poses].sort((left, right) => right.rms - left.rms)[0]
    selectedPoseId.value = worstPose?.pose_id || ''
    selectedPointId.value = ''
    await nextTick()
    try {
      initViewer()
      await loadHandModel(payload.hand_id)
      renderPoseDiagnostics()
      nextTick(frameScene)
    } catch (error) {
      errorMessage.value = `三维视图初始化失败：${error.message || String(error)}`
    }
  } catch (error) {
    errorMessage.value = error.message || String(error)
  } finally {
    loading.value = false
  }
}

watch(
  [showExpected, showObserved, showLines, showLabels],
  renderPoseDiagnostics,
)
watch(showModel, (visible) => {
  modelGroup?.traverse((item) => {
    if (item.material) item.material.opacity = visible ? 0.48 : 0
  })
})

onMounted(async () => {
  await refreshDiagnostics()
})

onBeforeUnmount(() => {
  cancelAnimationFrame(animationFrame)
  resizeObserver?.disconnect()
  renderer?.domElement.removeEventListener('pointerup', onViewerClick)
  controls?.dispose()
  renderer?.dispose()
})
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div>
        <span class="port-mark">7015</span>
        <div>
          <h1>灵巧手安装标定诊断</h1>
          <p>实心点是模型位置，白色线框点是实际观测；连线越长，误差越大</p>
        </div>
      </div>
      <nav>
        <a :href="pointCloudFrontendUrl">返回 7013</a>
        <button :disabled="loading" @click="refreshDiagnostics">
          {{ loading ? '加载中…' : '刷新结果' }}
        </button>
      </nav>
    </header>

    <div v-if="errorMessage && !diagnostics" class="empty-state">
      <strong>暂无可视化结果</strong>
      <p>{{ errorMessage }}</p>
      <a :href="pointCloudFrontendUrl">先到 7013 完成解算</a>
    </div>

    <main v-else-if="diagnostics" class="workspace">
      <aside class="left-panel">
        <section class="card result-card">
          <div class="card-heading">
            <span>整体结果</span>
            <strong :class="quality.tone">{{ quality.label }}</strong>
          </div>
          <p>{{ quality.detail }}</p>
          <div class="metric-grid">
            <div><span>RMS</span><b>{{ diagnostics.summary.rms.toFixed(2) }} mm</b></div>
            <div><span>最大</span><b>{{ diagnostics.summary.max.toFixed(2) }} mm</b></div>
            <div><span>样本</span><b>{{ diagnostics.summary.sample_count }}</b></div>
            <div><span>姿态</span><b>{{ diagnostics.summary.pose_count }}</b></div>
            <div v-if="looStats"><span>跨姿态 RMS</span><b>{{ looStats.rms.toFixed(2) }} mm</b></div>
            <div><span>模型点</span><b>{{ diagnostics.summary.point_count }}/16</b></div>
            <div><span>红点 RMS</span><b>{{ diagnostics.summary.by_color.red.rms.toFixed(2) }} mm</b></div>
            <div><span>绿点 RMS</span><b>{{ diagnostics.summary.by_color.green.rms.toFixed(2) }} mm</b></div>
          </div>
        </section>

        <section
          class="card order-card"
          :class="diagnostics.order_check.verdict"
        >
          <div class="card-heading">
            <span>点序自动检查</span>
            <strong>{{ diagnostics.order_check.verdict === 'consistent' ? '通过' : '需复核' }}</strong>
          </div>
          <h2>{{ diagnostics.order_check.label }}</h2>
          <p>{{ diagnostics.order_check.message }}</p>
          <small>
            当前 {{ diagnostics.order_check.current_rms_mm.toFixed(2) }} mm
            · 最佳互换 {{ diagnostics.order_check.best_swap_rms_mm.toFixed(2) }} mm
          </small>
        </section>

        <section class="card pose-card">
          <div class="card-heading">
            <span>选择姿态</span>
            <small>默认打开误差最大项</small>
          </div>
          <button
            v-for="pose in diagnostics.poses"
            :key="pose.pose_id"
            class="pose-row"
            :class="{
              active: pose.pose_id === selectedPoseId,
              bad: pose.rms > 10,
              warning: pose.rms > 5 && pose.rms <= 10,
            }"
            @click="selectPose(pose.pose_id)"
          >
            <span>
              <strong>{{ pose.pose_id }}</strong>
              <small>{{ pose.count }} 个点</small>
            </span>
            <i><b :style="{ width: `${Math.min(100, pose.rms * 5)}%` }"></b></i>
            <code>{{ pose.rms.toFixed(2) }} mm</code>
          </button>
        </section>
      </aside>

      <section class="viewer-panel">
        <div class="viewer-toolbar">
          <label><input v-model="showModel" type="checkbox" /> 手模型</label>
          <label><input v-model="showExpected" type="checkbox" /> 模型点</label>
          <label><input v-model="showObserved" type="checkbox" /> 实测点</label>
          <label><input v-model="showLines" type="checkbox" /> 误差连线</label>
          <label><input v-model="showLabels" type="checkbox" /> 编号</label>
          <button @click="frameScene">回到手心视角</button>
        </div>
        <div ref="viewerHost" class="viewer"></div>
        <div v-if="errorMessage" class="viewer-error">
          <strong>三维视图未能显示</strong>
          <span>{{ errorMessage }}</span>
          <button @click="refreshDiagnostics">重试</button>
        </div>
        <div class="viewer-caption">
          <span><i class="solid red"></i>模型红点</span>
          <span><i class="solid green"></i>模型绿点</span>
          <span><i class="ring"></i>实际观测</span>
          <span><i class="line good"></i>≤5 mm</span>
          <span><i class="line warning"></i>5–10 mm</span>
          <span><i class="line bad"></i>&gt;10 mm</span>
        </div>
      </section>

      <aside class="right-panel">
        <section v-if="selectedPose" class="card pose-detail">
          <div class="card-heading">
            <span>{{ selectedPose.pose_id }}</span>
            <strong :class="{ bad: selectedPose.rms > 10 }">
              RMS {{ selectedPose.rms.toFixed(2) }} mm
            </strong>
          </div>
          <p>点击下列编号可在三维视图中单独高亮，再次点击恢复全部。</p>
          <button
            v-for="observation in selectedPose.observations"
            :key="observation.sample_index"
            class="observation-row"
            :class="{
              active: observation.point_id === selectedPointId,
              bad: observation.residual_mm > 10,
              warning: observation.residual_mm > 5 && observation.residual_mm <= 10,
            }"
            @click="selectPoint(observation.point_id)"
          >
            <i :style="{ background: pointInfo(observation.point_id).color }"></i>
            <strong>{{ observation.short_label }}</strong>
            <span>#{{ observation.sample_index }}</span>
            <code>{{ observation.residual_mm.toFixed(2) }} mm</code>
          </button>
        </section>

        <section v-if="selectedObservation" class="card coordinate-card">
          <div class="card-heading">
            <span>{{ selectedObservation.short_label }} 坐标对比（手模型系）</span>
          </div>
          <div>
            <span>模型 XYZ（m）</span>
            <code>{{ selectedObservation.model_point_hand_m.map((value) => value.toFixed(5)).join(', ') }}</code>
          </div>
          <div>
            <span>实测 XYZ（m）</span>
            <code>{{ selectedObservation.observed_point_hand_m.map((value) => value.toFixed(5)).join(', ') }}</code>
          </div>
          <div>
            <span>误差 XYZ（mm）</span>
            <code>{{ selectedObservation.residual_vector_hand_mm.map((value) => value.toFixed(2)).join(', ') }}</code>
          </div>
        </section>

        <section class="card point-ranking">
          <div class="card-heading">
            <span>各编号总体误差</span>
            <small>从高到低</small>
          </div>
          <div
            v-for="point in sortedPointStats"
            :key="point.point_id"
            :class="{ bad: point.rms > 10, warning: point.rms > 5 && point.rms <= 10 }"
          >
            <i :style="{ background: pointInfo(point.point_id).color }"></i>
            <strong>{{ point.short_label }}</strong>
            <span>{{ point.count }} 次</span>
            <code>{{ point.rms.toFixed(2) }} mm</code>
          </div>
        </section>

        <details class="card transform-card">
          <summary>T_wrist2hand</summary>
          <code
            v-for="(row, index) in diagnostics.T_wrist2hand"
            :key="index"
          >{{ row.map((value) => Number(value).toFixed(6)).join('  ') }}</code>
          <small>
            相机 {{ diagnostics.calib_camera?.serial || '未知' }}
            · {{ diagnostics.solved_at }}
          </small>
        </details>
      </aside>
    </main>
  </div>
</template>
