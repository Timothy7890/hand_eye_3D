<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'
import { STLLoader } from 'three/addons/loaders/STLLoader.js'

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

// 手安装标定分两阶段：先一次性标完 16 个模型点，再按 episode 配对点云点。
const mode = ref('marker')
const hands = ref([])
const selectedHandId = ref('')
const handModel = ref(null)
const handBusy = ref(false)
const activeMountSlotId = ref('palm-red-01')
const mountDrafts = ref([])
const mountSamples = ref([])
const mountMinPoints = ref(3)
const mountSavedCloudPoints = ref([])
const mountCandidates = ref([])
const mountCandidateWarnings = ref([])
const mountCandidateBusy = ref(false)
const mountProfiles = ref([])
const selectedMountProfileId = ref('')
const loadedMountProfileId = ref('')
const mountProfileName = ref('')
const mountProfileBusy = ref(false)
const mountProfileDirty = ref(false)
const mountSaveBusy = ref(false)
const mountSolveBusy = ref(false)
const mountResult = ref(null)
const overlayVisible = ref(false)
const mountViewport = ref('model')
const handViewerHost = ref(null)
const mountSlots = [
  ...Array.from({ length: 8 }, (_, index) => ({
    point_id: `palm-red-${String(index + 1).padStart(2, '0')}`,
    label: `手心红点 ${String(index + 1).padStart(2, '0')}`,
    shortLabel: `红${index + 1}`,
    side: 'palm',
    color: '#ef4444',
  })),
  ...Array.from({ length: 8 }, (_, index) => ({
    point_id: `back-green-${String(index + 1).padStart(2, '0')}`,
    label: `手背绿点 ${String(index + 1).padStart(2, '0')}`,
    shortLabel: `绿${index + 1}`,
    side: 'back',
    color: '#22c55e',
  })),
]

let scene
let camera
let renderer
let controls
let cloudObject
let cloudMaterial
let markerGroup
let mountCandidateGroup
let resizeObserver
let requestSerial = 0
let pointerStart = null

let handScene
let handCamera
let handRenderer
let handControls
let handMeshGroup
let handPointGroup
let handResizeObserver
let handPointerStart = null
let handLoadSerial = 0
let overlayGroup
const stlCache = new Map()

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
const currentHand = computed(() =>
  hands.value.find((item) => item.hand_id === selectedHandId.value) || null,
)
const mountSavedForEpisode = computed(() =>
  mountSamples.value.filter((sample) => sample.pose_id === selectedEpisode.value),
)
const mountSamplesByPose = computed(() => {
  const poses = new Set(mountSamples.value.map((sample) => sample.pose_id))
  return poses.size
})
const mountDraftIds = computed(() =>
  new Set(mountDrafts.value.filter((item) => item.p_hand).map((item) => item.point_id)),
)
const allMountModelPointsSelected = computed(() =>
  mountDraftIds.value.size === mountSlots.length,
)
const selectedMountProfile = computed(() =>
  mountProfiles.value.find((profile) => profile.profile_id === selectedMountProfileId.value) || null,
)
const canSaveMountProfile = computed(() =>
  mountDraftIds.value.size > 0
  && Boolean(selectedHandId.value)
  && Boolean(mountProfileName.value.trim())
  && !mountProfileBusy.value,
)
const mountPairedIds = computed(() =>
  new Set(mountDrafts.value.filter((item) => item.vertexIndex != null).map((item) => item.point_id)),
)
const mountPairedDrafts = computed(() =>
  mountDrafts.value.filter((item) => item.vertexIndex != null),
)
const mountCandidateCounts = computed(() => ({
  red: mountCandidates.value.filter((item) => item.color === 'red').length,
  green: mountCandidates.value.filter((item) => item.color === 'green').length,
}))
const mountSavedIds = computed(() =>
  new Set(mountSavedForEpisode.value.map((item) => item.point_id)),
)
const activeMountDraft = computed(() =>
  mountDrafts.value.find((item) => item.point_id === activeMountSlotId.value) || null,
)
const activeMountSlot = computed(() =>
  mountSlots.find((item) => item.point_id === activeMountSlotId.value) || null,
)
const canSave = computed(() =>
  (selections.value.length > 0 || episodeSamples.value.length > 0)
  && !cloudBusy.value
  && !saveBusy.value
  && Boolean(selectedEpisode.value),
)
const canSaveMount = computed(() =>
  mountDrafts.value.some((item) => item.vertexIndex != null)
  && !cloudBusy.value
  && !mountSaveBusy.value
  && Boolean(selectedEpisode.value)
  && Boolean(cloudId.value)
  && Boolean(selectedHandId.value),
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
const mountResidualSummary = computed(() => {
  const residual = mountResult.value?.residual_mm
  if (!residual) return null
  return {
    rms: Number(residual.rms).toFixed(2),
    median: Number(residual.median).toFixed(2),
    max: Number(residual.max).toFixed(2),
  }
})
const overlayAvailable = computed(() =>
  Boolean(mountResult.value?.per_pose_overlay_T_camera_hand?.[selectedEpisode.value]),
)

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
  mountCandidateGroup = new THREE.Group()
  scene.add(mountCandidateGroup)

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

function restoreSavedMountPoints(geometry) {
  const position = geometry?.getAttribute('position')
  if (!position) {
    mountSavedCloudPoints.value = []
    return 0
  }
  mountSavedCloudPoints.value = mountSavedForEpisode.value
    .filter((sample) => Array.isArray(sample.p_camera) && sample.p_camera.length === 3)
    .map((sample) => {
      const target = sample.p_camera.map(Number)
      const savedCloudId = sample.cloud_id || sample.provenance?.cloud_id
      const savedStride = Number(
        sample.point_cloud_stride || sample.provenance?.point_cloud_stride,
      )
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
        ...sample,
        vertexIndex,
        point,
        displayPoint: [...point],
      }
    })
    .filter((item) => item.vertexIndex >= 0)
  refreshHighlights()
  return mountSavedCloudPoints.value.length
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
  keepOnlyMountModelPoints()
  mountSavedCloudPoints.value = []
  mountCandidates.value = []
  mountCandidateWarnings.value = []
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
    const restoredMountCount = restoreSavedMountPoints(geometry)
    if (mode.value === 'mount' && restoredMountCount) {
      infoMsg.value = `已恢复本姿态 ${restoredMountCount} 个安装配对点`
    } else {
      infoMsg.value = restoredCount
        ? `已恢复 ${restoredCount} 个已保存选点，可选择颜色后重新选点`
        : `已加载 ${pointCount.value.toLocaleString()} 个稳定点`
    }
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
  if (!pointerStart || !cloudObject) return
  if (mode.value === 'marker' && !activeColor.value) return
  if (mode.value === 'mount' && !activeMountDraft.value?.p_hand) {
    const moved = Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y)
    pointerStart = null
    if (moved <= 5) infoMsg.value = '请先选择槽位，再切到中央零位手模型选择模型点'
    return
  }
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
  let detectedCandidate = null
  if (mode.value === 'mount' && mountCandidateGroup) {
    const candidateHit = raycaster.intersectObjects(
      mountCandidateGroup.children,
      false,
    )[0]
    detectedCandidate = candidateHit?.object?.userData?.mountCandidate || null
    if (detectedCandidate) {
      const expectedColor = activeMountSlot.value?.side === 'palm' ? 'red' : 'green'
      if (detectedCandidate.color !== expectedColor) {
        infoMsg.value = `当前是${activeMountSlot.value?.label}，请选择${expectedColor === 'red' ? '红色' : '绿色'}候选点`
        return
      }
    }
  }
  const candidate = detectedCandidate
    ? {
        hit: { index: Number(detectedCandidate.vertex_index) },
        screenDistancePx: 0,
      }
    : raycaster
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

  if (mode.value === 'mount') {
    const draft = activeMountDraft.value
    const duplicateDraft = mountDrafts.value.find(
      (item) => item.point_id !== draft.point_id && item.vertexIndex === hit.index,
    )
    const duplicateSaved = mountSavedForEpisode.value.find(
      (item) => item.point_id !== draft.point_id
        && Number(item.vertex_index ?? item.provenance?.vertex_index) === hit.index,
    )
    if (duplicateDraft || duplicateSaved) {
      infoMsg.value = `该实体候选已分配给 ${duplicateDraft?.label || duplicateSaved?.label || duplicateSaved?.point_id}`
      return
    }
    mountDrafts.value = [
      ...mountDrafts.value.filter((item) => item.point_id !== draft.point_id),
      {
        ...draft,
        vertexIndex: hit.index,
        point,
        displayPoint,
        candidateId: detectedCandidate?.candidate_id,
      },
    ]
    infoMsg.value = `${draft.label} 已完成配对（距点击 ${screenDistancePx.toFixed(1)} px）`
    refreshHighlights()
    refreshHandPointMarkers()
    const next = chooseNextMountCloudSlot(draft.point_id)
    infoMsg.value = next
      ? `${draft.label} 已完成；下一项：${next.label}`
      : `${draft.label} 已完成；当前 episode 的点云配对已完成`
    return
  }

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
  const highlightItems = mode.value === 'mount'
    ? [
        ...mountSavedCloudPoints.value.map((selection) => ({
          selection,
          color: mountSlotInfo(selection.point_id).color,
          saved: true,
        })),
        ...mountDrafts.value
          .filter((selection) => selection.vertexIndex != null)
          .map((selection) => ({
            selection,
            color: mountSlotInfo(selection.point_id).color,
            saved: false,
          })),
      ]
    : selections.value.map((selection) => ({
        selection,
        color: colorInfo(selection.color).display_color,
        saved: false,
      }))
  for (const { selection, color, saved } of highlightItems) {
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(saved ? 0.0032 : 0.0042, 16, 12),
      new THREE.MeshBasicMaterial({
        color,
        transparent: saved,
        opacity: saved ? 0.65 : 1,
      }),
    )
    const displayPoint = selection.displayPoint || [
      selection.point[0],
      selection.point[1],
      selection.point[2],
    ]
    mesh.position.fromArray(displayPoint)
    markerGroup.add(mesh)
  }
  refreshMountCandidateMarkers()
}

function refreshMountCandidateMarkers() {
  if (!mountCandidateGroup) return
  while (mountCandidateGroup.children.length) {
    const child = mountCandidateGroup.children[0]
    mountCandidateGroup.remove(child)
    child.geometry?.dispose()
    child.material?.dispose()
  }
  mountCandidateGroup.visible = mode.value === 'mount'
  if (!mountCandidateGroup.visible) return
  const usedVertices = new Set([
    ...mountSavedCloudPoints.value.map((item) => item.vertexIndex),
    ...mountDrafts.value
      .filter((item) => item.vertexIndex != null)
      .map((item) => item.vertexIndex),
  ])
  for (const candidate of mountCandidates.value) {
    if (usedVertices.has(candidate.vertex_index)) continue
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.0065, 16, 12),
      new THREE.MeshBasicMaterial({
        color: candidate.color === 'red' ? '#ef4444' : '#22c55e',
        transparent: true,
        opacity: 0.55,
      }),
    )
    mesh.position.fromArray(candidate.p_camera)
    mesh.userData.mountCandidate = candidate
    mountCandidateGroup.add(mesh)
  }
}

function mountSlotInfo(pointId) {
  return mountSlots.find((slot) => slot.point_id === pointId) || {
    point_id: pointId,
    label: pointId,
    color: '#f8fafc',
  }
}

function modelOnlyMountDraft(item) {
  const slot = mountSlotInfo(item.point_id)
  return {
    ...slot,
    link: item.link,
    p_local: Array.isArray(item.p_local) ? [...item.p_local] : [...item.p_hand],
    p_hand: [...item.p_hand],
    meshFaceIndex: item.meshFaceIndex,
  }
}

function savedModelDraftsForHand(handId) {
  const byId = new Map()
  for (const sample of mountSamples.value) {
    if (
      sample.hand_id === handId
      && mountSlots.some((slot) => slot.point_id === sample.point_id)
      && Array.isArray(sample.p_hand)
      && sample.p_hand.length === 3
    ) {
      byId.set(sample.point_id, modelOnlyMountDraft(sample))
    }
  }
  return mountSlots
    .map((slot) => byId.get(slot.point_id))
    .filter(Boolean)
}

function keepOnlyMountModelPoints() {
  mountDrafts.value = mountDrafts.value
    .filter((item) => Array.isArray(item.p_hand))
    .map(modelOnlyMountDraft)
}

async function refreshMountProfiles(handId = selectedHandId.value) {
  if (!handId) {
    mountProfiles.value = []
    selectedMountProfileId.value = ''
    return
  }
  const response = await fetch(
    `/api/mount/model-point-profiles?hand_id=${encodeURIComponent(handId)}`,
  )
  if (!response.ok) throw await responseError(response, '模型点方案加载失败')
  const data = await response.json()
  mountProfiles.value = data.profiles || []
  if (!mountProfiles.value.some(
    (profile) => profile.profile_id === selectedMountProfileId.value,
  )) {
    selectedMountProfileId.value = mountProfiles.value[0]?.profile_id || ''
  }
}

function selectMountProfile() {
  if (selectedMountProfile.value) {
    mountProfileName.value = selectedMountProfile.value.name
  }
}

async function saveMountProfile() {
  if (!canSaveMountProfile.value) return
  mountProfileBusy.value = true
  errorMsg.value = ''
  try {
    const points = mountSlots.flatMap((slot) => {
      const draft = mountDrafts.value.find((item) => item.point_id === slot.point_id)
      if (!draft?.p_hand) return []
      return [{
        point_id: slot.point_id,
        label: slot.label,
        link: draft.link,
        p_local: draft.p_local,
        p_hand: draft.p_hand,
      }]
    })
    const response = await fetch('/api/mount/model-point-profiles', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        schema_version: 1,
        name: mountProfileName.value.trim(),
        hand_id: selectedHandId.value,
        points,
      }),
    })
    if (!response.ok) throw await responseError(response, '模型点方案保存失败')
    const data = await response.json()
    await refreshMountProfiles(selectedHandId.value)
    selectedMountProfileId.value = data.profile.profile_id
    loadedMountProfileId.value = data.profile.profile_id
    mountProfileName.value = data.profile.name
    mountProfileDirty.value = false
    infoMsg.value = data.created
      ? `模型点方案“${data.profile.name}”已保存（${data.profile.point_count}/16）`
      : `模型点方案“${data.profile.name}”已覆盖（${data.profile.point_count}/16）`
  } catch (error) {
    setError(error)
  } finally {
    mountProfileBusy.value = false
  }
}

async function loadMountProfile() {
  const profileId = selectedMountProfileId.value
  if (!profileId || mountProfileBusy.value) return
  if (
    mountProfileDirty.value
    && !window.confirm('当前模型点有未保存修改，确定加载其他方案并覆盖吗？')
  ) {
    return
  }
  mountProfileBusy.value = true
  errorMsg.value = ''
  try {
    const response = await fetch(
      `/api/mount/model-point-profiles/${encodeURIComponent(profileId)}`,
    )
    if (!response.ok) throw await responseError(response, '模型点方案读取失败')
    const data = await response.json()
    const profile = data.profile
    if (profile.hand_id !== selectedHandId.value) {
      throw new Error('模型点方案与当前手型号不匹配')
    }
    mountDrafts.value = profile.points.map(modelOnlyMountDraft)
    loadedMountProfileId.value = profile.profile_id
    mountProfileName.value = profile.name
    mountProfileDirty.value = false
    activeMountSlotId.value = mountSlots.find(
      (slot) => !mountDraftIds.value.has(slot.point_id),
    )?.point_id || mountSlots[0].point_id
    mountViewport.value = 'model'
    refreshHighlights()
    refreshHandPointMarkers()
    infoMsg.value = `已加载模型点方案“${profile.name}”的 ${profile.points.length} 个点`
  } catch (error) {
    setError(error)
  } finally {
    mountProfileBusy.value = false
  }
}

async function deleteMountProfile() {
  const profile = selectedMountProfile.value
  if (!profile || mountProfileBusy.value) return
  if (!window.confirm(`确定删除模型点方案“${profile.name}”吗？`)) return
  mountProfileBusy.value = true
  errorMsg.value = ''
  try {
    const response = await fetch(
      `/api/mount/model-point-profiles/${encodeURIComponent(profile.profile_id)}`,
      { method: 'DELETE' },
    )
    if (!response.ok) throw await responseError(response, '模型点方案删除失败')
    if (loadedMountProfileId.value === profile.profile_id) {
      loadedMountProfileId.value = ''
      mountProfileDirty.value = mountDraftIds.value.size > 0
    }
    await refreshMountProfiles(selectedHandId.value)
    infoMsg.value = `模型点方案“${profile.name}”已删除；当前模型点未清空`
  } catch (error) {
    setError(error)
  } finally {
    mountProfileBusy.value = false
  }
}

function nextMountSlotAfter(pointId, isPending) {
  const currentIndex = mountSlots.findIndex((slot) => slot.point_id === pointId)
  const ordered = currentIndex < 0
    ? mountSlots
    : [...mountSlots.slice(currentIndex + 1), ...mountSlots.slice(0, currentIndex)]
  return ordered.find(isPending)
}

function chooseNextMountModelSlot(currentPointId = '') {
  const next = nextMountSlotAfter(
    currentPointId,
    (slot) => !mountDraftIds.value.has(slot.point_id),
  )
  if (next) activeMountSlotId.value = next.point_id
  return next
}

function chooseNextMountCloudSlot(currentPointId = '') {
  const next = nextMountSlotAfter(
    currentPointId,
    (slot) =>
      !mountPairedIds.value.has(slot.point_id) && !mountSavedIds.value.has(slot.point_id),
  )
  if (next) {
    activeMountSlotId.value = next.point_id
    mountViewport.value = 'cloud'
  }
  return next
}

async function detectMountCandidates() {
  if (!selectedEpisode.value || !cloudId.value || mountCandidateBusy.value) return
  mountCandidateBusy.value = true
  errorMsg.value = ''
  try {
    const response = await fetch('/api/offline/detect-mount-candidates', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        episode: selectedEpisode.value,
        stride: cloudStride.value,
      }),
    })
    if (!response.ok) throw await responseError(response, '安装圆点检测失败')
    const data = await response.json()
    if (data.cloud_id !== cloudId.value) {
      throw new Error('检测返回的点云已变化，请重新加载点云')
    }
    mountCandidates.value = data.candidates || []
    mountCandidateWarnings.value = data.warnings || []
    refreshMountCandidateMarkers()
    const red = Number(data.counts?.red) || 0
    const green = Number(data.counts?.green) || 0
    infoMsg.value = `RGB 自动候选：红 ${red} 个，绿 ${green} 个`
    if (data.rejected_count) {
      infoMsg.value += `；另有 ${data.rejected_count} 个圆心缺少稳定深度`
    }
  } catch (error) {
    mountCandidates.value = []
    mountCandidateWarnings.value = []
    refreshMountCandidateMarkers()
    setError(error)
  } finally {
    mountCandidateBusy.value = false
  }
}

async function beginMountCloudPairing() {
  if (!allMountModelPointsSelected.value) {
    mountViewport.value = 'model'
    infoMsg.value = `请先标完 16 个模型点，当前已完成 ${mountDraftIds.value.size} 个`
    return
  }
  if (!selectedEpisode.value || !cloudId.value) {
    infoMsg.value = '请先选择并加载一个 episode 点云'
    return
  }
  mountViewport.value = 'cloud'
  const next = chooseNextMountCloudSlot()
  infoMsg.value = next
    ? `模型 16 点已完成，请在当前 episode 点云选择 ${next.label}`
    : '当前 episode 的可见点已配对；可保存或选择其他 episode'
  if (!mountCandidates.value.length) await detectMountCandidates()
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

// ---------- 手安装标定：零位手模型与固定槽配对 ----------

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
    new THREE.Quaternion().setFromEuler(new THREE.Euler(rpy[0], rpy[1], rpy[2], 'ZYX')),
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
          () => reject(new Error(`模型加载失败: ${url}`)),
        )
      }),
    )
  }
  return stlCache.get(url)
}

function clearGroup(group, disposeGeometry = false) {
  if (!group) return
  while (group.children.length) {
    const child = group.children[0]
    group.remove(child)
    if (disposeGeometry) child.geometry?.dispose()
    child.material?.dispose()
  }
}

function initHandViewer() {
  if (handRenderer || !handViewerHost.value) return
  handScene = new THREE.Scene()
  handScene.background = new THREE.Color(0x0b1220)
  handCamera = new THREE.PerspectiveCamera(45, 1, 0.001, 10)
  handCamera.position.set(0.22, -0.18, 0.22)

  handRenderer = new THREE.WebGLRenderer({ antialias: true })
  handRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  handRenderer.outputColorSpace = THREE.SRGBColorSpace
  handViewerHost.value.appendChild(handRenderer.domElement)

  handControls = new OrbitControls(handCamera, handRenderer.domElement)
  handControls.enableDamping = true
  handControls.dampingFactor = 0.08
  handScene.add(new THREE.HemisphereLight(0xdbeafe, 0x1e293b, 1.1))
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.4)
  keyLight.position.set(0.5, 0.8, 1)
  handScene.add(keyLight)
  const fillLight = new THREE.DirectionalLight(0x93c5fd, 0.5)
  fillLight.position.set(-0.6, -0.4, -0.8)
  handScene.add(fillLight)
  handScene.add(new THREE.AxesHelper(0.05))

  handMeshGroup = new THREE.Group()
  handPointGroup = new THREE.Group()
  handScene.add(handMeshGroup)
  handScene.add(handPointGroup)
  handRenderer.domElement.addEventListener('pointerdown', onHandPointerDown)
  handRenderer.domElement.addEventListener('pointerup', onHandPointerUp)
  handResizeObserver = new ResizeObserver(resizeHandViewer)
  handResizeObserver.observe(handViewerHost.value)
  handRenderer.setAnimationLoop(() => {
    handControls.update()
    handRenderer.render(handScene, handCamera)
  })
  resizeHandViewer()
}

function resizeHandViewer() {
  if (!handRenderer || !handViewerHost.value) return
  const width = Math.max(1, handViewerHost.value.clientWidth)
  const height = Math.max(1, handViewerHost.value.clientHeight)
  handRenderer.setSize(width, height, false)
  handCamera.aspect = width / height
  handCamera.updateProjectionMatrix()
}

function frameHandModel() {
  const box = new THREE.Box3().setFromObject(handMeshGroup)
  if (box.isEmpty()) return
  const center = box.getCenter(new THREE.Vector3())
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 0.06)
  handControls.target.copy(center)
  handCamera.position.set(
    center.x + radius * 1.6,
    center.y - radius * 1.2,
    center.z + radius * 1.6,
  )
  handCamera.near = radius / 100
  handCamera.far = radius * 50
  handCamera.updateProjectionMatrix()
  handControls.update()
}

function refreshHandPointMarkers() {
  clearGroup(handPointGroup, true)
  if (!handModel.value) return
  const saved = mountSavedForEpisode.value.filter(
    (sample) => sample.hand_id === selectedHandId.value && Array.isArray(sample.p_hand),
  )
  const byId = new Map(saved.map((item) => [item.point_id, { ...item, saved: true }]))
  for (const draft of mountDrafts.value) {
    if (draft.p_hand) byId.set(draft.point_id, { ...draft, saved: false })
  }
  for (const item of byId.values()) {
    const active = activeMountSlotId.value === item.point_id
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(active ? 0.0055 : 0.004, 16, 12),
      new THREE.MeshBasicMaterial({
        color: mountSlotInfo(item.point_id).color,
      }),
    )
    mesh.position.fromArray(item.p_hand)
    handPointGroup.add(mesh)
  }
}

async function loadHandModel() {
  if (!selectedHandId.value || !handRenderer) return
  const serial = ++handLoadSerial
  handBusy.value = true
  errorMsg.value = ''
  try {
    // 不传 joints，后端固定返回全零关节模型。
    const response = await fetch(`/api/hands/${encodeURIComponent(selectedHandId.value)}/model`)
    if (!response.ok) throw await responseError(response, '手模型加载失败')
    const payload = await response.json()
    const meshes = []
    for (const link of payload.links || []) {
      const THandLink = matrixFromRows(link.T_hand_link)
      for (const visual of link.visuals || []) {
        const geometry = await loadStl(visual.mesh_url)
        meshes.push({
          geometry,
          link: link.link,
          THandLink,
          matrix: THandLink.clone().multiply(
            urdfOriginMatrix(visual.xyz, visual.rpy, visual.scale),
          ),
        })
      }
    }
    if (serial !== handLoadSerial) return
    handModel.value = payload
    mountDrafts.value = savedModelDraftsForHand(payload.hand_id)
    loadedMountProfileId.value = ''
    mountProfileName.value = ''
    mountProfileDirty.value = mountDrafts.value.length > 0
    activeMountSlotId.value = mountSlots.find(
      (slot) => !mountDraftIds.value.has(slot.point_id),
    )?.point_id || mountSlots[0].point_id
    mountResult.value = null
    overlayVisible.value = false
    clearGroup(handMeshGroup)
    for (const item of meshes) {
      const mesh = new THREE.Mesh(
        item.geometry,
        new THREE.MeshStandardMaterial({
          color: 0x94a3b8,
          metalness: 0.15,
          roughness: 0.6,
        }),
      )
      mesh.matrixAutoUpdate = false
      mesh.matrix.copy(item.matrix)
      mesh.userData.link = item.link
      mesh.userData.THandLink = item.THandLink
      handMeshGroup.add(mesh)
    }
    clearOverlay()
    refreshHighlights()
    refreshHandPointMarkers()
    frameHandModel()
    infoMsg.value = mountDraftIds.value.size
      ? `${payload.label} 已恢复 ${mountDraftIds.value.size} 个模型点，请继续完成 16 点标注`
      : `${payload.label} 全零关节模型已加载，请先连续标注 16 个模型点`
  } catch (error) {
    if (serial === handLoadSerial) setError(error)
  } finally {
    if (serial === handLoadSerial) handBusy.value = false
  }
}

function activateMountSlot(pointId) {
  activeMountSlotId.value = pointId
  const draft = mountDrafts.value.find((item) => item.point_id === pointId)
  if (mountViewport.value === 'model') {
    infoMsg.value = draft?.p_hand
      ? `${draft.label} 模型点已选；再次点击 mesh 可修正`
      : `请在零位手模型上标注 ${mountSlotInfo(pointId).label}`
  } else if (!draft?.p_hand) {
    mountViewport.value = 'model'
    infoMsg.value = `${mountSlotInfo(pointId).label} 尚未标注模型点`
  } else if (draft.vertexIndex != null || mountSavedIds.value.has(pointId)) {
    infoMsg.value = `${draft.label} 点云点已完成；再次点击点云可覆盖`
  } else {
    infoMsg.value = `请在当前 episode 点云选择 ${draft.label}`
  }
  refreshHandPointMarkers()
}

function onHandPointerDown(event) {
  handPointerStart = { x: event.clientX, y: event.clientY }
}

function onHandPointerUp(event) {
  if (!handPointerStart || !handMeshGroup || !activeMountSlot.value) return
  const moved = Math.hypot(
    event.clientX - handPointerStart.x,
    event.clientY - handPointerStart.y,
  )
  handPointerStart = null
  if (moved > 5) return

  const rect = handRenderer.domElement.getBoundingClientRect()
  const mouse = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  )
  const raycaster = new THREE.Raycaster()
  raycaster.setFromCamera(mouse, handCamera)
  const hit = raycaster.intersectObjects(handMeshGroup.children, false)[0]
  if (!hit) {
    infoMsg.value = '没有命中手模型 mesh，请旋转或放大后重试'
    return
  }

  const slot = activeMountSlot.value
  const existing = activeMountDraft.value
  const pHand = hit.point.clone()
  const pLocal = pHand.clone().applyMatrix4(
    hit.object.userData.THandLink.clone().invert(),
  )
  mountDrafts.value = [
    ...mountDrafts.value.filter((item) => item.point_id !== slot.point_id),
    {
      ...slot,
      vertexIndex: existing?.vertexIndex,
      point: existing?.point,
      displayPoint: existing?.displayPoint,
      link: hit.object.userData.link,
      p_local: pLocal.toArray(),
      p_hand: pHand.toArray(),
      meshFaceIndex: hit.faceIndex,
    },
  ]
  mountProfileDirty.value = true
  const next = chooseNextMountModelSlot(slot.point_id)
  infoMsg.value = next
    ? `${slot.label} 模型点已选；下一项：${next.label}`
    : '16 个模型点已全部标注，请点击“开始当前 episode 点云配对”'
  refreshHighlights()
  refreshHandPointMarkers()
}

function removeMountDraft(pointId) {
  const draft = mountDrafts.value.find((item) => item.point_id === pointId)
  mountDrafts.value = [
    ...mountDrafts.value.filter((item) => item.point_id !== pointId),
    ...(draft?.p_hand ? [modelOnlyMountDraft(draft)] : []),
  ]
  activeMountSlotId.value = pointId
  mountViewport.value = 'cloud'
  refreshHighlights()
  refreshHandPointMarkers()
}

function clearMountDrafts() {
  mountDrafts.value = []
  mountProfileDirty.value = true
  activeMountSlotId.value = mountSlots[0].point_id
  mountViewport.value = 'model'
  refreshHighlights()
  refreshHandPointMarkers()
}

function clearMountCloudSelections() {
  keepOnlyMountModelPoints()
  refreshHighlights()
  refreshHandPointMarkers()
  infoMsg.value = '已撤销当前 episode 尚未保存的点云选择；16 个模型点仍保留'
}

async function loadHandsCatalog() {
  const response = await fetch('/api/hands')
  if (!response.ok) throw await responseError(response, '手型号目录加载失败')
  const data = await response.json()
  hands.value = data.hands || []
  if (!hands.value.some((item) => item.hand_id === selectedHandId.value)) {
    const savedHandId = mountSamples.value[0]?.hand_id
    selectedHandId.value = hands.value.find((item) => item.hand_id === savedHandId)?.hand_id
      || hands.value[0]?.hand_id
      || ''
  }
}

async function refreshMountSamples() {
  const response = await fetch('/api/mount/samples')
  if (!response.ok) throw await responseError(response, '安装样本加载失败')
  const data = await response.json()
  mountSamples.value = data.samples || []
  mountMinPoints.value = Number(data.min_points) || 3
  if (cloudObject) restoreSavedMountPoints(cloudObject.geometry)
  refreshHandPointMarkers()
}

async function saveMountSelections() {
  if (!canSaveMount.value) return
  mountSaveBusy.value = true
  errorMsg.value = ''
  infoMsg.value = ''
  try {
    const paired = mountDrafts.value.filter((item) => item.vertexIndex != null)
    const confirmResponse = await fetch('/api/offline/confirm-mount-points', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        episode: selectedEpisode.value,
        cloud_id: cloudId.value,
        stride: cloudStride.value,
        hand_id: selectedHandId.value,
        hand_joints: (handModel.value?.actuated_joints || []).map(() => 0),
        selections: paired.map((item) => ({
          point_id: item.point_id,
          label: item.label,
          link: item.link,
          p_local: item.p_local,
          p_hand: item.p_hand,
          vertex_index: item.vertexIndex,
        })),
      }),
    })
    if (!confirmResponse.ok) throw await responseError(confirmResponse, '安装选点确认失败')
    const confirmation = await confirmResponse.json()

    // mount API 不提供 replace_existing；对本 episode 同槽重选时先删除旧记录。
    const pairedIds = new Set(paired.map((item) => item.point_id))
    const replaced = mountSavedForEpisode.value.filter((item) => pairedIds.has(item.point_id))
    await Promise.all(replaced.map(async (item) => {
      const response = await fetch(`/api/mount/samples/${item.index}`, { method: 'DELETE' })
      if (!response.ok) throw await responseError(response, `替换 ${item.point_id} 失败`)
    }))

    const saveResponse = await fetch('/api/mount/samples/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ observations: confirmation.observations }),
    })
    if (!saveResponse.ok) throw await responseError(saveResponse, '安装样本保存失败')
    const saved = await saveResponse.json()
    keepOnlyMountModelPoints()
    mountResult.value = null
    overlayVisible.value = false
    clearOverlay()
    await refreshMountSamples()
    chooseNextMountCloudSlot()
    refreshHighlights()
    infoMsg.value = `已保存 ${saved.saved_count} 个安装配对，本会话共 ${saved.count} 条`
  } catch (error) {
    setError(error)
  } finally {
    mountSaveBusy.value = false
  }
}

async function deleteMountSample(index) {
  try {
    const response = await fetch(`/api/mount/samples/${index}`, { method: 'DELETE' })
    if (!response.ok) throw await responseError(response, '安装样本删除失败')
    mountResult.value = null
    overlayVisible.value = false
    clearOverlay()
    await refreshMountSamples()
    refreshHighlights()
    infoMsg.value = '安装样本已删除，可重新采集该槽'
  } catch (error) {
    setError(error)
  }
}

async function solveMount() {
  mountSolveBusy.value = true
  errorMsg.value = ''
  infoMsg.value = ''
  try {
    const response = await fetch('/api/mount/solve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
    if (!response.ok) throw await responseError(response, '手安装解算失败')
    mountResult.value = await response.json()
    infoMsg.value = '手安装解算完成'
    clearOverlay()
    if (overlayVisible.value) buildOverlay()
  } catch (error) {
    setError(error)
  } finally {
    mountSolveBusy.value = false
  }
}

function ensureOverlayGroup() {
  if (overlayGroup || !scene) return
  overlayGroup = new THREE.Group()
  overlayGroup.matrixAutoUpdate = false
  scene.add(overlayGroup)
}

function clearOverlay() {
  if (!overlayGroup) return
  clearGroup(overlayGroup)
  overlayGroup.visible = false
}

function buildOverlay() {
  if (!handMeshGroup?.children.length || !overlayAvailable.value) return
  ensureOverlayGroup()
  clearGroup(overlayGroup)
  for (const source of handMeshGroup.children) {
    const mesh = new THREE.Mesh(
      source.geometry,
      new THREE.MeshBasicMaterial({
        color: 0x38d996,
        transparent: true,
        opacity: 0.42,
        depthWrite: false,
      }),
    )
    mesh.matrixAutoUpdate = false
    mesh.matrix.copy(source.matrix)
    overlayGroup.add(mesh)
  }
  updateOverlay()
}

function updateOverlay() {
  if (!overlayGroup) return
  const rows = mountResult.value?.per_pose_overlay_T_camera_hand?.[selectedEpisode.value]
  overlayGroup.visible = Boolean(rows && overlayVisible.value)
  if (rows) overlayGroup.matrix.copy(matrixFromRows(rows))
}

function toggleOverlay() {
  overlayVisible.value = !overlayVisible.value
  if (overlayVisible.value && overlayAvailable.value && !overlayGroup?.children.length) {
    buildOverlay()
  }
  if (overlayVisible.value && !overlayAvailable.value) {
    infoMsg.value = '当前 episode 未参与本次解算，没有模型叠加位姿'
  }
  updateOverlay()
}

function formatMatrixValue(value) {
  return Number(value).toFixed(6)
}

async function setMode(nextMode) {
  if (mode.value === nextMode) return
  mode.value = nextMode
  if (nextMode === 'mount') mountViewport.value = 'model'
  refreshHighlights()
  if (nextMode !== 'mount') {
    if (overlayGroup) overlayGroup.visible = false
    return
  }
  await nextTick()
  initHandViewer()
  resizeHandViewer()
  try {
    if (!mountSamples.value.length) await refreshMountSamples()
    if (!hands.value.length) await loadHandsCatalog()
    if (selectedHandId.value && !handModel.value) await loadHandModel()
    await refreshMountProfiles(selectedHandId.value)
    restoreSavedMountPoints(cloudObject?.geometry)
    refreshHighlights()
  } catch (error) {
    setError(error)
  }
}

async function loadWorkspace() {
  errorMsg.value = ''
  infoMsg.value = ''
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
    if (!statusData.offline?.enabled) {
      throw new Error('后端未配置可读取的 episode 目录')
    }
    status.value = statusData
    episodes.value = episodeData.episodes || []
    markerColors.value = colorData.colors || []
    samples.value = sampleData.samples || []
    if (!episodes.value.some((item) => item.name === selectedEpisode.value)) {
      selectedEpisode.value = episodes.value[0]?.name || ''
    }
    activeColor.value = episodeSamples.value[0]?.color || selectableColors.value[0]?.color || ''
    if (!episodes.value.length) {
      infoMsg.value = '暂无 episode，请先在 7012 按 C 采集当前姿态，再点此页面的刷新按钮。'
    }
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
  refreshHandPointMarkers()
  updateOverlay()
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

watch(selectedHandId, async (value, oldValue) => {
  if (value && value !== oldValue && mode.value === 'mount' && handRenderer) {
    mountViewport.value = 'model'
    mountProfiles.value = []
    selectedMountProfileId.value = ''
    await loadHandModel()
    await refreshMountProfiles(value)
  }
})

watch(mountViewport, async () => {
  await nextTick()
  resizeViewer()
  resizeHandViewer()
})

onMounted(async () => {
  initViewer()
  await loadWorkspace()
  if (selectedEpisode.value) await loadPointCloud()
})

onBeforeUnmount(() => {
  requestSerial += 1
  handLoadSerial += 1
  resizeObserver?.disconnect()
  handResizeObserver?.disconnect()
  if (renderer) {
    renderer.setAnimationLoop(null)
    renderer.domElement.removeEventListener('pointerdown', onPointerDown)
    renderer.domElement.removeEventListener('pointerup', onPointerUp)
    renderer.dispose()
  }
  controls?.dispose()
  if (handRenderer) {
    handRenderer.setAnimationLoop(null)
    handRenderer.domElement.removeEventListener('pointerdown', onHandPointerDown)
    handRenderer.domElement.removeEventListener('pointerup', onHandPointerUp)
    handRenderer.dispose()
  }
  handControls?.dispose()
  clearGroup(handMeshGroup)
  clearGroup(handPointGroup, true)
  clearGroup(mountCandidateGroup, true)
  clearOverlay()
  for (const geometryPromise of stlCache.values()) {
    geometryPromise.then((geometry) => geometry.dispose()).catch(() => {})
  }
  disposeCloud()
})
</script>

<template>
  <main class="app-shell">
    <header class="topbar">
      <div>
        <h1>Hand-Eye 3D · 点云选点</h1>
        <p>已落盘 episode · 相机系 X 右 / Y 下 / Z 前 · 单位 m</p>
      </div>
      <div class="topbar-status">
        <span class="status-dot" :class="{ ready: status?.offline?.enabled }"></span>
        {{
          status?.offline?.enabled
            ? (status.mode === 'live' ? '实时采集 + episode 读取已连接' : '纯离线后端已连接')
            : '等待 episode 后端'
        }}
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
          <p v-if="!episodes.length" class="empty-state">
            暂无 episode。请先在 7012 按 C 采集当前姿态，再点击上方刷新。
          </p>
        </div>
      </aside>

      <section class="viewer-column">
        <div class="viewer-toolbar">
          <div class="viewer-title">
            <strong>
              {{
                mode === 'mount' && mountViewport === 'model'
                  ? (currentHand?.label || '零位灵巧手模型')
                  : (selectedEpisode || '未选择 episode')
              }}
            </strong>
            <span v-if="mode === 'mount' && mountViewport === 'model'">
              {{ handModel?.base_link || '等待模型' }} · 六个手关节全零
            </span>
            <span v-else-if="pointCount">{{ pointCount.toLocaleString() }} points</span>
          </div>
          <div v-if="mode === 'mount'" class="viewport-tabs">
            <button
              :class="{ active: mountViewport === 'model' }"
              @click="mountViewport = 'model'"
            >
              零位手模型
            </button>
            <button
              :class="{ active: mountViewport === 'cloud' }"
              :disabled="!allMountModelPointsSelected"
              title="标完 16 个模型点后进入点云配对"
              @click="beginMountCloudPairing"
            >
              实体点云
            </button>
          </div>
          <template v-if="mode === 'marker' || mountViewport === 'cloud'">
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
          </template>
        </div>
        <div
          v-show="mode === 'marker' || mountViewport === 'cloud'"
          ref="viewerHost"
          class="viewer"
        >
          <div v-if="cloudBusy" class="viewer-overlay">
            <span class="spinner"></span>
            正在对齐五帧深度并生成点云…
          </div>
          <div class="axis-legend">
            <span class="x">X 右</span>
            <span class="y">Y 下</span>
            <span class="z">Z 前</span>
          </div>
          <div class="viewer-help">
            {{
              mode === 'mount'
                ? `当前 ${activeMountSlot?.label || '未选槽位'} · 单击对应实体点`
                : '单击选点'
            }}
            · 左键拖动旋转 · 右键平移 · 滚轮缩放
          </div>
        </div>
        <div
          v-show="mode === 'mount' && mountViewport === 'model'"
          ref="handViewerHost"
          class="viewer hand-main-viewer"
        >
          <div v-if="handBusy" class="viewer-overlay">
            <span class="spinner"></span>
            正在加载手模型…
          </div>
          <div class="axis-legend">
            <span class="x">X</span>
            <span class="y">Y</span>
            <span class="z">Z</span>
          </div>
          <div class="viewer-help">
            当前 {{ activeMountSlot?.label }} · 单击模型贴点位置 · 左键旋转 · 右键平移 · 滚轮缩放
          </div>
        </div>
        <div v-if="errorMsg" class="message error">{{ errorMsg }}</div>
        <div v-else-if="infoMsg" class="message success">{{ infoMsg }}</div>
      </section>

      <aside class="selection-panel">
        <div class="mode-tabs">
          <button :class="{ active: mode === 'marker' }" @click="setMode('marker')">
            Marker 标定
          </button>
          <button :class="{ active: mode === 'mount' }" @click="setMode('mount')">
            手安装标定
          </button>
        </div>

        <template v-if="mode === 'mount'">
          <section class="side-card">
            <div class="panel-heading compact">
              <div>
                <h2>1. 选择零位手模型</h2>
                <span>模型显示在中央大视区 · mesh 表面可点击</span>
              </div>
              <span class="zero-badge">6 DOF = 0</span>
            </div>
            <select v-model="selectedHandId" class="hand-select" :disabled="handBusy">
              <option v-for="hand in hands" :key="hand.hand_id" :value="hand.hand_id">
                {{ hand.label }}（{{ hand.side === 'left' ? '左手' : '右手' }}）
              </option>
            </select>
            <p class="model-meta">
              {{ currentHand?.vendor || '—' }} · {{ handModel?.base_link || '等待模型' }}
            </p>
            <button class="secondary-button model-focus-button" @click="mountViewport = 'model'">
              在中央查看并选择模型点
            </button>
          </section>

          <section class="side-card mount-profile-card">
            <div class="panel-heading compact">
              <div>
                <h2>2. 模型点方案</h2>
                <span>标注任意数量即可保存草稿 · 重启后继续</span>
              </div>
              <span v-if="mountProfileDirty" class="profile-dirty">未保存修改</span>
            </div>
            <div class="profile-save-row">
              <input
                v-model="mountProfileName"
                maxlength="128"
                placeholder="输入方案名称"
                :disabled="mountProfileBusy"
              />
              <button
                class="primary-button"
                :disabled="!canSaveMountProfile"
                @click="saveMountProfile"
              >
                {{ mountProfileBusy ? '处理中…' : '保存/覆盖' }}
              </button>
            </div>
            <div class="profile-load-row">
              <select
                v-model="selectedMountProfileId"
                :disabled="mountProfileBusy || !mountProfiles.length"
                @change="selectMountProfile"
              >
                <option value="">选择已有方案</option>
                <option
                  v-for="profile in mountProfiles"
                  :key="profile.profile_id"
                  :value="profile.profile_id"
                >
                  {{ profile.name }}（{{ profile.point_count ?? profile.points?.length ?? 0 }}/16）
                </option>
              </select>
              <button
                class="secondary-button"
                :disabled="!selectedMountProfileId || mountProfileBusy"
                @click="loadMountProfile"
              >
                加载
              </button>
              <button
                class="text-button"
                :disabled="!selectedMountProfileId || mountProfileBusy"
                @click="deleteMountProfile"
              >
                删除
              </button>
            </div>
            <p class="model-meta">
              {{
                loadedMountProfileId
                  ? `当前已加载：${mountProfileName}`
                  : `${mountProfiles.length} 个可用方案`
              }}
            </p>
          </section>

          <section class="side-card mount-slots-card">
            <div class="panel-heading compact">
              <div>
                <h2>3. 先标注手模型的 16 个点</h2>
                <span>
                  已完成 {{ mountDraftIds.size }}/16 · 全部完成后再进入点云
                </span>
              </div>
            </div>
            <div class="mount-slot-section">
              <strong>手心</strong>
              <div class="mount-slot-grid">
                <button
                  v-for="slot in mountSlots.slice(0, 8)"
                  :key="slot.point_id"
                  class="mount-slot"
                  :class="{
                    active: activeMountSlotId === slot.point_id,
                    modeled: mountDraftIds.has(slot.point_id),
                    paired: mountPairedIds.has(slot.point_id),
                    saved: mountSavedIds.has(slot.point_id),
                  }"
                  :title="`${slot.label}（${slot.point_id}）`"
                  @click="activateMountSlot(slot.point_id)"
                >
                  <i :style="{ background: slot.color }"></i>
                  <span>{{ slot.shortLabel }}</span>
                  <small v-if="mountPairedIds.has(slot.point_id)">待保存</small>
                  <small v-else-if="mountSavedIds.has(slot.point_id)">已保存</small>
                  <small v-else-if="mountDraftIds.has(slot.point_id)">
                    {{ mountViewport === 'cloud' ? '点云待选' : '模型已选' }}
                  </small>
                  <small v-else>模型待选</small>
                </button>
              </div>
            </div>
            <div class="mount-slot-section">
              <strong>手背</strong>
              <div class="mount-slot-grid">
                <button
                  v-for="slot in mountSlots.slice(8)"
                  :key="slot.point_id"
                  class="mount-slot"
                  :class="{
                    active: activeMountSlotId === slot.point_id,
                    modeled: mountDraftIds.has(slot.point_id),
                    paired: mountPairedIds.has(slot.point_id),
                    saved: mountSavedIds.has(slot.point_id),
                  }"
                  :title="`${slot.label}（${slot.point_id}）`"
                  @click="activateMountSlot(slot.point_id)"
                >
                  <i :style="{ background: slot.color }"></i>
                  <span>{{ slot.shortLabel }}</span>
                  <small v-if="mountPairedIds.has(slot.point_id)">待保存</small>
                  <small v-else-if="mountSavedIds.has(slot.point_id)">已保存</small>
                  <small v-else-if="mountDraftIds.has(slot.point_id)">
                    {{ mountViewport === 'cloud' ? '点云待选' : '模型已选' }}
                  </small>
                  <small v-else>模型待选</small>
                </button>
              </div>
            </div>
            <p class="armed-hint">
              当前：<strong>{{ activeMountSlot?.label }}</strong>
              <template v-if="mountViewport === 'model'">
                → 请在手模型上点击此点
              </template>
              <template v-else>→ 请在当前 episode 点云点击对应实体点</template>
            </p>
            <div class="mount-stage-actions">
              <button
                class="primary-button"
                :disabled="!allMountModelPointsSelected"
                @click="beginMountCloudPairing"
              >
                开始当前 episode 点云配对
              </button>
              <button
                class="text-button"
                :disabled="!mountDraftIds.size"
                @click="clearMountDrafts"
              >
                清空模型点
              </button>
            </div>
          </section>

          <section class="side-card grow">
            <div class="panel-heading compact">
              <div>
                <h2>4. 当前 episode 点云配对</h2>
                <span>
                  {{ mountPairedIds.size }} 对待保存 · {{ mountSavedForEpisode.length }} 对已保存
                </span>
              </div>
              <button
                class="text-button"
                :disabled="!mountPairedDrafts.length"
                @click="clearMountCloudSelections"
              >
                撤销点云选择
              </button>
            </div>
            <div class="mount-candidate-tools">
              <button
                class="secondary-button"
                :disabled="!allMountModelPointsSelected || !cloudId || mountCandidateBusy"
                @click="detectMountCandidates"
              >
                {{ mountCandidateBusy ? 'RGB识别中…' : '识别RGB红绿圆' }}
              </button>
              <span>
                候选：红 {{ mountCandidateCounts.red }} · 绿 {{ mountCandidateCounts.green }}
              </span>
            </div>
            <p v-if="mountCandidateWarnings.length" class="candidate-warning">
              {{ mountCandidateWarnings[0] }}
            </p>
            <div class="selection-list mount-selection-list">
              <article
                v-for="item in mountPairedDrafts"
                :key="item.point_id"
                class="selection-row"
              >
                <i :style="{ background: item.color }"></i>
                <div>
                  <strong>{{ item.label }}</strong>
                  <code v-if="item.point">
                    cloud {{ item.point.map((value) => Number(value).toFixed(4)).join(', ') }}
                  </code>
                  <code>model {{ item.p_hand.map((value) => Number(value).toFixed(4)).join(', ') }}</code>
                  <small>
                    {{ item.link }} ·
                    {{ item.vertexIndex == null ? '等待点云点' : `cloud vertex #${item.vertexIndex}` }}
                  </small>
                </div>
                <button title="撤销此槽" @click="removeMountDraft(item.point_id)">×</button>
              </article>
              <p v-if="!mountPairedDrafts.length" class="empty-state">
                RGB识别会在点云中显示半透明红/绿候选球。按照当前槽位顺序直接点击
                同色候选；仍可点击普通点云手动修正。单个 episode 不要求看见全部16点。
              </p>
            </div>
            <button class="primary-button" :disabled="!canSaveMount" @click="saveMountSelections">
              {{ mountSaveBusy ? '确认并保存中…' : '确认并保存本姿态配对' }}
            </button>
          </section>

          <section class="side-card mount-samples-card">
            <div class="panel-heading compact">
              <div>
                <h2>5. 安装样本与解算</h2>
                <span>{{ mountSamples.length }} 条样本 · {{ mountSamplesByPose }} 个姿态</span>
              </div>
            </div>
            <div v-if="mountSavedForEpisode.length" class="mount-sample-list">
              <div
                v-for="sample in mountSavedForEpisode"
                :key="sample.index"
                class="mount-sample-row"
              >
                <i :style="{ background: mountSlotInfo(sample.point_id).color }"></i>
                <span>{{ sample.label || mountSlotInfo(sample.point_id).label }}</span>
                <small>#{{ sample.index }}</small>
                <button
                  class="mini-delete"
                  :title="`删除 ${sample.point_id}`"
                  @click="deleteMountSample(sample.index)"
                >×</button>
              </div>
            </div>
            <p v-else class="episode-sample-hint">当前 episode 尚无已保存安装样本</p>
            <button
              class="solve-button"
              :disabled="mountSolveBusy || mountSamples.length < mountMinPoints"
              @click="solveMount"
            >
              {{ mountSolveBusy ? '解算中…' : `解算 T_wrist2hand（至少 ${mountMinPoints} 点）` }}
            </button>
            <div v-if="mountResidualSummary" class="result-summary">
              <div><span>RMS</span><strong>{{ mountResidualSummary.rms }} mm</strong></div>
              <div><span>Median</span><strong>{{ mountResidualSummary.median }} mm</strong></div>
              <div><span>Max</span><strong>{{ mountResidualSummary.max }} mm</strong></div>
            </div>
            <button
              v-if="mountResult"
              class="secondary-button overlay-toggle"
              :disabled="!overlayAvailable"
              @click="toggleOverlay"
            >
              {{ overlayVisible ? '隐藏' : '显示' }}当前姿态模型叠加
            </button>
          </section>

          <section v-if="mountResult" class="side-card mount-result-card">
            <h2>解算输出</h2>
            <div class="matrix-block">
              <span>T_wrist2hand</span>
              <code
                v-for="(row, rowIndex) in mountResult.T_wrist2hand"
                :key="rowIndex"
              >{{ row.map(formatMatrixValue).join('  ') }}</code>
            </div>
            <div class="result-paths">
              <span>输出</span>
              <code>{{ mountResult.saved_to }}</code>
              <code v-if="mountResult.merged_calib">{{ mountResult.merged_calib }}</code>
            </div>
            <div class="tcp-list">
              <span>tcp_points（腕系，m）</span>
              <div
                v-for="tcp in mountResult.tcp_points_wrist_m"
                :key="tcp.id"
              >
                <strong>{{ tcp.label }}</strong>
                <code>{{ tcp.p_wrist_m.map((value) => Number(value).toFixed(6)).join(', ') }}</code>
              </div>
            </div>
          </section>
        </template>

        <template v-else>
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
        </template>
      </aside>
    </section>
  </main>
</template>
