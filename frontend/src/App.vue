<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import MarkerOverlay from './components/MarkerOverlay.vue'

const status = ref(null)
const samples = ref([])
const result = ref(null)
const errorMsg = ref('')
const recordBusy = ref(false)
const recordMessage = ref('')

// 当前待配对的一组数据
const pick = ref(null)        // {p_camera, depth_mm, pixel, valid_ratio}
const pickBusy = ref(false)
const wristT = ref(null)      // 自动读取到的 4x4
const wristManual = ref({ x: '', y: '', z: '', roll: '', pitch: '', yaw: '' })
const clickPos = ref(null)

const imgEl = ref(null)
const offlineEpisodes = ref([])
const selectedEpisode = ref('')
const episodesBusy = ref(false)
const previewVersion = ref(0)
const showDepthOverlay = ref(false)
const depthOverlayOpacity = ref(45)
const markerColors = ref([])
const detectedMarkers = ref([])
const markerImageSize = ref(null)
const markerWarnings = ref([])
const missingColors = ref([])
const markerBusy = ref(false)
const markerConfirmBusy = ref(false)
const confirmElapsedSec = ref(0)
const markerBatchBusy = ref(false)
const selectedMarkerKey = ref(null)
const addMarkerColorId = ref('')
const markerConfirmation = ref(null)
const markerBatchSaved = ref(false)
const imageNaturalSize = ref(null)
const liveFrameVersion = ref(Date.now())
let markerKeyCounter = 0
let detectionRequestId = 0
let liveFrameTimer = null

const offlineMode = computed(() => status.value?.mode === 'offline')
const selectedEpisodeInfo = computed(() =>
  offlineEpisodes.value.find((episode) => episode.name === selectedEpisode.value) || null,
)
const imageSrc = computed(() => {
  // status 还没回来时绝不能拉实时帧，否则离线页会短暂请求错误的数据源。
  if (!status.value) return ''
  if (!offlineMode.value) return `/api/frame.jpg?v=${liveFrameVersion.value}`
  if (!selectedEpisode.value) return ''
  return `/api/offline/episodes/${encodeURIComponent(selectedEpisode.value)}/preview?v=${previewVersion.value}`
})
const depthOverlaySrc = computed(() => {
  if (!status.value) return ''
  if (!offlineMode.value) return `/api/depth-overlay.png?v=${liveFrameVersion.value}`
  if (!selectedEpisode.value) return ''
  return `/api/offline/episodes/${encodeURIComponent(selectedEpisode.value)}/depth-overlay?v=${previewVersion.value}`
})
const overlayImageSize = computed(() => markerImageSize.value || imageNaturalSize.value)

function markerColorId(marker) {
  const byColor = markerColors.value.find(
    (color) => String(color.id) === String(marker.color),
  )
  if (byColor) return byColor.id
  const byId = markerColors.value.find((color) => String(color.id) === String(marker.id))
  return byId?.id ?? marker.color ?? marker.id
}

function episodeNameOfSample(sample) {
  return sample.episode || sample.pose_id || sample.provenance?.episode
}

function samplesForEpisode(name) {
  return samples.value.filter((sample) => episodeNameOfSample(sample) === name)
}

function isSavedMarker(marker) {
  return marker.source === 'saved'
    || (Array.isArray(marker.flags) && marker.flags.includes('already_saved'))
}

function sampleToMarker(sample) {
  const center = sample.center || sample.pixel || [0, 0]
  return normalizeMarker({
    id: sample.marker_id || sample.id,
    color: sample.color,
    center: [Number(center[0]), Number(center[1])],
    radius_px: Number(sample.radius_px) || 12,
    source: 'saved',
    confidence: 1,
    color_confidence: 1,
    circularity: 1,
    flags: ['already_saved'],
    saved_index: sample.index,
  })
}

function mergeDetectedWithSaved(episode, detected) {
  const saved = samplesForEpisode(episode).map(sampleToMarker)
  const savedColors = new Set(saved.map((marker) => String(markerColorId(marker))))
  const extras = detected.filter((marker) => !savedColors.has(String(markerColorId(marker))))
  return [...saved, ...extras]
}

const duplicateColorIds = computed(() => {
  const counts = new Map()
  for (const marker of detectedMarkers.value) {
    const id = String(markerColorId(marker))
    counts.set(id, (counts.get(id) || 0) + 1)
  }
  return [...counts.entries()].filter(([, count]) => count > 1).map(([id]) => id)
})
const confirmedObservations = computed(() => markerConfirmation.value?.observations || [])
const unsavedMarkers = computed(() => detectedMarkers.value.filter((marker) => !isSavedMarker(marker)))
const savedOverlayMarkers = computed(() => detectedMarkers.value.filter(isSavedMarker))
const markerConfirmSucceeded = computed(() =>
  markerConfirmation.value?.ok
  && !(markerConfirmation.value.errors || []).length
  && confirmedObservations.value.length === unsavedMarkers.value.length
  && unsavedMarkers.value.length > 0,
)
const canConfirmMarkers = computed(() =>
  unsavedMarkers.value.length > 0
  && !duplicateColorIds.value.length
  && !markerBusy.value
  && !markerConfirmBusy.value,
)

function colorLabel(id) {
  return markerColors.value.find((color) => String(color.id) === String(id))?.label || id || '—'
}

function resetMarkerWorkflow() {
  detectedMarkers.value = []
  markerImageSize.value = null
  markerWarnings.value = []
  missingColors.value = []
  selectedMarkerKey.value = null
  addMarkerColorId.value = ''
  markerConfirmation.value = null
  markerBatchSaved.value = false
}

function invalidateMarkerConfirmation() {
  markerConfirmation.value = null
  markerBatchSaved.value = false
}

function normalizeMarker(marker) {
  return {
    ...marker,
    center: [...marker.center],
    flags: Array.isArray(marker.flags) ? [...marker.flags] : marker.flags,
    _key: `marker-${++markerKeyCounter}`,
  }
}

function markerPayload(marker) {
  const { _key, ...payload } = marker
  return {
    ...payload,
    center: payload.center.map(Number),
    radius_px: Number(payload.radius_px),
  }
}

async function loadMarkerColors() {
  try {
    const res = await fetch('/api/markers/colors')
    const data = await res.json()
    if (!res.ok) throw new Error(data.error || '颜色配置加载失败')
    markerColors.value = (data.colors || []).map((color) => ({
      ...color,
      id: color.id ?? color.color,
      label: color.label ?? color.label_zh ?? color.color ?? color.id,
    }))
  } catch (e) {
    errorMsg.value = e instanceof Error ? e.message : String(e)
  }
}

async function detectMarkers(episode = selectedEpisode.value) {
  if (!offlineMode.value || !episode) return
  const requestId = ++detectionRequestId
  markerBusy.value = true
  errorMsg.value = ''
  resetMarkerWorkflow()
  try {
    await refreshSamples()
    const res = await fetch('/api/offline/detect-markers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ episode }),
    })
    const data = await res.json()
    if (requestId !== detectionRequestId || episode !== selectedEpisode.value) return
    if (!res.ok || !data.ok) throw new Error(data.error || '自动检测失败')
    const detected = (data.markers || data.candidates || []).map(normalizeMarker)
    detectedMarkers.value = mergeDetectedWithSaved(episode, detected)
    markerImageSize.value = data.image_size || null
    markerWarnings.value = data.warnings || []
    const savedColors = new Set(
      samplesForEpisode(episode).map((sample) => String(sample.color)),
    )
    missingColors.value = (data.missing_colors || []).filter(
      (color) => !savedColors.has(String(color)),
    )
    selectedMarkerKey.value = detectedMarkers.value[0]?._key || null
  } catch (e) {
    if (requestId === detectionRequestId) {
      errorMsg.value = e instanceof Error ? e.message : String(e)
    }
  } finally {
    if (requestId === detectionRequestId) markerBusy.value = false
  }
}

async function refreshStatus() {
  status.value = await (await fetch('/api/status')).json()
}

async function recordEpisode() {
  if (!status.value?.recording?.enabled || recordBusy.value) return
  recordBusy.value = true
  recordMessage.value = ''
  errorMsg.value = ''
  try {
    const res = await fetch('/api/record/episode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame_count: 5 }),
    })
    const data = await res.json()
    if (!res.ok || !data.ok) throw new Error(data.error || '离线数据拍摄失败')
    recordMessage.value = `${data.episode}：已保存 ${data.frame_count} 帧`
    await refreshStatus()
  } catch (e) {
    errorMsg.value = e instanceof Error ? e.message : String(e)
  } finally {
    recordBusy.value = false
  }
}

async function refreshSamples() {
  const data = await (await fetch('/api/samples')).json()
  samples.value = data.samples
}

async function refreshOfflineEpisodes() {
  if (!offlineMode.value) {
    offlineEpisodes.value = []
    selectedEpisode.value = ''
    resetMarkerWorkflow()
    return
  }
  episodesBusy.value = true
  try {
    const res = await fetch('/api/offline/episodes')
    const data = await res.json()
    if (!res.ok) throw new Error(data.error || '离线剧集加载失败')
    offlineEpisodes.value = data.episodes || []
    let selectionChanged = false
    if (!offlineEpisodes.value.some((episode) => episode.name === selectedEpisode.value)) {
      selectedEpisode.value = offlineEpisodes.value[0]?.name || ''
      selectionChanged = true
      imageNaturalSize.value = null
      pick.value = null
      wristT.value = null
      clickPos.value = null
    }
    if (selectionChanged && selectedEpisode.value) await detectMarkers(selectedEpisode.value)
  } catch (e) {
    errorMsg.value = e instanceof Error ? e.message : String(e)
  } finally {
    episodesBusy.value = false
  }
}

async function selectEpisode(episode) {
  if (markerBusy.value || episode.name === selectedEpisode.value) return
  selectedEpisode.value = episode.name
  imageNaturalSize.value = null
  pick.value = null
  wristT.value = null
  clickPos.value = null
  previewVersion.value += 1
  await detectMarkers(episode.name)
}

function episodeImportedCount(episode) {
  return episode.imported_marker_count ?? episode.imported_marker_ids?.length ?? 0
}

function episodeSampleCount(episode) {
  if (episode.sample_count != null) return episode.sample_count
  if (episode.observation_count != null) return episode.observation_count
  return samples.value.filter((sample) =>
    (sample.episode || sample.provenance?.episode) === episode.name,
  ).length
}

function onOfflineCanvasClick(center) {
  if (markerBusy.value) return
  if (!addMarkerColorId.value) {
    selectedMarkerKey.value = null
    return
  }
  const color = markerColors.value.find(
    (option) => String(option.id) === String(addMarkerColorId.value),
  )
  if (!color) return
  const radii = detectedMarkers.value
    .map((marker) => Number(marker.radius_px))
    .filter((radius) => Number.isFinite(radius) && radius > 0)
    .sort((a, b) => a - b)
  const radius = radii.length ? radii[Math.floor(radii.length / 2)] : 12
  const marker = normalizeMarker({
    id: `manual-${String(color.id)}-${markerKeyCounter + 1}`,
    color: color.id,
    center,
    radius_px: radius,
    confidence: 1,
    color_confidence: 1,
    circularity: 1,
    source: 'manual',
    flags: ['manual_added'],
  })
  detectedMarkers.value.push(marker)
  selectedMarkerKey.value = marker._key
  addMarkerColorId.value = ''
  invalidateMarkerConfirmation()
}

function moveMarker({ key, center }) {
  const marker = detectedMarkers.value.find((item) => item._key === key)
  if (!marker) return
  marker.center = center
  invalidateMarkerConfirmation()
}

function changeMarkerColor(marker, colorId) {
  if (isSavedMarker(marker)) return
  const color = markerColors.value.find((option) => String(option.id) === String(colorId))
  if (!color) return
  marker.color = color.id
  invalidateMarkerConfirmation()
}

function deleteMarker(marker) {
  detectedMarkers.value = detectedMarkers.value.filter((item) => item._key !== marker._key)
  if (selectedMarkerKey.value === marker._key) selectedMarkerKey.value = null
  invalidateMarkerConfirmation()
}

function confirmationForMarker(marker, index) {
  const sameMarker = (entry) => {
    if (!entry || typeof entry !== 'object') return false
    if (entry.marker_index === index || entry.index === index) return true
    const ids = [entry.marker_id, entry.id, entry.marker_color, entry.color]
      .filter((value) => value != null)
      .map(String)
    return ids.includes(String(marker.id)) || ids.includes(String(marker.color))
  }
  const observation = confirmedObservations.value.find(sameMarker)
  const error = (markerConfirmation.value?.errors || []).find(sameMarker)
  return { observation, error }
}

function markerErrorText(error) {
  if (!error) return ''
  if (typeof error === 'string') return error
  return error.error || error.message || error.reason || '深度失败'
}

async function confirmMarkers() {
  if (!canConfirmMarkers.value) return
  markerConfirmBusy.value = true
  confirmElapsedSec.value = 0
  markerConfirmation.value = null
  markerBatchSaved.value = false
  errorMsg.value = ''
  const started = Date.now()
  const abort = new AbortController()
  const tick = setInterval(() => {
    confirmElapsedSec.value = Math.round((Date.now() - started) / 1000)
  }, 200)
  const timeout = setTimeout(() => abort.abort(), 20000)
  try {
    const res = await fetch('/api/offline/confirm-markers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: abort.signal,
      body: JSON.stringify({
        episode: selectedEpisode.value,
        markers: unsavedMarkers.value.map(markerPayload),
      }),
    })
    const data = await res.json()
    markerConfirmation.value = data
    if (!res.ok || !data.ok) throw new Error(data.error || '标记确认失败')
    if ((data.errors || []).length) errorMsg.value = '部分标记深度失败，请调整后重新确认'
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') {
      errorMsg.value = '确认超时：页面到后端的连接被堵住了。请强制刷新后再点一次。'
    } else {
      errorMsg.value = e instanceof Error ? e.message : String(e)
    }
  } finally {
    clearInterval(tick)
    clearTimeout(timeout)
    markerConfirmBusy.value = false
  }
}

async function saveMarkerBatch() {
  if (!markerConfirmSucceeded.value || markerBatchBusy.value) return
  markerBatchBusy.value = true
  errorMsg.value = ''
  try {
    const res = await fetch('/api/samples/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        episode: selectedEpisode.value,
        observations: confirmedObservations.value,
      }),
    })
    const data = await res.json()
    if (!res.ok || !data.ok) throw new Error(data.error || '批量保存失败')
    markerBatchSaved.value = true
    result.value = null
    await refreshSamples()
    await refreshOfflineEpisodes()
    if (selectedEpisode.value) {
      detectedMarkers.value = mergeDetectedWithSaved(selectedEpisode.value, [])
      selectedMarkerKey.value = detectedMarkers.value[0]?._key || null
    }
  } catch (e) {
    errorMsg.value = e instanceof Error ? e.message : String(e)
  } finally {
    markerBatchBusy.value = false
  }
}

// ---- 视频点击 → 反投影 ----

function onImageLoad(event) {
  const image = event.currentTarget
  if (errorMsg.value === '预览图加载失败，正在重试…') errorMsg.value = ''
  if (image.naturalWidth && image.naturalHeight) {
    imageNaturalSize.value = [image.naturalWidth, image.naturalHeight]
  }
  if (!offlineMode.value) {
    clearTimeout(liveFrameTimer)
    liveFrameTimer = setTimeout(() => {
      liveFrameVersion.value = Date.now()
    }, 160)
  }
}

function onImageError() {
  errorMsg.value = '预览图加载失败，正在重试…'
  if (!offlineMode.value) {
    clearTimeout(liveFrameTimer)
    liveFrameTimer = setTimeout(() => {
      liveFrameVersion.value = Date.now()
    }, 500)
  }
}

function imagePixelFromEvent(ev, img) {
  const rect = img.getBoundingClientRect()
  const width = img.naturalWidth || status.value?.camera?.width
  const height = img.naturalHeight || status.value?.camera?.height
  if (!width || !height || !rect.width || !rect.height) return null

  // 兼容固定尺寸容器中的 object-fit: contain，排除黑边并按原图尺寸映射。
  const scale = Math.min(rect.width / width, rect.height / height)
  const renderedWidth = width * scale
  const renderedHeight = height * scale
  const offsetX = (rect.width - renderedWidth) / 2
  const offsetY = (rect.height - renderedHeight) / 2
  const x = ev.clientX - rect.left - offsetX
  const y = ev.clientY - rect.top - offsetY
  if (x < 0 || y < 0 || x >= renderedWidth || y >= renderedHeight) return null

  return {
    u: Math.min(width - 1, Math.max(0, Math.floor(x / renderedWidth * width))),
    v: Math.min(height - 1, Math.max(0, Math.floor(y / renderedHeight * height))),
    xPct: (ev.clientX - rect.left) / rect.width * 100,
    yPct: (ev.clientY - rect.top) / rect.height * 100,
  }
}

async function onVideoClick(ev) {
  const img = imgEl.value
  if (!img || offlineMode.value || pickBusy.value) return
  const point = imagePixelFromEvent(ev, img)
  if (!point) {
    errorMsg.value = '请点击图像内容区域（不要点击留白）'
    return
  }
  const { u, v, xPct, yPct } = point
  clickPos.value = { xPct, yPct }

  pickBusy.value = true
  errorMsg.value = ''
  pick.value = null
  try {
    const res = await fetch('/api/pick', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ u, v }),
    })
    const data = await res.json()
    if (data.ok) {
      pick.value = data
      // 自动位姿源：点击取点的同时立刻抓一次手腕位姿，保证时间对齐
      if (autoPose.value) await readWristPose()
    } else {
      errorMsg.value = data.error || '取点失败'
    }
  } catch (e) {
    errorMsg.value = String(e)
  } finally {
    pickBusy.value = false
  }
}

// ---- 手腕位姿 ----

const autoPose = computed(() => status.value?.pose_auto)

async function readWristPose() {
  errorMsg.value = ''
  try {
    const res = await fetch('/api/wrist_pose')
    const data = await res.json()
    if (data.ok) {
      wristT.value = data.T_base_wrist
    } else {
      errorMsg.value = data.error
    }
  } catch (e) {
    errorMsg.value = String(e)
  }
}

const manualValid = computed(() =>
  Object.values(wristManual.value).every((s) => s !== '' && isFinite(Number(s))),
)

const wristReady = computed(() =>
  autoPose.value ? !!wristT.value : manualValid.value,
)
const canSave = computed(() => !offlineMode.value && pick.value && wristReady.value)

// ---- 保存样本 ----

async function saveSample() {
  if (!canSave.value) return
  errorMsg.value = ''
  const body = { p_camera: pick.value.p_camera, pixel: pick.value.pixel }
  if ((offlineMode.value || autoPose.value) && wristT.value) {
    body.T_base_wrist = wristT.value
  } else {
    body.wrist_xyz = [Number(wristManual.value.x), Number(wristManual.value.y), Number(wristManual.value.z)]
    body.wrist_rpy = [Number(wristManual.value.roll), Number(wristManual.value.pitch), Number(wristManual.value.yaw)]
  }
  if (offlineMode.value) {
    const episode = pick.value.episode || selectedEpisode.value
    body.episode = episode
    body.provenance = pick.value.provenance || {
      mode: 'offline',
      episode,
      teleop_task_dir: status.value?.teleop_task_dir,
      camera_serial: pick.value.camera_serial || selectedEpisodeInfo.value?.camera_serial,
      preview_frame_idx: selectedEpisodeInfo.value?.preview_frame_idx,
      burst_frames_used: pick.value.burst_frames_used,
      qpos_median_rad: pick.value.qpos_median_rad,
      depth_mm: pick.value.depth_mm,
    }
  }
  const res = await fetch('/api/samples', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = await res.json()
  if (!data.ok) {
    errorMsg.value = data.error || '保存失败'
    return
  }
  pick.value = null
  clickPos.value = null
  wristT.value = null
  result.value = null
  await refreshSamples()
  await refreshOfflineEpisodes()
}

async function deleteSample(index) {
  await fetch(`/api/samples/${index}`, { method: 'DELETE' })
  result.value = null
  await refreshSamples()
  await refreshOfflineEpisodes()
}

// ---- 解算 ----

const solveBusy = ref(false)
const minSamples = computed(() => status.value?.min_samples ?? 5)

async function solve() {
  solveBusy.value = true
  errorMsg.value = ''
  try {
    const res = await fetch('/api/solve', { method: 'POST' })
    const data = await res.json()
    if (data.ok) {
      result.value = data
    } else {
      errorMsg.value = data.error || '解算失败'
    }
  } finally {
    solveBusy.value = false
  }
}

// 固定相机外参、只解 p_tool（点击指尖尖端的样本）
const toolResult = ref(null)
const toolBusy = ref(false)

async function solveToolOnly() {
  toolBusy.value = true
  errorMsg.value = ''
  try {
    const res = await fetch('/api/solve_tool', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    })
    const data = await res.json()
    if (data.ok) toolResult.value = data
    else errorMsg.value = data.error || '解算失败'
  } finally {
    toolBusy.value = false
  }
}

const matrixText = computed(() => {
  if (!result.value?.T_cam2base) return ''
  return result.value.T_cam2base
    .map((r) => r.map((v) => v.toFixed(5).padStart(9)).join('  '))
    .join('\n')
})
const isMultiResult = computed(() =>
  ['multi_tool_offset_joint', 'multi_marker_tool_offset_joint'].includes(result.value?.mode),
)
const multiMarkerRows = computed(() => {
  if (!isMultiResult.value) return []
  const offsets = result.value.p_tool_wrist_m_by_marker || {}
  const residuals = result.value.residual_mm?.per_marker
    || result.value.per_marker_residual_stats_mm
    || result.value.residual_by_marker_mm
    || {}
  const ids = new Set([...Object.keys(offsets), ...Object.keys(residuals)])
  return [...ids].map((id) => {
    const offsetValue = offsets[id]
    const residualValue = residuals[id]
    const offset = Array.isArray(offsetValue)
      ? offsetValue
      : offsetValue?.p_tool_wrist_m || offsetValue?.offset_m || []
    const residual = typeof residualValue === 'number'
      ? { rms: residualValue }
      : residualValue || {}
    return { id, offset, residual }
  })
})

function fmt(v, d = 4) { return Number(v).toFixed(d) }
function wristSummary(T) {
  return `[${T[0][3].toFixed(3)}, ${T[1][3].toFixed(3)}, ${T[2][3].toFixed(3)}]`
}
function sampleMarkerLabel(sample) {
  const id = sample.marker_color ?? sample.color ?? sample.marker_id
  return id != null ? colorLabel(id) : '—'
}

// ---- 手臂点动（--arm-control 时后端才有） ----

const arm = ref(null)             // /api/arm/status 的返回
const armBusy = ref(false)
const stepDeg = ref(2)            // 步长（度）
const stepOptions = [0.5, 1, 2, 5, 10]

async function refreshArm() {
  try {
    arm.value = await (await fetch('/api/arm/status')).json()
  } catch { /* 后端未起或断连，下轮再试 */ }
}

async function armPost(path, body) {
  armBusy.value = true
  errorMsg.value = ''
  try {
    const res = await fetch(`/api/arm/${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    })
    const data = await res.json()
    if (!data.ok) errorMsg.value = data.error || `${path} 失败`
    else arm.value = { enabled: true, ...data }
  } catch (e) {
    errorMsg.value = String(e)
  } finally {
    armBusy.value = false
  }
}

function nudgeJoint(index, sign) {
  const delta = sign * stepDeg.value * Math.PI / 180
  armPost('nudge', { index, delta })
}

function handMove() {
  if (!confirm('卸力拖动：重力前馈会让手臂近似失重（推到哪停哪），但补偿有偏差时仍可能缓慢飘移，请用手护住手臂。确认进入？')) return
  armPost('hand_move')
}

function engageArm() {
  if (!confirm('获取控制会立即发布 rt/arm_sdk 接管手臂（真机！），并在当前姿态刚性保持。\n请确认没有其他程序（遥操作 / reach_server 等）在控制手臂。')) return
  armPost('engage')
}

function disarmArm() {
  const extra = arm.value?.float ? '当前处于卸力模式，' : ''
  if (!confirm(`${extra}归还控制后手臂交还本体控制器，权重 1 秒渐出——请扶住手臂。确认归还？`)) return
  armPost('disarm')
}

function jointShortName(name) {
  return name.replace(/^(left|right)_/, '').replace(/_joint$/, '')
}

// ---- 指尖尖点标定（pivot：多姿态触同一固定点，只用 FK，不用相机） ----

const pivotSamples = ref([])
const pivotResult = ref(null)
const pivotBusy = ref(false)

async function refreshPivot() {
  try {
    const data = await (await fetch('/api/pivot/samples')).json()
    pivotSamples.value = data.samples || []
  } catch { /* 后端未起 */ }
}

async function pivotAdd() {
  pivotBusy.value = true
  errorMsg.value = ''
  try {
    const res = await fetch('/api/pivot/samples', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    })
    const data = await res.json()
    if (!data.ok) errorMsg.value = data.error || '采样失败'
    else { pivotResult.value = null; await refreshPivot() }
  } catch (e) {
    errorMsg.value = String(e)
  } finally {
    pivotBusy.value = false
  }
}

async function pivotDelete(index) {
  await fetch(`/api/pivot/samples/${index}`, { method: 'DELETE' })
  pivotResult.value = null
  await refreshPivot()
}

async function pivotClear() {
  if (!confirm('清空所有尖点样本？')) return
  await fetch('/api/pivot/clear', { method: 'POST' })
  pivotResult.value = null
  await refreshPivot()
}

async function pivotSolve() {
  pivotBusy.value = true
  errorMsg.value = ''
  try {
    const res = await fetch('/api/pivot/solve', { method: 'POST' })
    const data = await res.json()
    if (data.ok) pivotResult.value = data
    else errorMsg.value = data.error || '解算失败'
  } finally {
    pivotBusy.value = false
  }
}

// 腕姿态摘要（rpy 度，URDF 固定轴约定），方便看 roll 是否转开了
function wristRpyDeg(T) {
  const R = T
  const pitch = Math.atan2(-R[2][0], Math.hypot(R[0][0], R[1][0]))
  let roll, yaw
  if (Math.abs(Math.cos(pitch)) < 1e-8) {
    roll = 0
    yaw = Math.atan2(-R[0][1], R[1][1])
  } else {
    roll = Math.atan2(R[2][1], R[2][2])
    yaw = Math.atan2(R[1][0], R[0][0])
  }
  return [roll, pitch, yaw].map((a) => (a * 180 / Math.PI).toFixed(0)).join('/')
}

// 与 hand_eye_2D 保持一致：B 协力拖动、空格接住、C 拍摄。
async function onKeyDown(ev) {
  if (ev.repeat) return
  const tag = ev.target?.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'BUTTON' || tag === 'SELECT') return

  if (ev.code === 'Space') {
    ev.preventDefault()
    if (arm.value?.float && !armBusy.value) await armPost('stop')
    return
  }

  if (ev.code === 'KeyB') {
    ev.preventDefault()
    if (!arm.value?.enabled || !arm.value?.armed) {
      window.alert('请先获取手臂控制，再按 B 进入协力模式')
      return
    }
    if (arm.value.float || armBusy.value) return
    if (arm.value.jog_enabled) await armPost('disable_jog')
    await armPost('hand_move')
    return
  }

  if (ev.code === 'KeyC') {
    ev.preventDefault()
    if (!offlineMode.value && status.value?.recording?.enabled && !recordBusy.value) {
      await recordEpisode()
    }
  }
}

onMounted(async () => {
  window.addEventListener('keydown', onKeyDown)
  await refreshStatus()
  if (offlineMode.value) await loadMarkerColors()
  await refreshSamples()
  await refreshOfflineEpisodes()
  await refreshArm()
  await refreshPivot()
  setInterval(() => {
    if (!offlineMode.value) refreshArm()
  }, 800)
})

onBeforeUnmount(() => {
  clearTimeout(liveFrameTimer)
})
</script>

<template>
  <header class="topbar">
    <h1>Hand-Eye 3D 标定</h1>
    <span class="sub">
      眼在手外 · 联合估计指尖偏移 · 输出 T_{{ status?.base_link || 'base' }}←camera（彩色相机系）
    </span>
    <div class="spacer" />
    <span v-if="offlineMode" class="badge good">离线遥操作剧集</span>
    <span v-if="offlineMode && status?.teleop_task_dir" class="badge">
      数据目录: {{ status.teleop_task_dir }}
    </span>
    <span v-if="status && !offlineMode" class="badge">
      相机: {{ status.camera?.name || status.camera?.source }} {{ status.camera?.serial }}
    </span>
    <span v-if="!offlineMode && status?.camera?.width" class="badge">
      分辨率: {{ status.camera.width }}×{{ status.camera.height }}
    </span>
    <span v-if="status && !offlineMode" class="badge">
      位姿源: {{ status.pose_source }} ({{ status.wrist_link }})
    </span>
    <span v-if="status" class="badge">样本: {{ samples.length }}</span>
  </header>

  <div class="layout">
    <!-- 左：视频 -->
    <div class="video-panel">
      <div v-if="offlineMode" class="card offline-panel">
        <h2>
          离线剧集（{{ offlineEpisodes.length }}）
          <button class="btn refresh-btn" :disabled="episodesBusy" @click="refreshOfflineEpisodes">
            {{ episodesBusy ? '刷新中…' : '刷新' }}
          </button>
        </h2>
        <div v-if="offlineEpisodes.length" class="episode-list">
          <button
            v-for="episode in offlineEpisodes"
            :key="episode.name"
            class="episode-item"
            :class="{ selected: episode.name === selectedEpisode }"
            :disabled="markerBusy"
            @click="selectEpisode(episode)"
          >
            <span class="episode-main">
              <b>{{ episode.name }}</b>
              <span
                class="badge"
                :class="{ good: episodeImportedCount(episode) >= 9, bad: episodeImportedCount(episode) > 0 && episodeImportedCount(episode) < 9 }"
              >
                已保存 {{ episodeImportedCount(episode) }} 色
              </span>
            </span>
            <span class="episode-meta">
              {{ episode.frame_count }} 帧 · 预览 #{{ episode.preview_frame_idx }}
              · 样本 {{ episodeSampleCount(episode) }}
              <template v-if="episode.camera_serial"> · 相机 {{ episode.camera_serial }}</template>
            </span>
            <span
              v-for="warning in (episode.warnings || [])"
              :key="warning"
              class="episode-warning"
            >
              ⚠ {{ warning }}
            </span>
          </button>
        </div>
        <div v-else-if="!episodesBusy" class="coord dim">没有可用的离线剧集</div>
      </div>

      <div v-if="status" class="depth-controls">
        <label class="depth-toggle">
          <input v-model="showDepthOverlay" type="checkbox" />
          {{ offlineMode ? '叠加五帧中值深度' : '叠加实时 SDK 对齐深度' }}
        </label>
        <input
          v-model.number="depthOverlayOpacity"
          class="depth-opacity"
          type="range"
          min="10"
          max="90"
          step="5"
          :disabled="!showDepthOverlay"
        />
        <span class="coord dim">{{ depthOverlayOpacity }}%</span>
        <span class="depth-legend">近 <i></i> 远</span>
      </div>

      <div class="video-wrap">
        <img
          v-if="imageSrc"
          ref="imgEl"
          class="clickable-image"
          :src="imageSrc"
          :width="overlayImageSize?.[0] || undefined"
          :height="overlayImageSize?.[1] || undefined"
          :alt="offlineMode ? `${selectedEpisode} 代表帧` : '实时相机画面'"
          @load="onImageLoad"
          @error="onImageError"
          @click="onVideoClick"
        />
        <img
          v-if="showDepthOverlay && depthOverlaySrc"
          class="depth-overlay"
          :src="depthOverlaySrc"
          :style="{ opacity: depthOverlayOpacity / 100 }"
          :alt="offlineMode ? '五帧中值深度伪彩叠加' : '实时 SDK 对齐深度伪彩叠加'"
          @error="errorMsg = '深度叠加加载失败，请检查后端是否已重启'"
        />
        <MarkerOverlay
          v-if="offlineMode && overlayImageSize"
          :image-size="overlayImageSize"
          :markers="detectedMarkers"
          :colors="markerColors"
          :selected-key="selectedMarkerKey"
          :adding="!!addMarkerColorId"
          :disabled="markerBusy || markerConfirmBusy"
          @canvas-click="onOfflineCanvasClick"
          @select="selectedMarkerKey = $event"
          @move="moveMarker"
        />
        <div
          v-if="!offlineMode && clickPos"
          class="crosshair"
          :style="{ left: clickPos.xPct + '%', top: clickPos.yPct + '%' }"
        />
        <div v-if="offlineMode && markerBusy" class="image-loading">正在检测九色标记…</div>
        <div v-else-if="offlineMode && markerConfirmBusy" class="image-loading">
          确认中… {{ confirmElapsedSec }}s
        </div>
      </div>
      <div v-if="offlineMode" class="video-hint">
        每个姿态只确认当前手心或手背实际可见的颜色，不要求凑齐九色。拖动圆心可修正位置；
        先选“添加颜色”再点图像可补标。黄色虚线表示低置信度或有歧义，请逐一检查。
      </div>
      <div v-else class="video-hint">
        机械臂停稳后，点击画面中的标记点（指尖/手背贴纸中心）。深度<b>只取你点中的那个像素</b>
        （8 帧时域中值，无空间外扩），该像素测不到深度会直接报错而不是拿背景顶替——
        指尖太黑/太细测不到时，贴一小块哑光贴纸最有效。
        自动位姿源会在点击的同一时刻抓取手腕位姿。
        <b>手腕的朝向也要在各样本间充分变化</b>，否则指尖偏移解不出来。
      </div>
    </div>

    <!-- 右：操作区 -->
    <div class="side-panel">
      <!-- 手臂点动（--arm-control 时显示） -->
      <div v-if="arm?.enabled" class="card">
        <h2>
          0. {{ arm.armed && arm.arm ? (arm.arm === 'right' ? '右' : '左') + '臂点动' : '手臂控制' }}
          <span class="badge" :class="!arm.armed ? '' : (arm.float ? 'bad' : (arm.jog_enabled ? 'good' : 'good'))">
            {{ !arm.armed ? '未接管' : (arm.float ? '卸力中（扶住！）' : (arm.jog_enabled ? '点动开启' : '刚性保持')) }}
          </span>
        </h2>

        <template v-if="!arm.armed">
          <div class="field-row">
            <label></label>
            <button class="btn primary" :disabled="armBusy" @click="engageArm">获取控制（真机接管）</button>
          </div>
          <div class="video-hint">
            未接管时本服务只读 rt/lowstate，不发布任何控制指令，可与其他控制程序并存。
            点「获取控制」后开始发布 rt/arm_sdk 并在当前姿态刚性保持——
            <b>确保没有其他程序在控制手臂</b>。
          </div>
        </template>

        <template v-else>
          <div class="field-row">
            <label>模式</label>
            <button v-if="!arm.jog_enabled && !arm.float" class="btn primary"
                    :disabled="armBusy" @click="armPost('enable_jog')">开启点动</button>
            <button v-if="arm.jog_enabled" class="btn"
                    :disabled="armBusy" @click="armPost('disable_jog')">停止点动</button>
            <button v-if="!arm.jog_enabled && !arm.float" class="btn warn"
                    :disabled="armBusy" @click="handMove">进入协力模式 [B]</button>
            <button v-if="arm.float" class="btn primary"
                    :disabled="armBusy" @click="armPost('stop')">接住并保持 [空格]</button>
            <button class="btn warn" :disabled="armBusy" @click="disarmArm">归还控制</button>
          </div>
          <div class="field-row">
            <label>步长</label>
            <span class="step-group">
              <button v-for="s in stepOptions" :key="s" class="btn step-btn"
                      :class="{ active: stepDeg === s }" @click="stepDeg = s">{{ s }}°</button>
            </span>
          </div>
          <table class="jog-table">
            <tbody>
              <tr v-for="(name, i) in arm.joint_names" :key="name">
                <td class="jog-name">{{ jointShortName(name) }}</td>
                <td class="jog-val">
                  {{ arm.measured_rad ? (arm.measured_rad[i] * 180 / Math.PI).toFixed(1) : '?' }}°
                </td>
                <td>
                  <button class="btn jog-btn" :disabled="!arm.jog_enabled || armBusy"
                          @click="nudgeJoint(i, -1)">−</button>
                  <button class="btn jog-btn" :disabled="!arm.jog_enabled || armBusy"
                          @click="nudgeJoint(i, +1)">+</button>
                </td>
              </tr>
            </tbody>
          </table>
          <div class="video-hint">
            点动有限速（{{ arm.max_speed_rad_s }} rad/s）并钳制在关节限位内。
            卸力模式下 kp=0 只留阻尼 + 重力前馈（按实测角实时算），手臂近似失重、
            推到哪停哪；补偿有偏差时可能缓慢飘移，请护住手臂；
            按 <b>B</b> 进入协力模式，摆好后按<b>空格</b>接住并锁定；
            按 <b>C</b> 拍摄当前姿态。
            「归还控制」权重 1 秒渐出后交还本体控制器，请扶住手臂。
          </div>
        </template>
      </div>

      <!-- 当前样本 -->
      <div class="card">
        <template v-if="offlineMode">
          <h2>1. 九色标记确认</h2>
          <div class="marker-toolbar">
            <button
              class="btn"
              :disabled="markerBusy || markerConfirmBusy || !selectedEpisode"
              @click="detectMarkers()"
            >
              {{ markerBusy ? '检测中…' : '重跑自动检测' }}
            </button>
            <select v-model="addMarkerColorId" class="marker-select" :disabled="markerBusy">
              <option value="">添加颜色…</option>
              <option v-for="color in markerColors" :key="color.id" :value="String(color.id)">
                {{ color.label }}
              </option>
            </select>
            <span class="badge" :class="{ good: detectedMarkers.length > 0 }">
              画面 {{ detectedMarkers.length }} 个
            </span>
            <span v-if="savedOverlayMarkers.length" class="badge good">
              已保存 {{ savedOverlayMarkers.length }} 个
            </span>
          </div>

          <div v-if="addMarkerColorId" class="action-hint">
            请在左侧图像点击“{{ colorLabel(addMarkerColorId) }}”标记中心
          </div>
          <div v-if="duplicateColorIds.length" class="conflict-text">
            颜色冲突：{{ duplicateColorIds.map(colorLabel).join('、') }}。每集每色只能一个，修正后才能确认。
          </div>
          <div v-if="missingColors.length" class="warning-text">
            本姿态未检出（可能在另一面）：{{ missingColors.map(colorLabel).join('、') }}
          </div>
          <div v-for="warning in markerWarnings" :key="String(warning)" class="warning-text">
            {{ typeof warning === 'string' ? warning : (warning.message || JSON.stringify(warning)) }}
          </div>

          <div v-if="detectedMarkers.length" class="marker-list">
            <div
              v-for="(marker, index) in detectedMarkers"
              :key="marker._key"
              class="marker-row"
              :class="{ selected: marker._key === selectedMarkerKey }"
              @click="selectedMarkerKey = marker._key"
            >
              <span class="marker-index">{{ index + 1 }}</span>
              <select
                class="marker-select"
                :value="String(markerColorId(marker))"
                :disabled="isSavedMarker(marker)"
                @click.stop
                @change="changeMarkerColor(marker, $event.target.value)"
              >
                <option v-for="color in markerColors" :key="color.id" :value="String(color.id)">
                  {{ color.label }}
                </option>
              </select>
              <span class="coord marker-center-text">
                {{ Math.round(marker.center[0]) }}, {{ Math.round(marker.center[1]) }}
              </span>
              <span v-if="isSavedMarker(marker)" class="badge good">已保存</span>
              <span
                v-else
                class="confidence"
                :class="{ low: Number(marker.confidence ?? 1) < 0.65 || Number(marker.color_confidence ?? 1) < 0.65 || marker.flags?.length }"
              >
                {{ Math.round(Number(marker.confidence ?? 0) * 100) }}%
              </span>
              <button
                v-if="!isSavedMarker(marker)"
                class="del-btn"
                title="删除标记"
                @click.stop="deleteMarker(marker)"
              >✕</button>
            </div>
          </div>
          <div v-else-if="!markerBusy" class="coord dim">未检测到标记，请重跑或手动补标</div>

          <div class="field-row">
            <button class="btn primary" :disabled="!canConfirmMarkers" @click="confirmMarkers">
              {{ markerConfirmBusy ? `确认中… ${confirmElapsedSec}s` : '确认未保存标记' }}
            </button>
            <button
              class="btn primary"
              :disabled="!markerConfirmSucceeded || markerBatchBusy || markerBatchSaved"
              @click="saveMarkerBatch"
            >
              {{ markerBatchBusy ? '保存中…' : '整批保存观测' }}
            </button>
            <span v-if="savedOverlayMarkers.length && !unsavedMarkers.length" class="badge good">本姿态已在磁盘</span>
            <span v-else-if="markerBatchSaved" class="badge good">已整批保存</span>
          </div>

          <div v-if="markerConfirmation" class="confirm-summary">
            <span class="badge" :class="{ good: markerConfirmSucceeded, bad: !markerConfirmSucceeded }">
              深度成功 {{ confirmedObservations.length }}/{{ unsavedMarkers.length }}
            </span>
            <span class="badge">当前总样本 {{ samples.length }}</span>
          </div>
          <div v-if="markerConfirmation" class="depth-list">
            <div v-for="(marker, index) in detectedMarkers" :key="marker._key" class="depth-row">
              <span>{{ colorLabel(markerColorId(marker)) }}</span>
              <span v-if="isSavedMarker(marker)" class="coord ok">已在磁盘</span>
              <template v-else-if="confirmationForMarker(marker, index).observation">
                <span class="coord ok">
                  ✓ {{ Math.round(confirmationForMarker(marker, index).observation.depth_mm) }} mm
                </span>
              </template>
              <span v-else-if="confirmationForMarker(marker, index).error" class="err-text">
                ✕ {{ markerErrorText(confirmationForMarker(marker, index).error) }}
              </span>
              <span v-else class="coord dim">待确认</span>
            </div>
            <div
              v-for="message in (markerConfirmation.errors || []).filter((item) => typeof item === 'string')"
              :key="message"
              class="err-text"
            >
              {{ message }}
            </div>
          </div>
          <div v-if="errorMsg" class="err-text">⚠ {{ errorMsg }}</div>
        </template>
        <template v-else>
          <h2>1. 当前样本</h2>
          <div v-if="status?.recording?.task_dir" class="field-row">
            <label>离线数据</label>
            <button
              class="btn primary"
              :disabled="!status?.recording?.enabled || recordBusy"
              @click="recordEpisode"
            >
              {{ recordBusy ? '正在保存 5 帧…' : '拍摄当前姿态 [C]' }}
            </button>
            <span v-if="recordMessage" class="coord ok">{{ recordMessage }}</span>
            <span v-else class="coord dim">
              已有 {{ status?.recording?.episode_count || 0 }} 组
            </span>
          </div>
          <div v-if="status?.recording?.task_dir" class="video-hint">
            保存到 {{ status.recording.task_dir }}；机械臂保持静止后拍摄。
          </div>
          <div class="field-row">
            <label>P_camera</label>
            <span v-if="pickBusy" class="coord dim">取点中…</span>
            <span v-else-if="pick" class="coord ok">
              [{{ fmt(pick.p_camera[0]) }}, {{ fmt(pick.p_camera[1]) }}, {{ fmt(pick.p_camera[2]) }}] m
              · 深度 {{ Math.round(pick.depth_mm) }}mm
            </span>
            <span v-else class="coord dim">← 在左侧画面上点击标记点</span>
          </div>

          <template v-if="autoPose">
            <div class="field-row">
              <label>手腕位姿</label>
              <span v-if="wristT" class="coord ok">t = {{ wristSummary(wristT) }} m（自动）</span>
              <span v-else class="coord dim">点击取点时自动抓取</span>
              <button class="btn" @click="readWristPose">重读</button>
            </div>
          </template>
          <template v-else>
            <div class="field-row">
              <label>腕 xyz (m)</label>
              <input v-model="wristManual.x" placeholder="x" />
              <input v-model="wristManual.y" placeholder="y" />
              <input v-model="wristManual.z" placeholder="z" />
            </div>
            <div class="field-row">
              <label>腕 rpy (rad)</label>
              <input v-model="wristManual.roll" placeholder="roll" />
              <input v-model="wristManual.pitch" placeholder="pitch" />
              <input v-model="wristManual.yaw" placeholder="yaw" />
            </div>
          </template>

          <div class="field-row">
            <label></label>
            <button class="btn primary" :disabled="!canSave" @click="saveSample">保存这个样本</button>
          </div>
          <div v-if="errorMsg" class="err-text">⚠ {{ errorMsg }}</div>
        </template>
      </div>

      <!-- 样本列表 -->
      <div class="card">
        <h2>2. 已采样本（{{ samples.length }} / 最少 {{ minSamples }}，建议 ≥ 12）</h2>
        <table v-if="samples.length">
          <thead>
            <tr>
              <th>#</th><th>P_camera (m)</th><th>腕 t (m)</th>
              <th>颜色</th><th>剧集</th><th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="s in samples" :key="s.index">
              <td>{{ s.index }}</td>
              <td>{{ s.p_camera.map((v) => fmt(v, 3)).join(', ') }}</td>
              <td>{{ wristSummary(s.T_base_wrist) }}</td>
              <td>{{ sampleMarkerLabel(s) }}</td>
              <td>{{ s.episode || s.provenance?.episode || '—' }}</td>
              <td><button class="del-btn" title="删除" @click="deleteSample(s.index)">✕</button></td>
            </tr>
          </tbody>
        </table>
        <div v-else class="coord dim">还没有样本</div>
      </div>

      <!-- 解算 -->
      <div class="card">
        <h2>3. 解算 T_base←camera + 指尖偏移</h2>
        <button class="btn primary" :disabled="samples.length < minSamples || solveBusy" @click="solve">
          {{ solveBusy ? '解算中…' : `用 ${samples.length} 个样本解算` }}
        </button>
        <template v-if="result">
          <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap;">
            <span
              v-if="result.residual_mm?.rms != null"
              class="badge"
              :class="result.residual_mm.rms < 8 ? 'good' : 'bad'"
            >
              拟合 RMS {{ fmt(result.residual_mm.rms, 2) }} mm
            </span>
            <span v-if="result.leave_one_out_stats_mm" class="badge"
                  :class="result.leave_one_out_stats_mm.mean < 10 ? 'good' : 'bad'">
              留一验证均值 {{ fmt(result.leave_one_out_stats_mm.mean, 2) }} mm
            </span>
            <span v-if="!isMultiResult && result.p_tool_wrist_m" class="badge">
              p_tool(腕系) [{{ result.p_tool_wrist_m.map((v) => fmt(v, 3)).join(', ') }}] m
            </span>
            <span v-if="result.rpy_deg" class="badge">
              rpy(deg) [{{ result.rpy_deg.map((v) => fmt(v, 2)).join(', ') }}]
            </span>
            <span v-if="result.wrist_rotation_spread_deg != null" class="badge">
              腕姿态跨度 {{ fmt(result.wrist_rotation_spread_deg, 1) }}°
            </span>
            <span v-if="isMultiResult" class="badge good">
              多标记联合解算
            </span>
            <span v-if="result.pose_count != null" class="badge">姿态 {{ result.pose_count }}</span>
            <span v-if="result.marker_count != null" class="badge">标记 {{ result.marker_count }}</span>
          </div>
          <table v-if="multiMarkerRows.length" class="multi-result-table">
            <thead>
              <tr><th>标记</th><th>p_tool 腕系 (m)</th><th>RMS (mm)</th><th>最大 (mm)</th><th>样本</th></tr>
            </thead>
            <tbody>
              <tr v-for="row in multiMarkerRows" :key="row.id">
                <td>{{ colorLabel(row.id) }}</td>
                <td>{{ row.offset.length ? row.offset.map((v) => fmt(v, 4)).join(', ') : '—' }}</td>
                <td>{{ row.residual.rms != null ? fmt(row.residual.rms, 2) : '—' }}</td>
                <td>{{ row.residual.max != null ? fmt(row.residual.max, 2) : '—' }}</td>
                <td>{{ row.residual.count ?? row.residual.sample_count ?? '—' }}</td>
              </tr>
            </tbody>
          </table>
          <div v-if="matrixText" class="result-box" style="margin-top: 10px;">{{ matrixText }}</div>
          <div v-if="result.saved_to" class="video-hint">已保存到 {{ result.saved_to }}</div>
        </template>

        <div class="field-row" style="margin-top: 12px;">
          <button class="btn" :disabled="samples.length < 3 || toolBusy" @click="solveToolOnly">
            {{ toolBusy ? '解算中…' : '只解指尖偏移（固定相机外参）' }}
          </button>
        </div>
        <div class="video-hint">
          已有可信的相机外参时用这个：每次点击<b>指尖的尖端</b>采样（不同手腕姿态、
          含反手大 roll），3 个样本起步、建议 ≥ 8。自动复用最新一份联合解算的
          T_base←camera，只解 3 个未知量，比联合解稳。
        </div>
        <template v-if="toolResult">
          <div style="margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="badge" :class="toolResult.residual_mm.rms < 5 ? 'good' : 'bad'">
              拟合 RMS {{ fmt(toolResult.residual_mm.rms, 2) }} mm
              （最大 {{ fmt(toolResult.residual_mm.max, 1) }}）
            </span>
            <span class="badge">
              p_tool(腕系) [{{ toolResult.p_tool_wrist_m.map((v) => fmt(v, 4)).join(', ') }}] m
            </span>
            <span class="badge">姿态跨度 {{ fmt(toolResult.wrist_rotation_spread_deg, 1) }}°</span>
            <span v-if="toolResult.delta_vs_calib_norm_mm != null" class="badge"
                  :class="toolResult.delta_vs_calib_norm_mm < 5 ? 'good' : 'bad'">
              与原 p_tool 差 {{ fmt(toolResult.delta_vs_calib_norm_mm, 1) }} mm
              [{{ toolResult.delta_vs_calib_mm.map((v) => fmt(v, 1)).join(', ') }}]
            </span>
            <span v-if="toolResult.dropped_samples?.length" class="badge bad">
              已自动剔除 {{ toolResult.dropped_samples.map((d) => `#${d.index}（${Math.round(d.residual_mm)}mm）`).join('、') }}
            </span>
            <span class="badge">实际参与 {{ toolResult.sample_indices.length }} 个样本</span>
          </div>
          <div class="video-hint">
            外参来自 {{ toolResult.calib_used }}；
            已生成替换 p_tool 的完整标定文件 {{ toolResult.merged_calib }}，
            可直接给 reach_server --calib 用（--tool-out-mm 记得给 0）。
          </div>
        </template>
      </div>

      <!-- 指尖尖点标定 -->
      <div class="card">
        <h2>4. 指尖尖点标定（多姿态触同一点，不用相机）</h2>
        <div class="video-hint">
          找一个固定的尖角参照物（桌角/螺丝尖）。用「卸力拖动」把<b>指尖顶在该点上</b>，
          点「保持当前位置」锁定后再点下面「采样当前姿态」；然后换一个手腕姿态
          （<b>务必包含反手大角度 roll</b>，就是拨开关那个姿态）重新顶到同一点，重复采样。
          建议 ≥ 6 个、姿态越分散越准。解算只用手腕 FK，不依赖相机；
          残差反映"各姿态下指尖没真正钉在同一点"的程度。
        </div>
        <div class="field-row" style="margin-top: 8px;">
          <label></label>
          <button class="btn primary" :disabled="pivotBusy" @click="pivotAdd">采样当前姿态</button>
          <button class="btn primary" :disabled="pivotSamples.length < 4 || pivotBusy" @click="pivotSolve">
            {{ pivotBusy ? '…' : `用 ${pivotSamples.length} 个姿态解算` }}
          </button>
          <button class="btn" :disabled="!pivotSamples.length" @click="pivotClear">清空</button>
        </div>
        <table v-if="pivotSamples.length">
          <thead>
            <tr><th>#</th><th>腕 t (m)</th><th>腕 rpy (°)</th><th></th></tr>
          </thead>
          <tbody>
            <tr v-for="s in pivotSamples" :key="s.index">
              <td>{{ s.index }}</td>
              <td>{{ wristSummary(s.T_base_wrist) }}</td>
              <td>{{ wristRpyDeg(s.T_base_wrist) }}</td>
              <td><button class="del-btn" title="删除" @click="pivotDelete(s.index)">✕</button></td>
            </tr>
          </tbody>
        </table>
        <div v-else class="coord dim">还没有尖点样本</div>
        <template v-if="pivotResult">
          <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="badge" :class="pivotResult.residual_mm.rms < 5 ? 'good' : 'bad'">
              拟合 RMS {{ fmt(pivotResult.residual_mm.rms, 2) }} mm
            </span>
            <span v-if="pivotResult.leave_one_out_stats_mm" class="badge"
                  :class="pivotResult.leave_one_out_stats_mm.mean < 8 ? 'good' : 'bad'">
              留一验证均值 {{ fmt(pivotResult.leave_one_out_stats_mm.mean, 2) }} mm
            </span>
            <span class="badge">
              p_tool(腕系) [{{ pivotResult.p_tool_wrist_m.map((v) => fmt(v, 4)).join(', ') }}] m
            </span>
            <span class="badge">姿态跨度 {{ fmt(pivotResult.wrist_rotation_spread_deg, 1) }}°</span>
            <span v-if="pivotResult.delta_vs_handeye_norm_mm != null" class="badge"
                  :class="pivotResult.delta_vs_handeye_norm_mm < 5 ? 'good' : 'bad'">
              与手眼标定 p_tool 差 {{ fmt(pivotResult.delta_vs_handeye_norm_mm, 1) }} mm
              [{{ pivotResult.delta_vs_handeye_mm.map((v) => fmt(v, 1)).join(', ') }}]
            </span>
          </div>
          <div class="video-hint">
            已保存到 {{ pivotResult.saved_to }}
            <template v-if="pivotResult.merged_calib">
              ；同时生成了替换 p_tool 的完整标定文件 {{ pivotResult.merged_calib }}，
              可直接给 reach_server 的 --calib 使用（注意 --tool-out-mm 是否还需要）。
            </template>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.offline-panel {
  margin-bottom: 12px;
}

.offline-panel h2 {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.refresh-btn {
  padding: 4px 10px;
}

.episode-list {
  display: grid;
  gap: 6px;
  max-height: 260px;
  overflow-y: auto;
}

.episode-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  width: 100%;
  padding: 9px 10px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  color: var(--text);
  text-align: left;
  cursor: pointer;
}

.episode-item:hover,
.episode-item.selected {
  border-color: var(--accent);
  background: var(--panel-hover);
}

.episode-item:disabled {
  cursor: wait;
  opacity: .65;
}

.episode-item.selected {
  box-shadow: inset 3px 0 var(--accent);
}

.episode-main {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.episode-main b {
  overflow-wrap: anywhere;
}

.episode-meta {
  color: var(--text-dim);
  font-size: 11px;
}

.episode-warning {
  color: var(--warn);
  font-size: 11px;
  line-height: 1.4;
}

.image-loading {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  background: rgba(15, 17, 23, .62);
  color: var(--text);
  font-size: 13px;
  pointer-events: none;
}

.marker-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}

.marker-select {
  min-width: 100px;
  padding: 6px 8px;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--bg);
  color: var(--text);
}

.marker-select:focus {
  outline: none;
  border-color: var(--accent);
}

.action-hint,
.warning-text,
.conflict-text {
  margin: 6px 0;
  font-size: 12px;
}

.action-hint {
  color: var(--accent);
}

.warning-text {
  color: var(--warn);
}

.conflict-text {
  color: var(--err);
}

.marker-list {
  display: grid;
  gap: 5px;
  max-height: 280px;
  margin: 10px 0;
  overflow-y: auto;
}

.marker-row {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 6px 7px;
  border: 1px solid var(--border);
  border-radius: 7px;
  cursor: pointer;
}

.marker-row:hover,
.marker-row.selected {
  border-color: var(--accent);
  background: var(--panel-hover);
}

.marker-index {
  width: 18px;
  color: var(--text-dim);
  font-family: monospace;
}

.marker-center-text {
  flex: 1;
  color: var(--text-dim);
  text-align: right;
}

.confidence {
  min-width: 36px;
  color: var(--ok);
  font-family: monospace;
  font-size: 11px;
  text-align: right;
}

.confidence.low {
  color: var(--warn);
}

.confirm-summary {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin: 8px 0;
}

.depth-list {
  display: grid;
  gap: 4px;
}

.depth-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 5px 7px;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
}

.depth-row .err-text {
  margin-top: 0;
}

.multi-result-table {
  margin-top: 10px;
}
</style>
