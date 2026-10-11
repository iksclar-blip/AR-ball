package com.example.arball

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix as AMatrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RadialGradient
import android.graphics.Rect
import android.graphics.Shader
import android.graphics.YuvImage
import android.media.Image
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.opengl.Matrix as GLM
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.TextView
import android.widget.Toast
import com.google.ar.core.Anchor
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

class MainActivity : Activity(), GLSurfaceView.Renderer {

    // ---------- UI / session ----------
    private lateinit var gl: GLSurfaceView
    private lateinit var hint: TextView
    private var session: Session? = null
    private var installRequested = false
    private val resetReq = AtomicBoolean(true)

    @Volatile private var vw = 1
    @Volatile private var vh = 1
    private var viewportChanged = true
    private var texSet = false

    // ---------- GL ----------
    private var bgProg = 0
    private var objProg = 0
    private var oesTex = 0
    private var ballTex = 0
    private var shadowTex = 0
    private lateinit var bgCoords: FloatBuffer
    private lateinit var bgUv: FloatBuffer
    private lateinit var ballVB: FloatBuffer
    private lateinit var ballIB: ShortBuffer
    private var ballIdxCount = 0
    private lateinit var quadVB: FloatBuffer

    // ---------- AR ----------
    private var anchor: Anchor? = null
    private var stable = 0

    // ---------- physics (anchor-local space, Y up, floor at y = 0) ----------
    private val RAD = 0.11f
    private val bp = FloatArray(3)
    private val bv = FloatArray(3)
    private val bw = FloatArray(3)
    private val bq = floatArrayOf(1f, 0f, 0f, 0f) // w, x, y, z
    private var lastTs = 0L
    private var acc = 0f

    // ---------- hand tracking ----------
    private var landmarker: HandLandmarker? = null
    private val exec = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)

    private class Obs(val cx: Float, val cy: Float, val palmPx: Float, val ts: Long)

    @Volatile private var obs: Obs? = null
    private var lastObsTs = 0L
    private var lastObsNs = 0L
    private var px0 = 0f
    private var py0 = 0f
    private var pts0 = 0L
    private var hasPrev = false
    private var hvx = 0f
    private var hvy = 0f
    private var lastKickNs = 0L
    private var status = ""
    private var lastUiNs = 0L
    private var frameN = 0

    // =====================================================================
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = FrameLayout(this)
        gl = GLSurfaceView(this).apply {
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            preserveEGLContextOnPause = true
            setRenderer(this@MainActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
        root.addView(gl, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)

        hint = TextView(this).apply {
            setTextColor(Color.WHITE)
            setBackgroundColor(0x99000000.toInt())
            setPadding(32, 20, 32, 20)
            textSize = 15f
            text = "מתחיל…"
        }
        root.addView(
            hint,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.CENTER_HORIZONTAL
            ).apply { topMargin = 90 }
        )

        val btn = Button(this).apply {
            text = "↺ אפס כדור"
            setOnClickListener { resetReq.set(true) }
        }
        root.addView(
            btn,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            ).apply { bottomMargin = 120 }
        )
        setContentView(root)

        try {
            val base = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
            val opts = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(base)
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.4f)
                .setMinHandPresenceConfidence(0.4f)
                .build()
            landmarker = HandLandmarker.createFromOptions(this, opts)
        } catch (e: Throwable) {
            Toast.makeText(this, "מודל זיהוי היד לא נטען: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    override fun onResume() {
        super.onResume()
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 1)
            return
        }
        if (session == null) {
            try {
                when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED -> { installRequested = true; return }
                    else -> {}
                }
                val s = Session(this)
                val cfg = Config(s)
                cfg.planeFindingMode = Config.PlaneFindingMode.HORIZONTAL
                cfg.focusMode = Config.FocusMode.AUTO
                cfg.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                cfg.lightEstimationMode = Config.LightEstimationMode.DISABLED
                s.configure(cfg)
                session = s
                texSet = false
            } catch (e: Exception) {
                Toast.makeText(this, "ARCore לא זמין: ${e.message}", Toast.LENGTH_LONG).show()
                return
            }
        }
        try { session?.resume() } catch (e: Exception) {
            Toast.makeText(this, "המצלמה לא זמינה", Toast.LENGTH_LONG).show(); return
        }
        gl.onResume()
    }

    override fun onPause() {
        super.onPause()
        gl.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        super.onDestroy()
        session?.close(); session = null
        landmarker?.close()
        exec.shutdown()
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, res: IntArray) {
        if (res.isNotEmpty() && res[0] == PackageManager.PERMISSION_GRANTED) recreate()
        else Toast.makeText(this, "נדרשת הרשאת מצלמה", Toast.LENGTH_LONG).show()
    }

    // =====================================================================
    // GL setup
    override fun onSurfaceCreated(u: GL10?, c: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        bgProg = program(BG_VS, BG_FS)
        objProg = program(OBJ_VS, OBJ_FS)

        val t = IntArray(1)
        GLES20.glGenTextures(1, t, 0)
        oesTex = t[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        bgCoords = fb(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
        bgUv = fb(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f))
        texSet = false

        buildBall()
        quadVB = fb(
            floatArrayOf(
                -1f, 0f, -1f, 0f, 1f, 0f, 0f, 0f,
                1f, 0f, -1f, 0f, 1f, 0f, 1f, 0f,
                -1f, 0f, 1f, 0f, 1f, 0f, 0f, 1f,
                1f, 0f, 1f, 0f, 1f, 0f, 1f, 1f
            )
        )
        ballTex = makeTex(ballBitmap())
        shadowTex = makeTex(shadowBitmap())
    }

    override fun onSurfaceChanged(u: GL10?, w: Int, h: Int) {
        GLES20.glViewport(0, 0, w, h)
        vw = w; vh = h
        viewportChanged = true
    }

    // =====================================================================
    // Frame loop
    private val view = FloatArray(16)
    private val proj = FloatArray(16)
    private val anchorM = FloatArray(16)
    private val vp = FloatArray(16)
    private val base = FloatArray(16)

    override fun onDrawFrame(u: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        if (!texSet) { s.setCameraTextureName(oesTex); texSet = true }
        if (viewportChanged) {
            @Suppress("DEPRECATION")
            s.setDisplayGeometry(windowManager.defaultDisplay.rotation, vw, vh)
            viewportChanged = false
        }
        val frame: Frame = try { s.update() } catch (e: Exception) { return }

        if (frame.hasDisplayGeometryChanged()) {
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, bgCoords,
                Coordinates2d.TEXTURE_NORMALIZED, bgUv
            )
        }
        drawBackground()

        val camera = frame.camera
        if (camera.trackingState != TrackingState.TRACKING) { setStatus("מאבד מעקב — הזז את הטלפון לאט"); return }
        camera.getViewMatrix(view, 0)
        camera.getProjectionMatrix(proj, 0, 0.05f, 50f)
        GLM.multiplyMM(vp, 0, proj, 0, view, 0)

        if (resetReq.getAndSet(false)) { anchor?.detach(); anchor = null; stable = 0 }

        // ---- placement: crosshair at screen centre must hit a horizontal plane ----
        if (anchor == null) {
            var placed = false
            for (hit in frame.hitTest(vw / 2f, vh / 2f)) {
                val t = hit.trackable
                if (t is Plane && t.isPoseInPolygon(hit.hitPose) &&
                    t.type == Plane.Type.HORIZONTAL_UPWARD_FACING
                ) {
                    stable++
                    if (stable > 20) {
                        anchor = hit.createAnchor()
                        bp[0] = 0f; bp[1] = 0.45f; bp[2] = 0f
                        bv.fill(0f); bw.fill(0f)
                        bq[0] = 1f; bq[1] = 0f; bq[2] = 0f; bq[3] = 0f
                        lastTs = 0L
                        hasPrev = false
                    }
                    placed = true
                    break
                }
            }
            if (!placed) stable = 0
            if (anchor == null) { setStatus("כוון את המצלמה אל הרצפה או שולחן והזז לאט"); return }
        }
        val a = anchor!!
        if (a.trackingState != TrackingState.TRACKING) return
        val anchorPose = a.pose
        anchorPose.toMatrix(anchorM, 0)

        // ---- hand detection (every 2nd frame, async) ----
        frameN++
        if (frameN % 2 == 0) submitDetection(frame)

        // ---- physics ----
        if (lastTs != 0L) {
            val dt = ((frame.timestamp - lastTs) / 1e9f).coerceIn(0f, 0.05f)
            acc += dt
            val h = 1f / 120f
            while (acc >= h) { step(h); acc -= h }
        }
        lastTs = frame.timestamp
        if (hypot(bp[0], bp[2]) > 8f || bp[1] < -3f) {
            bp[0] = 0f; bp[1] = 0.45f; bp[2] = 0f; bv.fill(0f); bw.fill(0f)
        }

        // ---- hand → ball interaction ----
        handleHand(camera.pose, anchorPose)

        // ---- draw ----
        GLM.multiplyMM(base, 0, vp, 0, anchorM, 0)
        drawScene()
    }

    // =====================================================================
    // Physics
    private fun step(h: Float) {
        bv[1] -= 9.81f * h
        bp[0] += bv[0] * h; bp[1] += bv[1] * h; bp[2] += bv[2] * h

        val wx = bw[0]; val wy = bw[1]; val wz = bw[2]
        val w = bq[0]; val x = bq[1]; val y = bq[2]; val z = bq[3]
        val nw = w + 0.5f * h * (-wx * x - wy * y - wz * z)
        val nx = x + 0.5f * h * (wx * w + wy * z - wz * y)
        val ny = y + 0.5f * h * (wy * w + wz * x - wx * z)
        val nz = z + 0.5f * h * (wz * w + wx * y - wy * x)
        val n = sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
        bq[0] = nw / n; bq[1] = nx / n; bq[2] = ny / n; bq[3] = nz / n

        val drag = 1f - 0.05f * h
        bv[0] *= drag; bv[1] *= drag; bv[2] *= drag

        if (bp[1] < RAD) {
            bp[1] = RAD
            var jn = 9.81f * h
            if (bv[1] < 0f) {
                val e = if (-bv[1] > 0.6f) 0.72f else 0f
                jn = max(jn, -bv[1] * (1f + e))
                bv[1] = -bv[1] * e
            }
            // contact-point velocity (tangential)
            val ry = -RAD
            val cx = bw[1] * 0f - bw[2] * ry + bv[0]
            val cz = bw[0] * ry - bw[1] * 0f + bv[2]
            val len = sqrt(cx * cx + cz * cz)
            if (len > 1e-4f) {
                val jt = min(len / 3.5f, 0.45f * jn)   // Coulomb friction, solid sphere
                val dx = -cx / len; val dz = -cz / len
                bv[0] += dx * jt; bv[2] += dz * jt
                val k = jt / (0.4f * RAD * RAD)
                bw[0] += ry * dz * k
                bw[2] += -ry * dx * k
            }
            // rolling resistance
            val d = 1f - 0.5f * h
            bw[0] *= d; bw[2] *= d; bw[1] *= (1f - 1.5f * h)
            bv[0] *= (1f - 0.25f * h); bv[2] *= (1f - 0.25f * h)
            if (hypot(bv[0], bv[2]) < 0.02f && abs(bv[1]) < 0.02f &&
                sqrt(bw[0] * bw[0] + bw[1] * bw[1] + bw[2] * bw[2]) < 0.4f
            ) { bv.fill(0f); bw.fill(0f) }
        }
    }

    // =====================================================================
    // Hand tracking
    private fun yuvToNv21(img: Image): ByteArray {
        val w = img.width; val h = img.height
        val out = ByteArray(w * h * 3 / 2)
        val yP = img.planes[0]; val uP = img.planes[1]; val vP = img.planes[2]
        var pos = 0
        val yb = yP.buffer
        for (row in 0 until h) { yb.position(row * yP.rowStride); yb.get(out, pos, w); pos += w }
        val ub = uP.buffer; val vb = vP.buffer
        for (row in 0 until h / 2) for (col in 0 until w / 2) {
            out[pos++] = vb.get(row * vP.rowStride + col * vP.pixelStride)
            out[pos++] = ub.get(row * uP.rowStride + col * uP.pixelStride)
        }
        return out
    }

    private fun submitDetection(frame: Frame) {
        if (busy.get() || landmarker == null) return
        val img = try { frame.acquireCameraImage() } catch (e: NotYetAvailableException) { return }
        val w = img.width; val h = img.height
        val nv21 = try { yuvToNv21(img) } finally { img.close() }
        // affine map: view-normalised corners (TL, TR, BL) -> image-normalised
        val out = FloatArray(6)
        frame.transformCoordinates2d(
            Coordinates2d.VIEW_NORMALIZED, floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f),
            Coordinates2d.IMAGE_NORMALIZED, out
        )
        val ts = frame.timestamp
        val vwS = vw; val vhS = vh
        busy.set(true)
        exec.execute {
            try { detect(nv21, w, h, out, ts, vwS, vhS) } catch (_: Throwable) {} finally { busy.set(false) }
        }
    }

    private fun detect(nv21: ByteArray, w: Int, h: Int, o: FloatArray, ts: Long, vwS: Int, vhS: Int) {
        val bos = ByteArrayOutputStream()
        YuvImage(nv21, ImageFormat.NV21, w, h, null).compressToJpeg(Rect(0, 0, w, h), 75, bos)
        val src = BitmapFactory.decodeByteArray(bos.toByteArray(), 0, bos.size()) ?: return
        // re-render the camera image so that it is upright and aligned with the screen view
        val dw = 384
        val dh = (dw * vhS.toFloat() / vwS).roundToInt()
        val dst = Bitmap.createBitmap(dw, dh, Bitmap.Config.ARGB_8888)
        val m = AMatrix()
        m.setPolyToPoly(
            floatArrayOf(o[0] * w, o[1] * h, o[2] * w, o[3] * h, o[4] * w, o[5] * h), 0,
            floatArrayOf(0f, 0f, dw.toFloat(), 0f, 0f, dh.toFloat()), 0, 3
        )
        Canvas(dst).drawBitmap(src, m, null)
        val res = landmarker?.detect(BitmapImageBuilder(dst).build())
        val hands = res?.landmarks()
        if (hands == null || hands.isEmpty()) { obs = null; return }
        val l = hands[0]
        val ids = intArrayOf(0, 5, 9, 13, 17)
        var cx = 0f; var cy = 0f
        for (i in ids) { cx += l[i].x(); cy += l[i].y() }
        cx /= ids.size; cy /= ids.size
        val palm = hypot((l[5].x() - l[17].x()) * vwS, (l[5].y() - l[17].y()) * vhS)
        obs = Obs(cx, cy, palm, ts)
    }

    private fun handleHand(camPose: com.google.ar.core.Pose, anchorPose: com.google.ar.core.Pose) {
        val now = System.nanoTime()
        val o = obs
        if (o == null) {
            if (now - lastObsNs > 400_000_000L) hasPrev = false
            setStatus("הושט יד מול המצלמה ודחוף את הכדור  |  יד: לא זוהתה")
            return
        }
        if (o.ts == lastObsTs) return
        lastObsTs = o.ts; lastObsNs = now

        val px = o.cx * vw; val py = o.cy * vh
        if (hasPrev) {
            val dt = (o.ts - pts0) / 1e9f
            if (dt in 0.01f..0.5f) {
                hvx = 0.5f * hvx + 0.5f * (px - px0) / dt
                hvy = 0.5f * hvy + 0.5f * (py - py0) / dt
            }
        } else { hvx = 0f; hvy = 0f }
        px0 = px; py0 = py; pts0 = o.ts; hasPrev = true

        // ball in world / screen space
        val wl = FloatArray(3)
        anchorPose.transformPoint(bp, 0, wl, 0)
        val ct = camPose.translation
        val dist = sqrt((wl[0] - ct[0]).let { it * it } + (wl[1] - ct[1]).let { it * it } + (wl[2] - ct[2]).let { it * it })
        val xA = camPose.xAxis
        val c1 = project(wl[0], wl[1], wl[2]) ?: return
        val c2 = project(wl[0] + xA[0] * RAD, wl[1] + xA[1] * RAD, wl[2] + xA[2] * RAD) ?: return
        val ballR = hypot(c2[0] - c1[0], c2[1] - c1[1])
        val d2 = hypot(px - c1[0], py - c1[1])

        // monocular depth check: apparent palm size vs. expected size at the ball's distance
        val fPx = proj[0] * vw / 2f
        val expectedPalm = 0.085f * fPx / max(dist, 0.1f)
        val ratio = o.palmPx / expectedPalm
        val speed = hypot(hvx, hvy)
        val reach = ballR + 0.5f * o.palmPx
        setStatus("יד זוהתה | קרבה למסך: ${d2.roundToInt()}/${reach.roundToInt()} | עומק: ${"%.1f".format(ratio)} | מהירות: ${(speed / vh * 10).roundToInt() / 10f}")

        if (d2 < reach && ratio in 0.45f..2.2f && speed > 0.3f * vh && now - lastKickNs > 350_000_000L) {
            lastKickNs = now
            val zA = camPose.zAxis; val yA = camPose.yAxis
            var fx = -zA[0]; var fz = -zA[2]
            if (hypot(fx, fz) < 0.25f) { fx = yA[0]; fz = yA[2] }
            val fl = hypot(fx, fz).coerceAtLeast(1e-4f); fx /= fl; fz /= fl
            val rl = hypot(xA[0], xA[2]).coerceAtLeast(1e-4f)
            val rx = xA[0] / rl; val rz = xA[2] / rl
            // screen-right -> camera right, screen-up -> forward (away)
            var dx = rx * hvx + fx * (-hvy)
            var dz = rz * hvx + fz * (-hvy)
            val dl = hypot(dx, dz).coerceAtLeast(1e-4f); dx /= dl; dz /= dl
            val dirW = floatArrayOf(dx, 0f, dz)
            val dirL = FloatArray(3)
            anchorPose.inverse().rotateVector(dirW, 0, dirL, 0)
            val ll = hypot(dirL[0], dirL[2]).coerceAtLeast(1e-4f)
            val lx = dirL[0] / ll; val lz = dirL[2] / ll
            val sp = (speed / vh * 2.5f).coerceIn(1.2f, 6f)
            bv[0] += lx * sp; bv[2] += lz * sp; bv[1] += 0.8f + 0.25f * sp
            // topspin: roll in the push direction
            bw[0] += lz * 0.3f * sp / RAD
            bw[2] += -lx * 0.3f * sp / RAD
            setStatus("בעיטה! 🦶")
        }
    }

    private fun project(x: Float, y: Float, z: Float): FloatArray? {
        val p = floatArrayOf(x, y, z, 1f); val c = FloatArray(4)
        GLM.multiplyMV(c, 0, vp, 0, p, 0)
        if (c[3] <= 1e-4f) return null
        return floatArrayOf((c[0] / c[3] * 0.5f + 0.5f) * vw, (1f - (c[1] / c[3] * 0.5f + 0.5f)) * vh)
    }

    private fun setStatus(t: String) {
        val now = System.nanoTime()
        if (t.startsWith("בעיטה") || now - lastUiNs > 150_000_000L) {
            lastUiNs = now
            status = t
            runOnUiThread { hint.text = t }
        }
    }

    // =====================================================================
    // Drawing
    private fun drawBackground() {
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)
        GLES20.glUseProgram(bgProg)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(bgProg, "uTex"), 0)
        val aPos = GLES20.glGetAttribLocation(bgProg, "aPos")
        val aUV = GLES20.glGetAttribLocation(bgProg, "aUV")
        bgCoords.position(0); bgUv.position(0)
        GLES20.glEnableVertexAttribArray(aPos); GLES20.glEnableVertexAttribArray(aUV)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 0, bgCoords)
        GLES20.glVertexAttribPointer(aUV, 2, GLES20.GL_FLOAT, false, 0, bgUv)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(aPos); GLES20.glDisableVertexAttribArray(aUV)
        GLES20.glDepthMask(true)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
    }

    private val identity = FloatArray(16).also { GLM.setIdentityM(it, 0) }

    private fun drawScene() {
        val model = FloatArray(16); val mvp = FloatArray(16)
        val hAbove = max(bp[1] - RAD, 0f)

        // blob shadow on the floor
        GLM.setIdentityM(model, 0)
        GLM.translateM(model, 0, bp[0], 0.003f, bp[2])
        val sz = RAD * (1.7f + hAbove * 2.5f)
        GLM.scaleM(model, 0, sz, 1f, sz)
        GLM.multiplyMM(mvp, 0, base, 0, model, 0)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        drawObj(quadVB, null, mvp, identity, shadowTex, false, 0.55f / (1f + 5f * hAbove), 4)
        GLES20.glDisable(GLES20.GL_BLEND)

        // ball
        val rot = FloatArray(16)
        val w = bq[0]; val x = bq[1]; val y = bq[2]; val z = bq[3]
        rot[0] = 1 - 2 * (y * y + z * z); rot[1] = 2 * (x * y + w * z); rot[2] = 2 * (x * z - w * y)
        rot[4] = 2 * (x * y - w * z); rot[5] = 1 - 2 * (x * x + z * z); rot[6] = 2 * (y * z + w * x)
        rot[8] = 2 * (x * z + w * y); rot[9] = 2 * (y * z - w * x); rot[10] = 1 - 2 * (x * x + y * y)
        rot[15] = 1f
        val t = FloatArray(16)
        GLM.setIdentityM(t, 0); GLM.translateM(t, 0, bp[0], bp[1], bp[2])
        GLM.multiplyMM(model, 0, t, 0, rot, 0)
        GLM.multiplyMM(mvp, 0, base, 0, model, 0)
        drawObj(ballVB, ballIB, mvp, rot, ballTex, true, 1f, ballIdxCount)
    }

    private fun drawObj(
        vb: FloatBuffer, ib: ShortBuffer?, mvp: FloatArray, rot: FloatArray,
        tex: Int, lit: Boolean, alpha: Float, count: Int
    ) {
        GLES20.glUseProgram(objProg)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(objProg, "uTex"), 0)
        GLES20.glUniform1f(GLES20.glGetUniformLocation(objProg, "uLit"), if (lit) 1f else 0f)
        GLES20.glUniform1f(GLES20.glGetUniformLocation(objProg, "uAlpha"), alpha)
        GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(objProg, "uMVP"), 1, false, mvp, 0)
        GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(objProg, "uRot"), 1, false, rot, 0)
        val aPos = GLES20.glGetAttribLocation(objProg, "aPos")
        val aNorm = GLES20.glGetAttribLocation(objProg, "aNorm")
        val aUV = GLES20.glGetAttribLocation(objProg, "aUV")
        vb.position(0)
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 3, GLES20.GL_FLOAT, false, 32, vb)
        vb.position(3)
        GLES20.glEnableVertexAttribArray(aNorm)
        GLES20.glVertexAttribPointer(aNorm, 3, GLES20.GL_FLOAT, false, 32, vb)
        vb.position(6)
        GLES20.glEnableVertexAttribArray(aUV)
        GLES20.glVertexAttribPointer(aUV, 2, GLES20.GL_FLOAT, false, 32, vb)
        if (ib != null) {
            ib.position(0)
            GLES20.glDrawElements(GLES20.GL_TRIANGLES, count, GLES20.GL_UNSIGNED_SHORT, ib)
        } else {
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, count)
        }
        GLES20.glDisableVertexAttribArray(aPos)
        GLES20.glDisableVertexAttribArray(aNorm)
        GLES20.glDisableVertexAttribArray(aUV)
    }

    // =====================================================================
    // Resources
    private fun buildBall() {
        val stacks = 24; val slices = 32
        val v = ArrayList<Float>()
        for (i in 0..stacks) {
            val th = i * PI.toFloat() / stacks
            for (j in 0..slices) {
                val ph = j * 2f * PI.toFloat() / slices
                val nx = sin(th) * cos(ph); val ny = cos(th); val nz = sin(th) * sin(ph)
                v.add(nx * RAD); v.add(ny * RAD); v.add(nz * RAD)
                v.add(nx); v.add(ny); v.add(nz)
                v.add(j.toFloat() / slices); v.add(i.toFloat() / stacks)
            }
        }
        val idx = ArrayList<Short>()
        for (i in 0 until stacks) for (j in 0 until slices) {
            val a = (i * (slices + 1) + j).toShort()
            val b = (a + slices + 1).toShort()
            idx.add(a); idx.add(b); idx.add((a + 1).toShort())
            idx.add((a + 1).toShort()); idx.add(b); idx.add((b + 1).toShort())
        }
        ballVB = fb(v.toFloatArray())
        ballIB = ByteBuffer.allocateDirect(idx.size * 2).order(ByteOrder.nativeOrder()).asShortBuffer()
            .apply { put(idx.toShortArray()); position(0) }
        ballIdxCount = idx.size
    }

    private fun ballBitmap(): Bitmap {
        val bmp = Bitmap.createBitmap(1024, 512, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        c.drawColor(Color.rgb(245, 245, 242))
        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(23, 23, 27) }
        val rows = arrayOf(0.16f to 6, 0.5f to 6, 0.84f to 6)
        rows.forEachIndexed { i, (fy, n) ->
            val cy = fy * 512f
            val lat = (fy - 0.5f) * PI.toFloat()
            val sx = 1f / max(0.45f, cos(lat))
            for (k in 0 until n) {
                val cx = ((k + (i % 2) * 0.5f) / n) * 1024f
                val path = Path()
                for (j in 0 until 5) {
                    val a = (-PI / 2 + j * 2 * PI / 5).toFloat()
                    val x = cx + cos(a) * 46f * sx; val y = cy + sin(a) * 46f
                    if (j == 0) path.moveTo(x, y) else path.lineTo(x, y)
                }
                path.close(); c.drawPath(path, p)
            }
        }
        return bmp
    }

    private fun shadowBitmap(): Bitmap {
        val bmp = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            shader = RadialGradient(
                64f, 64f, 64f, intArrayOf(0xFF000000.toInt(), 0x00000000),
                floatArrayOf(0.15f, 1f), Shader.TileMode.CLAMP
            )
        }
        Canvas(bmp).drawCircle(64f, 64f, 64f, p)
        return bmp
    }

    private fun makeTex(bmp: Bitmap): Int {
        val t = IntArray(1)
        GLES20.glGenTextures(1, t, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, t[0])
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR_MIPMAP_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_REPEAT)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)
        GLES20.glGenerateMipmap(GLES20.GL_TEXTURE_2D)
        bmp.recycle()
        return t[0]
    }

    private fun fb(a: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            .apply { put(a); position(0) }

    private fun program(vs: String, fs: String): Int {
        fun sh(type: Int, src: String): Int {
            val s = GLES20.glCreateShader(type)
            GLES20.glShaderSource(s, src); GLES20.glCompileShader(s)
            return s
        }
        val p = GLES20.glCreateProgram()
        GLES20.glAttachShader(p, sh(GLES20.GL_VERTEX_SHADER, vs))
        GLES20.glAttachShader(p, sh(GLES20.GL_FRAGMENT_SHADER, fs))
        GLES20.glLinkProgram(p)
        return p
    }

    companion object {
        const val BG_VS = """
attribute vec4 aPos; attribute vec2 aUV; varying vec2 vUV;
void main(){ gl_Position = aPos; vUV = aUV; }"""
        const val BG_FS = """#extension GL_OES_EGL_image_external : require
precision mediump float; varying vec2 vUV; uniform samplerExternalOES uTex;
void main(){ gl_FragColor = texture2D(uTex, vUV); }"""
        const val OBJ_VS = """
uniform mat4 uMVP; uniform mat4 uRot;
attribute vec3 aPos; attribute vec3 aNorm; attribute vec2 aUV;
varying vec2 vUV; varying vec3 vN;
void main(){ gl_Position = uMVP * vec4(aPos, 1.0); vUV = aUV; vN = (uRot * vec4(aNorm, 0.0)).xyz; }"""
        const val OBJ_FS = """
precision mediump float;
uniform sampler2D uTex; uniform float uLit; uniform float uAlpha;
varying vec2 vUV; varying vec3 vN;
void main(){
  vec4 t = texture2D(uTex, vUV);
  if (uLit < 0.5) { gl_FragColor = vec4(0.0, 0.0, 0.0, t.a * uAlpha); return; }
  vec3 L = normalize(vec3(0.3, 1.0, 0.25));
  float d = max(dot(normalize(vN), L), 0.0);
  gl_FragColor = vec4(t.rgb * (0.45 + 0.65 * d), 1.0);
}"""
    }
}
