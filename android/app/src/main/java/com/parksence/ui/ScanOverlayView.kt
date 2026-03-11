package com.parksence.ui

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import android.view.animation.DecelerateInterpolator
import android.view.animation.LinearInterpolator

/**
 * Animated scanning overlay drawn on top of CameraX preview.
 * - SEARCHING: pulsing white reticle corners
 * - LOCKED: corners animate to green, fill fades in
 * - ANALYSING: green border + spinning arc
 */
class ScanOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    enum class State { SEARCHING, LOCKED, ANALYSING }

    var state: State = State.SEARCHING
        set(value) {
            if (field != value) {
                field = value
                onStateChanged()
            }
        }

    // Animated values
    private var pulse      = 0f    // 0→1→0 breathing
    private var lockAlpha  = 0f    // 0→1 for green fill
    private var spinAngle  = 0f    // 0→360 spinning arc

    private val pulseAnim = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 1500; repeatMode = ValueAnimator.REVERSE; repeatCount = ValueAnimator.INFINITE
        addUpdateListener { pulse = it.animatedValue as Float; invalidate() }
    }

    private val lockAnim = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 350; interpolator = DecelerateInterpolator()
        addUpdateListener { lockAlpha = it.animatedValue as Float; invalidate() }
    }

    private val spinAnim = ValueAnimator.ofFloat(0f, 360f).apply {
        duration = 1400; repeatCount = ValueAnimator.INFINITE; interpolator = LinearInterpolator()
        addUpdateListener { spinAngle = it.animatedValue as Float; invalidate() }
    }

    // Paints
    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 5f; strokeCap = Paint.Cap.ROUND
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val arcPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 6f; strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#4CAF50")
    }
    private val dimPaint = Paint().apply {
        color = Color.parseColor("#55000000")
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER; textSize = 42f; color = Color.WHITE
        setShadowLayer(8f, 0f, 3f, Color.parseColor("#AA000000"))
        typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    }
    private val sublabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER; textSize = 28f; color = Color.parseColor("#BBFFFFFF")
        setShadowLayer(6f, 0f, 2f, Color.parseColor("#88000000"))
    }
    private val pIndicatorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER; textSize = 52f; color = Color.WHITE
        typeface = Typeface.create("sans-serif-bold", Typeface.BOLD)
    }
    private val pBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 2.5f; color = Color.WHITE
    }
    private val pBoxFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL; color = Color.parseColor("#22FFFFFF")
    }
    private val dashedPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 1.5f; color = Color.parseColor("#44FFFFFF")
        pathEffect = DashPathEffect(floatArrayOf(8f, 8f), 0f)
    }

    init { pulseAnim.start() }

    private fun onStateChanged() {
        when (state) {
            State.SEARCHING -> {
                lockAnim.cancel(); spinAnim.cancel()
                lockAlpha = 0f
                if (!pulseAnim.isRunning) pulseAnim.start()
            }
            State.LOCKED -> {
                spinAnim.cancel()
                lockAnim.start()
            }
            State.ANALYSING -> {
                pulseAnim.cancel()
                lockAlpha = 1f
                spinAnim.start()
            }
        }
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        val cx = w / 2f

        // Portrait rectangle — matches a sign pole shape
        val rw = w * 0.38f          // 38% of screen width (narrow)
        val rh = h * 0.45f          // 45% of screen height (tall)
        val cy = h * 0.35f          // centred in upper third

        val rect = RectF(cx - rw / 2f, cy - rh / 2f, cx + rw / 2f, cy + rh / 2f)
        val len = minOf(rw, rh) * 0.16f

        // Dim everything outside the reticle area
        val pad = 20f
        canvas.drawRect(0f, 0f, w, rect.top - pad, dimPaint)
        canvas.drawRect(0f, rect.bottom + pad, w, h, dimPaint)
        canvas.drawRect(0f, rect.top - pad, rect.left - pad, rect.bottom + pad, dimPaint)
        canvas.drawRect(rect.right + pad, rect.top - pad, w, rect.bottom + pad, dimPaint)

        // ── P-sign target zone (top of reticle, matches real sign size) ──
        val pZoneBottom = rect.top + rh * 0.30f
        val pBoxW = rw * 0.55f
        val pBoxH = rh * 0.18f
        val pBoxRect = RectF(
            cx - pBoxW / 2f, rect.top + rh * 0.02f,
            cx + pBoxW / 2f, rect.top + rh * 0.02f + pBoxH
        )

        // Green fill (fades in on lock)
        if (lockAlpha > 0f) {
            fillPaint.color = Color.argb((lockAlpha * 30).toInt(), 76, 175, 80)
            canvas.drawRoundRect(rect, 12f, 12f, fillPaint)
        }

        // Corner colour: white→green based on lockAlpha
        val r = (255 + (76 - 255) * lockAlpha).toInt()
        val g = (255 + (175 - 255) * lockAlpha).toInt()
        val b = (255 + (80 - 255) * lockAlpha).toInt()
        cornerPaint.color = Color.rgb(r, g, b)

        // Pulse offset (breathing)
        val off = pulse * 4f
        val l = rect.left - off; val t = rect.top - off
        val ri = rect.right + off; val bo = rect.bottom + off

        // Draw 4 corner brackets
        drawCorner(canvas, l, t, len, 1f, 1f)
        drawCorner(canvas, ri, t, len, -1f, 1f)
        drawCorner(canvas, l, bo, len, 1f, -1f)
        drawCorner(canvas, ri, bo, len, -1f, -1f)

        // ── P indicator box at top of reticle ──
        if (state == State.SEARCHING) {
            // Dashed line separating P zone from rest
            canvas.drawLine(rect.left + 12f, pZoneBottom, rect.right - 12f, pZoneBottom, dashedPaint)
            // Small rounded box for the P
            canvas.drawRoundRect(pBoxRect, 8f, 8f, pBoxFillPaint)
            canvas.drawRoundRect(pBoxRect, 8f, 8f, pBoxPaint)
            // "P" letter
            canvas.drawText("P", pBoxRect.centerX(), pBoxRect.centerY() + 18f, pIndicatorPaint)
        } else if (state == State.LOCKED) {
            // Green P box when locked
            pBoxPaint.color = Color.parseColor("#4CAF50")
            pBoxFillPaint.color = Color.parseColor("#224CAF50")
            canvas.drawRoundRect(pBoxRect, 8f, 8f, pBoxFillPaint)
            canvas.drawRoundRect(pBoxRect, 8f, 8f, pBoxPaint)
            pIndicatorPaint.color = Color.parseColor("#4CAF50")
            canvas.drawText("P", pBoxRect.centerX(), pBoxRect.centerY() + 18f, pIndicatorPaint)
            // Reset
            pBoxPaint.color = Color.WHITE
            pBoxFillPaint.color = Color.parseColor("#22FFFFFF")
            pIndicatorPaint.color = Color.WHITE
        }

        // Spinning arc during analysis
        if (state == State.ANALYSING) {
            val arcRect = RectF(rect).apply { inset(-14f, -14f) }
            canvas.drawArc(arcRect, spinAngle, 80f, false, arcPaint)
            canvas.drawArc(arcRect, spinAngle + 180f, 80f, false, arcPaint)
        }

        // Labels
        val labelY = rect.bottom + 72f
        when (state) {
            State.SEARCHING -> {
                canvas.drawText("Align P sign at the top", cx, labelY, labelPaint)
            }
            State.LOCKED -> {
                labelPaint.color = Color.parseColor("#4CAF50")
                canvas.drawText("Sign detected", cx, labelY, labelPaint)
                canvas.drawText("Hold steady...", cx, labelY + 40f, sublabelPaint)
                labelPaint.color = Color.WHITE
            }
            State.ANALYSING -> {
                labelPaint.color = Color.parseColor("#4CAF50")
                canvas.drawText("Analysing...", cx, labelY, labelPaint)
                labelPaint.color = Color.WHITE
            }
        }
    }

    private fun drawCorner(canvas: Canvas, x: Float, y: Float, len: Float, dx: Float, dy: Float) {
        canvas.drawLine(x, y, x + dx * len, y, cornerPaint)
        canvas.drawLine(x, y, x, y + dy * len, cornerPaint)
    }
}
