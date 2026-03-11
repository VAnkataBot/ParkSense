package com.parksence.ocr

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * ML Kit OCR wrapper with two-pass red-text detection.
 *
 * Pass 1: standard image (normal dark text).
 * Pass 2: red-channel mask (catches Sunday/holiday times printed in red).
 * Results merged and deduped.
 */
object OcrProcessor {

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    // ── Public API ────────────────────────────────────────────────────────────

    suspend fun extractText(bitmap: Bitmap): String {
        val pass1 = runOcr(bitmap).map { it.uppercase().trim() }.filter { it.isNotEmpty() }

        val redMask = buildRedMask(bitmap)
        val pass2 = if (redMask != null) {
            runOcr(redMask).map { it.uppercase().trim() }.filter { it.isNotEmpty() }
        } else emptyList()

        redMask?.recycle()

        val combined = pass1 + pass2.filter { it !in pass1 }
        return combined.joinToString("\n").trim()
    }

    // ── Red mask ──────────────────────────────────────────────────────────────

    /**
     * Renders red pixels as black on white, returns null if not enough red.
     */
    private fun buildRedMask(src: Bitmap): Bitmap? {
        val w = src.width
        val h = src.height
        val pixels = IntArray(w * h).also { src.getPixels(it, 0, w, 0, 0, w, h) }
        val hsv = FloatArray(3)

        var redCount = 0
        val result = IntArray(w * h) { Color.WHITE }

        for (i in pixels.indices) {
            val c = pixels[i]
            Color.RGBToHSV(Color.red(c), Color.green(c), Color.blue(c), hsv)
            if (isRed(hsv)) {
                result[i] = Color.BLACK
                redCount++
            }
        }

        if (redCount < 30) return null

        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.setPixels(result, 0, w, 0, 0, w, h)
        return bmp
    }

    private fun isRed(hsv: FloatArray): Boolean {
        val h = hsv[0]; val s = hsv[1]; val v = hsv[2]
        return (h <= 20f || h >= 330f) && s >= 0.39f && v >= 0.31f
    }

    // ── ML Kit wrapper ────────────────────────────────────────────────────────

    private suspend fun runOcr(bitmap: Bitmap): List<String> =
        suspendCancellableCoroutine { cont ->
            val image = InputImage.fromBitmap(bitmap, 0)
            recognizer.process(image)
                .addOnSuccessListener { result ->
                    val lines = result.textBlocks
                        .flatMap { it.lines }
                        .map { it.text }
                    cont.resume(lines)
                }
                .addOnFailureListener { cont.resumeWithException(it) }
        }
}
