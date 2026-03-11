package com.parksence.detection

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect

/**
 * Color-based parking sign detector.
 * Port of Python core/color_detector.py — no ML, pure pixel operations.
 *
 * Detects signs by HSV color ranges:
 *   Blue   → parking / vehicle-restriction signs
 *   Red    → no-parking / no-stopping signs
 *   Yellow → loading zone (lastplats) signs
 */
data class DetectedSign(
    val rect: Rect,
    val color: String,
    val label: String,   // DINO-style label for classifier compatibility
)

object ColorDetector {

    // ── HSV thresholds (Android scale: H 0-360, S 0-1, V 0-1) ────────────────

    private val BLUE_H   = 190f..270f;  private val BLUE_S   = 0.31f..1f;  private val BLUE_V   = 0.16f..1f
    private val RED_H1   = 0f..20f;     private val RED_S    = 0.39f..1f;  private val RED_V    = 0.31f..1f
    private val RED_H2   = 330f..360f
    private val YELLOW_H = 40f..70f;    private val YELLOW_S = 0.47f..1f;  private val YELLOW_V = 0.47f..1f

    // ── ROI (must match Python color_detector.py) ─────────────────────────────

    private const val ROI_X_START = 0.28f
    private const val ROI_X_END   = 0.72f
    private const val ROI_Y_END   = 0.72f

    private const val MIN_AREA_FRAC = 0.002f
    private const val MAX_ASPECT    = 3.0f
    private const val MIN_FILL      = 0.25f   // min fraction of bounding box that is colored

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Detect parking signs in [bitmap] by color segmentation.
     * Returns list sorted bottom-first (descending rect.bottom).
     */
    fun detect(bitmap: Bitmap): List<DetectedSign> {
        // Scale down for performance (~320px wide)
        val scale = minOf(1f, 320f / bitmap.width)
        val bmp = if (scale < 1f)
            Bitmap.createScaledBitmap(bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt(), true)
        else bitmap

        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h).also { bmp.getPixels(it, 0, w, 0, 0, w, h) }

        val roiX1 = (w * ROI_X_START).toInt()
        val roiX2 = (w * ROI_X_END).toInt()
        val roiY2 = (h * ROI_Y_END).toInt()
        val roiArea = (roiX2 - roiX1) * roiY2
        val minArea = roiArea * MIN_AREA_FRAC

        val results = mutableListOf<DetectedSign>()

        for ((color, label) in listOf(
            "blue"   to "blue parking sign",
            "red"    to "parking sign",      // neutral — OCR will confirm if it's no-parking
            "yellow" to "parking sign",      // neutral — OCR will confirm if it's loading zone
        )) {
            val mask = buildMask(pixels, w, roiX1, 0, roiX2, roiY2, color)
            val boxes = findBoxes(mask, roiX2 - roiX1, roiY2, roiX1, 0, minArea.toInt())
            for (box in boxes) {
                // Scale back to original bitmap coordinates
                val orig = Rect(
                    (box.left   / scale).toInt(),
                    (box.top    / scale).toInt(),
                    (box.right  / scale).toInt(),
                    (box.bottom / scale).toInt(),
                )
                results.add(DetectedSign(orig, color, label))
            }
        }

        if (bmp != bitmap) bmp.recycle()
        return results.sortedByDescending { it.rect.bottom }
    }

    /**
     * Lightweight trigger for live camera frames.
     * Returns true if the center of the frame has enough blue pixels to suggest a sign.
     */
    /**
     * Detects a P-sign (blue square with white "P" inside) in the center of the frame.
     * Searches a narrow portrait region matching how you'd aim at a sign pole.
     */
    fun hasSignInCenter(bitmap: Bitmap): Boolean {
        val scale = minOf(1f, 200f / bitmap.width)
        val bmp = if (scale < 1f)
            Bitmap.createScaledBitmap(bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt(), true)
        else bitmap

        val sw = if (bmp.config == Bitmap.Config.HARDWARE)
            bmp.copy(Bitmap.Config.ARGB_8888, false) else bmp

        val w = sw.width
        val h = sw.height

        // Search only the top zone where the P sign sits (top ~30% of frame, center strip)
        // Matches the P indicator in the overlay reticle
        val cx1 = (w * 0.30f).toInt()
        val cx2 = (w * 0.70f).toInt()
        val cy1 = (h * 0.05f).toInt()
        val cy2 = (h * 0.35f).toInt()
        val rw = cx2 - cx1
        val rh = cy2 - cy1
        if (rw <= 0 || rh <= 0) {
            if (sw != bmp) sw.recycle()
            if (bmp != bitmap) bmp.recycle()
            return false
        }

        val pixels = IntArray(rw * rh)
        sw.getPixels(pixels, 0, rw, cx1, cy1, rw, rh)

        if (sw != bmp) sw.recycle()
        if (bmp != bitmap) bmp.recycle()

        // ── Pass 1: find bounding box of blue pixels ──
        val hsv = FloatArray(3)
        var minBx = rw; var maxBx = 0; var minBy = rh; var maxBy = 0
        var blueCount = 0

        for (y in 0 until rh) {
            for (x in 0 until rw) {
                val px = pixels[y * rw + x]
                Color.RGBToHSV(Color.red(px), Color.green(px), Color.blue(px), hsv)
                if (isBlue(hsv)) {
                    blueCount++
                    if (x < minBx) minBx = x
                    if (x > maxBx) maxBx = x
                    if (y < minBy) minBy = y
                    if (y > maxBy) maxBy = y
                }
            }
        }

        // Need a minimum cluster of blue
        if (blueCount < 30) return false

        val bw = maxBx - minBx + 1
        val bh = maxBy - minBy + 1
        if (bw < 6 || bh < 6) return false

        // The P sign itself is roughly square — but the whole pole can be tall.
        // We only care that there's a compact blue blob, not super wide (sky).
        // Reject very wide horizontal bands (sky horizon).
        val aspect = bw.toFloat() / bh
        if (aspect > 3.5f) return false   // too wide = sky band

        // Blue should fill a decent portion of its bounding box
        val boxArea = bw * bh
        if (blueCount.toFloat() / boxArea < 0.20f) return false

        // ── Pass 2: white inside the blue region (the "P" letter) ──
        var whiteCount = 0
        for (y in minBy..maxBy) {
            for (x in minBx..maxBx) {
                val px = pixels[y * rw + x]
                Color.RGBToHSV(Color.red(px), Color.green(px), Color.blue(px), hsv)
                if (hsv[1] < 0.25f && hsv[2] > 0.65f) whiteCount++
            }
        }

        // P sign has a white letter — need at least 4% white inside
        if (whiteCount.toFloat() / boxArea < 0.04f) return false

        return true
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    private fun buildMask(
        pixels: IntArray, w: Int,
        x1: Int, y1: Int, x2: Int, y2: Int,
        color: String,
    ): BooleanArray {
        val mw = x2 - x1
        val mh = y2 - y1
        val mask = BooleanArray(mw * mh)
        val hsv = FloatArray(3)
        for (py in y1 until y2) {
            for (px in x1 until x2) {
                val c = pixels[py * w + px]
                Color.RGBToHSV(Color.red(c), Color.green(c), Color.blue(c), hsv)
                val hit = when (color) {
                    "blue"   -> isBlue(hsv)
                    "red"    -> isRed(hsv)
                    "yellow" -> isYellow(hsv)
                    else     -> false
                }
                mask[(py - y1) * mw + (px - x1)] = hit
            }
        }
        return mask
    }

    private fun findBoxes(
        mask: BooleanArray, mw: Int, mh: Int,
        originX: Int, originY: Int,
        minArea: Int,
    ): List<Rect> {
        // Row and column projections — fast rectangular region finder
        val rowSums = IntArray(mh) { y -> (0 until mw).count { x -> mask[y * mw + x] } }
        val colSums = IntArray(mw) { x -> (0 until mh).count { y -> mask[y * mw + x] } }

        val rowThr = mw * 0.15f
        val colThr = mh * 0.05f

        val rowRanges = findRuns(rowSums, rowThr)
        val colRanges = findRuns(colSums, colThr)

        val boxes = mutableListOf<Rect>()
        for ((ry1, ry2) in rowRanges) {
            for ((cx1, cx2) in colRanges) {
                val area = (cx2 - cx1) * (ry2 - ry1)
                if (area < minArea) continue
                val w = cx2 - cx1
                val h = ry2 - ry1
                val aspect = h.toFloat() / maxOf(w, 1)
                if (aspect > MAX_ASPECT) {
                    // Tall region = multiple sign plates merged vertically.
                    // Split into sign-height strips (signs are roughly square).
                    val signH = (w * 1.1f).toInt().coerceAtLeast(1)
                    val nSigns = (h / signH).coerceAtLeast(1)
                    val stripH = h / nSigns
                    for (i in 0 until nSigns) {
                        val sy1 = ry1 + i * stripH
                        val sy2 = (ry1 + (i + 1) * stripH).coerceAtMost(ry2)
                        val stripArea = w * (sy2 - sy1)
                        if (stripArea >= minArea) {
                            var filled = 0
                            for (y in sy1 until sy2) for (x in cx1 until cx2) if (mask[y * mw + x]) filled++
                            if (filled.toFloat() / stripArea >= MIN_FILL)
                                boxes.add(Rect(originX + cx1, originY + sy1, originX + cx2, originY + sy2))
                        }
                    }
                    continue
                }
                // Check fill ratio within this candidate box
                var filled = 0
                for (y in ry1 until ry2) for (x in cx1 until cx2) if (mask[y * mw + x]) filled++
                if (filled.toFloat() / area < MIN_FILL) continue
                boxes.add(Rect(originX + cx1, originY + ry1, originX + cx2, originY + ry2))
            }
        }
        return mergeOverlapping(boxes)
    }

    private fun findRuns(sums: IntArray, threshold: Float): List<Pair<Int, Int>> {
        val runs = mutableListOf<Pair<Int, Int>>()
        var start = -1
        for (i in sums.indices) {
            if (sums[i] >= threshold && start == -1) start = i
            else if (sums[i] < threshold && start != -1) {
                runs.add(start to i)
                start = -1
            }
        }
        if (start != -1) runs.add(start to sums.size)
        return runs
    }

    private fun mergeOverlapping(boxes: List<Rect>): List<Rect> {
        if (boxes.size <= 1) return boxes
        val result = boxes.map { Rect(it) }.toMutableList()
        var changed = true
        while (changed) {
            changed = false
            val merged = mutableListOf<Rect>()
            val used = BooleanArray(result.size)
            for (i in result.indices) {
                if (used[i]) continue
                val r = Rect(result[i])
                for (j in i + 1 until result.size) {
                    if (used[j]) continue
                    val other = result[j]
                    val inter = Rect()
                    if (inter.setIntersect(r, other)) {
                        val interArea = inter.width() * inter.height()
                        val unionArea = r.width() * r.height() + other.width() * other.height() - interArea
                        if (unionArea > 0 && interArea.toFloat() / unionArea > 0.1f) {
                            r.union(other)
                            used[j] = true
                            changed = true
                        }
                    }
                }
                merged.add(r)
                used[i] = true
            }
            result.clear()
            result.addAll(merged)
        }
        return result
    }

    private fun isBlue(hsv: FloatArray)   = hsv[0] in BLUE_H   && hsv[1] in BLUE_S   && hsv[2] in BLUE_V
    private fun isRed(hsv: FloatArray)    = (hsv[0] in RED_H1  || hsv[0] in RED_H2)  && hsv[1] in RED_S && hsv[2] in RED_V
    private fun isYellow(hsv: FloatArray) = hsv[0] in YELLOW_H && hsv[1] in YELLOW_S && hsv[2] in YELLOW_V
}
