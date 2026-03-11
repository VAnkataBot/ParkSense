package com.parksence

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.view.View
import android.view.animation.DecelerateInterpolator
import android.widget.EditText
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.parksence.api.LmStudioClient
import com.parksence.databinding.ActivityMainBinding
import com.parksence.detection.ColorDetector
import com.parksence.ui.ScanOverlayView
import kotlinx.coroutines.*
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    @Volatile private var isAnalysing = false
    private var lockFrameCount = 0
    private val LOCK_FRAMES_NEEDED = 12

    private val cameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else showPermissionDenied()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        cameraExecutor = Executors.newSingleThreadExecutor()

        val prefs = getSharedPreferences("settings", MODE_PRIVATE)
        LmStudioClient.serverUrl = prefs.getString("server_url", "http://192.168.1.100:1234")!!

        binding.btnScanAgain.setOnClickListener { resetToScan() }
        binding.btnSettings.setOnClickListener { showSettingsDialog() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            startCamera()
        else
            cameraPermission.launch(Manifest.permission.CAMERA)
    }

    // ── Settings ──────────────────────────────────────────────────────────────

    private fun showSettingsDialog() {
        val input = EditText(this).apply {
            setText(LmStudioClient.serverUrl)
            hint = "http://192.168.x.x:1234"
            setPadding(48, 24, 48, 24)
        }
        AlertDialog.Builder(this)
            .setTitle("LM Studio server URL")
            .setMessage("Enter your Mac's local IP and port")
            .setView(input)
            .setPositiveButton("Save") { _, _ ->
                val url = input.text.toString().trimEnd('/')
                LmStudioClient.serverUrl = url
                getSharedPreferences("settings", MODE_PRIVATE).edit()
                    .putString("server_url", url).apply()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // ── Camera ────────────────────────────────────────────────────────────────

    private fun startCamera() {
        ProcessCameraProvider.getInstance(this).also { future ->
            future.addListener({
                val provider = future.get()
                val preview = Preview.Builder().build()
                    .also { it.surfaceProvider = binding.previewView.surfaceProvider }

                imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()

                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(cameraExecutor, ::analyseFrame) }

                provider.unbindAll()
                provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA,
                    preview, imageCapture, analysis
                )
            }, ContextCompat.getMainExecutor(this))
        }
    }

    // ── Live frame analysis (Persona-style: detect → lock → auto-capture) ───

    private fun analyseFrame(proxy: ImageProxy) {
        if (isAnalysing) { proxy.close(); return }
        var hasSign = false
        try {
            var bitmap = proxy.toBitmap()
            // toBitmap() returns sensor orientation (landscape) — rotate to match display
            val rotation = proxy.imageInfo.rotationDegrees
            if (rotation != 0) {
                val matrix = Matrix()
                matrix.postRotate(rotation.toFloat())
                val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                bitmap.recycle()
                bitmap = rotated
            }
            hasSign = ColorDetector.hasSignInCenter(bitmap)
            bitmap.recycle()
        } catch (e: Exception) {
            Log.e("ParkSence", "Frame analysis failed", e)
        } finally {
            proxy.close()
        }

        runOnUiThread {
            if (hasSign) {
                lockFrameCount++
                when {
                    lockFrameCount >= LOCK_FRAMES_NEEDED && !isAnalysing -> {
                        captureAndAnalyse()
                    }
                    lockFrameCount == 4 -> {
                        buzz()
                        binding.overlay.state = ScanOverlayView.State.LOCKED
                    }
                    lockFrameCount > 4 -> binding.overlay.state = ScanOverlayView.State.LOCKED
                }
            } else {
                lockFrameCount = 0
                if (!isAnalysing) binding.overlay.state = ScanOverlayView.State.SEARCHING
            }
        }
    }

    // ── Capture + freeze frame + LLM call ────────────────────────────────────

    private fun captureAndAnalyse() {
        isAnalysing = true
        imageCapture?.takePicture(ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val fullBitmap: Bitmap
                    try {
                        var bmp = image.toBitmap()
                        // Rotate to match display orientation (same fix as analyseFrame)
                        val rotation = image.imageInfo.rotationDegrees
                        if (rotation != 0) {
                            val matrix = Matrix()
                            matrix.postRotate(rotation.toFloat())
                            val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
                            bmp.recycle()
                            bmp = rotated
                        }
                        fullBitmap = bmp
                    } catch (e: Exception) {
                        Log.e("ParkSence", "toBitmap failed", e)
                        image.close()
                        isAnalysing = false
                        binding.overlay.state = ScanOverlayView.State.SEARCHING
                        return
                    }
                    image.close()

                    // Crop to reticle area for LLM (reticle: x 31-69%, y 12.5-57.5%)
                    val w = fullBitmap.width
                    val h = fullBitmap.height
                    val cropX = (w * 0.31f).toInt()
                    val cropY = (h * 0.125f).toInt()
                    val cropW = (w * 0.38f).toInt()
                    val cropH = (h * 0.45f).toInt()
                    val croppedBitmap = Bitmap.createBitmap(fullBitmap, cropX, cropY, cropW, cropH)

                    // ── Persona-style freeze frame (full image) ──
                    showCaptureEffect(fullBitmap)

                    val now  = LocalDateTime.now()
                    val day  = now.dayOfWeek.name.lowercase().replaceFirstChar { it.uppercase() }
                    val time = now.format(DateTimeFormatter.ofPattern("HH:mm"))

                    lifecycleScope.launch(Dispatchers.IO) {
                        try {
                            // Send only the cropped sign area to the LLM
                            val result = LmStudioClient.analyze(croppedBitmap, day, time)
                            croppedBitmap.recycle()
                            withContext(Dispatchers.Main) { showResult(result) }
                        } catch (e: Exception) {
                            croppedBitmap.recycle()
                            Log.e("ParkSence", "LM Studio call failed", e)
                            withContext(Dispatchers.Main) { showError(e.message ?: "Unknown error") }
                        }
                    }
                }
                override fun onError(exc: ImageCaptureException) {
                    Log.e("ParkSence", "Capture failed", exc)
                    isAnalysing = false
                    binding.overlay.state = ScanOverlayView.State.SEARCHING
                }
            })
    }

    // ── Persona-style capture effect: flash + freeze + processing overlay ────

    private fun showCaptureEffect(bitmap: Bitmap) {
        buzz()

        // 1. Show frozen frame (the "screenshot")
        binding.capturedFrame.setImageBitmap(bitmap)
        binding.capturedFrame.visibility = View.VISIBLE
        binding.capturedFrame.scaleX = 1.04f
        binding.capturedFrame.scaleY = 1.04f
        binding.capturedFrame.alpha = 1f
        binding.capturedFrame.animate()
            .scaleX(1f).scaleY(1f)
            .setDuration(350)
            .setInterpolator(DecelerateInterpolator())
            .start()

        // 2. White flash (like a camera shutter)
        binding.flashOverlay.apply {
            visibility = View.VISIBLE
            alpha = 0.8f
            animate().alpha(0f)
                .setDuration(300)
                .withEndAction { visibility = View.GONE }
                .start()
        }

        // 3. Switch overlay to ANALYSING (spinning arcs over frozen frame)
        binding.overlay.state = ScanOverlayView.State.ANALYSING
    }

    // ── Haptics ──────────────────────────────────────────────────────────────

    private fun buzz() {
        try {
            val v = getSystemService(Vibrator::class.java) ?: return
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                v.vibrate(VibrationEffect.createPredefined(VibrationEffect.EFFECT_TICK))
            } else {
                @Suppress("DEPRECATION")
                v.vibrate(30L)
            }
        } catch (_: Exception) { }
    }

    // ── Result display ──────────────────────────────────────────────────────

    private fun showResult(result: com.parksence.api.ParkingResult) {
        // Fade out frozen frame + overlay, slide in results
        binding.overlay.animate().alpha(0f).setDuration(250).withEndAction {
            binding.overlay.visibility = View.GONE
            binding.overlay.alpha = 1f
        }.start()

        binding.capturedFrame.animate()
            .alpha(0f)
            .setDuration(300)
            .withEndAction {
                binding.capturedFrame.visibility = View.GONE
                binding.capturedFrame.setImageBitmap(null)
            }.start()

        binding.resultCard.apply {
            alpha = 0f; translationY = 80f; visibility = View.VISIBLE
            animate().alpha(1f).translationY(0f).setDuration(450)
                .setStartDelay(150)
                .setInterpolator(DecelerateInterpolator()).start()
        }

        when (result.canPark) {
            true  -> {
                binding.resultIcon.text = "\u2705"
                binding.resultTitle.text = "You can park here"
                binding.resultTitle.setTextColor(getColor(R.color.green))
            }
            false -> {
                binding.resultIcon.text = "\uD83D\uDEAB"
                binding.resultTitle.text = "You cannot park here"
                binding.resultTitle.setTextColor(getColor(R.color.red))
            }
            null  -> {
                binding.resultIcon.text = "\u26A0\uFE0F"
                binding.resultTitle.text = "Could not determine"
                binding.resultTitle.setTextColor(getColor(R.color.orange))
            }
        }

        binding.resultMessage.text = result.message
        binding.resultNotes.text   = result.notes.joinToString("\n") { "\u2139 $it" }
        binding.resultNotes.visibility = if (result.notes.isEmpty()) View.GONE else View.VISIBLE
        binding.btnScanAgain.visibility = View.VISIBLE

        // Signs the LLM saw
        binding.signsContainer.removeAllViews()
        if (result.signs.isNotEmpty()) {
            binding.signsHeader.visibility = View.VISIBLE
            result.signs.forEachIndexed { i, sign ->
                val tv = TextView(this).apply {
                    text = sign
                    textSize = 14f
                    setTextColor(0xCCFFFFFF.toInt())
                    setBackgroundResource(R.drawable.bg_notes_card)
                    setPadding(40, 28, 40, 28)
                    setLineSpacing(0f, 1.3f)
                    val lp = android.widget.LinearLayout.LayoutParams(
                        android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                        android.widget.LinearLayout.LayoutParams.WRAP_CONTENT,
                    ).apply { bottomMargin = 12 }
                    layoutParams = lp
                    alpha = 0f; translationX = 60f
                    animate().alpha(1f).translationX(0f)
                        .setStartDelay((i * 80 + 300).toLong())
                        .setDuration(300).setInterpolator(DecelerateInterpolator()).start()
                }
                binding.signsContainer.addView(tv)
            }
        }
    }

    private fun showError(msg: String) {
        isAnalysing = false
        binding.overlay.visibility = View.GONE
        binding.capturedFrame.animate()
            .alpha(0f).setDuration(200)
            .withEndAction {
                binding.capturedFrame.visibility = View.GONE
                binding.capturedFrame.setImageBitmap(null)
            }.start()
        binding.resultCard.visibility = View.VISIBLE
        binding.resultIcon.text = "\u274C"
        binding.resultTitle.text = "Connection error"
        binding.resultTitle.setTextColor(getColor(R.color.red))
        binding.resultMessage.text =
            "Could not reach LM Studio.\n\nCheck:\n\u2022 Mac and phone on same WiFi\n\u2022 LM Studio server is running\n\u2022 Tap \u2699 and set correct IP\n\n$msg"
        binding.btnScanAgain.visibility = View.VISIBLE
    }

    private fun resetToScan() {
        isAnalysing = false
        lockFrameCount = 0
        binding.overlay.alpha = 1f
        binding.overlay.state = ScanOverlayView.State.SEARCHING
        binding.overlay.visibility = View.VISIBLE
        binding.capturedFrame.visibility = View.GONE
        binding.capturedFrame.setImageBitmap(null)
        binding.resultCard.visibility = View.GONE
        binding.btnScanAgain.visibility = View.GONE
        binding.signsHeader.visibility = View.GONE
        binding.signsContainer.removeAllViews()
    }

    private fun showPermissionDenied() {
        binding.resultCard.visibility = View.VISIBLE
        binding.resultIcon.text = "\uD83D\uDCF7"
        binding.resultTitle.text = "Camera permission required"
        binding.resultMessage.text = "Please allow camera access in Settings."
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
