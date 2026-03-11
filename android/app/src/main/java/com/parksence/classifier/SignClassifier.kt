package com.parksence.classifier

/**
 * Port of Python core/classifier.py
 * Classifies a detected sign box into a semantic class using OCR text + detector label.
 */
data class SignData(
    val text: String,
    val detectorLabel: String,
    var signClass: String = "unknown",
)

object SignClassifier {

    // ── Text normalisation ────────────────────────────────────────────────────

    private fun norm(text: String): String {
        var t = text.uppercase()
        t = t.replace(Regex("[|\\\\]"), "")
        t = t.replace(Regex("[~_]"), "-")
        t = t.replace(Regex("[^\\wÅÄÖåäö\\s:()\\-–./,]"), "")
        return t.replace(Regex("\\s+"), " ").trim()
    }

    // ── OCR text rules ────────────────────────────────────────────────────────
    // Each entry: patterns (all must match) → class. First match wins.

    private data class OcrRule(val patterns: List<String>, val cls: String)

    private val OCR_RULES = listOf(
        OcrRule(listOf("STOPFÖRBUD"),                          "no_stopping"),
        OcrRule(listOf("STOP FÖRBUD"),                         "no_stopping"),
        OcrRule(listOf("PARKERING FÖRBJUDEN"),                 "no_parking"),
        OcrRule(listOf("PARKERINGSFÖRBUD"),                    "no_parking"),
        OcrRule(listOf("LASTPLATS"),                           "loading_zone"),
        OcrRule(listOf("LAST PLATS"),                          "loading_zone"),
        OcrRule(listOf("ÖVRIG TID"),                           "parking_ovrig"),
        OcrRule(listOf("OVRIG TID"),                           "parking_ovrig"),
        OcrRule(listOf("SNEDPARKERING"),                       "diagonal_parking"),
        OcrRule(listOf("PARALLELLPARKERING"),                  "parallel_parking"),
        OcrRule(listOf("PARALELLPARKERING"),                   "parallel_parking"),
        OcrRule(listOf("BOENDE"),                              "residents"),
        OcrRule(listOf("TILLSTÅND"),                           "residents"),
        OcrRule(listOf("TILLSTAND"),                           "residents"),
        OcrRule(listOf("BETALA DIGITALT"),                     "payment_info"),
        OcrRule(listOf("PARKERING.STOCKHOLM"),                 "payment_info"),
        OcrRule(listOf("P-SKIVA"),                             "parking_disc"),
        OcrRule(listOf("PSKIVA"),                              "parking_disc"),
        OcrRule(listOf("P SKIVA"),                             "parking_disc"),
        OcrRule(listOf("""0\s*[-–]\s*\d+\s*M\b"""),           "distance_plate"),
        OcrRule(listOf("TORS",  """\d+\s*[-–]\s*\d+"""),      "exception_plate"),
        OcrRule(listOf("FRED",  """\d+\s*[-–]\s*\d+"""),      "exception_plate"),
        OcrRule(listOf("MÅN",   """\d+\s*[-–]\s*\d+"""),      "exception_plate"),
        OcrRule(listOf("TIS",   """\d+\s*[-–]\s*\d+"""),      "exception_plate"),
        OcrRule(listOf("ONS",   """\d+\s*[-–]\s*\d+"""),      "exception_plate"),
        OcrRule(listOf("LÖR",   """\d+\s*[-–]\s*\d+"""),      "exception_plate"),
        OcrRule(listOf("SÖN",   """\d+\s*[-–]\s*\d+"""),      "exception_plate"),
        OcrRule(listOf("AVGIFT","""\d+\s*[-–]\s*\d+"""),      "exception_plate"),
        OcrRule(listOf("TAXA",  """\d+\s*[-–]\s*\d+"""),      "exception_plate"),
    )

    // ── Detector label → class fallback ──────────────────────────────────────
    // Ordered most-specific first. "diagonal" and "parallel" before "truck"/"trailer".

    private val LABEL_MAP = listOf(
        "no stopping"       to "no_stopping",
        "no parking"        to "no_parking",
        "loading zone"      to "loading_zone",
        "lastplats"         to "loading_zone",
        "handicap"          to "handicap",
        "wheelchair"        to "handicap",
        "electric vehicle"  to "ev_charging",
        "charging"          to "ev_charging",
        "motorcycle"        to "motorcycle",
        "diagonal"          to "diagonal_parking",
        "parallel"          to "parallel_parking",
        "truck"             to "truck",
        "trailer parking sign" to "trailer",
        "parking disc"      to "parking_disc",
        "arrow"             to "arrow_plate",
        "residents"         to "residents",
        "blue parking sign" to "parking",
        "parking sign"      to "parking",
        "parking"           to "parking",
    )

    private val VEHICLE_KEYWORDS = setOf(
        "handicap", "wheelchair", "electric vehicle", "charging",
        "motorcycle", "truck", "trailer", "residents"
    )

    // ── Public API ────────────────────────────────────────────────────────────

    fun classify(sign: SignData): String {
        val text  = norm(sign.text)
        val label = sign.detectorLabel.lowercase()

        // 1. OCR text rules
        for (rule in OCR_RULES) {
            if (rule.patterns.all { Regex(it).containsMatchIn(text) }) return rule.cls
        }

        // 2. Bare time plate
        if (Regex("""^\s*\d{1,2}\s*[-–]\s*\d{1,2}""").containsMatchIn(text) && text.length < 25)
            return "exception_plate"

        // 3. Detector label fallback
        if (("parking sign" in label || "parking" in label) && text.isNotEmpty()) {
            for (kw in VEHICLE_KEYWORDS) if (kw in label) return "parking"
        }
        for ((kw, cls) in LABEL_MAP) if (kw in label) return cls

        // 4. Any digit-dash-digit in OCR
        if (Regex("""\d+\s*[-–]\s*\d+""").containsMatchIn(text)) return "exception_plate"

        return "unknown"
    }

    fun classifyAll(signs: List<SignData>): List<SignData> {
        signs.forEach { it.signClass = classify(it) }
        return signs
    }
}
