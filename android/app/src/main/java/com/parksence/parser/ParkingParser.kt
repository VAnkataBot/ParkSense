package com.parksence.parser

import com.parksence.classifier.SignData
import java.time.LocalDateTime

/**
 * Port of Python core/parser.py — Swedish parking sign rule interpreter.
 */
data class ParkingVerdict(
    val canPark: Boolean?,   // null = unknown
    val message: String,
    val notes: List<String>,
)

object ParkingParser {

    // ── Class sets ────────────────────────────────────────────────────────────

    private val ANCHOR_ALLOW    = setOf("parking", "parking_ovrig", "diagonal_parking", "parallel_parking")
    private val ANCHOR_PROHIBIT = setOf("no_parking", "no_stopping")
    private val ANCHOR_LOADING  = setOf("loading_zone")
    private val ANCHOR_CLASSES  = ANCHOR_ALLOW + ANCHOR_PROHIBIT + ANCHOR_LOADING

    private val MODIFIER_NOTES = mapOf(
        "handicap"     to "Disabled permit holders only.",
        "ev_charging"  to "Electric vehicles (with charging) only.",
        "truck"        to "Heavy vehicles (>3.5 t) only.",
        "motorcycle"   to "Motorcycles / mopeds only.",
        "trailer"      to "Trailers only.",
        "parking_disc" to "Parking disc required — set to next half-hour on arrival.",
        "residents"    to "Residents/permit holders only.",
    )

    // ── Day map ───────────────────────────────────────────────────────────────

    private val DAY_MAP = mapOf(
        "MÅN" to 0, "MON" to 0,
        "TIS" to 1, "TUE" to 1,
        "ONS" to 2, "WED" to 2,
        "TOR" to 3, "TORS" to 3, "THU" to 3,
        "FRE" to 4, "FRED" to 4, "FRI" to 4,
        "LÖR" to 5, "SAT" to 5,
        "SÖN" to 6, "SÖ" to 6,  "SUN" to 6,
    )
    private val DAY_NAMES = mapOf(0 to "Mån", 1 to "Tis", 2 to "Ons", 3 to "Tor", 4 to "Fre", 5 to "Lör", 6 to "Sön")

    // ── Regex patterns ────────────────────────────────────────────────────────

    private val TIME_RE  = Regex("""(\d{1,2}[.:,]?\d{0,2})\s*[-–]\s*(\d{1,2}[.:,]?\d{0,2})""")
    private val DAY_PAT  = DAY_MAP.keys.sortedByDescending { it.length }.joinToString("|") { Regex.escape(it) }
                               .let { Regex("(?:$it)") }
    private val DIST_RE  = Regex("""0\s*[-–]\s*(\d+)\s*M+\b""")

    // ── Text helpers ──────────────────────────────────────────────────────────

    private fun norm(text: String): String {
        var t = text.uppercase()
        t = t.replace(Regex("[|\\\\]"), "")
        t = t.replace(Regex("[~_]"), "-")
        t = t.replace(Regex("[^\\wÅÄÖåäö\\s:()\\-–./,]"), "")
        return t.replace(Regex("\\s+"), " ").trim()
    }

    private fun parseMins(s: String): Int {
        val clean = s.replace('.', ':').replace(',', ':')
        val parts = clean.split(':')
        return parts[0].toInt() * 60 + (parts.getOrNull(1)?.toIntOrNull() ?: 0)
    }

    private fun fmt(m: Int) = "%02d:%02d".format(m / 60, m % 60)

    private fun stripDayTimes(text: String) =
        Regex("""\b${DAY_PAT.pattern}\b\s*\d{1,2}[.:,]?\d{0,2}\s*[-–]\s*\d{1,2}[.:,]?\d{0,2}""")
            .replace(text, "")

    private fun weekdayTime(text: String): Pair<Int, Int>? {
        var t = DIST_RE.replace(text, "")
        t = Regex("""\([^)]*\)""").replace(t, "")
        t = stripDayTimes(t)
        val m = TIME_RE.find(t) ?: return null
        return parseMins(m.groupValues[1]) to parseMins(m.groupValues[2])
    }

    private fun saturdayTime(text: String): Pair<Int, Int>? {
        val m = Regex("""\(\s*${TIME_RE.pattern}\s*\)""").find(text) ?: return null
        return parseMins(m.groupValues[1]) to parseMins(m.groupValues[2])
    }

    private fun singleDay(text: String): Int? {
        for ((name, idx) in DAY_MAP) {
            if (Regex("""\b$name\b""").containsMatchIn(text)) return idx
        }
        return null
    }

    private fun zoneMetres(text: String) =
        DIST_RE.find(text)?.groupValues?.get(1)?.toIntOrNull()

    private fun isOvrigTid(text: String) = "ÖVRIG TID" in text || "OVRIG TID" in text

    // ── Interval helpers ──────────────────────────────────────────────────────

    private fun intervalsForAnchor(modText: String, weekday: Int, anchorCls: String): List<Pair<Int, Int>> {
        val day = singleDay(modText)
        if (day != null) {
            val m = Regex("""\b${DAY_PAT.pattern}\b\s*${TIME_RE.pattern}""").find(modText)
            if (m != null && weekday == day && anchorCls !in ANCHOR_ALLOW) {
                return listOf(parseMins(m.groupValues[1]) to parseMins(m.groupValues[2]))
            }
        }
        val tWd  = weekdayTime(modText)
        val tSat = saturdayTime(modText)
        if (tWd == null && tSat == null) return emptyList()
        if (weekday == 6) return emptyList()
        if (weekday == 5) return if (tSat != null) listOf(tSat) else emptyList()
        return if (tWd != null) listOf(tWd) else emptyList()
    }

    private fun dayProhibitionIntervals(modText: String, weekday: Int): List<Pair<Int, Int>> {
        val day = singleDay(modText) ?: return emptyList()
        if (day != weekday) return emptyList()
        val m = Regex("""\b${DAY_PAT.pattern}\b\s*${TIME_RE.pattern}""").find(modText) ?: return emptyList()
        return listOf(parseMins(m.groupValues[1]) to parseMins(m.groupValues[2]))
    }

    private fun inAny(intervals: List<Pair<Int, Int>>, minutes: Int) =
        intervals.any { (s, e) -> minutes in s until e }

    private fun complement(intervals: List<Pair<Int, Int>>): List<Pair<Int, Int>> {
        if (intervals.isEmpty()) return emptyList()
        val merged = intervals.sortedBy { it.first }
        val result = mutableListOf<Pair<Int, Int>>()
        var cursor = 0
        for ((s, e) in merged) {
            if (cursor < s) result.add(cursor to s)
            cursor = maxOf(cursor, e)
        }
        if (cursor < 1440) result.add(cursor to 1440)
        return result
    }

    // ── Grouping ──────────────────────────────────────────────────────────────

    private data class AnchorGroup(
        val anchorCls: String,
        val anchorText: String,
        val modText: String,
        val mods: List<SignData>,
    )

    private fun groupSigns(signs: List<SignData>): List<AnchorGroup> {
        val groups = mutableListOf<AnchorGroup>()
        val currentMods = mutableListOf<SignData>()

        for (sign in signs) {
            val cls = sign.signClass
            val text = norm(sign.text)
            if (cls in ANCHOR_CLASSES) {
                groups.add(AnchorGroup(
                    anchorCls  = cls,
                    anchorText = text,
                    modText    = currentMods.joinToString(" ") { norm(it.text) },
                    mods       = currentMods.toList(),
                ))
                currentMods.clear()
            } else {
                currentMods.add(sign)
            }
        }
        if (currentMods.isNotEmpty() && groups.isNotEmpty()) {
            val last = groups.last()
            groups[groups.lastIndex] = last.copy(
                modText = (last.modText + " " + currentMods.joinToString(" ") { norm(it.text) }).trim(),
                mods    = last.mods + currentMods,
            )
        }
        return groups
    }

    private fun buildNotes(group: AnchorGroup): List<String> {
        val notes = mutableListOf<String>()
        val combined = group.anchorText + " " + group.modText

        if (group.anchorCls in MODIFIER_NOTES) MODIFIER_NOTES[group.anchorCls]?.let { notes.add(it) }
        for (mod in group.mods) {
            val n = MODIFIER_NOTES[mod.signClass]
            if (n != null && n !in notes) notes.add(n)
        }
        zoneMetres(combined)?.let { notes.add("Reserved zone: 0–$it m.") }
        if ("AVGIFT" in combined || "TAXA" in combined) notes.add("Paid parking — check meter/app for tariff.")
        if ("PARKERING.STOCKHOLM" in combined || "BETALA DIGITALT" in combined)
            notes.add("Pay digitally — parkering.stockholm.se or Stockholm parking app.")
        if ("BOENDE" in combined || "TILLSTÅND" in combined || "TILLSTAND" in combined) {
            val day = singleDay(combined)
            notes.add(if (day == 6) "Sunday: residents/permit holders only." else "Residents/permit holders only.")
        }
        if ("P-SKIVA" in combined || "PSKIVA" in combined) notes.add("Parking disc required.")
        return notes
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Parse a list of signs (bottom-to-top order) into a parking verdict.
     */
    fun parse(signs: List<SignData>, now: LocalDateTime): ParkingVerdict {
        if (signs.isEmpty()) return ParkingVerdict(null, "No signs detected.", emptyList())

        val weekday = now.dayOfWeek.value - 1   // Mon=0 … Sun=6
        val minutes = now.hour * 60 + now.minute
        var groups  = groupSigns(signs)

        if (groups.isEmpty()) {
            val vehicleMods  = setOf("handicap", "ev_charging", "motorcycle", "truck", "trailer", "residents", "parking_disc")
            val modifierOnly = setOf("exception_plate", "distance_plate", "arrow_plate",
                                     "payment_info", "parking_disc", "unknown")
            when {
                signs.any { it.signClass in vehicleMods } || signs.any { it.signClass in modifierOnly } -> {
                    // Exception/modifier plates always belong to a parking anchor — infer one.
                    // This handles merged OCR crops where the P sign text was mixed with sub-plate text.
                    groups = listOf(AnchorGroup(
                        anchorCls  = "parking",
                        anchorText = "",
                        modText    = signs.joinToString(" ") { norm(it.text) },
                        mods       = signs,
                    ))
                }
                else -> return ParkingVerdict(null, "No anchor sign found.", emptyList())
            }
        }

        // Collect restricted intervals (for övrig tid complement)
        val allRestrictedIntervals = mutableListOf<Pair<Int, Int>>()
        val evaluated = groups.map { g ->
            val ovrig = g.anchorCls == "parking_ovrig" || isOvrigTid(g.anchorText + " " + g.modText)
            val intervals = intervalsForAnchor(g.modText, weekday, g.anchorCls)
            if (!ovrig && g.anchorCls in (ANCHOR_PROHIBIT + ANCHOR_LOADING)) {
                if (intervals.isNotEmpty()) allRestrictedIntervals.addAll(intervals)
                else allRestrictedIntervals.add(0 to 1440)
            }
            object {
                val cls = g.anchorCls; val modText = g.modText; val notes = buildNotes(g)
                val ovrig = ovrig; val intervals = intervals; val anchorText = g.anchorText
            }
        }

        var verdict: Boolean? = null
        var verdictMsg = ""
        val allNotes = mutableListOf<String>()

        for (g in evaluated) {
            allNotes.addAll(g.notes)

            if (g.ovrig && g.cls in ANCHOR_ALLOW) {
                val freeIntervals = complement(allRestrictedIntervals)
                if (allRestrictedIntervals.isEmpty()) {
                    verdict = true; verdictMsg = "Parking allowed (no restrictions)."
                } else if (inAny(freeIntervals, minutes)) {
                    verdict = true; verdictMsg = "Parking allowed — övrig tid (outside restricted hours)."
                } else {
                    verdict = false; verdictMsg = "No parking now — restriction active."
                }
                continue
            }

            if (g.cls in ANCHOR_LOADING) {
                if (g.intervals.isEmpty() || inAny(g.intervals, minutes)) {
                    verdict = false; verdictMsg = "Loading zone — no parking now."
                } else if (verdict != true) {
                    verdict = true; verdictMsg = "Outside loading zone hours — parking allowed."
                }
                continue
            }

            if (g.cls in ANCHOR_PROHIBIT) {
                if (g.intervals.isEmpty()) {
                    verdict = false; verdictMsg = "No stopping/parking at all times."
                } else if (inAny(g.intervals, minutes)) {
                    val day = singleDay(g.modText)
                    verdictMsg = if (day != null) {
                        val lbl = "${DAY_NAMES[day]} ${fmt(g.intervals[0].first)}–${fmt(g.intervals[0].second)}"
                        "No parking — street cleaning $lbl."
                    } else {
                        "Parking prohibited ${fmt(g.intervals[0].first)}–${fmt(g.intervals[0].second)}."
                    }
                    verdict = false
                } else if (verdict != true) {
                    verdict = true; verdictMsg = "Outside restricted hours — parking allowed."
                }
                continue
            }

            if (g.cls in ANCHOR_ALLOW && !g.ovrig) {
                val prohib = dayProhibitionIntervals(g.modText, weekday)
                if (prohib.isNotEmpty() && inAny(prohib, minutes)) {
                    val lbl = "${DAY_NAMES[weekday]} ${fmt(prohib[0].first)}–${fmt(prohib[0].second)}"
                    verdict = false; verdictMsg = "No parking — street cleaning $lbl."
                    continue
                }
                val tWd  = weekdayTime(g.modText)
                val tSat = saturdayTime(g.modText)
                when {
                    g.intervals.isEmpty() && tWd == null && tSat == null -> {
                        if (verdict == null) { verdict = true; verdictMsg = "Parking allowed (no time restriction)." }
                    }
                    inAny(g.intervals, minutes) -> {
                        if (verdict != false) {
                            val lbl = g.intervals.firstOrNull()?.let { "${fmt(it.first)}–${fmt(it.second)}" } ?: ""
                            verdict = true; verdictMsg = "Parking allowed${if (lbl.isNotEmpty()) " — $lbl" else ""}."
                        }
                    }
                    else -> {
                        if (verdict != false) { verdict = false; verdictMsg = "Outside allowed parking hours." }
                    }
                }
            }
        }

        if (verdict == null) verdictMsg = "Could not determine parking rules — check signs manually."
        return ParkingVerdict(verdict, verdictMsg, allNotes.distinct())
    }
}
