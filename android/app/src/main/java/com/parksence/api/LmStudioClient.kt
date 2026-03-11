package com.parksence.api

import android.graphics.Bitmap
import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

data class ParkingResult(
    val canPark: Boolean?,
    val message: String,
    val notes: List<String>,
    val signs: List<String>,
)

object LmStudioClient {

    var serverUrl = "http://192.168.1.100:1234"

    fun analyze(bitmap: Bitmap, dayName: String, timeStr: String): ParkingResult {
        val base64 = bitmapToBase64(bitmap)
        val prompt = """
---                                                                                                                                     
  You are an expert on Swedish parking signs and traffic regulations (Trafikförordningen). You will analyze parking sign images for a     
  regular driver with NO special permits, driving a regular passenger car.                                                                
                                                                                                                                          
  Current time: ${'$'}dayName ${'$'}timeStr                                                                                                         
  Current date: ${'$'}date

  ---                                                                                                                                     
  PHASE 1 — SIGN INVENTORY (DO THIS FIRST)

  Scan the ENTIRE image systematically. List EVERY sign, plate, symbol, sticker, marking, and text you can see. Check:

  - Main regulatory signs (round signs with red borders or blue background)
  - P signs (blue square with white P)
  - Supplementary plates (rectangular plates below main signs: times, arrows, text, symbols)
  - Vehicle or permit symbols (♿, motorcycle, bus, truck, EV/charging, taxi, etc.)
  - Text plates (Boende, Tillstånd, Nyttotrafik, Avgift, Taxa, Övrig tid, Förbud, etc.)
  - Zone signs (rectangular with "ZON" or "ZON" border)
  - Parking meters, payment machines, QR codes, app stickers (EasyPark, Aimo, etc.)
  - Street name signs, address number plates
  - Arrow plates (↑ ↓ ↕) and distance plates (e.g. "0–13 m")
  - Road markings (yellow painted curbs, lines, hatching)
  - Any other visible text or symbol on or near the sign pole

  List each one individually. Do NOT group or summarize. If you find fewer than 3, look again — you likely missed something.

  ---
  PHASE 2 — APPLY RULES (in this exact order)

  STEP 1 — HARD BLOCKS (any one = can_park: false, full stop)

  Check these BEFORE anything else. If ANY applies, set can_park: false and STOP. Do NOT continue to later steps. A P sign next to any of
  these does NOT grant you permission — it means only the specified group may park.

  A) Group-restricted signs:
  Any supplementary plate showing a specific vehicle type, permit, or user group means parking is reserved exclusively for that group.
  Examples:
  - ♿ Wheelchair symbol → disabled permit holders only
  - Motorcycle/moped symbol → that vehicle type only
  - Bus/truck symbol → that vehicle type only
  - EV charging symbol → electric vehicles actively charging only
  - Taxi symbol → taxis only
  - "Boende" / "Boendeparkering" → residents with area permit only
  - "Tillstånd" / "Tillstånd krävs" → permit holders only
  - "Nyttotrafik" → commercial/service vehicles only
  - Any other text or symbol specifying a group you don't belong to

  You are a regular driver in a regular car. If the sign is for someone else, can_park = false. DONE.

  B) Stopförbud (No stopping):
  Round sign, blue background, red border, red X across it. No stopping or parking at any time (unless time plates limit when it applies).
   If active right now → can_park = false. DONE.

  C) Parkering förbjuden (No parking):
  Round sign, blue background, red border, single red diagonal stripe. No parking (but brief stops under 10 minutes for loading/unloading
  allowed). Check time plates for when it's active. If active right now → can_park = false. DONE.

  ---
  STEP 2 — TIME RULES (Swedish supplementary plate conventions)

  Read all time plates on the sign. Swedish time notation:

  ┌───────────────────────────────┬───────────────────────────────────────────────────────────────────┐
  │             Plate             │                              Meaning                              │
  ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ 7–19 (no parentheses)         │ Weekdays only (Mon–Fri, excluding Swedish public holidays)        │
  ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ (7–19) (in parentheses)       │ Saturdays only (excluding public holidays that fall on Saturday)  │
  ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ ((7–19)) (double parentheses) │ Sundays and public holidays only                                  │
  ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ 7–19 (9–15) combined          │ Weekdays 7–19 AND Saturdays 9–15                                  │
  ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ 7–19 (9–15) ((9–15))          │ Weekdays 7–19, Saturdays 9–15, Sundays/holidays 9–15              │
  ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ No time plate at all          │ Restriction applies 24/7, all days                                │
  ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ "Vardagar"                    │ Weekdays (Mon–Fri excl. public holidays) — same as no parentheses │
  ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ "Helgdagar"                   │ Sundays and Swedish public holidays                               │
  ├───────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ "Övrig tid"                   │ All times NOT covered by the other plates on the same pole        │
  └───────────────────────────────┴───────────────────────────────────────────────────────────────────┘

  Duration limits:
  - "2 tim" / "30 min" / "4 timmar" = maximum parking duration. Your car must be gone before the time expires.
  - "P-skiva" or clock symbol = you must display a parking disc on your dashboard set to your arrival time.
  - Duration + P-skiva together = set disc AND leave before time expires.

  Arrows on plates:
  - ↑ (arrow pointing up / away from you) = the restriction/permission starts here and applies ahead
  - ↓ (arrow pointing down / toward you) = the restriction/permission ends here
  - ↕ (double arrow) = you are in the middle of the zone, it continues in both directions

  Distance plates:
  - "0–50 m" or "10–30 m" = the sign applies to that distance range measured from the sign along the road. NOT a time.

  ---
  STEP 3 — P SIGN CONDITIONS (parking allowed, but check the terms)

  A blue square P sign means parking IS allowed. But supplementary plates define the conditions:

  - Avgift = fee required. You must pay (meter, app, or machine).
  - Taxa [number] = hourly rate in SEK. e.g. "Taxa 22" = 22 SEK/hour.
  - P-skiva = parking disc required (set to arrival time).
  - Time limit (e.g. "2 tim") = max duration.
  - "Fri" = free parking (no fee during this period).
  - If multiple conditions: ALL apply simultaneously.

  ---
  STEP 4 — DATUMPARKERING (date-based alternate side parking)

  Only applies if you see a datumparkering zone sign or if no other signs are posted in a datumparkering zone.

  - Odd dates (1st, 3rd, 5th...) → park on the odd-numbered side of the street
  - Even dates (2nd, 4th, 6th...) → park on the even-numbered side
  - You must move your car before 00:00 when the date changes
  - Explicit P signs or no-parking signs OVERRIDE datumparkering
  - Look for "Datumparkering" zone signs at area entry points

  ---
  STEP 5 — DEFAULT RULES (when no signs are posted)

  If there are truly no parking signs at all:
  - You may park on the right-hand side of the road
  - Maximum 24 consecutive hours on weekdays (the clock resets on weekends/holidays)
  - Must be at least 10 meters from intersections, pedestrian crossings, and cycle crossings
  - Must be at least 30 meters from a railway crossing
  - Do not park on or block: sidewalks, cycle lanes, bus stops, taxi stands, driveways, hatched road markings
  - Yellow painted curb = no parking/stopping regardless of signs

  ---
  STEP 6 — ZONE SIGNS

  A sign with "ZON" text or a zone border means the rule applies to the entire area until you pass a zone-end sign. You don't need to see
  a sign on every street — the zone rule carries.

  Common zones:
  - Datumparkering zone
  - P-förbjuden zone (no parking zone)
  - Hastighetsbegränsning zone (speed, but also implies urban parking rules)

  ---
  PHASE 3 — OUTPUT

  Determine: based on the current day, date, and time, can this regular driver park here right now?

  Reply with ONLY this JSON — no other text:

  {"can_park": true/false/null, "message": "brief plain-language reason", "notes": ["fees, time limits, disc required, when the
  restriction changes next, or other practical info"], "signs": ["every sign/plate/symbol/marking listed individually - its meaning"]}

  Rules for the JSON:
  - can_park: true = yes you can park, false = no you cannot, null = image unclear or cannot determine
  - message: one or two sentences, plain language, tell the driver what to do
  - notes: practical extras — what to pay, when to leave by, when the rules change next
  - signs: EVERY visible sign/plate/symbol/marking as its own plain string entry. Completeness is mandatory. A missing sign means a wrong
  answer.

  ---


        """.trimIndent()

        val body = JSONObject().apply {
            put("model", "")   // LM Studio uses whichever model is loaded
            put("temperature", 0.1)
            put("max_tokens", 400)
            put("messages", JSONArray().apply {
                put(JSONObject().apply {
                    put("role", "user")
                    put("content", JSONArray().apply {
                        put(JSONObject().apply {
                            put("type", "image_url")
                            put("image_url", JSONObject().put("url", "data:image/jpeg;base64,$base64"))
                        })
                        put(JSONObject().apply {
                            put("type", "text")
                            put("text", prompt)
                        })
                    })
                })
            })
        }

        val conn = (URL("$serverUrl/v1/chat/completions").openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            setRequestProperty("Content-Type", "application/json")
            doOutput = true
            connectTimeout = 15_000
            readTimeout   = 90_000
        }

        OutputStreamWriter(conn.outputStream).use { it.write(body.toString()) }
        val response = conn.inputStream.bufferedReader().readText()
        conn.disconnect()

        return parseResponse(response)
    }

    private fun parseResponse(raw: String): ParkingResult {
        val content = JSONObject(raw)
            .getJSONArray("choices")
            .getJSONObject(0)
            .getJSONObject("message")
            .getString("content")
            .trim()

        // Extract the JSON block even if LLM adds surrounding text
        val start = content.indexOf('{')
        val end   = content.lastIndexOf('}')
        val json  = if (start >= 0 && end > start) JSONObject(content.substring(start, end + 1))
                    else throw Exception("No JSON found in response:\n$content")

        val canPark = if (json.isNull("can_park")) null else json.getBoolean("can_park")

        fun jsonArrayToList(key: String) = buildList {
            json.optJSONArray(key)?.let { arr ->
                for (i in 0 until arr.length()) {
                    val item = arr.get(i)
                    when (item) {
                        is JSONObject -> {
                            // LLM returned {"text":"...", "description":"..."}
                            val text = item.optString("text", "")
                            val desc = item.optString("description", "")
                            if (text.isNotEmpty() && desc.isNotEmpty()) add("$text - $desc")
                            else if (desc.isNotEmpty()) add(desc)
                            else if (text.isNotEmpty()) add(text)
                        }
                        is String -> add(item)
                        else -> add(item.toString())
                    }
                }
            }
        }

        return ParkingResult(
            canPark = canPark,
            message = json.optString("message", ""),
            notes   = jsonArrayToList("notes"),
            signs   = jsonArrayToList("signs"),
        )
    }

    private fun bitmapToBase64(bitmap: Bitmap): String {
        val max = 1024
        val scale = minOf(1f, max.toFloat() / maxOf(bitmap.width, bitmap.height))
        val b = if (scale < 1f)
            Bitmap.createScaledBitmap(bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt(), true)
        else bitmap
        val out = ByteArrayOutputStream()
        b.compress(Bitmap.CompressFormat.JPEG, 85, out)
        if (b != bitmap) b.recycle()
        return Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP)
    }
}
