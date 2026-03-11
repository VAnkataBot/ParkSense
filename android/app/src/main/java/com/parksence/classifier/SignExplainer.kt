package com.parksence.classifier

/**
 * Human-readable explanations for each sign class.
 * Used in the result screen so users can learn what each sign means.
 */
object SignExplainer {

    data class SignInfo(
        val icon: String,
        val name: String,
        val description: String,
    )

    private val EXPLANATIONS = mapOf(
        "parking" to SignInfo(
            "🅿️", "Parking sign",
            "Blue P sign — marks where parking is allowed. Any time/day plates below it define when."
        ),
        "parking_ovrig" to SignInfo(
            "🅿️", "Parking — övrig tid",
            "Blue P with 'övrig tid' (other times) — parking is allowed during hours NOT covered by other restrictions on the same pole."
        ),
        "diagonal_parking" to SignInfo(
            "🅿️", "Diagonal parking",
            "Park at an angle (roughly 45°) to the kerb, not parallel. Usually means more cars fit in the space."
        ),
        "parallel_parking" to SignInfo(
            "🅿️", "Parallel parking",
            "Park parallel to the kerb, bumper-to-bumper with other cars."
        ),
        "no_parking" to SignInfo(
            "🚫", "No parking",
            "Parking is forbidden. You may stop briefly to drop off passengers or load/unload, but you cannot leave the vehicle."
        ),
        "no_stopping" to SignInfo(
            "⛔", "No stopping",
            "You cannot stop here at all — not even briefly. Stricter than no parking."
        ),
        "loading_zone" to SignInfo(
            "🟡", "Loading zone (Lastplats)",
            "This space is reserved for loading and unloading only. Parking is not allowed. Usually time-limited."
        ),
        "exception_plate" to SignInfo(
            "🕐", "Time/day plate",
            "Specifies when the rule above it applies. E.g. '7–19' means Mon–Fri 07:00–19:00. Times in parentheses '(11–17)' apply on Saturdays."
        ),
        "distance_plate" to SignInfo(
            "📏", "Zone extent plate",
            "Shows how far the rule above applies, e.g. '0–15 m' means from this sign up to 15 metres ahead."
        ),
        "handicap" to SignInfo(
            "♿", "Disabled parking",
            "Reserved for vehicles displaying a valid disabled parking permit (blue badge)."
        ),
        "ev_charging" to SignInfo(
            "⚡", "Electric vehicle charging",
            "Reserved for electric vehicles that are actively charging. Regular vehicles cannot park here."
        ),
        "motorcycle" to SignInfo(
            "🏍️", "Motorcycle parking",
            "Reserved for motorcycles and mopeds only."
        ),
        "truck" to SignInfo(
            "🚛", "Heavy vehicle parking",
            "Reserved for heavy vehicles (over 3.5 tonnes). Regular cars cannot park here."
        ),
        "trailer" to SignInfo(
            "🚌", "Trailer parking",
            "Reserved for vehicles towing trailers or caravans."
        ),
        "parking_disc" to SignInfo(
            "🕐", "Parking disc required",
            "You must place a parking disc (P-skiva) on your dashboard set to the next half-hour when you arrive. Available free at most Swedish petrol stations."
        ),
        "residents" to SignInfo(
            "🏠", "Residents only",
            "Reserved for local residents or permit holders (boende/tillstånd). You need a valid area permit to park here."
        ),
        "payment_info" to SignInfo(
            "💳", "Payment information",
            "Shows how to pay for parking — usually refers to the Stockholm parking app or parkering.stockholm.se. No physical meter."
        ),
        "arrow_plate" to SignInfo(
            "➡️", "Direction arrow",
            "Shows which direction the rule above applies — left, right, or both ways from this sign."
        ),
        "unknown" to SignInfo(
            "❓", "Unrecognised sign",
            "Could not determine what this sign means. Check it manually."
        ),
    )

    fun explain(signClass: String): SignInfo =
        EXPLANATIONS[signClass] ?: SignInfo("❓", signClass, "Unknown sign type.")
}
