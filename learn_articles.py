"""Content module for the /learn topic cluster.

Each article is a plain dict with slug, title, description (used for meta),
keywords (used for og:tags), and html_body (Jinja-safe HTML rendered in
templates/learn/article.html).

Articles are intentionally surfer-first but technically honest: they explain
the same physics the wave models use, with concrete numbers a forecaster
would recognize.
"""

ARTICLES = [
    {
        'slug': 'how-to-read-a-surf-forecast',
        'title': 'How to Read a Surf Forecast',
        'description': 'A practical guide to interpreting wave height, period, direction, wind, and tides — the five numbers that determine whether to paddle out.',
        'keywords': 'surf forecast, how to read, wave height, period, wind',
        'html_body': '''
<p>Every surf forecast on the internet boils down to five numbers: <strong>wave height, wave period, wave direction, wind, and tide</strong>. Knowing what each one does — and which matters most at your local break — is the difference between a wasted drive and a perfect dawn patrol.</p>

<h2>Wave height: bigger isn't always better</h2>
<p>The headline number on any forecast is <strong>significant wave height (Hs)</strong>, usually given in feet. This is the average of the largest one-third of waves measured in deep water — <em>not</em> the face height you'll see at the beach. Sets typically run 1.3–1.5× the Hs, and shoaling near the beach amplifies the face by another 1.5–2×. A forecast of 3 ft Hs might mean 5 ft faces on the bigger waves at a typical beach break.</p>
<p>What's the right size? Depends on your skill level and the spot. Most surf forecast sites — this one included — let you set a skill level so the same forecast reads "ankle to knee, longboard only" vs "head high, go time."</p>

<h2>Wave period: the most important number after height</h2>
<p>Period is the time in seconds between successive wave crests passing a fixed point. It's the cleanest single indicator of <em>quality</em>, regardless of size:</p>
<ul>
<li><strong>Under 6 s:</strong> Choppy windswell. Often breaks in random directions, lacks power.</li>
<li><strong>6–9 s:</strong> Wind swell. Surfable but disorganized.</li>
<li><strong>9–12 s:</strong> Medium-period swell. Decent shape, lined up.</li>
<li><strong>12+ s:</strong> Ground swell. Powerful, well-organized lines from a distant storm.</li>
<li><strong>15+ s:</strong> Long-period ground swell. Travels through coastal blockage; can produce overhead waves at apparently-small forecasts.</li>
</ul>
<p>For a deeper look at why period matters more than people think, see <a href="/learn/wave-period-explained">Wave Period Explained</a>.</p>

<h2>Wave direction vs beach orientation</h2>
<p>A 4 ft, 14 s NE swell is a dream — but only if the beach faces northeast. The same swell at a south-facing beach will arrive heavily refracted and produce maybe half the height. Every forecast page on this site shows the <a href="/learn/beach-orientation-and-waves">beach-facing direction</a>; pay attention to how it lines up with the forecast wave direction.</p>

<h2>Wind: offshore is gold, light cross-shore is fine</h2>
<p>Wind direction relative to the beach determines surface quality:</p>
<ul>
<li><strong>Offshore</strong> (wind blowing from land out to sea): holds up wave faces. The cleanest conditions.</li>
<li><strong>Cross-shore</strong> (parallel to the beach): adds texture but doesn't destroy shape. Often surfable, sometimes preferable when very light.</li>
<li><strong>Onshore</strong> (sea to land): destroys shape. Strong onshore = blown out.</li>
</ul>
<p>Wind speed matters too. Calm to 5 mph offshore is glassy. 5–15 mph offshore is clean. Above 25 mph offshore makes paddling out brutal. Even an "onshore" wind under 5 mph is usually fine.</p>

<h2>Tides: timing matters</h2>
<p>Most breaks favor a specific tide stage. Beach breaks often work best on the push (incoming) at mid-tide. Reef breaks can be more particular — some only work for a 2-hour window around low or high. Many spots become dangerous at dead low. The forecast tells you the size and shape; the tide tells you <em>when</em> within the day to go. See <a href="/learn/best-tide-for-surfing">Best Tide for Surfing</a>.</p>

<h2>Put it together: the call</h2>
<p>A surfable forecast typically combines:</p>
<ul>
<li>Enough size for your skill (and not too much)</li>
<li>Period <strong>9 s or longer</strong> (ideally 12+)</li>
<li>Wave direction aligned with the beach (within ~45° of beach-normal)</li>
<li>Light or offshore wind</li>
<li>A reasonable tide window</li>
</ul>
<p>This site's <strong>session score</strong> combines all five into a single 0–100 number per forecast hour, with the highest-scoring 3-hour window highlighted in the session planner. Use it as a starting point, then check the actual numbers — if any one is way off, the score won't tell the whole story.</p>
''',
    },
    {
        'slug': 'wave-period-explained',
        'title': 'Wave Period Explained',
        'description': 'Why wave period matters more than height for surf quality, what peak vs mean period mean, and how to interpret the numbers in your forecast.',
        'keywords': 'wave period, peak period, mean period, surf forecast',
        'html_body': '''
<p>Of all the numbers in a surf forecast, <strong>wave period</strong> is the one most casual surfers misread. Height grabs attention — bigger wave looks bigger — but period determines how the wave behaves: how much energy it carries, how cleanly it breaks, how predictably it lines up. A 3 ft 14 s ground swell will produce <em>better</em> surf than a 5 ft 6 s wind chop on the same beach. Period is the multiplier.</p>

<h2>What period actually measures</h2>
<p>Wave period is the time in seconds between two successive crests passing a fixed point. It's directly tied to the wavelength via the deep-water dispersion relation:</p>
<p style="margin-left: 1em;"><code>L = (g · T²) / (2π) ≈ 1.56 · T² meters</code></p>
<p>So a 10 s wave has a wavelength of ~156 m, while a 14 s wave is ~306 m — almost twice as long. Longer wavelength means each wave carries water motion deeper below the surface, all the way down to roughly half its wavelength. A 14 s swell stirs the water column ~150 m down; a 6 s wind sea barely affects anything below 18 m.</p>

<h2>Why long-period swells are more powerful</h2>
<p>When a long-period swell crosses a continental shelf, the seabed begins to interact with the wave at depths up to half the wavelength. This slows the wave, compresses it, and amplifies the height — the process called <a href="/glossary#coast">shoaling</a>. A 4 ft 14 s offshore swell can produce 6 ft faces at the beach. The same 4 ft offshore on a 6 s period barely shoals at all, because the wave's energy doesn't reach the bottom until it's already in breaking depth.</p>
<p>This is why a forecast of 3 ft / 14 s often produces visibly larger and more powerful waves at the beach than 4 ft / 7 s. The period is doing the work.</p>

<h2>Peak period vs mean period</h2>
<p>You'll see two period numbers in scientific wave data:</p>
<ul>
<li><strong>Peak period (Tp):</strong> the period of the most energetic frequency component. This is what NDBC buoys and most surf forecasts report.</li>
<li><strong>Mean period (Tm):</strong> the energy-weighted average period across all frequencies. Usually 1–2 s shorter than Tp.</li>
</ul>
<p>The distinction matters when you compare forecasts. Open-Meteo's free Marine API defaults to <em>mean</em> period in a field called <code>wave_period</code>, while NOAA's WaveWatch III publishes <em>peak</em> period. Switching sources mid-forecast can shift the apparent period by 1–2 s without anything actually changing — moving you across the 9 s and 12 s classification boundaries that separate wind swell from ground swell. This site standardizes on Tp across all sources so the labels stay consistent.</p>

<h2>Period thresholds for forecasting</h2>
<p>There's no hard line between swell types, but useful working thresholds:</p>
<ul>
<li><strong>≥ 15 s</strong> — long-period ground swell. Major distant storm origin. Wraps around obstructions, fires deep-water spots.</li>
<li><strong>12–15 s</strong> — classic ground swell. Most weekend "perfect" forecasts.</li>
<li><strong>9–12 s</strong> — medium-period mixed. Surfable, decent shape, less power.</li>
<li><strong>6–9 s</strong> — wind swell. Generated within ~24 hours of arrival.</li>
<li><strong>&lt; 6 s</strong> — local chop. Disorganized.</li>
</ul>

<h2>Period plus steepness: a fuller picture</h2>
<p>Period alone doesn't perfectly distinguish "decayed remote swell" from "locally-generated big sea." Wave <strong>steepness</strong> — the ratio of height to wavelength <code>s = 2π · Hs / (g · T²)</code> — does a better job:</p>
<ul>
<li><strong>s &lt; 0.008</strong> — well-decayed, clean ground swell from a distant fetch.</li>
<li><strong>0.008–0.025</strong> — mixed seas.</li>
<li><strong>s &gt; 0.025</strong> — locally-generated wind sea.</li>
</ul>
<p>This site computes both period-based and steepness-based classifications and flags when they disagree — usually a sign of mixed seas or model uncertainty.</p>

<h2>Practical use</h2>
<p>Next time you check a forecast, glance at the period first. If it's 12 s or higher and the wind is light, drive to the beach even if the height looks small. If it's 6 s and the height is "big," expect chop. Period is the single best indicator of whether the surf will be worth your time.</p>
''',
    },
    {
        'slug': 'what-is-a-swell-window',
        'title': 'What Is a Swell Window?',
        'description': 'The range of directions from which open-ocean swells can reach a given break — how it shapes which storms produce surf and which don\'t.',
        'keywords': 'swell window, swell direction, beach orientation',
        'html_body': '''
<p>A surfer who can read a weather chart starts to notice something: the same hurricane that lights up one stretch of coast leaves another flat. The same Pacific storm that fires the North Shore does nothing in Newport Beach. The reason is the <strong>swell window</strong> — the range of open-ocean directions from which swells can actually reach a given beach.</p>

<h2>The geometry</h2>
<p>Stand on the beach, face the ocean. The swell window is the arc of compass bearings on the horizon that aren't blocked by land, islands, headlands, or shallow banks. A swell coming from a direction inside that window has a clear path to your break. A swell from outside it has to wrap, diffract, or — usually — never arrives at all.</p>
<p>For Surf City, North Carolina (E-facing, ~90°), the swell window runs roughly from NNE (~22°) around through E and on to SSE (~157°). Swells from those directions arrive head-on or close to it. A swell from the south (180°) is mostly blocked by Cape Fear; a swell from the west (270°) is obviously blocked by the continent itself.</p>

<h2>Why it matters</h2>
<p>Knowing the swell window tells you which weather systems matter:</p>
<ul>
<li>For East Coast spots: Atlantic hurricanes tracking up the seaboard generate E and ESE swells — squarely in window for most US East Coast beaches. Nor'easters generate NE and N swells — in window for most spots north of Cape Hatteras.</li>
<li>For Southern California: North Pacific winter storms generate NW swells, which are in window for almost the entire CA coast. Southern hemisphere summer swells come from the SSW — in window only for breaks with a clear south-facing aspect.</li>
<li>For Hawaii's North Shore: only NNW–NE swells reach. The South Shore opens to S and SW.</li>
</ul>

<h2>Refraction extends the window — partially</h2>
<p>Swells don't stop dead at the window edge. <strong>Refraction</strong> bends waves as they enter shallow water, turning them toward shore-normal. A swell arriving at 30° off your beach-normal will lose maybe half its energy to refraction but still produce surfable lines. A swell arriving at 75° loses most of its energy. Beyond ~100°, it's effectively blocked.</p>
<p>Long-period ground swells refract more aggressively than short-period wind swells, because the bottom interacts with longer waves at greater depths. A 14 s SE swell can wrap noticeably into an E-facing beach; the same 14 s swell wrapping into a N-facing beach loses too much energy to amount to anything.</p>

<h2>Diffraction and shadows</h2>
<p>Around the edges of headlands and islands, waves <strong>diffract</strong> — spreading into the geometric shadow. This is why spots tucked behind a point can pick up swell that "shouldn't" reach them, just smaller and cleaner than the unsheltered coastline next door. It's also why you sometimes see surfers riding on a day the open-ocean forecast says is flat.</p>

<h2>Mapping your spot's window</h2>
<p>This site's <strong>ocean basin map</strong> on every forecast page shows the open-ocean wave field around your location. You can visually see where the energy is coming from and whether it has a clear path to your beach. Combined with the beach-facing direction (shown numerically on the page), it answers the question: "Can this swell actually reach me?"</p>

<h2>Practical use</h2>
<ul>
<li>Look up your beach's facing direction. (On this site, it's shown directly in the page header.)</li>
<li>The swell window is roughly ±90° from beach-normal, narrower if you're behind a cape or island.</li>
<li>A forecast wave direction within ±30° of beach-normal will arrive almost full-strength.</li>
<li>Within ±75°, you'll still get surfable wrap but expect 50% energy loss.</li>
<li>Beyond ±100°, the swell is effectively blocked regardless of size.</li>
</ul>
<p>The session score on this site automatically applies a swell-exposure multiplier based on this geometry — so the same NE groundswell will score high at Surf City and noticeably lower at SE-facing Wrightsville Beach.</p>
''',
    },
    {
        'slug': 'best-tide-for-surfing',
        'title': 'Best Tide for Surfing',
        'description': 'Why tides matter, how to read tide-stage timing, and which tides typically work for different break types.',
        'keywords': 'tide for surfing, low tide, high tide, mid tide',
        'html_body': '''
<p>Two surfers stand on the same beach on the same day, looking at the same waves. One paddles out and scores; the other waits two hours and gets shoulder-high perfection. The difference is tide. Most breaks have a preferred tide stage, and getting it wrong can turn a 4 ft swell into ankle slop or close-outs.</p>

<h2>What the tide actually does</h2>
<p>Tide is the periodic rise and fall of sea level driven by the moon and sun's gravity. On most US coasts, you get two highs and two lows per day (semi-diurnal), with a tide range that varies from ~1 ft (East Coast in summer) to ~10+ ft (Pacific Northwest, Maine). That changing water depth interacts with the sandbar or reef where waves break, shaping their size, shape, and break point.</p>

<h2>Why tide stage matters</h2>
<p>Waves break when they reach a depth roughly equal to their face height — a 4 ft face wave breaks in ~5 ft of water. When the tide is high, the same offshore swell arrives over deeper water and either doesn't break at all, breaks weakly, or breaks closer to shore over a different bar. When the tide is low, it breaks earlier and faster over the bar, sometimes faster than a surfer can ride.</p>

<h2>How break type interacts with tide</h2>

<h3>Beach breaks</h3>
<p>Most beach breaks favor <strong>mid tide</strong>, either pushing (incoming) or dropping (outgoing). Mid tide gives you enough depth that the wave doesn't break too sharply, and enough sandbar exposure that it actually shapes up. Dead low can produce close-outs; dead high often means the waves just roll past unbroken.</p>
<p>The push (incoming) is often slightly preferred — moving water energizes the sandbar and keeps things organized.</p>

<h3>Reef breaks</h3>
<p>Reef breaks are more particular. The fixed bottom contour means there's typically a 2–3 hour window where the depth is just right. Some reefs only work on a draining tide; some only on a rising one. Knowing your local reef's preferred window is critical — outside it, the wave either doesn't break or breaks dangerously shallow.</p>

<h3>Point breaks</h3>
<p>Point breaks generally favor lower tides. The point's bottom contour is exposed enough for waves to wrap and break consistently. At high tide, waves often just roll past the point without breaking.</p>

<h3>Pier and jetty breaks</h3>
<p>Sandbar deposits next to piers and jetties tend to be steeper than open beach. Mid to low tide is usually best — high tide submerges the sandbar enough that waves don't focus.</p>

<h2>Slack tide and moving water</h2>
<p>At the top of high tide and the bottom of low tide, the water momentarily stops moving — <strong>slack tide</strong>. Many breaks lose definition during slack: the lack of current lets sand drift and waves roll past without focusing. Moving tide (within an hour of the midpoint between high and low) is generally more productive than slack.</p>

<h2>Spring vs neap</h2>
<p>Around full and new moons, the sun and moon align and tides become <strong>spring</strong> tides — bigger range, stronger currents, more dramatic differences between high and low. Around quarter moons, you get <strong>neap</strong> tides — smaller range, less variation. Spring tides amplify whatever the break's tide preference is; neaps soften it. A spot that's "low-tide only" might work for hours on a spring low and only briefly on a neap.</p>

<h2>Practical use</h2>
<ul>
<li>Check your forecast's tide times before you leave. If the next high is 4 hours away and your spot wants mid-low, plan accordingly.</li>
<li>Look at the tide curve, not just the times. A "rising mid tide" 3 hours after low is a different animal than "approaching high" 1 hour before high — both might be ~50% of the range.</li>
<li>If you're new to a spot, ask a local what tide works best. Reef breaks especially.</li>
<li>This site's session score factors in tide movement when local tide-stage preference isn't curated, and tide-stage matching when it is — checked the tide overlay on the wave chart to see where the highs and lows fall during your window.</li>
</ul>
''',
    },
    {
        'slug': 'beach-orientation-and-waves',
        'title': 'How Beach Orientation Affects Waves',
        'description': 'Why which direction your beach faces is the single biggest factor in how a given swell will arrive — and how to use it when reading forecasts.',
        'keywords': 'beach orientation, wave direction, refraction',
        'html_body': '''
<p>Beaches aren't all the same. Some face dead east, some face south, some face every direction in between depending on how the coastline curves. <strong>Beach orientation</strong> — the compass direction perpendicular to the shoreline, pointing toward the ocean — is the single most important property of a break after its bathymetry. It determines which swells reach the beach with full energy and which arrive heavily diminished or not at all.</p>

<h2>Defining beach orientation</h2>
<p>Stand at the water's edge, facing the ocean. The direction you're facing is the beach-facing direction. A surf forecast at Surf City NC shows ~90° (east) because the beach faces east — the open Atlantic is due east. Wrightsville Beach NC, just 65 miles south, shows ~135° (southeast) because the coastline at Wrightsville curves to face south of east. Two beaches in the same state, on the same swell, can produce dramatically different surf.</p>

<h2>How orientation modulates swells</h2>
<p>The angle between the incoming swell direction and the beach-facing direction is the key:</p>
<ul>
<li><strong>0–30° off beach-normal:</strong> swell arrives almost head-on. Full energy reaches the beach. Best conditions.</li>
<li><strong>30–75° off:</strong> swell arrives at an oblique angle. Refraction bends some of the energy back toward shore-normal, but you lose ~30–50% of the height.</li>
<li><strong>75–100° off:</strong> swell arrives nearly parallel to the beach. Most of the energy passes by without breaking. Maybe 20% of forecast height reaches you.</li>
<li><strong>Beyond 100° off:</strong> swell is effectively blocked. Whatever waves you see are diffracted scraps.</li>
</ul>

<h2>The same swell, different beaches</h2>
<p>Take a 4 ft 13 s NE (45°) ground swell. At Surf City (E-facing, 90°): swell angle is 45° off beach-normal — a small ~50% energy loss. Expect 2.5–3 ft faces, clean lines. At Wrightsville (SE-facing, 135°): swell angle is 90° off — most of the energy passes by. Expect 1–1.5 ft, hardly worth getting wet.</p>
<p>That's why two locations 65 miles apart can produce totally different surf on the same forecast.</p>

<h2>How forecasts capture this</h2>
<p>Most surf forecast sites assume a generic beach orientation per region. This site computes the actual beach-facing direction from the local coastline geometry using OpenStreetMap data — so the orientation is specific to where your forecast pin is, not the average for the whole state. You can see the number on every forecast page (e.g. "ENE-facing beach (78°)") and the session score applies a refraction-loss multiplier accordingly.</p>

<h2>Aspect-dependent breaks</h2>
<p>Some spots are particularly oriented:</p>
<ul>
<li><strong>Outer Banks NC</strong> beaches face roughly E. Best on NE through SE swells. North-facing storms still produce surf via wrap.</li>
<li><strong>Virginia Beach</strong> faces ENE (~78°). Best on NE swells from nor'easters and on E swells from Atlantic hurricanes.</li>
<li><strong>Cocoa Beach FL</strong> faces ENE. Best on hurricane-driven E/ESE swells.</li>
<li><strong>Wrightsville Beach NC</strong> faces SE. Best on E/SE swells; poor on N/NE swells.</li>
<li><strong>Southern California</strong> Newport faces WSW. Best on SW summer swells and NW winter swells that can wrap deep into the bight.</li>
<li><strong>North Shore Oahu</strong> faces NNW. Best on N–NNW winter ground swells.</li>
</ul>

<h2>Cross-orientation breaks</h2>
<p>Around capes, headlands, and inside bays, the beach can face multiple directions over a short stretch. <a href="/learn/what-is-a-swell-window">Swell windows</a> can vary by 100° in a single mile of coastline. This is why some "secret spots" within a region work when neighboring beaches don't — the orientation catches a swell direction that doesn't fit the regional norm.</p>

<h2>Practical use</h2>
<p>Before any forecast check:</p>
<ul>
<li>Note your spot's beach-facing direction (degrees and cardinal).</li>
<li>Note the forecast wave direction.</li>
<li>Compute the difference. If under 30°, expect full power. 30–75°, expect a moderate drop. 75°+, expect not much.</li>
</ul>
<p>This single calculation — done by the session score automatically on this site — is often more decisive than wave height in determining whether to go.</p>
''',
    },
    {
        'slug': 'reading-buoy-reports',
        'title': 'Reading Buoy Reports',
        'description': 'NDBC and CDIP buoys are the ground truth for ocean conditions. How to read them and how to translate offshore numbers to beach face heights.',
        'keywords': 'NDBC buoy, CDIP, wave buoy, ocean observations',
        'html_body': '''
<p>Forecast models are smart but they're forecasts — predictions. <strong>Buoys</strong> are real, in the ocean, measuring waves and wind every 30 minutes. When forecasts disagree with what's actually happening, the buoy is usually right. Learning to read a buoy report is one of the highest-leverage skills a surfer can develop.</p>

<h2>The two buoy networks</h2>
<p><strong>NDBC</strong> (National Data Buoy Center, NOAA) operates roughly 100 moored wave/weather buoys along US coasts and offshore. They measure wave height, peak period, dominant direction, wind speed/direction/gust, air and water temperature, and barometric pressure. Most are offshore — 20–250 nm from the coast.</p>
<p><strong>CDIP</strong> (Coastal Data Information Program, Scripps) runs a network of nearshore wave buoys, mostly on the West Coast and Pacific. CDIP buoys are typically 10–20 nm offshore and produce higher-resolution <strong>directional wave spectra</strong> than NDBC — they tell you not just "5 ft at 12 s from the WNW" but the full distribution of energy across periods and directions.</p>

<h2>The four numbers</h2>
<p>Every buoy reports these:</p>
<ul>
<li><strong>WVHT</strong> (wave height): significant wave height in meters or feet. The average of the largest 1/3 of waves.</li>
<li><strong>DPD</strong> (dominant period): peak period in seconds. The period carrying the most energy.</li>
<li><strong>APD</strong> (average period): mean period across all frequencies. Usually 1–2 s shorter than DPD.</li>
<li><strong>MWD</strong> (mean wave direction): dominant direction the waves are coming FROM, in degrees true. 90 = from east.</li>
</ul>
<p>Plus wind: <strong>WSPD</strong> (speed, m/s), <strong>WDIR</strong> (direction from), <strong>GST</strong> (gust speed). And environmental: <strong>WTMP</strong> (water temp), <strong>ATMP</strong> (air temp), <strong>PRES</strong> (barometric pressure mb).</p>

<h2>Translating offshore to beach</h2>
<p>A buoy reads open-ocean conditions; you ride waves at the beach. The two can differ substantially:</p>
<ul>
<li><strong>Offshore height &gt; beach face height — usually.</strong> Once a swell crosses the continental shelf, shoaling can amplify height, but refraction loss and the angle between swell and beach orientation usually reduce face height vs offshore Hs.</li>
<li><strong>Period is preserved — mostly.</strong> Period doesn't change much from offshore to beach. The 14 s the buoy reports is essentially the 14 s you'll feel.</li>
<li><strong>Direction bends.</strong> Refraction turns swell direction toward shore-normal as it crosses the shelf. A 90° offshore direction may arrive at the beach as 75–80°.</li>
</ul>
<p>For a typical East Coast beach with a relatively narrow shelf, beach face heights are often <strong>30–50% smaller</strong> than offshore Hs from a deep-water buoy. A 3 ft buoy reading at 12 s might translate to 4 ft face heights on the bigger sets — but on an offshore reef buoy 100 nm out, the 3 ft might mean barely 1.5 ft at the beach. Distance from the coastline matters a lot.</p>

<h2>Picking the right buoy</h2>
<p>The buoy panel on every forecast page picks the closest stations and lists their distance. The closer the buoy, the more directly its readings apply. When the only available buoy is 150+ nm offshore, this site shows a warning — those readings are open-ocean and overstate beach heights significantly.</p>
<p>Look for buoys within 50 nm if you can. Within 20 nm is ideal. CDIP nearshore buoys (often within 5 nm) are gold for West Coast surfers.</p>

<h2>The wave spectrum</h2>
<p>The 1D spectrum chart on the buoy panel shows wave energy across frequencies. Useful patterns to recognize:</p>
<ul>
<li>A <strong>single sharp peak</strong> at long period (low frequency) = one organized ground swell. Clean conditions.</li>
<li><strong>Two or more peaks</strong> = mixed seas. Multiple swell trains. Lined-up sets interrupted by random chop.</li>
<li>A <strong>broad peak at short period</strong> (high frequency) = local wind sea. Choppy, disorganized.</li>
<li>The 2D directional spectrum (CDIP only) adds direction-of-arrival per frequency. A real luxury when available.</li>
</ul>

<h2>Buoys vs models, when they disagree</h2>
<p>The buoy is measuring what's actually there. The model is predicting what it thinks <em>should</em> be there. When they disagree by more than ~25%, trust the buoy for the immediate window and treat the next 24 h of forecast with skepticism. Persistent model bias against buoy observations is what professional surf forecasters spend their careers correcting.</p>

<h2>Practical use</h2>
<ul>
<li>Check your nearest buoy before any session — it's the ground truth for the moment.</li>
<li>Compare buoy readings to what the forecast said for that hour. If they're close, the forecast is probably reliable for the next day or so. If they're way off, weight the buoy more.</li>
<li>For "should I go now" decisions, the buoy beats any forecast.</li>
<li>For "should I drive 90 minutes tomorrow," the forecast is your only option — but knowing the current bias from buoy comparison helps calibrate it.</li>
</ul>
''',
    },
    {
        'slug': 'storms-and-groundswell',
        'title': 'Storms and Groundswell',
        'description': 'How distant storms generate the ground swells that produce world-class surf — fetch, duration, and travel-time geometry.',
        'keywords': 'ground swell, storm, hurricane, fetch, swell propagation',
        'html_body': '''
<p>Every long-period ground swell on the planet starts the same way: a storm somewhere in the open ocean blows wind across the water for long enough, hard enough, and over a long enough fetch that it generates waves that survive the journey to your beach. Understanding the geometry of storm-driven swells turns weather maps from background noise into a tool you can use.</p>

<h2>The three ingredients</h2>
<p>Wave-generating storms need three things working together:</p>
<ul>
<li><strong>Wind speed:</strong> the higher the better. 30+ kt sustained winds produce serious swells; 50+ kt produce the kind of legendary winter swells that fire Mavericks or Pipeline.</li>
<li><strong>Fetch:</strong> the over-water distance the wind blows in roughly the same direction. Long fetch + sustained winds = bigger, longer-period swells. A 1,000+ nm fetch is what separates Pacific ground swells from East Coast hurricane swells.</li>
<li><strong>Duration:</strong> how long the wind blows. 12–24 hours of sustained 40 kt winds will fully develop the sea state for that wind speed; beyond that, you can't grow the swell any more (the "fully developed sea" limit).</li>
</ul>

<h2>What happens when the storm ends</h2>
<p>Once the wind stops, the waves <strong>decay</strong> — they lose energy slowly as they propagate outward. But not all wavelengths decay at the same rate. Short-period (high-frequency) components dissipate first; long-period (low-frequency) components keep their energy for thousands of miles. This is why a Gulf of Alaska storm produces 14 s ground swells at the North Shore three days later, even though the storm was 2,500 nm away.</p>
<p>The swell field that arrives at your beach is the <strong>dispersive</strong> tail of the storm: long periods first, then progressively shorter periods. A buoy watching a remote swell approach will see the period <em>drop</em> over time as the storm's full spectrum arrives.</p>

<h2>Group velocity and travel time</h2>
<p>Waves propagate at their <strong>group velocity</strong>: <code>cg ≈ 0.78 · T m/s</code> in deep water. A 14 s wave travels at ~10.9 m/s ≈ 39 km/h. So:</p>
<ul>
<li>A storm 1,000 nm away producing a 14 s swell will arrive in ~48 hours.</li>
<li>A storm 2,000 nm away takes ~96 hours.</li>
<li>A storm 4,000 nm away takes ~190 hours (8 days).</li>
</ul>
<p>This is why surf forecasters watch storm systems thousands of miles offshore — the energy is already on the way, you just have to wait. The session score on this site uses group velocity to estimate the origin distance for arriving long-period swells.</p>

<h2>East Coast vs West Coast storm patterns</h2>
<p><strong>East Coast:</strong> two main storm types feed surf.</p>
<ul>
<li><strong>Hurricanes (Aug–Nov):</strong> tropical systems generate clean E and ESE swells that travel up the seaboard. The best East Coast surf often comes from hurricanes 500–1,500 nm offshore — close enough to deliver size, far enough to deliver clean lines.</li>
<li><strong>Nor'easters (Oct–Apr):</strong> extratropical cyclones up the Atlantic coast generate big NE swells. Often storm conditions at the same time as the surf — onshore winds, rain, cold water — but the swells themselves are big and long-period.</li>
</ul>
<p><strong>West Coast:</strong> three main sources.</p>
<ul>
<li><strong>North Pacific winter storms (Nov–Mar):</strong> Aleutian and Gulf of Alaska lows generate W and NW swells. Travel 2,000–3,000 nm to reach California.</li>
<li><strong>Southern Hemisphere swells (May–Sep):</strong> winter storms in the Southern Ocean generate S and SW swells that travel up the Pacific. Clean, long-period, but small.</li>
<li><strong>Tropical systems (occasional):</strong> Eastern Pacific hurricanes produce S swells for Southern California.</li>
</ul>

<h2>The narrative this site builds</h2>
<p>The swell narrative panel on every forecast page identifies the dominant swell type, estimates its origin distance using group velocity, and flags whether the swell is building, holding, or fading over the next 24 h. It pulls all this from the wave model output you'd otherwise have to dig into yourself.</p>

<h2>Practical use</h2>
<ul>
<li>When you see "ground swell from the NE at 14 s, originating ~2,500 km away" — that means there's a storm in the central Atlantic, the swell is 2–3 days into a 3-day journey, and the peak is still ahead.</li>
<li>Long-period swells are <em>direction-fragile</em>: 14 s at 30° off beach-normal arrives at full power; 14 s at 75° off arrives at half power.</li>
<li>Short-period swells (under 9 s) are usually within 12–24 h of being generated. They're not from a distant storm — they're from <em>this morning's</em> wind, somewhere nearby.</li>
<li>Watch the period trend on the wave chart. Rising period over hours = an approaching swell from a distant storm. Falling period = the swell train has passed through its peak.</li>
</ul>
''',
    },
    {
        'slug': 'surf-safety-basics',
        'title': 'Surf Safety Basics',
        'description': 'A short, practical guide to staying safe in the water: rip currents, leash and board management, crowds and lineup etiquette, hypothermia, and big surf.',
        'keywords': 'surf safety, rip current, leash, hypothermia, beginner',
        'html_body': '''
<p>The forecast tells you what the surf will be like. It doesn't tell you whether you should be in it. <strong>Surf safety</strong> is mostly common sense plus a handful of specific skills — and the worst incidents happen to people who haven't thought about them in advance. Here's the short list.</p>

<h2>Rip currents</h2>
<p>The single most common cause of surf-related drownings on US beaches. A <strong>rip current</strong> is a narrow channel of water flowing seaward, formed by water from breaking waves piling up against the beach and draining back through a low spot in the sandbar. Rips can move at 1–2 m/s — faster than most people can swim.</p>
<p>How to spot one before you paddle out:</p>
<ul>
<li>A discolored, often murky channel running perpendicular to the beach.</li>
<li>A gap in the line of breaking waves — the rip is the path of least resistance, so waves don't break there.</li>
<li>Foam or debris streaming seaward in a defined line.</li>
<li>Choppy, agitated water where smooth water surrounds it.</li>
</ul>
<p>If you get caught in one: <strong>don't fight it</strong>. Swim parallel to the beach until you're out of the channel, then angle back to shore. Surfers actually use rips deliberately to paddle out faster — knowing them is half the battle.</p>

<h2>Leash and board management</h2>
<p>Your leash keeps your board with you in a wipeout but turns your board into a missile if it snaps free. Check your leash before every session — frayed cord, weak swivel, cracked plug. Replace anything that looks marginal. Cost: $20. Cost of a loose board hitting another surfer's head: a hospital visit.</p>
<p>If you're a beginner: get a board cover or a soft-top before learning. The fiberglass nose of a shortboard does not care that you just lost your balance.</p>

<h2>Lineup etiquette and crowd safety</h2>
<p>Most "safety" incidents in crowded lineups are collisions, not drownings. The unwritten rules exist for safety as much as fairness:</p>
<ul>
<li><strong>The surfer closer to the peak has priority.</strong> Don't drop in on someone already riding.</li>
<li><strong>Paddling out, don't paddle through the takeoff zone.</strong> Go around.</li>
<li><strong>Hold your board.</strong> Don't ditch it if a wave comes and someone is paddling out behind you.</li>
<li><strong>If you fall, cover your head.</strong> Your own board comes back at you.</li>
<li><strong>Locals aren't required to be friendly, but most are.</strong> Watch first, paddle out at the edge, work your way in.</li>
</ul>

<h2>Cold water and hypothermia</h2>
<p>Water below ~60°F will sap your strength faster than you think. Below ~50°F, you have minutes before fine motor control goes. Wear the right wetsuit for the temperature:</p>
<ul>
<li><strong>≥ 75°F:</strong> boardshorts or rashguard.</li>
<li><strong>68–75°F:</strong> springsuit or 2 mm top.</li>
<li><strong>62–68°F:</strong> 3/2 mm fullsuit.</li>
<li><strong>55–62°F:</strong> 4/3 mm fullsuit + booties.</li>
<li><strong>48–55°F:</strong> 5/4 mm + booties + hood.</li>
<li><strong>&lt; 48°F:</strong> 5/4/3 mm + booties + hood + gloves.</li>
</ul>
<p>Wind chill matters too. A 60°F day with 15 mph offshore wind feels much colder than a 60°F day with no wind. Get out before you start shivering uncontrollably — that's the warning sign.</p>

<h2>Big surf</h2>
<p>"Big" is relative — if you're an intermediate surfer used to chest-high days, head-high surf is your big surf. Don't paddle out into conditions that are 50%+ bigger than your normal session unless you've worked up to it gradually.</p>
<p>For genuine big-wave conditions (overhead+, long-period, hold-downs lasting 15+ seconds), specific safety protocols apply: never surf alone, use the right equipment (gun shape, leash for big waves), know the rescue patterns, and consider whether you've had hold-down training. If you have to ask whether you're ready, you're not.</p>

<h2>The big four warning signs to bail</h2>
<p>If any of these apply, sit it out:</p>
<ul>
<li>You can't keep up with the duck dives — you'll be exhausted before you make it out.</li>
<li>You're surfing alone and conditions have gotten bigger or stormier than you expected.</li>
<li>You can't see the bottom (low-vis water + reef = bad math).</li>
<li>Your hands or feet have gone numb — you've crossed into hypothermia territory.</li>
</ul>

<h2>What this site can and can't tell you</h2>
<p>The forecast tells you the size, period, direction, wind, and tide. It doesn't tell you whether your local sandbar is currently shaped for safe takeoffs, whether the lineup is crowded, whether the rip channels have moved since last week, or whether your wetsuit is dry. <strong>Make the safety call on the beach, not from the forecast.</strong> Watch the lineup for a few minutes before you paddle out. If it doesn't look right, it isn't.</p>
''',
    },
]

ARTICLES_BY_SLUG = {a['slug']: a for a in ARTICLES}
