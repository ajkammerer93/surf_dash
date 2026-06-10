"""Content module for /regions/<slug> landing pages.

Each region groups a slice of LOCATION_BY_SLUG (by state code or an explicit
slug allowlist) under regional context that explains the dominant swell
seasons, hazards, and orientation patterns. Crawlable internal links to
the forecast pages inside the region seed PageRank into the long tail.
"""

REGIONS = [
    {
        'slug': 'outer-banks',
        'title': 'Outer Banks Surf Forecast',
        'short_title': 'Outer Banks (OBX), NC',
        'description': 'Surf forecast for the Outer Banks of North Carolina — Kitty Hawk, Nags Head, Rodanthe, Waves, Avon, Hatteras, Ocracoke. Live cams, 7-day swell, tide and wind.',
        'state_filter': None,  # uses explicit slug list below
        'slug_list': ['kitty-hawk', 'nags-head-abalone-st', 'nags-head-jennettes-pier',
                      'rodanthe', 'waves', 'avon-pier', 'hatteras', 'ocracoke'],
        'intro_html': '''
<p>The <strong>Outer Banks</strong> of North Carolina — the long thin barrier islands stretching from Kitty Hawk south through Cape Hatteras to Ocracoke — face open Atlantic with no significant offshore shelf to weaken incoming swells. It's the most exposed coastline on the US East Coast.</p>
<p>The orientation shifts continuously as the islands curve. Kitty Hawk and Nags Head face roughly east; from Rodanthe south, the islands swing southeast, and at Cape Hatteras itself the orientation hooks around so beaches face north <em>and</em> south within a few miles. Combined, the OBX has a swell window covering virtually every Atlantic direction — there's almost always something working somewhere.</p>
<p>Season-wise, the OBX runs on two main engines. <strong>Hurricanes</strong> (Aug–Nov) deliver clean E–SE swells from systems tracking up the seaboard; the best East Coast surf of the year typically comes from a hurricane 500–1,500 nm offshore. <strong>Nor'easters</strong> (Oct–Apr) generate big NE swells with stormy conditions. Winter cold fronts produce occasional short windows of clean offshore wind after a system passes.</p>
<p>Hazards: strong rip currents, cold water in winter (50s °F), and at exposed breaks like Hatteras the surf can go from manageable to overhead very fast. The lighthouse cove and inlet zones around Hatteras are also dangerous for swimmers — strong tidal currents.</p>
''',
    },
    {
        'slug': 'north-carolina-coast',
        'title': 'North Carolina Coast Surf Forecast',
        'short_title': 'North Carolina (non-OBX)',
        'description': 'Surf forecast for North Carolina south of the Outer Banks — Topsail, Surf City, Wrightsville, Carolina Beach, Oak Island, Ocean Isle, Atlantic Beach, Emerald Isle.',
        'state_filter': None,
        'slug_list': ['atlantic-beach-nc', 'emerald-isle-nc', 'north-topsail-island',
                      'topsail-beach', 'surf-city-line', 'surf-city-pier-north',
                      'wrightsville-beach', 'wb-mercers-pier', 'carolina-beach',
                      'carolina-beach-center-pier', 'kure-beach', 'oak-island',
                      'ocean-isle-beach', 'sunset-beach'],
        'intro_html': '''
<p>North Carolina south of the Outer Banks runs from the Crystal Coast (Atlantic Beach, Emerald Isle) through Cape Lookout and on to the SE-facing beaches around Wrightsville, Carolina Beach, and the Brunswick beaches (Oak Island, Ocean Isle, Sunset Beach). The orientation shifts dramatically: Topsail and Surf City face east-southeast; Wrightsville faces southeast; the Brunswick stretch faces nearly south.</p>
<p>That orientation difference matters enormously. The same NE groundswell that fires Surf City Pier produces half the height at Wrightsville Beach (more refraction loss into a SE-facing aspect), and barely reaches the south-facing Brunswick beaches at all. South-flowing summer hurricane swells, by contrast, light up the southern beaches and disappear at Surf City. Pay attention to swell direction — it determines which breaks work.</p>
<p>Hurricane season is the surf engine here, August through November. Nor'easters work for the more easterly oriented spots. Cold fronts in winter and spring produce light offshore mornings; summer is small with occasional tropical-system wind swell.</p>
<p>Water temps run from low 50s °F in winter to mid-80s in summer. Most spots have substantial sandbar variation — check buoy reports for the actual current conditions, not just the seasonal averages.</p>
''',
    },
    {
        'slug': 'virginia-coast',
        'title': 'Virginia Coast Surf Forecast',
        'short_title': 'Virginia',
        'description': 'Surf forecast for Virginia — Virginia Beach and the southeast Virginia coast. Live cam, 7-day swell, tide, and wind.',
        'state_filter': 'VA',
        'slug_list': None,
        'intro_html': '''
<p>Virginia's surf is concentrated around <strong>Virginia Beach</strong> and the southeastern Virginia coastline. The orientation runs roughly ENE — open to swells from the north through east-southeast. The coastline is relatively straight, so neighboring breaks have similar swell windows, but offshore shoals do produce some local refraction differences.</p>
<p>The dominant swell sources are the same as the rest of the mid-Atlantic: <strong>hurricanes</strong> from August through November tracking up the seaboard, and <strong>nor'easters</strong> from late autumn through winter. Summer is generally small. The best windows often come right after a passing storm, when offshore wind kicks in behind the system and the swell is still elevated.</p>
<p>Water temps range from low 40s °F in winter to high 70s in summer. The shelf is wider than the Outer Banks so wave shoaling is less dramatic; expect face heights at the beach to be closer to offshore buoy readings than at Hatteras or Rodanthe.</p>
''',
    },
    {
        'slug': 'jersey-shore',
        'title': 'Jersey Shore Surf Forecast',
        'short_title': 'New Jersey',
        'description': 'Surf forecast for the New Jersey shore — Manasquan, Belmar, Asbury Park, Long Beach Island, Atlantic City, and the Cape May–Wildwood beaches.',
        'state_filter': 'NJ',
        'slug_list': None,
        'intro_html': '''
<p>The <strong>Jersey Shore</strong> runs north-south, with most beaches facing roughly E to ENE. From the Sandy Hook–Long Branch stretch in the north through the Manasquan/Belmar area, down past Asbury Park, Seaside, and onto Long Beach Island, the orientation shifts subtly enough that nearby spots can produce noticeably different surf on the same forecast.</p>
<p>Hurricane season (August–October) is the high water mark — clean east-southeast swells, often with light offshore mornings before the sea breeze. Nor'easters from late autumn through winter generate big NE swells; quality varies because the same systems usually bring strong onshore winds at the same time as the swell.</p>
<p>Spring and fall transitions produce some of the cleanest windows of the year — moderate swells from passing systems with offshore winds behind them. Summer is mostly small with afternoon onshore sea breezes; dawn patrol is your friend.</p>
<p>The Jersey shore has well-organized sandbar deposits next to its many piers and jetties (Manasquan being the classic). These produce more focused, peaky waves than open-beach stretches. Water runs cold for surfers — low 40s °F in February, peaking only mid-70s in August.</p>
''',
    },
    {
        'slug': 'long-island',
        'title': 'Long Island Surf Forecast',
        'short_title': 'Long Island, NY',
        'description': 'Surf forecast for Long Island, New York — south-shore beaches from Long Beach east through Montauk.',
        'state_filter': 'NY',
        'slug_list': None,
        'intro_html': '''
<p>Long Island's south shore faces nearly due south, which makes it dramatically different from the rest of the East Coast. The dominant swell sources still come from the north and east — hurricanes and nor'easters — but they have to wrap onto a south-facing beach, with significant refraction loss along the way. Long Island consequently sees smaller surf than the same systems produce on the Jersey Shore or Outer Banks.</p>
<p>The trade-off: when something does wrap onto Long Island, it's typically very clean. Long-period ground swells refract into the south-facing geometry and produce lined-up, organized waves. Short-period wind swells barely make it.</p>
<p>Best season runs late summer through fall — hurricane swells with clean offshore wind. Winter nor'easters are big but often stormy. Montauk on the east end picks up more direct east swells than the rest of the south shore due to its more easterly orientation.</p>
<p>Water temps mirror the Jersey shore — low 40s °F February, high 70s August. Watch for strong rip currents at the inlets (Jones, Fire Island).</p>
''',
    },
    {
        'slug': 'florida-space-coast',
        'title': 'Florida Space Coast Surf Forecast',
        'short_title': 'Florida (Space Coast & East)',
        'description': 'Surf forecast for Florida\'s Space Coast — Cocoa Beach, Sebastian Inlet, Vero Beach, Indialantic, and the east-facing Florida coast.',
        'state_filter': 'FL',
        'slug_list': None,
        'intro_html': '''
<p>Florida's east coast — known to surfers as the <strong>Space Coast</strong> in the central region around Cocoa Beach and Sebastian Inlet — faces ENE on the open Atlantic. It's exposed to swells from the north, east, and southeast, with the Bahamas blocking a fraction of the southerly window for southern Florida beaches.</p>
<p>The surf engine here is offshore hurricanes (Aug–Nov), particularly storms tracking through the Bahamas or curving back from the Carolinas. Sebastian Inlet — generally considered Florida's best wave — fires on hurricane swells thanks to its jetty-focused sandbar. Winter cold fronts produce shorter-period wind swells, and occasional nor'easters wrap down the coast.</p>
<p>Florida has a long offshore shelf, so swell energy dissipates more before reaching the beach than at exposed locations like Hatteras. A 3 ft, 12 s buoy reading offshore typically produces only chest-high faces at the beach. The exception is jetty and pier spots like Sebastian, which focus what energy does arrive.</p>
<p>Water temps stay warm year-round — low 70s °F in winter, mid-80s in summer. No wetsuit needed most of the year. Watch for rip currents and (occasionally) sharks.</p>
''',
    },
    {
        'slug': 'southern-california',
        'title': 'Southern California Surf Forecast',
        'short_title': 'Southern California',
        'description': 'Surf forecast for Southern California — from Point Conception south through San Diego. North Pacific winter swells and Southern hemisphere summer swells.',
        'state_filter': None,
        'slug_list': ['san-diego-pacific-beach-ca', 'del-mar-ca', 'encinitas-moonlight-beach-ca',
                      'oceanside-ca', 'san-clemente-ca', 'laguna-beach-ca',
                      'huntington-beach-ca', 'seal-beach-ca', 'hermosa-beach-ca',
                      'manhattan-beach-ca', 'venice-beach-ca', 'santa-monica-ca',
                      'malibu-ca'],
        'intro_html': '''
<p>Southern California's coast — from Point Conception south through San Diego — faces a broad WSW with significant variation. North Pacific winter storms (Nov–Mar) produce W and NW swells which are in window for the entire coast. Southern hemisphere winter (May–Sep, opposite seasons) produces cleaner but smaller S swells which are in window mostly for breaks that face south through southwest — Newport, Black's, La Jolla.</p>
<p>The narrow continental shelf and abundant kelp tend to produce well-organized waves. Long-period NW swells refract heavily into the SoCal bight, often producing surf as far south as San Diego from systems that were aimed at Oregon. The Channel Islands offshore (Catalina, San Clemente) block some southerly energy from breaks like Huntington, but enhance others through diffraction.</p>
<p>Best months: November through March for size, June through August for clean smaller surf. Summer is famously crowded; winter sees big-wave specialists at outer reefs.</p>
<p>Water temps are mild for the Pacific — low 60s °F in winter, low 70s in summer. A 3/2 fullsuit covers most of the year for most surfers.</p>
''',
    },
    {
        'slug': 'northern-california',
        'title': 'Northern California Surf Forecast',
        'short_title': 'Northern California',
        'description': 'Surf forecast for Northern California — Bay Area beaches and the central coast. North Pacific winter swells and cold water year-round.',
        'state_filter': None,
        'slug_list': ['pismo-beach-ca', 'santa-cruz-ca', 'half-moon-bay-ca', 'point-arena-ca'],
        'intro_html': '''
<p>Northern California's coast — from Half Moon Bay north through Mendocino — faces open Pacific with no offshore islands to block winter storms. North Pacific winter swells (Nov–Mar) arrive at full force, producing some of the biggest rideable waves on Earth (Mavericks, just outside Half Moon Bay). Summer brings smaller, cleaner conditions.</p>
<p>The coastline is rugged and oriented mostly west — exposed to W and NW swells. South swells are rare and refract heavily; the region's surf is mostly fed by winter storms in the Gulf of Alaska and Aleutians. Long-period ground swells (15+ s) are common in winter.</p>
<p>Water is cold year-round — low 50s °F in winter, peaking only at low 60s in late summer. A 4/3 fullsuit is the minimum for most surfers; many wear 5/4 with hood and booties through winter. White sharks are present in higher numbers than further south, though attacks remain rare.</p>
''',
    },
    {
        'slug': 'oregon-coast',
        'title': 'Oregon Coast Surf Forecast',
        'short_title': 'Oregon',
        'description': 'Surf forecast for the Oregon coast — Cannon Beach, Lincoln City, Newport, Bandon, and the open Pacific coast.',
        'state_filter': 'OR',
        'slug_list': None,
        'intro_html': '''
<p>The Oregon coast faces W to WSW open Pacific. It's directly exposed to North Pacific winter storms, which produce consistently large swells from November through March. Long-period ground swells dominate; short-period local wind swells are common during storms but quickly clean up between systems.</p>
<p>Summer (June–September) is the cleanest but smallest time of year — passing high pressure stabilizes the wind and small SW swells arrive from the southern hemisphere. The "surfable window" each year for many spots is summer through fall; winter is mostly experts only.</p>
<p>Water is cold year-round — high 40s °F in winter, low 60s peak. A 5/4 fullsuit with hood and booties is standard from October through June. The coastline is exposed and rocky; pay attention to tide and reef hazards before paddling out at any unfamiliar spot.</p>
''',
    },
    {
        'slug': 'hawaii-north-shore',
        'title': 'Hawaii North Shore Surf Forecast',
        'short_title': 'Hawaii (North Shore)',
        'description': 'Surf forecast for Hawaii\'s North Shore — Pipeline, Sunset, Waimea, and the seven-mile miracle. North Pacific winter swells, world-class surf.',
        'state_filter': 'HI',
        'slug_list': None,
        'intro_html': '''
<p>The <strong>North Shore of Oahu</strong> — the legendary seven-mile stretch from Haleiwa east through Sunset Beach — faces NNW open Pacific. North Pacific winter storms (Nov–Mar) generate ground swells that travel 2,000+ nm to arrive at full strength, producing the world's most famous big-wave surf.</p>
<p>The swell window opens roughly from N through NNW and includes some WNW. Long-period (15+ s) ground swells refract dramatically into the various breaks: Pipeline fires on a NNW direction at a specific tide, Sunset on more N swell with size, Waimea on the biggest swells of the season (15+ ft, 17+ s).</p>
<p>Summer (April–October) is small — the prevailing pattern of trade winds and southern hemisphere swells means the North Shore is mostly flat. The South Shore (Waikiki, Diamond Head) picks up summer south swells, while the North Shore sits out.</p>
<p>Water temps are warm year-round — high 70s °F in winter, low 80s in summer. No wetsuit needed; many surfers use rashguards mainly for sun and reef protection. Hazards are substantial: shallow reef, strong currents, big rip channels at the major breaks. The North Shore is unforgiving when conditions are big — don't paddle out beyond your level.</p>
''',
    },
    {
        'slug': 'delmarva',
        'title': 'Delmarva Surf Forecast',
        'short_title': 'Delmarva (MD & DE)',
        'description': 'Surf forecast for the Delmarva coast — Ocean City Maryland, Assateague, and the Delaware beaches (Rehoboth, Dewey, Bethany). Live cams, 7-day swell, tide and wind.',
        'state_filter': None,
        'slug_list': ['ocean-city-md', 'oc-md-inlet-cam', 'oc-md-assateague-cam',
                      'rehoboth-beach-de', 'dewey-beach-de', 'bethany-beach-de'],
        'intro_html': '''
<p>The <strong>Delmarva peninsula</strong> — Delaware's Rehoboth, Dewey, and Bethany beaches running south into Ocean City, Maryland and Assateague Island — faces almost due east, wide open to Atlantic swell from NE through SE. It's a classic mid-Atlantic setup: long, straight barrier-island beach break with sandbars that shift after every significant storm.</p>
<p>The standout wave is the <strong>Ocean City inlet</strong>, where the south jetty grooms NE swells into longer, more organized rights than anything on the open beach. Assateague, just across the inlet, offers uncrowded peaks for surfers willing to walk.</p>
<p>The engines are the same as the rest of the mid-Atlantic: <strong>hurricane swells</strong> from August through November and <strong>nor'easters</strong> from late fall through early spring. Summer runs small with short-period wind swell — dawn patrol before the sea breeze is the play. The cleanest windows of the year often come in early fall: hurricane swell in the water and light offshore mornings.</p>
<p>Water temps swing hard: low 40s °F in winter to high 70s in late summer. A 5/4 with hood and booties is standard January through March; trunks or a spring suit cover July through September. Rip currents around the inlet jetties are the main hazard.</p>
''',
    },
    {
        'slug': 'south-carolina-georgia',
        'title': 'South Carolina & Georgia Surf Forecast',
        'short_title': 'South Carolina & Georgia',
        'description': 'Surf forecast for South Carolina and Georgia — Folly Beach, Isle of Palms, Myrtle Beach, Garden City, Hilton Head, and Tybee Island. Live cams, 7-day swell, tide and wind.',
        'state_filter': None,
        'slug_list': ['myrtle-beach-sc', 'garden-city-beach-sc', 'isle-of-palms-sc',
                      'folly-beach-sc', 'hilton-head-island-sc', 'tybee-island-ga'],
        'intro_html': '''
<p>The <strong>South Carolina and Georgia coast</strong> curves from the Grand Strand (Myrtle Beach, Garden City) through Charleston's beaches (Isle of Palms, Folly) down to Hilton Head and Tybee Island. The continental shelf here is among the widest on the US East Coast, which drains energy from incoming swell — the same system that fires the Outer Banks arrives noticeably smaller here. What does get through tends to be clean and user-friendly.</p>
<p><strong>Folly Beach</strong> is the region's surf hub: the Washout, a stretch left bare by Hurricane Hugo, picks up more swell than anywhere else in the state and hosts the bulk of the local scene. The northern Grand Strand beaches face more east and catch NE windswell slightly better; Hilton Head and Tybee face southeast across an even wider shelf and need genuine groundswell to break well.</p>
<p><strong>Hurricane season</strong> (August–November) is the main event — long-period SE groundswell from offshore systems is what this shelf handles best. Winter cold fronts produce short-period windswell windows that clean up quickly behind the front. Summer is small but warm and forgiving — an ideal longboard and learner season.</p>
<p>Water temps are the mildest north of Florida: mid-50s °F in winter, mid-80s at the summer peak. A 4/3 covers the coldest months; trunks from May through October. Strong tidal currents near the many inlets are the main hazard, and the large tide swing (5-7 ft) means many spots only break properly on part of the tide.</p>
''',
    },
    {
        'slug': 'new-england',
        'title': 'New England Surf Forecast',
        'short_title': 'New England',
        'description': 'Surf forecast for New England — Narragansett Rhode Island, Hampton Beach New Hampshire, and the Maine beaches (Old Orchard, Ogunquit). Live cams, 7-day swell, tide and wind.',
        'state_filter': None,
        'slug_list': ['narragansett-ri', 'hampton-beach-nh',
                      'old-orchard-beach-me', 'ogunquit-beach-me'],
        'intro_html': '''
<p><strong>New England</strong> surf runs from Rhode Island's south-facing shore up through New Hampshire's short coastline to the beach breaks of southern Maine. Geography splits the region in two: Rhode Island faces south and drinks in hurricane swell, while New Hampshire and Maine face east into the Gulf of Maine and run on nor'easter energy.</p>
<p><strong>Narragansett</strong> and the Rhode Island south shore are the regional standouts — a mix of beach break, cobblestone points, and reefs that turn long-period S–SE hurricane swell into some of the best waves on the East Coast. September and October, when tropical swells meet the first offshore-wind cold fronts, are the prime window.</p>
<p>North of Cape Ann, <strong>Hampton Beach</strong> and the Maine beaches (Old Orchard, Ogunquit) need NE wind swell or a nor'easter to break. The Gulf of Maine blocks most hurricane energy, so fall through winter is the consistent season — at the price of genuinely arctic conditions.</p>
<p>Water is the coldest of any US surf region: high 30s °F in late winter in Maine, peaking in the mid-60s in August. A 5/4 or 6/5 with hood, gloves, and 7mm booties is standard from December through April. The reward for the rubber: empty lineups and some of the cleanest storm-groomed waves anywhere.</p>
''',
    },
    {
        'slug': 'gulf-coast',
        'title': 'Gulf Coast Surf Forecast',
        'short_title': 'Gulf Coast (TX & AL)',
        'description': 'Surf forecast for the Gulf Coast — Galveston, Port Aransas, South Padre Island Texas, and Gulf Shores Alabama. Live cams, 7-day swell, tide and wind.',
        'state_filter': None,
        'slug_list': ['galveston-tx', 'port-aransas-tx', 'south-padre-island-tx',
                      'gulf-shores-al'],
        'intro_html': '''
<p>The <strong>Gulf of Mexico</strong> is a fetch-limited basin — no open-ocean groundswell reaches it, so everything that breaks on the Texas and Alabama coast is generated inside the Gulf itself. That means short-period wind swell (4-8 seconds) is the staple diet, and timing matters more here than anywhere else: conditions can go from flat to fun and back inside a single day.</p>
<p>The two engines are <strong>winter cold fronts</strong> and <strong>tropical systems</strong>. From October through March, fronts pushing off the coast generate days of onshore slop followed by a short, clean window as the wind swings offshore behind the front — the classic Gulf pattern. In hurricane season, a storm anywhere in the Gulf can light up the entire coast with the year's best surf, often while the weather at the beach is still sunny.</p>
<p><strong>South Padre Island</strong> is the most consistent Texas spot — deeper water close to shore lets swell arrive with less bottom drag. <strong>Port Aransas</strong> and <strong>Galveston</strong> work the same patterns with a bit less size; Galveston's piers organize the sandbars and host the longest-running scene. <strong>Gulf Shores</strong>, Alabama picks up slightly better exposure to southeast fetch and shares the Florida panhandle's sugar-sand bars.</p>
<p>Water temps run warm: low 60s °F in winter on the upper Texas coast (a 3/2 season) to bathwater mid-80s in summer. Hazards are modest — rip currents near piers and jetties, and stingrays in the shallows (shuffle your feet). It's one of the friendliest learning coastlines in the country when a clean window lines up.</p>
''',
    },
]

REGIONS_BY_SLUG = {r['slug']: r for r in REGIONS}
