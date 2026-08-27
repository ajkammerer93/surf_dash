"""Source pins for the map basemap.

These exist because of how the CARTO basemap failed on 2026-08-27: CARTO began
serving keyless requests with a large diagonal "API KEY REQUIRED /
carto.com/basemaps/apikey" watermark burned into every tile. The tiles still
returned HTTP 200 with a valid PNG of the expected size, so nothing threw,
nothing 4xx'd, no upstream probe complained and no test failed. It shipped and
sat on production looking broken to every visitor until a human happened to
look at the map.

Nothing here can detect a future watermark -- that needs eyes on a rendered
tile. What these pin is the structure that made the incident expensive: four
call sites each naming the tile host inline, so a swap meant finding all four
and getting the URL template right in each. Same reasoning as
test_erddap_mirrors.py's "no call site may name a host directly".

The Esri replacement has two traps that are silent when wrong:
  - the tile path is {z}/{y}/{x} -- Y BEFORE X. With x and y transposed the map
    still loads real tiles of the right size, just of the wrong place.
  - labels are a separate reference layer, not baked into the base, so dimming
    the base must not dim the labels with it.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "templates", "index.html")


def _src():
    with open(INDEX) as f:
        return f.read()


class TestNoKeyedBasemap:
    def test_no_carto_basemap_anywhere(self):
        assert "cartocdn.com" not in _src(), (
            "CARTO watermarks keyless basemap tiles with 'API KEY REQUIRED' "
            "burned into the image while still returning HTTP 200")

    def test_tile_host_is_named_in_exactly_one_place(self):
        """Four call sites each naming the host is what made the swap costly."""
        src = _src()
        hosts = re.findall(r"https://([a-z0-9.\-{}]+)/[^'\"]*\{z\}", src)
        assert hosts, "no tile URL templates found"
        assert len(set(hosts)) == 1, (
            "more than one tile host in the file: %s" % sorted(set(hosts)))
        # base + labels are the only two literals allowed to carry it
        assert len(hosts) == 2, (
            "expected exactly 2 tile URL templates (base + labels), found %d -- "
            "call sites must go through addBasemap()" % len(hosts))

    def test_every_map_uses_the_helper(self):
        src = _src()
        assert "function addBasemap(" in src
        # one definition + one call per map (empty, location picker, swell, buoy)
        assert len(re.findall(r"addBasemap\(", src)) >= 5, (
            "every map must obtain its basemap through addBasemap()")

    def test_no_raw_tilelayer_with_an_inline_url(self):
        src = _src()
        bad = re.findall(r"L\.tileLayer\(\s*['\"]https://", src)
        assert not bad, (
            "%d tileLayer call(s) name a tile URL inline instead of going "
            "through addBasemap()" % len(bad))


class TestEsriTileTemplate:
    def test_tile_path_is_z_y_x_not_z_x_y(self):
        """Esri's REST tile path is {z}/{y}/{x}. Transposed, the map still
        loads valid tiles -- of the wrong location."""
        src = _src()
        for name in ("BASEMAP_BASE_URL", "BASEMAP_LABELS_URL"):
            m = re.search(name + r"\s*=\s*\n?\s*'([^']+)'", src)
            assert m, "%s not found" % name
            url = m.group(1)
            assert url.endswith("/tile/{z}/{y}/{x}"), (
                "%s must end in /tile/{z}/{y}/{x} (Y BEFORE X); got %r"
                % (name, url[-24:]))

    def test_attribution_credits_the_tile_source(self):
        src = _src()
        m = re.search(r"BASEMAP_ATTRIBUTION\s*=\s*\n?\s*'([^']+)'", src)
        assert m, "BASEMAP_ATTRIBUTION not found"
        attr = m.group(1)
        assert "Esri" in attr and "OpenStreetMap" in attr, (
            "attribution must credit Esri and OpenStreetMap, per the service's "
            "own copyrightText")

    def test_labels_layer_is_not_dimmed_with_the_base(self):
        """The mute exists so the basemap recedes behind the wave field. It is
        applied via the base layer's className precisely so the separate labels
        layer keeps full contrast -- dimming the tile pane would take both."""
        src = _src()
        fn = re.search(r"function addBasemap\([\s\S]*?\n        \}", src)
        assert fn, "addBasemap() body not found"
        body = fn.group(0)
        base_part, _, labels_part = body.partition("opts.labels !== false")
        assert "basemap-muted" in base_part, (
            "the mute className belongs on the BASE layer")
        assert "basemap-muted" not in labels_part, (
            "the labels layer must not be dimmed -- it carries the place names")
        assert ".basemap-muted" in src and "filter:" in src, (
            "the .basemap-muted rule must exist and apply a filter")
