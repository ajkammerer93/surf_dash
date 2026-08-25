"""Tests that render.yaml still describes the service Render actually runs.

This file exists because the drift it guards against already happened and cost
real capacity. render.yaml declared --timeout 180 --workers 1 --threads 8 while
the dashboard ran --timeout 120 --workers 2 with no --threads at all: two
concurrent requests instead of eight, and a worker timeout that killed the
fallback chains v0.11.17 had raised it to accommodate. Nothing failed, nothing
warned, and the file read as authoritative to everyone who opened it.

An unadopted render.yaml is just a file in a repo. Once it IS adopted, Blueprint
sync overwrites dashboard settings from this file, so a field that is true of the
live service but missing here gets silently reset on the next sync. These tests
pin the fields where that would actually hurt.
"""
import os
import re

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDER_YAML = os.path.join(ROOT, 'render.yaml')


@pytest.fixture(scope='module')
def service():
    with open(RENDER_YAML) as f:
        spec = yaml.safe_load(f)
    services = spec['services']
    assert len(services) == 1
    return services[0]


def test_plan_and_region_are_declared(service):
    """Both are unset-able only once.

    Region cannot be changed after creation at all, so a wrong value here is not
    a setting, it is a migration. Plan absent from an adopted Blueprint risks a
    reset to a default tier, and the basin cache entry alone will not fit in
    512 MB comfortably.
    """
    assert service['plan'] == 'standard'
    assert service['region'] == 'ohio'


def test_health_check_path_matches_a_real_route(service):
    """And specifically NOT the upstream diagnostic.

    A probe that calls a third party hands that third party a restart button: a
    bad afternoon at NOAA would have Render recycling a process that was serving
    cached forecasts perfectly well, wiping the in-process cache each time.
    """
    assert service['healthCheckPath'] == '/healthz'
    src = open(os.path.join(ROOT, 'app.py')).read()
    assert "@app.route('/healthz')" in src
    assert service['healthCheckPath'] != '/api/health-upstreams'


def test_gunicorn_flags_are_the_reasoned_ones(service):
    """Each of these has a documented incident behind it; see the yaml comments."""
    cmd = service['startCommand']
    assert '--timeout 180' in cmd, 'v0.11.17: 150s fallback chains need the headroom'
    assert '--workers 1' in cmd, 'v0.8.1 reversal: the in-process cache is not shared across forks'
    assert '--threads 8' in cmd, 'v0.11.18: four threads were pinned during the 2026-07-16 degradation'
    assert '--preload' in cmd


def test_threads_is_set_at_all(service):
    """The specific silent failure that was live.

    Without --threads gunicorn uses sync workers -- one request per worker, so
    the dashboard's `--workers 2` with no threads meant a concurrency of two. It
    looks like a missing tuning flag and behaves like a four-fold capacity cut.
    """
    assert re.search(r'--threads\s+\d+', service['startCommand'])


def test_runtime_env_vars_are_accounted_for(service):
    """Anything the web service reads without a default has to be declared.

    Adoption retains dashboard values for keys marked sync: false, but a key
    absent from this file entirely is one nobody is tracking.
    """
    declared = {e['key'] for e in service['envVars']}
    assert 'WINDY_API_KEY' in declared
    assert 'PYTHON_VERSION' in declared


def test_no_dead_gunicorn_config_file():
    """gunicorn.conf.py set timeout = 120 and was overridden by the CLI flags.

    It contradicted the real value and read as authoritative to anyone who found
    it before finding render.yaml.
    """
    assert not os.path.exists(os.path.join(ROOT, 'gunicorn.conf.py'))
