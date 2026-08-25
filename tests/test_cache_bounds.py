"""Tests for the response cache's byte bound.

The bound this replaced counted ENTRIES, and entries here differ in size by four
orders of magnitude. A tide prediction is a few kilobytes; the basin grid is
~27 MB resident, because every one of its ~1.4M floats is a 24-byte object plus
a pointer -- about 6.5x its JSON size. CACHE_MAX_ENTRIES = 1000 therefore
described anything between a few megabytes and several gigabytes depending on
which thousand entries you got, and a crawler sweeping 145 spots across seven
key families was the way to find out on a 2 GB instance.

What these pin is mostly the accounting. A byte bound whose running total drifts
from the real contents is worse than no bound at all: it reports a number that
looks like control while enforcing nothing.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as A  # noqa: E402


@pytest.fixture(autouse=True)
def clean_cache():
    with A._cache_lock:
        A._cache.clear()
        A._cache_key_locks.clear()
        A._cache_bytes = 0
    yield
    with A._cache_lock:
        A._cache.clear()
        A._cache_key_locks.clear()
        A._cache_bytes = 0


def _store(key, value):
    with A._cache_lock:
        A._cache_store(key, value, A._deep_size(value))


def _real_total():
    return sum(e['bytes'] for e in A._cache.values())


# --- the sizer ------------------------------------------------------------

def test_deep_size_follows_containers_rather_than_measuring_the_pointer_array():
    """sys.getsizeof on a nested list reports the pointers and nothing they
    lead to, which for these payloads is off by a factor of six."""
    nested = [[float(i) for i in range(200)] for _ in range(50)]
    assert A._deep_size(nested) > 10 * sys.getsizeof(nested)


def test_deep_size_counts_a_shared_object_once():
    shared = [0.0] * 1000
    once = A._deep_size([shared])
    twice = A._deep_size([shared, shared])
    # the second reference costs a pointer, not another kilobyte of floats
    assert twice - once < 0.1 * once


def test_deep_size_survives_a_cycle():
    a = []
    a.append(a)
    assert A._deep_size(a) > 0


# --- accounting -----------------------------------------------------------

def test_running_total_matches_the_contents(monkeypatch):
    for i in range(20):
        _store(f'k{i}', [float(j) for j in range(100)])
    assert A._cache_bytes == _real_total()


def test_overwriting_a_key_does_not_double_count():
    """The bug this guards: re-caching an expired key adds its size again while
    the old size is still on the books, so the total climbs on every refresh
    until eviction fires early and evicts things that were fine."""
    _store('k', [0.0] * 5000)
    after_first = A._cache_bytes
    _store('k', [0.0] * 5000)
    assert A._cache_bytes == pytest.approx(after_first, rel=0.01)
    assert A._cache_bytes == _real_total()
    assert len(A._cache) == 1


def test_eviction_subtracts_what_it_removed(monkeypatch):
    monkeypatch.setattr(A, 'CACHE_MAX_BYTES', 200_000)
    for i in range(40):
        _store(f'k{i}', [float(j) for j in range(300)])
    assert A._cache_bytes == _real_total()
    assert A._cache_bytes <= A.CACHE_MAX_BYTES


# --- the bounds themselves ------------------------------------------------

def test_byte_bound_evicts_before_the_count_bound_would(monkeypatch):
    """The whole point: 12 large entries must not survive just because 12 < 1000."""
    monkeypatch.setattr(A, 'CACHE_MAX_BYTES', 150_000)
    monkeypatch.setattr(A, 'CACHE_MAX_ENTRIES', 1000)
    for i in range(12):
        _store(f'big{i}', [float(j) for j in range(2000)])
    assert len(A._cache) < 12
    assert A._cache_bytes <= A.CACHE_MAX_BYTES


def test_count_bound_still_applies_to_many_small_entries(monkeypatch):
    monkeypatch.setattr(A, 'CACHE_MAX_ENTRIES', 25)
    for i in range(60):
        _store(f'small{i}', i)
    assert len(A._cache) <= 25


def test_eviction_is_oldest_first(monkeypatch):
    monkeypatch.setattr(A, 'CACHE_MAX_ENTRIES', 3)
    for k in ('a', 'b', 'c', 'd'):
        _store(k, [0.0] * 10)
    assert 'a' not in A._cache
    assert 'd' in A._cache


def test_an_entry_bigger_than_the_whole_budget_is_still_kept(monkeypatch):
    """Refetching the basin costs 55-100s against ERDDAP.

    Declining to cache an oversized entry converts a memory bound into an
    availability problem, which is the worse trade -- so the loop stops at one
    entry and logs instead.
    """
    monkeypatch.setattr(A, 'CACHE_MAX_BYTES', 1000)
    _store('huge', [float(i) for i in range(5000)])
    assert 'huge' in A._cache
    assert len(A._cache) == 1


def test_an_oversized_entry_does_not_evict_a_later_normal_one(monkeypatch):
    monkeypatch.setattr(A, 'CACHE_MAX_BYTES', 1000)
    _store('huge', [float(i) for i in range(5000)])
    _store('small', 1)
    assert 'small' in A._cache


def test_key_locks_are_reaped_with_their_entries(monkeypatch):
    monkeypatch.setattr(A, 'CACHE_MAX_ENTRIES', 2)
    for k in ('a', 'b', 'c'):
        with A._cache_lock:
            A._cache_key_locks.setdefault(k, __import__('threading').Lock())
        _store(k, [0.0] * 10)
    assert 'a' not in A._cache_key_locks


def test_the_budget_is_a_sane_fraction_of_the_instance():
    """2 GB instance, and the app's own baseline is 150-250 MB."""
    assert 100 * 1024**2 <= A.CACHE_MAX_BYTES <= 800 * 1024**2
