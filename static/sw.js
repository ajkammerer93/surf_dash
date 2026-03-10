const CACHE_NAME = 'fsf-v1';
const APP_SHELL = ['/', '/static/manifest.json', '/static/icons/icon-192.png', '/static/icons/icon-512.png'];
const CDN_HOSTS = ['unpkg.com', 'cdn.jsdelivr.net', 'cdnjs.cloudflare.com'];
const API_CACHE = 'fsf-api-v1';
const API_MAX_AGE = 15 * 60 * 1000; // 15 minutes

self.addEventListener('install', function(event) {
    event.waitUntil(
        caches.open(CACHE_NAME).then(function(cache) {
            return cache.addAll(APP_SHELL);
        })
    );
    self.skipWaiting();
});

self.addEventListener('activate', function(event) {
    event.waitUntil(
        caches.keys().then(function(keys) {
            return Promise.all(
                keys.filter(function(k) { return k !== CACHE_NAME && k !== API_CACHE; })
                    .map(function(k) { return caches.delete(k); })
            );
        })
    );
    self.clients.claim();
});

self.addEventListener('fetch', function(event) {
    var url = new URL(event.request.url);

    // API requests: network-first with 15min cache fallback
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(
            fetch(event.request).then(function(response) {
                if (response.ok) {
                    var clone = response.clone();
                    caches.open(API_CACHE).then(function(cache) {
                        var headers = new Headers(clone.headers);
                        headers.set('sw-cached-at', Date.now().toString());
                        var cachedResponse = new Response(clone.body, {
                            status: clone.status,
                            statusText: clone.statusText,
                            headers: headers
                        });
                        cache.put(event.request, cachedResponse);
                    });
                }
                return response;
            }).catch(function() {
                return caches.open(API_CACHE).then(function(cache) {
                    return cache.match(event.request).then(function(cached) {
                        if (cached) {
                            var cachedAt = parseInt(cached.headers.get('sw-cached-at') || '0');
                            if (Date.now() - cachedAt < API_MAX_AGE) return cached;
                        }
                        return new Response(JSON.stringify({error: 'offline'}), {
                            status: 503,
                            headers: {'Content-Type': 'application/json'}
                        });
                    });
                });
            })
        );
        return;
    }

    // CDN libs: cache-first
    if (CDN_HOSTS.some(function(h) { return url.hostname === h; })) {
        event.respondWith(
            caches.match(event.request).then(function(cached) {
                if (cached) return cached;
                return fetch(event.request).then(function(response) {
                    if (response.ok) {
                        var clone = response.clone();
                        caches.open(CACHE_NAME).then(function(cache) {
                            cache.put(event.request, clone);
                        });
                    }
                    return response;
                });
            })
        );
        return;
    }

    // App shell: stale-while-revalidate
    if (url.origin === self.location.origin) {
        event.respondWith(
            caches.match(event.request).then(function(cached) {
                var fetchPromise = fetch(event.request).then(function(response) {
                    var clone = response.clone();
                    caches.open(CACHE_NAME).then(function(cache) {
                        cache.put(event.request, clone);
                    });
                    return response;
                });
                return cached || fetchPromise;
            })
        );
        return;
    }
});
