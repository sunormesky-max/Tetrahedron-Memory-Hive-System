"use strict";

Object.defineProperty(exports, "__esModule", { value: true });
exports.createTetraMemManager = createTetraMemManager;

function createTetraMemManager(config, params) {
  const baseUrl = (config && config.url) || "http://127.0.0.1:8000";
  const timeout = (config && config.timeout) || 30000;
  const apiKey = (config && config.apiKey) || null;

  let logger = null;
  let loggerPromise = null;

  function getLogger() {
    if (logger) return logger;
    if (!loggerPromise) {
      loggerPromise = import(params.loggerImportPath || "@opencode/subsystem-logger")
        .then(function (mod) {
          logger = mod.createLogger("tetramem");
          return logger;
        })
        .catch(function () {
          logger = {
            info: function () {
              var args = Array.prototype.slice.call(arguments);
              console.log.apply(console, ["[tetramem]"].concat(args));
            },
            warn: function () {
              var args = Array.prototype.slice.call(arguments);
              console.warn.apply(console, ["[tetramem]"].concat(args));
            },
            error: function () {
              var args = Array.prototype.slice.call(arguments);
              console.error.apply(console, ["[tetramem]"].concat(args));
            },
            debug: function () {
              var args = Array.prototype.slice.call(arguments);
              console.log.apply(console, ["[tetramem:debug]"].concat(args));
            },
          };
          return logger;
        });
    }
    return loggerPromise;
  }

  function log(level) {
    var args = Array.prototype.slice.call(arguments, 1);
    getLogger().then(function (l) {
      if (l[level]) l[level].apply(l, args);
    });
  }

  function fetchApi(path, options) {
    var url = baseUrl.replace(/\/+$/, "") + path;
    var headers = Object.assign({}, (options && options.headers) || {}, {
      "Content-Type": "application/json",
    });
    if (apiKey) {
      headers["Authorization"] = "Bearer " + apiKey;
    }
    var controller = new AbortController();
    var timer = setTimeout(function () {
      controller.abort();
    }, timeout);
    return fetch(url, {
      method: (options && options.method) || "GET",
      headers: headers,
      body: (options && options.body) ? JSON.stringify(options.body) : undefined,
      signal: controller.signal,
    })
      .then(function (res) {
        clearTimeout(timer);
        if (!res.ok) {
          throw new Error("TetraMem API " + res.status + " for " + path);
        }
        return res.json();
      })
      .catch(function (err) {
        clearTimeout(timer);
        throw err;
      });
  }

  function search(query, options) {
    var limit = (options && options.limit) || 10;
    var threshold = (options && options.threshold) || 0.0;
    log("info", "search: query=%s limit=%d", query, limit);
    return fetchApi("/api/v1/search", {
      method: "POST",
      body: {
        query: query,
        limit: limit,
        threshold: threshold,
        filters: (options && options.filters) || {},
      },
    })
      .then(function (data) {
        if (!data || !Array.isArray(data.results)) {
          return [];
        }
        return data.results.map(function (r) {
          return {
            text: r.text || r.content || "",
            score: r.score || 0,
            source: r.source || "tetramem",
            metadata: r.metadata || {},
          };
        });
      })
      .catch(function (err) {
        log("warn", "search failed: %s", err.message);
        return [];
      });
  }

  function readFile(filePath, options) {
    log("info", "readFile: path=%s", filePath);
    return fetchApi("/api/v1/read", {
      method: "POST",
      body: {
        path: filePath,
        encoding: (options && options.encoding) || "utf-8",
      },
    })
      .then(function (data) {
        return {
          text: data.content || data.text || "",
          score: 1.0,
          source: filePath,
          metadata: data.metadata || {},
        };
      })
      .catch(function (err) {
        log("warn", "readFile failed: %s", err.message);
        return {
          text: "",
          score: 0,
          source: filePath,
          metadata: { error: err.message },
        };
      });
  }

  function status() {
    log("info", "status check");
    return fetchApi("/api/v1/status")
      .then(function (data) {
        return {
          ok: true,
          backend: "tetramem",
          connected: true,
          details: data,
        };
      })
      .catch(function (err) {
        return {
          ok: false,
          backend: "tetramem",
          connected: false,
          error: err.message,
        };
      });
  }

  function sync(options) {
    log("info", "sync started");
    return fetchApi("/api/v1/sync", {
      method: "POST",
      body: {
        paths: (options && options.paths) || [],
        fullSync: (options && options.fullSync) || false,
      },
    })
      .then(function (data) {
        return {
          ok: true,
          synced: data.synced || 0,
          errors: data.errors || 0,
        };
      })
      .catch(function (err) {
        log("error", "sync failed: %s", err.message);
        return {
          ok: false,
          synced: 0,
          errors: 1,
          error: err.message,
        };
      });
  }

  function probeEmbeddingAvailability() {
    log("info", "probeEmbeddingAvailability");
    return fetchApi("/api/v1/capabilities/embeddings")
      .then(function (data) {
        return {
          available: data.available !== false,
          model: data.model || "unknown",
          dimensions: data.dimensions || 0,
        };
      })
      .catch(function () {
        return {
          available: false,
          model: null,
          dimensions: 0,
        };
      });
  }

  function probeVectorAvailability() {
    log("info", "probeVectorAvailability");
    return fetchApi("/api/v1/capabilities/vectors")
      .then(function (data) {
        return {
          available: data.available !== false,
          engine: data.engine || "unknown",
          indexSize: data.indexSize || 0,
        };
      })
      .catch(function () {
        return {
          available: false,
          engine: null,
          indexSize: 0,
        };
      });
  }

  function close() {
    log("info", "close");
    return Promise.resolve();
  }

  return {
    search: search,
    readFile: readFile,
    status: status,
    sync: sync,
    probeEmbeddingAvailability: probeEmbeddingAvailability,
    probeVectorAvailability: probeVectorAvailability,
    close: close,
  };
}
