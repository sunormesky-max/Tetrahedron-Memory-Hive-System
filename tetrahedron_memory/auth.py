"""
Pure-Python JWT-like authentication and multi-tenant isolation.
Uses HMAC-SHA256 for signing — no external libraries.
"""

import base64
import enum
import hashlib
import hmac
import json
import secrets
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class TenantMode(enum.Enum):
    SINGLE = "single"
    MULTI = "multi"


class AuthManager:
    _TOKEN_TTL = 86400  # 24 hours

    def __init__(self, secret_key: str = None):
        self._secret = secret_key or secrets.token_hex(32)
        self._api_keys: Dict[str, Dict] = {}
        self._sessions: Dict[str, Dict] = {}

    def create_api_key(self, tenant_id: str, name: str, role: str = "agent",
                       quota: int = 100000) -> str:
        raw_key = secrets.token_urlsafe(32)
        key_hash = self.hash_key(raw_key)
        self._api_keys[key_hash] = {
            "tenant_id": tenant_id,
            "name": name,
            "role": role,
            "quota": quota,
            "created": time.time(),
        }
        return raw_key

    def hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def sign_payload(self, payload: Dict) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        h_b64 = base64.urlsafe_b64encode(
            json.dumps(header, separators=(",", ":")).encode()
        ).rstrip(b"=").decode()
        p_b64 = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode()
        ).rstrip(b"=").decode()
        signing_input = f"{h_b64}.{p_b64}"
        sig = hmac.new(
            self._secret.encode(), signing_input.encode(), hashlib.sha256
        ).digest()
        s_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
        return f"{signing_input}.{s_b64}"

    def verify_signature(self, token: str) -> Optional[Dict]:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        h_b64, p_b64, s_b64 = parts
        signing_input = f"{h_b64}.{p_b64}"
        expected_sig = hmac.new(
            self._secret.encode(), signing_input.encode(), hashlib.sha256
        ).digest()
        expected_b64 = base64.urlsafe_b64encode(expected_sig).rstrip(b"=").decode()
        if not hmac.compare_digest(expected_b64, s_b64):
            return None
        try:
            padding = 4 - len(p_b64) % 4
            if padding != 4:
                p_b64 += "=" * padding
            payload = json.loads(base64.urlsafe_b64decode(p_b64))
        except Exception:
            return None
        if payload.get("exp", 0) < time.time():
            return None
        return payload

    def create_token(self, api_key: str) -> Optional[str]:
        key_hash = self.hash_key(api_key)
        key_info = self._api_keys.get(key_hash)
        if key_info is None:
            return None
        payload = {
            "tenant_id": key_info["tenant_id"],
            "role": key_info["role"],
            "name": key_info["name"],
            "quota": key_info["quota"],
            "iat": int(time.time()),
            "exp": int(time.time()) + self._TOKEN_TTL,
            "jti": secrets.token_hex(8),
        }
        token = self.sign_payload(payload)
        if len(self._sessions) > 10000:
            self.cleanup_expired()
        self._sessions[token] = {
            "tenant_id": key_info["tenant_id"],
            "expires": payload["exp"],
            "role": key_info["role"],
        }
        return token

    def validate_token(self, token: str) -> Optional[Dict]:
        session = self._sessions.get(token)
        if session is None:
            payload = self.verify_signature(token)
            if payload is None:
                return None
            session = {
                "tenant_id": payload.get("tenant_id", ""),
                "expires": payload.get("exp", 0),
                "role": payload.get("role", "agent"),
            }
        if session["expires"] < time.time():
            self._sessions.pop(token, None)
            return None
        return session

    def revoke_token(self, token: str):
        self._sessions.pop(token, None)

    def cleanup_expired(self):
        now = time.time()
        expired = [t for t, s in self._sessions.items() if s["expires"] < now]
        for t in expired:
            del self._sessions[t]

    def list_api_keys(self) -> List[Dict[str, Any]]:
        keys: List[Dict[str, Any]] = []
        for key_hash, info in self._api_keys.items():
            keys.append({
                "key_hash": key_hash[:12] + "...",
                "tenant_id": info["tenant_id"],
                "name": info["name"],
                "role": info["role"],
                "quota": info["quota"],
                "created": info["created"],
                "age_seconds": round(time.time() - info["created"], 1),
            })
        return keys

    def rotate_key(self, old_key: str) -> Optional[str]:
        old_hash = self.hash_key(old_key)
        old_info = self._api_keys.get(old_hash)
        if old_info is None:
            return None
        new_raw_key = secrets.token_urlsafe(32)
        new_hash = self.hash_key(new_raw_key)
        new_info = dict(old_info)
        new_info["created"] = time.time()
        self._api_keys[new_hash] = new_info
        del self._api_keys[old_hash]
        for token, session in list(self._sessions.items()):
            if session.get("tenant_id") == old_info["tenant_id"]:
                if session.get("name") == old_info.get("name"):
                    del self._sessions[token]
        return new_raw_key


class TenantManager:
    def __init__(self):
        self._tenants: Dict[str, Dict] = {}
        self._tenant_prefixes: Dict[str, str] = {}

    def create_tenant(self, tenant_id: str, config: Dict) -> str:
        prefix = config.get("prefix", tenant_id[:8] + ":")
        self._tenants[tenant_id] = {
            "id": tenant_id,
            "config": config,
            "created": time.time(),
        }
        self._tenant_prefixes[tenant_id] = prefix
        return tenant_id

    def get_tenant(self, tenant_id: str) -> Optional[Dict]:
        return self._tenants.get(tenant_id)

    def get_prefix(self, tenant_id: str) -> str:
        return self._tenant_prefixes.get(tenant_id, tenant_id[:8] + ":")

    def check_quota(self, tenant_id: str, current_count: int) -> bool:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return False
        max_q = tenant.get("config", {}).get("max_memories", 100000)
        return current_count < max_q

    def list_tenants(self) -> List[Dict]:
        return list(self._tenants.values())


class TenantAwareField:
    """
    Wrapper around HoneycombNeuralField that adds tenant awareness.
    In SINGLE mode: passes through directly, zero overhead.
    In MULTI mode: prefixes all node IDs and isolates operations.
    """

    def __init__(self, field: Any, mode: "TenantMode" = TenantMode.SINGLE):
        self._field = field
        self._mode = mode
        self._tenant_prefixes: Dict[str, str] = {}
        self._active_tenant: Optional[str] = None

    def _tenant_label(self, tenant_id: str) -> str:
        return f"__tenant_{tenant_id}__"

    def store(self, content: str, labels: Optional[List[str]] = None,
              weight: float = 1.0, metadata: Optional[Dict] = None,
              tenant_id: Optional[str] = None) -> str:
        effective_tenant = tenant_id or self._active_tenant
        if self._mode == TenantMode.SINGLE or effective_tenant is None:
            return self._field.store(content, labels=labels, weight=weight, metadata=metadata)
        isolated_labels = [self._tenant_label(effective_tenant)] + (labels or [])
        return self._field.store(content, labels=isolated_labels, weight=weight, metadata=metadata)

    def query(self, query_text: str, k: int = 5, labels: Optional[List[str]] = None,
              tenant_id: Optional[str] = None) -> List[Dict]:
        effective_tenant = tenant_id or self._active_tenant
        if self._mode == TenantMode.SINGLE or effective_tenant is None:
            return self._field.query(query_text, k=k, labels=labels)
        isolated_labels = [self._tenant_label(effective_tenant)] + (labels or [])
        results = self._field.query(query_text, k=k, labels=isolated_labels)
        prefix = self._tenant_label(effective_tenant)
        for r in results:
            r["labels"] = [l for l in r.get("labels", []) if l != prefix]
        return results

    def delete(self, tetra_id: str, tenant_id: Optional[str] = None) -> bool:
        effective_tenant = tenant_id or self._active_tenant
        if self._mode == TenantMode.SINGLE or effective_tenant is None:
            node = self._field._nodes.get(tetra_id)
            if node is None:
                return False
            self._field._clear_node(tetra_id, node)
            return True
        node = self._field._nodes.get(tetra_id)
        if node is None:
            return False
        prefix = self._tenant_label(effective_tenant)
        if prefix not in node.labels:
            return False
        self._field._clear_node(tetra_id, node)
        return True

    def update(self, tetra_id: str, content: Optional[str] = None,
               weight: Optional[float] = None, labels: Optional[List[str]] = None,
               tenant_id: Optional[str] = None) -> bool:
        effective_tenant = tenant_id or self._active_tenant
        if self._mode == TenantMode.SINGLE or effective_tenant is None:
            node = self._field._nodes.get(tetra_id)
            if node is None:
                return False
            if content is not None:
                node.content = content
            if weight is not None:
                node.weight = max(0.1, min(10.0, float(weight)))
            if labels is not None:
                node.labels = list(labels)
            return True
        node = self._field._nodes.get(tetra_id)
        if node is None:
            return False
        prefix = self._tenant_label(effective_tenant)
        if prefix not in node.labels:
            return False
        if content is not None:
            node.content = content
        if weight is not None:
            node.weight = max(0.1, min(10.0, float(weight)))
        if labels is not None:
            node.labels = [prefix] + list(labels)
        return True

    def stats(self, tenant_id: Optional[str] = None) -> Dict:
        effective_tenant = tenant_id or self._active_tenant
        if self._mode == TenantMode.SINGLE or effective_tenant is None:
            return self._field.stats()
        all_stats = self._field.stats()
        prefix = self._tenant_label(effective_tenant)
        tenant_count = 0
        tenant_weight_sum = 0.0
        for n in self._field._nodes.values():
            if n.is_occupied and prefix in n.labels:
                tenant_count += 1
                tenant_weight_sum += n.weight
        return {
            "tenant_id": effective_tenant,
            "tenant_memories": tenant_count,
            "tenant_weight_sum": round(tenant_weight_sum, 2),
            "total_nodes": all_stats.get("total_nodes", 0),
        }

    def browse(self, direction: str = "newest", limit: int = 20,
               labels: Optional[List[str]] = None, min_weight: float = 0.0,
               offset: int = 0, tenant_id: Optional[str] = None) -> tuple:
        effective_tenant = tenant_id or self._active_tenant
        if self._mode == TenantMode.SINGLE or effective_tenant is None:
            return self._field.browse_timeline(direction, limit, labels, min_weight, offset)
        prefix = self._tenant_label(effective_tenant)
        isolated_labels = [prefix] + (labels or [])
        items, total = self._field.browse_timeline(direction, limit, isolated_labels, min_weight, offset)
        for item in items:
            item["labels"] = [l for l in item.get("labels", []) if l != prefix]
        return items, total

    def export_tenant(self, tenant_id: str) -> Dict:
        if self._mode == TenantMode.SINGLE:
            return self._field.export_full_state()
        prefix = self._tenant_label(tenant_id)
        nodes = {}
        for nid, node in self._field._nodes.items():
            if node.is_occupied and prefix in node.labels:
                nodes[nid] = {
                    "content": node.content,
                    "labels": [l for l in node.labels if l != prefix],
                    "weight": float(node.weight),
                    "activation": float(node.activation),
                    "metadata": dict(node.metadata) if node.metadata else {},
                    "creation_time": float(node.creation_time),
                    "centroid": node.position.tolist(),
                }
        return {"tenant_id": tenant_id, "nodes": nodes}

    def switch_mode(self, mode: "TenantMode"):
        self._mode = mode

    @property
    def mode(self) -> "TenantMode":
        return self._mode

    def get_tenant_stats(self, tenant_id: Optional[str] = None) -> Dict:
        return self.stats(tenant_id)

    def switch_tenant(self, tenant_id: Optional[str]) -> None:
        self._active_tenant = tenant_id

    def export_all_tenants(self) -> Dict[str, Dict]:
        if self._mode == TenantMode.SINGLE:
            return {"_single": self._field.export_full_state()}
        tenant_prefix_set: set = set()
        for node in self._field._nodes.values():
            if not node.is_occupied:
                continue
            for label in node.labels:
                if label.startswith("__tenant_") and label.endswith("__"):
                    tid = label[len("__tenant_"):-2]
                    if tid:
                        tenant_prefix_set.add(tid)
        result: Dict[str, Dict] = {}
        for tid in sorted(tenant_prefix_set):
            result[tid] = self.export_tenant(tid)
        return result

    def import_tenant(self, tenant_id: str, data: Dict) -> int:
        if self._mode == TenantMode.SINGLE:
            if "nodes" in data and hasattr(data, "items"):
                count = 0
                for nid, ndata in data["nodes"].items():
                    self._field.store(
                        content=ndata.get("content", ""),
                        labels=ndata.get("labels", []),
                        weight=ndata.get("weight", 1.0),
                        metadata=ndata.get("metadata"),
                    )
                    count += 1
                return count
            return 0
        prefix = self._tenant_label(tenant_id)
        nodes = data.get("nodes", {})
        imported_count = 0
        for nid, ndata in nodes.items():
            labels = ndata.get("labels", [])
            isolated_labels = [prefix] + labels
            self._field.store(
                content=ndata.get("content", ""),
                labels=isolated_labels,
                weight=ndata.get("weight", 1.0),
                metadata=ndata.get("metadata"),
            )
            imported_count += 1
        return imported_count

    def get_tenant_summary(self) -> Dict[str, Dict[str, Any]]:
        if self._mode == TenantMode.SINGLE:
            all_stats = self._field.stats()
            return {
                "_single": {
                    "total_memories": all_stats.get("occupied_nodes", 0),
                    "total_weight": round(
                        sum(
                            n.weight
                            for n in self._field._nodes.values()
                            if n.is_occupied
                        ),
                        2,
                    ),
                    "avg_activation": round(all_stats.get("avg_activation", 0), 4),
                }
            }
        tenant_data: Dict[str, Dict[str, Any]] = {}
        for node in self._field._nodes.values():
            if not node.is_occupied:
                continue
            for label in node.labels:
                if label.startswith("__tenant_") and label.endswith("__"):
                    tid = label[len("__tenant_"):-2]
                    if not tid:
                        continue
                    if tid not in tenant_data:
                        tenant_data[tid] = {
                            "memory_count": 0,
                            "total_weight": 0.0,
                            "max_weight": 0.0,
                            "labels_used": set(),
                        }
                    td = tenant_data[tid]
                    td["memory_count"] += 1
                    td["total_weight"] += node.weight
                    td["max_weight"] = max(td["max_weight"], node.weight)
                    for l in node.labels:
                        if not l.startswith("__"):
                            td["labels_used"].add(l)
        summary: Dict[str, Dict[str, Any]] = {}
        for tid, td in sorted(tenant_data.items()):
            summary[tid] = {
                "memory_count": td["memory_count"],
                "total_weight": round(td["total_weight"], 2),
                "avg_weight": round(td["total_weight"] / max(1, td["memory_count"]), 3),
                "max_weight": round(td["max_weight"], 2),
                "unique_labels": sorted(td["labels_used"]),
            }
        return summary
