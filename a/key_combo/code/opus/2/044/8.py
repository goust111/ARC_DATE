import numpy as np
from collections import deque

BG = 0


def _components(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not vis[r, c]:
                q = deque([(r, c)])
                vis[r, c] = True
                pts = []
                while q:
                    rr, cc = q.popleft()
                    pts.append((rr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                            vis[nr, nc] = True
                            q.append((nr, nc))
                comps.append(np.asarray(pts, dtype=np.int32))
    return comps


def _bbox(pts: np.ndarray):
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    h, w = g.shape

    # Parse vertical legend in columns 0 and 1.
    mapping = {}
    for r in range(h):
        a = int(g[r, 0])
        b = int(g[r, 1])
        if a != BG and b != BG:
            mapping[a] = b

    out = g.copy()

    # Redraw only main area; squares are re-created from detected components.
    out[:, 2:] = BG
    out[:2, :] = g[:2, :]

    objects = _components(g[:, 2:] != BG)
    candidates = []
    for pts_local in objects:
        rr = pts_local[:, 0]
        cc = pts_local[:, 1]
        r0, c0 = rr.min(), cc.min()
        r1, c1 = rr.max(), cc.max()
        h0 = r1 - r0 + 1
        w0 = c1 - c0 + 1
        if h0 != w0 or (h0 % 2 == 0):
            continue

        color = int(g[rr[0], cc[0] + 2])
        if color not in mapping:
            continue

        # ensure full solid odd square.
        if pts_local.shape[0] != h0 * w0:
            continue

        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        old_n = h0
        new_n = 7 if old_n == 1 else old_n - 2
        candidates.append((new_n, center_r + 0, center_c + 2, mapping[color]))

    # Draw largest first; smaller ones on top.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    for new_n, cr, cc, color in candidates:
        half = new_n // 2
        r0, r1 = cr - half, cr + half
        c0, c1 = cc - half, cc + half
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    return out


def _cycle_size_to_color(size: int) -> int:
    return 7 if size == 1 else size - 2


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = 15
    w = 15
    g = np.zeros((h, w), dtype=np.int64)

    # Build legend lines with blank separator rows.
    orig_colors = [1, 2, 3, 7]
    new_colors = [7, 3, 5, 9]
    for i, (a, b) in enumerate(zip(orig_colors, new_colors)):
        r = 2 * i
        if r >= 2:
            break
    r = 0
    for a, b in zip(orig_colors, new_colors):
        g[r, 0] = a
        g[r, 1] = b
        r += 2

    # Ensure we have at least one size-1 square.
    sizes = [1, 3, 5, 7]
    rng.shuffle(sizes)

    used_centers = []
    for s in sizes[:3]:
        ns = _cycle_size_to_color(s)
        half_ns = ns // 2
        while True:
            cr = int(rng.integers(max(2 + half_ns, 2), h - half_ns))
            cc = int(rng.integers(max(2 + half_ns, 2), w - half_ns))
            if all(abs(cr - x) + abs(cc - y) >= 4 for x, y in used_centers):
                used_centers.append((cr, cc))
                break

        color = int(orig_colors[sizes.index(s) % len(orig_colors)])
        n0 = s
        r0 = cr - n0 // 2
        c0 = cc - n0 // 2
        r1 = r0 + n0
        c1 = c0 + n0
        g[r0:r1 + 1, c0:c1 + 1] = color

    return g
