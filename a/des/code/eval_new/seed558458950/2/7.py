import numpy as np
from collections import deque


def _most_frequent_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bfs_components(mask: np.ndarray):
    """Return connected components (4-neighborhood) as coordinate arrays."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or visited[r0, c0]:
                continue
            queue = deque([(r0, c0)])
            visited[r0, c0] = True
            pts = []
            while queue:
                r, c = queue.popleft()
                pts.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(points: np.ndarray):
    rs = points[:, 0]
    cs = points[:, 1]
    return int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())


def _single_component(mask: np.ndarray) -> bool:
    comps = _bfs_components(mask)
    return len(comps) == 1 and len(comps[0]) > 0


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    h0, w0 = occupied.shape
    if r < 0 or c < 0 or r + h > h0 or c + w > w0:
        return False
    return not occupied[r : r + h, c : c + w].any()


def _place_random_rect(rng, occupied, h: int, w: int, r_lo: int, r_hi: int, c_lo: int, c_hi: int):
    if r_lo > r_hi or c_lo > c_hi:
        return None
    for _ in range(200):
        r = int(rng.integers(r_lo, r_hi + 1))
        c = int(rng.integers(c_lo, c_hi + 1))
        if _can_place(occupied, r, c, h, w):
            return r, c
    return None


def _random_holes(rng, h: int, w: int, force_center_hole: bool):
    inner_h = h - 2
    inner_w = w - 2
    for _ in range(30):
        hole_prob = float(rng.uniform(0.18, 0.45))
        holes = (rng.random((inner_h, inner_w)) < hole_prob)
        if holes.size == 0:
            continue
        if not holes.any():
            rr = int(rng.integers(0, inner_h))
            cc = int(rng.integers(0, inner_w))
            holes[rr, cc] = True
        if force_center_hole:
            holes[inner_h // 2, inner_w // 2] = True
        full_mask = np.ones((h, w), dtype=bool)
        if inner_h > 0:
            full_mask[1:-1, 1:-1] = ~holes
        if not _single_component(full_mask):
            continue
        return holes
    # Conservative fallback to a simple cross-like hole set that is always connected-safe.
    holes = np.zeros((inner_h, inner_w), dtype=bool)
    holes[inner_h // 2, inner_w // 2] = True
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """Invert stamp holes onto matching envelope interiors by marker-centered alignment.
Input structure:
- The background is uniform color (most frequent pixel).
- Small 3x3 to 5x5 color stamps have many background holes and are not grey.
- Larger grey rectangular envelopes each contain one single marker pixel with the same color as its paired stamp.
Rule:
- Detect the background, envelope color, each envelope, and its marker color.
- For each marker, find the same-color stamp, align by matching stamp-center to marker position, and paint the envelope at stamp-hole coordinates with the marker color.
- Remove all stamp pixels to background while the marker position is first reset to grey; it is recolored only when it is a hole in the matched stamp.
Key insight:
- The transfer uses semantic inversion: holes in the stamp become ink on the envelope, so the marker acts as a center anchor and only background-holes are stamped.
"""
    grid = np.array(grid, copy=False)
    out = np.array(grid, copy=True)
    if grid.ndim != 2:
        return out

    bg = _most_frequent_color(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    if not colors:
        return out

    comps_by_color = {}
    counts_by_color = {}
    for c in colors:
        mask = grid == c
        comps = _bfs_components(mask)
        comps_by_color[c] = comps
        counts_by_color[c] = int(mask.sum())
    if not comps_by_color:
        return out

    # Greyscale envelope color is the color that clearly has large rectangle-like components.
    grey = None
    for c, _ in sorted(counts_by_color.items(), key=lambda it: it[1], reverse=True):
        comps = comps_by_color[c]
        ok = False
        for comp in comps:
            r0, r1, c0, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            area = len(comp)
            if h >= 6 and w >= 6 and area >= h * w - 4:
                ok = True
                break
        if ok:
            grey = c
            break
    if grey is None:
        # Fallback: the most frequent non-background color.
        grey = int(sorted(counts_by_color.items(), key=lambda it: it[1], reverse=True)[0][0])

    grey_comps = []
    for comp in comps_by_color.get(grey, []):
        r0, r1, c0, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = len(comp)
        if h >= 6 and w >= 6 and area >= h * w - 4:
            mask = np.zeros_like(grid, dtype=bool)
            mask[comp[:, 0], comp[:, 1]] = True
            grey_comps.append((comp, (r0, r1, c0, c1), mask))

    if not grey_comps:
        return out

    for comp, (r0, r1, c0, c1), envelope_mask in grey_comps:
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        marker_pos = np.argwhere((sub != bg) & (sub != grey))
        if marker_pos.size == 0:
            continue

        for mr_local, mc_local in marker_pos:
            mr = r0 + int(mr_local)
            mc = c0 + int(mc_local)
            mcol = int(grid[mr, mc])

            if mcol not in comps_by_color:
                continue

            # Choose stamp component by size/shape constraint for this color.
            stamp_comp = None
            stamp_bbox = None
            for scomp in comps_by_color[mcol]:
                sr0, sr1, sc0, sc1 = _bbox(scomp)
                sh = sr1 - sr0 + 1
                sw = sc1 - sc0 + 1
                if 3 <= sh <= 5 and 3 <= sw <= 5 and len(scomp) >= 9:
                    stamp_comp = scomp
                    stamp_bbox = (sr0, sr1, sc0, sc1)
                    break

            if stamp_comp is None:
                continue

            sr0, sr1, sc0, sc1 = stamp_bbox
            sh = sr1 - sr0 + 1
            sw = sc1 - sc0 + 1
            # Stamp center anchor.
            cr = (sr0 + sr1) // 2
            cc = (sc0 + sc1) // 2

            # Remove stamp object first.
            out[stamp_comp[:, 0], stamp_comp[:, 1]] = bg

            # Marker is removed from the envelope unless recolored by a hole.
            out[mr, mc] = grey

            for rr in range(sr0, sr1 + 1):
                for cc0 in range(sc0, sc1 + 1):
                    if grid[rr, cc0] != bg:
                        continue
                    tr = mr + (rr - cr)
                    tc = mc + (cc0 - cc)
                    if 0 <= tr < grid.shape[0] and 0 <= tc < grid.shape[1] and envelope_mask[tr, tc]:
                        out[tr, tc] = mcol

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(40):
        h = int(rng.integers(20, 31))
        w = int(rng.integers(20, 31))
        bg = 0
        grey = 1
        grid = np.full((h, w), bg, dtype=int)
        occupied = np.zeros_like(grid, dtype=bool)

        pair_count = int(rng.integers(1, 4))
        colors = [2, 3, 4, 5, 6, 7, 8, 9]
        rng.shuffle(colors)
        colors = colors[:pair_count]

        top_r0, top_r1 = 1, h // 2 - 2
        bot_r0, bot_r1 = h // 2 + 1, h - 2
        if top_r1 - top_r0 < 3 or bot_r1 - bot_r0 < 4:
            continue

        placed = 0
        for pair_idx, color in enumerate(colors):
            # Choose stamp size (odd sizes only to keep a unique center).
            stamp_h = int(rng.choice(np.array([3, 5], dtype=int)))
            stamp_w = int(rng.choice(np.array([3, 5], dtype=int)))

            # Place stamp in upper half.
            sr_cand = _place_random_rect(
                rng,
                occupied,
                stamp_h,
                stamp_w,
                top_r0,
                top_r1 - stamp_h + 1,
                1,
                w - stamp_w - 2,
            )
            if sr_cand is None:
                continue
            sr, sc = sr_cand

            # Fill stamp with color.
            grid[sr : sr + stamp_h, sc : sc + stamp_w] = color
            holes = _random_holes(rng, stamp_h, stamp_w, force_center_hole=(rng.random() < 0.55))
            for hr, hc in np.argwhere(holes):
                grid[sr + hr + 1, sc + hc + 1] = bg
            occupied[sr : sr + stamp_h, sc : sc + stamp_w] = grid[sr : sr + stamp_h, sc : sc + stamp_w] != bg

            placed += 1

            # Place one envelope in lower half with marker that can anchor this stamp.
            placed_envelope = False
            for _ in range(200):
                e_h_min = max(6, stamp_h + 2)
                e_w_min = max(6, stamp_w + 2)
                e_h_max = min(14, bot_r1 - bot_r0 + 1)
                e_w_max = min(14, w - 2)
                if e_h_min > e_h_max or e_w_min > e_w_max:
                    break
                env_h = int(rng.integers(e_h_min, e_h_max + 1))
                env_w = int(rng.integers(e_w_min, e_w_max + 1))

                er_cand = _place_random_rect(
                    rng,
                    occupied,
                    env_h,
                    env_w,
                    bot_r0,
                    bot_r1 - env_h + 1,
                    1,
                    w - env_w - 2,
                )
                if er_cand is None:
                    continue
                er, ec = er_cand

                mr_min = er + stamp_h // 2
                mr_max = er + env_h - 1 - stamp_h // 2
                mc_min = ec + stamp_w // 2
                mc_max = ec + env_w - 1 - stamp_w // 2
                if mr_min > mr_max or mc_min > mc_max:
                    continue

                mr = int(rng.integers(mr_min, mr_max + 1))
                mc = int(rng.integers(mc_min, mc_max + 1))

                # Render envelope.
                grid[er : er + env_h, ec : ec + env_w] = grey
                grid[mr, mc] = color

                # Update occupied and ensure no overlap.
                if (occupied[er : er + env_h, ec : ec + env_w] & ~ (grid[er : er + env_h, ec : ec + env_w] == bg)).any():
                    # This envelope placement touched an existing object due marker collision edge cases.
                    occupied[er : er + env_h, ec : ec + env_w] = False
                    grid[er : er + env_h, ec : ec + env_w] = bg
                    continue

                occupied[er : er + env_h, ec : ec + env_w] = True
                placed_envelope = True
                break

            if not placed_envelope:
                # If envelope failed, remove stamp and continue trying next color.
                grid[sr : sr + stamp_h, sc : sc + stamp_w] = bg
                occupied[sr : sr + stamp_h, sc : sc + stamp_w] = False
                placed -= 1
                continue

        if placed >= 1:
            break

    # Optional transpose for extra structural variety while keeping limits.
    if rng.random() < 0.35:
        if rng.random() < 0.5:
            grid = grid.T

    return grid.astype(np.int64)
