from __future__ import annotations

from collections import deque
import random

import numpy as np

BG = 0
RED = 2


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _neighbors4(r: int, c: int):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not bool(mask[r0, c0]) or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((int(r), int(c)))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _to_set(comp: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in comp}


def _find_sprites(g: np.ndarray) -> list[dict[str, int]]:
    sprites: list[dict[str, int]] = []
    for comp in _components(g != BG):
        comp_set = _to_set(comp)
        if len(comp_set) != 9:
            continue
        rows = [r for r, _ in comp_set]
        cols = [c for _, c in comp_set]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 != 2 or c1 - c0 != 2:
            continue
        if comp_set != {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}:
            continue
        colors = [int(g[r, c]) for r, c in comp_set]
        uniq = sorted(set(colors))
        if len(uniq) != 2:
            continue
        center = (r0 + 1, c0 + 1)
        center_color = int(g[center])
        outer_color = uniq[0] if uniq[1] == center_color else uniq[1]
        if colors.count(center_color) != 1 or int(g[center]) == outer_color:
            continue
        if colors.count(outer_color) != 8:
            continue
        sprites.append(
            {
                "outer": outer_color,
                "inner": center_color,
                "r0": int(r0),
                "c0": int(c0),
                "coords": comp_set,
            }
        )
    return sprites


def _find_bars(g: np.ndarray) -> dict[int, dict[str, int]]:
    bars: dict[int, dict[str, int]] = {}
    h, w = g.shape
    for comp in _components(g != BG):
        comp_set = _to_set(comp)
        rows = [r for r, _ in comp_set]
        cols = [c for _, c in comp_set]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        H = r1 - r0 + 1
        W = c1 - c0 + 1
        if H != 3 or W < 10:
            continue
        if len(comp_set) != H * W:
            continue
        if not (3 <= r0 < h - 1):
            continue
        top = {int(v) for v in g[r0, c0:c1 + 1]}
        bot = {int(v) for v in g[r1, c0:c1 + 1]}
        mid = {int(v) for v in g[r0 + 1, c0:c1 + 1]}
        if len(top) != 1 or len(bot) != 1:
            continue
        boundary = int(next(iter(top)))
        if int(next(iter(bot))) != boundary:
            continue
        if boundary in mid:
            continue
        if len(mid) != 1:
            continue
        fill = int(next(iter(mid)))
        if fill in bars:
            # fill colors are unique by puzzle definition.
            continue
        bars[fill] = {
            "top": int(r0),
            "c0": int(c0),
            "c1": int(c1),
            "boundary": int(boundary),
            "fill": int(fill),
        }
    return bars


def _ring_layer(center: tuple[int, int], dist: int) -> list[tuple[int, int]]:
    r, c = map(int, center)
    d = int(dist)
    layer: list[tuple[int, int]] = []
    for rr in range(r - d, r + d + 1):
        layer.append((rr, c - d))
        layer.append((rr, c + d))
    for cc in range(c - d + 1, c + d):
        layer.append((r - d, cc))
        layer.append((r + d, cc))
    # unique preserve order for tiny rings.
    seen: set[tuple[int, int]] = set()
    uniq: list[tuple[int, int]] = []
    for p in layer:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match two-tone sprites to color bars, dock them at the top of the matched bar,
    recolor by swap, then grow even-distance concentric rings with source-order
    ownership blocking.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    sprites = _find_sprites(g)
    bars = _find_bars(g)
    if not sprites or not bars:
        return out

    bar_mask = np.zeros((H, W), dtype=bool)
    for bar in bars.values():
        bar_mask[bar["top"], bar["c0"] : bar["c1"] + 1] = True
        bar_mask[bar["top"] + 1, bar["c0"] : bar["c1"] + 1] = True
        bar_mask[bar["top"] + 2, bar["c0"] : bar["c1"] + 1] = True

    docked: list[dict[str, int | tuple[int, int]]] = []
    for sp in sprites:
        outer = int(sp["outer"])
        if outer not in bars:
            continue
        bar = bars[outer]
        target_top = int(bar["top"]) - 2
        target_left = int(sp["c0"])
        if target_top < 0 or target_left < 0:
            continue
        if target_left + 2 >= W:
            continue
        if target_left < int(bar["c0"]) or target_left + 2 > int(bar["c1"]):
            continue

        for r, c in sp["coords"]:
            out[r, c] = BG

        for dr in range(3):
            for dc in range(3):
                rr = target_top + dr
                cc = target_left + dc
                if not _in_bounds((H, W), rr, cc):
                    continue
                out[rr, cc] = int(bar["fill"]) if (dr == 1 and dc == 1) else int(bar["boundary"])
        docked.append(
            {
                "center": (int(target_top + 1), int(target_left + 1)),
                "color": int(bar["fill"]),
                "left": int(target_left),
            }
        )

    docked.sort(key=lambda d: d["left"])
    owner = np.zeros((H, W), dtype=int)

    for idx, src in enumerate(docked, start=1):
        center = (int(src["center"][0]), int(src["center"][1]))
        color = int(src["color"])
        max_d = max(H, W)
        for dist in range(2, max_d + 1, 2):
            layer = _ring_layer(center, dist)
            blocked = False

            for rr, cc in layer:
                if not _in_bounds((H, W), rr, cc):
                    blocked = True
                    break
                owner_id = int(owner[rr, cc])
                if owner_id not in (0, idx):
                    blocked = True
                    break

            if blocked:
                break

            for rr, cc in layer:
                if bar_mask[rr, cc]:
                    continue
                out[rr, cc] = color
                owner[rr, cc] = idx

    return out


def generate_input() -> np.ndarray:
    for _ in range(5000):
        H = random.randint(15, 20)
        W = random.randint(15, 20)
        if W < 12:
            continue
        g = np.full((H, W), BG, dtype=int)

        n = random.randint(2, 3)
        all_colors = [1, 3, 4, 5, 6, 7, 8, 9]
        fill_colors = random.sample(all_colors, n)
        boundary_colors = random.sample([c for c in all_colors if c not in fill_colors], n)

        # Bars start at row 7 and use full width minus border columns for stable matching.
        row = 7
        bars = []
        c0, c1 = 1, W - 2
        for i in range(n):
            if row + 2 >= H:
                break
            bars.append((row, row + 2, fill_colors[i], boundary_colors[i], c0, c1))
            row += 3 + random.randint(1, 2)
        if len(bars) != n:
            continue

        for row_top, row_bot, fill, boundary, left, right in bars:
            g[row_top, left : right + 1] = boundary
            g[row_top + 1, left : right + 1] = fill
            g[row_bot, left : right + 1] = boundary

        # Place 2~3 top-area 3x3 two-tone sprites with distinct outer colors.
        candidate_cols = [c for c in range(1, W - 2)]
        random.shuffle(candidate_cols)
        sprite_cols: list[int] = []
        for c in candidate_cols:
            if len(sprite_cols) >= n:
                break
            if any(abs(c - x) < 3 for x in sprite_cols):
                continue
            sprite_cols.append(c)
        if len(sprite_cols) != n:
            continue

        for i, c in enumerate(sorted(sprite_cols)):
            outer = int(fill_colors[i])
            inner = int(random.choice([c2 for c2 in all_colors if c2 != outer]))
            r0 = random.randint(0, 4)
            for rr in range(r0, r0 + 3):
                for cc in range(c, c + 3):
                    g[rr, cc] = outer
            g[r0 + 1, c + 1] = inner

        try:
            out = main(g)
        except Exception:
            continue
        if out.shape != g.shape:
            continue

        # Ensure the generated instance has an observable ring phase.
        if not np.any(out != g):
            continue

        return g

    raise RuntimeError("failed to generate valid puzzle 7 input")

