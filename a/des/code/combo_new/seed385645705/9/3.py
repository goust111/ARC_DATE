from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GEN_BG = 0
GEN_SEP = 5
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 600
PLACE_ATTEMPTS = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in values.reshape(-1))
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "expected a unique dominant color"
    return int(winners[0])


def _separator_color(g: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in g.reshape(-1) if int(v) != bg)
    assert counts, "expected at least one non-background color"
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "separator color must be structurally unique"
    return int(winners[0])


def _analyze_frame(g: np.ndarray, sep: int, points: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(sub[border] == sep):
        return None

    sep_mask = sub == sep
    extra = sep_mask & (~border)

    divider_cols: list[int] = []
    for j in range(1, w - 1):
        if np.all(sub[:, j] == sep):
            divider_cols.append(j)

    if extra.any():
        allowed = np.zeros_like(extra)
        for j in divider_cols:
            allowed[1:-1, j] = True
        if not np.array_equal(extra, allowed):
            return None
        kind = "palette_candidate"
    else:
        kind = "simple_frame"

    return {
        "bbox": (r0, c0, r1, c1),
        "kind": kind,
        "divider_cols": divider_cols,
        "touches_edge": r0 == 0 or c0 == 0 or r1 == g.shape[0] - 1 or c1 == g.shape[1] - 1,
    }


def _parse_palette(g: np.ndarray, bg: int, sep: int, frame: dict) -> dict | None:
    r0, c0, r1, c1 = frame["bbox"]
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    if h < 5:
        return None

    cuts = [0] + list(frame["divider_cols"]) + [w - 1]
    if len(cuts) < 3:
        return None

    entries: dict[int, np.ndarray] = {}
    widths: list[int] = []
    pattern_h = h - 3
    if pattern_h < 2:
        return None

    for left, right in zip(cuts, cuts[1:]):
        if right - left <= 1:
            return None
        j0 = left + 1
        j1 = right - 1
        widths.append(j1 - j0 + 1)

        strip = sub[1, j0 : j1 + 1]
        strip_colors = set(map(int, np.unique(strip)))
        if len(strip_colors) != 1:
            return None
        label_color = strip_colors.pop()
        if label_color in (bg, sep):
            return None

        pattern = sub[2 : h - 1, j0 : j1 + 1]
        values = set(map(int, np.unique(pattern)))
        if not values <= {bg, label_color}:
            return None
        if values != {bg, label_color}:
            return None
        if label_color in entries:
            return None
        entries[label_color] = pattern.copy()

    if len(set(widths)) != 1 or len(entries) < 2:
        return None
    return {"bbox": frame["bbox"], "entries": entries}


def _tile_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ph, pw = pattern.shape
    h, w = shape
    rr = np.arange(h) % ph
    cc = np.arange(w) % pw
    return pattern[rr[:, None], cc[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each labeled grey-framed container by looking up its color in the edge palette and tiling the matching binary pattern.

    Input structure:
    - A black background contains several thin grey rectangular frames. Some frames have a one-row solid color strip directly above their top border; these are the containers to fill.
    - On the right edge or bottom edge there is a special grey-framed palette rectangle. Its interior is split into equal-width subpanels by grey divider columns.
    - In each palette subpanel, the top interior row is a solid label color strip and the rows below form a small two-color pattern using that label color and the background.

    Rule:
    - Identify the edge-touching palette frame and read each subpanel as a mapping from label color to binary pattern.
    - For every simple grey frame with a solid strip immediately above it, use the strip color to select the matching palette pattern.
    - Tile that pattern from the container's interior top-left corner until the full interior is covered, clipping any overhang.
    - Remove all label strips and erase the palette contents back to background while keeping grey structural borders and divider lines.

    Key insight:
    - The colored strips are not decorative: they are keys into a separate palette legend, and the palette must be decoded before any container can be filled correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _unique_mode(g)
    sep = _separator_color(g, bg)

    frames: list[dict] = []
    for comp in _components(g == sep):
        info = _analyze_frame(g, sep, comp)
        if info is not None:
            frames.append(info)

    palette_candidates: list[dict] = []
    simple_frames: list[dict] = []
    for frame in frames:
        if frame["kind"] == "palette_candidate" and frame["touches_edge"]:
            parsed = _parse_palette(g, bg, sep, frame)
            if parsed is not None:
                palette_candidates.append(parsed)
        elif frame["kind"] == "simple_frame":
            simple_frames.append(frame)

    assert palette_candidates, "expected one edge palette"
    palette = max(
        palette_candidates,
        key=lambda item: (item["bbox"][2] - item["bbox"][0] + 1) * (item["bbox"][3] - item["bbox"][1] + 1),
    )
    mapping = palette["entries"]

    out = g.copy()

    for frame in simple_frames:
        r0, c0, r1, c1 = frame["bbox"]
        if r0 == 0:
            continue
        strip = out[r0 - 1, c0 : c1 + 1]
        strip_colors = set(map(int, np.unique(strip)))
        if len(strip_colors) != 1:
            continue
        label_color = strip_colors.pop()
        if label_color in (bg, sep) or label_color not in mapping:
            continue

        interior_h = r1 - r0 - 1
        interior_w = c1 - c0 - 1
        if interior_h <= 0 or interior_w <= 0:
            continue

        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(mapping[label_color], (interior_h, interior_w))
        out[r0 - 1, c0 : c1 + 1] = bg

    pr0, pc0, pr1, pc1 = palette["bbox"]
    palette_region = out[pr0 + 1 : pr1, pc0 + 1 : pc1]
    palette_region[palette_region != sep] = bg
    return out


def _pattern_mask(h: int, w: int) -> np.ndarray:
    rr, cc = np.indices((h, w))
    for _ in range(40):
        kind = random.choice(
            [
                "checker",
                "vstripe",
                "hstripe",
                "diag",
                "antidiag",
                "corners",
                "plus",
                "random",
            ]
        )
        if kind == "checker":
            mask = (rr + cc + random.randint(0, 1)) % 2 == 0
        elif kind == "vstripe":
            period = random.choice([2, 3])
            mask = cc % period == random.randrange(period)
        elif kind == "hstripe":
            period = random.choice([2, 3])
            mask = rr % period == random.randrange(period)
        elif kind == "diag":
            period = random.choice([2, 3])
            mask = (rr - cc) % period == random.randrange(period)
        elif kind == "antidiag":
            period = random.choice([2, 3])
            mask = (rr + cc) % period == random.randrange(period)
        elif kind == "corners":
            mask = np.zeros((h, w), dtype=bool)
            mask[0, 0] = True
            mask[0, -1] = True
            mask[-1, 0] = True
            mask[-1, -1] = True
            if h > 2 and w > 2 and random.choice([False, True]):
                mask[h // 2, w // 2] = True
        elif kind == "plus":
            mask = np.zeros((h, w), dtype=bool)
            mask[h // 2, :] = True
            mask[:, w // 2] = True
        else:
            mask = np.zeros((h, w), dtype=bool)
            on_count = random.randint(1, h * w - 1)
            for idx in random.sample(range(h * w), on_count):
                mask[idx // w, idx % w] = True

        if 0 < int(mask.sum()) < h * w:
            return mask

    mask = np.zeros((h, w), dtype=bool)
    mask[::2, ::2] = True
    if int(mask.sum()) in (0, h * w):
        mask[0, 0] = True
        mask[-1, -1] = False
    return mask


def _build_palette_entries(colors: list[int], ph: int, pw: int) -> dict[int, np.ndarray] | None:
    entries: dict[int, np.ndarray] = {}
    seen: set[bytes] = set()
    for color in colors:
        for _ in range(60):
            mask = _pattern_mask(ph, pw)
            key = mask.astype(np.uint8).tobytes()
            if key in seen:
                continue
            patch = np.full((ph, pw), GEN_BG, dtype=int)
            patch[mask] = color
            entries[color] = patch
            seen.add(key)
            break
        else:
            return None
    return entries


def _make_palette_patch(entries: dict[int, np.ndarray]) -> np.ndarray:
    colors = list(entries.keys())
    ph, pw = next(iter(entries.values())).shape
    n = len(colors)
    h = ph + 3
    w = 2 + n * pw + (n - 1)
    patch = np.full((h, w), GEN_BG, dtype=int)
    patch[0, :] = GEN_SEP
    patch[-1, :] = GEN_SEP
    patch[:, 0] = GEN_SEP
    patch[:, -1] = GEN_SEP

    c = 1
    for i, color in enumerate(colors):
        patch[1, c : c + pw] = color
        patch[2 : 2 + ph, c : c + pw] = entries[color]
        c += pw
        if i + 1 < n:
            patch[:, c] = GEN_SEP
            c += 1
    return patch


def _fits(used: np.ndarray, top: int, left: int, bottom: int, right: int) -> bool:
    return top >= 0 and left >= 0 and bottom < used.shape[0] and right < used.shape[1] and not used[top : bottom + 1, left : right + 1].any()


def _reserve(used: np.ndarray, top: int, left: int, bottom: int, right: int) -> None:
    used[top : bottom + 1, left : right + 1] = True


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, label_color: int | None) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = GEN_SEP
    grid[r1, c0 : c1 + 1] = GEN_SEP
    grid[r0 : r1 + 1, c0] = GEN_SEP
    grid[r0 : r1 + 1, c1] = GEN_SEP
    if label_color is not None:
        grid[r0 - 1, c0 : c1 + 1] = label_color


def _manual_expected(
    grid: np.ndarray,
    containers: list[tuple[int, int, int, int, int]],
    palette_bbox: tuple[int, int, int, int],
    entries: dict[int, np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, h, w, color in containers:
        out[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = _tile_pattern(entries[color], (h - 2, w - 2))
        out[r0 - 1, c0 : c0 + w] = GEN_BG
    pr0, pc0, pr1, pc1 = palette_bbox
    region = out[pr0 + 1 : pr1, pc0 + 1 : pc1]
    region[region != GEN_SEP] = GEN_BG
    return out


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(["right", "bottom"])
    entry_count = random.randint(2, 4)
    ph = random.randint(2, 4)
    pw = random.randint(2, 4)
    palette_colors = random.sample(GEN_COLORS, entry_count)
    entries = _build_palette_entries(palette_colors, ph, pw)
    if entries is None:
        return None
    palette_patch = _make_palette_patch(entries)
    pal_h, pal_w = palette_patch.shape

    container_count = random.randint(2, 5)
    distractor_count = random.randint(0, 2)

    used_colors = random.sample(palette_colors, min(2, len(palette_colors)))
    while len(used_colors) < container_count:
        used_colors.append(random.choice(palette_colors))
    random.shuffle(used_colors)

    container_specs = []
    for color in used_colors:
        container_specs.append(
            {
                "kind": "container",
                "color": color,
                "h": random.randint(4, 9),
                "w": random.randint(4, 9),
            }
        )
    distractor_specs = []
    for _ in range(distractor_count):
        distractor_specs.append(
            {
                "kind": "distractor",
                "color": None,
                "h": random.randint(4, 8),
                "w": random.randint(4, 8),
            }
        )

    min_h = max(18, pal_h + 6)
    min_w = max(18, pal_w + 6)
    if orientation == "right":
        min_w = max(min_w, pal_w + 10)
    else:
        min_h = max(min_h, pal_h + 10)
    if min_h > 30 or min_w > 30:
        return None

    H = random.randint(min_h, 30)
    W = random.randint(min_w, 30)
    grid = np.full((H, W), GEN_BG, dtype=int)
    used = np.zeros((H, W), dtype=bool)

    if orientation == "right":
        pal_r0 = random.randint(0, H - pal_h)
        pal_c0 = W - pal_w
    else:
        pal_r0 = H - pal_h
        pal_c0 = random.randint(0, W - pal_w)
    grid[pal_r0 : pal_r0 + pal_h, pal_c0 : pal_c0 + pal_w] = palette_patch
    _reserve(
        used,
        max(0, pal_r0 - 1),
        max(0, pal_c0 - 1),
        min(H - 1, pal_r0 + pal_h),
        min(W - 1, pal_c0 + pal_w),
    )

    objects = container_specs + distractor_specs
    random.shuffle(objects)
    objects.sort(key=lambda spec: spec["h"] * spec["w"], reverse=True)

    placed_containers: list[tuple[int, int, int, int, int]] = []

    for spec in objects:
        h = spec["h"]
        w = spec["w"]
        label = spec["kind"] == "container"
        for _ in range(PLACE_ATTEMPTS):
            r0 = random.randint(1, H - h - 1)
            c0 = random.randint(1, W - w - 1)
            top = r0 - 2 if label else r0 - 1
            bottom = r0 + h
            left = c0 - 1
            right = c0 + w
            if not _fits(used, top, left, bottom, right):
                continue
            _draw_frame(grid, r0, c0, h, w, spec["color"])
            _reserve(used, top, left, bottom, right)
            if label:
                placed_containers.append((r0, c0, h, w, int(spec["color"])))
            break
        else:
            return None

    if len({color for *_rest, color in placed_containers}) < 2:
        return None

    sep_count = int(np.count_nonzero(grid == GEN_SEP))
    other_counts = [int(np.count_nonzero(grid == color)) for color in palette_colors]
    if sep_count <= max(other_counts):
        return None

    palette_bbox = (pal_r0, pal_c0, pal_r0 + pal_h - 1, pal_c0 + pal_w - 1)
    expected = _manual_expected(grid, placed_containers, palette_bbox, entries)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid labeled-container tiling instance")
