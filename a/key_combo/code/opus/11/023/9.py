import random
from collections import deque

import numpy as np


BG = 0
SOURCE_BASE = 1
MARKER = 2
TARGET_BASE = 3


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    groups: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            groups.append(np.array(cells, dtype=int))
    return groups


def _extract_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sources: list[dict] = []
    targets: list[dict] = []
    for coords in _components(g != BG):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != BG), "every object must be a dense rectangle"
        colors = set(map(int, np.unique(patch)))
        info = {
            "bbox": (r0, c0, r1, c1),
            "h": int(r1 - r0 + 1),
            "w": int(c1 - c0 + 1),
            "patch": patch.copy(),
        }
        if colors == {TARGET_BASE}:
            targets.append(info)
        else:
            assert colors <= {SOURCE_BASE, MARKER}, "unexpected source colors"
            assert SOURCE_BASE in colors and MARKER in colors, "source must contain blue base and red markers"
            info["marker_mask"] = patch == MARKER
            sources.append(info)
    assert 3 <= len(sources) <= 4, "expected 3-4 source rectangles"
    assert 2 <= len(targets) <= 3, "expected 2-3 target rectangles"
    return sources, targets


def _paint_by_width(grid: np.ndarray, use_all_sources: bool = True) -> np.ndarray:
    sources, targets = _extract_objects(grid)
    out = np.asarray(grid, dtype=int).copy()
    for target in targets:
        r0, c0, _r1, _c1 = target["bbox"]
        matches = [src for src in sources if int(src["w"]) == int(target["w"])]
        assert matches, "each target must match at least one source by width"
        matches.sort(key=lambda src: int(src["h"]))
        if not use_all_sources:
            matches = [matches[-1]]
        tallest = max(int(src["h"]) for src in matches)
        assert int(target["h"]) >= tallest, "target must be tall enough for every matched source"
        for src in matches:
            mask = src["marker_mask"]
            hh, ww = mask.shape
            out[r0 : r0 + hh, c0 : c0 + ww][mask] = MARKER
    return out


def _paint_by_row_overlap(grid: np.ndarray) -> np.ndarray:
    sources, targets = _extract_objects(grid)
    out = np.asarray(grid, dtype=int).copy()
    for target in targets:
        tr0, tc0, tr1, _tc1 = target["bbox"]
        matches = [
            src
            for src in sources
            if max(int(src["bbox"][0]), int(tr0)) <= min(int(src["bbox"][2]), int(tr1))
        ]
        assert matches, "naive overlap rule found no source"
        for src in sorted(matches, key=lambda item: int(item["h"])):
            mask = src["marker_mask"]
            hh, ww = mask.shape
            out[tr0 : tr0 + hh, tc0 : tc0 + ww][mask] = MARKER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Transfer each marked blue rectangle's pattern onto same-width green rectangles.

    Input structure:
    - The background is black.
    - Left-side objects are dense blue rectangles containing red marker pixels.
    - Right-side objects are dense plain green rectangles that act as transfer targets.

    Rule:
    - For each green rectangle, collect all blue/red source rectangles with the same bounding-box width.
    - Sort those matched sources by height from small to large.
    - Transfer each source marker pattern into the green rectangle by preserving every red pixel's offsets from the source top and left edges.
    - Later transfers overwrite earlier ones; cells not covered by transferred red markers stay green.
    - Source rectangles remain unchanged.

    Key insight:
    - The pairing is determined by equal width rather than spatial proximity, so visually separated rectangles interact when their widths match.
    """
    return _paint_by_width(grid, use_all_sources=True)


# --------------- generator ---------------

GRID_ROWS = (18, 24)
GRID_COLS = (20, 28)
SOURCE_COUNT_CHOICES = (3, 4)
WIDTH_CHOICES = (3, 4, 5, 6)
HEIGHT_CHOICES = (3, 4, 5, 6, 7)
TARGET_EXTRA_HEIGHT = (0, 2)
GEN_ATTEMPTS = 500


def _random_marker_set(h: int, w: int, forbidden: set[tuple[int, int]], mandatory: tuple[int, int] | None) -> set[tuple[int, int]]:
    area = h * w
    limit = min(area - 1, max(3, min(6, area // 2)))
    count = random.randint(2, limit)
    cells = set()
    if mandatory is not None:
        cells.add(mandatory)
    pool = [(r, c) for r in range(h) for c in range(w) if (r, c) not in forbidden]
    random.shuffle(pool)
    for r, c in pool:
        if len(cells) >= count:
            break
        if random.random() < 0.3 and c + 1 < w and (r, c + 1) not in forbidden and len(cells) + 1 < count:
            cells.add((r, c))
            cells.add((r, c + 1))
        elif random.random() < 0.2 and r + 1 < h and (r + 1, c) not in forbidden and len(cells) + 1 < count:
            cells.add((r, c))
            cells.add((r + 1, c))
        else:
            cells.add((r, c))
    assert 1 <= len(cells) < area
    return cells


def _build_group_masks(width: int, heights: list[int]) -> dict[int, set[tuple[int, int]]]:
    ordered = list(heights)
    mandatory: dict[int, tuple[int, int]] = {}
    blocked: set[tuple[int, int]] = set()
    for h in ordered:
        choices = [(r, c) for r in range(h) for c in range(width) if (r, c) not in blocked]
        assert choices, "not enough distinct mandatory cells"
        mandatory[h] = random.choice(choices)
        blocked.add(mandatory[h])

    masks: dict[int, set[tuple[int, int]]] = {}
    all_mandatory = set(mandatory.values())
    for h in ordered:
        forbidden = all_mandatory - {mandatory[h]}
        masks[h] = _random_marker_set(h, width, forbidden, mandatory[h])

    if len(ordered) > 1:
        union = set().union(*(masks[h] for h in ordered))
        tallest = masks[max(ordered)]
        if union == tallest:
            return _build_group_masks(width, heights)
        for h in ordered:
            others = set().union(*(masks[oh] for oh in ordered if oh != h))
            if not (masks[h] - others):
                return _build_group_masks(width, heights)
    return masks


def _render_patch(h: int, w: int, color: int, marker_cells: set[tuple[int, int]] | None = None) -> np.ndarray:
    patch = np.full((h, w), color, dtype=int)
    if marker_cells:
        for r, c in marker_cells:
            patch[int(r), int(c)] = MARKER
    return patch


def _group_partitions(source_count: int, target_count: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(remain: int, slots: int, prefix: list[int]) -> None:
        if slots == 1:
            counts = tuple(prefix + [remain])
            if any(v > 1 for v in counts) and any(v == 1 for v in counts):
                out.append(counts)
            return
        for value in range(1, remain - slots + 2):
            rec(remain - value, slots - 1, prefix + [value])

    rec(source_count, target_count, [])
    return out


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_specs(
    shape: tuple[int, int],
    specs: list[dict],
    col_lo: int,
    col_hi: int,
    occupied: np.ndarray,
) -> list[dict] | None:
    placed: list[dict] = []
    for spec in sorted(specs, key=lambda item: item["h"] * item["w"], reverse=True):
        h = int(spec["h"])
        w = int(spec["w"])
        candidates = []
        for r0 in range(1, shape[0] - h):
            for c0 in range(col_lo, col_hi - w + 2):
                if _can_place(occupied, r0, c0, h, w):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        occupied[r0 : r0 + h, c0 : c0 + w] = True
        placed.append({**spec, "bbox": (r0, c0, r0 + h - 1, c0 + w - 1)})
    return placed


def _compose_input(shape: tuple[int, int], sources: list[dict], targets: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in sources + targets:
        r0, c0, r1, c1 = obj["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = obj["patch"]
    return grid


def _solve_from_sources_and_targets(sources: list[dict], targets: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for src in sources:
        r0, c0, r1, c1 = src["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = src["patch"]
    for tgt in targets:
        r0, c0, r1, c1 = tgt["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = TARGET_BASE
        matches = [src for src in sources if int(src["w"]) == int(tgt["w"])]
        for src in sorted(matches, key=lambda item: int(item["h"])):
            mask = src["patch"] == MARKER
            hh, ww = mask.shape
            out[r0 : r0 + hh, c0 : c0 + ww][mask] = MARKER
    return out


def _nearest_source_output(grid: np.ndarray) -> np.ndarray:
    sources, targets = _extract_objects(grid)
    out = np.asarray(grid, dtype=int).copy()
    for tgt in targets:
        tr0, tc0, tr1, tc1 = tgt["bbox"]
        tcenter = ((tr0 + tr1) / 2.0, (tc0 + tc1) / 2.0)
        candidates = [src for src in sources if int(src["w"]) == int(tgt["w"])]
        assert candidates, "expected at least one same-width source"
        best = min(
            candidates,
            key=lambda src: (
                (tcenter[0] - (src["bbox"][0] + src["bbox"][2]) / 2.0) ** 2
                + (tcenter[1] - (src["bbox"][1] + src["bbox"][3]) / 2.0) ** 2,
                int(src["h"]),
            ),
        )
        mask = best["marker_mask"]
        hh, ww = mask.shape
        out[tr0 : tr0 + hh, tc0 : tc0 + ww][mask] = MARKER
    return out


def _validate_generation(inp: np.ndarray, meta_sources: list[dict], meta_targets: list[dict]) -> bool:
    expected = _solve_from_sources_and_targets(meta_sources, meta_targets, inp.shape)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    sources, targets = _extract_objects(inp)
    if any(np.any(actual[s["bbox"][0] : s["bbox"][2] + 1, s["bbox"][1] : s["bbox"][3] + 1] != s["patch"]) for s in sources):
        return False

    group_sizes = sorted(sum(int(src["w"]) == int(tgt["w"]) for src in sources) for tgt in targets)
    if group_sizes[0] != 1 or group_sizes[-1] < 2:
        return False

    if np.array_equal(actual, _paint_by_width(inp, use_all_sources=False)):
        return False

    try:
        overlap_out = _paint_by_row_overlap(inp)
    except Exception:
        overlap_out = None
    if overlap_out is not None and np.array_equal(actual, overlap_out):
        return False

    if np.array_equal(actual, _nearest_source_output(inp)):
        return False

    if np.array_equal(actual, inp):
        return False

    for tgt in targets:
        r0, c0, r1, c1 = tgt["bbox"]
        if not np.any(actual[r0 : r1 + 1, c0 : c1 + 1] == MARKER):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)
    source_count = random.choice(SOURCE_COUNT_CHOICES)
    target_count = 2 if source_count == 3 else random.choice((2, 3))
    partitions = _group_partitions(source_count, target_count)
    if not partitions:
        return None

    widths = random.sample(WIDTH_CHOICES, target_count)
    counts = list(random.choice(partitions))
    random.shuffle(counts)
    heights = random.sample(HEIGHT_CHOICES, source_count)
    random.shuffle(heights)

    grouped_heights: dict[int, list[int]] = {}
    height_idx = 0
    for width, count in zip(widths, counts):
        group = heights[height_idx : height_idx + count]
        height_idx += count
        grouped_heights[width] = group

    source_specs: list[dict] = []
    target_specs: list[dict] = []
    for width in widths:
        group = grouped_heights[width]
        masks = _build_group_masks(width, group)
        for h in group:
            source_specs.append(
                {
                    "kind": "source",
                    "h": h,
                    "w": width,
                    "patch": _render_patch(h, width, SOURCE_BASE, masks[h]),
                }
            )
        target_h = max(group) + random.randint(*TARGET_EXTRA_HEIGHT)
        target_specs.append(
            {
                "kind": "target",
                "h": target_h,
                "w": width,
                "patch": _render_patch(target_h, width, TARGET_BASE),
            }
        )

    split = random.randint(7, cols - 10)
    gap = random.choice((2, 3))
    left_lo, left_hi = 1, split
    right_lo, right_hi = split + gap + 1, cols - 2
    if right_hi - right_lo + 1 < max(t["w"] for t in target_specs):
        return None

    occupied = np.zeros((rows, cols), dtype=bool)
    placed_sources = _place_specs((rows, cols), source_specs, left_lo, left_hi, occupied)
    if placed_sources is None:
        return None
    placed_targets = _place_specs((rows, cols), target_specs, right_lo, right_hi, occupied)
    if placed_targets is None:
        return None

    inp = _compose_input((rows, cols), placed_sources, placed_targets)
    if not _validate_generation(inp, placed_sources, placed_targets):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to produce a valid puzzle instance")
