from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(grid).reshape(-1))
    return int(counts.most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neigh = _neighbors4 if int(connectivity) == 4 else _neighbors8
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in neigh(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _coords_4_connected(coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    pts = set(coords)
    seen = {coords[0]}
    q = deque([coords[0]])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(pts)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _stamp_offsets(
    sprite_coords: list[tuple[int, int]],
    anchor: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    sr0, sc0, sr1, sc1 = _bbox(sprite_coords)
    half_w = int(sc1 - sc0 + 1)
    anchor_local_c = int(anchor[1] - sc0)
    assert 0 <= anchor_local_c < half_w, "anchor must lie inside the visible half-width"

    left_offsets: list[tuple[int, int]] = []
    full_offsets: set[tuple[int, int]] = set()
    for r, c in sprite_coords:
        dr = int(r - anchor[0])
        dc = int(c - anchor[1])
        left_offsets.append((dr, dc))
        local_c = int(c - sc0)
        mirror_dc = int((2 * half_w - 1 - local_c) - anchor_local_c)
        full_offsets.add((dr, dc))
        full_offsets.add((dr, mirror_dc))

    return sorted(left_offsets), sorted(full_offsets), int(half_w)


def _extract_sources_and_targets(
    grid: np.ndarray,
) -> tuple[
    dict[int, dict[str, object]],
    dict[int, list[tuple[int, int]]],
    set[tuple[int, int]],
]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _components_from_mask(g != bg, connectivity=8)

    templates: dict[int, dict[str, object]] = {}
    singletons_by_color: dict[int, list[tuple[int, int]]] = {}
    source_cells: set[tuple[int, int]] = set()

    for comp in comps:
        vals = [int(g[r, c]) for r, c in comp]
        colors = sorted(set(vals))
        if len(comp) == 1:
            (r, c) = comp[0]
            singletons_by_color.setdefault(int(g[r, c]), []).append((int(r), int(c)))
            continue

        if len(colors) != 2:
            continue

        counts = Counter(vals)
        marker_colors = [color for color, count in counts.items() if int(count) == 1]
        if len(marker_colors) != 1:
            continue

        marker_color = int(marker_colors[0])
        sprite_color = int(next(color for color in colors if int(color) != marker_color))
        anchor = next((int(r), int(c)) for r, c in comp if int(g[r, c]) == marker_color)
        sprite_coords = [(int(r), int(c)) for r, c in comp if int(g[r, c]) == sprite_color]

        assert len(sprite_coords) >= 3, "source sprite must have 3-6 pixels"
        assert _coords_4_connected(sprite_coords), "source sprite must be 4-connected"
        assert any(max(abs(anchor[0] - r), abs(anchor[1] - c)) == 1 for r, c in sprite_coords), (
            "anchor must be 8-adjacent to the sprite"
        )

        left_offsets, full_offsets, half_w = _stamp_offsets(sprite_coords, anchor)
        assert half_w in (2, 3), "visible half-width must imply full width 4 or 6"
        assert marker_color not in templates, "each anchor color must define a single template"
        templates[marker_color] = {
            "anchor": anchor,
            "marker_color": marker_color,
            "sprite_color": sprite_color,
            "left_offsets": left_offsets,
            "full_offsets": full_offsets,
            "half_w": half_w,
        }
        source_cells.update(comp)

    assert templates, "expected at least one two-color source component"
    return templates, singletons_by_color, source_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each half-sprite to same-colored markers, then mirror-complete it horizontally at each destination.

    Input structure:
    - The grid has a dominant black background with 1-2 multicolor 8-connected source components.
    - Each source contains a 4-connected monochrome half-sprite plus one adjacent single-pixel anchor of another color.
    - Additional isolated single pixels are destination markers when their color matches a source anchor, or distractors otherwise.

    Rule:
    - For each two-color source component, treat the singleton color as the anchor and the larger color set as the half-sprite.
    - Read the visible half-sprite inside its tight bbox, double that visible width to recover the full symmetric bbox, and record sprite offsets relative to the anchor.
    - Erase every source component, then at each isolated marker of the same anchor color stamp the half-sprite and also fill the horizontal mirror of each stamped sprite pixel across that local bbox midline; keep the destination marker itself unchanged.

    Key insight:
    - The source figure is intentionally only the left half of a wider symmetric template, so copying alone is insufficient and the missing right half must be regenerated independently at every destination.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    templates, singletons_by_color, source_cells = _extract_sources_and_targets(g)

    out = g.copy()
    for r, c in source_cells:
        out[r, c] = bg

    for marker_color, template in templates.items():
        sprite_color = int(template["sprite_color"])
        full_offsets = list(template["full_offsets"])
        for tr, tc in singletons_by_color.get(int(marker_color), []):
            for dr, dc in full_offsets:
                rr = int(tr + dr)
                cc = int(tc + dc)
                if _in_bounds(out.shape, rr, cc):
                    out[rr, cc] = sprite_color
            out[int(tr), int(tc)] = int(marker_color)

    return out


BG = 0
SPRITE_COLORS = (1, 3, 4)
ANCHOR_COLORS = (2, 5, 6, 7, 8)
DISTRACTOR_COLOR = 9
GRID_SIDE_RANGE = (14, 19)
SOURCE_COUNT_CHOICES = (2, 2, 1)
DEST_COUNT_CHOICES = (2, 2, 3)
FULL_WIDTH_CHOICES = (4, 6)
FULL_HEIGHT_CHOICES = (2, 3, 4)
SPRITE_AREA_RANGE = (3, 6)
MAX_ATTEMPTS = 600


def _shape_to_offsets(mask: np.ndarray, anchor_local: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = anchor_local
    offsets = []
    for r, c in np.argwhere(mask):
        offsets.append((int(r - ar), int(c - ac)))
    return sorted(offsets)


def _grow_connected_half(height: int, half_w: int, area: int) -> np.ndarray | None:
    mask = np.zeros((height, half_w), dtype=bool)
    start = (random.randrange(height), random.randrange(half_w))
    mask[start] = True

    while int(mask.sum()) < int(area):
        frontier = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < height and 0 <= nc < half_w and not mask[nr, nc]:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            return None
        pick = random.choice(sorted(frontier))
        mask[pick] = True

    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (height, half_w):
        return None
    if np.array_equal(mask, np.fliplr(mask)):
        return None
    return mask


def _make_source_spec() -> dict[str, object] | None:
    full_w = random.choice(FULL_WIDTH_CHOICES)
    half_w = full_w // 2
    height = random.choice(FULL_HEIGHT_CHOICES)
    area = random.randint(*SPRITE_AREA_RANGE)
    if area > height * half_w:
        area = height * half_w
    mask = _grow_connected_half(height, half_w, area)
    if mask is None:
        return None

    anchor_candidates = []
    for ar in range(height):
        for ac in range(half_w):
            if mask[ar, ac]:
                continue
            if any(mask[nr, nc] for nr, nc in _neighbors8(ar, ac) if 0 <= nr < height and 0 <= nc < half_w):
                anchor_candidates.append((ar, ac))
    if not anchor_candidates:
        return None

    anchor_local = random.choice(anchor_candidates)
    left_offsets = _shape_to_offsets(mask, anchor_local)
    full_offsets: set[tuple[int, int]] = set()
    for dr, dc in left_offsets:
        local_c = int(dc + anchor_local[1])
        mirror_dc = int((full_w - 1 - local_c) - anchor_local[1])
        full_offsets.add((int(dr), int(dc)))
        full_offsets.add((int(dr), int(mirror_dc)))

    right_only = {(dr, dc) for dr, dc in full_offsets if dc >= half_w - anchor_local[1]}
    if not right_only:
        return None

    return {
        "height": height,
        "half_w": half_w,
        "full_w": full_w,
        "anchor_local": anchor_local,
        "mask": mask,
        "left_offsets": left_offsets,
        "full_offsets": sorted(full_offsets),
    }


def _absolute_cells(anchor: tuple[int, int], offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ar, ac = anchor
    return [(int(ar + dr), int(ac + dc)) for dr, dc in offsets]


def _cells_with_marker(anchor: tuple[int, int], offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [anchor, *_absolute_cells(anchor, offsets)]


def _can_place_cells(
    shape: tuple[int, int],
    occupied: np.ndarray,
    cells: list[tuple[int, int]],
    padding: int = 1,
) -> bool:
    seen = set()
    for r, c in cells:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in seen:
            return False
        seen.add((r, c))
        r0 = max(0, r - padding)
        r1 = min(shape[0], r + padding + 1)
        c0 = max(0, c - padding)
        c1 = min(shape[1], c + padding + 1)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _mark_cells(occupied: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[int(r), int(c)] = True


def _candidate_destination_anchors(
    shape: tuple[int, int],
    occupied: np.ndarray,
    full_offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            cells = _cells_with_marker((r, c), full_offsets)
            if _can_place_cells(shape, occupied, cells, padding=1):
                candidates.append((int(r), int(c)))
    random.shuffle(candidates)
    return candidates


def _candidate_source_anchors(
    shape: tuple[int, int],
    occupied: np.ndarray,
    left_offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            cells = _cells_with_marker((r, c), left_offsets)
            if _can_place_cells(shape, occupied, cells, padding=1):
                candidates.append((int(r), int(c)))
    random.shuffle(candidates)
    return candidates


def _render_output(shape: tuple[int, int], sources: list[dict[str, object]]) -> np.ndarray | None:
    out = np.zeros(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)

    for source in sources:
        dest_count = int(source["dest_count"])
        full_offsets = list(source["full_offsets"])
        marker_color = int(source["marker_color"])
        sprite_color = int(source["sprite_color"])
        dests: list[tuple[int, int]] = []

        for _ in range(dest_count):
            candidates = _candidate_destination_anchors(shape, occupied, full_offsets)
            if not candidates:
                return None

            edge_bias = [p for p in candidates if min(p[0], p[1], shape[0] - 1 - p[0], shape[1] - 1 - p[1]) <= 2]
            anchor = random.choice(edge_bias if edge_bias and random.random() < 0.45 else candidates)
            dests.append(anchor)
            cells = _cells_with_marker(anchor, full_offsets)
            _mark_cells(occupied, cells)
            out[anchor] = marker_color
            for rr, cc in _absolute_cells(anchor, full_offsets):
                out[rr, cc] = sprite_color
            out[anchor] = marker_color

        source["destinations"] = dests

    return out


def _render_input(shape: tuple[int, int], sources: list[dict[str, object]], keep_distractor: bool) -> np.ndarray | None:
    inp = np.zeros(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)

    for source in sources:
        marker_color = int(source["marker_color"])
        for anchor in source["destinations"]:
            inp[anchor] = marker_color
            _mark_cells(occupied, [anchor])

    for source in sources:
        left_offsets = list(source["left_offsets"])
        candidates = _candidate_source_anchors(shape, occupied, left_offsets)
        if not candidates:
            return None
        anchor = random.choice(candidates)
        source["source_anchor_abs"] = anchor
        inp[anchor] = int(source["marker_color"])
        for rr, cc in _absolute_cells(anchor, left_offsets):
            inp[rr, cc] = int(source["sprite_color"])
        _mark_cells(occupied, _cells_with_marker(anchor, left_offsets))

    if keep_distractor:
        candidates = [
            (r, c)
            for r in range(shape[0])
            for c in range(shape[1])
            if _can_place_cells(shape, occupied, [(r, c)], padding=1)
        ]
        if not candidates:
            return None
        distractor = random.choice(candidates)
        inp[distractor] = DISTRACTOR_COLOR
        _mark_cells(occupied, [distractor])

    return inp


def _build_sources() -> list[dict[str, object]] | None:
    n_sources = random.choice(SOURCE_COUNT_CHOICES)
    sprite_colors = random.sample(SPRITE_COLORS, n_sources)
    marker_colors = random.sample(ANCHOR_COLORS, n_sources)
    sources: list[dict[str, object]] = []

    for sprite_color, marker_color in zip(sprite_colors, marker_colors):
        spec = _make_source_spec()
        if spec is None:
            return None
        spec["sprite_color"] = int(sprite_color)
        spec["marker_color"] = int(marker_color)
        spec["dest_count"] = int(random.choice(DEST_COUNT_CHOICES))
        sources.append(spec)

    return sources


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, sources: list[dict[str, object]], keep_distractor: bool) -> bool:
    try:
        templates, singletons_by_color, _ = _extract_sources_and_targets(inp)
    except Exception:
        return False

    if set(templates) != {int(source["marker_color"]) for source in sources}:
        return False

    for source in sources:
        marker_color = int(source["marker_color"])
        sprite_color = int(source["sprite_color"])
        full_offsets = set(map(tuple, source["full_offsets"]))
        left_offsets = set(map(tuple, source["left_offsets"]))
        full_cells = full_offsets - left_offsets
        if not full_cells:
            return False

        tpl = templates.get(marker_color)
        if tpl is None:
            return False
        if int(tpl["sprite_color"]) != sprite_color:
            return False
        if set(map(tuple, tpl["left_offsets"])) != left_offsets:
            return False
        if set(map(tuple, tpl["full_offsets"])) != full_offsets:
            return False

        dests = sorted(singletons_by_color.get(marker_color, []))
        if dests != sorted(source["destinations"]):
            return False

        for anchor in dests:
            for rr, cc in _absolute_cells(anchor, list(source["full_offsets"])):
                if not _in_bounds(expected.shape, rr, cc):
                    return False
                if int(expected[rr, cc]) != sprite_color:
                    return False
            if int(expected[anchor]) != marker_color:
                return False

    if keep_distractor and DISTRACTOR_COLOR not in set(map(int, inp.reshape(-1))):
        return False
    if keep_distractor and DISTRACTOR_COLOR not in set(map(int, expected.reshape(-1))):
        return False

    return np.array_equal(main(inp), expected)


def _attempt_generate() -> np.ndarray | None:
    sources = _build_sources()
    if sources is None:
        return None
    if len(sources) == 2:
        shape = (random.randint(16, 18), random.randint(16, 18))
    else:
        shape = (random.randrange(*GRID_SIDE_RANGE), random.randrange(*GRID_SIDE_RANGE))

    expected = _render_output(shape, sources)
    if expected is None:
        return None

    keep_distractor = random.random() < 0.7
    inp = _render_input(shape, sources, keep_distractor)
    if inp is None:
        return None

    if keep_distractor:
        distractors = np.argwhere(inp == DISTRACTOR_COLOR)
        if distractors.shape != (1, 2):
            return None
        dr, dc = map(int, distractors[0])
        expected[dr, dc] = DISTRACTOR_COLOR

    if not _validate_semantics(inp, expected, sources, keep_distractor):
        return None

    if len(set(map(int, expected.reshape(-1)))) < 3:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
