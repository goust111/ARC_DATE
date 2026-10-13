from __future__ import annotations

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _extract_components(grid: np.ndarray, bg: int = 0) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            colors: set[int] = set()
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                colors.add(int(grid[rr, cc]))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            assert len(colors) == 1, "each fragment must be monochromatic"
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            components.append(
                {
                    "color": int(next(iter(colors))),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )

    return components


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "mask must be non-empty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return (
        mask[r0 : r1 + 1, c0 : c1 + 1].copy(),
        (int(r0), int(c0), int(r1), int(c1)),
    )


def _shape_matches_negative(anchor: dict[str, object], other: dict[str, object]) -> bool:
    anchor_mask = np.asarray(anchor["mask"], dtype=bool)
    other_mask = np.asarray(other["mask"], dtype=bool)
    negative = ~anchor_mask
    if not negative.any():
        return False
    negative_crop, _ = _crop_mask(negative)
    return np.array_equal(np.rot90(other_mask, 2), negative_crop)


def _assemble_rectangle(anchor: dict[str, object], other: dict[str, object]) -> dict[str, object]:
    anchor_mask = np.asarray(anchor["mask"], dtype=bool)
    rect = np.full(anchor_mask.shape, int(other["color"]), dtype=int)
    rect[anchor_mask] = int(anchor["color"])
    return {
        "area": int(rect.shape[0] * rect.shape[1]),
        "anchor_pos": tuple(map(int, anchor["bbox"][:2])),
        "grid": rect,
    }


def _build_output_from_pairs(pairs: list[tuple[dict[str, object], dict[str, object]]]) -> np.ndarray:
    rectangles = [_assemble_rectangle(anchor, other) for anchor, other in pairs]
    assert rectangles, "expected at least one pair"
    rectangles.sort(key=lambda item: (int(item["area"]), tuple(item["anchor_pos"])))

    out_h = max(int(item["grid"].shape[0]) for item in rectangles)
    out_w = sum(int(item["grid"].shape[1]) for item in rectangles) + len(rectangles) - 1
    out = np.zeros((out_h, out_w), dtype=int)

    c = 0
    for item in rectangles:
        rect = np.asarray(item["grid"], dtype=int)
        h, w = map(int, rect.shape)
        out[:h, c : c + w] = rect
        c += w + 1

    return out


def _pair_components(components: list[dict[str, object]]) -> list[tuple[dict[str, object], dict[str, object]]]:
    n = len(components)
    assert n > 0 and n % 2 == 0, "expected a positive even number of fragments"

    outgoing: list[list[int]] = []
    incoming = [0] * n
    for i, anchor in enumerate(components):
        candidates = [
            j
            for j, other in enumerate(components)
            if i != j and _shape_matches_negative(anchor, other)
        ]
        assert len(candidates) <= 1, "pairing must be unambiguous"
        outgoing.append(candidates)
        for j in candidates:
            incoming[j] += 1

    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for i in range(n):
        if outgoing[i]:
            assert incoming[i] == 0, "anchor fragments must not also be partners"
            j = outgoing[i][0]
            assert incoming[j] == 1 and not outgoing[j], "partner fragments must be matched once"
            pairs.append((components[i], components[j]))
        else:
            assert incoming[i] == 1, "each non-anchor fragment must be used exactly once"

    assert len(pairs) * 2 == n
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Pair complementary fragments, complete rectangles, and pack them by area."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    components = _extract_components(g, bg=0)
    pairs = _pair_components(components)
    return _build_output_from_pairs(pairs)


# --------------- generator ---------------

_RNG = np.random.default_rng()

BG = 0
COLORS = np.arange(1, 10)

PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX_EXCLUSIVE = 5
GRID_H_MIN = 18
GRID_H_MAX_EXCLUSIVE = 31
GRID_W_MIN = 18
GRID_W_MAX_EXCLUSIVE = 31
RECT_SIDE_MIN = 3
RECT_SIDE_MAX_EXCLUSIVE = 8
MAX_GENERATION_ATTEMPTS = 300
PAIR_SAMPLE_ATTEMPTS = 120
SHAPE_SAMPLE_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 500


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]):
                continue
            if not bool(mask[nr, nc]) or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))

    return len(seen) == int(mask.sum())


def _sample_anchor_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(SHAPE_SAMPLE_ATTEMPTS):
        fill_from_top = bool(_RNG.integers(0, 2))
        full_run = int(_RNG.integers(1, h))
        widths = _RNG.integers(1, w, size=h)

        if fill_from_top:
            widths[:full_run] = w
            free_rows = list(range(full_run, h))
        else:
            widths[h - full_run :] = w
            free_rows = list(range(0, h - full_run))

        if free_rows and len({int(widths[r]) for r in free_rows}) == 1 and len(free_rows) >= 2:
            row = int(_RNG.choice(free_rows))
            choices = [val for val in range(1, w) if val != int(widths[row])]
            widths[row] = int(_RNG.choice(choices))

        if len(set(map(int, widths))) == 1:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, width in enumerate(widths):
            mask[r, : int(width)] = True

        negative = ~mask
        if not negative.any():
            continue
        negative_crop, _ = _crop_mask(negative)
        if not _is_connected(mask) or not _is_connected(negative_crop):
            continue
        return mask

    return None


def _sample_pair(used_areas: set[int]) -> tuple[int, np.ndarray, np.ndarray] | None:
    for _ in range(PAIR_SAMPLE_ATTEMPTS):
        h = int(_RNG.integers(RECT_SIDE_MIN, RECT_SIDE_MAX_EXCLUSIVE))
        w = int(_RNG.integers(RECT_SIDE_MIN, RECT_SIDE_MAX_EXCLUSIVE))
        area = h * w
        if area in used_areas:
            continue

        anchor_mask = _sample_anchor_mask(h, w)
        if anchor_mask is None:
            continue

        negative_crop, _ = _crop_mask(~anchor_mask)
        partner_mask = np.rot90(negative_crop, 2)
        rotation = int(_RNG.integers(0, 4))
        return area, np.rot90(anchor_mask, rotation), np.rot90(partner_mask, rotation)

    return None


def _place_fragment(
    grid: np.ndarray,
    blocked: np.ndarray,
    fragment: dict[str, object],
) -> bool:
    mask = np.asarray(fragment["mask"], dtype=bool)
    mh, mw = map(int, mask.shape)
    h, w = map(int, grid.shape)

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = int(_RNG.integers(0, h - mh + 1))
        c0 = int(_RNG.integers(0, w - mw + 1))
        if blocked[r0 : r0 + mh, c0 : c0 + mw][mask].any():
            continue

        grid[r0 : r0 + mh, c0 : c0 + mw][mask] = int(fragment["color"])
        blocked[
            max(0, r0 - 1) : min(h, r0 + mh + 1),
            max(0, c0 - 1) : min(w, c0 + mw + 1),
        ] = True
        fragment["bbox"] = (int(r0), int(c0), int(r0 + mh - 1), int(c0 + mw - 1))
        return True

    return False


def _expected_from_generated_pairs(pairs: list[dict[str, object]]) -> np.ndarray:
    solved_pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for pair in pairs:
        anchor = pair["anchor"]
        other = pair["partner"]
        assert _shape_matches_negative(anchor, other)
        solved_pairs.append((anchor, other))
    return _build_output_from_pairs(solved_pairs)


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.integers(PAIR_COUNT_MIN, PAIR_COUNT_MAX_EXCLUSIVE))
    colors = _RNG.choice(COLORS, size=pair_count * 2, replace=False)

    used_areas: set[int] = set()
    pairs: list[dict[str, object]] = []
    all_fragments: list[dict[str, object]] = []
    for pair_index in range(pair_count):
        sampled = _sample_pair(used_areas)
        if sampled is None:
            return None

        area, mask_a, mask_b = sampled
        used_areas.add(int(area))
        pair = {
            "area": int(area),
            "anchor": {"color": int(colors[2 * pair_index]), "mask": mask_a},
            "partner": {"color": int(colors[2 * pair_index + 1]), "mask": mask_b},
        }
        pairs.append(pair)
        all_fragments.extend([pair["anchor"], pair["partner"]])

    max_h = max(int(np.asarray(fragment["mask"]).shape[0]) for fragment in all_fragments)
    max_w = max(int(np.asarray(fragment["mask"]).shape[1]) for fragment in all_fragments)
    grid_h = int(_RNG.integers(max(GRID_H_MIN, max_h + 2), GRID_H_MAX_EXCLUSIVE))
    grid_w = int(_RNG.integers(max(GRID_W_MIN, max_w + 2), GRID_W_MAX_EXCLUSIVE))
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    order = list(_RNG.permutation(len(all_fragments)))
    for idx in order:
        if not _place_fragment(grid, blocked, all_fragments[idx]):
            return None

    try:
        expected = _expected_from_generated_pairs(pairs)
    except Exception:
        return None
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
