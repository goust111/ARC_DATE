from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import permutations

import numpy as np


BG = 0
COUNT_TO_COLOR = {1: 1, 2: 2, 3: 3}
INPUT_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
PANEL_SIZE_CHOICES = (4, 5, 6)
SLOT_SIZE_CHOICES = (8, 9)
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_panels(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    panels: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if grid[r0, c0] == BG or seen[r0, c0]:
                continue

            q = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] != BG and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            br0, bc0, br1, bc1 = min(rows), min(cols), max(rows), max(cols)
            patch = grid[br0 : br1 + 1, bc0 : bc1 + 1]
            assert np.all(patch != BG), "each component must be a filled non-black rectangle"
            assert len(coords) == patch.size, "panel component must occupy its entire bbox"
            panels.append({"bbox": (br0, bc0, br1, bc1), "patch": patch.copy()})

    assert panels, "expected at least one panel"
    return panels


def _quadrant_slices(h: int, w: int) -> list[tuple[slice, slice]]:
    top_h = h // 2
    left_w = w // 2
    return [
        (slice(0, top_h), slice(0, left_w)),
        (slice(0, top_h), slice(left_w, w)),
        (slice(top_h, h), slice(0, left_w)),
        (slice(top_h, h), slice(left_w, w)),
    ]


def _unique_majority(region: np.ndarray) -> int:
    colors, counts = np.unique(region, return_counts=True)
    assert colors.size > 0
    max_count = int(counts.max())
    winners = colors[counts == max_count]
    assert winners.size == 1, "each quadrant must have a unique majority color"
    return int(winners[0])


def _panel_signature(panel: np.ndarray) -> tuple[int, int, int, int]:
    h, w = map(int, panel.shape)
    assert h >= 4 and w >= 4
    signature = []
    for rs, cs in _quadrant_slices(h, w):
        region = panel[rs, cs]
        assert region.size > 0
        assert np.all(region != BG)
        signature.append(_unique_majority(region))
    return tuple(signature)


def _panel_global_majority(panel: np.ndarray) -> int:
    return _unique_majority(panel)


def _majority_key(region: np.ndarray) -> tuple[int, ...]:
    colors, counts = np.unique(region, return_counts=True)
    max_count = int(counts.max())
    return tuple(int(color) for color in colors[counts == max_count])


def _count_color(count: int) -> int:
    return COUNT_TO_COLOR.get(int(count), 4)


def _summarize_groups(panels: list[dict], keys: list[object]) -> np.ndarray:
    groups: dict[object, list[dict]] = defaultdict(list)
    for panel, key in zip(panels, keys):
        groups[key].append(panel)

    ordered = sorted(
        groups.values(),
        key=lambda group: min((int(p["bbox"][0]), int(p["bbox"][1])) for p in group),
    )
    out = np.zeros((1, len(ordered)), dtype=int)
    for idx, group in enumerate(ordered):
        out[0, idx] = _count_color(len(group))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Classify filled panels by quadrant-majority color signatures and summarize each class count.

    Input structure:
    - A black background contains 4 to 6 separated solid rectangular panels.
    - Every panel is fully non-black inside its bbox and mixes 2 or 3 visible colors.
    - Each panel's bbox is split into TL/TR/BL/BR quadrants using the panel's own row/column axes.

    Rule:
    - Detect each non-black connected component and verify it is a filled rectangle.
    - For every panel quadrant, take the unique majority color; the ordered 4-tuple
      (TL, TR, BL, BR) is that panel's signature.
    - Group panels by identical signatures, sort groups by the topmost-leftmost panel
      inside each group, and output a 1xN row whose cells encode group sizes:
      1->blue, 2->red, 3->green, 4 or more->yellow.

    Key insight:
    - The noisy panel interiors are only a carrier; the real label is the ordered
      quadrant-majority signature, so both the panel-local reference frame and the
      per-quadrant vote are necessary before counting classes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    panels = _find_panels(g)
    signatures = [_panel_signature(panel["patch"]) for panel in panels]
    return _summarize_groups(panels, signatures)


def _valid_count_patterns(total: int, n_groups: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, slots: int, acc: list[int]) -> None:
        if slots == 1:
            acc.append(remaining)
            if max(acc) >= 2:
                out.append(tuple(acc))
            acc.pop()
            return
        for value in range(1, remaining - slots + 2):
            acc.append(value)
            rec(remaining - value, slots - 1, acc)
            acc.pop()

    rec(total, n_groups, [])
    return out


_COUNT_PATTERNS = {
    (total, groups): _valid_count_patterns(total, groups)
    for total in range(4, 7)
    for groups in (2, 3)
    if groups <= total
}


def _signature_candidates(palette: tuple[int, ...]) -> list[tuple[int, int, int, int]]:
    if len(palette) == 2:
        base = (palette[0], palette[0], palette[1], palette[1])
    else:
        base = (palette[0], palette[0], palette[1], palette[2])
    return sorted(set(permutations(base)))


def _build_panel(signature: tuple[int, int, int, int], panel_h: int, panel_w: int) -> np.ndarray:
    panel = np.zeros((panel_h, panel_w), dtype=int)
    palette = sorted(set(signature))

    for idx, (rs, cs) in enumerate(_quadrant_slices(panel_h, panel_w)):
        maj = int(signature[idx])
        region_h = rs.stop - rs.start
        region_w = cs.stop - cs.start
        region_area = region_h * region_w
        maj_count = _RNG.integers(region_area // 2 + 1, region_area)
        others = [color for color in palette if color != maj]
        assert others
        flat = np.array(
            [maj] * int(maj_count)
            + list(_RNG.choice(others, size=region_area - int(maj_count), replace=True)),
            dtype=int,
        )
        _RNG.shuffle(flat)
        panel[rs, cs] = flat.reshape(region_h, region_w)

    assert _panel_signature(panel) == signature
    return panel


def _choose_layout(total_panels: int) -> tuple[int, int, int, int]:
    _ = total_panels
    slot_rows, slot_cols = ((2, 3) if int(_RNG.integers(2)) == 0 else (3, 2))
    slot_h = int(_RNG.choice(SLOT_SIZE_CHOICES))
    slot_w = int(_RNG.choice(SLOT_SIZE_CHOICES))
    return slot_rows, slot_cols, slot_h, slot_w


def _place_panels(specs: list[dict]) -> np.ndarray:
    total_panels = len(specs)
    slot_rows, slot_cols, slot_h, slot_w = _choose_layout(total_panels)
    grid_h = slot_rows * slot_h + 2
    grid_w = slot_cols * slot_w + 2
    grid = np.zeros((grid_h, grid_w), dtype=int)

    slots = [(sr, sc) for sr in range(slot_rows) for sc in range(slot_cols)]
    chosen_slots = _RNG.choice(len(slots), size=total_panels, replace=False)

    for spec, slot_idx in zip(specs, chosen_slots):
        sr, sc = slots[int(slot_idx)]
        slot_r0 = 1 + sr * slot_h
        slot_c0 = 1 + sc * slot_w

        panel = spec["panel"]
        panel_h, panel_w = panel.shape
        inner_h = slot_h - 2
        inner_w = slot_w - 2
        assert panel_h <= inner_h and panel_w <= inner_w
        dr = int(_RNG.integers(inner_h - panel_h + 1))
        dc = int(_RNG.integers(inner_w - panel_w + 1))
        r0 = slot_r0 + 1 + dr
        c0 = slot_c0 + 1 + dc
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel
        spec["bbox"] = (r0, c0, r0 + panel_h - 1, c0 + panel_w - 1)

    return grid


def _intended_output(specs: list[dict]) -> np.ndarray:
    panels = [{"bbox": spec["bbox"], "patch": spec["panel"]} for spec in specs]
    keys = [spec["signature"] for spec in specs]
    return _summarize_groups(panels, keys)


def _wrong_outputs_differ(grid: np.ndarray) -> bool:
    panels = _find_panels(grid)
    signatures = [_panel_signature(panel["patch"]) for panel in panels]
    true_out = _summarize_groups(panels, signatures)

    sorted_signature_keys = [tuple(sorted(sig)) for sig in signatures]
    if np.array_equal(_summarize_groups(panels, sorted_signature_keys), true_out):
        return False

    alternative_key_sets = [
        [_majority_key(panel["patch"]) for panel in panels],
        [sig[:2] for sig in signatures],
        [sig[2:] for sig in signatures],
        [(sig[0], sig[2]) for sig in signatures],
        [(sig[1], sig[3]) for sig in signatures],
    ]
    differing = sum(
        not np.array_equal(_summarize_groups(panels, keys), true_out)
        for keys in alternative_key_sets
    )
    return differing >= 2


def _gen_attempt() -> np.ndarray | None:
    total_panels = int(_RNG.integers(4, 7))
    n_groups = 3 if float(_RNG.random()) < 0.65 else 2
    if n_groups > total_panels:
        return None

    count_patterns = _COUNT_PATTERNS[(total_panels, n_groups)]
    group_counts = list(count_patterns[int(_RNG.integers(len(count_patterns)))])
    _RNG.shuffle(group_counts)

    palette_size = 3 if float(_RNG.random()) < 0.7 else 2
    palette = tuple(int(c) for c in _RNG.choice(INPUT_COLORS, size=palette_size, replace=False))
    candidates = _signature_candidates(palette)
    if len(candidates) < n_groups:
        return None

    picked = _RNG.choice(len(candidates), size=n_groups, replace=False)
    signatures = [candidates[int(idx)] for idx in picked]

    specs: list[dict] = []
    for signature, count in zip(signatures, group_counts):
        for _ in range(count):
            panel_h = int(_RNG.choice(PANEL_SIZE_CHOICES))
            panel_w = int(_RNG.choice(PANEL_SIZE_CHOICES))
            panel = _build_panel(signature, panel_h, panel_w)
            specs.append(
                {
                    "signature": signature,
                    "panel": panel,
                }
            )

    if all(spec["panel"].shape[0] % 2 == 0 and spec["panel"].shape[1] % 2 == 0 for spec in specs):
        first = specs[0]
        forced_h = 5
        forced_w = first["panel"].shape[1]
        if forced_w not in PANEL_SIZE_CHOICES:
            forced_w = 5
        first["panel"] = _build_panel(first["signature"], forced_h, forced_w)

    _RNG.shuffle(specs)
    grid = _place_panels(specs)

    try:
        expected = _intended_output(specs)
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if actual.shape[1] not in (2, 3):
        return None
    if not _wrong_outputs_differ(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
