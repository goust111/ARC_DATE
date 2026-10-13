from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste


BG = 0
KEY_COLOR_POOL = [1, 2, 3, 4, 5, 6, 8, 9]
INNER_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (22, 28)
GEN_RETRIES = 400

CASE_OPTIONS = [
    (2, [2, 1]),
    (2, [3, 1]),
    (2, [2, 2]),
    (2, [4, 1]),
    (2, [3, 2]),
    (3, [2, 1, 1]),
    (3, [3, 1, 1]),
    (3, [2, 2, 1]),
]


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _render_nested(shape: tuple[int, int], colors: list[int]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), BG, dtype=int)
    for depth, color in enumerate(colors):
        r0 = depth
        c0 = depth
        r1 = h - depth
        c1 = w - depth
        if depth == len(colors) - 1:
            out[r0:r1, c0:c1] = int(color)
            break
        out[r0, c0:c1] = int(color)
        out[r1 - 1, c0:c1] = int(color)
        out[r0:r1, c0] = int(color)
        out[r0:r1, c1 - 1] = int(color)
    return out


def _extract_layers(patch: np.ndarray) -> list[int] | None:
    if patch.ndim != 2 or patch.size == 0 or bool(np.any(patch == BG)):
        return None

    colors: list[int] = []
    cur = np.array(patch, dtype=int)
    while True:
        if cur.size == 0:
            return None
        first = int(cur[0, 0])
        if bool(np.all(cur == first)):
            colors.append(first)
            break
        if min(cur.shape) < 3:
            return None
        border = cur[_border_mask(cur.shape)]
        if not bool(np.all(border == first)):
            return None
        colors.append(first)
        cur = cur[1:-1, 1:-1]

    return colors if len(colors) >= 2 else None


def _key_candidates(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    keys = []
    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color == BG:
                continue
            if (r > 0 and int(g[r - 1, c]) != BG) or (r + 1 < H and int(g[r + 1, c]) != BG):
                continue
            if (c > 0 and int(g[r, c - 1]) != BG) or (c + 1 < W and int(g[r, c + 1]) != BG):
                continue
            keys.append((r, c, color))
    return keys


def _find_legend_keys(grid: np.ndarray) -> list[tuple[int, int, int]]:
    keys = _key_candidates(grid)
    by_col: dict[int, list[tuple[int, int, int]]] = {}
    for item in keys:
        by_col.setdefault(int(item[1]), []).append(item)

    candidates = []
    for col, items in by_col.items():
        items = sorted(items)
        if not (2 <= len(items) <= 3):
            continue
        rows = [r for r, _c, _color in items]
        if len(set(color for _r, _c, color in items)) != len(items):
            continue
        if any(rows[i + 1] - rows[i] < 2 for i in range(len(rows) - 1)):
            continue
        candidates.append((len(items), col, items))

    assert len(candidates) == 1, "expected exactly one legend key column"
    return sorted(candidates[0][2])


def _analyze_objects(grid: np.ndarray, keys: list[tuple[int, int, int]]) -> list[dict]:
    work = np.asarray(grid, dtype=int).copy()
    for r, c, _color in keys:
        work[int(r), int(c)] = BG

    objects = []
    for comp in find_components(work, bg=BG, connectivity=4, monochromatic=False):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = bbox_h(comp["bbox"])
        w = bbox_w(comp["bbox"])
        if int(comp["area"]) != h * w:
            raise AssertionError("nested rectangle object must fill its bbox")
        patch = crop_bbox(work, comp["bbox"])
        layers = _extract_layers(patch)
        if layers is None:
            raise AssertionError("foreground component is not a valid nested rectangle")
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (h, w),
                "layers": layers,
                "outer": int(layers[0]),
                "bbox_area": int(h * w),
            }
        )

    assert 3 <= len(objects) <= 5, "expected 3-5 nested rectangles"
    return objects


def _sort_objects_for_processing(objects: list[dict]) -> list[dict]:
    return sorted(
        objects,
        key=lambda obj: (
            int(obj["bbox_area"]),
            int(obj["bbox"][1]),
            int(obj["bbox"][0]),
        ),
    )


def _legend_dot_positions(key_col: int, count: int) -> list[int]:
    return [int(key_col + 2 * step) for step in range(1, count + 1)]


def _count_outer_colors(objects: list[dict], keys: list[tuple[int, int, int]]) -> dict[int, int]:
    counts = {int(color): 0 for _r, _c, color in keys}
    for obj in objects:
        outer = int(obj["outer"])
        assert outer in counts, "outer layer color must be a legend key color"
        counts[outer] += 1
    assert any(value >= 2 for value in counts.values()), "at least one key color must be shared"
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count legend-key outer colors and use those counts to reverse nested rectangle prefixes.

    Input structure:
    - The background is black, with one legend area near the left or right side formed by 2-3 isolated key-color pixels aligned in one column.
    - Away from the legend are 3-5 separated multicolor nested rectangles; each object fills its bbox and consists of 2-4 concentric rectangular layers with a solid innermost core.
    - Every object's outermost color is one of the legend key colors.

    Rule:
    - Count how many nested rectangles use each legend color as their outermost layer.
    - On each legend row, keep the key pixel and add that many same-color dots to its right, spaced every other cell.
    - For each nested rectangle, let T be the count of its outermost color and L be its number of layers; reverse the order of the outermost min(T, L) layers and leave deeper layers unchanged.
    - Process nested rectangles by increasing bbox area; ties are broken by left-to-right, then top-to-bottom bbox origin.

    Key insight:
    - The legend is not only a report: its counted frequencies are fed back as each rectangle's layer-reversal depth, so more frequently shared outer colors trigger deeper reversals.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BG

    keys = _find_legend_keys(g)
    objects = _sort_objects_for_processing(_analyze_objects(g, keys))
    counts = _count_outer_colors(objects, keys)

    out = g.copy()
    for r, c, color in keys:
        for dc in _legend_dot_positions(c, counts[int(color)]):
            assert 0 <= dc < g.shape[1], "legend row must have room for its tally"
            assert int(g[r, dc]) == BG, "legend tally cells must start empty"
            out[int(r), int(dc)] = int(color)

    for obj in objects:
        t = min(counts[int(obj["outer"])], len(obj["layers"]))
        new_layers = list(reversed(obj["layers"][:t])) + list(obj["layers"][t:])
        patch = _render_nested(obj["shape"], new_layers)
        r0, c0, r1, c1 = obj["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _object_candidate_positions(blocked: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = blocked.shape
    h, w = map(int, shape)
    positions = []
    for r0 in range(1, H - h):
        for c0 in range(1, W - w):
            rr0 = max(0, r0 - 1)
            rr1 = min(H, r0 + h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(W, c0 + w + 1)
            if not bool(blocked[rr0:rr1, cc0:cc1].any()):
                positions.append((r0, c0))
    return positions


def _sample_layers(outer: int, layer_count: int, key_colors: list[int]) -> list[int]:
    colors = [int(outer)]
    while len(colors) < layer_count:
        preferred = [c for c in key_colors if c not in colors and c != colors[-1]]
        fallback = [c for c in INNER_COLOR_POOL if c not in colors and c != colors[-1]]
        if preferred and random.random() < 0.45:
            colors.append(int(random.choice(preferred)))
        else:
            colors.append(int(random.choice(fallback)))
    return colors


def _sample_object_specs(
    key_colors: list[int],
    outer_colors: list[int],
    counts: dict[int, int],
) -> list[dict]:
    specs = [{"outer": int(color), "layers_n": None} for color in outer_colors]
    groups: dict[int, list[int]] = {}
    for idx, spec in enumerate(specs):
        groups.setdefault(int(spec["outer"]), []).append(idx)

    assigned = [False] * len(specs)
    for color, idxs in groups.items():
        t = int(counts[color])
        if t >= 2:
            partial_choices = [n for n in range(t + 1, 5)]
            if partial_choices:
                idx = random.choice(idxs)
                specs[idx]["layers_n"] = int(random.choice(partial_choices))
                assigned[idx] = True
            if t >= 3:
                remaining = [idx for idx in idxs if not assigned[idx]]
                if remaining:
                    idx = random.choice(remaining)
                    specs[idx]["layers_n"] = random.randint(2, t - 1)
                    assigned[idx] = True

    for idx, spec in enumerate(specs):
        if spec["layers_n"] is None:
            spec["layers_n"] = random.randint(2, 4)

    all_inner_cross_keys = False
    for spec in specs:
        layer_count = int(spec["layers_n"])
        core_h = random.randint(1, 3)
        core_w = random.randint(1, 3)
        spec["shape"] = (core_h + 2 * (layer_count - 1), core_w + 2 * (layer_count - 1))
        spec["layers"] = _sample_layers(int(spec["outer"]), layer_count, key_colors)
        if any(color in key_colors and color != spec["outer"] for color in spec["layers"][1:]):
            all_inner_cross_keys = True

    if not all_inner_cross_keys:
        candidates = [spec for spec in specs if len(spec["layers"]) >= 2 and any(color != spec["outer"] for color in key_colors)]
        if candidates:
            spec = random.choice(candidates)
            replacement_choices = [color for color in key_colors if color != spec["outer"] and color != spec["layers"][0]]
            if replacement_choices:
                spec["layers"][1] = int(random.choice(replacement_choices))
                if len(spec["layers"]) >= 3 and spec["layers"][2] == spec["layers"][1]:
                    fix_choices = [c for c in INNER_COLOR_POOL if c not in {spec["layers"][1], spec["layers"][0]}]
                    spec["layers"][2] = int(random.choice(fix_choices))

    for spec in specs:
        spec["patch"] = _render_nested(spec["shape"], spec["layers"])
    return specs


def _choose_case() -> tuple[list[int], list[int], dict[int, int]]:
    key_count, pattern = random.choice(CASE_OPTIONS)
    key_colors = random.sample(KEY_COLOR_POOL, key_count)
    shuffled_counts = pattern[:]
    random.shuffle(shuffled_counts)
    counts = {int(color): int(count) for color, count in zip(key_colors, shuffled_counts)}
    outer_colors = []
    for color in key_colors:
        outer_colors.extend([int(color)] * counts[int(color)])
    random.shuffle(outer_colors)
    return key_colors, outer_colors, counts


def _reserve_legend(
    grid: np.ndarray,
    blocked: np.ndarray,
    key_colors: list[int],
    counts: dict[int, int],
) -> list[tuple[int, int, int]] | None:
    H, W = grid.shape
    max_count = max(counts.values())
    legend_h = 2 * len(key_colors) + 1
    legend_w = 2 * max_count + 3 + random.randint(0, 1)
    if legend_h + 2 > H or legend_w + 3 > W:
        return None

    side = random.choice(["left", "right"])
    r0 = random.randint(1, H - legend_h - 1)
    c0 = 1 if side == "left" else W - legend_w - 2

    rr0 = max(0, r0 - 1)
    rr1 = min(H, r0 + legend_h + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(W, c0 + legend_w + 1)
    blocked[rr0:rr1, cc0:cc1] = True

    keys = []
    key_col = c0 + 1
    for i, color in enumerate(key_colors):
        rr = r0 + 1 + 2 * i
        grid[rr, key_col] = int(color)
        keys.append((rr, key_col, int(color)))
    return keys


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = np.full((H, W), BG, dtype=int)
    blocked = np.zeros((H, W), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    key_colors, outer_colors, counts = _choose_case()
    keys = _reserve_legend(grid, blocked, key_colors, counts)
    if keys is None:
        return None

    specs = _sample_object_specs(key_colors, outer_colors, counts)
    specs.sort(key=lambda spec: spec["shape"][0] * spec["shape"][1], reverse=True)

    for spec in specs:
        positions = _object_candidate_positions(blocked, spec["shape"])
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        paste(grid, spec["patch"], r0, c0)
        h, w = spec["shape"]
        spec["bbox"] = (r0, c0, r0 + h - 1, c0 + w - 1)
        rr0 = max(0, r0 - 1)
        rr1 = min(H, r0 + h + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(W, c0 + w + 1)
        blocked[rr0:rr1, cc0:cc1] = True

    out = main(grid)
    if not _semantic_checks(grid, out):
        return None
    return grid


def _expected_legend_counts(keys: list[tuple[int, int, int]], objects: list[dict]) -> dict[int, int]:
    return _count_outer_colors(objects, keys)


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        keys = _find_legend_keys(inp)
        objects = _sort_objects_for_processing(_analyze_objects(inp, keys))
        counts = _expected_legend_counts(keys, objects)
        assert any(value >= 2 for value in counts.values())
        assert len(set(counts.values())) >= 2 or max(counts.values()) >= 3

        expected = inp.copy()
        for r, c, color in keys:
            expected_cols = set(_legend_dot_positions(c, counts[int(color)]))
            for cc in range(c + 1, out.shape[1]):
                value = int(out[r, cc])
                if cc in expected_cols:
                    assert value == int(color)
                    expected[r, cc] = int(color)
                elif int(inp[r, cc]) == BG:
                    if cc <= max(expected_cols, default=c):
                        assert value == BG
            assert int(out[r, c]) == int(color)

        changed = False
        saw_full = False
        saw_partial = False
        for obj in objects:
            r0, c0, r1, c1 = obj["bbox"]
            out_patch = out[r0 : r1 + 1, c0 : c1 + 1]
            t = min(counts[int(obj["outer"])], len(obj["layers"]))
            expected_layers = list(reversed(obj["layers"][:t])) + list(obj["layers"][t:])
            rendered = _render_nested(obj["shape"], expected_layers)
            assert np.array_equal(out_patch, rendered)
            expected[r0 : r1 + 1, c0 : c1 + 1] = rendered
            if t > 1:
                changed = True
            if counts[int(obj["outer"])] > len(obj["layers"]):
                saw_full = True
            if 1 < counts[int(obj["outer"])] < len(obj["layers"]):
                saw_partial = True

        assert changed
        assert saw_full or saw_partial
        assert not np.array_equal(inp, out)
        assert np.array_equal(out, expected)
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
