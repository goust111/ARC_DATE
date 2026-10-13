from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


GRAY = 5
GENERATION_ATTEMPTS = 800


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "empty mask has no bbox"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if not bool(mask[sr, sc]) or bool(seen[sr, sc]):
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _chebyshev_distance(a: np.ndarray, b: np.ndarray) -> int:
    diffs = np.abs(a[:, None, :] - b[None, :, :])
    return int(diffs.max(axis=2).min())


def _gray_assignments(g: np.ndarray, colors: list[int]) -> dict[int, np.ndarray]:
    gray_components = _components(g == GRAY)
    assert len(gray_components) == 4, "expected one gray patch for each rotated copy"

    visible_by_color = {color: np.argwhere(g == color) for color in colors}
    assert all(coords.size > 0 for coords in visible_by_color.values()), "each sprite color must be visible"

    assigned = {color: np.zeros(g.shape, dtype=bool) for color in colors}
    counts = {color: 0 for color in colors}
    for comp in gray_components:
        distances = [(_chebyshev_distance(comp, visible_by_color[color]), color) for color in colors]
        distances.sort(key=lambda item: item[0])
        assert distances[0][0] < distances[1][0], "gray patch must be closest to one sprite"
        color = int(distances[0][1])
        assigned[color][comp[:, 0], comp[:, 1]] = True
        counts[color] += 1

    assert all(counts[color] == 1 for color in colors), "each sprite must own exactly one gray patch"
    return assigned


def _observations(g: np.ndarray) -> tuple[int, list[dict]]:
    bg = _background(g)
    assert bg != GRAY, "gray is reserved for occlusion patches"
    colors = sorted(int(c) for c in np.unique(g) if int(c) not in (bg, GRAY))
    assert len(colors) == 4, "expected four differently colored sprite remnants"

    gray_by_color = _gray_assignments(g, colors)
    records: list[dict] = []
    owned_gray = np.zeros(g.shape, dtype=bool)
    for color in colors:
        owned_gray |= gray_by_color[color]

    assert np.array_equal(owned_gray, g == GRAY), "all gray cells must belong to a sprite patch"

    for color in colors:
        visible_full = g == color
        unknown_full = gray_by_color[color]
        box = _bbox(visible_full | unknown_full)
        visible = _crop(visible_full, box).astype(bool)
        unknown = _crop(unknown_full, box).astype(bool)
        assert visible.any() and unknown.any(), "each copy needs visible and occluded cells"
        records.append(
            {
                "color": int(color),
                "bbox": box,
                "visible": visible,
                "unknown": unknown,
            }
        )
    return int(bg), records


def _candidate_outputs(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    bg, records = _observations(g)
    candidates: dict[bytes, np.ndarray] = {}

    for rotations in itertools.permutations((0, 1, 2, 3)):
        norm_visible: list[np.ndarray] = []
        norm_unknown: list[np.ndarray] = []
        shapes: list[tuple[int, int]] = []

        for record, rotation in zip(records, rotations):
            visible = np.rot90(record["visible"], -int(rotation))
            unknown = np.rot90(record["unknown"], -int(rotation))
            norm_visible.append(visible.astype(bool))
            norm_unknown.append(unknown.astype(bool))
            shapes.append(tuple(map(int, visible.shape)))

        if len(set(shapes)) != 1:
            continue

        full_mask = np.zeros(shapes[0], dtype=bool)
        for visible in norm_visible:
            full_mask |= visible
        if not full_mask.any():
            continue
        if _bbox(full_mask) != (0, 0, full_mask.shape[0] - 1, full_mask.shape[1] - 1):
            continue

        ok = True
        for record, rotation, unknown in zip(records, rotations, norm_unknown):
            reconstructed = np.rot90(full_mask, int(rotation))
            if reconstructed.shape != record["visible"].shape:
                ok = False
                break
            known = ~record["unknown"]
            if not np.array_equal(reconstructed[known], record["visible"][known]):
                ok = False
                break
            if not bool((unknown & full_mask).any()):
                ok = False
                break
            if not bool((unknown & ~full_mask).any()):
                ok = False
                break
        if not ok:
            continue

        out = g.copy()
        for record, rotation in zip(records, rotations):
            r0, c0, _r1, _c1 = record["bbox"]
            reconstructed = np.rot90(full_mask, int(rotation))
            rr, cc = np.nonzero(record["unknown"])
            color = int(record["color"])
            for dr, dc in zip(rr, cc):
                out[r0 + int(dr), c0 + int(dc)] = color if bool(reconstructed[int(dr), int(dc)]) else bg

        if np.any(out == GRAY):
            continue
        key = out.tobytes() + repr(out.shape).encode()
        candidates.setdefault(key, out)

    return list(candidates.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    网格上四个互不接触的旋转残片互相补全，并把灰色遮挡补丁还原为本色或背景。

    Input structure:
    - 均匀背景上有四个单色精灵，颜色互不相同，分别是同一掩码旋转 0、90、180、270 度后的摆放。
    - 每个精灵都有一个灰色连通补丁，补丁覆盖该精灵的一部分以及同一局部框内的相邻背景。
    - 没有任何一个副本完整；每个掩码格至少在某个旋转副本中可见，配准方式唯一。

    Rule:
    - 将每个颜色残片连同离它最近的灰色补丁视为一个副本，枚举四个副本到四个旋转角的配准。
    - 在归一坐标下合并所有可见颜色格，得到完整共享掩码；任何已知背景格与合并掩码冲突的配准都作废。
    - 对每个灰色像素按所属副本的旋转角回贴：落在重建掩码内则涂该副本颜色，否则涂背景色。

    Key insight:
    - 完整形状从未直接出现；灰色只是未知窗口，必须用其他旋转副本的可见残片在同一归一掩码中互相补足。
    """
    candidates = _candidate_outputs(grid)
    assert len(candidates) == 1, "expected a unique rotation registration and output"
    return candidates[0]


def _neighbors4(r: int, c: int, h: int, w: int):
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = map(int, start)
    er, ec = map(int, end)
    mask[r, c] = True
    while (r, c) != (er, ec):
        options: list[tuple[int, int]] = []
        if r < er:
            options.append((r + 1, c))
        if r > er:
            options.append((r - 1, c))
        if c < ec:
            options.append((r, c + 1))
        if c > ec:
            options.append((r, c - 1))
        r, c = random.choice(options)
        mask[r, c] = True


def _connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _make_base_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    for _ in range(200):
        target = random.randint(int(0.46 * area), int(0.66 * area))
        seeds = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        random.shuffle(seeds)

        mask = np.zeros((h, w), dtype=bool)
        mask[seeds[0]] = True
        for seed in seeds[1:]:
            starts = np.argwhere(mask)
            start = tuple(map(int, starts[random.randrange(len(starts))]))
            _draw_path(mask, start, seed)

        if int(mask.sum()) > target:
            continue

        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            for rr, cc in np.argwhere(mask):
                for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                    if not bool(mask[nr, nc]):
                        frontier.append((nr, nc))
            if not frontier:
                break
            mask[random.choice(frontier)] = True

        if int((~mask).sum()) < 5:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not _connected(mask):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask
    return None


def _random_rect(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    rh = random.randint(2, min(4, h))
    rw = random.randint(2, min(4, w))
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    rect = np.zeros((h, w), dtype=bool)
    rect[r0 : r0 + rh, c0 : c0 + rw] = True
    return rect


def _sample_occluders(mask: np.ndarray) -> list[np.ndarray] | None:
    h, w = map(int, mask.shape)
    for _ in range(500):
        rects: list[np.ndarray] = []
        hidden_signatures: set[tuple[tuple[int, int], ...]] = set()
        for _idx in range(4):
            rect = None
            for _try in range(80):
                candidate = _random_rect((h, w))
                hidden = candidate & mask
                covered_bg = candidate & ~mask
                if not hidden.any() or not covered_bg.any():
                    continue
                if not (mask & ~candidate).any():
                    continue
                signature = tuple(map(tuple, np.argwhere(hidden)))
                if signature in hidden_signatures:
                    continue
                rect = candidate
                hidden_signatures.add(signature)
                break
            if rect is None:
                break
            rects.append(rect)

        if len(rects) != 4:
            continue
        all_hidden = rects[0].copy()
        any_hidden = np.zeros(mask.shape, dtype=bool)
        for rect in rects:
            all_hidden &= rect
            any_hidden |= rect & mask
        if not any_hidden.any():
            continue
        if (all_hidden & mask).any():
            continue
        return rects
    return None


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = map(int, patch.shape)
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(bg)
    assert not np.any(view[mask] != int(bg)), "generated copies must not overlap"
    view[mask] = patch[mask]


def _render_copy(mask: np.ndarray, rect: np.ndarray, rotation: int, color: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    sprite = np.rot90(mask, int(rotation))
    occluder = np.rot90(rect, int(rotation))

    inp = np.full(sprite.shape, int(bg), dtype=int)
    inp[sprite] = int(color)
    inp[occluder] = GRAY

    out = np.full(sprite.shape, int(bg), dtype=int)
    out[sprite] = int(color)
    return inp, out


def _render_scene(
    mask: np.ndarray,
    rects: list[np.ndarray],
    rotations: list[int],
    colors: list[int],
    bg: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    max_dim = max(mask.shape)
    gap_r = random.randint(3, 6)
    gap_c = random.randint(3, 6)
    total_h = 2 * max_dim + gap_r
    total_w = 2 * max_dim + gap_c
    height = random.randint(max(20, total_h + 2), 30)
    width = random.randint(max(20, total_w + 2), 30)

    top = random.randint(1, height - total_h - 1)
    left = random.randint(1, width - total_w - 1)
    slots = [
        (top, left),
        (top, left + max_dim + gap_c),
        (top + max_dim + gap_r, left),
        (top + max_dim + gap_r, left + max_dim + gap_c),
    ]
    random.shuffle(slots)

    inp = np.full((height, width), int(bg), dtype=int)
    expected = np.full((height, width), int(bg), dtype=int)
    copies: list[dict] = []

    for rect, rotation, color, slot in zip(rects, rotations, colors, slots):
        local_in, local_out = _render_copy(mask, rect, rotation, color, bg)
        lh, lw = map(int, local_in.shape)
        sr, sc = slot
        r0 = sr + random.randint(0, max_dim - lh)
        c0 = sc + random.randint(0, max_dim - lw)
        _paste_non_bg(inp, local_in, r0, c0, bg)
        _paste_non_bg(expected, local_out, r0, c0, bg)
        copies.append(
            {
                "rotation": int(rotation),
                "color": int(color),
                "position": (int(r0), int(c0)),
                "rect": rect.copy(),
            }
        )

    return inp, expected, {"mask": mask.copy(), "rects": [r.copy() for r in rects], "copies": copies, "bg": int(bg)}


def _semantic_generation_check(inp: np.ndarray, expected: np.ndarray, info: dict) -> bool:
    mask = info["mask"]
    rects = info["rects"]

    visible_union = np.zeros(mask.shape, dtype=bool)
    hidden_union = np.zeros(mask.shape, dtype=bool)
    for rect in rects:
        hidden_mask = rect & mask
        hidden_bg = rect & ~mask
        if not hidden_mask.any() or not hidden_bg.any():
            return False
        visible_union |= mask & ~rect
        hidden_union |= hidden_mask

    if not np.array_equal(visible_union, mask):
        return False
    if not hidden_union.any() or np.any(expected == GRAY):
        return False

    changed = inp != expected
    if not np.array_equal(changed, inp == GRAY):
        return False

    candidates = _candidate_outputs(inp)
    if len(candidates) != 1:
        return False
    return bool(np.array_equal(candidates[0], expected))


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    h = random.randint(5, 8)
    w_options = [w for w in range(5, 9) if w != h]
    w = random.choice(w_options)

    mask = _make_base_mask(h, w)
    if mask is None:
        return None

    rects = _sample_occluders(mask)
    if rects is None:
        return None

    bg = random.choice([0, 1, 2, 3, 4, 6, 7, 8, 9])
    palette = [c for c in range(10) if c not in (bg, GRAY)]
    colors = random.sample(palette, 4)
    rotations = random.sample([0, 1, 2, 3], 4)

    inp, expected, info = _render_scene(mask, rects, rotations, colors, bg)
    try:
        if not _semantic_generation_check(inp, expected, info):
            return None
    except AssertionError:
        return None
    return inp, expected, info


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid rotating-fragment completion input")
