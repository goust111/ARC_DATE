from __future__ import annotations

from collections import Counter
import random

import numpy as np


SIDES = ("top", "right", "bottom", "left")
ROTATIONS = {"top": 0, "right": 3, "bottom": 2, "left": 1}


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        coords = np.argwhere(grid == int(color))
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 5 or w < 5:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(h, w)
        color_mask = sub == int(color)
        if not bool(color_mask[0, 0] and color_mask[0, -1] and color_mask[-1, 0] and color_mask[-1, -1]):
            continue
        if not np.all(~color_mask | border):
            continue
        if np.any(border & (sub != int(bg)) & ~color_mask):
            continue

        missing = border & ~color_mask
        gap_count = int(missing.sum())
        if gap_count > 4:
            continue

        frames.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "inner": sub[1:-1, 1:-1].copy(),
                "missing": missing,
                "gap_count": gap_count,
            }
        )
    return frames


def _classify_frames(frames: list[dict], bg: int) -> tuple[list[dict], list[dict]]:
    sources: list[dict] = []
    targets: list[dict] = []
    for frame in frames:
        inner = frame["inner"]
        if int(frame["gap_count"]) == 0:
            pixel_count = int(np.count_nonzero(inner != int(bg)))
            if pixel_count <= 0:
                continue
            enriched = dict(frame)
            enriched["pixel_count"] = pixel_count
            sources.append(enriched)
            continue

        if bool(np.any(inner != int(bg))):
            continue

        missing = frame["missing"]
        side_counts = {
            "top": int(missing[0, 1:-1].sum()),
            "right": int(missing[1:-1, -1].sum()),
            "bottom": int(missing[-1, 1:-1].sum()),
            "left": int(missing[1:-1, 0].sum()),
        }
        best = max(side_counts.values())
        if best <= 0:
            continue
        main_sides = [side for side, count in side_counts.items() if int(count) == int(best)]
        if len(main_sides) != 1:
            continue

        enriched = dict(frame)
        enriched["main_side"] = main_sides[0]
        targets.append(enriched)

    return sources, targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match full source frames to gapped target frames by count, rotate the source
    interior toward the target's main gap side, and copy it into the target.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    frames = _detect_frames(g, bg)
    sources, targets = _classify_frames(frames, bg)

    assert sources and targets, "expected both source and target frames"
    assert len(sources) == len(targets), "source/target frame counts must match"

    by_pixels: dict[int, dict] = {}
    by_gaps: dict[int, dict] = {}
    for src in sources:
        key = int(src["pixel_count"])
        assert key not in by_pixels, "source pixel counts must be unique"
        by_pixels[key] = src
    for tgt in targets:
        key = int(tgt["gap_count"])
        assert key not in by_gaps, "target gap counts must be unique"
        by_gaps[key] = tgt

    assert set(by_pixels) == set(by_gaps), "source/target keys must match"

    out = g.copy()
    for src in sources:
        r0, c0, r1, c1 = map(int, src["bbox"])
        out[r0 + 1 : r1, c0 + 1 : c1] = int(bg)

    for key, src in by_pixels.items():
        tgt = by_gaps[int(key)]
        rotated = np.rot90(src["inner"], ROTATIONS[str(tgt["main_side"])])
        r0, c0, r1, c1 = map(int, tgt["bbox"])
        assert rotated.shape == (int(r1 - r0 - 1), int(c1 - c0 - 1))
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated

    return out


INNER_DIM_CHOICES = [
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),
    (3, 5),
    (5, 3),
    (4, 5),
    (5, 4),
]
COUNT_CHOICES = (2, 3, 4)
GAP_TEMPLATES = {
    1: [(1, 0, 0, 0)],
    2: [(2, 0, 0, 0)],
    3: [(3, 0, 0, 0), (2, 1, 0, 0)],
    4: [(4, 0, 0, 0), (3, 1, 0, 0), (2, 1, 1, 0)],
}
GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 31)
PLACEMENT_PADDING = 1
GEN_ATTEMPTS = 300


def _grow_connected_cells(h: int, w: int, n: int) -> list[tuple[int, int]] | None:
    for _ in range(80):
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < int(n):
            frontier = set()
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(sorted(frontier)))
        if len(cells) == int(n):
            return sorted(cells)
    return None


def _is_asymmetric(patch: np.ndarray, bg: int) -> bool:
    if int(np.count_nonzero(patch != int(bg))) <= 1:
        return True
    for k in (1, 2, 3):
        rotated = np.rot90(patch, k)
        if rotated.shape == patch.shape and np.array_equal(rotated, patch):
            return False
    return True


def _make_pattern(h: int, w: int, count: int, palette: list[int], bg: int) -> np.ndarray | None:
    for _ in range(120):
        coords = _grow_connected_cells(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), int(bg), dtype=int)
        n_colors = min(len(palette), random.choice((2, 3)))
        colors = random.sample(palette, n_colors)
        order = coords[:]
        random.shuffle(order)
        for idx, (r, c) in enumerate(order):
            patch[int(r), int(c)] = int(colors[idx % n_colors])
        if len(set(int(v) for v in patch[patch != int(bg)])) < 2:
            continue
        if not _is_asymmetric(patch, bg):
            continue
        return patch
    return None


def _side_capacity(inner_h: int, inner_w: int) -> dict[str, int]:
    return {
        "top": int(inner_w),
        "right": int(inner_h),
        "bottom": int(inner_w),
        "left": int(inner_h),
    }


def _sample_gap_counts(total: int, main_side: str, inner_h: int, inner_w: int) -> dict[str, int] | None:
    caps = _side_capacity(inner_h, inner_w)
    others = [side for side in SIDES if side != main_side]
    choices: list[dict[str, int]] = []
    for tpl in GAP_TEMPLATES[int(total)]:
        main_count, *rest = tpl
        if int(main_count) > int(caps[main_side]):
            continue
        for perm in set(__import__("itertools").permutations(rest)):
            counts = {main_side: int(main_count)}
            ok = True
            for side, value in zip(others, perm):
                if int(value) > int(caps[side]):
                    ok = False
                    break
                counts[side] = int(value)
            if ok:
                choices.append(counts)
    if not choices:
        return None
    return random.choice(choices)


def _edge_positions(inner_h: int, inner_w: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, inner_w + 1)]
    if side == "right":
        return [(r, inner_w + 1) for r in range(1, inner_h + 1)]
    if side == "bottom":
        return [(inner_h + 1, c) for c in range(1, inner_w + 1)]
    return [(r, 0) for r in range(1, inner_h + 1)]


def _sample_gaps(inner_h: int, inner_w: int, counts: dict[str, int]) -> set[tuple[int, int]]:
    gaps: set[tuple[int, int]] = set()
    for side in SIDES:
        need = int(counts.get(side, 0))
        if need == 0:
            continue
        choices = _edge_positions(inner_h, inner_w, side)
        picks = random.sample(choices, need)
        for cell in picks:
            gaps.add(cell)
    return gaps


def _build_source_patch(pattern: np.ndarray, frame_color: int, bg: int) -> np.ndarray:
    h, w = map(int, pattern.shape)
    patch = np.full((h + 2, w + 2), int(bg), dtype=int)
    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)
    patch[1:-1, 1:-1] = pattern
    return patch


def _build_target_patch(inner_h: int, inner_w: int, frame_color: int, bg: int, gaps: set[tuple[int, int]]) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), int(bg), dtype=int)
    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)
    for r, c in gaps:
        patch[int(r), int(c)] = int(bg)
    return patch


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = blocked.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    return not bool(blocked[r0 : r0 + h, c0 : c0 + w].any())


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int) -> None:
    H, W = blocked.shape
    rr0 = max(0, int(r0 - padding))
    cc0 = max(0, int(c0 - padding))
    rr1 = min(H, int(r0 + h + padding))
    cc1 = min(W, int(c0 + w + padding))
    blocked[rr0:rr1, cc0:cc1] = True


def _place_patches(shape: tuple[int, int], patches: list[tuple[str, np.ndarray]]) -> dict[str, tuple[int, int]] | None:
    H, W = map(int, shape)
    blocked = np.zeros((H, W), dtype=bool)
    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda idx: int(patches[idx][1].shape[0] * patches[idx][1].shape[1]), reverse=True)

    placements: dict[str, tuple[int, int]] = {}
    for idx in order:
        name, patch = patches[idx]
        h, w = map(int, patch.shape)
        positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if not _can_place(blocked, r0, c0, h, w):
                continue
            placements[name] = (int(r0), int(c0))
            _mark_blocked(blocked, r0, c0, h, w, PLACEMENT_PADDING)
            placed = True
            break
        if not placed:
            return None
    return placements


def _render_expected(inp: np.ndarray, bg: int, pair_info: list[dict], placements: dict[str, tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    for info in pair_info:
        src_r0, src_c0 = placements[info["source_name"]]
        src_h, src_w = map(int, info["pattern"].shape)
        out[src_r0 + 1 : src_r0 + 1 + src_h, src_c0 + 1 : src_c0 + 1 + src_w] = int(bg)

    for info in pair_info:
        tgt_r0, tgt_c0 = placements[info["target_name"]]
        rotated = np.rot90(info["pattern"], ROTATIONS[info["main_side"]])
        tgt_h, tgt_w = map(int, rotated.shape)
        out[tgt_r0 + 1 : tgt_r0 + 1 + tgt_h, tgt_c0 + 1 : tgt_c0 + 1 + tgt_w] = rotated
    return out


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(2, 3)
    counts = random.sample(list(COUNT_CHOICES), n_pairs)
    sides = random.sample(list(SIDES), n_pairs)

    bg = random.randrange(10)
    colors = [color for color in range(10) if color != int(bg)]
    random.shuffle(colors)
    frame_colors = colors[: 2 * n_pairs]
    pattern_palette = colors[2 * n_pairs :]
    if len(pattern_palette) < 2:
        return None

    pair_info: list[dict] = []
    patches: list[tuple[str, np.ndarray]] = []
    total_area = 0

    for idx, (count, side) in enumerate(zip(counts, sides)):
        dims = [dims for dims in INNER_DIM_CHOICES if dims[0] * dims[1] >= int(count)]
        src_h, src_w = random.choice(dims)
        pattern = _make_pattern(src_h, src_w, int(count), pattern_palette, bg)
        if pattern is None:
            return None

        if side in {"top", "bottom"}:
            tgt_h, tgt_w = int(src_h), int(src_w)
        else:
            tgt_h, tgt_w = int(src_w), int(src_h)

        gap_counts = _sample_gap_counts(int(count), side, tgt_h, tgt_w)
        if gap_counts is None:
            return None
        gaps = _sample_gaps(tgt_h, tgt_w, gap_counts)

        source_name = f"source_{idx}"
        target_name = f"target_{idx}"
        source_patch = _build_source_patch(pattern, int(frame_colors[2 * idx]), bg)
        target_patch = _build_target_patch(tgt_h, tgt_w, int(frame_colors[2 * idx + 1]), bg, gaps)

        patches.append((source_name, source_patch))
        patches.append((target_name, target_patch))
        total_area += int(source_patch.shape[0] * source_patch.shape[1] + target_patch.shape[0] * target_patch.shape[1])
        pair_info.append(
            {
                "count": int(count),
                "main_side": side,
                "pattern": pattern,
                "source_name": source_name,
                "target_name": target_name,
            }
        )

    min_h = max(patch.shape[0] for _, patch in patches)
    min_w = max(patch.shape[1] for _, patch in patches)
    for _ in range(60):
        H = random.randint(max(GRID_H_RANGE[0], min_h + 2), GRID_H_RANGE[1] - 1)
        W = random.randint(max(GRID_W_RANGE[0], min_w + 2), GRID_W_RANGE[1] - 1)
        if H * W < total_area * 2:
            continue
        placements = _place_patches((H, W), patches)
        if placements is None:
            continue

        inp = np.full((H, W), int(bg), dtype=int)
        for name, patch in patches:
            r0, c0 = placements[name]
            h, w = map(int, patch.shape)
            inp[r0 : r0 + h, c0 : c0 + w] = patch

        expected = _render_expected(inp, bg, pair_info, placements)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
