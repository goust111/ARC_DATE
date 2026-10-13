from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Container:
    border_color: int
    pattern_color: int
    signal_color: int
    bbox: tuple[int, int, int, int]
    pattern_mask: np.ndarray
    signal_pos: tuple[int, int]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rect_frame_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(local, expected))


def _find_containers(grid: np.ndarray) -> list[Container]:
    containers: list[Container] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != 0):
        for coords in _components(grid == color):
            if not _is_rect_frame_component(coords):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            nonzero = interior[interior != 0]
            if nonzero.size < 2:
                continue
            colors, counts = np.unique(nonzero, return_counts=True)
            signal_candidates = [int(col) for col, cnt in zip(colors, counts, strict=False) if int(cnt) == 1]
            pattern_candidates = [int(col) for col, cnt in zip(colors, counts, strict=False) if int(cnt) > 1]
            if len(signal_candidates) != 1 or len(pattern_candidates) != 1:
                continue
            signal_color = signal_candidates[0]
            pattern_color = pattern_candidates[0]
            signal_coords = np.argwhere(interior == signal_color)
            if signal_coords.shape != (1, 2):
                continue
            pattern_mask = (interior == pattern_color)
            if not np.any(pattern_mask):
                continue
            containers.append(
                Container(
                    border_color=int(color),
                    pattern_color=pattern_color,
                    signal_color=signal_color,
                    bbox=(r0, c0, r1, c1),
                    pattern_mask=pattern_mask.copy(),
                    signal_pos=(int(signal_coords[0, 0]), int(signal_coords[0, 1])),
                )
            )
    assert containers, "expected at least one valid container"
    return containers


def _flip_axis(shape: tuple[int, int], signal_pos: tuple[int, int]) -> str:
    h, w = shape
    r, c = signal_pos
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    on_center_row = h % 2 == 1 and r == h // 2
    on_center_col = w % 2 == 1 and c == w // 2

    if on_center_row and not on_center_col:
        return "horizontal"
    if on_center_col and not on_center_row:
        return "vertical"

    dr = abs(r - center_r)
    dc = abs(c - center_c)
    if dc > dr:
        return "horizontal"
    if dr > dc:
        return "vertical"

    return "horizontal" if w >= h else "vertical"


def _flipped(mask: np.ndarray, axis: str) -> np.ndarray:
    if axis == "horizontal":
        return np.fliplr(mask)
    assert axis == "vertical"
    return np.flipud(mask)


def _paired_indices(containers: list[Container]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    degree = [0] * len(containers)
    for i, a in enumerate(containers):
        matches = [
            j
            for j, b in enumerate(containers)
            if i != j
            and a.signal_color == b.border_color
            and b.signal_color == a.border_color
        ]
        degree[i] = len(matches)
        assert all(d <= 1 for d in degree), "pairing must be unambiguous"
        if matches and i < matches[0]:
            pairs.append((i, matches[0]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap paired container patterns and flip them according to each destination signal.

    Input structure:
    - The background is black.
    - Each container is a monochrome rectangular frame around an interior region.
    - Every interior contains one multi-pixel pattern color and one singleton signal pixel.

    Rule:
    - Pair containers by reciprocal frame-color/signal-color matching.
    - For each paired destination container, flip the incoming pattern according to
      that container's signal position: left/right implies a horizontal flip,
      top/bottom implies a vertical flip.
    - Clear all signal pixels to black.
    """
    g = np.asarray(grid, dtype=int)
    containers = _find_containers(g)
    out = g.copy()

    for i, j in _paired_indices(containers):
        a = containers[i]
        b = containers[j]
        assert a.pattern_mask.shape == b.pattern_mask.shape, "paired interiors must match in size"

        ar0, ac0, ar1, ac1 = a.bbox
        br0, bc0, br1, bc1 = b.bbox
        a_patch = out[ar0 + 1 : ar1, ac0 + 1 : ac1]
        b_patch = out[br0 + 1 : br1, bc0 + 1 : bc1]
        a_patch[:] = 0
        b_patch[:] = 0

        into_a = _flipped(b.pattern_mask, _flip_axis(a.pattern_mask.shape, a.signal_pos))
        into_b = _flipped(a.pattern_mask, _flip_axis(b.pattern_mask.shape, b.signal_pos))
        a_patch[into_a] = b.pattern_color
        b_patch[into_b] = a.pattern_color

    for container in containers:
        r0, c0, _r1, _c1 = container.bbox
        sr, sc = container.signal_pos
        out[r0 + 1 + sr, c0 + 1 + sc] = 0

    return out


# --------------- generator ---------------

COLOR_POOL = list(range(1, 10))
GRID_MIN, GRID_MAX = 20, 30
PAIR_COUNT_CHOICES = (1, 2)
INTERIOR_SHAPES = ((5, 5), (5, 7), (7, 5), (7, 7), (7, 9), (9, 7))
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120
PATTERN_RETRIES = 80
PAIR_AXIS_OPTIONS = (("horizontal", "vertical"), ("vertical", "horizontal"))


@dataclass(frozen=True)
class ContainerSpec:
    border_color: int
    pattern_color: int
    signal_color: int
    interior_shape: tuple[int, int]
    pattern_mask: np.ndarray
    signal_pos: tuple[int, int]


def _signal_pos(shape: tuple[int, int], axis: str) -> tuple[int, int]:
    h, w = shape
    assert h % 2 == 1 and w % 2 == 1
    cr, cc = h // 2, w // 2
    if axis == "horizontal":
        side = random.choice((-1, 1))
        cols = list(range(0, cc)) if side < 0 else list(range(cc + 1, w))
        return cr, random.choice(cols)
    assert axis == "vertical"
    side = random.choice((-1, 1))
    rows = list(range(0, cr)) if side < 0 else list(range(cr + 1, h))
    return random.choice(rows), cc


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _valid_pattern(mask: np.ndarray) -> bool:
    if not _mask_connected(mask):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    if sub.all():
        return False
    hh, ww = sub.shape
    if hh >= 3 and ww >= 3:
        frame = np.zeros_like(sub, dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        if np.array_equal(sub, frame):
            return False
    return True


def _random_pattern(shape: tuple[int, int], signal_pos: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    allowed = [(r, c) for r in range(h) for c in range(w) if (r, c) != signal_pos]
    max_area = min(8, len(allowed) - 1)
    if max_area < 3:
        return None
    for _ in range(PATTERN_RETRIES):
        target = random.randint(3, max_area)
        sr, sc = random.choice(allowed)
        chosen = {(sr, sc)}
        frontier = {(sr - 1, sc), (sr + 1, sc), (sr, sc - 1), (sr, sc + 1)}
        while len(chosen) < target and frontier:
            candidates = [
                (r, c)
                for r, c in frontier
                if 0 <= r < h and 0 <= c < w and (r, c) != signal_pos and (r, c) not in chosen
            ]
            if not candidates:
                break
            nr, nc = random.choice(candidates)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) != target:
            continue
        mask = np.zeros(shape, dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if _valid_pattern(mask):
            return mask
    return None


def _build_patch(spec: ContainerSpec) -> np.ndarray:
    h, w = spec.interior_shape
    patch = np.zeros((h + 2, w + 2), dtype=int)
    patch[0, :] = spec.border_color
    patch[-1, :] = spec.border_color
    patch[:, 0] = spec.border_color
    patch[:, -1] = spec.border_color
    interior = patch[1:-1, 1:-1]
    interior[spec.pattern_mask] = spec.pattern_color
    sr, sc = spec.signal_pos
    assert interior[sr, sc] == 0
    interior[sr, sc] = spec.signal_color
    return patch


def _container_from_spec(spec: ContainerSpec, top: int, left: int) -> Container:
    h, w = spec.interior_shape
    return Container(
        border_color=spec.border_color,
        pattern_color=spec.pattern_color,
        signal_color=spec.signal_color,
        bbox=(top, left, top + h + 1, left + w + 1),
        pattern_mask=spec.pattern_mask.copy(),
        signal_pos=spec.signal_pos,
    )


def _place_patches(patches: list[np.ndarray]) -> tuple[int, int, list[tuple[int, int]]] | None:
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    min_h = max(GRID_MIN, max_h + 2)
    min_w = max(GRID_MIN, max_w + 2)
    height = random.randint(min_h, GRID_MAX)
    width = random.randint(min_w, GRID_MAX)
    occupied = np.zeros((height, width), dtype=bool)
    placements = [(-1, -1)] * len(patches)

    for idx in sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True):
        ph, pw = patches[idx].shape
        candidates: list[tuple[int, int]] = []
        for r in range(height - ph + 1):
            for c in range(width - pw + 1):
                rr0 = max(0, r - 1)
                cc0 = max(0, c - 1)
                rr1 = min(height, r + ph + 1)
                cc1 = min(width, c + pw + 1)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        placements[idx] = random.choice(candidates)
        r, c = placements[idx]
        occupied[r : r + ph, c : c + pw] = True
    return height, width, placements


def _unpaired_count(pair_count: int) -> int:
    if pair_count == 1:
        return random.choice((0, 1, 2))
    return random.choice((0, 1))


def _gen_specs() -> list[ContainerSpec] | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    unpaired_count = _unpaired_count(pair_count)
    total_borders = 2 * pair_count + unpaired_count
    if total_borders + unpaired_count > len(COLOR_POOL) - 2:
        return None

    colors = COLOR_POOL.copy()
    random.shuffle(colors)
    border_colors = colors[:total_borders]
    cursor = total_borders
    spare_signals = colors[cursor : cursor + unpaired_count]
    cursor += unpaired_count
    pattern_pool = colors[cursor:]
    if len(pattern_pool) < 2:
        return None

    specs: list[ContainerSpec] = []
    border_idx = 0
    spare_idx = 0

    for _ in range(pair_count):
        a_border = border_colors[border_idx]
        b_border = border_colors[border_idx + 1]
        border_idx += 2
        shape = random.choice(INTERIOR_SHAPES)
        axis_a, axis_b = random.choice(PAIR_AXIS_OPTIONS)
        sig_a = _signal_pos(shape, axis_a)
        sig_b = _signal_pos(shape, axis_b)
        mask_a = _random_pattern(shape, sig_a)
        mask_b = _random_pattern(shape, sig_b)
        if mask_a is None or mask_b is None:
            return None
        spec_a = ContainerSpec(
            border_color=a_border,
            pattern_color=random.choice(pattern_pool),
            signal_color=b_border,
            interior_shape=shape,
            pattern_mask=mask_a,
            signal_pos=sig_a,
        )
        spec_b = ContainerSpec(
            border_color=b_border,
            pattern_color=random.choice(pattern_pool),
            signal_color=a_border,
            interior_shape=shape,
            pattern_mask=mask_b,
            signal_pos=sig_b,
        )
        if _flipped(mask_b, _flip_axis(shape, sig_a))[sig_a]:
            return None
        if _flipped(mask_a, _flip_axis(shape, sig_b))[sig_b]:
            return None
        specs.extend((spec_a, spec_b))

    for _ in range(unpaired_count):
        border_color = border_colors[border_idx]
        border_idx += 1
        shape = random.choice(INTERIOR_SHAPES)
        axis = random.choice(("horizontal", "vertical"))
        sig = _signal_pos(shape, axis)
        mask = _random_pattern(shape, sig)
        if mask is None:
            return None
        specs.append(
            ContainerSpec(
                border_color=border_color,
                pattern_color=random.choice(pattern_pool),
                signal_color=spare_signals[spare_idx],
                interior_shape=shape,
                pattern_mask=mask,
                signal_pos=sig,
            )
        )
        spare_idx += 1

    random.shuffle(specs)
    return specs


def _build_grid(specs: list[ContainerSpec]) -> np.ndarray | None:
    patches = [_build_patch(spec) for spec in specs]
    for _ in range(PLACEMENT_RETRIES):
        placed = _place_patches(patches)
        if placed is None:
            continue
        height, width, placements = placed
        grid = np.zeros((height, width), dtype=int)
        containers = []
        for spec, patch, (r, c) in zip(specs, patches, placements, strict=False):
            grid[r : r + patch.shape[0], c : c + patch.shape[1]] = patch
            containers.append(_container_from_spec(spec, r, c))
        if len(_find_containers(grid)) != len(specs):
            continue
        paired = _paired_indices(containers)
        if len(paired) != sum(spec.signal_color in {other.border_color for other in specs} for spec in specs) // 2:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        specs = _gen_specs()
        if specs is None:
            continue
        grid = _build_grid(specs)
        if grid is None:
            continue
        out = main(grid)
        if not isinstance(out, np.ndarray) or np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
