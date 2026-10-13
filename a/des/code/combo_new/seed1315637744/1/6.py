import random
from collections import deque
from dataclasses import dataclass

import numpy as np


BG = 0
FRAME_COLOR = 5
GEN_RETRIES = 500
MIN_GAP = 1
CORNER_ORDER = ("tl", "tr", "br", "bl")
CW_ROTATIONS = {"tl": 0, "tr": 1, "br": 2, "bl": 3}


@dataclass
class TemplateInfo:
    bbox: tuple[int, int, int, int]
    frame_color: int
    pattern_color: int
    mask: np.ndarray


@dataclass
class EnvelopeInfo:
    bbox: tuple[int, int, int, int]
    fill_color: int
    marker_color: int
    corner: str


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_full_frame(subgrid: np.ndarray, color: int) -> bool:
    h, w = subgrid.shape
    if h != w or h < 4:
        return False
    if np.any(subgrid[0, :] != color) or np.any(subgrid[-1, :] != color):
        return False
    if np.any(subgrid[:, 0] != color) or np.any(subgrid[:, -1] != color):
        return False
    return True


def _find_template(grid: np.ndarray, bg: int) -> TemplateInfo:
    candidates: list[TemplateInfo] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            side = sub.shape[0]
            if not _is_full_frame(sub, color):
                continue
            if len(comp) != 4 * side - 4:
                continue
            interior = sub[1:-1, 1:-1]
            inside_colors = [int(v) for v in np.unique(interior) if int(v) != bg]
            if len(inside_colors) != 1:
                continue
            pattern_color = inside_colors[0]
            if pattern_color == color:
                continue
            mask = interior == pattern_color
            if mask.shape[0] < 2 or mask.sum() < 2:
                continue
            if not bool(mask[0, 0]):
                continue
            if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
                continue
            candidates.append(
                TemplateInfo(
                    bbox=(r0, c0, r1, c1),
                    frame_color=color,
                    pattern_color=pattern_color,
                    mask=mask.copy(),
                )
            )
    assert len(candidates) == 1, f"expected exactly one template, found {len(candidates)}"
    return candidates[0]


def _corner_from_marker(subgrid: np.ndarray, marker_color: int) -> str | None:
    pos = np.argwhere(subgrid == marker_color)
    if pos.shape != (1, 2):
        return None
    r, c = map(int, pos[0])
    h, w = subgrid.shape
    if (r, c) == (0, 0):
        return "tl"
    if (r, c) == (0, w - 1):
        return "tr"
    if (r, c) == (h - 1, w - 1):
        return "br"
    if (r, c) == (h - 1, 0):
        return "bl"
    return None


def _find_envelopes(grid: np.ndarray, bg: int, template_bbox: tuple[int, int, int, int]) -> list[EnvelopeInfo]:
    mask = grid != bg
    r0, c0, r1, c1 = template_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    envelopes: list[EnvelopeInfo] = []
    for comp in _components(mask):
        br0, bc0, br1, bc1 = _bbox(comp)
        area = (br1 - br0 + 1) * (bc1 - bc0 + 1)
        if len(comp) != area:
            continue
        sub = grid[br0 : br1 + 1, bc0 : bc1 + 1]
        if np.any(sub == bg):
            continue
        colors, counts = np.unique(sub, return_counts=True)
        if len(colors) != 2:
            continue
        order = np.argsort(-counts)
        colors = colors[order]
        counts = counts[order]
        if int(counts[0]) != area - 1 or int(counts[1]) != 1:
            continue
        fill_color = int(colors[0])
        marker_color = int(colors[1])
        corner = _corner_from_marker(sub, marker_color)
        if corner is None:
            continue
        envelopes.append(
            EnvelopeInfo(
                bbox=(br0, bc0, br1, bc1),
                fill_color=fill_color,
                marker_color=marker_color,
                corner=corner,
            )
        )
    assert envelopes, "expected at least one envelope"
    return envelopes


def _parse_input(grid: np.ndarray) -> tuple[int, TemplateInfo, list[EnvelopeInfo]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    template = _find_template(g, bg)
    envelopes = _find_envelopes(g, bg, template.bbox)
    return bg, template, envelopes


def _stamp_origin(
    bbox: tuple[int, int, int, int], corner: str, shape: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    if corner == "tl":
        return r0, c0
    if corner == "tr":
        return r0, c1 - w + 1
    if corner == "br":
        return r1 - h + 1, c1 - w + 1
    if corner == "bl":
        return r1 - h + 1, c0
    raise ValueError(f"unknown corner {corner}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the stamp template and print it into each envelope corner indicated by a marker.

    Input structure:
    - The background is black.
    - One small square template is enclosed by a 1-cell gray frame; inside it, a single colored non-symmetric stamp pattern sits on black cells.
    - Several larger solid rectangles act as envelopes; each is filled with one color except for exactly one corner marker pixel of another color.

    Rule:
    - Extract the stamp mask from inside the framed template, ignoring the gray border.
    - For each envelope, detect which corner contains the marker: top-left means 0 degrees, top-right means 90 degrees clockwise, bottom-right means 180 degrees, bottom-left means 270 degrees clockwise.
    - Paint the rotated stamp into that envelope corner using the marker's color, overwriting the marker cell and leaving the template unchanged.

    Key insight:
    - The marker location encodes rotation, while the marker color independently encodes the printed stamp color.
    """
    g = np.asarray(grid, dtype=int)
    _, template, envelopes = _parse_input(g)
    out = g.copy()
    base_mask = template.mask

    for env in envelopes:
        rotated = np.rot90(base_mask, -CW_ROTATIONS[env.corner])
        eh = env.bbox[2] - env.bbox[0] + 1
        ew = env.bbox[3] - env.bbox[1] + 1
        ph, pw = rotated.shape
        assert eh >= ph and ew >= pw, "envelope is smaller than the stamp"
        sr, sc = _stamp_origin(env.bbox, env.corner, rotated.shape)
        patch = out[sr : sr + ph, sc : sc + pw]
        patch[rotated] = env.marker_color

    return out


# --------------- generator ---------------


def _pattern_is_valid(mask: np.ndarray) -> bool:
    if mask.shape[0] < 3 or mask.shape[0] != mask.shape[1]:
        return False
    if not bool(mask[0, 0]):
        return False
    if int(mask.sum()) < 3:
        return False
    return not any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _random_stamp_mask(size: int, rng: random.Random) -> np.ndarray:
    target = rng.randint(size + 1, size * size - 1)
    for _ in range(200):
        mask = np.zeros((size, size), dtype=bool)
        cells = {(0, 0)}
        frontier = {(0, 1), (1, 0)}
        while len(cells) < target and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            r, c = cell
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        for r, c in cells:
            mask[r, c] = True
        if _pattern_is_valid(mask):
            return mask
    raise RuntimeError("failed to generate a non-symmetric stamp mask")


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = occupied.shape
    r1, c1 = r0 + h - 1, c0 + w - 1
    if r1 >= H or c1 >= W:
        return False
    rr0 = max(0, r0 - MIN_GAP)
    cc0 = max(0, c0 - MIN_GAP)
    rr1 = min(H, r1 + MIN_GAP + 1)
    cc1 = min(W, c1 + MIN_GAP + 1)
    return not np.any(occupied[rr0:rr1, cc0:cc1])


def _place_objects(
    shapes: dict[str, tuple[int, int]], H: int, W: int, rng: random.Random
) -> dict[str, tuple[int, int]]:
    occupied = np.zeros((H, W), dtype=bool)
    positions: dict[str, tuple[int, int]] = {}
    order = sorted(shapes.items(), key=lambda item: item[1][0] * item[1][1], reverse=True)
    for name, (h, w) in order:
        placed = False
        for _ in range(300):
            r0 = rng.randint(0, H - h)
            c0 = rng.randint(0, W - w)
            if not _can_place(occupied, r0, c0, h, w):
                continue
            occupied[r0 : r0 + h, c0 : c0 + w] = True
            positions[name] = (r0, c0)
            placed = True
            break
        if not placed:
            raise RuntimeError("placement failed")
    return positions


def _draw_template(
    grid: np.ndarray,
    top_left: tuple[int, int],
    mask: np.ndarray,
    pattern_color: int,
) -> None:
    r0, c0 = top_left
    size = mask.shape[0] + 2
    r1, c1 = r0 + size - 1, c0 + size - 1
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    interior[mask] = pattern_color


def _draw_envelope(
    grid: np.ndarray,
    top_left: tuple[int, int],
    shape: tuple[int, int],
    fill_color: int,
    marker_color: int,
    corner: str,
) -> None:
    r0, c0 = top_left
    h, w = shape
    r1, c1 = r0 + h - 1, c0 + w - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = fill_color
    if corner == "tl":
        grid[r0, c0] = marker_color
    elif corner == "tr":
        grid[r0, c1] = marker_color
    elif corner == "br":
        grid[r1, c1] = marker_color
    elif corner == "bl":
        grid[r1, c0] = marker_color
    else:
        raise ValueError(f"unknown corner {corner}")


def _validate_generated(
    grid: np.ndarray,
    expected_mask: np.ndarray,
    expected_env_count: int,
    expected_corners: set[str],
) -> None:
    bg, template, envelopes = _parse_input(grid)
    assert bg == BG
    assert np.array_equal(template.mask, expected_mask)
    assert len(envelopes) == expected_env_count
    assert {env.corner for env in envelopes} == expected_corners
    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid)
    template_out = out[
        template.bbox[0] : template.bbox[2] + 1, template.bbox[1] : template.bbox[3] + 1
    ]
    template_in = grid[
        template.bbox[0] : template.bbox[2] + 1, template.bbox[1] : template.bbox[3] + 1
    ]
    assert np.array_equal(template_in, template_out)


def _gen_attempt(rng: random.Random) -> np.ndarray:
    stamp_size = rng.randint(3, 5)
    stamp_mask = _random_stamp_mask(stamp_size, rng)

    n_env = rng.randint(1, 4)
    corners = rng.sample(CORNER_ORDER, k=n_env)

    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    envelope_color, stamp_color = rng.sample(palette, 2)
    remaining = [c for c in palette if c not in (envelope_color, stamp_color)]
    marker_colors = rng.sample(remaining, k=n_env)

    env_shapes: list[tuple[int, int]] = []
    for _ in range(n_env):
        h = rng.randint(stamp_size + 1, min(9, stamp_size + 4))
        w = rng.randint(stamp_size + 1, min(9, stamp_size + 4))
        env_shapes.append((h, w))

    template_outer = stamp_size + 2
    total_area = template_outer * template_outer + sum(h * w for h, w in env_shapes)
    max_dim = max([template_outer] + [max(h, w) for h, w in env_shapes])
    min_side = max(max_dim + 2, int(np.ceil(np.sqrt(total_area * 2.4))))
    if min_side > 30:
        raise RuntimeError("object pack too large")
    H = rng.randint(min_side, 30)
    W = rng.randint(min_side, 30)

    shapes = {"template": (template_outer, template_outer)}
    for idx, shape in enumerate(env_shapes):
        shapes[f"env{idx}"] = shape
    positions = _place_objects(shapes, H, W, rng)

    grid = np.full((H, W), BG, dtype=int)
    _draw_template(grid, positions["template"], stamp_mask, stamp_color)
    for idx, (shape, corner, marker_color) in enumerate(zip(env_shapes, corners, marker_colors)):
        _draw_envelope(grid, positions[f"env{idx}"], shape, envelope_color, marker_color, corner)

    _validate_generated(grid, stamp_mask, n_env, set(corners))
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _gen_attempt(random.Random())
        except Exception:
            continue
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
